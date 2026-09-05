"""Conservative diameter recovery shared by the production and v2 OCR paths."""
import logging
import re

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_diameter_symbols(text: str) -> str:
    """Canonicalize observed symbols, without guessing that numeric zero is Ø."""
    text = text.translate(str.maketrans({"⌀": "Ø", "ø": "Ø", "Φ": "Ø", "φ": "Ø"}))
    return re.sub(r"(?<!\w)O/\s*(?=\d)", "Ø", text)


def diameter_candidate(text: str):
    """Propose recovery for quantity–diameter callouts or leading-zero fragments.

    A zero directly followed by a nonzero digit is suspicious here, but may
    still be a part identifier. Never apply this proposal without image evidence.
    Ordinary decimals (2-0.5) and embedded identifiers are excluded. Bare 06
    and clipped -03.5 are only proposals too, never unconditional corrections.
    """
    match = re.fullmatch(r"\s*([1-9]\d*)\s*[-–—]\s*[0O]\s*([1-9]\d*(?:[.,]\d+)?)\s*", text)
    if match:
        return f"{match[1]}-Ø{match[2].replace(',', '.')}"
    match = re.fullmatch(r"\s*(-?)\s*[0O]([1-9]\d*(?:[.,]\d+)?)\s*", text)
    if match:
        return f"{match[1]}Ø{match[2].replace(',', '.')}"
    return None


def _comparison_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_diameter_symbols(text)).replace(",", ".").replace("–", "-").replace("—", "-")


def _has_diameter_stroke(crop, text):
    """Verify the candidate character's split ring, not just its OCR spelling.

    A diameter ring has two enclosed regions separated diagonally. A normal
    zero has one, and an eight has vertically stacked regions. Require a
    one-to-one component/character alignment before inspecting the suspect zero.
    """
    compact = re.sub(r"\s+", "", text).replace("–", "-").replace("—", "-")
    prefix = re.match(r"(?:[1-9]\d*)?-", compact)
    index = prefix.end() if prefix else 0
    if index >= len(compact) or compact[index] not in "0O":
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    components = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # Dimension underlines and small edge fragments are not characters.
        if w > gray.shape[1] * 0.5 and w / h > 8:
            continue
        if cv2.contourArea(contour) >= max(6, gray.shape[0] ** 2 * 0.0025) and w >= 2 and h >= 2:
            components.append((x, y, w, h, i))
    components.sort()
    if len(components) != len(compact):
        return False
    x, y, w, h, parent = components[index]
    if not (0.45 <= w / h <= 1.3) or h < 12:
        return False
    holes = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == parent:
            m = cv2.moments(contour)
            if m["m00"] >= w * h * 0.035:
                holes.append((m["m10"] / m["m00"], m["m01"] / m["m00"], m["m00"]))
    if len(holes) != 2:
        return False
    left, right = sorted(holes)
    dx, dy = right[0] - left[0], right[1] - left[1]
    return (0.18 <= dx / w <= 0.65 and 0.05 <= dy / h <= 0.5
            and min(left[2], right[2]) / max(left[2], right[2]) >= 0.3)


def reread_diameter(image, box, text, confidence, ocr):
    """Deskew a RapidOCR quadrilateral and retry only ambiguous callouts.

    Accept a matching OCR callout or a verified split-ring character in the
    original image; numerical values and quantities must agree. Failures
    preserve the first read.
    At most two small OCR calls are made; no optional model is downloaded.
    """
    candidate = diameter_candidate(text)
    if candidate is None:
        return text, confidence
    try:
        points = np.asarray(box, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            return text, confidence
        # RapidOCR order: top-left, top-right, bottom-right, bottom-left.
        width = max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3]))
        height = max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1]))
        if min(width, height) < 2 or width * height > 1_000_000:
            return text, confidence
        scale = min(3.0, 96.0 / height, 1536.0 / width)
        w, h = max(2, round(width * scale)), max(2, round(height * scale))
        target = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (w, h),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
        crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        for index, variant in enumerate((crop, binary)):
            # The second attempt reads the entire rectified text line directly;
            # another detection pass can split a diameter symbol off again.
            result = ocr(variant) if index == 0 else ocr(variant, use_det=False, use_cls=False)
            items = result[0] if isinstance(result, tuple) else result
            for item in items or []:
                if len(item) == 2:
                    observed, score = str(item[0]), float(item[1])
                elif len(item) >= 3:
                    observed, score = str(item[1]), float(item[2])
                else:
                    continue
                if score >= 0.8 and _comparison_text(observed) == candidate:
                    logger.info("Diameter OCR recovered %r -> %r (%.3f)", text, candidate, score)
                    return candidate, min(confidence, score)
        if _has_diameter_stroke(crop, text):
            logger.info("Diameter stroke verified %r -> %r", text, candidate)
            return candidate, min(confidence, 0.9)
    except Exception:
        logger.debug("Local diameter OCR failed for %r", text, exc_info=True)
    return text, confidence
