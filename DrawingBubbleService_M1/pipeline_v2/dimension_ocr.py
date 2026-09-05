from __future__ import annotations

import re
from typing import List

import cv2
import numpy as np

from .contracts import TextCandidate
from .ocr_utils import get_ocr
try:
    from ..diameter_ocr import normalize_diameter_symbols, diameter_candidate, reread_diameter
except ImportError:
    from diameter_ocr import normalize_diameter_symbols, diameter_candidate, reread_diameter


_DIM_PATTERNS = (
    r"^\d+(?:[\.,]\d+)?$",
    r"^\d+(?:[\.,]\d+)?\s*°$",
    r"^R\d+(?:[\.,]\d+)?$",
    r"^[Ø⌀]?\s*\d+(?:[\.,]\d+)?(?:\s*[A-Z]\d+)?$",
    r"^\d+(?:[\.,]\d+)?\s*[Xx×]\s*\d+(?:[\.,]\d+)?\s*°?$",
)


def normalize_dimension_text(text: str) -> str:
    t = normalize_diameter_symbols(text.strip()).upper()
    t = t.replace(",", ".")
    # Keep ambiguous O intact until local image verification has had a chance
    # to distinguish diameter marks from zeros and engineering identifiers.
    t = re.sub(r"\s+", " ", t)
    return t


def looks_like_dimension(text: str) -> bool:
    t = normalize_dimension_text(text)
    if not re.search(r"\d", t):
        return False
    if any(re.fullmatch(p, t) for p in _DIM_PATTERNS):
        return True
    return bool(re.search(r"\d+(?:\.\d+)?", t)) and len(t) <= 24


def _parse_ocr_result(raw):
    if raw is None:
        return []
    if isinstance(raw, tuple):
        raw = raw[0]
    return raw or []


def detect_dimension_text(image: np.ndarray) -> List[TextCandidate]:
    ocr = get_ocr()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    ocr_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    candidates: List[TextCandidate] = []
    retries_left = 12
    try:
        raw = _parse_ocr_result(ocr(ocr_img))
    except Exception:
        raw = []

    for idx, item in enumerate(raw, start=1):
        if len(item) < 3:
            continue
        box, text, conf = item[0], str(item[1]), float(item[2])
        original_text = text
        if retries_left and diameter_candidate(text):
            retries_left -= 1
            text, conf = reread_diameter(image, box, text, conf, ocr)
        norm = normalize_dimension_text(text)
        if not looks_like_dimension(norm):
            continue
        pts = np.array(box, dtype=np.float32)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        candidates.append(TextCandidate(
            candidate_id=f"t_{idx:03d}",
            text=norm,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            center=(float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
            confidence=conf,
            kind="dimension",
            raw_text=original_text,
            review_required=diameter_candidate(text) is not None,
        ))
    return candidates
