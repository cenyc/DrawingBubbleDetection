"""
Auto-annotation: take a drawing with only dimension text and generate
balloons + numbered callouts + leader lines pointing to each dimension.

This is the forward problem — simpler than reverse detection because
we control the balloon placement and leader routing.

Pipeline:
  1. OCR all text using existing ensemble
  2. Filter to real dimension callouts (groups of engineering text)
  3. Place a balloon next to each dimension in open space
  4. Number balloons in reading order (top-to-bottom, left-to-right)
  5. Draw straight leader lines from each balloon to its dimension
  6. Return annotated image + ground-truth mapping

Output metadata can be used as training data for the reverse detector.

───────────────────────────────────────────────────────────────────────
CURRENT STATE — do not chase further without a real consumer request.

Validator covers H1-H6, H8, H10, H11. H11 (leader crosses ink) uses a
20 px threshold so grazing crossings (5-15 px centerline clips) do not
trip the check while genuine leader-through-geometry does.

Current suite + baseline are recorded in AUTO_ANNOTATE_RULES.md. Any
residual H5/H11 violations trace back to the same upstream defect: OCR
garbles a dim bbox, placement routes a leader hundreds of pixels across
the drawing to reach the bogus position.

Runtime mode is governed by `strict`:
  • strict=False (production, FastAPI): validator runs, report is
    attached to returned balloon list as `.report`, never raises.
  • strict=True (tests, batch): validator raises AutoAnnotateError
    listing every violation.

Known limitations (deliberately accepted, NOT bugs):
  • Dense shaft-style drawings: drop-unresolvable may discard legitimate
    dims when available clean space cannot fit them all.
    The center-of-drawing drop bias is a real cost, documented.
  • OCR quality ceiling: bare numerics next to circular features
    ('075(n5)' should be 'Ø75(n5)', '840' should be 'Ø40'). This
    is the highest-leverage remaining lever but has been PARKED —
    do not attempt a fix without a named downstream consumer whose
    blocked on the garbage-text survivor class. The OCR Ø-restorer
    belongs in a future geometric_dim_classifier.py, not inlined
    here as a text heuristic.

If you came here to "fix" an item in §11, stop and check for an
actual caller/customer first. The current behaviour was explicitly
chosen, not overlooked.
───────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Debug flag: when set to a non-empty string, the placement loop emits
# a trace of every candidate considered for dims whose text contains
# this substring. Intended ONLY for diagnosis runs — leave as "" for
# normal operation. Set via `auto_annotate._DEBUG_TRACE_DIM = "40"` or
# similar from a caller script.
_DEBUG_TRACE_DIM: str = ""

try:
    from .ocr_rules import normalize_ocr_tokens, OCRToken
    from .callout_rules import CalloutGroup, build_callout_groups
except ImportError:
    from ocr_rules import normalize_ocr_tokens, OCRToken
    from callout_rules import CalloutGroup, build_callout_groups


# ── Data model ──────────────────────────────────────────────────

@dataclass
class AutoBubble:
    """An auto-generated balloon with its assigned dimension."""
    number: int              # sequential 1..N
    cx: int                  # balloon center x
    cy: int                  # balloon center y
    radius: int              # balloon radius
    dimension_text: str      # the dimension this balloon points to
    dim_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    leader_start: Tuple[int, int] = (0, 0)  # point on balloon rim
    leader_end: Tuple[int, int] = (0, 0)    # point on dim bbox
    leader_points: List[Tuple[int, int]] = field(default_factory=list)
    # Full polyline including start + any elbows + end. When empty,
    # renderer falls back to the straight leader_start → leader_end.


# ── Validation types ────────────────────────────────────────────

@dataclass
class Violation:
    """A single rule violation found by `_validate_output`."""
    rule: str                        # e.g. "H2", "H5"
    balloons: Tuple[int, ...]        # balloon numbers involved
    detail: str                      # human-readable explanation

    def __str__(self) -> str:
        tag = f"{self.rule}"
        if self.balloons:
            tag += f" [{','.join(str(b) for b in self.balloons)}]"
        return f"{tag}: {self.detail}"


@dataclass
class ValidationReport:
    """All violations discovered on a single auto_annotate() call."""
    violations: List[Violation] = field(default_factory=list)

    def has_violations(self) -> bool:
        return bool(self.violations)

    def by_rule(self) -> Dict[str, List[Violation]]:
        out: Dict[str, List[Violation]] = {}
        for v in self.violations:
            out.setdefault(v.rule, []).append(v)
        return out

    def summary(self) -> str:
        if not self.violations:
            return "ValidationReport: OK (0 violations)"
        rules = self.by_rule()
        parts = [f"{r}×{len(vs)}" for r, vs in sorted(rules.items())]
        return f"ValidationReport: {len(self.violations)} violation(s) — " + ", ".join(parts)

    def __str__(self) -> str:
        if not self.violations:
            return self.summary()
        lines = [self.summary()]
        for v in self.violations:
            lines.append(f"  • {v}")
        return "\n".join(lines)


class AutoAnnotateError(Exception):
    """Raised by auto_annotate() in strict mode when the validator
    finds any rule violations. The full `ValidationReport` is
    attached as `.report` so callers can enumerate every issue
    instead of fixing one at a time."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(str(report))


class BalloonList(list):
    """A list subclass so callers can read the validator's
    `ValidationReport` off the returned balloon list without
    changing the existing `(annotated, balloons)` tuple contract.

        annotated, balloons = auto_annotate(img)
        if balloons.report.has_violations():
            ...
    """
    report: "ValidationReport"


# ── OCR + dimension extraction ──────────────────────────────────

def _ocr_rotated_pass(
    image: np.ndarray,
    det,
    angles: Tuple[int, ...] = (-45, 45, -90, 90),
) -> List:
    """Rotate the image by each angle, run OCR, and map detected
    bboxes back to the original image coordinates. Catches diagonal
    dimensions (e.g. chamfer callouts written along a 45° edge) that
    upright OCR misses. Returns OCRToken objects in *original* coords.
    """
    from ocr_rules import OCRToken  # local import avoids cycle
    # Reuse the singleton OCR from detector._get_ocr() instead of
    # spawning a fresh RapidOCR instance — model loading is ~8–10 s
    # per construction, and the rotated-pass is called for every
    # auto-annotate request, so this saves real time on hot paths.
    ocr = det.ocr if hasattr(det, "ocr") else None
    if ocr is None:
        from detector import _get_ocr
        ocr = _get_ocr()

    h, w = image.shape[:2]
    out = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin_a + w * cos_a)
        nh = int(h * cos_a + w * sin_a)
        M[0, 2] += (nw - w) / 2
        M[1, 2] += (nh - h) / 2
        rot = cv2.warpAffine(
            image, M, (nw, nh),
            borderValue=(255, 255, 255),
        )
        inv_M = cv2.invertAffineTransform(M)
        try:
            result = ocr(rot)
        except Exception:
            continue
        items = result[0] if (isinstance(result, tuple) and len(result) >= 1) else (result or [])
        for item in items or []:
            try:
                bbox = item[0]
                text_info = item[1]
                text = str(text_info[0] if isinstance(text_info, (list, tuple)) else text_info).strip()
                conf = float(text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.9)
                if not text or conf < 0.5:
                    continue
                # Map each bbox corner back to original coords
                mapped = []
                for (px, py) in bbox:
                    x = inv_M[0, 0] * px + inv_M[0, 1] * py + inv_M[0, 2]
                    y = inv_M[1, 0] * px + inv_M[1, 1] * py + inv_M[1, 2]
                    mapped.append((float(x), float(y)))
                xs = [p[0] for p in mapped]
                ys = [p[1] for p in mapped]
                out.append(OCRToken(
                    text=text,
                    cx=sum(xs) / len(xs),
                    cy=sum(ys) / len(ys),
                    conf=conf,
                    x1=min(xs), y1=min(ys),
                    x2=max(xs), y2=max(ys),
                ))
            except Exception:
                continue
    return out


def _extract_dimensions(image: np.ndarray) -> List[CalloutGroup]:
    """Run OCR and return callout groups for every dimension in the image.

    Strategy:
      1. Full ensemble OCR (same as detector pipeline).
      2. Hough-detect any existing annotation bubbles; mark OCR tokens
         that sit inside those circles as "bubble numbers" and exclude
         them from the dimension pool.
      3. Build callout groups from the remaining tokens.
      4. Add every numeric token not absorbed by a group as its own
         singleton dimension — catches orphans like "(8)", "Ø53",
         "Ø56" where OCR loses the Ø / paren prefix and the grouper
         skips them as bare integers.
    """
    import re
    import math
    try:
        from .detector import BubbleDetector, DetectionConfig, _auto_ocr_scale, _auto_hough_params
        from .callout_rules import CalloutGroup as CG
    except ImportError:
        from detector import BubbleDetector, DetectionConfig, _auto_ocr_scale, _auto_hough_params
        from callout_rules import CalloutGroup as CG

    det = BubbleDetector(DetectionConfig(
        ocr_scale=0, run_multi_scale_ocr=True,
        min_radius=0, max_radius=0,
        hough_param2=0, min_dist=0,
        enable_seed_trace_assignment=False,
        enable_image_linking=False,
        enable_heavy_path_disambiguation=False,
        enable_annotation=False,
        print_timing=False,
    ))

    h, w = image.shape[:2]
    ocr_scale = _auto_ocr_scale(w, h)
    # rescue_missed=True enables single-digit recovery (e.g. lone "1"
    # or "3" between dim arrows that RapidOCR's text-detector rejects).
    raw = det._run_ocr(image, effective_scale=ocr_scale, rescue_missed=True)

    # Run a rotated-image OCR pass to catch diagonal dimensions that
    # upright OCR misses (e.g. "Ø1.000 / 1.002" slanted callouts).
    # Kept SEPARATE from the main pool so the tolerance-grouper
    # doesn't accidentally pair rotated tokens with upright ones
    # that happen to sit close by in 2D space.
    rotated_raw = _ocr_rotated_pass(image, det)

    norm = normalize_ocr_tokens(raw, image_w=w, image_h=h)

    # ── Detect existing bubbles so we can exclude their numbers ───
    # Only circles that are bubble-shaped (small, annotation-tinted
    # ring). Large Hough hits are part geometry (holes, bosses, fillets)
    # and must NOT be treated as bubbles — they'd swallow dimension
    # tokens that happen to sit on top of those features.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mn, mx, p2, md = _auto_hough_params(w, h)
    raw_circles = det._find_circles(gray, mn, mx, p2, md)
    raw_circles.extend(det._find_color_circles(image, mn, mx, existing=raw_circles))

    # Bubble annotations are small — never more than ~4% of the
    # shorter image dimension. Anything larger is a part feature
    # (hole, boss, fillet) that Hough happens to find circular.
    # Lower bound: a legible balloon digit needs ~12 px radius. Sub-12
    # circles are Hough hits on the interior curves of text characters
    # (the "O" in M60, the "0" in Ø50) — indistinguishable from real
    # bubbles by tint alone, so cap by size instead.
    hard_cap = max(50, int(min(w, h) * 0.04))
    hard_floor = 12

    tinted_small = [
        (cx, cy, cr) for (cx, cy, cr) in raw_circles
        if hard_floor <= cr <= hard_cap
        and det._circle_has_annotation_tint(image, cx, cy, cr)
    ]
    # Tighten further using the median small-circle radius so a
    # stray 2× outlier doesn't define the cap. Real annotation
    # bubbles on one drawing share a fairly uniform size.
    if tinted_small:
        tinted_small.sort(key=lambda c: c[2])
        median_r = tinted_small[len(tinted_small) // 2][2]
        max_bubble_r = max(int(median_r * 1.8), 30)
        bubble_circles = [c for c in tinted_small if c[2] <= max_bubble_r]
    else:
        bubble_circles = []
    # Reject tinted circles that contain long text tokens. Real
    # balloon bubbles hold 1-2 digit numbers (max 3 chars with a
    # revision letter suffix like "11A"). On D1c, the B datum
    # indicator and dim-extension arrow arcs fire Hough's circle
    # detector and pass tint, then enclose the Ø72(h12) dim text —
    # causing the token to be stripped from `dim_tokens`. A bubble
    # must contain at most a short number.
    def _has_non_bubble_text(cx, cy, cr):
        for tok in norm:
            if math.dist((tok.cx, tok.cy), (cx, cy)) > cr * 1.1:
                continue
            t = tok.text.strip()
            if len(t) > 3:
                return True
            # letters that aren't a single-letter suffix are non-bubble
            letters = [c for c in t if c.isalpha()]
            if len(letters) > 1:
                return True
        return False
    bubble_circles = [
        (cx, cy, cr) for (cx, cy, cr) in bubble_circles
        if not _has_non_bubble_text(cx, cy, cr)
    ]

    def _inside_any_bubble(tok) -> bool:
        for (cx, cy, cr) in bubble_circles:
            if math.dist((tok.cx, tok.cy), (cx, cy)) <= cr * 1.1:
                return True
        return False

    # Strip bubble-number tokens BEFORE grouping so they don't steal
    # a diameter "53" away from Ø53's real position.
    dim_tokens = [t for t in norm if not _inside_any_bubble(t)]

    groups = build_callout_groups(dim_tokens, scale_factor=1.0)

    # Filter watermark-like tokens. Engineering dimensions use a
    # limited vocabulary: digits, '.', ',', '/', 'Ø', '±', '°',
    # 'x' / '×', '(', ')', '±', and uppercase letter suffixes like
    # 'R', 'M', 'H', 'DIA', 'TYP', 'A'–'H' datums. Mixed-case text
    # with unusual consonants (e.g. "wikiHow" → "1ziH") is a
    # watermark / noise and should be dropped.
    def _is_watermark(txt: str) -> bool:
        s = txt.strip()
        if not s:
            return True
        # Must be predominantly digits OR a known dimension pattern.
        digit_count = sum(c.isdigit() for c in s)
        letter_count = sum(c.isalpha() for c in s)
        # If more than half is letters AND it contains lowercase
        # mixed with digits (classic watermark signature), drop.
        if letter_count > digit_count and any(c.islower() for c in s):
            # Exception: pure words like "THRU", "TYP", "ALL", etc.
            if re.fullmatch(r"[A-Za-z]+", s):
                return False
            return True
        return False

    digit_groups = [
        g for g in groups
        if re.search(r"\d", g.text) and not _is_watermark(g.text)
    ]

    # Add orphan numeric tokens that didn't end up inside a group.
    # These are bare integers/decimals that OCR lost a prefix from
    # ("Ø53" → "53", "(8)" → "8") — still real dimensions.
    grouped_texts = {
        (round(t.cx, 0), round(t.cy, 0))
        for g in groups for t in g.tokens
    }
    for tok in dim_tokens:
        if (round(tok.cx, 0), round(tok.cy, 0)) in grouped_texts:
            continue
        txt = tok.text.strip()
        if not txt or _is_watermark(txt):
            continue
        # Accept:
        #   - pure numerics          → "40", "37,5"
        #   - radius                 → "R1", "R2"
        #   - bare metric threads    → "M60"
        #   - diameter w/ fit suffix → "Ø75(n5)", "072(h12)"
        #   - isolated single digits → "5", "8"  (real short-axis dims)
        # Reject pure-letter tokens (datum letters A/B/C).
        if not re.search(r"\d", txt):
            continue
        digit_groups.append(CG(
            text=txt,
            tokens=[tok],
            x1=tok.x1, y1=tok.y1, x2=tok.x2, y2=tok.y2,
            cx=tok.cx, cy=tok.cy,
            callout_type="orphan_numeric",
        ))

    # Add rotated-pass tokens as standalone groups. For each rotated
    # token, we check whether an upright token sits at the same
    # position:
    #   • If NO upright there → add the rotated token (new discovery)
    #   • If an upright IS there but is SHORTER / a fragment of the
    #     rotated text → the rotated reading is more complete; add
    #     the rotated token AND remove any upright-based group the
    #     fragment produced
    #   • If the upright reads the same length or longer → the
    #     upright is already capturing it; skip the rotated duplicate
    # This is what lets vertical "36" survive when upright OCR only
    # managed to read the "3" fragment.
    def _nearby_upright_tokens(t):
        out = []
        for u in norm:
            if math.dist((t.cx, t.cy), (u.cx, u.cy)) <= 25.0:
                out.append(u)
        return out

    def _drop_groups_for_token(tok):
        """Remove digit_groups whose only token matches `tok`'s
        position (a fragment-based orphan)."""
        nonlocal digit_groups
        digit_groups = [
            g for g in digit_groups
            if not (
                len(g.tokens) == 1
                and math.dist(
                    (g.tokens[0].cx, g.tokens[0].cy),
                    (tok.cx, tok.cy),
                ) < 5.0
            )
        ]

    for t in rotated_raw:
        txt = t.text.strip()
        if not txt:
            continue
        if not re.search(r"\d", txt):
            continue
        nearby = _nearby_upright_tokens(t)
        if nearby:
            # Rotated wins only if its text is clearly longer than
            # every nearby upright fragment.
            longest_upright = max(len(u.text.strip()) for u in nearby)
            if len(txt) <= longest_upright:
                continue
            # Remove any orphan groups the fragment upright created
            for u in nearby:
                _drop_groups_for_token(u)
        digit_groups.append(CG(
            text=txt,
            tokens=[t],
            x1=t.x1, y1=t.y1, x2=t.x2, y2=t.y2,
            cx=t.cx, cy=t.cy,
            callout_type="rotated",
        ))

    # ── OCR-artefact normalisation ───────────────────────────────
    # Fix common RapidOCR misreads before dedup / balloon placement:
    #   • "070", "050" — Ø read as digit 0 → prepend Ø for clarity
    #   • "60%" / "(2,87%)" — ° read as % → restore °
    #   • "050/040" merged chamfer/diameter pair → split & normalise
    for g in digit_groups:
        t = g.text.strip()
        # Trailing % after digits → °
        t = re.sub(r"(\d)\s*%$", r"\1°", t)
        # % inside parens like "(2,87%)" → (2,87°)
        t = re.sub(r"(\d)\s*%\s*\)", r"\1°)", t)
        # Leading-zero diameter: "070" → "Ø70", "070.5" → "Ø70.5"
        m = re.match(r"^0(\d{2,}(?:[.,]\d+)?)$", t)
        if m:
            t = f"Ø{m.group(1)}"
        # Slash-joined diameter pairs: "050/040" → "Ø50/Ø40"
        m = re.match(r"^0(\d{2,})/0(\d{2,})$", t)
        if m:
            t = f"Ø{m.group(1)}/Ø{m.group(2)}"
        # Strip OCR-artefact "=" or "-" prefixes that come from arrows or
        # symmetry marks bleeding into the text token (e.g. "==2" → "2",
        # "-4DIA" → "4DIA"). The "=" sign is the engineering symmetric-
        # dimension mark — flag it so we can DUPLICATE the dim_group
        # below for the mirrored feature.
        symmetric = bool(re.match(r"^[=]+\s*\d", t))
        t = re.sub(r"^[=\-]+\s*", "", t)
        g.text = t
        # Tag for the post-processing pass below
        g._symmetric_dim = symmetric  # type: ignore[attr-defined]

    # ── Deduplicate ──────────────────────────────────────────────
    # OCR frequently emits the same dimension twice. Two cases:
    #   (a) Same text, close positions — e.g. "1.0±0.1" and
    #       "1.0 ±0.1" read twice at nearly the same location.
    #   (b) Overlapping bboxes regardless of text — e.g. a FCF read
    #       once as "Ø0.003(M" and once more broadly as
    #       "Ø0.003M A B C", the second bbox engulfing the first.
    # Keep the LARGER bbox in both cases (more complete reading).
    def _norm_text(t: str) -> str:
        return re.sub(r"\s+", "", t).upper()

    def _iou(g, h) -> float:
        ox = max(0.0, min(g.x2, h.x2) - max(g.x1, h.x1))
        oy = max(0.0, min(g.y2, h.y2) - max(g.y1, h.y1))
        inter = ox * oy
        if inter <= 0:
            return 0.0
        a = (g.x2 - g.x1) * (g.y2 - g.y1)
        b = (h.x2 - h.x1) * (h.y2 - h.y1)
        return inter / max(1.0, min(a, b))  # intersection-over-min

    # Typ-label dedup: collapse short labels (≤ 4 chars, e.g. "R4",
    # "C2", "TYP") only when they occur CLOSE TOGETHER — then it's
    # almost certainly the same physical text read twice (multi-scale
    # OCR + rotated pass often duplicates within a few pixels).
    # When the same short label appears at well-separated positions
    # (e.g. R2 fillet on two different corners of a part), they are
    # legitimately TWO dimensions that each need a balloon.
    TYP_LABEL = re.compile(r"^[A-Z]?\d{1,3}$|^TYP$", re.IGNORECASE)
    TYP_PROXIMITY = 80.0  # px — below this, same-text is a duplicate read
    kept_typ_positions: Dict[str, List[Tuple[float, float]]] = {}
    filtered_pre_dedup: List[CG] = []
    for g in digit_groups:
        stripped = g.text.strip()
        if TYP_LABEL.fullmatch(stripped) and len(stripped) <= 4:
            key = _norm_text(stripped)
            positions = kept_typ_positions.get(key, [])
            if any(math.hypot(g.cx - px, g.cy - py) < TYP_PROXIMITY
                   for (px, py) in positions):
                continue  # true duplicate — drop
            positions.append((g.cx, g.cy))
            kept_typ_positions[key] = positions
        filtered_pre_dedup.append(g)
    digit_groups = filtered_pre_dedup

    deduped: List[CG] = []
    # Process largest-bbox-first so the "engulfer" survives and its
    # smaller fragment sibling is rejected.
    for g in sorted(digit_groups, key=lambda g: -((g.x2 - g.x1) * (g.y2 - g.y1))):
        g_norm = _norm_text(g.text)
        is_dup = False
        for kept in deduped:
            # (a) Same-text collision
            if _norm_text(kept.text) == g_norm:
                cd = math.hypot(g.cx - kept.cx, g.cy - kept.cy)
                avg_diag = 0.5 * (
                    math.hypot(g.x2 - g.x1, g.y2 - g.y1)
                    + math.hypot(kept.x2 - kept.x1, kept.y2 - kept.y1)
                )
                if cd < max(60.0, avg_diag * 1.2):
                    is_dup = True
                    break
            # (b) Bbox overlap >= 60% of the smaller bbox's area —
            # same physical callout read twice with different text.
            if _iou(g, kept) >= 0.6:
                is_dup = True
                break
        if not is_dup:
            deduped.append(g)

    # Symmetric-dim expansion: groups whose original text started with
    # "=" denote a single dim value applied to two mirrored features
    # (engineering convention). Place a SECOND balloon for the
    # mirrored feature. Heuristic mirror: shift the dim bbox along
    # the SHORTER axis of its bbox by 3× its short dimension. For a
    # vertical-stack dim like "=2" this puts the duplicate above or
    # below the original; for a horizontal one, left or right.
    expanded: List[CG] = []
    for g in deduped:
        expanded.append(g)
        if not getattr(g, "_symmetric_dim", False):
            continue
        bw = g.x2 - g.x1
        bh = g.y2 - g.y1
        if bh >= bw:
            # Vertical or square text — duplicate offset horizontally
            offset_x = bw + 30
            new_cx = g.cx + offset_x
            new_cy = g.cy
        else:
            # Horizontal text — duplicate offset vertically
            offset_y = bh + 30
            new_cx = g.cx
            new_cy = g.cy + offset_y
        dup = CG(
            text=g.text,
            tokens=list(g.tokens),
            x1=new_cx - bw / 2, y1=new_cy - bh / 2,
            x2=new_cx + bw / 2, y2=new_cy + bh / 2,
            cx=new_cx, cy=new_cy,
            callout_type=g.callout_type,
        )
        expanded.append(dup)
    return expanded


# ── Balloon placement ───────────────────────────────────────────

def _preferred_side(
    dim_bbox: Tuple[float, float, float, float],
) -> str:
    """Pick a single preferred side for the balloon based on the
    dimension-text bbox orientation (engineering convention):
      - Horizontal text (wider than tall) → balloon placed ABOVE.
      - Vertical / square text → balloon placed to the LEFT.
    """
    x1, y1, x2, y2 = dim_bbox
    w = x2 - x1
    h = y2 - y1
    return "above" if w >= h * 1.2 else "left"


def _initial_position(
    dim_bbox: Tuple[float, float, float, float],
    radius: int,
    side: str,
) -> Tuple[int, int]:
    """Where the balloon starts before repulsion. Placed with a
    visible leader gap between the balloon rim and the dimension
    bbox (matches ref-style where the short arrow shaft is clearly
    distinct from the balloon circle)."""
    x1, y1, x2, y2 = dim_bbox
    # Gap = balloon radius + leader shaft. A shaft ~1.5× the radius
    # gives a visible, proportional leader like the references.
    leader_shaft = max(18, int(radius * 1.5))
    gap = radius + leader_shaft
    cx_dim = (x1 + x2) / 2.0
    cy_dim = (y1 + y2) / 2.0
    if side == "above":
        return (int(cx_dim), int(y1 - gap))
    if side == "below":
        return (int(cx_dim), int(y2 + gap))
    if side == "left":
        return (int(x1 - gap), int(cy_dim))
    # right
    return (int(x2 + gap), int(cy_dim))


def _build_validation_masks(
    image: np.ndarray, dilate_px: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ink_mask, inside_mask) used by `_validate_output`.

      • ink_mask: dilated binary ink — 1 where any drawing line,
        GD&T frame, dim text, or border lives.
      • inside_mask: 1 where a pixel is INSIDE a closed part outline
        (i.e. not in the largest white connected component). A
        balloon center here violates H4 even if no ink happens to
        lie under the disc.

    Kept separate so the validator can blame H3 vs. H4 distinctly
    instead of collapsing them into one "obstacle" blob.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(1, dilate_px)
    kernel = np.ones((k * 2 + 1, k * 2 + 1), np.uint8)
    ink_dilated = cv2.dilate(ink, kernel, iterations=1)

    # Flood-fill style "inside-part" detection via largest-white-CC
    seal_k = np.ones((3, 3), np.uint8)
    sealed_ink = cv2.dilate(ink, seal_k, iterations=1)
    white = (sealed_ink == 0).astype(np.uint8)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    bg_label = 0
    best_area = 0
    for i in range(1, num_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            bg_label = i
    background = (labels == bg_label).astype(np.uint8)
    inside_mask = (1 - background).astype(np.uint8)

    return (
        (ink_dilated > 0).astype(np.uint8),
        inside_mask,
    )


def _disc_overlaps_mask(
    mask: np.ndarray, cx: int, cy: int, radius: int,
) -> int:
    """Return count of mask-1 pixels inside the disc (cx, cy, radius).
    Zero means pristine whitespace under the balloon."""
    h, w = mask.shape
    if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
        return radius * radius  # treat OOB as heavily covered
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius + 1, cy + radius + 1
    sub = mask[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (radius ** 2)
    return int(np.sum(sub * disc.astype(np.uint8)))


def _leader_ink_overlap(
    pts: List[Tuple[int, int]],
    ink_mask: np.ndarray,
    skip_px: int = 3,
) -> int:
    """Count the number of ink pixels a leader polyline crosses,
    excluding a small exemption disk at each endpoint.

    Used by H11. The endpoint exemptions are mandatory:
      • Start: the leader has to begin AT the balloon rim, which
        means the first few pixels always graze the ring ink.
      • End: the leader is *supposed* to terminate at/near dim text
        or a dim arrow; crossing that is the point of the leader.

    Everything between the exemption disks is fair game. If a
    leader routes through a part body / dim-extension-line / GD&T
    frame, those pixels count.
    """
    if not pts or len(pts) < 2:
        return 0
    h, w = ink_mask.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(pts) - 1):
        ax, ay = int(pts[i][0]), int(pts[i][1])
        bx, by = int(pts[i + 1][0]), int(pts[i + 1][1])
        cv2.line(canvas, (ax, ay), (bx, by), 1, thickness=1)
    # Erase the start/end exemption disks
    sx, sy = int(pts[0][0]), int(pts[0][1])
    ex, ey = int(pts[-1][0]), int(pts[-1][1])
    cv2.circle(canvas, (sx, sy), max(1, skip_px), 0, thickness=-1)
    cv2.circle(canvas, (ex, ey), max(1, skip_px), 0, thickness=-1)
    # Intersect with the obstacle ink mask
    overlap = int(np.sum(canvas & (ink_mask > 0).astype(np.uint8)))
    return overlap


def _seg_intersects_disc(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, r: float,
) -> bool:
    """True if line segment AB enters the disc (cx, cy, r).
    Standard point-to-segment distance test."""
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(ax - cx, ay - cy) <= r
    t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / (dx * dx + dy * dy)))
    nx = ax + t * dx
    ny = ay + t * dy
    return math.hypot(nx - cx, ny - cy) <= r


def _validate_output(
    balloons: List[AutoBubble],
    image: np.ndarray,
    border_margin: int,
) -> ValidationReport:
    """Check every hard-reject rule (H1-H6, H8, H10) against the
    produced balloon set. Collects ALL violations found; does not
    short-circuit. Caller decides whether to raise.

    Skipped vs. the full spec:
      • H7 (existing-bubble proximity) — checked inside the
        preserve_existing branch at placement time; not re-checked here.
      • H9 (GD&T-frame-rectangle overlap) — requires a separate frame
        detector that isn't yet implemented; covered in practice by H3
        because GD&T frames are ink too.
    """
    report = ValidationReport()
    if not balloons:
        return report

    h, w = image.shape[:2]
    ink_mask, inside_mask = _build_validation_masks(image, dilate_px=2)

    # H1 — unique numbers
    counts: Dict[int, int] = {}
    for b in balloons:
        counts[b.number] = counts.get(b.number, 0) + 1
    for num, n in counts.items():
        if n > 1:
            report.violations.append(Violation(
                rule="H1",
                balloons=(num,),
                detail=f"number {num} appears {n} times",
            ))

    # H10 — contiguous 1..N
    nums = sorted({b.number for b in balloons})
    expected = list(range(1, len(nums) + 1))
    if nums != expected:
        missing = set(expected) - set(nums)
        extra = set(nums) - set(expected)
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"unexpected {sorted(extra)}")
        report.violations.append(Violation(
            rule="H10",
            balloons=tuple(nums),
            detail=f"numbers not 1..{len(nums)}: " + ", ".join(parts),
        ))

    # H2 — disc pairs must not overlap (gap ≥ 2 px between rims)
    for i in range(len(balloons)):
        for j in range(i + 1, len(balloons)):
            a, b = balloons[i], balloons[j]
            d = math.hypot(a.cx - b.cx, a.cy - b.cy)
            min_d = a.radius + b.radius + 2
            if d < min_d:
                report.violations.append(Violation(
                    rule="H2",
                    balloons=(a.number, b.number),
                    detail=(
                        f"disc overlap: center-dist {d:.0f}px, "
                        f"need ≥ {min_d}px (gap {d - a.radius - b.radius:.0f})"
                    ),
                ))

    # H3 — disc contains zero ink pixels
    for b in balloons:
        n_ink = _disc_overlaps_mask(ink_mask, b.cx, b.cy, b.radius)
        if n_ink > 0:
            report.violations.append(Violation(
                rule="H3",
                balloons=(b.number,),
                detail=f"{n_ink} ink pixels inside disc at ({b.cx},{b.cy}) r={b.radius}",
            ))

    # H4 — disc center must be in the largest-white-component (not
    # inside a closed part outline)
    for b in balloons:
        if 0 <= b.cx < w and 0 <= b.cy < h:
            if inside_mask[b.cy, b.cx] != 0:
                report.violations.append(Violation(
                    rule="H4",
                    balloons=(b.number,),
                    detail=f"center ({b.cx},{b.cy}) is inside a closed part outline",
                ))

    # H5 — leader segment must not cross another dim's bbox
    for b in balloons:
        pts = b.leader_points or [b.leader_start, b.leader_end]
        for si in range(len(pts) - 1):
            ax, ay = pts[si]
            bx, by = pts[si + 1]
            for other in balloons:
                if other is b:
                    continue
                ox1, oy1, ox2, oy2 = other.dim_bbox
                if _seg_intersects_bbox(ax, ay, bx, by, ox1, oy1, ox2, oy2):
                    report.violations.append(Violation(
                        rule="H5",
                        balloons=(b.number, other.number),
                        detail=(
                            f"leader of #{b.number} crosses dim bbox of "
                            f"#{other.number} ({other.dimension_text!r})"
                        ),
                    ))
                    break  # one violation per leader-vs-other-bubble is enough

    # H6 — leader segment must not cross another balloon's disc
    for b in balloons:
        pts = b.leader_points or [b.leader_start, b.leader_end]
        for si in range(len(pts) - 1):
            ax, ay = pts[si]
            bx, by = pts[si + 1]
            for other in balloons:
                if other is b:
                    continue
                if _seg_intersects_disc(ax, ay, bx, by, other.cx, other.cy, other.radius):
                    report.violations.append(Violation(
                        rule="H6",
                        balloons=(b.number, other.number),
                        detail=(
                            f"leader of #{b.number} passes through disc of #{other.number}"
                        ),
                    ))
                    break

    # H8 — disc center inside [border_margin, ...] box
    for b in balloons:
        min_edge = min(b.cx, b.cy, w - b.cx, h - b.cy)
        if min_edge < border_margin:
            report.violations.append(Violation(
                rule="H8",
                balloons=(b.number,),
                detail=(
                    f"center ({b.cx},{b.cy}) is {min_edge}px from an edge, "
                    f"border_margin={border_margin}"
                ),
            ))

    # H11 — leader must not cross part geometry / ink.
    # Exemption disks at both endpoints (see spec §1 notes) prevent
    # false positives from the unavoidable "leader touches balloon
    # rim ink" and "leader terminates at or near dim ink" cases.
    # Threshold of 20 px: grazing crossings of 5-15 px (leader
    # clipping a centerline or dim extension line) are visually
    # subtle and not worth reopening placement work to fix.
    # ≥ 20 px means the leader is genuinely running ALONG ink, not
    # just tangent to it. See AUTO_ANNOTATE_RULES.md §11 for the
    # rationale — these numbers were measured empirically from the
    # H11 addition, not prescribed in a vacuum.
    H11_THRESHOLD = 20
    H11_SKIP_PX = 3
    for b in balloons:
        pts = b.leader_points or [b.leader_start, b.leader_end]
        overlap = _leader_ink_overlap(pts, ink_mask, skip_px=H11_SKIP_PX)
        if overlap >= H11_THRESHOLD:
            report.violations.append(Violation(
                rule="H11",
                balloons=(b.number,),
                detail=(
                    f"leader of #{b.number} crosses {overlap}px of part "
                    f"geometry / ink (threshold ≥{H11_THRESHOLD}, "
                    f"skip {H11_SKIP_PX}px at endpoints)"
                ),
            ))

    return report


def _build_obstacle_mask(image: np.ndarray, dilate_px: int = 4) -> np.ndarray:
    """Return a uint8 mask the same size as the image where 1 =
    pixel is an OBSTACLE (a drawing line, GD&T frame, dimension
    text, border line, or pixel inside a closed part outline).

    Strategy:
      1. Binarise + dilate → thick ink mask (lines, text, frames).
      2. On the inverted (white) image, find connected components.
      3. The LARGEST white component is the drawing's background
         — where balloons may land. Every other white component is
         an enclosed white region inside a part and counts as
         obstacle too.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(2, dilate_px)
    kernel = np.ones((k * 2 + 1, k * 2 + 1), np.uint8)
    dilated = cv2.dilate(ink, kernel, iterations=1)
    # Lightly-dilated version for the "which white region is bg"
    # component analysis. Using the heavily-dilated mask would eat
    # into the background.
    seal_k = np.ones((3, 3), np.uint8)
    sealed_ink = cv2.dilate(ink, seal_k, iterations=1)
    white = (sealed_ink == 0).astype(np.uint8)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    # Label 0 is the inverse of `white` (ink itself); we want the
    # largest non-zero label as "background".
    bg_label = 0
    best_area = 0
    for i in range(1, num_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area = area
            bg_label = i
    background = (labels == bg_label).astype(np.uint8)
    # Obstacle = ONLY dilated ink. Pixels inside a closed part
    # contour but not on actual ink are no longer treated as hard
    # obstacles — balloons are allowed in the empty centerline
    # area inside a part body, as long as the disc doesn't cover
    # real ink (lines, text). This matches engineering-drawing
    # convention where balloons often sit in the visually-empty
    # interior of the part near the dim they annotate.
    obstacle = (dilated > 0).astype(np.uint8)
    return obstacle


def _obstacle_coverage(
    obstacle_mask: np.ndarray, cx: int, cy: int, radius: int,
) -> float:
    """Fraction of the balloon disc covered by obstacle pixels.
    Returns 1.0 if fully out of bounds, 0.0 if pristine whitespace."""
    h, w = obstacle_mask.shape
    if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
        return 1.0
    x0, y0 = cx - radius, cy - radius
    x1, y1 = cx + radius + 1, cy + radius + 1
    sub = obstacle_mask[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (radius ** 2)
    disc_area = int(disc.sum())
    if disc_area == 0:
        return 1.0
    covered = int(np.sum(sub * disc.astype(np.uint8)))
    return covered / disc_area


def _position_is_clear(
    obstacle_mask: np.ndarray, cx: int, cy: int, radius: int,
    max_coverage: float = 0.05,
) -> bool:
    """True if the balloon disc is at most `max_coverage` (fraction)
    covered by obstacle pixels. 5% tolerates thin dim extension
    lines crossing the disc (users find these visually acceptable
    and they often enable the natural below/above placement) while
    still rejecting any position on top of a GD&T frame, part
    outline, or border line."""
    return _obstacle_coverage(obstacle_mask, cx, cy, radius) <= max_coverage


def _build_ink_density(image: np.ndarray, cell: int = 16) -> np.ndarray:
    """Return a 2D density map where each cell holds the count of
    'obstacle' pixels in that grid tile.

    Obstacle = drawing ink (dark) OR pixels INSIDE a closed part
    contour (white interior enclosed by an outline). Placing a
    balloon inside a part's bounding outline looks wrong even if the
    local pixels are white — so we flood-fill from the image
    border and anything NOT reached by the flood counts as inside
    the part.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = ink.shape
    # Flood fill background from border, starting on a white pixel.
    # We dilate the ink first so outlines form closed boundaries
    # (dimension lines and part edges get connected to each other).
    kernel = np.ones((3, 3), np.uint8)
    sealed = cv2.dilate(ink, kernel, iterations=1)
    # "outside" mask: pixels reachable from the border that are NOT ink
    outside = np.zeros_like(sealed)
    # Flood-fill on an inverted image so white regions count as
    # connected; we seed all border pixels that are currently white
    # in the sealed image.
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = []
    for x in range(0, w, max(1, w // 40)):
        if sealed[0, x] == 0:
            seeds.append((x, 0))
        if sealed[h - 1, x] == 0:
            seeds.append((x, h - 1))
    for y in range(0, h, max(1, h // 40)):
        if sealed[y, 0] == 0:
            seeds.append((0, y))
        if sealed[y, w - 1] == 0:
            seeds.append((w - 1, y))
    flood = sealed.copy()
    for (sx, sy) in seeds:
        if flood[sy, sx] == 0:
            cv2.floodFill(flood, ff_mask, (sx, sy), 200)
    outside = (flood == 200).astype(np.uint8)
    # Obstacle density: real INK pixels weighted 1.0, inside-part-
    # but-no-ink pixels weighted 0.15. This lets the scorer treat
    # the visually-empty centerline area inside a part body as a
    # valid placement region (per user feedback on D1c: "there's
    # empty space there and no lines will intersect"), while still
    # charging a soft penalty for inside-part so clean outside
    # candidates win when they exist.
    ink_mask = (ink > 0).astype(np.float32)
    inside_only = ((outside == 0) & (ink == 0)).astype(np.float32) * 0.15
    obstacle = ink_mask + inside_only

    gh, gw = (h + cell - 1) // cell, (w + cell - 1) // cell
    density = np.zeros((gh, gw), dtype=np.int32)
    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * cell, min((gy + 1) * cell, h)
            x0, x1 = gx * cell, min((gx + 1) * cell, w)
            density[gy, gx] = int(obstacle[y0:y1, x0:x1].sum())
    return density


def _density_at(
    density: np.ndarray, x: int, y: int, radius: int, cell: int,
) -> float:
    """Average ink density in the grid cells covering the balloon."""
    gh, gw = density.shape
    gx0 = max(0, (x - radius) // cell)
    gy0 = max(0, (y - radius) // cell)
    gx1 = min(gw - 1, (x + radius) // cell)
    gy1 = min(gh - 1, (y + radius) // cell)
    region = density[gy0:gy1 + 1, gx0:gx1 + 1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def _resolve_overlaps(
    balloons: List[AutoBubble],
    image_shape: Tuple[int, int],
    dim_bboxes: List[Tuple[float, float, float, float]],
    existing_bubbles: List[Tuple[int, int, int]],
    ink_density: Optional[np.ndarray] = None,
    ink_cell: int = 16,
    obstacle_mask: Optional[np.ndarray] = None,
    max_iter: int = 200,
) -> None:
    """Iterative repulsion: push balloons away from each other,
    away from dimension bboxes (except their own target), and away
    from pre-existing annotation bubbles. Keeps balloons inside
    image bounds. Matches the ref style where balloons sit neatly
    next to their dimension without piling up.
    """
    h, w = image_shape
    for _ in range(max_iter):
        moved = False
        for i, a in enumerate(balloons):
            dx = 0.0
            dy = 0.0
            # Repel from other balloons. Target gap between rims is
            # 12 px — keeps Hough's circle detector seeing each as
            # a separate blob and provides headroom for the
            # deterministic separation pass below (which only needs
            # +2 px for H2 compliance). Tighter than the old r/2
            # target so dense drawings (D1 with 22 candidate dims)
            # don't exhaust placement space.
            for j, b in enumerate(balloons):
                if i == j:
                    continue
                vx = a.cx - b.cx
                vy = a.cy - b.cy
                dist = math.hypot(vx, vy) or 0.01
                # Target rim-to-rim gap of ~28 px — wide enough that
                # the eye reads adjacent balloons as separate items
                # (user flagged D1c's R2/15°/Ø88,5 cluster as hard
                # to parse). Was 12 px, which technically satisfied
                # H2 but still looked crowded.
                min_d = a.radius + b.radius + 28
                if dist < min_d:
                    force = (min_d - dist) * 1.0
                    dx += vx / dist * force
                    dy += vy / dist * force
            # Repel from pre-existing bubbles
            for (bx, by, br) in existing_bubbles:
                vx = a.cx - bx
                vy = a.cy - by
                dist = math.hypot(vx, vy) or 0.01
                min_d = a.radius + br + 6
                if dist < min_d:
                    force = (min_d - dist) * 0.8
                    dx += vx / dist * force
                    dy += vy / dist * force
            # Repel from dim bboxes (except own target). Bbox inflated
            # by 12 px — matches _BBOX_INFLATE used elsewhere so
            # visual overlap and clearance checks are consistent.
            own = a.dim_bbox
            for bbox in dim_bboxes:
                if bbox == own:
                    continue
                bx1 = bbox[0] - 12; by1 = bbox[1] - 12
                bx2 = bbox[2] + 12; by2 = bbox[3] + 12
                if (a.cx + a.radius > bx1 and a.cx - a.radius < bx2
                        and a.cy + a.radius > by1 and a.cy - a.radius < by2):
                    ccx = (bx1 + bx2) / 2
                    ccy = (by1 + by2) / 2
                    vx = a.cx - ccx
                    vy = a.cy - ccy
                    dist = math.hypot(vx, vy) or 0.01
                    dx += vx / dist * 6
                    dy += vy / dist * 6
            # Repel from part geometry / drawing ink. Sample ink
            # density at the balloon position and at four offset
            # probes; push toward the least-inky neighbour. This
            # keeps balloons off the part outline and dimension lines.
            if ink_density is not None:
                here = _density_at(ink_density, a.cx, a.cy, a.radius, ink_cell)
                if here > 0.08:  # noticeable ink inside the balloon
                    best = (here, 0.0, 0.0)
                    step = max(ink_cell, a.radius)
                    for px, py in (
                        (step, 0), (-step, 0), (0, step), (0, -step),
                    ):
                        d_probe = _density_at(
                            ink_density, a.cx + px, a.cy + py,
                            a.radius, ink_cell,
                        )
                        if d_probe < best[0]:
                            best = (d_probe, float(px), float(py))
                    # Scale the push with how inky the current spot is
                    gain = min(12.0, here * 40.0)
                    dx += best[1] / step * gain
                    dy += best[2] / step * gain
            # Strong border repulsion — match the placement margin so
            # balloons pushed by other forces don't end up hugging
            # the image edge.
            border_margin = max(a.radius * 3, int(min(w, h) * 0.05))
            if a.cx < border_margin:
                dx += (border_margin - a.cx) * 1.0
            if a.cx > w - border_margin:
                dx -= (a.cx - (w - border_margin)) * 1.0
            if a.cy < border_margin:
                dy += (border_margin - a.cy) * 1.0
            if a.cy > h - border_margin:
                dy -= (a.cy - (h - border_margin)) * 1.0
            # Attractive pull toward own dim center. Without this,
            # repulsion can drift a balloon arbitrarily far from its
            # target — resulting in 10r+ leaders and ambiguous
            # dim-to-balloon mapping. Pull engages past 3r (spec
            # sweet spot) and gets much stronger past 6r (spec cap).
            own = a.dim_bbox
            own_cx = (own[0] + own[2]) / 2.0
            own_cy = (own[1] + own[3]) / 2.0
            vx_own = own_cx - a.cx
            vy_own = own_cy - a.cy
            d_own = math.hypot(vx_own, vy_own) or 0.01
            if d_own > 3 * a.radius:
                # Pull strength scales with overshoot; past 6r it
                # beats most other repulsions.
                if d_own > 6 * a.radius:
                    pull = (d_own - 6 * a.radius) * 1.2 + (3 * a.radius) * 0.3
                else:
                    pull = (d_own - 3 * a.radius) * 0.3
                dx += vx_own / d_own * pull
                dy += vy_own / d_own * pull
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                nx = int(a.cx + dx)
                ny = int(a.cy + dy)
                # Hard clamp — never closer to an edge than the margin
                nx = max(border_margin, min(w - border_margin, nx))
                ny = max(border_margin, min(h - border_margin, ny))
                # Hard cap on distance-to-own-dim (spec §3.1 ceiling).
                # If the proposed move would land past 6r, clamp it
                # to the 6r circle around the dim. Prevents drift.
                new_d = math.hypot(nx - own_cx, ny - own_cy)
                max_d = 6 * a.radius
                if new_d > max_d:
                    t = max_d / new_d
                    nx = int(own_cx + (nx - own_cx) * t)
                    ny = int(own_cy + (ny - own_cy) * t)
                # Reject moves that INCREASE overlap with another
                # dim's bbox (inflated by 5 px for visual-ink extent).
                # Balloon-on-dim-text is a hard no per user feedback.
                def _bbox_penetration(px: int, py: int) -> float:
                    depth = 0.0
                    for bx in dim_bboxes:
                        if bx == own:
                            continue
                        ox1 = bx[0] - 12; oy1 = bx[1] - 12
                        ox2 = bx[2] + 12; oy2 = bx[3] + 12
                        if (px + a.radius > ox1 and px - a.radius < ox2
                                and py + a.radius > oy1 and py - a.radius < oy2):
                            dpx = min(px + a.radius, ox2) - max(px - a.radius, ox1)
                            dpy = min(py + a.radius, oy2) - max(py - a.radius, oy1)
                            depth += min(dpx, dpy)
                    return depth
                bbox_cur = _bbox_penetration(a.cx, a.cy)
                bbox_new = _bbox_penetration(nx, ny)
                if bbox_new > 0 and bbox_new > bbox_cur - 0.5:
                    continue

                # Reject moves that land the balloon on an obstacle
                # (line, GD&T frame, part outline). Two exceptions:
                # (1) balloon is currently past 6r from its dim —
                #     long-leader is worse than ink overlap, so allow
                #     the move inward even if it lands on dirtier ink
                # (2) the move significantly reduces distance-to-own-
                #     dim — allow a little ink for a big distance gain
                if obstacle_mask is not None:
                    cov_new = _obstacle_coverage(obstacle_mask, nx, ny, a.radius)
                    cov_cur = _obstacle_coverage(obstacle_mask, a.cx, a.cy, a.radius)
                    d_cur_own = math.hypot(a.cx - own_cx, a.cy - own_cy)
                    d_new_own = math.hypot(nx - own_cx, ny - own_cy)
                    distance_improve = d_cur_own - d_new_own
                    past_cap = d_cur_own > 6 * a.radius
                    if (cov_new > 0.02 and cov_new > cov_cur
                            and not past_cap
                            and distance_improve < 10):
                        continue
                a.cx, a.cy = nx, ny
                moved = True
        if not moved:
            break


# ── Numbering ──────────────────────────────────────────────────

def _number_balloons(balloons: List[AutoBubble], row_tol: int = 30) -> None:
    """
    Assign sequential numbers in reading order (top-to-bottom,
    left-to-right within each row).
    """
    # Sort into rows by y-coordinate, then within row by x
    sorted_b = sorted(balloons, key=lambda b: (b.cy, b.cx))

    # Group into rows — a row is a contiguous set whose cy values are
    # within row_tol of each other
    rows: List[List[AutoBubble]] = []
    for b in sorted_b:
        if rows and abs(b.cy - rows[-1][0].cy) < row_tol:
            rows[-1].append(b)
        else:
            rows.append([b])

    # Sort each row left-to-right, assign sequential numbers
    n = 1
    for row in rows:
        row.sort(key=lambda b: b.cx)
        for b in row:
            b.number = n
            n += 1


# ── Leader line drawing ─────────────────────────────────────────

def _compute_leader_endpoints(
    balloon: AutoBubble,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Compute where the leader line starts (on the balloon rim) and
    ends (on the nearest edge of the dimension bbox).

    Guarantees the leader has at least MIN_LEADER pixels of visible
    length:
      • If the balloon center is *inside* the dim bbox (the disc
        engulfs the text, which can happen on tall combined-stack
        groups), the standard clamp returns target == center and the
        leader is zero-length. Instead, we exit through the nearest
        bbox edge with a rim point pushed outward enough to be visible.
      • If the balloon disc is touching/overlapping the bbox so the
        rim sits at or past the bbox edge, we pull the rim back along
        the disc-to-target direction so the leader stroke isn't
        swallowed by the balloon ring.
    """
    MIN_LEADER = 6
    x1, y1, x2, y2 = balloon.dim_bbox
    cx, cy, r = balloon.cx, balloon.cy, balloon.radius

    # Degenerate case: balloon center lies inside the dim bbox. The
    # normal clamp gives target == center and produces a zero-length
    # leader. Pick the nearest bbox edge and route outward through it.
    if x1 <= cx <= x2 and y1 <= cy <= y2:
        d_top, d_bot = cy - y1, y2 - cy
        d_left, d_right = cx - x1, x2 - cx
        d_min = min(d_top, d_bot, d_left, d_right)
        if d_min == d_top:
            tip = (cx, y1); ux, uy = 0, -1
        elif d_min == d_bot:
            tip = (cx, y2); ux, uy = 0, 1
        elif d_min == d_left:
            tip = (x1, cy); ux, uy = -1, 0
        else:
            tip = (x2, cy); ux, uy = 1, 0
        # Place the rim outside the bbox, far enough that the leader
        # has a visible stroke between rim and bbox edge.
        rim_offset = max(int(d_min) + MIN_LEADER, r)
        rim = (int(cx + ux * rim_offset), int(cy + uy * rim_offset))
        return (rim, (int(round(tip[0])), int(round(tip[1]))))

    # Standard case: balloon center is outside the bbox.
    tx = max(x1, min(cx, x2))
    ty = max(y1, min(cy, y2))
    dx, dy = tx - cx, ty - cy
    d = math.hypot(dx, dy)
    if d < 1:
        return ((cx, cy), (int(tx), int(ty)))
    ux, uy = dx / d, dy / d
    rim_x = cx + ux * r
    rim_y = cy + uy * r

    # If the disc is almost touching the bbox, the rim sits at/past
    # the tip and the leader stroke disappears. Pull the rim back
    # along the disc–target axis so the leader is visible.
    leader_len = math.hypot(tx - rim_x, ty - rim_y)
    if leader_len < MIN_LEADER and d > MIN_LEADER:
        rim_dist = max(1.0, d - MIN_LEADER)
        rim_x = cx + ux * rim_dist
        rim_y = cy + uy * rim_dist

    return ((int(rim_x), int(rim_y)), (int(tx), int(ty)))


def _compute_leader_polyline(
    balloon: AutoBubble,
    obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[Tuple[int, int]]:
    """Compute a right-angle leader polyline from balloon rim to
    dimension bbox edge. The bend point is chosen to avoid cutting
    through other dimension text boxes when possible.

    Returns a list of 2 (straight) or 3 (elbowed) points in order
    [rim, (elbow), tip]. Straight is used when the balloon and
    dimension are already axis-aligned or the elbow would cross
    an obstacle worse than the straight line.
    """
    x1, y1, x2, y2 = balloon.dim_bbox
    # Tip: closest point on bbox to the balloon
    tip_x = max(x1, min(balloon.cx, x2))
    tip_y = max(y1, min(balloon.cy, y2))

    dx = tip_x - balloon.cx
    dy = tip_y - balloon.cy
    if math.hypot(dx, dy) < 1:
        return [(balloon.cx, balloon.cy), (int(tip_x), int(tip_y))]

    # Balloon rim exit — chose the cardinal direction (N/S/E/W) with
    # the most room, then extend by a short "shoulder" so the leader
    # starts horizontal or vertical (engineering convention).
    if abs(dx) >= abs(dy):
        # Horizontal-first elbow
        rim_x = balloon.cx + (balloon.radius if dx > 0 else -balloon.radius)
        rim_y = balloon.cy
        elbow = (int(tip_x), int(rim_y))
    else:
        # Vertical-first elbow
        rim_x = balloon.cx
        rim_y = balloon.cy + (balloon.radius if dy > 0 else -balloon.radius)
        elbow = (int(rim_x), int(tip_y))

    rim = (int(rim_x), int(rim_y))
    tip = (int(tip_x), int(tip_y))

    # Obstacle check: does the elbowed path cross a dimension bbox
    # other than the target? If yes, try the alternate elbow
    # orientation; if that also crosses, fall back to straight.
    def _crosses_any(pts: List[Tuple[int, int]]) -> int:
        if not obstacles:
            return 0
        score = 0
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            for (ox1, oy1, ox2, oy2) in obstacles:
                # Skip the target bbox itself
                if (abs(ox1 - x1) < 0.5 and abs(oy1 - y1) < 0.5
                        and abs(ox2 - x2) < 0.5 and abs(oy2 - y2) < 0.5):
                    continue
                if _seg_intersects_bbox(ax, ay, bx, by, ox1, oy1, ox2, oy2):
                    score += 1
        return score

    primary = [rim, elbow, tip]
    primary_cost = _crosses_any(primary)
    if primary_cost == 0:
        return primary

    # Try alternate bend orientation
    if abs(dx) >= abs(dy):
        alt_rim = (balloon.cx, balloon.cy + (balloon.radius if dy > 0 else -balloon.radius))
        alt_elbow = (alt_rim[0], int(tip_y))
    else:
        alt_rim = (balloon.cx + (balloon.radius if dx > 0 else -balloon.radius), balloon.cy)
        alt_elbow = (int(tip_x), alt_rim[1])
    alt = [(int(alt_rim[0]), int(alt_rim[1])), alt_elbow, tip]
    alt_cost = _crosses_any(alt)
    if alt_cost < primary_cost:
        return alt
    return primary


def _seg_intersects_bbox(
    ax: float, ay: float, bx: float, by: float,
    x1: float, y1: float, x2: float, y2: float,
) -> bool:
    """True if segment [a,b] intersects the axis-aligned box."""
    # Liang-Barsky clipping test
    dx = bx - ax
    dy = by - ay
    p = [-dx, dx, -dy, dy]
    q = [ax - x1, x2 - ax, ay - y1, y2 - ay]
    u1, u2 = 0.0, 1.0
    for pk, qk in zip(p, q):
        if pk == 0:
            if qk < 0:
                return False
        else:
            t = qk / pk
            if pk < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
    return u1 <= u2


# ── Non-crossing optimization ───────────────────────────────────

def _segments_cross(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    """True if segment AB crosses segment CD."""
    def ccw(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    d1 = ccw(ax, ay, bx, by, cx, cy)
    d2 = ccw(ax, ay, bx, by, dx, dy)
    d3 = ccw(cx, cy, dx, dy, ax, ay)
    d4 = ccw(cx, cy, dx, dy, bx, by)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _fix_crossings(balloons: List[AutoBubble]) -> None:
    """
    Detect crossing leader pairs and try swapping balloon positions
    to uncross them (leader lines shouldn't cross in engineering
    drawings by convention).
    """
    for _ in range(3):  # iterate a few times to resolve cascades
        changed = False
        for i in range(len(balloons)):
            for j in range(i + 1, len(balloons)):
                bi, bj = balloons[i], balloons[j]
                li_s, li_e = bi.leader_start, bi.leader_end
                lj_s, lj_e = bj.leader_start, bj.leader_end
                if _segments_cross(
                    li_s[0], li_s[1], li_e[0], li_e[1],
                    lj_s[0], lj_s[1], lj_e[0], lj_e[1],
                ):
                    # Swap balloon positions
                    bi.cx, bj.cx = bj.cx, bi.cx
                    bi.cy, bj.cy = bj.cy, bi.cy
                    # Recompute leader endpoints
                    bi.leader_start, bi.leader_end = _compute_leader_endpoints(bi)
                    bj.leader_start, bj.leader_end = _compute_leader_endpoints(bj)
                    # Re-check if swap still crosses — if so, revert
                    if _segments_cross(
                        bi.leader_start[0], bi.leader_start[1],
                        bi.leader_end[0], bi.leader_end[1],
                        bj.leader_start[0], bj.leader_start[1],
                        bj.leader_end[0], bj.leader_end[1],
                    ):
                        # Swap back
                        bi.cx, bj.cx = bj.cx, bi.cx
                        bi.cy, bj.cy = bj.cy, bi.cy
                        bi.leader_start, bi.leader_end = _compute_leader_endpoints(bi)
                        bj.leader_start, bj.leader_end = _compute_leader_endpoints(bj)
                    else:
                        changed = True
        if not changed:
            break


# ── Rendering ──────────────────────────────────────────────────

def _render(
    image: np.ndarray,
    balloons: List[AutoBubble],
    # BGR maroon — dark red with a hint of warmth. Matches the
    # target annotation style requested by the user.
    balloon_color: Tuple[int, int, int] = (40, 20, 130),
    leader_color: Tuple[int, int, int] = (40, 20, 130),
) -> np.ndarray:
    """Draw balloons, leader lines, and numbers onto a copy of the
    image in the reference style:
      - Large purple circle
      - Bold number centered inside
      - Plain straight leader line from rim to dimension (no arrowhead)
    """
    out = image.copy()

    # Line/ring thickness scales with balloon size so a big image
    # doesn't get hair-thin strokes, and a small one isn't clobbered.
    # Leader now matches the ring thickness for better visibility —
    # thin leaders on a dense drawing are easy to lose against
    # dimension-line ink.
    ring_thickness = max(2, int(max(b.radius for b in balloons) / 14)) if balloons else 2
    leader_thickness = max(2, ring_thickness)

    for b in balloons:
        # Plain leader line — no arrowhead. Drawn first so the
        # balloon ring overdraws any overshoot into the circle.
        pts = b.leader_points or [b.leader_start, b.leader_end]
        for i in range(len(pts) - 1):
            start = tuple(int(round(value)) for value in pts[i])
            end = tuple(int(round(value)) for value in pts[i + 1])
            cv2.line(out, start, end,
                     leader_color, leader_thickness, cv2.LINE_AA)
        # Balloon circle
        cv2.circle(out, (b.cx, b.cy), b.radius,
                   balloon_color, ring_thickness, cv2.LINE_AA)
        # Bold number centered in balloon.
        # Triplex is OpenCV's thickest sans-ish font and matches the
        # chunky reference style better than DUPLEX/SIMPLEX.
        text = str(b.number)
        font = cv2.FONT_HERSHEY_TRIPLEX
        # Size the text to comfortably fit inside the ring (not
        # touching edges). Smaller than before so 2-digit numbers
        # still look clean.
        font_scale = b.radius / 22.0
        text_thickness = max(1, int(b.radius / 14))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        tx = b.cx - tw // 2
        ty = b.cy + th // 2
        cv2.putText(out, text, (tx, ty), font, font_scale,
                    balloon_color, text_thickness, cv2.LINE_AA)

    return out


# ── Existing-bubble detection ──────────────────────────────────

def _find_existing_bubbles(
    image: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Return Hough + color circles that are REAL existing balloons
    (small annotation-tinted rings with a bubble-number digit inside).
    Used by preserve_existing mode to avoid double-annotating
    already-ballooned drawings.

    A circle is only counted as an existing bubble if an OCR-read
    digit token falls inside it. Without this check, Hough false
    positives on part geometry (holes, fillets, outlines) would be
    treated as existing annotations and the preserve_existing filter
    would skip legitimate dimensions nearby.
    """
    try:
        from .detector import (
            BubbleDetector, DetectionConfig, _auto_hough_params,
            _auto_ocr_scale,
        )
        from .ocr_rules import normalize_ocr_tokens, is_bubble_token
    except ImportError:
        from detector import (
            BubbleDetector, DetectionConfig, _auto_hough_params,
            _auto_ocr_scale,
        )
        from ocr_rules import normalize_ocr_tokens, is_bubble_token

    det = BubbleDetector(DetectionConfig(
        enable_seed_trace_assignment=False,
        enable_image_linking=False,
        enable_heavy_path_disambiguation=False,
        enable_annotation=False, print_timing=False,
    ))
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mn, mx, p2, md = _auto_hough_params(w, h)

    raw = det._find_circles(gray, mn, mx, p2, md)
    raw.extend(det._find_color_circles(image, mn, mx, existing=raw))

    hard_cap = max(50, int(min(w, h) * 0.04))
    hard_floor = 12
    tinted = [
        (cx, cy, cr) for (cx, cy, cr) in raw
        if hard_floor <= cr <= hard_cap
        and det._circle_has_annotation_tint(image, cx, cy, cr)
    ]
    if not tinted:
        return []

    # Require a bubble-shaped digit token inside the ring. A circle
    # with no digit inside it is a part feature, not an annotation.
    ocr_scale = _auto_ocr_scale(w, h)
    raw_ocr = det._run_ocr(image, effective_scale=ocr_scale)
    norm = normalize_ocr_tokens(raw_ocr, image_w=w, image_h=h)
    bubble_digit_tokens = [t for t in norm if is_bubble_token(t.text)]

    def _has_digit_inside(cx: int, cy: int, cr: int) -> bool:
        for t in bubble_digit_tokens:
            if math.hypot(t.cx - cx, t.cy - cy) <= cr * 0.9:
                return True
        return False

    with_digits = [c for c in tinted if _has_digit_inside(*c)]
    if not with_digits:
        return []
    with_digits.sort(key=lambda c: c[2])
    median_r = with_digits[len(with_digits) // 2][2]
    max_r = max(int(median_r * 1.8), 30)
    return [c for c in with_digits if c[2] <= max_r]


# ── Public API ─────────────────────────────────────────────────

def auto_annotate(
    image: np.ndarray,
    balloon_radius: int = 0,
    preserve_existing: bool = True,
    strict: bool = False,
) -> Tuple[np.ndarray, List[AutoBubble]]:
    """
    Generate balloons + leader lines for all dimensions in the image.

    Parameters
    ----------
    image : BGR image
    balloon_radius : size of auto-placed balloons in pixels. 0 (default)
        auto-scales to ~1.6% of the shorter image dimension.
    preserve_existing : when True, dimensions that already have an
        annotation bubble in one of their candidate placement zones
        are skipped (the existing bubble is assumed to cover that
        dimension). Useful for "fill in the missing balloons" on
        partially-annotated drawings.
    strict : when True, raise `AutoAnnotateError` if the final balloon
        set violates any hard-reject rule (H1-H6, H8, H10). When False
        (default), the validation report is attached to the returned
        balloon list as `.report` and the caller decides what to do.
        Use strict=True in tests and batch tooling; leave False in
        production so the FastAPI service doesn't 500 on real drawings.

    Returns
    -------
    annotated : BGR image with balloons and leader lines drawn
    balloons : BalloonList of AutoBubble with `.report: ValidationReport`
    """
    # Reuse the step helpers wired up in detector.py so the same
    # Live Logs + Pipeline Steps panel works for auto-annotate runs.
    from detector import _step_begin, _step_end

    h, w = image.shape[:2]

    # Auto-scale balloon radius if caller passed default (0). Target
    # ~3.5% of the shorter image side — matches the reference style
    # (large purple balloons big enough to hold readable numbers).
    if balloon_radius <= 0:
        balloon_radius = max(22, int(min(w, h) * 0.035))

    # 1. Extract dimensions via OCR
    _step_begin("1. Extract dimensions (OCR)")
    dim_groups = _extract_dimensions(image)
    _step_end("1. Extract dimensions (OCR)")

    # 1b. If preserve_existing, find bubbles already drawn in the
    # image and skip dimensions that one of them is already close to.
    _step_begin("2. Detect existing balloons (preserve)")
    existing_bubbles: List[Tuple[int, int, int]] = []
    if preserve_existing:
        existing_bubbles = _find_existing_bubbles(image)
        if existing_bubbles:
            filtered_dims = []
            for g in dim_groups:
                # "Already annotated" = any existing bubble center is
                # within ~balloon_radius*3 of the dimension bbox edge
                # (covers the same 8 cardinal placement zones used
                # below).
                dim_cx = (g.x1 + g.x2) / 2
                dim_cy = (g.y1 + g.y2) / 2
                min_d = min(
                    math.hypot(dim_cx - bx, dim_cy - by) - br
                    for (bx, by, br) in existing_bubbles
                )
                if min_d < balloon_radius * 3:
                    continue
                filtered_dims.append(g)
            dim_groups = filtered_dims

    _step_end("2. Detect existing balloons (preserve)")

    # Pre-compute two obstacle representations:
    #   • density grid for fast scoring during repulsion
    #   • full-resolution dilated mask for HARD pixel-level rejection
    #     (every balloon position must have zero obstacle pixels
    #     inside its disc — lines, GD&T frames, border lines, and
    #     part-interior regions are all obstacles).
    _step_begin("3. Build obstacle masks")
    ink_cell = 16
    ink_density = _build_ink_density(image, cell=ink_cell)
    obstacle_mask = _build_obstacle_mask(image, dilate_px=2)
    _step_end("3. Build obstacle masks")

    # 2. Place each balloon. For each dim we try MULTIPLE candidate
    # positions (4 cardinal sides × 2 shaft lengths) and score by:
    #   • ink density at the candidate (covers part geometry, FCF
    #     frames, and other text)
    #   • overlap with dimension bboxes other than the target
    #   • conflict with balloons already placed
    # Pick the lowest-score candidate. This gives training-data
    # quality spacing — no balloon-on-balloon, no balloon-on-ink.
    balloons: List[AutoBubble] = []
    dim_bboxes = [(g.x1, g.y1, g.x2, g.y2) for g in dim_groups]
    # Inflation applied to the visual-overlap checks only (scoring
    # penalty + repulsion bbox-penetration gate). Previously 5 px
    # (covered glyph anti-aliasing + overhang) but round-trip tests
    # showed M1 OCR inside the balloon disc still picked up
    # neighboring glyph ink when the disc edge sat 5-10 px from
    # another dim's bbox — producing misread balloon numbers.
    # 12 px (~half a balloon radius) reserves enough margin for
    # M1's OCR window around the balloon not to overlap adjacent
    # dim text.
    _BBOX_INFLATE = 12

    # Keep balloons inside the drawing area. 3× balloon radius is a
    # DETECTION requirement — round-trip through M1's Hough circle
    # detector needs enough clear ring around the balloon for the
    # detector to see the full circle. Less margin = missed
    # detections on balloons near the image edge. 2r was tried but
    # caused D1a balloon #9 (R1) to go undetected.
    border_margin = max(balloon_radius * 3, int(min(w, h) * 0.05))

    def _score_position(cx: int, cy: int, own_bbox, is_preferred: bool) -> float:
        # Ink term at weight 1.0 — detection critical. This is a
        # round-trip system: auto-annotated output feeds back into
        # the M1 detector, which uses Hough circle detection + OCR
        # inside the disc + skeleton-graph leader tracing. A balloon
        # whose disc is crossed by a construction line will fail
        # Hough (broken rim), OCR (ink overlapping the number), or
        # leader tracing (leader merges with the crossing line).
        ink = _density_at(ink_density, cx, cy, balloon_radius, ink_cell)
        score = ink
        # Balloon disc overlapping another dim bbox — NEAR-HARD penalty.
        # Inflation by _BBOX_INFLATE accounts for tight-OCR-crop vs
        # visible-glyph extent.
        #
        # BEYOND the hard overlap zone, a graded CLEARANCE penalty
        # keeps the balloon's disc far enough from neighboring dim
        # glyphs that the round-trip OCR inside the balloon isn't
        # contaminated by adjacent digits. Without this term, balloon
        # #6 on D1b sat 83 px from the "3,2" text (no overlap) but
        # M1's OCR still swept glyph ink from "3,2" into the reading
        # of the balloon number, producing "#75" instead of "#6".
        CLEARANCE_PX = 15
        for bx in dim_bboxes:
            if bx == own_bbox:
                continue
            ox1, oy1, ox2, oy2 = bx
            ox1 -= _BBOX_INFLATE; oy1 -= _BBOX_INFLATE
            ox2 += _BBOX_INFLATE; oy2 += _BBOX_INFLATE
            if (cx + balloon_radius > ox1 and cx - balloon_radius < ox2
                    and cy + balloon_radius > oy1 and cy - balloon_radius < oy2):
                score += 500.0
                continue
            # Graded clearance — distance from balloon edge to bbox
            dx = max(0.0, ox1 - cx, cx - ox2)
            dy = max(0.0, oy1 - cy, cy - oy2)
            edge_gap = math.hypot(dx, dy) - balloon_radius
            if 0 <= edge_gap < CLEARANCE_PX:
                score += 40.0 * (CLEARANCE_PX - edge_gap) / CLEARANCE_PX
        # LEADER-CROSSING penalty: straight line from candidate to
        # target dim bbox edge. Crossing another dim's bbox routes
        # a leader across an unrelated dim's text — second-most-
        # damaging defect after balloon-on-text.
        x1o, y1o, x2o, y2o = own_bbox
        tx = max(x1o, min(cx, x2o))
        ty = max(y1o, min(cy, y2o))
        for bx in dim_bboxes:
            if bx == own_bbox:
                continue
            bx1, by1, bx2, by2 = bx
            if _seg_intersects_bbox(cx, cy, tx, ty, bx1, by1, bx2, by2):
                score += 300.0
        # Conflict with already-placed balloons:
        # 1) OVERLAP: disc-touching or closer (+200 near-hard).
        # 2) CLUSTERING: rims within ~35 px of each other (+gradient).
        #    Prevents visually-crowded groups of 2-3 balloons that
        #    individually fit the min-gap rule but collectively look
        #    like one blob (seen on D1c where R2, 15°, Ø88,5 dims
        #    all sit in the same upper-right region).
        for placed in balloons:
            d = math.hypot(cx - placed.cx, cy - placed.cy)
            overlap_gap = balloon_radius * 2 + 4
            cluster_gap = balloon_radius * 2 + 35
            if d < overlap_gap:
                score += 200.0 * (overlap_gap - d) / overlap_gap
            elif d < cluster_gap:
                score += 150.0 * (cluster_gap - d) / (cluster_gap - overlap_gap)
        # Leader crossing an already-placed balloon (passes through
        # or touches another balloon's disc).
        for placed in balloons:
            if _seg_intersects_disc(cx, cy, tx, ty,
                                     placed.cx, placed.cy, placed.radius):
                score += 250.0
        # Conflict with existing annotation bubbles in the drawing
        for (bx, by, br) in existing_bubbles:
            d = math.hypot(cx - bx, cy - by)
            min_gap = balloon_radius + br + 10
            if d < min_gap:
                score += 2.0 * (min_gap - d) / min_gap
        # Border penalty — keep the balloon in internal whitespace.
        # Hard penalty (+20) for any position inside the margin zone,
        # plus a linear gradient inside to push candidates deeper in.
        min_edge = min(cx, cy, w - cx, h - cy)
        if min_edge < border_margin:
            score += 20.0 * (border_margin - min_edge) / border_margin + 5.0
        # Distance-to-own-dim (spec §3.1: balloon should sit at
        # [1.5r, 6r] from the dim center). Without this, balloons
        # that find clean ink far away win over close-but-inky
        # candidates, producing 10r+ leaders. The ink term already
        # discourages overlap — this term keeps distance bounded.
        own_cx = (x1o + x2o) / 2.0
        own_cy = (y1o + y2o) / 2.0
        d_own = math.hypot(cx - own_cx, cy - own_cy)
        if d_own > 6 * balloon_radius:
            # Hard penalty past spec ceiling — must beat ink density
            # so a clean-but-far candidate never beats close candidate
            score += 150.0 * (d_own / balloon_radius - 6)
        elif d_own > 3 * balloon_radius:
            score += 4.0 * (d_own / balloon_radius - 3)
        # Small discount for orientation-preferred side
        if is_preferred:
            score -= 0.05
        return score

    # Process dims in order of PLACEMENT DIFFICULTY (most constrained
    # first). A dim with only one clean cardinal slot should be
    # placed before a dim with many clean slots — otherwise the
    # easy dim steals the hard dim's single viable spot, forcing
    # the hard dim into a visually bad position. Sort by count of
    # legal cardinal candidates (fewer = harder).
    def _constraint_count(g) -> int:
        bbox = (g.x1, g.y1, g.x2, g.y2)
        x1, y1, x2, y2 = bbox
        cx_dim = (x1 + x2) / 2.0
        cy_dim = (y1 + y2) / 2.0
        gap = balloon_radius + max(18, int(balloon_radius * 1.5))
        count = 0
        for tx, ty in [
            (int(cx_dim), int(y1 - gap)),
            (int(cx_dim), int(y2 + gap)),
            (int(x1 - gap), int(cy_dim)),
            (int(x2 + gap), int(cy_dim)),
        ]:
            cx_cl = max(border_margin, min(w - border_margin, tx))
            cy_cl = max(border_margin, min(h - border_margin, ty))
            if _position_is_clear(obstacle_mask, cx_cl, cy_cl, balloon_radius):
                count += 1
        return count
    dim_groups_sorted = sorted(dim_groups, key=_constraint_count)

    _step_begin("4. Place balloons (initial)")
    for g in dim_groups_sorted:
        bbox = (g.x1, g.y1, g.x2, g.y2)
        preferred = _preferred_side(bbox)
        best = None
        best_score = float("inf")
        second_best_score = float("inf")
        # Diagnosis trace (Phase-1 instrumentation — see _DEBUG_TRACE_DIM).
        trace_enabled = bool(
            _DEBUG_TRACE_DIM and _DEBUG_TRACE_DIM in g.text
        )
        if trace_enabled:
            cx_dim_t = (g.x1 + g.x2) / 2.0
            cy_dim_t = (g.y1 + g.y2) / 2.0
            logger.warning(
                "[TRACE] dim=%r bbox=(%.0f,%.0f,%.0f,%.0f) center=(%.0f,%.0f) "
                "preferred=%s r=%d border=%d",
                g.text, g.x1, g.y1, g.x2, g.y2, cx_dim_t, cy_dim_t,
                preferred, balloon_radius, border_margin,
            )
            logger.warning(
                "[TRACE]   §3.1 distance cap (1.5r..6r) = [%d..%d] px",
                int(1.5 * balloon_radius), int(6 * balloon_radius),
            )
        # Extension cap 3.2 (~6r total). Past that, the distance
        # penalty (150×/r) ensures no candidate wins based on
        # cleanliness alone; and for dims near the image edge, the
        # extra extensions reach into the hollow part body interior
        # where ink is low but the placement is visually broken.
        for side in ("above", "left", "below", "right"):
            for extend in (1.0, 1.6, 2.4, 3.2):
                x1, y1, x2, y2 = bbox
                leader_shaft = max(18, int(balloon_radius * 1.5 * extend))
                gap = balloon_radius + leader_shaft
                cx_dim = (x1 + x2) / 2.0
                cy_dim = (y1 + y2) / 2.0
                if side == "above":
                    tx, ty = int(cx_dim), int(y1 - gap)
                elif side == "below":
                    tx, ty = int(cx_dim), int(y2 + gap)
                elif side == "left":
                    tx, ty = int(x1 - gap), int(cy_dim)
                else:
                    tx, ty = int(x2 + gap), int(cy_dim)
                # Clamp the AXIS-parallel coordinate into the margin.
                # For above/below candidates, the vertical distance
                # from the dim is what matters for the leader; the
                # horizontal center only needs to stay within the
                # image. When a dim sits near the right edge (e.g.
                # M60 at x=417 on a 473-wide image with margin 66),
                # above/below positions at x=417 are OOB and would
                # be discarded, leaving only LEFT candidates — which
                # wedge the balloon between M60 and its left neighbor
                # (Ø50). Clamping the cross-axis to margin keeps the
                # balloon unambiguously on M60's side of the part.
                if side in ("above", "below"):
                    tx = max(border_margin, min(w - border_margin, tx))
                else:
                    ty = max(border_margin, min(h - border_margin, ty))
                # H8 hard reject (border margin) — now only fires on
                # the travel-axis, i.e. above/below going past top/
                # bottom edge, or left/right going past side edges.
                oob_margin = (
                    tx < border_margin or tx > w - border_margin
                    or ty < border_margin or ty > h - border_margin
                )
                # H3 hard reject (pixel-clear disc)
                disc_clear = (
                    (not oob_margin)
                    and _position_is_clear(obstacle_mask, tx, ty, balloon_radius)
                )
                # §3.1 [1.5r, 6r] distance check (INFORMATIONAL ONLY —
                # not currently enforced as either hard or soft term,
                # captured here for Q1)
                dist_to_center = math.hypot(tx - cx_dim, ty - cy_dim)
                in_spec_range = (
                    1.5 * balloon_radius <= dist_to_center <= 6 * balloon_radius
                )
                if trace_enabled:
                    # Compute the full score breakdown only for the
                    # candidates that would have been scored
                    if disc_clear:
                        s_ink = _density_at(ink_density, tx, ty, balloon_radius, ink_cell)
                        s_bbox_overlap = 0.0
                        for bx in dim_bboxes:
                            if bx == bbox: continue
                            ox1, oy1, ox2, oy2 = bx
                            if (tx + balloon_radius > ox1 and tx - balloon_radius < ox2
                                    and ty + balloon_radius > oy1 and ty - balloon_radius < oy2):
                                s_bbox_overlap += 50.0
                        s_leader_cross = 0.0
                        tgt_x = max(x1, min(tx, x2))
                        tgt_y = max(y1, min(ty, y2))
                        for bx in dim_bboxes:
                            if bx == bbox: continue
                            bx1, by1, bx2, by2 = bx
                            if _seg_intersects_bbox(tx, ty, tgt_x, tgt_y, bx1, by1, bx2, by2):
                                s_leader_cross += 40.0
                        s_balloon_conflict = 0.0
                        for placed in balloons:
                            d = math.hypot(tx - placed.cx, ty - placed.cy)
                            min_gap = balloon_radius * 2 + 4
                            if d < min_gap:
                                s_balloon_conflict += 4.0 * (min_gap - d) / min_gap
                        s_existing = 0.0
                        for (bx_, by_, br_) in existing_bubbles:
                            d = math.hypot(tx - bx_, ty - by_)
                            min_gap = balloon_radius + br_ + 10
                            if d < min_gap:
                                s_existing += 2.0 * (min_gap - d) / min_gap
                        s_border = 0.0
                        min_edge = min(tx, ty, w - tx, h - ty)
                        if min_edge < border_margin:
                            s_border = 20.0 * (border_margin - min_edge) / border_margin + 5.0
                        s_pref = -0.05 if (side == preferred) else 0.0
                        total = s_ink + s_bbox_overlap + s_leader_cross + s_balloon_conflict + s_existing + s_border + s_pref
                        logger.warning(
                            "[TRACE]   side=%-5s extend=%.1f pos=(%d,%d) "
                            "dist=%dpx=%.1fr in_spec=%s "
                            "OOB=%s clear=%s "
                            "score[ink=%.2f bbox=%.1f leadX=%.1f ball=%.2f "
                            "exist=%.2f border=%.2f pref=%.2f] TOTAL=%.2f",
                            side, extend, tx, ty,
                            int(dist_to_center), dist_to_center / balloon_radius,
                            "Y" if in_spec_range else "N",
                            "Y" if oob_margin else "N",
                            "Y" if disc_clear else "N",
                            s_ink, s_bbox_overlap, s_leader_cross, s_balloon_conflict,
                            s_existing, s_border, s_pref, total,
                        )
                    else:
                        logger.warning(
                            "[TRACE]   side=%-5s extend=%.1f pos=(%d,%d) "
                            "dist=%dpx=%.1fr in_spec=%s "
                            "OOB=%s clear=%s  [REJECTED — not scored]",
                            side, extend, tx, ty,
                            int(dist_to_center), dist_to_center / balloon_radius,
                            "Y" if in_spec_range else "N",
                            "Y" if oob_margin else "N",
                            "Y" if disc_clear else "N",
                        )
                # OOB stays a hard reject (balloon must fit inside the
                # drawing). H3 pixel-clearance is now a SOFT score
                # penalty, not a hard drop — previously, dense drawings
                # would reject all 24 cardinal candidates and fall
                # through to the spiral fallback, which walked 20+r
                # away from the dim. Better to sit slightly on ink
                # than ship a 500-px leader (or no balloon at all).
                if oob_margin:
                    continue
                score = _score_position(tx, ty, bbox, side == preferred)
                if not disc_clear:
                    score += 30.0
                if score < best_score:
                    second_best_score = best_score
                    best_score = score
                    best = (tx, ty)
                elif score < second_best_score:
                    second_best_score = score
        # Diagonal candidates — when a dim sits in a crowded cardinal-
        # blocked region (e.g. Ø50 on D1a: all 4 cardinals either
        # inside-part or conflict with M60's balloon), a diagonal
        # offset can reach clean whitespace that no cardinal can.
        x1, y1, x2, y2 = bbox
        cx_dim = (x1 + x2) / 2.0
        cy_dim = (y1 + y2) / 2.0
        bbox_hw = (x2 - x1) / 2.0
        bbox_hh = (y2 - y1) / 2.0
        for dx_sgn, dy_sgn in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            for extend in (1.5, 2.2, 3.2):
                shaft = int(balloon_radius * 1.5 * extend)
                tx = int(cx_dim + dx_sgn * (bbox_hw + shaft))
                ty = int(cy_dim + dy_sgn * (bbox_hh + shaft))
                if (tx < border_margin or tx > w - border_margin
                        or ty < border_margin or ty > h - border_margin):
                    continue
                disc_clear_d = _position_is_clear(
                    obstacle_mask, tx, ty, balloon_radius,
                )
                score = _score_position(tx, ty, bbox, False)
                if not disc_clear_d:
                    score += 30.0
                if score < best_score:
                    second_best_score = best_score
                    best_score = score
                    best = (tx, ty)
                elif score < second_best_score:
                    second_best_score = score
        if trace_enabled:
            logger.warning(
                "[TRACE]   WINNER pos=%s score=%.2f  margin_over_runner_up=%.2f",
                best, best_score,
                (second_best_score - best_score) if second_best_score < float("inf") else float("inf"),
            )
        if best is None:
            # Spiral fallback — none of the 4 sides × 6 extensions
            # was pixel-clear. Scan a tight spiral around the
            # preferred initial point; accept the first pixel-clear
            # + no-balloon-collision + in-margin location. If even
            # that fails, pick the least-dirty candidate seen.
            start_x, start_y = _initial_position(bbox, balloon_radius, preferred)
            step = max(10, balloon_radius // 3)
            found = None
            least_dirty = None  # (coverage, x, y)
            if trace_enabled:
                cx_dim_t = (g.x1 + g.x2) / 2.0
                cy_dim_t = (g.y1 + g.y2) / 2.0
                d0 = math.hypot(start_x - cx_dim_t, start_y - cy_dim_t)
                logger.warning(
                    "[TRACE]   SPIRAL fallback start=(%d,%d) dist=%dpx=%.1fr step=%d",
                    start_x, start_y, int(d0), d0 / balloon_radius, step,
                )
            for ring in range(1, 80):
                r = ring * step
                # 12 directions per ring for finer coverage
                for i in range(12):
                    ang = i * (2 * math.pi / 12)
                    tx = int(start_x + r * math.cos(ang))
                    ty = int(start_y + r * math.sin(ang))
                    if (tx < border_margin or tx > w - border_margin
                            or ty < border_margin or ty > h - border_margin):
                        continue
                    cov = _obstacle_coverage(obstacle_mask, tx, ty, balloon_radius)
                    balloon_conflict = any(
                        math.hypot(tx - p.cx, ty - p.cy) < balloon_radius * 3
                        for p in balloons
                    )
                    if least_dirty is None or (cov + (0.5 if balloon_conflict else 0)) < least_dirty[0]:
                        least_dirty = (cov + (0.5 if balloon_conflict else 0), tx, ty)
                    if cov <= 0.02 and not balloon_conflict:
                        found = (tx, ty)
                        if trace_enabled:
                            cx_dim_t = (g.x1 + g.x2) / 2.0
                            cy_dim_t = (g.y1 + g.y2) / 2.0
                            d = math.hypot(tx - cx_dim_t, ty - cy_dim_t)
                            logger.warning(
                                "[TRACE]   SPIRAL found ring=%d ang=%d° "
                                "pos=(%d,%d) dist=%dpx=%.1fr cov=%.3f",
                                ring, int(math.degrees(ang)),
                                tx, ty, int(d), d / balloon_radius, cov,
                            )
                        break
                if found is not None:
                    break
            if found is not None:
                cx, cy = found
            elif least_dirty is not None:
                cx, cy = least_dirty[1], least_dirty[2]
                if trace_enabled:
                    cx_dim_t = (g.x1 + g.x2) / 2.0
                    cy_dim_t = (g.y1 + g.y2) / 2.0
                    d = math.hypot(cx - cx_dim_t, cy - cy_dim_t)
                    logger.warning(
                        "[TRACE]   SPIRAL least-dirty cov=%.3f "
                        "pos=(%d,%d) dist=%dpx=%.1fr",
                        least_dirty[0], cx, cy, int(d), d / balloon_radius,
                    )
            else:
                cx, cy = start_x, start_y
                cx = max(balloon_radius + 2, min(w - balloon_radius - 2, cx))
                cy = max(balloon_radius + 2, min(h - balloon_radius - 2, cy))
        else:
            cx, cy = best
        balloons.append(AutoBubble(
            number=0,  # assigned in _number_balloons
            cx=cx, cy=cy, radius=balloon_radius,
            dimension_text=g.text,
            dim_bbox=bbox,
        ))
    _step_end("4. Place balloons (initial)")

    # 3. Iterative repulsion — push overlapping balloons apart,
    # away from dim boxes they don't belong to, away from existing
    # annotation bubbles, and away from part-geometry ink.
    _step_begin("5. Resolve balloon overlaps (repulsion)")
    _resolve_overlaps(
        balloons, (h, w), dim_bboxes, existing_bubbles,
        ink_density=ink_density, ink_cell=ink_cell,
        obstacle_mask=obstacle_mask,
    )
    _step_end("5. Resolve balloon overlaps (repulsion)")

    # Deterministic final separation: if any balloon pair still
    # overlaps after the soft-repulsion loop (H2), push them apart
    # along the line between centres until the minimum gap holds.
    # Also check each move lands on a clear position — if not, undo.
    ink_mask_hard, inside_mask_hard = _build_validation_masks(image, dilate_px=2)
    def _hard_ok(px: int, py: int, r: int) -> bool:
        if px < border_margin or px > w - border_margin:
            return False
        if py < border_margin or py > h - border_margin:
            return False
        # inside-part-body check removed — balloons ARE allowed in
        # the visually-empty centerline area inside a part body
        # when no actual ink (lines, text) overlaps the disc. Trust
        # the ink-coverage check below as the sole placement gate.
        if _obstacle_coverage(ink_mask_hard, px, py, r) > 0.05:
            return False
        return True

    for _ in range(40):
        moved = False
        for i in range(len(balloons)):
            a = balloons[i]
            for j in range(i + 1, len(balloons)):
                b = balloons[j]
                vx = a.cx - b.cx
                vy = a.cy - b.cy
                d = math.hypot(vx, vy) or 0.01
                min_d = a.radius + b.radius + 3
                if d >= min_d:
                    continue
                push = (min_d - d) / 2.0 + 0.5
                ux, uy = vx / d, vy / d
                old_a = (a.cx, a.cy)
                old_b = (b.cx, b.cy)
                new_a = (
                    max(border_margin, min(w - border_margin, int(a.cx + ux * push))),
                    max(border_margin, min(h - border_margin, int(a.cy + uy * push))),
                )
                new_b = (
                    max(border_margin, min(w - border_margin, int(b.cx - ux * push))),
                    max(border_margin, min(h - border_margin, int(b.cy - uy * push))),
                )
                # Only accept the push if each balloon's new slot
                # is fully clear (no ink, inside background region).
                # Otherwise stay put — `_drop_unresolvable` below
                # handles the fact that the pair couldn't be
                # separated without landing on an obstacle.
                if _hard_ok(new_a[0], new_a[1], a.radius):
                    a.cx, a.cy = new_a
                    moved = True
                if _hard_ok(new_b[0], new_b[1], b.radius):
                    b.cx, b.cy = new_b
                    moved = True
        if not moved:
            break

    _step_begin("6. Drop unresolvable placements")
    # Final drop pass: any balloon still violating H2/H3/H4/H6 after
    # soft repulsion + deterministic separation is UNRESOLVABLE in
    # the space available. Drop it rather than ship a bad placement.
    # Spec: "algorithm correct on easy inputs, degrades gracefully
    # on hard ones" — graceful degradation = fewer balloons, not
    # wrong balloons.
    def _dim_text_quality(text: str) -> float:
        """0..1 score of how 'dimension-shaped' a text is. Used as
        the tiebreak when deciding which balloon to drop from an
        unresolvable conflict — drop the lower-quality one rather
        than defaulting to 'last reading order'. Without this, clean
        real dims (Ø50, 2×45°) get dropped in favour of OCR garbage
        (075(n5), (2w)/060, D>1:10) that happened to be placed first.
        """
        t = (text or "").strip()
        if not t:
            return 0.0
        score = 0.5

        # Strong positive: clean dim patterns
        clean_patterns = [
            r"^[Øø]\d+(?:\.\d+)?$",          # Ø50, Ø72.59
            r"^\d+\.\d+$",                    # 2.5, 117.5
            r"^\d+$",                         # 190, 63
            r"^\d+\s*[×x]\s*\d+°?$",          # 2×45°, 1x45
            r"^\d+\s*±\s*\d+(?:\.\d+)?$",     # 184 ±0.1
            r"^\d+°$",                        # 45°, 60°
            r"^M\d+(?:[×x]\d+(?:\.\d+)?)?$",  # M12×1.75
            r"^R\d+(?:\.\d+)?$",              # R4
            r"^\(\d+\)$",                     # (8)
            r"^[+-]\d+(?:\.\d+)?\s+[A-Z]$",  # +0.014 C
        ]
        for pat in clean_patterns:
            if re.fullmatch(pat, t):
                score += 0.4
                break

        # Negative: OCR artifact indicators
        if "]" in t or "[" in t:
            score -= 0.2   # stray brackets from FCF fragment reads
        if re.search(r"[()][A-Za-z]+[()/]", t):
            score -= 0.3   # "(2w)/060"-style garbage
        if re.search(r"[A-Za-z]{2,}", t) and not re.search(
            r"\b(?:DIA|THRU|TYP|ALL|AROUND|OUTER|PROFILE|MAJOR|MINOR|REF)\b",
            t.upper(),
        ):
            # Mixed-case letter clusters that aren't standard dim words
            if not re.fullmatch(r"[A-Z]\d+(?:[×x]\d+(?:\.\d+)?)?", t):
                score -= 0.15
        if t.count("/") > 1:
            score -= 0.2   # multiple slashes usually = merged garbage
        if re.search(r"\d[A-Za-z]\d", t):
            score -= 0.15  # digit-letter-digit mid-token (e.g. "2w6")
        # Trailing stray chars: "-", "]-", "("
        if re.search(r"[\-\]\(]\s*$", t) and not re.search(r"\d\s*[\-\]\)]\s*$", t):
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _balloon_violation(b, others):
        """Return first violated rule key, or None if clean."""
        # H3 now uses a coverage threshold instead of any-pixel test.
        # Thin dim extension lines (1-2 px wide) inside a balloon
        # disc are visually acceptable — balloon leaders commonly
        # overlap extension lines in reference drawings. Dim TEXT
        # (bold glyphs) still triggers H3 because it fills enough
        # area to exceed the threshold.
        if _obstacle_coverage(ink_mask_hard, b.cx, b.cy, b.radius) > 0.05:
            return "H3"
        # H4 (balloon center inside closed contour) removed — see
        # the _hard_ok note above. Balloons inside the empty part
        # interior are acceptable as long as no ink is on the disc.
        for other in others:
            if other is b:
                continue
            if math.hypot(b.cx - other.cx, b.cy - other.cy) < b.radius + other.radius + 2:
                return "H2"
        pts = b.leader_points or [b.leader_start, b.leader_end]
        for si in range(len(pts) - 1):
            ax, ay = pts[si]
            px, py = pts[si + 1]
            for other in others:
                if other is b:
                    continue
                if _seg_intersects_disc(ax, ay, px, py, other.cx, other.cy, other.radius):
                    return "H6"
        return None

    def _is_balloon_hard_clean(b, others):
        return _balloon_violation(b, others) is None

    # Debug log of drops for post-mortem analysis
    self_drop_log = getattr(auto_annotate, "_last_drop_log", [])
    auto_annotate._last_drop_log = []

    # Relocation attempt: before giving up on any balloon, try
    # moving it to an alternate position (all 4 sides × several
    # shaft lengths × diagonal offsets) that avoids all current
    # conflicts. Because each balloon's initial position is picked
    # BEFORE later balloons exist, some conflicts are solvable just
    # by reshuffling. On D1 this typically rescues 3-5 dims that
    # would otherwise be dropped.
    def _try_relocate(b, others) -> bool:
        """Return True if we found a new (cx, cy) for b that clears
        all hard constraints against `others` and the obstacle mask.
        Mutates b in place if successful."""
        x1, y1, x2, y2 = b.dim_bbox
        cx_dim = (x1 + x2) / 2.0
        cy_dim = (y1 + y2) / 2.0
        candidates = []
        # Same cap as the primary loop — 3.2 extend (~6r).
        for side in ("above", "right", "below", "left"):
            for extend in (1.0, 1.5, 2.2, 3.2):
                shaft = max(18, int(b.radius * 1.5 * extend))
                gap = b.radius + shaft
                if side == "above":
                    tx, ty = int(cx_dim), int(y1 - gap)
                elif side == "below":
                    tx, ty = int(cx_dim), int(y2 + gap)
                elif side == "left":
                    tx, ty = int(x1 - gap), int(cy_dim)
                else:
                    tx, ty = int(x2 + gap), int(cy_dim)
                candidates.append((tx, ty))
        # Diagonal corners — also capped to stay within 6r.
        for dx_sgn, dy_sgn in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            for extend in (1.5, 2.5, 3.2):
                shaft = int(b.radius * 1.5 * extend)
                tx = int(cx_dim + dx_sgn * (abs(x2 - x1) / 2 + shaft))
                ty = int(cy_dim + dy_sgn * (abs(y2 - y1) / 2 + shaft))
                candidates.append((tx, ty))
        original = (b.cx, b.cy)
        # Score all valid candidates by distance-to-dim and pick
        # the closest legal option — was first-valid before, which
        # produced long-leader diagonals when closer cardinals had
        # a marginal ink issue.
        best_cand = None
        best_d = float("inf")
        for tx, ty in candidates:
            if tx < border_margin or tx > w - border_margin:
                continue
            if ty < border_margin or ty > h - border_margin:
                continue
            if not _hard_ok(tx, ty, b.radius):
                continue
            # balloon-vs-balloon
            if any(
                math.hypot(tx - o.cx, ty - o.cy) < b.radius + o.radius + 4
                for o in others if o is not b
            ):
                continue
            # leader-cross-bbox / leader-cross-balloon
            tgt_x = max(x1, min(tx, x2))
            tgt_y = max(y1, min(ty, y2))
            leader_bad = False
            for other in others:
                if other is b:
                    continue
                ox1 = other.dim_bbox[0] - 12
                oy1 = other.dim_bbox[1] - 12
                ox2 = other.dim_bbox[2] + 12
                oy2 = other.dim_bbox[3] + 12
                if _seg_intersects_bbox(tx, ty, tgt_x, tgt_y, ox1, oy1, ox2, oy2):
                    leader_bad = True
                    break
                if _seg_intersects_disc(tx, ty, tgt_x, tgt_y,
                                         other.cx, other.cy, other.radius):
                    leader_bad = True
                    break
            if leader_bad:
                continue
            d = math.hypot(tx - cx_dim, ty - cy_dim)
            if d < best_d:
                best_d = d
                best_cand = (tx, ty)
        if best_cand is not None:
            b.cx, b.cy = best_cand
            b.leader_start, b.leader_end = _compute_leader_endpoints(b)
            b.leader_points = [b.leader_start, b.leader_end]
            return True
        b.cx, b.cy = original
        return False

    # Iteratively drop the balloon with most conflicts until all
    # remaining ones are hard-clean. Preserve reading order when
    # tied (drop later-reading-order balloons first).
    while True:
        # Leaders haven't been recomputed yet (step 5 below), so
        # approximate leader direction as straight start→end for
        # the H6 check. Good enough since L-bend polylines haven't
        # been built yet.
        for b in balloons:
            if not b.leader_points:
                b.leader_start, b.leader_end = _compute_leader_endpoints(b)
                b.leader_points = [b.leader_start, b.leader_end]
        bad_with_reason = [
            (b, _balloon_violation(b, balloons)) for b in balloons
        ]
        bad = [(b, r) for (b, r) in bad_with_reason if r is not None]
        if not bad:
            break

        # Try to RELOCATE each bad balloon before dropping any.
        # Relocation explores all candidate sides × extensions and
        # accepts the first that clears every hard constraint
        # against the current set. If we can rescue even one
        # conflict by moving a balloon, do it and re-check.
        relocated = False
        for (b, _rule) in bad:
            if _try_relocate(b, balloons):
                relocated = True
                break
        if relocated:
            continue  # re-evaluate with the moved balloon in place

        # Non-strict mode (production): if relocation can't fix the
        # conflict, ship the balloon anyway. Detection is more
        # important than hard-rule purity — a balloon sitting slightly
        # on ink is visibly imperfect but still conveys the dimension.
        # strict mode (tests) still drops to surface the conflict.
        if not strict:
            break

        # Drop the LOWEST-QUALITY bad balloon. Text-quality tiebreak
        # prefers keeping clean, dimension-shaped text and sacrifices
        # garbled fragments.
        bad.sort(
            key=lambda item: (
                _dim_text_quality(item[0].dimension_text),
                -(round(item[0].cy / max(30, item[0].radius))),
                -item[0].cx,
            ),
        )
        victim, rule = bad[0]
        # Identify who the victim was conflicting with, for diagnostics
        partner_text = ""
        if rule == "H2":
            for other in balloons:
                if other is victim:
                    continue
                if math.hypot(victim.cx - other.cx, victim.cy - other.cy) < victim.radius + other.radius + 2:
                    partner_text = f" vs {other.dimension_text!r}(q={_dim_text_quality(other.dimension_text):.2f})"
                    break
        auto_annotate._last_drop_log.append((victim.dimension_text, rule, partner_text))
        balloons.remove(victim)
        logger.info(
            "auto_annotate: dropped %r q=%.2f (rule %s)%s",
            victim.dimension_text, _dim_text_quality(victim.dimension_text),
            rule, partner_text,
        )

    _step_end("6. Drop unresolvable placements")

    # 4. Number balloons in reading order (top-to-bottom, L→R).
    _step_begin("7. Number balloons in reading order")
    _number_balloons(balloons)
    _step_end("7. Number balloons in reading order")

    # 5. Leader from balloon rim to the target dim. Prefer a short
    # straight line, but if it would cross another dim bbox, fall
    # back to an L-bend polyline routed around the obstacle — keeps
    # the leader from being drawn on top of an unrelated dim text.
    _step_begin("8. Route leader lines")
    obstacle_bboxes = [
        b.dim_bbox for i, b in enumerate(balloons)
    ]
    for b in balloons:
        b.leader_start, b.leader_end = _compute_leader_endpoints(b)
        straight = [b.leader_start, b.leader_end]
        # Does the straight leader cross another dim bbox?
        crosses = False
        ax, ay = b.leader_start
        tx, ty = b.leader_end
        for bx in obstacle_bboxes:
            if bx == b.dim_bbox:
                continue
            bx1, by1, bx2, by2 = bx
            if _seg_intersects_bbox(ax, ay, tx, ty, bx1, by1, bx2, by2):
                crosses = True
                break
        if crosses:
            # Try the L-bend polyline (has its own cross check and
            # alternate-orientation fallback); only use it if the
            # polyline reduces crossings.
            b.leader_points = _compute_leader_polyline(
                b, obstacles=obstacle_bboxes,
            )
        else:
            b.leader_points = straight
    _step_end("8. Route leader lines")

    # 6. Render annotated output
    _step_begin("9. Render annotated image")
    annotated = _render(image, balloons)
    _step_end("9. Render annotated image")

    # 7. Validate against hard-reject rules H1-H6, H8, H10.
    # In non-strict (default) mode the report is attached to the
    # returned balloon list so the caller can log/inspect it. In
    # strict mode any violation raises AutoAnnotateError.
    _step_begin("10. Validate output (H-rules)")
    report = _validate_output(balloons, image, border_margin)
    _step_end("10. Validate output (H-rules)")
    result = BalloonList(balloons)
    result.report = report
    if report.has_violations():
        if strict:
            raise AutoAnnotateError(report)
        logger.warning("auto_annotate: %s", report.summary())

    return annotated, result
