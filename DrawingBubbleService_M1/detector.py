"""
Bubble detector — production pipeline.

10-step pipeline:
  1  OCR the full image (RapidOCR, multi-scale)
  2  Normalise OCR tokens
  3  Detect balloon shapes (Hough circles, multi-scale, adaptive)
  4  Read bubble numbers (OCR inside detected circles)
  5  Group OCR tokens into callout candidates
  6  Extract leader-like segments (HoughLinesP)
  7  Detect leader seed at balloon boundary (HoughLinesP v2 → color fallback)
  8  Trace leader path from seed (BFS on connected pixels)
  9  Assign dimension text at trace endpoint
  10 Fallback linker for unresolved balloons + conflict resolution

  - Scale calibration: all pixel thresholds adapt to median bubble radius
  - detect_leader_seed_v2 (HoughLinesP) used as primary seed detector
  - Leader directions threaded through to the fallback linker for angular scoring
  - resolve_assignment_conflicts() called after all passes
  - HoughCircles uses computed eff_min_r / eff_max_r (not hardcoded 15/55)
  - Bubble radius taken from matched Hough circle, not hardcoded 35
"""

from __future__ import annotations

import logging
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Unicode-aware text rendering for the annotated overlay. cv2.putText
# uses HERSHEY fonts, which are ASCII-only and silently replace any
# non-ASCII character (Ø, °, ×, ±, ⊥, ⊕, ⌗) with "?". Engineering
# drawings are full of those symbols, so we route the dim-text labels
# through Pillow + a TrueType font that supports them.
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_FONT_CACHE: Dict[int, "ImageFont.FreeTypeFont"] = {}


def _get_unicode_font(size: int):
    """Cached TrueType font that handles engineering symbols."""
    if not _PIL_AVAILABLE:
        return None
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in (
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(path, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _measure_unicode_text(text: str, font_px: int) -> Tuple[int, int]:
    """Return (width, height) of `text` as it would render at `font_px`."""
    font = _get_unicode_font(font_px)
    if font is None:
        return (len(text) * font_px // 2, font_px)
    try:
        l, t, r, b = font.getbbox(text)
        return (r - l, b - t)
    except AttributeError:
        return font.getsize(text)


def _put_text_unicode(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_px: int,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Draw Unicode text on a BGR image. `org` is the BASELINE position
    (matches cv2.putText convention) so the call site arithmetic stays
    the same."""
    if not text or not _PIL_AVAILABLE:
        # Fallback: cv2 putText (will mangle non-ASCII, but won't crash)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_px / 30.0, color_bgr, 1, cv2.LINE_AA)
        return
    font = _get_unicode_font(font_px)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        ascent, _descent = font.getmetrics()
    except Exception:
        ascent = font_px
    x = int(org[0])
    y = int(org[1]) - ascent
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((x, y), text, font=font, fill=color_rgb)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
import geometric_utils

try:
    from .ocr_rules import (
        OCRToken,
        NormalizedToken,
        normalize_ocr_tokens,
        is_bubble_token,
        normalize_bubble_value,
        is_dimension_token,
    )
    from .callout_rules import CalloutGroup, build_callout_groups
    from .leader_path_rules import extract_path_segments
    from .linker_rules import (
        BalloonNode,
        LinkResult,
        link_balloons_to_callouts,
        resolve_assignment_conflicts,
    )
    from .leader_seed_rules import (
        BalloonGeometry,
        LeaderSeed,
        detect_best_leader_seed,
        trace_leader_from_seed,
        color_trace_to_callout,
    )
    from .annotation_layer import (
        extract_annotation_layer,
        trace_annotation_path,
    )
    from .skeleton_graph import skeleton_assign
except ImportError:
    from ocr_rules import (
        OCRToken,
        NormalizedToken,
        normalize_ocr_tokens,
        is_bubble_token,
        normalize_bubble_value,
        is_dimension_token,
    )
    from callout_rules import CalloutGroup, build_callout_groups
    from leader_path_rules import extract_path_segments
    from linker_rules import (
        BalloonNode,
        LinkResult,
        link_balloons_to_callouts,
        resolve_assignment_conflicts,
    )
    from leader_seed_rules import (
        BalloonGeometry,
        LeaderSeed,
        detect_best_leader_seed,
        trace_leader_from_seed,
        color_trace_to_callout,
    )
    from annotation_layer import (
        extract_annotation_layer,
        trace_annotation_path,
    )
    from skeleton_graph import skeleton_assign

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step timing — structured log records that the live UI parses into a
# per-step progress list. Format chosen so it's both human-readable in
# raw logs AND machine-parseable in JS without a JSON serialiser:
#
#   [STEP] start: <name>
#   [STEP] done:  <name> | ms=<elapsed>
#
# A skipped step is logged as "[STEP] skip: <name>" so the UI can show
# the step as gray / "n/a" instead of leaving it pending forever.
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def _step(name: str):
    """Wrap a phase of the pipeline so its start/end + duration is logged."""
    t0 = time.perf_counter()
    logger.info("[STEP] start: %s", name)
    try:
        yield
    except Exception:
        ms = (time.perf_counter() - t0) * 1000.0
        logger.info("[STEP] fail:  %s | ms=%.1f", name, ms)
        raise
    else:
        ms = (time.perf_counter() - t0) * 1000.0
        logger.info("[STEP] done:  %s | ms=%.1f", name, ms)


# Inline begin/end for places where wrapping the body in `with` would
# force big indentation changes. Pair each _step_begin with a
# matching _step_end at the same scope.
_step_starts: Dict[str, float] = {}


def _step_begin(name: str) -> None:
    _step_starts[name] = time.perf_counter()
    logger.info("[STEP] start: %s", name)


def _step_end(name: str) -> None:
    t0 = _step_starts.pop(name, None)
    if t0 is None:
        logger.info("[STEP] done:  %s", name)
    else:
        ms = (time.perf_counter() - t0) * 1000.0
        logger.info("[STEP] done:  %s | ms=%.1f", name, ms)


def _step_skip(name: str, reason: str = "") -> None:
    suffix = f" | reason={reason}" if reason else ""
    logger.info("[STEP] skip:  %s%s", name, suffix)


# ─────────────────────────────────────────────────────────────────────────────
# OCR singleton
# ─────────────────────────────────────────────────────────────────────────────

_ocr_instance: Optional[RapidOCR] = None


def _get_ocr() -> RapidOCR:
    """Return a singleton RapidOCR instance running on CPU only."""
    global _ocr_instance
    if _ocr_instance is None:
        # CPU-only — explicit. Set deployment-wide policy: no GPU.
        _ocr_instance = RapidOCR(
            det_use_cuda=False,
            cls_use_cuda=False,
            rec_use_cuda=False,
        )
        logger.info("RapidOCR initialized on CPU")
    return _ocr_instance


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive parameter helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auto_ocr_scale(image_w: int, image_h: int) -> int:
    """Pick full-page OCR scale.

    Full-page OCR is used to build the global text graph.  Local recovery
    paths still run 3x/4x OCR on bubble and leader-endpoint crops, so capping
    page-wide OCR at 2x avoids paying a quadratic cost on tiny images while
    preserving high-resolution reads where geometry says they are needed.
    """
    estimated_text_h = image_w * 0.015
    if estimated_text_h < 20:
        return 2
    return 1


def _auto_hough_params(image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    """Return (min_radius, max_radius, param2, min_dist) tuned to image size."""
    short = min(image_w, image_h)
    min_r  = max(6,  int(short * 0.020))
    max_r  = max(min_r + 20, int(short * 0.15))
    param2 = 22 if short < 500 else 26
    min_dist = max(12, int(min_r * 1.5))
    return min_r, max_r, param2, min_dist


def _compute_scale_factor(bubble_radii: List[float]) -> float:
    """
    Derive a pixel-scale multiplier from detected bubble radii.

    Reference: 25 px radius at standard screen resolution.
    All distance thresholds are multiplied by this factor so the system
    adapts automatically to different drawing scales.
    """
    valid = [r for r in bubble_radii if 5 < r < 200]
    if not valid:
        return 1.0
    median_r = float(sorted(valid)[len(valid) // 2])
    return max(0.4, min(5.0, median_r / 25.0))


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic-render helpers (visualisation only — never used by detection)
# ─────────────────────────────────────────────────────────────────────────────


def _is_rim_dir(dir_x: float, dir_y: float,
                radial_x: float, radial_y: float) -> bool:
    """True if `dir` is tangent-to or pointing into the rim relative to
    the radial-outward vector. Used to detect when the seed's direction
    is unsuitable for tracing into the leader stub."""
    rn = math.hypot(radial_x, radial_y)
    if rn < 1e-6:
        return False
    rx, ry = radial_x / rn, radial_y / rn
    return (dir_x * rx + dir_y * ry) < 0.2


def _perimeter_scan_for_stub(
    mask: np.ndarray,
    balloon_centre: Tuple[float, float],
    balloon_radius: float,
) -> Optional[Tuple[Tuple[int, int], Tuple[float, float]]]:
    """Scan every angle around the balloon rim looking for an off-rim
    maroon pixel within a short outward range. Returns
    ((entry_x, entry_y), (dir_x, dir_y)) of the strongest outward trail
    found, or None if no rim angle has a stub.

    Used as a fallback when the seed-contact wedge probe fails because
    the contact point doesn't coincide with the leader-line exit."""
    h, w = mask.shape[:2]
    cx, cy = balloon_centre
    r = balloon_radius
    # Match the BFS suppress radius (rim band is blocked there too).
    # Start the perimeter probe at 1.15·r so any entry we return is
    # clearly OUTSIDE the rim band BFS can't enter.
    min_probe_dist = int(r * 1.15) + 2
    max_probe = max(60, int(r * 2.5))

    best_entry: Optional[Tuple[int, int]] = None
    best_dir: Optional[Tuple[float, float]] = None
    best_depth = 0  # number of consecutive maroon pixels along the ray

    for ang_deg in range(0, 360, 5):
        ang = math.radians(ang_deg)
        ux, uy = math.cos(ang), math.sin(ang)
        entry: Optional[Tuple[int, int]] = None
        for step in range(min_probe_dist, max_probe + 1):
            sx = int(round(cx + ux * step))
            sy = int(round(cy + uy * step))
            if not (0 <= sx < w and 0 <= sy < h):
                break
            if mask[sy, sx] > 0:
                entry = (sx, sy)
                break
        if entry is None:
            continue
        # Score this ray by how many consecutive maroon pixels we get
        # walking outward from the entry. A longer streak = a more
        # confident leader stub.
        depth = 0
        for further in range(0, 60):
            sx = entry[0] + int(round(ux * further))
            sy = entry[1] + int(round(uy * further))
            if not (0 <= sx < w and 0 <= sy < h):
                break
            if mask[sy, sx] > 0:
                depth += 1
            else:
                # Allow a 1-pixel gap then stop.
                gx = entry[0] + int(round(ux * (further + 1)))
                gy = entry[1] + int(round(uy * (further + 1)))
                if (0 <= gx < w and 0 <= gy < h and mask[gy, gx] > 0):
                    depth += 1
                    continue
                break
        if depth > best_depth:
            best_depth = depth
            best_entry = entry
            best_dir = (ux, uy)

    if best_entry is None or best_depth < 3:
        return None
    return best_entry, best_dir


def _outward_rim_trace(
    mask: np.ndarray,
    contact: Tuple[int, int],
    radial: Tuple[float, float],
    balloon_centre: Tuple[float, float],
    balloon_radius: float,
    max_steps: int = 400,
) -> Optional[List[Tuple[int, int]]]:
    """Render-only leader trace.

    Strategy:
      1. Full-perimeter scan around the balloon rim — find the angle
         whose outward maroon trail is longest. That's the leader.
      2. From the resulting (entry, direction), BFS-follow ink off the
         rim to the trail's endpoint.

    The earlier wedge-around-contact-point approach was fragile when
    the seed detector's contact point was nowhere near where the
    leader actually exits the rim. The perimeter sweep is exhaustive
    and finds the real exit regardless of seed quality."""
    h, w = mask.shape[:2]
    cx, cy = balloon_centre
    r = balloon_radius
    suppress_r = r * 1.1
    rx, ry = radial

    scan = _perimeter_scan_for_stub(mask, balloon_centre, balloon_radius)
    if scan is None:
        return None
    entry, (rx, ry) = scan

    # BFS-follow ink from `entry`, blocking everything within suppress_r.
    visited = np.zeros((h, w), dtype=bool)
    cy_arr, cx_arr = np.ogrid[:h, :w]
    visited[(cx_arr - cx) ** 2 + (cy_arr - cy) ** 2 <= suppress_r ** 2] = True
    visited[entry[1], entry[0]] = False

    queue: List[Tuple[float, int, int, List[Tuple[int, int]]]] = []
    import heapq
    heapq.heappush(queue, (0.0, entry[0], entry[1], [contact, entry]))
    visited[entry[1], entry[0]] = True
    best_path = [contact, entry]
    best_score = 0.0

    neighbours = [(-1, -1), (0, -1), (1, -1),
                  (-1,  0),          (1,  0),
                  (-1,  1), (0,  1), (1,  1)]

    while queue:
        neg_prog, pcx, pcy, path = heapq.heappop(queue)
        prog = -neg_prog
        if len(path) > max_steps:
            continue
        for dx, dy in neighbours:
            nx, ny = pcx + dx, pcy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if visited[ny, nx]:
                continue
            if mask[ny, nx] == 0:
                continue
            visited[ny, nx] = True
            step_prog = dx * rx + dy * ry
            new_prog = prog + step_prog
            new_path = path + [(nx, ny)]
            if new_prog > best_score:
                best_score = new_prog
                best_path = new_path
            heapq.heappush(queue, (-new_prog, nx, ny, new_path))
    return best_path if len(best_path) > 2 else None


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    # Circle detection (0 = auto-compute)
    min_radius: int = 0
    max_radius: int = 0
    hough_param1: int = 50
    hough_param2: int = 0
    min_dist: int = 0
    dedup_dist: int = 25
    edge_margin: int = 5
    max_candidate_circles: int = 18

    # OCR
    ocr_scale: int = 0          # 0 = auto
    min_ocr_conf: float = 0.35
    run_multi_scale_ocr: bool = True
    enable_adaptive_multiscale_ocr: bool = True
    enable_rotated_fullpage_ocr: bool = False
    enable_fullpage_ocr_recovery_variants: bool = False
    enable_optional_ocr_ensemble: bool = False

    # Matching — base values scaled by scale_factor at runtime
    max_assoc_distance: int = 300
    max_assoc_cost: float = 360.0
    leader_bonus_weight: float = 42.0

    # Fallback bubble search
    uncircled_near_circle_dist: int = 60
    enable_circle_recovery_ocr: bool = True
    # Cap on unmatched circles we attempt to recover via OCR. Candidates are
    # evidence-prioritized before OCR, so the first few carry nearly all useful
    # recall; later attempts mostly spend CPU on drawing geometry/artifacts.
    max_unmatched_circle_recovery: int = 2
    max_edge_blob_recovery_ocr: int = 6
    max_reverse_discovery_callouts: int = 24
    max_reverse_discovery_ocr: int = 10
    max_red_blob_local_ocr: int = 64
    max_annotation_ring_recovery_ocr: int = 18
    max_annotation_ring_recovery_candidates: int = 10
    max_step6_recovery_seconds: float = 75.0

    # Feature flags
    fast_debug_mode: bool = False
    enable_annotation: bool = True
    enable_seed_trace_assignment: bool = True
    enable_image_linking: bool = True
    max_balloons_for_image_linking: int = 12
    enable_heavy_path_disambiguation: bool = False
    print_timing: bool = False

    # Screenshot normalization — auto-detect noisy / undersized /
    # warm-tinted client screenshots and apply a small upfront fix-up
    # pass (upscale, white-balance, bilateral-denoise, deskew, crop).
    # Off by default because it changes the input pixels before OCR/geometry.
    enable_screenshot_normalization: bool = False

    # Real-ESRGAN-anime ONNX enhancement — 4x super-resolution +
    # denoise applied as a side-copy before circle detection. The
    # anime-trained model is suited to CAD line art (no hallucinated
    # texture, preserves thin strokes). Off by default so we can
    # A/B test against the existing suite without breaking baselines.
    # When on, the enhanced image is used for circle/blob detection;
    # OCR continues to run on the original to preserve character
    # shapes (we proved earlier that boosted images degrade OCR).
    enable_realesrgan_enhancement: bool = False
    realesrgan_model_path: str = "models/realesr-anime-v3.onnx"

    # VLM verification — uses a local vision model (via Ollama)
    # to verify/correct dimension assignments.  Runs offline.
    # Requires: ollama installed + `ollama pull minicpm-v`
    # Set to True only if Ollama is running on the machine.
    enable_vlm_verification: bool = False
    vlm_model: str = "minicpm-v"
    vlm_host: str = "http://127.0.0.1:11434"

    # Targeted endpoint OCR — runs a local RapidOCR rescue pass
    # around unresolved/low-confidence bubbles. Adds ~5s per
    # request. Off by default; set True when the caller cares about maximum
    # accuracy over latency.
    enable_targeted_endpoint_ocr: bool = False
    max_targeted_endpoint_ocr: int = 4
    max_local_rescue_ocr_variants: int = 120
    # Output-boundary assignment validation. Late rescue/fallback passes may
    # attach a nearby dimension-looking token even when no leader geometry
    # supports that link. Keep this on in production: unsupported weak
    # assignments are cleared to NO_DIMENSION instead of published as facts.
    enable_assignment_geometry_validation: bool = True
    # First-stage architecture migration: claim trace-supported assignments
    # before the global fallback. Dense pages require a stricter trace-quality
    # gate so a wandering line cannot steal a nearby dimension.
    enable_leader_first_assignment: bool = True
    max_bubbles_for_leader_first_assignment: int = 10000
    min_leader_trace_quality: float = 0.68
    min_dense_leader_trace_quality: float = 0.76
    dense_leader_trace_bubble_count: int = 12

    # Photo-input preprocessing: deskew + lighting normalisation +
    # light denoise. Set True for phone-camera input. Default False
    # each sub-step is also internally gated, but defaulting off avoids any
    # risk of resampling blur / CLAHE shifting a borderline OCR read on a
    # clean scan.
    enable_photo_preprocessing: bool = False



@dataclass
class BubbleResult:
    bubble_number: str
    x: int
    y: int
    radius: int
    dimension: str
    confidence: float
    needs_review: bool = False
    review_reason: str = ""
    candidate_count: int = 0

    def to_dict(self) -> dict:
        # Cast every field through a plain-Python constructor. Several
        # detector stages use numpy comparisons (e.g. cost-matrix
        # thresholds) whose result is np.bool_ / np.float64 — those
        # are not JSON-serialisable and would crash JSONResponse in
        # /api/detect. Casting at the boundary is cheap and keeps
        # individual assignment sites from having to remember to
        # wrap with bool()/float().
        return {
            "bubble_number": str(self.bubble_number),
            "dimension": str(self.dimension) if self.dimension is not None else "",
            "x": int(self.x),
            "y": int(self.y),
            "radius": int(self.radius),
            "confidence": round(float(self.confidence), 3),
            "needs_review": bool(self.needs_review),
            "review_reason": str(self.review_reason) if self.review_reason is not None else "",
            "candidate_count": int(self.candidate_count),
        }


@dataclass
class AssignmentCandidate:
    """Evidence-ranked candidate before mutating a bubble assignment."""
    bubble_index: int
    callout_index: int
    score: float
    endpoint_distance: float
    text_quality: float
    reason: str


@dataclass
class TraceQuality:
    """Decision-facing quality summary for one physical leader trace."""
    score: float
    path_points: int
    endpoint_distance: float
    target_distance: float
    straightness: float
    continuity: float
    escape_score: float
    target_score: float
    touches_callout: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 3),
            "path_points": int(self.path_points),
            "endpoint_distance": round(float(self.endpoint_distance), 1),
            "target_distance": (
                round(float(self.target_distance), 1)
                if math.isfinite(float(self.target_distance)) else None
            ),
            "straightness": round(float(self.straightness), 3),
            "continuity": round(float(self.continuity), 3),
            "escape_score": round(float(self.escape_score), 3),
            "target_score": round(float(self.target_score), 3),
            "touches_callout": bool(self.touches_callout),
            "reason": self.reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Detector
# ─────────────────────────────────────────────────────────────────────────────

class BubbleDetector:
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.cfg = config or DetectionConfig()
        self.ocr  = _get_ocr()

        # State set fresh on every detect_from_array call
        self.image: Optional[np.ndarray] = None
        self._norm_tokens: List[NormalizedToken] = []
        self._seed_traces: Dict[str, Any] = {}
        # Circle helpers are also reused directly by auto_annotate.py,
        # before detect_from_array() has had a chance to derive
        # image-specific values. Start with the configured baseline;
        # detect_from_array() will still raise it for upscaled inputs.
        self._eff_dedup_dist = self.cfg.dedup_dist

        # Real-ESRGAN enhancer — lazy. We construct the wrapper now
        # so missing-file errors surface at startup, but the ONNX
        # session itself loads on first use (saves cold-start cost
        # when the flag is off).
        self._enhancer = None
        if self.cfg.enable_realesrgan_enhancement:
            try:
                from image_enhancer import RealESRGANEnhancer
                self._enhancer = RealESRGANEnhancer(
                    model_path=self.cfg.realesrgan_model_path,
                )
                logger.info("Real-ESRGAN enhancer ready (model=%s)",
                            self.cfg.realesrgan_model_path)
            except Exception as exc:
                logger.warning(
                    "Real-ESRGAN enhancer disabled — failed to initialise: %s",
                    exc,
                )
                self._enhancer = None

    # ── Public API ─────────────────────────────────────────────────

    def detect(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return self.detect_from_array(img)

    def detect_from_array(self, img: np.ndarray):
        if img is None:
            raise ValueError("Image array is None.")

        t0 = time.perf_counter()
        self.image = img
        # Preserve the original input *before* any preprocessing so
        # later passes can read the un-blurred chroma signal (the
        # moiré-cleanup blur de-saturates the price-table / B&W areas
        # enough to fool the strict tint check on the normalised
        # image). Used by `_suppress_suspicious_bubbles`.
        self._original_image = img.copy()
        # Reset per-request rescue OCR cache so results from a prior
        # image don't leak into this one.
        self._rescue_crop_cache = {}
        self._local_angle_crop_cache = {}
        self._bubble_recovery_cache = {}
        self._step6_recovery_deadline = None
        self._step6_budget_logged = False
        self._seed_traces = {}
        self._seed_traces_by_index = {}
        self._render_seed_traces = {}

        # ── STEP 0a: Assess image quality ────────────────────────
        # Compute statistics once up front so later steps (upscale,
        # contrast boost, split-stream OCR) can branch on them
        # consistently. The result is stashed on `self._quality` and
        # surfaced in the diagnostics payload so users can see what
        # the detector thinks of each image without re-running.
        _step_begin("0a. Assess image quality")
        self._quality = self._assess_image_quality(img)
        logger.info("Image quality: %s", self._quality)
        _step_end("0a. Assess image quality")

        # ── STEP 0b: Photo-input preprocessing ───────────────────
        # Deskew + lighting normalisation + light denoise. Each sub-
        # step is internally gated so clean scans pass through with
        # no transformation. For phone-camera input this rotates the
        # page upright and evens out glare before OCR runs.
        if self.cfg.enable_photo_preprocessing:
            _step_begin("0b. Photo-input preprocessing")
            try:
                try:
                    from .photo_preprocessing import preprocess_photo_input
                except ImportError:
                    from photo_preprocessing import preprocess_photo_input
                img, photo_diag = preprocess_photo_input(img)
                self.image = img
                # Re-snapshot the original because suppress passes
                # need the post-deskew coordinates to match.
                self._original_image = img.copy()
                logger.info("Photo preprocessing: %s", photo_diag)
            except Exception as exc:
                logger.warning("Photo preprocessing failed (%s) — "
                               "continuing with original image", exc)
            _step_end("0b. Photo-input preprocessing")

        # ── STEP 0: Screenshot normalisation (opt-in) ────────────
        # Client uploads are often screenshots of CAD output rather
        # than the original raster — they're JPEG-compressed, sub-
        # resolution, sometimes warm-tinted or slightly rotated.
        # The normaliser auto-detects this and applies a small fix-up
        # pass before the rest of the pipeline runs. Gated on the
        # config flag so callers can A/B-test it without changing default
        # behaviour.
        if self.cfg.enable_screenshot_normalization:
            _step_begin("0. Screenshot normalisation")
            try:
                try:
                    from .screenshot_normalizer import normalize_screenshot
                except ImportError:
                    from screenshot_normalizer import normalize_screenshot
                img, norm_info = normalize_screenshot(img, filename=None)
                self.image = img
                logger.info("Screenshot normalisation: %s",
                            {"detected": norm_info.get("detected_screenshot"),
                             "ops": [o.get("op") for o in norm_info.get("ops", [])
                                     if not o.get("skipped")]})
            except Exception as exc:
                logger.warning("Screenshot normalisation failed (%s) — "
                               "continuing with original image.", exc)
            _step_end("0. Screenshot normalisation")
        else:
            _step_skip("0. Screenshot normalisation", "disabled in config")

        # ── STEP 0b: Auto-upscale tiny inputs ────────────────────
        # HoughCircles and the OCR ensemble both need a minimum
        # pixel budget — balloon edges, leader lines, and 1–2 digit
        # bubble numbers all collapse below ~10 px. Tiny uploads
        # (small JPEGs, low-DPI scans) silently return zero bubbles
        # because every downstream filter rejects what survives.
        # Upscale once with bicubic so the rest of the pipeline has
        # enough signal. Original-image references are kept in sync
        # so `_low_chroma_in_original` still indexes the right
        # coordinates.
        MIN_SHORT_SIDE = 800
        h0, w0 = img.shape[:2]
        short_side = min(w0, h0)
        self._was_upscaled = False
        if short_side < MIN_SHORT_SIDE:
            _step_begin("0b. Auto-upscale tiny input")
            scale = MIN_SHORT_SIDE / float(short_side)
            new_w = int(round(w0 * scale))
            new_h = int(round(h0 * scale))
            logger.info(
                "Auto-upscaling tiny input %sx%s -> %sx%s (factor %.2fx)",
                w0, h0, new_w, new_h, scale,
            )
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            self.image = img
            self._original_image = img.copy()
            self._was_upscaled = True
            _step_end("0b. Auto-upscale tiny input")

        # ── STEP 0c: Contrast & saturation boost (removed) ────────
        # Removed: applying CLAHE + saturation boost + unsharp mask
        # to the working image hurt OCR more than it helped. The
        # enhanced character shapes confused digit recognition on clean scans.
        # Re-introduce as a split-stream pass (enhanced copy for
        # circle detection only, original for OCR) when reworking
        # the pipeline for color-restricted detection (Step 3).
        _step_skip("0c. Enhance contrast & saturation", "deferred to split-stream pass")

        # ── STEP 0d: Real-ESRGAN enhancement (heuristic-gated) ────
        # The model helps cluttered/thin-stroke drawings but can distort
        # clean drawings; OCR on the 4x upscaled output may misread
        # well-rendered digits and symbols.
        # Solution: only run when the quality assessment flags the
        # image as cluttered. Clean drawings bypass the model and stay on the
        # fast baseline path.
        if self._enhancer is not None:
            q = self._quality or {}
            # Three independent triggers, all structural (no per-image
            # tuning):
            #   1. is_high_clutter   — cluttered drawings with tables,
            #      BOM, notes (clutter_score > 100). Original signal.
            #   2. is_low_resolution AND is_low_contrast — phone photo
            #      of a monitor screen. Tiny (short side < 800 px) AND
            #      washed-out (gray_std < 40). Clean scans have at
            #      least one of those False, so the AND keeps them out.
            #   3. is_low_resolution AND clutter_score > 50 — tiny
            #      input with moderate text density (catches small
            #      screenshot crops where contrast survived).
            is_screenshot_photo = (
                bool(q.get("is_low_resolution"))
                and (
                    bool(q.get("is_low_contrast"))
                    or (q.get("clutter_score", 0.0) or 0.0) > 50.0
                )
            )
            needs_enhancement = (
                bool(q.get("is_high_clutter"))
                or is_screenshot_photo
            )
            if needs_enhancement:
                _step_begin("0d. Real-ESRGAN enhancement")
                try:
                    pre_h, pre_w = img.shape[:2]
                    enhanced = self._enhancer.enhance(img)
                    logger.info("Real-ESRGAN: %dx%d -> %dx%d (clutter=%s)",
                                pre_w, pre_h, enhanced.shape[1], enhanced.shape[0],
                                (self._quality or {}).get("clutter_score"))
                    img = enhanced
                    self.image = img
                    self._original_image = img.copy()
                    # Mark as upscaled so downstream gates that key off
                    # this flag (eff_p2 floor, dedup_dist floor, red-blob
                    # fallback) treat it like a tiny-input upscale.
                    self._was_upscaled = True
                except Exception as exc:
                    logger.warning("Real-ESRGAN failed (%s) — continuing with "
                                   "original image.", exc)
                _step_end("0d. Real-ESRGAN enhancement")
            else:
                _step_skip("0d. Real-ESRGAN enhancement",
                           f"clutter={q.get('clutter_score')} "
                           f"low_res={q.get('is_low_resolution')} "
                           f"low_contrast={q.get('is_low_contrast')} "
                           f"— not flagged as cluttered or screenshot-photo")
        else:
            _step_skip("0d. Real-ESRGAN enhancement",
                       "disabled in config" if not self.cfg.enable_realesrgan_enhancement
                       else "enhancer failed to initialise")

        _step_begin("1. Preprocess image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        logger.info("Image size: %sx%s", w, h)

        # Preprocess: create a clean binarized version for OCR
        # and leader line detection. The original is kept for
        # color-based detection (annotation layer, color circles).
        try:
            from preprocess import preprocess_drawing
            self._preprocessed = preprocess_drawing(img)
        except Exception:
            self._preprocessed = None

        # Extract annotation color layer (balloons + leader lines + text
        # all share the same chromatic color — purple, red, blue, etc.)
        self._annotation_layer = extract_annotation_layer(img)
        if self._annotation_layer and self.cfg.print_timing:
            al = self._annotation_layer
            print(f"[0] Annotation layer: H={al.dominant_hue} "
                  f"S={al.dominant_sat} conf={al.confidence:.2f} "
                  f"pixels={al.pixel_count}")

        # Resolve auto parameters
        ocr_scale = self.cfg.ocr_scale if self.cfg.ocr_scale > 0 else _auto_ocr_scale(w, h)
        eff_min_r, eff_max_r, auto_p2, auto_dist = _auto_hough_params(w, h)
        if self.cfg.min_radius > 0:
            eff_min_r = self.cfg.min_radius
        if self.cfg.max_radius > 0:
            eff_max_r = self.cfg.max_radius
        eff_p2   = self.cfg.hough_param2 if self.cfg.hough_param2 > 0 else auto_p2
        eff_dist = self.cfg.min_dist     if self.cfg.min_dist > 0     else auto_dist
        # When we upscaled a tiny/compressed source, JPEG ringing
        # artefacts produce hundreds of false-positive Hough circles
        # at the default permissive param2 (~22-26). Bumping the floor
        # to 38 collapses that noise while still finding real balloon
        # rings on tiny/compressed images, where the loose default can produce
        # hundreds of candidates and push downstream passes into long runtimes.
        # Lower values (28, 32) interact non-linearly with the multi-
        # method circle pipeline and yield *fewer* surviving bubbles;
        # missed balloons in this regime are picked up by the red-
        # blob fallback further down rather than by loosening Hough.
        if self._was_upscaled and self.cfg.hough_param2 <= 0:
            eff_p2 = max(eff_p2, 38)
        # Distances scale with the upscale factor: two near-duplicate
        # circles that were ~12 px apart in the source become ~30 px
        # apart at 2.4x, which slips past the default dedup_dist=25
        # and produces a "ghost" balloon next to a real one. Bumping
        # to 40 collapses these pairs without merging genuinely
        # separate balloons.
        self._eff_dedup_dist = self.cfg.dedup_dist
        if self._was_upscaled:
            self._eff_dedup_dist = max(self.cfg.dedup_dist, 40)
        _step_end("1. Preprocess image")

        # ── STEP 1: Full-image OCR ────────────────────────────────
        # RapidOCR for main tokens. Then augment with PaddleOCR +
        # Florence-2 for BUBBLE IDs only — these engines catch
        # balloon numbers that RapidOCR misses. Only bubble-like
        # tokens (1-2 digit numbers) are added to avoid polluting
        # the dimension callout pool.
        t = time.perf_counter()
        _step_begin("2. OCR — full image")
        # Run OCR on original image
        raw_tokens = self._run_ocr(img, ocr_scale)
        _step_end("2. OCR — full image")

        # Augment with rotated-image OCR — catches diagonal and
        # vertical dimension text (e.g. "20.41" angled callouts,
        # "Ø110" / "Ø134.6 ±0.1" written along vertical leader
        # lines) that upright OCR on the sharpened variant misses.
        # IMPORTANT: rotated tokens are filtered strictly so they
        # don't pollute the callout grouper with partial reads. We
        # only accept tokens that look like a COMPLETE dimension
        # (has Ø/0-prefix + digits, OR decimal point, OR ± sign),
        # and only if they don't spatially duplicate an upright
        # token. Without this filter the grouper merges rotated
        # fragments with unrelated upright values.
        _step_begin("3. OCR — rotated + ensemble passes")
        try:
            run_rotated_fullpage = (
                self.cfg.enable_rotated_fullpage_ocr
                or self.cfg.enable_photo_preprocessing
                or self.cfg.enable_targeted_endpoint_ocr
            )
            rotated_tokens = (
                self._ocr_rotated_pass(img) if run_rotated_fullpage else []
            )
            for rt in rotated_tokens:
                txt = rt.text.strip()
                # Must look like a real dimension token, not a
                # fragment or stray digit
                looks_complete = (
                    re.search(r"\d+\.\d+", txt) is not None   # has decimal
                    or "±" in txt                              # has tolerance sign
                    or re.search(r"^[Øø0]\d{2,}", txt)         # diameter prefix
                    or re.search(r"\bØ\d", txt)
                )
                if not looks_complete:
                    continue
                # Reject if it duplicates a nearby upright token
                near = any(
                    math.dist((rt.cx, rt.cy), (ut.cx, ut.cy)) <= 40.0
                    for ut in raw_tokens
                )
                if near:
                    continue
                raw_tokens.append(rt)
        except Exception as e:
            logger.warning("Rotated OCR pass failed: %s", e)

        if self.cfg.enable_optional_ocr_ensemble:
            try:
                from ocr_ensemble import _run_paddle, _run_florence
                existing_positions = {
                    (int(t.cx), int(t.cy)) for t in raw_tokens
                }
                for engine_fn in [_run_paddle, _run_florence]:
                    try:
                        results = engine_fn(img)
                    except Exception:
                        continue
                    for r in results:
                        txt = r.text.strip()
                        # Only add bubble-like tokens: 1-2 digits, optional letter
                        if not re.fullmatch(r"\d{1,2}[A-Za-z]?", txt):
                            continue
                        # Skip if already near an existing token
                        near = any(
                            abs(r.cx - ex) < 25 and abs(r.cy - ey) < 25
                            for ex, ey in existing_positions
                        )
                        if near:
                            continue
                        raw_tokens.append(OCRToken(
                            text=txt, cx=r.cx, cy=r.cy, conf=r.confidence,
                            x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2,
                        ))
                        existing_positions.add((int(r.cx), int(r.cy)))
            except ImportError:
                pass
        _step_end("3. OCR — rotated + ensemble passes")

        if self.cfg.print_timing:
            print(f"[1] OCR: {time.perf_counter()-t:.2f}s | tokens={len(raw_tokens)}")

        # ── STEP 2: Normalise tokens ──────────────────────────────
        t = time.perf_counter()
        _step_begin("4. Normalise OCR tokens")
        norm_tokens = normalize_ocr_tokens(raw_tokens, image_w=w, image_h=h)
        # ── Split-stream tagging ──────────────────────────────────
        # Tag each token by checking its bbox against the annotation
        # colour mask (red/maroon/pink/magenta/purple). A token is
        # maroon-stream if EITHER:
        #   (a) ≥3% of its bbox pixels are annotation-coloured, or
        #   (b) its centre is within a maroon-stroke-dilated region
        # The two-rule combination catches both fully-rendered
        # balloon digits (where the digit ink is maroon) AND digits
        # rendered in black inside a maroon balloon (where only the
        # surrounding ring is maroon but the token centre sits in
        # the ring's interior, surrounded by maroon).
        # Downstream filters consult `token.is_maroon` to route the
        # token to the right consumer — this is how we keep table
        # noise out of bubble identification.
        ann_mask = self._annotation_hsv_mask(img)
        # Dilate the mask generously so the center-pixel rule also
        # fires for digits sitting in the empty interior of a
        # balloon circle (centre is a few px from the maroon stroke).
        center_mask = cv2.dilate(
            ann_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
        )
        n_maroon = 0
        for t_tok in norm_tokens:
            x1 = max(0, int(t_tok.x1))
            y1 = max(0, int(t_tok.y1))
            x2 = min(w, int(t_tok.x2))
            y2 = min(h, int(t_tok.y2))
            if x2 <= x1 or y2 <= y1:
                t_tok.is_maroon = False
                continue
            patch = ann_mask[y1:y2, x1:x2]
            overlap_pct = float(patch.sum()) / (255.0 * patch.size) if patch.size else 0.0
            cx_i = max(0, min(w - 1, int(t_tok.cx)))
            cy_i = max(0, min(h - 1, int(t_tok.cy)))
            center_hit = bool(center_mask[cy_i, cx_i])
            t_tok.is_maroon = overlap_pct >= 0.03 or center_hit
            if t_tok.is_maroon:
                n_maroon += 1
        logger.info("Split-stream tagging: %d/%d tokens flagged is_maroon",
                    n_maroon, len(norm_tokens))
        self._norm_tokens = norm_tokens
        _step_end("4. Normalise OCR tokens")
        if self.cfg.print_timing:
            print(f"[2] Normalize: {time.perf_counter()-t:.2f}s | norm={len(norm_tokens)}")

        # ── STEP 3: Detect balloon shapes ─────────────────────────
        # Attempted a CLAHE + saturation + denoise side-copy here
        # to improve thin/faded balloon detection. At every param
        # regime (aggressive AND mild), the extra circles found via
        # enhancement caused false-positive bubble assignments on
        # working cases by adding false circles and shifting assignments.
        # Reverted to detecting on the original image. The right
        # next architectural step is to enhance, find circles,
        # then ONLY accept new circles that the original image's
        # tint check also confirms — not implemented yet.
        t = time.perf_counter()
        _step_begin("5. Detect balloon circles")
        circles = self._find_circles(gray, eff_min_r, eff_max_r, eff_p2, eff_dist)

        # Augment with color-based circle detection
        color_circles = self._find_color_circles(
            img, eff_min_r, eff_max_r, existing=circles
        )
        circles.extend(color_circles)

        # Third source — aggressive annotation-mask pass for cluttered
        # drawings. The Hough + _find_color_circles
        # passes both reject thin-stroke balloons (Hough because the
        # gradient is too weak, _find_color_circles because the
        # broken-stroke contours don't enclose enough area). This
        # pass closes stroke gaps with a bigger kernel and relaxes
        # the area filter so partially-rendered rings still produce
        # candidate circles.
        mask_circles = self._find_circles_from_annotation_mask(
            img, eff_min_r, eff_max_r, existing=circles
        )
        circles.extend(mask_circles)
        _step_end("5. Detect balloon circles")

        if self.cfg.print_timing:
            print(f"[3] Circles: {time.perf_counter()-t:.2f}s | n={len(circles)}")

        # ── STEP 4: Read bubble numbers ───────────────────────────
        t = time.perf_counter()
        _step_begin("6. Read bubble numbers")
        budget_s = max(1.0, float(self.cfg.max_step6_recovery_seconds or 0.0))
        self._step6_recovery_deadline = time.perf_counter() + budget_s
        self._step6_budget_logged = False
        with _step("6a. Identify primary bubble labels"):
            bubbles, used_token_ids = self._identify_bubbles(circles, norm_tokens, w, h, img)
        with _step("6b. Recover OCR-token bubbles"):
            self._find_uncircled_bubbles(bubbles, norm_tokens, used_token_ids, circles)
            self._find_edge_bubbles(bubbles, norm_tokens, used_token_ids, img)
        with _step("6c. Recover edge/evidence bubbles"):
            self._find_edge_blob_bubbles(bubbles, norm_tokens, used_token_ids, img)
            self._find_evidence_based_bubbles(bubbles, norm_tokens, used_token_ids, img)
        with _step("6d. Recover colored-circle labels"):
            self._recover_colored_circle_label_bubbles(
                bubbles, norm_tokens, used_token_ids, circles, img,
            )
        if self._should_run_annotation_ring_recovery(bubbles, img):
            with _step("6e. Recover annotation-ring labels"):
                self._recover_annotation_ring_label_bubbles(
                    bubbles, norm_tokens, used_token_ids, circles, img,
                )
        else:
            _step_skip(
                "6e. Recover annotation-ring labels",
                "primary bubble evidence already sufficient",
            )
        with _step("6f. Recover red/blob labels"):
            self._find_red_blob_bubbles(bubbles, norm_tokens, used_token_ids, img)
        # Mask-based circle discovery: for photo-quality input the
        # primary Hough on the gray edge map misses balloons whose
        # rims got softened by camera blur. The maroon mask (boosted
        # for photo input) still shows them clearly, so we run
        # HoughCircles directly ON the mask as a recall booster.
        # Gated to photo input — adds nothing on clean scans.
        if self.cfg.enable_photo_preprocessing or self._strict_photo_mode():
            with _step("6g. Recover mask-circle bubbles"):
                self._find_mask_circle_bubbles(bubbles, norm_tokens,
                                               used_token_ids, img)
        bubbles = self._remove_concatenated_ids(bubbles)
        before_ui = len(bubbles)
        bubbles = self._suppress_ui_table_artifact_bubbles(bubbles)
        if len(bubbles) < before_ui:
            logger.info(
                "Pre-assignment UI/table suppression dropped %d bubble(s)",
                before_ui - len(bubbles),
            )
        self._step6_recovery_deadline = None
        _step_end("6. Read bubble numbers")
        if self.cfg.print_timing:
            print(f"[4] Bubbles: {time.perf_counter()-t:.2f}s | n={len(bubbles)}")

        # Scale calibration — done once after bubbles are detected
        scale = _compute_scale_factor([float(b.radius) for b in bubbles])
        eff_assoc_dist = int(self.cfg.max_assoc_distance * scale)
        eff_assoc_cost = self.cfg.max_assoc_cost * scale
        logger.info("Scale factor: %.2f | eff_assoc_dist=%d", scale, eff_assoc_dist)

        # ── STEP 5: Callout regions ───────────────────────────────
        if True:  # pipeline block
            t = time.perf_counter()
            _step_begin("7. Group OCR tokens into callouts")
            callout_groups = self._build_callouts(norm_tokens, used_token_ids, bubbles, scale)
            _step_end("7. Group OCR tokens into callouts")
            if self.cfg.print_timing:
                print(f"[5] Callouts: {time.perf_counter()-t:.2f}s | n={len(callout_groups)}")

            # ── STEP 6: Leader segments ───────────────────────────────
            t = time.perf_counter()
            _step_begin("8. Extract leader-line segments")
            leader_segments = extract_path_segments(img)
            _step_end("8. Extract leader-line segments")
            if self.cfg.print_timing:
                print(f"[6] Leader segs: {time.perf_counter()-t:.2f}s | n={len(leader_segments)}")

            # ── STEP 6b: Suppress phantom single-digit tokens ─────────
            # Uses HoughLinesP segment endpoints to distinguish real
            # dimension digits from OCR phantom reads.
            _step_begin("9. Suppress phantom digits + rebuild callouts")
            norm_tokens = self._suppress_phantom_digits(
                norm_tokens, bubbles, leader_segments,
            )
            self._norm_tokens = norm_tokens
            # Rebuild callout groups with cleaned tokens
            callout_groups = self._build_callouts(
                norm_tokens, used_token_ids, bubbles, scale,
            )
            # Expose to downstream suppression passes so they can
            # consult the structural type (thread/diameter/chamfer/...)
            # of a dim text instead of guessing from raw characters.
            self._callout_groups = callout_groups
            _step_end("9. Suppress phantom digits + rebuild callouts")


            # ── Pre-trace bubble discovery (moved up from Steps 12-13) ──
            # Find ALL bubbles before tracing + Hungarian, so the
            # cost matrix and trace step see the complete bubble
            # population. Previously these ran after Hungarian, which
            # meant late-discovered bubbles got distance-only fallback
            # assignments — wrong on drawings whose initial Hough only
            # caught a fraction of the bubbles.

            # Reverse leader discovery: for callouts whose centre is
            # not near any existing bubble, search for missed bubbles.
            # The pre-trace version uses geometric "no nearby bubble"
            # instead of "callout text not in any bubble.dimension"
            # (which depended on Hungarian having already run).
            _step_begin("9b. Reverse leader discovery (pre-trace)")
            self._reverse_discover_bubbles(
                bubbles, callout_groups, leader_segments,
                gray, img, eff_min_r, eff_max_r, eff_p2, eff_dist,
                use_pre_hungarian_filter=True,
            )
            _step_end("9b. Reverse leader discovery (pre-trace)")

            # Likelihood scan: annotation-tinted rings missed by Hough.
            if self._should_run_likelihood_scan(bubbles, img):
                _step_begin("9c. Likelihood scan (pre-trace)")
                self._likelihood_scan_bubbles(
                    bubbles, callout_groups, img, gray, eff_assoc_dist,
                )
                _step_end("9c. Likelihood scan (pre-trace)")
            else:
                _step_skip(
                    "9c. Likelihood scan (pre-trace)",
                    "primary bubble evidence already sufficient",
                )

            # ── STEPS 7-8: Seed detection + path tracing ──────────────
            leader_directions: Dict[int, Tuple[float, float]] = {}
            if self.cfg.enable_seed_trace_assignment:
                t = time.perf_counter()
                _step_begin("10. Seed detection + leader tracing")
                self._seed_traces, leader_directions = self._trace_balloon_leader_paths(
                    bubbles=bubbles,
                    callout_groups=callout_groups,
                )
                _step_end("10. Seed detection + leader tracing")
                if self.cfg.print_timing:
                    print(f"[7-8] Seed+trace: {time.perf_counter()-t:.2f}s | "
                          f"traced={len(self._seed_traces)}")
            else:
                _step_skip("10. Seed detection + leader tracing", "disabled in config")

            if (
                self.cfg.enable_leader_first_assignment
                and len(bubbles) <= self.cfg.max_bubbles_for_leader_first_assignment
            ):
                _step_begin("10b. Leader-first candidate assignment")
                self._assign_leader_first_candidates(bubbles, callout_groups)
                _step_end("10b. Leader-first candidate assignment")
            else:
                reason = (
                    "disabled in config"
                    if not self.cfg.enable_leader_first_assignment
                    else "deferred until dense trace-quality gates are stricter"
                )
                _step_skip("10b. Leader-first candidate assignment", reason)

            # ── STEP 9: Global optimal assignment (Hungarian) ─────────
            # Now runs with the COMPLETE bubble population AND complete
            # leader-trace data — Hungarian sees the full picture in
            # one shot, no late-arriving bubbles to patch around.
            t = time.perf_counter()
            _step_begin("11. Optimal bubble→callout assignment")
            self._optimal_assign(
                bubbles, callout_groups, leader_directions,
                eff_assoc_dist, eff_assoc_cost,
            )
            _step_end("11. Optimal bubble→callout assignment")
            if self.cfg.print_timing:
                print(f"[9] Optimal assign: {time.perf_counter()-t:.2f}s")

                # Note: we previously tried re-running the Hungarian
                # here ("13d") to take advantage of the supplementary
                # trace data, but this perturbed already-correct
                # assignments on drawings where Step 11's Hungarian
                # got things right. Smart-assign (Step 15) and
                # propagation (Step 16) already pick up the new
                # trace data for bubbles that didn't get a dim from
                # Step 11, so no re-Hungarian is needed.

            # ── Pre-rescue: drop unmatched bubbles on B&W background ──
            # Gated to screenshot mode. Spurious Hough circles in the
            # price-table area survive Step 13 with NO_DIMENSION; the
            # rescue then prioritises them (NO_DIMENSION first) and
            # they steal recovered text that a nearby real bubble
            # should have received. Killing them here lets the rescue
            # assign the recovered dim to the real bubble downstream.
            if (self.cfg.enable_screenshot_normalization
                    and self.cfg.enable_targeted_endpoint_ocr):
                _step_begin("13b. Suppress low-chroma unmatched bubbles")
                before = len(bubbles)
                bubbles = self._suppress_low_chroma_unmatched_pre_rescue(bubbles)
                if len(bubbles) < before:
                    logger.info(
                        "Pre-rescue suppression dropped %d unmatched bubble(s)",
                        before - len(bubbles),
                    )
                _step_end("13b. Suppress low-chroma unmatched bubbles")

            # ── Targeted OCR at trace endpoints for small text ────────
            # Opt-in — costs ~5s per request for a marginal accuracy
            # gain; enable only when the added latency is acceptable.
            auto_targeted_ocr = (
                self._strict_photo_mode()
                and any(
                    not b.dimension or b.dimension == "NO_DIMENSION"
                    or (b.needs_review and b.confidence < 0.35)
                    for b in bubbles
                )
            )
            if self.cfg.enable_targeted_endpoint_ocr or auto_targeted_ocr:
                _step_begin("14. Targeted endpoint OCR")
                self._targeted_endpoint_ocr(bubbles, img, callout_groups)
                _step_end("14. Targeted endpoint OCR")
            else:
                _step_skip("14. Targeted endpoint OCR", "disabled in config")

            # ── Post-processing: smart assign any remaining NO_DIMENSION ─
            _step_begin("15. Smart-assign unresolved bubbles")
            self._smart_assign_unresolved(bubbles, callout_groups, eff_assoc_dist)
            _step_end("15. Smart-assign unresolved bubbles")

            # ── Post-processing: shared dimension propagation ─────────
            _step_begin("16. Propagate shared dimensions")
            self._propagate_shared_dimensions(bubbles)
            _step_end("16. Propagate shared dimensions")


        # ── Post-assignment dimension normalization ───────────────
        _step_begin("17. Normalise assigned dimensions")
        self._normalize_assigned_dimensions(bubbles)
        _step_end("17. Normalise assigned dimensions")

        # ── VLM verification pass ─────────────────────────────────
        # Uses a local vision model to verify/correct each bubble's
        # dimension by cropping around the bubble and asking the VLM
        # what dimension the leader line points to.
        if self.cfg.enable_vlm_verification:
            t = time.perf_counter()
            _step_begin("18. VLM verification pass")
            self._vlm_verify_and_correct(bubbles, img)
            _step_end("18. VLM verification pass")
            if self.cfg.print_timing:
                print(f"[VLM] Verify: {time.perf_counter()-t:.2f}s")
        else:
            _step_skip("18. VLM verification pass", "disabled in config")

        # ── Confidence assessment ─────────────────────────────────
        _step_begin("19. Assess confidence per bubble")
        for bubble in bubbles:
            self._assess_confidence(bubble)
        _step_end("19. Assess confidence per bubble")

        # ── Suppress suspicious false-positive bubbles ────────────
        # Catches the failure modes that show up on noisy / screenshot
        # inputs where the bubble-discovery pathways (evidence-based,
        # annotation-blob) latch onto text that isn't a real balloon:
        #
        #   (a) the dim text contains long alphabetic runs that no
        #       engineering shorthand explains — e.g. "9TrEr / L9'(REF)";
        #   (b) the bubble label is just a fragment of its own dim
        #       text (e.g. bubble '59' assigned dim '59.5' — the
        #       label is the leading digits of the dim string that
        #       happen to sit inside a spurious Hough circle).
        #
        # Only fires on low-confidence bubbles (needs_review=True) so
        # high-confidence real bubbles are never touched.
        _step_begin("19b. Suppress suspicious false-positives")
        before = len(bubbles)
        bubbles = self._suppress_suspicious_bubbles(bubbles)
        if len(bubbles) < before:
            logger.info(
                "Suppressed %d suspicious bubble(s) on final pass",
                before - len(bubbles),
            )
        _step_end("19b. Suppress suspicious false-positives")

        # Clear orphan single-digit dims — runs before the proximity
        # fallback so the fallback gets a chance to re-fill them.
        _step_begin("19c. Clear orphan single-digit dims")
        self._clear_weak_orphan_dims(bubbles)
        _step_end("19c. Clear orphan single-digit dims")

        # Proximity fallback — for any bubble that finished the pipeline
        # with an empty dimension (either never assigned, cleared by
        # the axis-label rule, or cleared as an orphan single-digit),
        # borrow the nearest dim-shaped token within ~5× radius. Runs
        # AFTER suppression so a freshly-available dim text isn't
        # wasted on a bubble we're about to drop.
        _step_begin("19d. Proximity fallback for empty dimensions")
        self._proximity_fallback_for_empty_dims(bubbles)
        _step_end("19d. Proximity fallback for empty dimensions")

        # Late fallbacks can assign raw OCR text after step 17 has already
        # normalised dimensions. Run the same generic cleanups once more at
        # the output boundary so all assignment paths produce consistent text.
        _step_begin("19e. Normalise late assigned dimensions")
        self._normalize_assigned_dimensions(bubbles)
        _step_end("19e. Normalise late assigned dimensions")

        _step_begin("19e0. Assign keyword note callouts")
        self._assign_keyword_note_callouts(bubbles, callout_groups)
        _step_end("19e0. Assign keyword note callouts")

        _step_begin("19e0a. Assign coordinate leader callouts")
        self._assign_coordinate_leader_callouts(bubbles)
        _step_end("19e0a. Assign coordinate leader callouts")

        _step_begin("19e0b. Correct weak endpoint assignments")
        self._correct_weak_assignments_from_trace_endpoint(bubbles, callout_groups)
        _step_end("19e0b. Correct weak endpoint assignments")

        _step_begin("19e1. Repair omitted dimension prefixes")
        self._repair_omitted_dimension_prefixes(bubbles)
        _step_end("19e1. Repair omitted dimension prefixes")

        if self.cfg.enable_assignment_geometry_validation:
            _step_begin("19e2. Validate assignment geometry")
            self._validate_assignment_geometry(bubbles, callout_groups)
            _step_end("19e2. Validate assignment geometry")
        else:
            _step_skip("19e2. Validate assignment geometry", "disabled in config")

        _step_begin("19e3. OCR traced leader endpoints")
        self._assign_traced_endpoint_ocr_callouts(bubbles)
        _step_end("19e3. OCR traced leader endpoints")

        _step_begin("19e4. Rescue local angle callouts")
        self._assign_local_angle_callouts(bubbles)
        _step_end("19e4. Rescue local angle callouts")

        _step_begin("19e5. Rescue plain numeric leader callouts")
        self._assign_local_plain_numeric_callouts(bubbles)
        _step_end("19e5. Rescue plain numeric leader callouts")

        _step_begin("19e5a. Refine weak local OCR decimals")
        self._refine_low_conf_local_rescue_decimals(bubbles, callout_groups)
        _step_end("19e5a. Refine weak local OCR decimals")

        _step_begin("19e5b. Targeted rotated leader OCR")
        self._assign_targeted_rotated_leader_ocr(bubbles, callout_groups)
        _step_end("19e5b. Targeted rotated leader OCR")

        _step_begin("19e6. Correct late endpoint assignments")
        self._correct_weak_assignments_from_trace_endpoint(bubbles, callout_groups)
        _step_end("19e6. Correct late endpoint assignments")

        _step_begin("19e7. Clear final axis-label assignments")
        self._clear_axis_label_assignments(bubbles)
        _step_end("19e7. Clear final axis-label assignments")

        _step_begin("19e8. Clear table-description assignments")
        self._clear_table_description_assignments(bubbles)
        _step_end("19e8. Clear table-description assignments")

        _step_begin("19e8a. Strict OCR endpoints after clearing table text")
        self._assign_traced_endpoint_ocr_callouts(bubbles, strict_rich_only=True)
        _step_end("19e8a. Strict OCR endpoints after clearing table text")

        # ── Annotation ────────────────────────────────────────────
        # Red-blob recovery can preserve an unreadable balloon location as
        # "#?" for manual inspection. Do not emit that as an extracted bubble:
        # it is a hint, not a real balloon id, and it pollutes automated counts.
        _step_begin("19f. Suppress invalid bubble ids")
        before = len(bubbles)
        bubbles = [
            b for b in bubbles
            if re.fullmatch(r"\d{1,3}[A-Za-z]?", str(b.bubble_number or "").strip())
        ]
        if len(bubbles) < before:
            logger.info(
                "Suppressed %d bubble(s) with invalid placeholder ids",
                before - len(bubbles),
            )
        _step_end("19f. Suppress invalid bubble ids")

        _step_begin("19g. Suppress UI/table overlay artifacts")
        before = len(bubbles)
        bubbles = self._suppress_ui_table_artifact_bubbles(bubbles)
        if len(bubbles) < before:
            logger.info(
                "Suppressed %d UI/table artifact bubble(s)",
                before - len(bubbles),
            )
        _step_end("19g. Suppress UI/table overlay artifacts")

        _step_begin("19h. Suppress tiny OCR text artifacts")
        before = len(bubbles)
        bubbles = self._suppress_tiny_text_artifact_bubbles(bubbles)
        if len(bubbles) < before:
            logger.info(
                "Suppressed %d tiny OCR text artifact bubble(s)",
                before - len(bubbles),
            )
        _step_end("19h. Suppress tiny OCR text artifacts")

        _step_begin("19i. Validate circular annotation-rim evidence")
        before = len(bubbles)
        bubbles = self._suppress_non_circular_bubble_artifacts(bubbles)
        if len(bubbles) < before:
            logger.info(
                "Suppressed %d non-circular bubble artifact(s)",
                before - len(bubbles),
            )
        _step_end("19i. Validate circular annotation-rim evidence")

        _step_begin("19j. Suppress overlapping duplicate bubble candidates")
        before = len(bubbles)
        bubbles = self._suppress_overlapping_bubble_candidates(bubbles)
        if len(bubbles) < before:
            logger.info(
                "Suppressed %d overlapping duplicate bubble candidate(s)",
                before - len(bubbles),
            )
        _step_end("19j. Suppress overlapping duplicate bubble candidates")

        _step_begin("20. Annotate output image")
        annotated = self._annotate(img, bubbles, circles, callout_groups)
        _step_end("20. Annotate output image")

        total = time.perf_counter() - t0
        if self.cfg.print_timing:
            print(f"[TOTAL] {total:.2f}s | bubbles={len(bubbles)}")

        return bubbles, annotated

    # ── OCR ────────────────────────────────────────────────────────

    def _ocr_rotated_pass(
        self, img: np.ndarray,
        angles: Tuple[int, ...] = (-45, 45, -90, 90),
    ) -> List[OCRToken]:
        """Run OCR on the image rotated by each angle and map detected
        bboxes back to the upright coordinate frame.

        Catches diagonal dimension text (e.g. "20.41" at 30-45°) and
        vertical dimension text (e.g. "Ø134.6 ±0.1" along a vertical
        leader line) that the main upright-OCR pass misses because
        RapidOCR's sharpen-preprocessed variant fragments small
        rotated glyphs.

        Returns OCRTokens with bbox coords mapped back to the
        original image frame so the callout grouper can use them
        directly alongside upright tokens.
        """
        h, w = img.shape[:2]
        out: List[OCRToken] = []
        for angle in angles:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
            nw = int(h * sin_a + w * cos_a)
            nh = int(h * cos_a + w * sin_a)
            M[0, 2] += (nw - w) / 2
            M[1, 2] += (nh - h) / 2
            rot = cv2.warpAffine(
                img, M, (nw, nh),
                borderValue=(255, 255, 255),
            )
            inv_M = cv2.invertAffineTransform(M)
            try:
                result = self.ocr(rot)
            except Exception:
                continue
            items = result[0] if (isinstance(result, tuple) and len(result) >= 1) else (result or [])
            for item in items or []:
                try:
                    bbox = item[0]
                    text_info = item[1]
                    text = str(text_info[0] if isinstance(text_info, (list, tuple)) else text_info).strip()
                    conf = float(text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.9)
                    if not text or conf < self.cfg.min_ocr_conf:
                        continue
                    # Only keep tokens that look like a dimension
                    # value — we don't want rotated-image artifacts
                    # polluting the bubble-number detection.
                    if not re.search(r"\d", text):
                        continue
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

    def _rotated_pass_ocr(
        self,
        img: np.ndarray,
        effective_scale: int,
        all_tokens: List[OCRToken],
    ) -> None:
        """Run OCR on a 90°-rotated copy of `img` and add any new
        engineering-shaped tokens to `all_tokens` in original-image
        coordinates.

        Engineering drawings often label vertical dimensions with text
        rotated 90° (e.g. "59.5" running up a wall). RapidOCR's
        text-detection model is trained for upright text; rotated
        labels need an explicit rotated pass. We filter to
        engineering-looking text only (decimal point, unit symbol,
        or ≤3 chars) so the pass doesn't add table-content noise.
        """
        try:
            h_orig = img.shape[0]
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        except Exception as exc:
            logger.debug("Rotated OCR rotate() failed: %s", exc)
            return

        scales = [1]
        if effective_scale > 1 and max(rotated.shape[:2]) <= 1500:
            scales.append(effective_scale)

        for rotated_scale in scales:
            try:
                if rotated_scale > 1:
                    img_proc = cv2.resize(
                        rotated, None, fx=rotated_scale, fy=rotated_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    img_proc = rotated
                result = self.ocr(img_proc)
            except Exception:
                continue
            if not result:
                continue
            items = result[0] if isinstance(result, tuple) else result
            if not items:
                continue
            for item in items:
                try:
                    if len(item) < 2:
                        continue
                    bbox_points = item[0]
                    text_info = item[1]
                    if isinstance(text_info, (list, tuple)):
                        text = str(text_info[0]).strip()
                        conf = float(text_info[1]) if len(text_info) > 1 else 0.9
                    else:
                        text = str(text_info).strip()
                        conf = float(item[2]) if len(item) > 2 else 0.9
                    if not text or conf < self.cfg.min_ocr_conf:
                        continue
                    # Filter to engineering-looking text only.
                    has_decimal = "." in text
                    has_unit_or_symbol = bool(
                        re.search(r"[°±ØΦ%]|R\d|M\d|MM\b", text, re.I)
                    )
                    is_short = len(text) <= 3
                    is_long_digits_only = (
                        len(text) >= 4
                        and re.fullmatch(r"\d+", text) is not None
                    )
                    if is_long_digits_only:
                        continue
                    if not (has_decimal or has_unit_or_symbol or is_short):
                        continue

                    # Inverse 90°-CW rotation: rotated_x → original_y,
                    # rotated_y → (h_orig - 1 - original_x).
                    if hasattr(bbox_points[0], "__iter__"):
                        xs_rot = [float(p[0]) / rotated_scale for p in bbox_points]
                        ys_rot = [float(p[1]) / rotated_scale for p in bbox_points]
                    else:
                        xs_rot = [float(bbox_points[0]) / rotated_scale,
                                  float(bbox_points[2]) / rotated_scale]
                        ys_rot = [float(bbox_points[1]) / rotated_scale,
                                  float(bbox_points[3]) / rotated_scale]
                    xs_orig = ys_rot
                    ys_orig = [h_orig - 1 - x for x in xs_rot]
                    x1, y1 = min(xs_orig), min(ys_orig)
                    x2, y2 = max(xs_orig), max(ys_orig)
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    # Skip if positionally overlaps an existing upright
                    # token — only add genuinely new vertical reads.
                    overlap = False
                    for t in all_tokens:
                        if t.x1 <= cx <= t.x2 and t.y1 <= cy <= t.y2:
                            overlap = True
                            break
                    if overlap:
                        continue
                    all_tokens.append(OCRToken(
                        text=text, cx=cx, cy=cy, conf=conf,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                    ))
                except Exception:
                    continue

    def _run_ocr(
        self, img: np.ndarray, effective_scale: int,
        single_scale: bool = False,
        rescue_missed: bool = False,
    ) -> List[OCRToken]:
        """
        Run RapidOCR at effective_scale and optionally also at scale=1.

        Multi-scale ensures large-font bubble numbers and small dimension text
        are both captured.

        `single_scale=True` skips the scale=1 fallback pass — used by
        the rescue OCR path where the crop is already small and a
        single scale pass is sufficient. Cuts rescue OCR latency in
        half (no duplicate pass at scale=1 on already-small crops).

        `rescue_missed=True` runs an extra pass that scans for small
        isolated ink components NOT yet covered by an OCR token and
        re-OCRs them (multi-zoom, multi-preprocessing). Catches
        single-digit dim values like "1", "3" that RapidOCR's
        text-detection model rejects on its own. Only enabled for
        the FORWARD auto-annotate path; the reverse-detection path
        runs on drawings that already have balloons drawn, where
        the rescue would generate phantom "1"s on rendered leader
        lines and pollute the bubble→dim mapping.
        """
        requested_scales = [effective_scale]
        if (not single_scale) and self.cfg.run_multi_scale_ocr and effective_scale > 1:
            requested_scales.append(1)

        h0, w0 = img.shape[:2]
        scales: List[int] = []
        for scale in requested_scales:
            actual_scale = scale
            if max(h0, w0) > 1500 and scale > 1:
                actual_scale = 1
            elif max(h0, w0) > 1000 and scale > 2:
                actual_scale = 2
            if actual_scale not in scales:
                scales.append(actual_scale)
        logger.info(
            "OCR scales: requested=%s actual=%s image=%dx%d single_scale=%s",
            requested_scales,
            scales,
            w0,
            h0,
            single_scale,
        )

        all_tokens: List[OCRToken] = []

        def _has_enough_primary_ocr(tokens: List[OCRToken]) -> bool:
            if len(tokens) < 12:
                return False
            digit_tokens = 0
            dimension_like = 0
            for tok in tokens:
                text = (tok.text or "").strip()
                if re.search(r"\d", text):
                    digit_tokens += 1
                if (
                    re.search(r"\d+[.,]\d+", text)
                    or re.search(r"[ØΦÂ±±°º˚×X/]", text)
                    or re.search(r"\b(?:REF|TYP|THRU|DIA|M\d|R\d)\b", text.upper())
                ):
                    dimension_like += 1
            return digit_tokens >= 8 and dimension_like >= 1

        for scale_index, actual_scale in enumerate(scales):
            if (
                scale_index > 0
                and self.cfg.enable_adaptive_multiscale_ocr
                and _has_enough_primary_ocr(all_tokens)
            ):
                logger.info(
                    "Adaptive OCR: skipped fallback scale %s after %d primary tokens",
                    actual_scale,
                    len(all_tokens),
                )
                continue
            h, w = img.shape[:2]
            if actual_scale > 1:
                img_scaled = cv2.resize(img, None, fx=actual_scale, fy=actual_scale,
                                        interpolation=cv2.INTER_CUBIC)
            else:
                img_scaled = img

            # Build multiple preprocessing variants to handle different
            # image qualities: clean digital, phone photos with noise/moiré,
            # low contrast scans.
            gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY) if len(img_scaled.shape) == 3 else img_scaled

            variants: List[np.ndarray] = []

            # Variant 1: CLAHE + bilateral + sharpen (standard)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            variants.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))

            # Variant 2 (punctuation-only recovery): the same CLAHE +
            # bilateral but WITHOUT the sharpen step. The sharpen kernel
            # fragments small multi-char tokens containing "±" or "."
            # into pieces. Running a non-sharpened pass and only keeping
            # tokens containing those characters recovers the damaged
            # ones without polluting the pool with duplicate bare
            # digits that hurt reference-suite accuracy.
            use_recovery_variants = (
                self.cfg.enable_fullpage_ocr_recovery_variants
                or self.cfg.enable_photo_preprocessing
                or self.cfg.enable_targeted_endpoint_ocr
                or self._strict_photo_mode()
            )
            if use_recovery_variants:
                variants.append(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))

            # Variant 3 (untouched color recovery): pass the scaled
            # color image directly — no grayscale conversion, no
            # filtering. RapidOCR's text detection is measurably
            # sensitive to color channel information on small rotated
            # text (empirically, "±0,2" on T1 is only found in color
            # mode). ±-filtered downstream so this can't pollute.
                variants.append(img_scaled
                                if len(img_scaled.shape) == 3
                                else cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR))

            for variant_idx, img_proc in enumerate(variants):
                # Variant 0 = aggressive sharpen (primary).
                # Variant 1+ = recovery variants — only keep tokens
                # that contain characters the sharpen kernel is known
                # to destroy. This prevents the gentler variants from
                # polluting the pool with duplicate bare digits that
                # hurt assignment accuracy on reference images.
                is_recovery = variant_idx >= 1
                try:
                    result = self.ocr(img_proc)
                except Exception as e:
                    logger.warning("RapidOCR failed at scale %d: %s", actual_scale, e)
                    continue

                if not result or not isinstance(result, (list, tuple)):
                    continue

                # RapidOCR output: list of [bbox_points, text, conf] triples
                items = result[0] if (isinstance(result, tuple) and len(result) >= 1) else result
                if not items:
                    continue

                for item in items:
                    try:
                        if len(item) < 2:
                            continue
                        bbox_points = item[0]
                        text_info   = item[1]

                        if isinstance(text_info, (list, tuple)):
                            text = str(text_info[0]).strip()
                            conf = float(text_info[1]) if len(text_info) > 1 else 0.9
                        else:
                            text = str(text_info).strip()
                            conf = float(item[2]) if len(item) > 2 else 0.9

                        if not text or conf < self.cfg.min_ocr_conf:
                            continue

                        # Recovery-variant filter: only accept tokens
                        # containing "±" (the most commonly destroyed
                        # symbol). Everything else is better handled
                        # by variant 0.
                        if is_recovery and "±" not in text:
                            continue

                        if hasattr(bbox_points[0], '__iter__'):
                            xs = [float(p[0]) / actual_scale for p in bbox_points]
                            ys = [float(p[1]) / actual_scale for p in bbox_points]
                        else:
                            xs = [float(bbox_points[0]) / actual_scale,
                                  float(bbox_points[2]) / actual_scale]
                            ys = [float(bbox_points[1]) / actual_scale,
                                  float(bbox_points[3]) / actual_scale]

                        all_tokens.append(OCRToken(
                            text=text,
                            cx=sum(xs) / len(xs),
                            cy=sum(ys) / len(ys),
                            conf=conf,
                            x1=min(xs), y1=min(ys),
                            x2=max(xs), y2=max(ys),
                        ))
                    except Exception:
                        continue

        # ── Rotated-pass OCR for vertical / sideways text ──
        # Engineering drawings often label vertical dimensions with
        # text rotated 90° (e.g. "59.5" running up a wall, "48.7"
        # running down a column). Gated behind the photo-input flag
        # because the extra OCR pass can shift borderline reads on
        # already-clean scan inputs.
        # When you opt into photo preprocessing you're saying "I
        # have noisy/skewed input, accept more aggressive recovery."
        if self.cfg.enable_photo_preprocessing:
            self._rotated_pass_ocr(img, effective_scale, all_tokens)

        # Rescue pass: small isolated ink components NOT already
        # covered by an OCR token are often missed single-digit dims
        # (e.g. a lone "1" between two arrows). Only enabled when
        # `rescue_missed=True` (forward auto-annotate path) — the
        # reverse-detection path operates on drawings with rendered
        # balloons whose colored leader lines would produce phantom
        # "1" tokens.
        if rescue_missed:
            rescue = self._rescue_missed_digit_tokens(img, all_tokens)
            if rescue:
                all_tokens.extend(rescue)

        return self._dedup_ocr_tokens(all_tokens)

    def _rescue_missed_digit_tokens(
        self, img: np.ndarray, existing: List[OCRToken],
    ) -> List[OCRToken]:
        """Re-OCR small uncovered ink components to recover missed
        single-digit dimensions. Returns NEW tokens to merge in."""
        h, w = img.shape[:2]

        # 1. Coverage mask from existing token bboxes (padded a bit so
        # arrow-stems adjacent to a token don't trigger the rescue).
        covered = np.zeros((h, w), dtype=np.uint8)
        for tok in existing:
            x1 = max(0, int(tok.x1) - 8)
            y1 = max(0, int(tok.y1) - 8)
            x2 = min(w, int(tok.x2) + 8)
            y2 = min(h, int(tok.y2) + 8)
            covered[y1:y2, x1:x2] = 255

        # 2. Binarised "drawing ink" mask = pixels that are dark on
        # ALL THREE channels (genuine black/dark-grey original
        # drawing content), masked to UN-covered regions only.
        # Pixels that are RED (annotation balloons + leader lines)
        # have high R but low G/B — they look dark in plain
        # grayscale and would be picked up as "ink" by Otsu, but
        # they are NOT real dim text. Without this filter the
        # rescue pass finds phantom "1"s on every rendered leader
        # line and pollutes detection output with false bubbles
        # like "81", "15", "82".
        if len(img.shape) == 3:
            b_ch, g_ch, r_ch = cv2.split(img)
            ink = ((b_ch < 110) & (g_ch < 110) & (r_ch < 110)).astype(np.uint8) * 255
        else:
            _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        uncovered = cv2.bitwise_and(ink, cv2.bitwise_not(covered))

        # 3. Connected components — keep only ones with digit-like
        # shape: small width/height, decent fill ratio (so we skip
        # long thin arrow stems), reasonable aspect.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered, connectivity=8)
        candidates: List[Tuple[int, int, int, int]] = []
        # Estimate digit-size range from existing token heights so we
        # don't pick up labels much smaller/larger than real dim text.
        if existing:
            heights = sorted(int(tok.y2 - tok.y1) for tok in existing)
            median_h = heights[len(heights) // 2]
            # ±60% around the median dim text height — narrow enough
            # to skip noise, wide enough to cover the missed digit.
            min_h = max(6, int(median_h * 0.4))
            max_h = max(min_h + 8, int(median_h * 1.6))
        else:
            min_h, max_h = 8, 35

        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if not (min_h <= ch <= max_h):
                continue
            if not (4 <= cw <= max_h):  # width similar to height (digits)
                continue
            # Reject extremely sparse components (likely arrow lines)
            fill = area / max(1, cw * ch)
            if fill < 0.18:
                continue
            # Reject extremely tall thin slivers (single-pixel verticals)
            if cw < 3 or ch / cw > 6.0:
                continue
            candidates.append((int(x), int(y), int(cw), int(ch)))

        if not candidates:
            return []

        # 4. Group nearby candidates into a single crop region (so a
        # multi-digit number split into per-stroke components becomes
        # one OCR target). Simple: merge components whose bboxes are
        # within `merge_gap` px on either axis.
        merge_gap = max(8, min_h // 2)
        merged: List[Tuple[int, int, int, int]] = []
        for cand in candidates:
            x, y, cw, ch = cand
            placed = False
            for j, (mx, my, mw, mh) in enumerate(merged):
                if (x < mx + mw + merge_gap
                    and x + cw + merge_gap > mx
                    and y < my + mh + merge_gap
                    and y + ch + merge_gap > my):
                    nx = min(x, mx)
                    ny = min(y, my)
                    nx2 = max(x + cw, mx + mw)
                    ny2 = max(y + ch, my + mh)
                    merged[j] = (nx, ny, nx2 - nx, ny2 - ny)
                    placed = True
                    break
            if not placed:
                merged.append(cand)

        # 5. For each merged region, try several preprocessing
        # variants × zoom levels until OCR returns something. Tight
        # crops fail RapidOCR's text-detection model — the model
        # needs whitespace context. Heavy padding + binarized
        # versions of the crop typically unlock single-digit reads
        # (3, 5, 8) that the standard pipeline misses.
        rescued: List[OCRToken] = []
        # Padding chosen to give RapidOCR's text-detector enough
        # whitespace context WITHOUT pulling in adjacent bubbles or
        # dim text. ~half a digit height of margin works on every
        # tested drawing; 1.5× was too loose (crops included
        # neighboring bubble numbers, producing reads like "81"
        # for bubble 8 + a stray "1" rescue candidate).
        pad = max(10, min_h // 2)

        def _crop_variants(crop_bgr: np.ndarray):
            """Yield (label, image) preprocessing variants for OCR."""
            yield "raw", crop_bgr
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
            # Otsu threshold (clean black-on-white)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            yield "binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            # Slightly dilated (thickens thin digit strokes)
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.erode(binary, kernel, iterations=1)  # erode on white→thicker dark strokes
            yield "dilated", cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        for (rx, ry, rw, rh) in merged:
            x0 = max(0, rx - pad)
            y0 = max(0, ry - pad)
            x1 = min(w, rx + rw + pad)
            y1 = min(h, ry + rh + pad)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            items = None
            scale = 6
            # Try variants × zoom levels — first match wins.
            outer_break = False
            for variant_label, variant_img in _crop_variants(crop):
                if outer_break:
                    break
                for try_scale in (6, 8, 10):
                    scaled = cv2.resize(variant_img, None, fx=try_scale, fy=try_scale,
                                        interpolation=cv2.INTER_CUBIC)
                    try:
                        result = self.ocr(scaled)
                    except Exception:
                        continue
                    if not result:
                        continue
                    trial = result[0] if (isinstance(result, tuple) and len(result) >= 1) else result
                    if trial and any(
                        re.search(r"\d", str(t[1][0] if isinstance(t[1], (list, tuple)) else t[1]))
                        for t in trial if len(t) >= 2
                    ):
                        items = trial
                        scale = try_scale
                        logger.info(
                            "OCR rescue: matched on variant=%s zoom=%dx for (%d,%d %dx%d)",
                            variant_label, try_scale, rx, ry, rw, rh,
                        )
                        outer_break = True
                        break

            if not items:
                # Last-resort heuristic: emit a low-confidence "1"
                # for a tall-thin isolated component, but ONLY if
                # there's whitespace to its immediate left AND right
                # (so it's clearly between dim arrows, not an arrow
                # stem connected to a dim line).
                if 8 <= rw <= 18 and rh / max(1, rw) >= 2.2 and rh / max(1, rw) <= 4.5:
                    # Isolation check: sample at upper and lower
                    # thirds (deliberately AVOID the middle, where a
                    # horizontal dim line would cross both sides of
                    # a "1" between arrows). True digit "1"s have
                    # whitespace at their top + bottom thirds; arrow
                    # stems extend vertically through.
                    isolation = max(15, int(rh * 0.5))
                    left_x = max(0, rx - isolation)
                    right_x = min(w, rx + rw + isolation)
                    rows = [ry + rh // 5, ry + (4 * rh) // 5]
                    rows = [r for r in rows if 0 <= r < h]
                    left_clear = right_clear = True
                    for r in rows:
                        if ink[r, left_x:rx].any():
                            left_clear = False
                            break
                        if ink[r, rx + rw:right_x].any():
                            right_clear = False
                            break
                    if left_clear and right_clear:
                        cx = rx + rw / 2
                        cy = ry + rh / 2
                        rescued.append(OCRToken(
                            text="1",
                            cx=cx, cy=cy,
                            conf=0.4,
                            x1=rx, y1=ry,
                            x2=rx + rw, y2=ry + rh,
                        ))
                        logger.info(
                            "OCR rescue: assumed '1' for isolated component at (%d,%d %dx%d)",
                            rx, ry, rw, rh,
                        )
                continue
            for item in items:
                try:
                    if len(item) < 2:
                        continue
                    bbox_points = item[0]
                    text_info   = item[1]
                    if isinstance(text_info, (list, tuple)):
                        text = str(text_info[0]).strip()
                        conf = float(text_info[1]) if len(text_info) > 1 else 0.5
                    else:
                        text = str(text_info).strip()
                        conf = 0.5
                    # Keep only tokens that actually contain a digit
                    # AND aren't junk (e.g. '|', '.', '-'). 0.30 conf
                    # threshold — lower than the main pass since the
                    # crop is denoiser-clean and the rescue is a last
                    # resort.
                    if not text or conf < 0.30:
                        continue
                    if not re.search(r"\d", text):
                        continue
                    if hasattr(bbox_points[0], '__iter__'):
                        xs = [float(p[0]) / scale + x0 for p in bbox_points]
                        ys = [float(p[1]) / scale + y0 for p in bbox_points]
                    else:
                        xs = [float(bbox_points[0]) / scale + x0,
                              float(bbox_points[2]) / scale + x0]
                        ys = [float(bbox_points[1]) / scale + y0,
                              float(bbox_points[3]) / scale + y0]
                    rescued.append(OCRToken(
                        text=text,
                        cx=sum(xs) / len(xs),
                        cy=sum(ys) / len(ys),
                        conf=conf,
                        x1=min(xs), y1=min(ys),
                        x2=max(xs), y2=max(ys),
                    ))
                except Exception:
                    continue
        if rescued:
            logger.info("OCR rescue: %d new tokens %s",
                        len(rescued),
                        [t.text for t in rescued][:8])
        return rescued

    def _dedup_ocr_tokens(self, tokens: List[OCRToken], pos_eps: float = 14.0) -> List[OCRToken]:
        out: List[OCRToken] = []
        for token in sorted(tokens, key=lambda t: (-t.conf, t.cy, t.cx)):
            keep = True
            for existing in out:
                same_text = token.text.strip().upper() == existing.text.strip().upper()
                close = math.dist((token.cx, token.cy), (existing.cx, existing.cy)) <= pos_eps
                if same_text and close:
                    keep = False
                    break
            if keep:
                out.append(token)
        return out

    # ── Circle Detection ───────────────────────────────────────────

    def _find_circles(
        self,
        gray: np.ndarray,
        min_radius: int,
        max_radius: int,
        param2: int,
        min_dist: int,
    ) -> List[Tuple[int, int, int]]:
        from geometric_utils import find_circular_contours, fit_ellipse_to_contour
        h, w = gray.shape[:2]
        all_circles: List[Tuple[int, int, int]] = []

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Contour circles (backup)
        contour_circles = find_circular_contours(gray, min_area=min_radius**2 * 0.3, min_circularity=0.55)
        all_circles.extend(contour_circles)
        
        # Ellipse fits (occluded circles)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_radius**2 * 0.2:
                continue
            if len(cnt) < 5:
                continue
            circ = (4 * math.pi * cv2.contourArea(cnt)) / (cv2.arcLength(cnt, True) ** 2)
            if circ < 0.5:
                continue
            cx, cy, equiv_r, _ = fit_ellipse_to_contour(cnt)
            if 8 < equiv_r < max_radius:
                all_circles.append((int(round(cx)), int(round(cy)), int(round(equiv_r))))

        # Adaptive preprocessing
        img_mean = np.mean(gray)
        img_std  = np.std(gray)

        if img_mean < 80:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
            enhanced = clahe.apply(gray)
        elif img_std > 60:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced = cv2.bilateralFilter(enhanced, 9, 80, 80)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Multi-scale Hough with adaptive parameters.
        # Reduced to 2x2 = 4 total passes (was 3x4 = 12).
        # The median-radius filter downstream discards duplicate circles,
        # so extra passes provided diminishing returns at high cost.
        scales = [1.0, 0.8] if min(h, w) > 800 else [1.0]
        # param2 deltas: +2/+4 are the strict passes (clean drawings).
        # -3 is a RELAXED pass that catches soft-edged rings on
        # JPEG-compressed or noisy inputs (where strict Hough loses
        # rings whose gradient strength was reduced by compression).
        # The downstream tint check + token-cluster guard will reject
        # any false positives produced by the relaxed pass, so this
        # is purely additive recall without precision cost.
        param_sets = [(1.0, 2), (1.0, 4), (1.0, -3)]
        # When photo-input preprocessing is on, add a SOFTER pass too:
        # phone-camera blur and screen-photo moiré attenuate ring edges
        # further than ordinary JPEG noise. The extra (-6) pass picks
        # up rings the -3 strict-noise pass still missed. Same downstream
        # filters keep false positives in check.
        if self.cfg.enable_photo_preprocessing:
            param_sets = param_sets + [(1.0, -6), (1.2, -3)]

        for scale in scales:
            if scale != 1.0:
                sh, sw = int(h * scale), int(w * scale)
                scaled_gray = cv2.resize(enhanced, (sw, sh))
                inv_scale = 1.0 / scale
            else:
                scaled_gray = enhanced
                inv_scale = 1.0

            blur_size = 5 if min(scaled_gray.shape[:2]) < 500 else 7
            blurred = cv2.GaussianBlur(scaled_gray, (blur_size, blur_size), 2)

            for dp, p2_delta in param_sets:
                base_p2 = max(18, param2 + p2_delta)
                if min(blurred.shape[:2]) < 400:
                    base_p2 = max(14, base_p2 - 4)

                # Use computed min/max radius (not hardcoded!)
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=dp,
                    minDist=int(min_dist / max(scale, 0.5)),
                    param1=self.cfg.hough_param1,
                    param2=base_p2,
                    minRadius=int(min_radius * scale),
                    maxRadius=int(max_radius * scale),
                )
                if circles is None:
                    continue
                for c in circles[0]:
                    sx = int(round(c[0] * inv_scale))
                    sy = int(round(c[1] * inv_scale))
                    sr = int(round(c[2] * inv_scale))
                    if sx - sr >= 0 and sy - sr >= 0 and sx + sr < w and sy + sr < h:
                        all_circles.append((sx, sy, sr))

        # Deduplicate + ellipse fits
        unique: List[Tuple[int, int, int]] = []
        for c in all_circles:
            dup = any(
                math.dist(c[:2], u[:2]) < self._eff_dedup_dist and abs(c[2] - u[2]) < 10
                for u in unique
            )
            if not dup:
                unique.append(c)

        return unique[:self.cfg.max_candidate_circles]


    def _find_color_circles(

        self,
        image: np.ndarray,
        min_radius: int,
        max_radius: int,
        existing: List[Tuple[int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """
        Detect annotation-layer circles using HSV colour filtering.

        Engineering drawings often use a distinct hue (purple, magenta,
        blue, red) for balloons/bubbles.  When the drawing has a
        chromatic annotation layer, filter by hue → find contours →
        fit enclosing circles.  Only returns circles not already in
        *existing* (dedup by centre proximity).

        Returns an empty list if the image is achromatic or no new
        circles are found.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Try several common annotation hue bands (purple, magenta, blue, red).
        # For each band: build mask, find contours, filter by circularity.
        hue_bands = [
            # (name, lo_h, hi_h, min_sat)  —  OpenCV H: 0–180
            # Low min_sat (15) to catch faded/scanned annotations
            ("purple",  120, 160, 15),
            ("magenta", 145, 175, 15),
            ("blue",    90,  130, 20),
            ("red_lo",  0,   10,  15),
            ("red_hi",  170, 180, 15),
        ]

        candidates: List[Tuple[int, int, int]] = []

        for _name, lo_h, hi_h, min_sat in hue_bands:
            lower = np.array([lo_h, min_sat, 50], dtype=np.uint8)
            upper = np.array([hi_h, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            # Skip if too few or too many pixels match (noise or full-image match)
            pix_count = int(np.count_nonzero(mask))
            total_pix = h * w
            if pix_count < 50 or pix_count > total_pix * 0.3:
                continue

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 1:
                    continue
                # Circularity: 4π·area / perimeter² ≈ 1.0 for perfect circle
                circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
                if circularity < 0.55:
                    continue

                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(radius))

                if r_i < min_radius or r_i > max_radius:
                    continue

                # Check not already detected
                is_dup = any(
                    math.dist((cx_i, cy_i), (ex, ey)) < self._eff_dedup_dist
                    for ex, ey, _ in existing
                )
                if is_dup:
                    continue

                candidates.append((cx_i, cy_i, r_i))

        # Dedup among candidates
        unique: List[Tuple[int, int, int]] = []
        for c in candidates:
            dup = any(math.dist(c[:2], u[:2]) < self._eff_dedup_dist for u in unique)
            if not dup:
                unique.append(c)

        return unique

    def _find_circles_from_annotation_mask(
        self,
        image: np.ndarray,
        min_radius: int,
        max_radius: int,
        existing: List[Tuple[int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """Aggressive annotation-mask circle pass for cluttered inputs.

        `_find_color_circles` uses a 3x3 close kernel and rejects
        contours under 200 px enclosed area. On drawings with thin
        balloon strokes, broken
        stroke fragments fail both filters and yield zero circles.

        This pass widens the bridge kernel to 7x7 with 3 iterations
        so multi-pixel stroke gaps are mended, and drops the area
        floor to 80 px so partially-rendered rings still register.
        Contours that survive get the same circularity + radius
        gates as `_find_color_circles` so we don't dilute the
        candidate pool with rectangular table cells or text blobs.
        """
        h, w = image.shape[:2]
        mask = self._annotation_hsv_mask(image)
        # The shared helper already applies a 5x5 close; layer on
        # another 7x7 close and an opening to discard speckle.
        big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big_kernel, iterations=3)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[int, int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue
            circularity = (4.0 * math.pi * area) / (peri * peri)
            if circularity < 0.55:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(radius))
            if r_i < min_radius or r_i > max_radius:
                continue
            # Skip if already detected by Hough or _find_color_circles
            if any(math.dist((cx_i, cy_i), (ex, ey)) < self._eff_dedup_dist
                   for ex, ey, _ in existing):
                continue
            candidates.append((cx_i, cy_i, r_i))

        # Dedup against ourselves
        unique: List[Tuple[int, int, int]] = []
        for c in candidates:
            if not any(math.dist(c[:2], u[:2]) < self._eff_dedup_dist for u in unique):
                unique.append(c)
        if unique:
            logger.info("Annotation-mask circle pass: +%d new candidates", len(unique))
        return unique

    # ── Bubble Identification ──────────────────────────────────────

    def _identify_bubbles(
        self,
        circles: List[Tuple[int, int, int]],
        norm_tokens: List[NormalizedToken],
        w: int, h: int,
        img: np.ndarray,
    ) -> Tuple[List[BubbleResult], set]:
        bubbles: List[BubbleResult] = []
        used_token_ids: set = set()
        used_circles: set = set()   # (cx, cy) already claimed by a bubble
        seen: set = set()

        # Estimate the plausible bubble radius from tight token-circle pairs.
        # _auto_hough_params permits radii up to ~15% of the short image side
        # (e.g. 154 on a 1600x1027 drawing), so drawing-feature shapes can
        # slip in as circles.  A legitimate bubble's OCR token sits within a
        # few pixels of the circle centre — collecting those establishes the
        # real bubble scale and lets us reject oversize "circles" later.
        tight_radii: List[int] = []
        for t in norm_tokens:
            if getattr(t, "dual_use", False):
                continue
            if not is_bubble_token(t.text):
                continue
            for (cx, cy, cr) in circles:
                if math.dist((t.cx, t.cy), (cx, cy)) <= 15:
                    tight_radii.append(cr)
                    break
        if tight_radii:
            tight_radii.sort()
            median_bubble_r = tight_radii[len(tight_radii) // 2]
            max_plausible_r = max(int(median_bubble_r * 2.5), 60)
        else:
            median_bubble_r = None
            max_plausible_r = None

        circle_grid_cell = max(
            32,
            int((max((c[2] for c in circles), default=20) * 2) + 24),
        )
        circle_grid: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        for c in circles:
            cx, cy, _cr = c
            key = (int(cx // circle_grid_cell), int(cy // circle_grid_cell))
            circle_grid.setdefault(key, []).append(c)

        def _nearby_circles_for_point(
            px: float,
            py: float,
            *,
            radius_hint: Optional[float] = None,
        ) -> List[Tuple[int, int, int]]:
            gx = int(px // circle_grid_cell)
            gy = int(py // circle_grid_cell)
            max_r = float(radius_hint if radius_hint is not None else max_plausible_r or circle_grid_cell)
            search = max(1, int(math.ceil((max_r * 1.4 + 80.0) / circle_grid_cell)))
            out: List[Tuple[int, int, int]] = []
            for yy in range(gy - search, gy + search + 1):
                for xx in range(gx - search, gx + search + 1):
                    out.extend(circle_grid.get((xx, yy), []))
            return out

        # Find tokens inside circles
        for idx, t in enumerate(norm_tokens):
            if idx in used_token_ids:
                continue
            if getattr(t, "dual_use", False):
                continue
            if not is_bubble_token(t.text):
                continue
            # Split-stream gate: a real bubble label sits on annotation-
            # coloured ink (maroon / magenta). Black-on-white text that
            # *looks* like a 1-2 digit number is dimension / callout
            # content and must never become a bubble — even when it
            # happens to sit inside a Hough circle (e.g. parenthesised
            # numeric callout text where the parens form a small Hough
            # circle and promote the callout to a phantom bubble).
            if not getattr(t, "is_maroon", False):
                continue
            norm = normalize_bubble_value(t.text)
            # "0" is never a real bubble number; empty string means
            # normalize_bubble_value received a pure-zero token ("0", "00")
            # which would create BubbleResult(bubble_number="") — an orphan row.
            if not norm or norm == "0":
                continue
            if norm in seen:
                continue

            best_circle = None
            best_dist   = float("inf")
            for (cx, cy, cr) in _nearby_circles_for_point(
                float(t.cx),
                float(t.cy),
                radius_hint=float(max_plausible_r or circle_grid_cell),
            ):
                if (cx, cy) in used_circles:
                    continue   # circle already claimed — skip
                # Skip oversize circles that aren't bubble-shaped: drawing
                # features (gears, holes, countersinks) can be detected by
                # Hough but absorb far-away dimension-digit tokens like the
                # "8" of "(8)" or the "53" of "Ø53" through their inflated
                # match tolerance.
                if max_plausible_r is not None and cr > max_plausible_r:
                    continue
                if self._circle_too_clipped(img, cx, cy, cr):
                    continue
                if not self._bubble_label_scale_ok(t, cr):
                    continue
                d = math.dist((t.cx, t.cy), (cx, cy))
                # Tolerance scales with radius: phone photos / perspective
                # distortion can shift OCR text position relative to circle.
                match_tol = cr * 1.3 + 10
                if d <= match_tol and d < best_dist:
                    # Every circle must pass at least the basic annotation
                    # tint check — this filters false circles from HoughCircles
                    # that coincide with dimension text or drawing geometry.
                    if not self._circle_has_annotation_tint(img, cx, cy, cr):
                        continue
                    # Stricter check when token is near/outside the edge
                    if d > cr * 0.9:
                        if not self._circle_has_annotation_tint(img, cx, cy, cr, strict=True):
                            continue
                    best_dist   = d
                    best_circle = (cx, cy, cr)

            if best_circle is None:
                continue

            cx, cy, cr = best_circle

            # ── Dimension-collision guard (main pass) ─────────────────────
            # Dimension text can look circular enough for HoughCircles and
            # annotation tint checks to pass.
            # Reuse the same collision logic as the recovery pass.
            if self._recovered_id_collides_with_dimension(
                norm, cx, cy, cr, norm_tokens, used_token_ids
            ):
                continue

            # ── Token-cluster guard ───────────────────────────────────
            # A real annotation bubble has ONE digit token inside it
            # with whitespace around. If DIMENSION-like tokens (±, Ø,
            # decimals, long strings) pack within ~1.5× the radius of
            # this candidate, the circle almost certainly sits on top
            # of a dimension text block (e.g. "25 ±0,2" reading as
            # ["25","2","±0.2","+1"] clustered together) and the
            # chosen "bubble number" is actually a fragment of that
            # dimension. Reject so the true red annotation circle
            # elsewhere can claim this number.
            # Do NOT count sibling bubble-number digit tokens: OCR
            # frequently splits a 2-digit balloon ("14") into "1" +
            # "4", and both end up inside the same circle — that's
            # the expected pattern for a bubble, not a dimension.
            cluster_radius = max(cr * 1.5, 20.0)
            dim_token_count = 0
            for ot in norm_tokens:
                if ot is t:
                    continue
                if math.dist((ot.cx, ot.cy), (cx, cy)) > cluster_radius:
                    continue
                # Ignore tokens that could plausibly be fragments of
                # this same bubble's number (1-2 digit numerics or
                # ID suffix letters).
                if is_bubble_token(ot.text):
                    continue
                dim_token_count += 1
            if dim_token_count >= 2:
                continue

            # ── Multi-digit fragment merge ────────────────────────────
            # OCR occasionally splits a 2-digit balloon ("14") into
            # "1" + "4" when the glyphs are narrow or the drawing is
            # compressed. If other small digit tokens sit INSIDE this
            # same circle, gather them and concatenate left-to-right
            # to form the real bubble number, then mark them all used.
            merged_norm = norm
            merged_token_ids = [idx]
            inside_tokens = []
            for oi, ot in enumerate(norm_tokens):
                if oi == idx or oi in used_token_ids:
                    continue
                if getattr(ot, "dual_use", False):
                    continue
                if not is_bubble_token(ot.text):
                    continue
                if math.dist((ot.cx, ot.cy), (cx, cy)) > cr:
                    continue
                inside_tokens.append((oi, ot))
            if inside_tokens:
                # Include the matched token, sort all by x, concatenate
                all_frags = [(t.cx, idx, t.text)] + [
                    (ot.cx, oi, ot.text) for oi, ot in inside_tokens
                ]
                all_frags.sort(key=lambda e: e[0])
                combined = "".join(frag[2] for frag in all_frags)
                # Only accept if the combined result is still a valid
                # bubble token (prevents merging with stray unrelated
                # digits that happened to fall within the radius).
                if is_bubble_token(combined):
                    merged_norm = normalize_bubble_value(combined)
                    merged_token_ids = [frag[1] for frag in all_frags]
                    if merged_norm in seen:
                        # Collision with already-claimed bubble — skip
                        continue

            bubbles.append(BubbleResult(
                bubble_number=merged_norm,
                x=cx,
                y=cy,
                radius=cr,
                confidence=float(t.conf),
                dimension="NO_DIMENSION",
            ))
            seen.add(merged_norm)
            for mi in merged_token_ids:
                used_token_ids.add(mi)
            used_circles.add((cx, cy))

        # Recovery pass: try OCR inside unmatched circles
        if self.cfg.enable_circle_recovery_ocr and not self._skip_heavy_annotation_recovery():
            unmatched = [c for c in circles if (c[0], c[1]) not in used_circles]
            # Apply the same plausibility filter to recovery — reading OCR
            # inside a giant drawing-feature "circle" will just return a
            # random digit picked up from text that happens to fall inside
            # the crop window.
            if max_plausible_r is not None:
                unmatched = [c for c in unmatched if c[2] <= max_plausible_r]
            # Pre-filter: drop circles whose rim has no annotation tint.
            # On dense engineering drawings Hough returns dozens of
            # phantom circles from drawing geometry. Without this
            # filter the recovery budget gets consumed by non-balloon
            # circles, and real maroon balloons further down the list
            # never get an OCR pass. The downstream loop also has a
            # tint check (defence in depth) but pre-filtering ensures
            # the budget reaches the real candidates first.
            unmatched = [
                c for c in unmatched
                if self._circle_has_annotation_tint(img, c[0], c[1], c[2])
            ]
            def _recovery_priority(c: Tuple[int, int, int]) -> Tuple[float, float]:
                cx, cy, cr = c
                try:
                    evidence = self._compute_bubble_evidence(img, cx, cy, cr)
                except Exception:
                    evidence = 0.0
                return (-evidence, float(cr))

            unmatched.sort(key=_recovery_priority)
            budget = max(0, int(self.cfg.max_unmatched_circle_recovery))
            for (cx, cy, cr) in unmatched[:budget]:
                # Recovery OCR can "invent" a bubble from any readable text
                # inside a circular region — e.g. a corner watermark or
                # drawing-title mark.  Require annotation-colour evidence
                # on the rim so only genuine balloon-shaped marks qualify.
                if not self._circle_has_annotation_tint(img, cx, cy, cr):
                    continue
                if self._circle_too_clipped(img, cx, cy, cr):
                    continue
                # If the main OCR pass already classified text at this
                # location as NON-maroon (black-on-white), it's dim /
                # callout content sitting inside an incidental Hough
                # circle (e.g. parenthesised numeric callout text where the
                # parens form a circle). Skip recovery OCR here so we don't
                # promote dim text to a phantom bubble.
                has_non_maroon_text_at_circle = False
                for ot in norm_tokens:
                    if getattr(ot, "is_maroon", False):
                        continue
                    if math.dist((ot.cx, ot.cy), (cx, cy)) > cr:
                        continue
                    if not is_bubble_token(ot.text):
                        continue
                    has_non_maroon_text_at_circle = True
                    break
                if has_non_maroon_text_at_circle:
                    continue
                result = self._recover_bubble_from_circle(img, cx, cy, cr)
                if result is None:
                    continue
                norm_text, conf = result
                if norm_text in seen:
                    continue
                # Reject if recovered ID looks like a nearby dimension value.
                # e.g. circle at (252,516) reads "30" but it's actually "3.0".
                if self._recovered_id_collides_with_dimension(
                    norm_text, cx, cy, cr, norm_tokens, used_token_ids
                ):
                    continue
                bubbles.append(BubbleResult(
                    bubble_number=norm_text,
                    x=cx, y=cy, radius=cr,
                    confidence=conf,
                    dimension="NO_DIMENSION",
                ))
                seen.add(norm_text)
                used_circles.add((cx, cy))
        elif self.cfg.enable_circle_recovery_ocr:
            logger.info(
                "Unmatched-circle OCR recovery: skipped "
                "(low-saturation high-clutter annotation layer)"
            )

        return bubbles, used_token_ids

    def _find_uncircled_bubbles(
        self,
        bubbles: List[BubbleResult],
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
        circles: List[Tuple[int, int, int]],
    ) -> None:
        seen = {b.bubble_number for b in bubbles}

        for idx, t in enumerate(norm_tokens):
            if idx in used_token_ids:
                continue
            if getattr(t, "dual_use", False):
                continue
            if not getattr(t, "is_maroon", False):
                # Split-stream: black-on-white text (dimensions,
                # table cells, title-block entries) cannot be a
                # bubble number even when its text shape looks like
                # a 1-2 digit integer. The annotation HSV mask
                # filtered this token out as non-maroon.
                continue
            if not is_bubble_token(t.text):
                continue
            norm = normalize_bubble_value(t.text)
            if not norm or norm == "0":
                continue
            if norm in seen:
                continue

            # Only near an UNCLAIMED circle — a circle already assigned to a
            # detected bubble must not be re-used as a promotion target.
            # Require the token to be INSIDE an annotation-tinted circle
            # (within 1.3× circle radius of its center).  Previously we
            # accepted any circle within uncircled_near_circle_dist=60px,
            # which allowed dimension text tokens to be promoted by
            # unrelated Hough circles in the drawing geometry.
            existing_bubble_centres = [(b.x, b.y, b.radius) for b in bubbles]
            near_circle = False
            near_circle_radius = None
            for cx, cy, cr in circles:
                # Token must be essentially inside the circle
                if math.dist((t.cx, t.cy), (cx, cy)) > cr * 1.3:
                    continue
                if self._circle_too_clipped(
                    self.image if self.image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                    cx, cy, cr,
                ):
                    continue
                if not self._bubble_label_scale_ok(t, cr):
                    continue
                # Circle must not be inside an existing bubble
                if any(math.hypot(cx - bx, cy - by) < br * 1.5
                       for bx, by, br in existing_bubble_centres):
                    continue
                # Circle must have annotation-color rim (strict mode —
                # loose tint check falsely passes on JPEG artifacts
                # around text).
                if not self._circle_has_annotation_tint(
                    self.image if self.image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                    cx, cy, cr, strict=True,
                ):
                    continue
                near_circle = True
                near_circle_radius = cr
                break

            if near_circle:
                # ── Dimension-collision guard ─────────────────────────
                # If this candidate's digit string also appears as a
                # SUBSTRING inside any nearby dimension-like token, it
                # is almost certainly the dimension value leaking into
                # the bubble pool (e.g. the "53" of "Ø53", the "8" of
                # "(8)", the "9" of "R9").  Refuse to promote it.
                is_dim_substring = False
                for j, other in enumerate(norm_tokens):
                    if j == idx or j in used_token_ids:
                        continue
                    other_text = other.text.strip()
                    # Must be a dimension-looking token that *contains*
                    # the candidate digits but is not itself the same
                    # bare number.  A nearby containing token within
                    # ~4× radius counts as evidence of dimension leak.
                    if other_text == t.text.strip():
                        continue
                    if norm not in re.findall(r"\d+", other_text):
                        continue
                    if re.fullmatch(r"\d+", other_text):
                        # Another bare number — not a dimension token.
                        continue
                    if math.dist((t.cx, t.cy), (other.cx, other.cy)) < 4 * self.cfg.uncircled_near_circle_dist:
                        is_dim_substring = True
                        break
                if is_dim_substring:
                    continue

                # Estimate radius from nearby circle
                best_r = int(near_circle_radius or 25)
                for (cx, cy, cr) in circles:
                    if math.dist((t.cx, t.cy), (cx, cy)) < self.cfg.uncircled_near_circle_dist:
                        if self._circle_too_clipped(
                            self.image if self.image is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                            cx, cy, cr,
                        ):
                            continue
                        if not self._bubble_label_scale_ok(t, cr):
                            continue
                        best_r = cr
                        break
                bubbles.append(BubbleResult(
                    bubble_number=norm,
                    x=int(t.cx), y=int(t.cy),
                    radius=best_r,
                    confidence=float(t.conf),
                    dimension="NO_DIMENSION",
                ))
                seen.add(norm)
                used_token_ids.add(idx)

    # ── Annotation colour helpers ─────────────────────────────────

    @staticmethod
    def _annotation_hsv_mask(img: np.ndarray) -> np.ndarray:
        """Return a uint8 mask of "annotation-coloured" pixels.

        CAD packages emit balloon strokes in any of: red, maroon,
        pink, magenta, or purple. The hue band [125-180] ∪ [0-12]
        covers the full red→purple spectrum. A modest closing pass
        bridges 1-2 px stroke gaps that JPEG compression introduces.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m_red = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([12, 255, 255]))
        m_blue_purple = cv2.inRange(hsv, np.array([85, 35, 35]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m_red, m_blue_purple)
        return cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

    # ── Quality assessment ────────────────────────────────────────

    def _assess_image_quality(self, img: np.ndarray) -> Dict[str, Any]:
        """Compute upfront statistics that downstream steps gate on.

        The goal isn't a single "quality score" — it's a small set
        of named signals each downstream step can react to:

          * resolution    → upscale step (Step 0b)
          * low_contrast  → CLAHE in the side-copy used for circle
                            detection (Step 3)
          * low_maroon    → saturation boost in that same side-copy
          * clutter_score → diagnostic only for now; later it can
                            tighten the HSV thresholds to suppress
                            faded table-row pixels that drift into
                            the annotation band

        Cheap to compute (~10 ms on 800x800) so we always run it.
        """
        h, w = img.shape[:2]
        short_side = min(w, h)
        megapixels = (w * h) / 1_000_000.0

        # Grayscale stats — overall contrast / brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray_mean = float(np.mean(gray))
        gray_std  = float(np.std(gray))

        # HSV — annotation colour detection via the shared helper
        # that covers the full red→purple CAD palette.
        if img.ndim == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            maroon = self._annotation_hsv_mask(img)
            maroon_pixels = int(maroon.sum() // 255)
            maroon_pct = maroon_pixels / float(w * h)
            # Sample mean saturation only over coloured pixels
            # (S>40) — using the whole image dilutes the signal
            # with the white background.
            coloured = hsv[:, :, 1] > 40
            mean_sat = float(np.mean(hsv[coloured, 1])) if coloured.any() else 0.0
        else:
            maroon_pixels = 0
            maroon_pct = 0.0
            mean_sat = 0.0

        # Clutter proxy — count connected text-like components in
        # the binarised "everything that isn't annotation colour"
        # mask. Tables and dense notes show up as many small
        # rectangular blobs; clean drawings have far fewer.
        if img.ndim == 3:
            _, bin_gray = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            nlabels, _, stats, _ = cv2.connectedComponentsWithStats(bin_gray, connectivity=8)
            # Count blobs whose bounding box looks like a text glyph
            # (roughly 5–40 px wide, 6–30 px tall). This is rough
            # but enough to separate "clean drawing" from "drawing
            # plus surrounding tables/notes/title block".
            text_like = 0
            for i in range(1, nlabels):
                bx, by, bw, bh, area = stats[i]
                if 5 <= bw <= 40 and 6 <= bh <= 30 and area >= 15:
                    text_like += 1
            clutter_score = text_like / max(1.0, megapixels)
        else:
            clutter_score = 0.0

        q: Dict[str, Any] = {
            "width": w,
            "height": h,
            "megapixels": round(megapixels, 3),
            "gray_mean": round(gray_mean, 1),
            "gray_std": round(gray_std, 1),
            "mean_sat_over_colour": round(mean_sat, 1),
            "maroon_pixel_count": maroon_pixels,
            "maroon_pixel_pct": round(maroon_pct * 100, 3),
            "clutter_score": round(clutter_score, 1),
            # Boolean flags that downstream steps can branch on.
            "is_low_resolution": short_side < 800,
            "is_low_contrast":   gray_std < 40,
            "is_low_maroon":     maroon_pct < 0.0008,
            # Structural clutter estimate: dense tables/BOM/notes push this
            # much higher than clean drawings.
            "is_high_clutter":   clutter_score > 100,
        }
        return q

    def _find_edge_bubbles(
        self,
        bubbles: List[BubbleResult],
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
        image: np.ndarray,
    ) -> None:
        """
        Detect bubbles that are partially cropped at image edges.

        Bubble tokens near edges (within 1.5× median radius) that
        weren't matched to circles get an estimated position and
        radius.  This handles the common case where a drawing photo
        cuts off a balloon at the margin.
        """
        seen = {b.bubble_number for b in bubbles}
        h, w = image.shape[:2]
        median_r = 25
        if bubbles:
            radii = [b.radius for b in bubbles]
            median_r = int(sorted(radii)[len(radii) // 2])

        edge_margin = int(median_r * 1.5)

        for idx, t in enumerate(norm_tokens):
            if idx in used_token_ids:
                continue
            if getattr(t, "dual_use", False):
                continue
            if not getattr(t, "is_maroon", False):
                # Split-stream: black-on-white text (dimensions,
                # table cells, title-block entries) cannot be a
                # bubble number even when its text shape looks like
                # a 1-2 digit integer. The annotation HSV mask
                # filtered this token out as non-maroon.
                continue
            if not is_bubble_token(t.text):
                continue
            norm = normalize_bubble_value(t.text)
            if not norm or norm == "0":
                continue
            if norm in seen:
                continue

            # Check if the token is near any image edge
            near_edge = (
                t.cx < edge_margin
                or t.cx > w - edge_margin
                or t.cy < edge_margin
                or t.cy > h - edge_margin
            )
            if not near_edge:
                continue

            # The token must be near a partial circle with annotation
            # color.  Without this check, dimension text near edges
            # gets misidentified as cropped bubbles.
            has_nearby_tinted_arc = False
            for ang_offset in range(0, 360, 30):
                ang = math.radians(ang_offset)
                px = int(t.cx + median_r * math.cos(ang))
                py = int(t.cy + median_r * math.sin(ang))
                if 0 <= px < w and 0 <= py < h:
                    if (self._annotation_layer is not None
                            and self._annotation_layer.mask[py, px] > 0):
                        has_nearby_tinted_arc = True
                        break
            if not has_nearby_tinted_arc:
                continue

            if not self._bubble_label_scale_ok(t, median_r):
                continue

            if self._strict_photo_mode() and any(
                math.hypot(t.cx - b.x, t.cy - b.y) < max(120.0, b.radius * 2.2)
                for b in bubbles
            ):
                continue

            bubbles.append(BubbleResult(
                bubble_number=norm,
                x=int(t.cx),
                y=int(t.cy),
                radius=median_r,
                confidence=0.4,
                dimension="NO_DIMENSION",
                needs_review=True,
                review_reason="edge_bubble",
            ))
            seen.add(norm)
            used_token_ids.add(idx)

    def _recover_edge_blob_label(
        self,
        image: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> Optional[Tuple[str, float]]:
        """OCR a clipped edge-balloon crop for a 1-2 digit label only."""
        h, w = image.shape[:2]
        pad = max(2, int(radius * 0.05))
        x1 = max(0, cx - radius - pad)
        y1 = max(0, cy - radius - pad)
        x2 = min(w, cx + radius + pad)
        y2 = min(h, cy + radius + pad)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        best: Optional[Tuple[str, float]] = None
        for scale in (4,):
            up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants = (
                up,
                cv2.cvtColor(255 - thresh, cv2.COLOR_GRAY2BGR),
            )
            for variant in variants:
                try:
                    result = self.ocr(variant)
                except Exception:
                    continue
                items = result[0] if (
                    isinstance(result, tuple) and len(result) >= 1
                ) else (result or [])
                for item in items or []:
                    try:
                        text_info = item[1]
                        text = str(
                            text_info[0]
                            if isinstance(text_info, (list, tuple))
                            else text_info
                        ).strip()
                        conf = float(
                            text_info[1]
                            if isinstance(text_info, (list, tuple))
                            and len(text_info) > 1
                            else 0.9
                        )
                    except Exception:
                        continue
                    norm = self._normalize_local_bubble_candidate(text)
                    if norm is None:
                        continue
                    if conf < 0.45:
                        continue
                    if best is None or conf > best[1]:
                        best = (norm, conf)
        return best

    def _find_edge_blob_bubbles(
        self,
        bubbles: List[BubbleResult],
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
        image: np.ndarray,
    ) -> None:
        """Recover clipped colored balloons whose label OCR was missed.

        Unlike `_find_edge_bubbles`, this starts from annotation-color
        fragments touching the image boundary. It is edge-only and accepts
        only a 1-2 digit OCR result from the combined fragment crop, which
        keeps table/dimension text from becoming balloons.
        """
        if self._skip_heavy_annotation_recovery():
            logger.info(
                "Edge-blob recovery: skipped "
                "(low-saturation high-clutter annotation layer)"
            )
            return
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return

        seen = {str(b.bubble_number) for b in bubbles}
        median_r = 25
        if bubbles:
            radii = sorted(max(1, int(b.radius)) for b in bubbles)
            median_r = radii[len(radii) // 2]

        mask = self._annotation_hsv_mask(image)
        if int(np.count_nonzero(mask)) < 40:
            return
        nlabels, _labels, stats, cents = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        comps: List[Dict[str, Any]] = []
        edge_pad = max(2, int(median_r * 0.18))
        for i in range(1, nlabels):
            x, y, bw, bh, area = stats[i]
            if area < max(12, median_r * 2):
                continue
            if area > median_r * median_r * 5:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > 3.2:
                continue
            touches = []
            if x <= edge_pad:
                touches.append("left")
            if x + bw >= w - edge_pad:
                touches.append("right")
            if y <= edge_pad:
                touches.append("top")
            if y + bh >= h - edge_pad:
                touches.append("bottom")
            for side in touches:
                comps.append({
                    "side": side,
                    "x1": int(x),
                    "y1": int(y),
                    "x2": int(x + bw),
                    "y2": int(y + bh),
                    "cx": float(cents[i][0]),
                    "cy": float(cents[i][1]),
                    "area": int(area),
                })

        if not comps:
            return

        edge_ocr_attempts = 0
        edge_ocr_budget = max(0, int(self.cfg.max_edge_blob_recovery_ocr))
        for side in ("left", "right", "top", "bottom"):
            if edge_ocr_attempts >= edge_ocr_budget:
                break
            side_comps = [c for c in comps if c["side"] == side]
            if not side_comps:
                continue
            axis = "cy" if side in {"left", "right"} else "cx"
            side_comps.sort(key=lambda c: c[axis])
            clusters: List[List[Dict[str, Any]]] = []
            for comp in side_comps:
                if not clusters:
                    clusters.append([comp])
                    continue
                prev = clusters[-1][-1]
                if abs(float(comp[axis]) - float(prev[axis])) <= median_r * 2.8:
                    clusters[-1].append(comp)
                else:
                    clusters.append([comp])

            for cluster in clusters:
                if edge_ocr_attempts >= edge_ocr_budget:
                    break
                x1 = min(c["x1"] for c in cluster)
                y1 = min(c["y1"] for c in cluster)
                x2 = max(c["x2"] for c in cluster)
                y2 = max(c["y2"] for c in cluster)
                bw = x2 - x1
                bh = y2 - y1
                max_span = max(bw, bh)
                min_span = min(bw, bh)
                if max_span > median_r * 4.0 or min_span < median_r * 0.45:
                    continue
                total_area = max(1, sum(int(c["area"]) for c in cluster))
                cx = int(round(
                    sum(float(c["cx"]) * int(c["area"]) for c in cluster)
                    / total_area
                ))
                cy = int(round(
                    sum(float(c["cy"]) * int(c["area"]) for c in cluster)
                    / total_area
                ))
                radius = int(round(max(median_r * 1.55, max_span * 0.86)))

                if any(
                    math.hypot(cx - b.x, cy - b.y) < max(radius, b.radius) * 1.4
                    for b in bubbles
                ):
                    continue

                center_candidates = [(cx, cy)]
                tangent_delta = max(2, int(round(radius * 0.14)))
                normal_delta = max(1, int(round(radius * 0.07)))
                if side in {"left", "right"}:
                    normal_sign = 1 if side == "right" else -1
                    for dx in (0, normal_sign * normal_delta):
                        for dy in (-tangent_delta, tangent_delta, 0):
                            center_candidates.append((cx + dx, cy + dy))
                else:
                    normal_sign = -1 if side == "top" else 1
                    for dy in (0, normal_sign * normal_delta):
                        for dx in (-tangent_delta, tangent_delta, 0):
                            center_candidates.append((cx + dx, cy + dy))

                recovered = None
                rec_cx, rec_cy = cx, cy
                for tcx, tcy in center_candidates:
                    if edge_ocr_attempts >= edge_ocr_budget:
                        break
                    edge_ocr_attempts += 1
                    recovered = self._recover_edge_blob_label(image, tcx, tcy, radius)
                    if recovered is not None:
                        rec_cx, rec_cy = tcx, tcy
                        break
                if recovered is None:
                    continue
                norm, conf = recovered
                if norm in seen:
                    continue
                if self._recovered_id_collides_with_dimension(
                    norm, rec_cx, rec_cy, radius, norm_tokens, used_token_ids
                ):
                    continue

                bubbles.append(BubbleResult(
                    bubble_number=norm,
                    x=rec_cx,
                    y=rec_cy,
                    radius=max(8, min(radius, int(median_r * 2.2))),
                    confidence=max(0.35, min(0.75, conf)),
                    dimension="NO_DIMENSION",
                    needs_review=True,
                    review_reason="edge_blob_recovery",
                ))
                seen.add(norm)
                logger.info(
                    "Edge-blob recovery added bubble #%s at (%d, %d), r=%d",
                    norm, rec_cx, rec_cy, radius,
                )

    def _find_red_blob_bubbles(
        self,
        bubbles: List[BubbleResult],
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
        image: np.ndarray,
    ) -> None:
        """Last-chance fallback for upscaled tiny inputs.

        After Hough + edge + evidence-based passes, scan the red
        chromatic layer for isolated round blobs that are clearly
        annotation balloons but missed by every prior detector
        (typically because the balloon stroke is too faint after
        bicubic upscaling for Hough's gradient threshold). For each
        unmatched blob, claim the nearest unused single-digit OCR
        token within 1.5x the blob radius. Only runs when the input
        was upscaled, so it cannot regress good-quality images.
        """
        # Run whenever either:
        #   (a) the image was upscaled (tiny/compressed input — the
        #       original design case), OR
        #   (b) the original image has enough maroon to be worth
        #       scanning even at native resolution. This second
        #       gate is what lets the fallback fire on high-res drawings
        #       with thin annotation strokes where
        #       _identify_bubbles produces zero candidates.
        # Using OR (not AND) so tiny images still get the fallback via
        # branch (a). Blobs already covered by existing bubbles
        # are filtered downstream by the "too close" check, so
        # running unconditionally is safe.
        maroon_count = (self._quality or {}).get("maroon_pixel_count", 0)
        was_upscaled = getattr(self, "_was_upscaled", False)
        if not was_upscaled and maroon_count < 200:
            logger.info("Red-blob fallback: skipped (no upscale, only %d maroon pixels)",
                        maroon_count)
            return
        logger.info("Red-blob fallback: scanning (existing bubbles=%d, "
                    "maroon_px=%d, was_upscaled=%s)",
                    len(bubbles), maroon_count, was_upscaled)

        h, w = image.shape[:2]
        # Use the shared annotation mask so we catch purple/magenta
        # CAD palettes, not just red/maroon.
        red = self._annotation_hsv_mask(image)

        median_r = 25
        if bubbles:
            radii = [b.radius for b in bubbles]
            median_r = int(sorted(radii)[len(radii) // 2])

        nlabels, labels, stats, cents = cv2.connectedComponentsWithStats(red, connectivity=8)
        seen = {b.bubble_number for b in bubbles}
        logger.info("Red-blob fallback: %d components, median_r=%d", nlabels - 1, median_r)
        for i in range(1, nlabels):
            x, y, bw, bh, area = stats[i]
            cx, cy = float(cents[i][0]), float(cents[i][1])
            logger.info("  blob[%d] @ (%d,%d) w=%d h=%d area=%d", i, int(cx), int(cy), bw, bh, area)

        local_ocr_attempts = 0
        local_ocr_budget = max(0, int(self.cfg.max_red_blob_local_ocr))
        if self._skip_heavy_annotation_recovery():
            local_ocr_budget = 0
        for i in range(1, nlabels):
            if self._step6_recovery_budget_exceeded():
                break
            x, y, bw, bh, area = stats[i]
            if area < 40 or area > median_r * median_r * 4:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > 2.5:
                # Long thin shapes are leader-line segments, not balloons
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            logger.info("  candidate blob @ (%d,%d) area=%d aspect=%.2f", int(cx), int(cy), area, aspect)
            blob_r = max(8, int(round(min(bw, bh) * 0.55)))

            # Skip if an existing bubble already covers this blob
            too_close = any(
                math.dist((cx, cy), (b.x, b.y)) < max(blob_r, b.radius) * 1.5
                for b in bubbles
            )
            if too_close:
                logger.info("    skip: too close to existing bubble")
                continue

            local_centers = [(int(round(cx)), int(round(cy)))]
            if bw > bh * 1.45:
                local_centers.extend([
                    (int(x + blob_r), int(y + bh / 2)),
                    (int(x + bw - blob_r), int(y + bh / 2)),
                ])
            elif bh > bw * 1.45:
                local_centers.extend([
                    (int(x + bw / 2), int(y + blob_r)),
                    (int(x + bw / 2), int(y + bh - blob_r)),
                ])
            # Find the nearest unused single-digit OCR token within
            # 1.5x median radius — that's our candidate bubble number.
            # We accept dual_use tokens here (unlike _find_edge_bubbles)
            # because on heavily-compressed inputs the OCR often
            # classifies the balloon digit as a dimension fragment.
            # The red-blob proximity gate is the safety net against
            # accidentally promoting a real dimension digit.
            best_idx, best_dist, best_norm = -1, float("inf"), None
            for idx, t in enumerate(norm_tokens):
                if idx in used_token_ids:
                    continue
                # A token inside (or adjacent to) a red blob *should*
                # be is_maroon=True; if it isn't, OCR found it on a
                # nearby black dim, not on the balloon itself.
                if not getattr(t, "is_maroon", False):
                    continue
                if not is_bubble_token(t.text):
                    continue
                norm = normalize_bubble_value(t.text)
                if not norm or norm == "0" or norm in seen:
                    continue
                d = math.dist((cx, cy), (t.cx, t.cy))
                if d < best_dist and d < blob_r * 2.0:
                    best_idx, best_dist, best_norm = idx, d, norm

            # If no digit token was readable inside this blob, try
            # the dimension-only fallback: find the nearest unmatched
            # non-bubble token (a dimension callout) within 2.5x the
            # median radius and emit a placeholder bubble. The number
            # is set to "?" so the human inspector knows to fill it
            # in, but the dim and position are preserved so the work
            # of identifying the location isn't wasted.
            if best_idx < 0:
                recovered = None
                recovered_center = None
                if (not self.cfg.enable_targeted_endpoint_ocr
                        and local_ocr_attempts < local_ocr_budget):
                    for rcx, rcy in local_centers:
                        if local_ocr_attempts >= local_ocr_budget:
                            break
                        if not (0 <= rcx < w and 0 <= rcy < h):
                            continue
                        local_ocr_attempts += 1
                        result = self._recover_bubble_from_circle(image, rcx, rcy, blob_r)
                        if result is None:
                            continue
                        norm_text, conf = result
                        if norm_text in seen:
                            continue
                        recovered = (norm_text, conf)
                        recovered_center = (rcx, rcy)
                        break
                if recovered is None or recovered_center is None:
                    logger.info("    skip: no readable bubble id near red blob")
                    continue

                norm_text, conf = recovered
                rcx, rcy = recovered_center
                bubbles.append(BubbleResult(
                    bubble_number=norm_text,
                    x=int(rcx),
                    y=int(rcy),
                    radius=blob_r,
                    confidence=max(0.35, float(conf) * 0.8),
                    dimension="NO_DIMENSION",
                    needs_review=True,
                    review_reason="red_blob_local_ocr",
                ))
                seen.add(norm_text)
                logger.info(
                    "Red-blob fallback OCR added bubble #%s at (%d, %d), r=%d",
                    norm_text, int(rcx), int(rcy), blob_r,
                )
                continue

            bubbles.append(BubbleResult(
                bubble_number=best_norm,
                x=int(cx),
                y=int(cy),
                radius=blob_r,
                confidence=0.35,
                dimension="NO_DIMENSION",
                needs_review=True,
                review_reason="red_blob_fallback",
            ))
            seen.add(best_norm)
            used_token_ids.add(best_idx)
            logger.info(
                "Red-blob fallback added bubble #%s at (%d, %d), token-dist=%.1f",
                best_norm, int(cx), int(cy), best_dist,
            )

    def _find_evidence_based_bubbles(
        self,
        bubbles: List[BubbleResult],
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
        image: np.ndarray,
    ) -> None:
        """
        Find bubbles from OCR tokens that weren't matched to any
        HoughCircle, using the multi-signal bubble evidence scorer.

        For each unmatched bubble-like token, tests several candidate
        radii at the token position.  If the evidence score exceeds
        a threshold, creates a bubble.

        This catches balloons that HoughCircles missed due to:
          - Partial occlusion
          - Non-standard circle style (thin outline, dashed)
          - Dense drawing geometry confusing HoughCircles
        """
        seen = {b.bubble_number for b in bubbles}
        median_r = 25
        if bubbles:
            radii = sorted(b.radius for b in bubbles)
            median_r = radii[len(radii) // 2]

        for idx, t in enumerate(norm_tokens):
            if idx in used_token_ids:
                continue
            if getattr(t, "dual_use", False):
                continue
            if not getattr(t, "is_maroon", False):
                # Split-stream: black-on-white text (dimensions,
                # table cells, title-block entries) cannot be a
                # bubble number even when its text shape looks like
                # a 1-2 digit integer. The annotation HSV mask
                # filtered this token out as non-maroon.
                continue
            if not is_bubble_token(t.text):
                continue
            norm = normalize_bubble_value(t.text)
            if not norm or norm == "0":
                continue
            if norm in seen:
                continue

            # Test candidate radii around the median. Pick the radius
            # by best ANNOTATION-MASK OVERLAP rather than best evidence
            # score — overlap directly confirms a magenta-coloured rim
            # at that radius, while evidence (a generic pixel/edge
            # signal) can spike at the wrong radius when the digit
            # itself produces strong gradients near the center.
            mask = None
            if (hasattr(self, '_annotation_layer')
                    and self._annotation_layer is not None):
                mask = self._annotation_layer.mask

            def _ann_overlap_at(r: int) -> float:
                if mask is None:
                    return 0.0
                count = total_s = 0
                for ang in range(0, 360, 10):
                    rad = math.radians(ang)
                    px = int(t.cx + r * math.cos(rad))
                    py = int(t.cy + r * math.sin(rad))
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        total_s += 1
                        if mask[py, px] > 0:
                            count += 1
                return (count / total_s) if total_s > 0 else 0.0

            best_score = 0.0
            best_r = median_r
            best_overlap = 0.0
            # Five candidates spanning 70%-130% of median catches bubbles
            # slightly larger or smaller than the median.
            candidate_radii = [
                int(median_r * f) for f in (0.7, 0.85, 1.0, 1.15, 1.3)
            ]
            for test_r in candidate_radii:
                if test_r < 8:
                    continue
                overlap = _ann_overlap_at(test_r)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_r = test_r
                    best_score = self._compute_bubble_evidence(
                        image, int(t.cx), int(t.cy), test_r,
                    )

            # If no rim overlap at any candidate radius, fall back to
            # the evidence-only path so this rescue still catches
            # bubbles with too-faint magenta (e.g. JPEG-bleached).
            if best_overlap == 0.0:
                for test_r in candidate_radii:
                    if test_r < 8:
                        continue
                    score = self._compute_bubble_evidence(
                        image, int(t.cx), int(t.cy), test_r,
                    )
                    if score > best_score:
                        best_score = score
                        best_r = test_r

            ann_overlap = best_overlap

            # Combined acceptance: a strong magenta-rim overlap can
            # compensate for a slightly-below-threshold evidence score
            # (and vice versa). Direct annotation-pixel overlap is the
            # most trustworthy signal that a balloon is actually there.
            passes = (
                (best_score >= 0.25 and ann_overlap >= 0.05)
                or (best_score >= 0.20 and ann_overlap >= 0.10)
                or (ann_overlap >= 0.20)
            )
            if passes:
                bubbles.append(BubbleResult(
                    bubble_number=norm,
                    x=int(t.cx),
                    y=int(t.cy),
                    radius=best_r,
                    confidence=best_score,
                    dimension="NO_DIMENSION",
                ))
                seen.add(norm)
                used_token_ids.add(idx)

    def _find_mask_circle_bubbles(
        self,
        bubbles: List[BubbleResult],
        norm_tokens: List["NormalizedToken"],
        used_token_ids: set,
        image: np.ndarray,
    ) -> None:
        """Mask-based balloon detection for photo-quality input.

        The primary HoughCircles runs on the grayscale edge map, which
        misses balloons whose rims got softened by camera blur. The
        annotation-colour mask still shows them as clean rings, so we
        run HoughCircles a SECOND time directly on the saturation-
        boosted maroon mask. New circles (not within ~1× median radius
        of an existing bubble) get added.

        Pure mask-based recall booster — no text whitelists, no
        per-image tuning. Any drawing whose balloons are colour-
        annotated benefits when its rim quality is below the
        gray-edge Hough's threshold.
        """
        if not bubbles:
            return  # need median radius
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255)
            boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            hsv_b = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)
            m_red = cv2.inRange(hsv_b,
                                np.array([0, 35, 35]),
                                np.array([12, 255, 255]))
            m_blue_purp = cv2.inRange(hsv_b,
                                      np.array([85, 30, 35]),
                                      np.array([180, 255, 255]))
            mask = cv2.morphologyEx(
                cv2.bitwise_or(m_red, m_blue_purp),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
        except Exception as exc:
            logger.debug("mask-circle: boost failed: %s", exc)
            return

        radii = sorted(b.radius for b in bubbles)
        median_r = radii[len(radii) // 2]
        h, w = image.shape[:2]
        min_r = max(8, int(median_r * 0.6))
        max_r = max(min_r + 6, int(median_r * 1.6))

        # HoughCircles on the binary mask — the rings on a clean mask
        # are easier to detect than on a noisy gray edge map.
        # param1 is gradient threshold; on a binary input we use a
        # modest value because the mask edges are crisp.
        try:
            circles = cv2.HoughCircles(
                mask,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=max(20, int(median_r * 1.5)),
                param1=50,
                param2=18,
                minRadius=min_r,
                maxRadius=max_r,
            )
        except Exception as exc:
            logger.debug("mask-circle: HoughCircles failed: %s", exc)
            return
        if circles is None:
            return

        seen = {b.bubble_number for b in bubbles}
        added = 0
        for c in circles[0]:
            cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
            if cx - cr < 0 or cy - cr < 0 or cx + cr >= w or cy + cr >= h:
                continue
            # Skip if already detected near here.
            if any(math.hypot(cx - b.x, cy - b.y) < median_r * 1.0
                   for b in bubbles):
                continue
            # Try to read a number inside via the standard recovery OCR.
            result = self._recover_bubble_from_circle(image, cx, cy, cr)
            if result is None:
                continue
            norm_text, conf = result
            if norm_text in seen:
                continue
            bubbles.append(BubbleResult(
                bubble_number=norm_text,
                x=cx, y=cy, radius=cr,
                confidence=conf * 0.85,
                dimension="NO_DIMENSION",
                needs_review=True,
                review_reason="mask_circle_recovery",
            ))
            seen.add(norm_text)
            added += 1
        if added > 0:
            logger.info("Mask-circle recovery added %d bubble(s)", added)

    def _recover_colored_circle_label_bubbles(
        self,
        bubbles: List[BubbleResult],
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
        circles: List[Tuple[int, int, int]],
        image: np.ndarray,
    ) -> None:
        """Recover missed bubble IDs by OCR'ing colored circle interiors."""
        if self._skip_heavy_annotation_recovery():
            logger.info(
                "Colored-circle label recovery: skipped "
                "(low-saturation high-clutter annotation layer)"
            )
            return
        if not bubbles:
            return

        seen = {str(b.bubble_number) for b in bubbles}
        radii = sorted(max(1, int(b.radius)) for b in bubbles)
        median_r = radii[len(radii) // 2]
        h, w = image.shape[:2]

        mask = self._annotation_hsv_mask(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        blurred = cv2.GaussianBlur(mask, (5, 5), 1.5)

        min_r = max(12, int(median_r * 0.35))
        max_r = max(min_r + 8, int(median_r * 1.15))
        candidate_circles: List[Tuple[int, int, int]] = []

        hough = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int(median_r * 0.65)),
            param1=50,
            param2=14,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if hough is not None:
            for c in hough[0]:
                candidate_circles.append((
                    int(round(c[0])),
                    int(round(c[1])),
                    int(round(c[2])),
                ))

        for cx, cy, cr in circles:
            if min_r <= cr <= max_r:
                candidate_circles.append((int(cx), int(cy), int(cr)))

        unique: List[Tuple[int, int, int]] = []
        for cx, cy, cr in candidate_circles:
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            if self._circle_too_clipped(image, cx, cy, cr):
                continue
            if not self._circle_has_annotation_tint(image, cx, cy, cr, strict=True):
                continue
            evidence = self._compute_bubble_evidence(image, cx, cy, cr)
            if evidence < 0.32:
                continue
            if any(
                math.hypot(cx - b.x, cy - b.y) < min(float(cr), float(b.radius)) * 0.95
                for b in bubbles
            ):
                continue
            if any(
                math.hypot(cx - ux, cy - uy) < min(float(cr), float(ur)) * 0.8
                for ux, uy, ur in unique
            ):
                continue
            unique.append((cx, cy, cr))

        unique.sort(key=lambda c: (c[2], -self._compute_bubble_evidence(image, *c)))

        added = 0
        for cx, cy, cr in unique[:12]:
            result = self._recover_bubble_from_circle(image, cx, cy, cr)
            if result is None:
                continue
            norm_text, conf = result
            if norm_text in seen:
                continue
            if self._recovered_id_collides_with_dimension(
                norm_text, cx, cy, cr, norm_tokens, used_token_ids,
            ):
                continue

            bubbles.append(BubbleResult(
                bubble_number=norm_text,
                x=cx,
                y=cy,
                radius=cr,
                confidence=float(max(conf, 0.55)),
                dimension="NO_DIMENSION",
                needs_review=True,
                review_reason="colored_circle_label_recovery",
            ))
            seen.add(norm_text)
            added += 1

        if added:
            logger.info("Colored-circle label recovery added %d bubble(s)", added)

    @staticmethod
    def _annotation_ring_score(mask: np.ndarray, cx: int, cy: int, r: int) -> float:
        """Fraction of sampled rim points that hit annotation-colour pixels."""
        h, w = mask.shape[:2]
        hits = 0
        total = 0
        for dr in (0, -1, 1, -2, 2):
            rr = r + dr
            if rr < 4:
                continue
            for ang_deg in range(0, 360, 5):
                ang = math.radians(ang_deg)
                px = int(round(cx + rr * math.cos(ang)))
                py = int(round(cy + rr * math.sin(ang)))
                if 0 <= px < w and 0 <= py < h:
                    total += 1
                    if mask[py, px] > 0:
                        hits += 1
        return (hits / total) if total else 0.0

    def _skip_heavy_annotation_recovery(self) -> bool:
        """Avoid OCR-heavy color recovery on low-saturation UI clutter."""
        al = getattr(self, "_annotation_layer", None)
        if al is None:
            return False
        quality = getattr(self, "_quality", {}) or {}
        dominant_sat = int(getattr(al, "dominant_sat", 255) or 255)
        pixel_count = int(getattr(al, "pixel_count", 0) or 0)
        clutter = float(quality.get("clutter_score", 0.0) or 0.0)
        was_upscaled = bool(getattr(self, "_was_upscaled", False))
        if was_upscaled:
            return False
        return (
            dominant_sat < 70
            and pixel_count > 25000
            and clutter > 90.0
        )

    def _step6_recovery_budget_exceeded(self) -> bool:
        deadline = getattr(self, "_step6_recovery_deadline", None)
        if deadline is None:
            return False
        if time.perf_counter() <= float(deadline):
            return False
        if not getattr(self, "_step6_budget_logged", False):
            logger.info(
                "Step 6 recovery budget exhausted after %.1f s",
                float(self.cfg.max_step6_recovery_seconds or 0.0),
            )
            self._step6_budget_logged = True
        return True

    @staticmethod
    def _numeric_ids_contiguous_from_one(ids: Iterable[str]) -> bool:
        numeric: List[int] = []
        for value in ids:
            text = str(value or "").strip()
            if not re.fullmatch(r"\d{1,3}", text):
                return False
            numeric.append(int(text))
        if len(numeric) < 6:
            return False
        unique = set(numeric)
        if len(unique) != len(numeric):
            return False
        max_id = max(unique)
        return unique == set(range(1, max_id + 1))

    def _recover_annotation_ring_label_bubbles(
        self,
        bubbles: List[BubbleResult],
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
        circles: List[Tuple[int, int, int]],
        image: np.ndarray,
    ) -> None:
        """Recover labels from thin/open annotation-colour balloon rings.

        Some CAD exports render a valid balloon as a pale, broken coloured
        circle. The gray Hough pass may miss it and the contour pass may
        reject it because the stroke is not a closed filled component. This
        pass uses the annotation-colour mask only to propose ring candidates;
        a candidate is promoted only when local rim evidence and local OCR
        agree. No image-specific IDs, coordinates, or expected values are used.
        """
        if image is None:
            return

        h, w = image.shape[:2]
        mask = self._annotation_hsv_mask(image)
        if int(np.count_nonzero(mask)) < 50:
            return

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        blurred = cv2.GaussianBlur(mask, (5, 5), 1.2)

        # This pass is for missed balloon rings, not OCR-sized text blobs.
        # Small real balloons are handled by the main OCR/token paths and
        # the tiny-artifact guard, while mask Hough below this radius tends
        # to spend its OCR budget on colored text fragments.
        min_r = 14
        max_r = max(18, min(72, int(min(h, w) * 0.08)))
        raw_candidates: List[Tuple[int, int, int]] = []

        try:
            hough = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(18, int(min_r * 2.5)),
                param1=40,
                param2=10,
                minRadius=min_r,
                maxRadius=max_r,
            )
        except Exception as exc:
            logger.debug("annotation-ring recovery: HoughCircles failed: %s", exc)
            hough = None

        if hough is not None:
            for c in hough[0]:
                raw_candidates.append((
                    int(round(c[0])),
                    int(round(c[1])),
                    int(round(c[2])),
                ))

        for cx, cy, cr in circles:
            cr_i = int(round(cr))
            if min_r <= cr_i <= max_r:
                raw_candidates.append((int(round(cx)), int(round(cy)), cr_i))

        if not raw_candidates:
            return

        seen_ids = {str(b.bubble_number) for b in bubbles}
        scored: List[Tuple[float, int, int, int]] = []
        unique: List[Tuple[int, int, int]] = []
        for cx, cy, cr in raw_candidates:
            if cx - cr < 0 or cy - cr < 0 or cx + cr >= w or cy + cr >= h:
                continue
            if self._circle_too_clipped(image, cx, cy, cr):
                continue
            if any(
                math.hypot(cx - b.x, cy - b.y) < max(8.0, min(float(cr), float(b.radius)) * 0.9)
                for b in bubbles
            ):
                continue
            if any(
                math.hypot(cx - ux, cy - uy) < max(8.0, min(float(cr), float(ur)) * 0.75)
                for ux, uy, ur in unique
            ):
                continue
            if not self._circle_has_annotation_tint(image, cx, cy, cr, strict=False):
                continue

            ring_score = self._annotation_ring_score(mask, cx, cy, cr)
            evidence = self._compute_bubble_evidence(image, cx, cy, cr)
            if evidence < 0.12 and ring_score < 0.12:
                continue

            unique.append((cx, cy, cr))
            radius_score = min(float(cr) / 24.0, 1.0)
            priority = (ring_score * 2.0) + (radius_score * 0.25) + (min(evidence, 0.5) * 0.25)
            scored.append((priority, cx, cy, cr))

        if not scored:
            return

        scored.sort(key=lambda c: (-c[0], c[3]))
        added = 0
        strong_uncovered_threshold = 0.80
        complete_sequence_threshold = 0.85
        ring_ocr_attempts = 0
        max_ring_ocr = max(0, int(self.cfg.max_annotation_ring_recovery_ocr))
        if self._skip_heavy_annotation_recovery():
            max_ring_ocr = min(max_ring_ocr, 12)
        max_ring_candidates = max(
            0,
            int(self.cfg.max_annotation_ring_recovery_candidates),
        )
        for _score, cx, cy, cr in scored[:max_ring_candidates]:
            if self._step6_recovery_budget_exceeded():
                break
            if ring_ocr_attempts >= max_ring_ocr:
                break
            # If existing labels already form a complete production-style
            # sequence (1..N), only spend local OCR on very strong uncovered
            # ring evidence. This keeps the recovery pass available for real
            # missed balloons (for example N+1, or a hole that just got filled)
            # without scanning dozens of weak colored text/dimension artifacts.
            if (
                self._numeric_ids_contiguous_from_one(seen_ids)
                and _score < complete_sequence_threshold
            ):
                break
            # On non-contiguous drawings we cannot assume missing integers
            # are actual missing balloons. Only OCR strong uncovered rings;
            # weak candidates are usually colored dimension text, leaders, or
            # table geometry and are expensive false positives.
            if _score < strong_uncovered_threshold:
                break
            result = None
            radius_trials = (
                (cr, cr - 2, cr + 2)
                if self._skip_heavy_annotation_recovery()
                else (cr, cr - 2, cr + 2, cr - 4, cr + 4)
            )
            for rr in radius_trials:
                if self._step6_recovery_budget_exceeded():
                    break
                if ring_ocr_attempts >= max_ring_ocr:
                    break
                if rr < min_r or rr > max_r:
                    continue
                ring_ocr_attempts += 1
                result = self._recover_bubble_from_circle(image, cx, cy, rr)
                if result is not None:
                    cr = rr
                    break
            if result is None:
                continue

            norm_text, conf = result
            if norm_text in seen_ids:
                same_label = [
                    b for b in bubbles
                    if str(b.bubble_number) == norm_text
                ]
                if any(
                    math.hypot(cx - b.x, cy - b.y) < max(float(cr), float(b.radius)) * 1.2
                    for b in same_label
                ):
                    continue

                candidate_evidence = self._compute_bubble_evidence(image, cx, cy, cr)
                candidate_ring = self._annotation_ring_score(mask, cx, cy, cr)
                replace_target: Optional[BubbleResult] = None
                best_gain = 0.0
                for existing in same_label:
                    existing_evidence = self._compute_bubble_evidence(
                        image,
                        int(existing.x),
                        int(existing.y),
                        int(max(1, existing.radius)),
                    )
                    existing_has_dim = (
                        bool(existing.dimension)
                        and existing.dimension != "NO_DIMENSION"
                    )
                    if existing_has_dim and not existing.needs_review:
                        continue
                    gain = candidate_evidence - existing_evidence
                    if (
                        candidate_ring >= 0.16
                        and gain > max(0.08, best_gain)
                    ):
                        replace_target = existing
                        best_gain = gain
                if replace_target is not None:
                    replace_target.x = cx
                    replace_target.y = cy
                    replace_target.radius = cr
                    replace_target.confidence = float(max(conf * 0.9, 0.50))
                    replace_target.dimension = "NO_DIMENSION"
                    replace_target.needs_review = True
                    replace_target.review_reason = "annotation_ring_label_recovery_replaced_weaker_duplicate"
                    added += 1
                    continue
                if (
                    candidate_ring >= 0.16
                    and candidate_evidence >= 0.20
                    and not self._recovered_id_collides_with_dimension(
                        norm_text, cx, cy, cr, norm_tokens, used_token_ids,
                    )
                ):
                    bubbles.append(BubbleResult(
                        bubble_number=norm_text,
                        x=cx,
                        y=cy,
                        radius=cr,
                        confidence=float(max(conf * 0.9, 0.50)),
                        dimension="NO_DIMENSION",
                        needs_review=True,
                        review_reason="annotation_ring_label_recovery_duplicate_label",
                    ))
                    added += 1
                continue
            if self._recovered_id_collides_with_dimension(
                norm_text, cx, cy, cr, norm_tokens, used_token_ids,
            ):
                continue

            bubbles.append(BubbleResult(
                bubble_number=norm_text,
                x=cx,
                y=cy,
                radius=cr,
                confidence=float(max(conf * 0.9, 0.50)),
                dimension="NO_DIMENSION",
                needs_review=True,
                review_reason="annotation_ring_label_recovery",
            ))
            seen_ids.add(norm_text)
            added += 1

        if added:
            logger.info("Annotation-ring label recovery added %d bubble(s)", added)

    def _find_annotation_blob_bubbles(
        self,
        bubbles: List[BubbleResult],
        image: np.ndarray,
    ) -> None:
        """
        Detect bubbles from annotation-color connected components.

        HoughCircles fails when a bubble sits next to dense dimension
        text (noisy gradient field).  This detection path uses a
        different signal: connected components in the annotation mask
        filtered by circularity, solidity, and radius band.

        Catches bubbles that Hough silently drops in dense drawing regions.
        """
        if not hasattr(self, '_annotation_layer') or self._annotation_layer is None:
            return
        mask = self._annotation_layer.mask
        if mask is None or np.count_nonzero(mask) < 50:
            return

        if not bubbles:
            return
        radii = sorted(b.radius for b in bubbles)
        median_r = radii[len(radii) // 2]
        seen = {b.bubble_number for b in bubbles}
        h, w = image.shape[:2]

        # Min area for a candidate blob
        min_area = int(math.pi * (median_r * 0.5) ** 2)
        max_area = int(math.pi * (median_r * 1.8) ** 2)

        # Edge margin — blobs in the outer 3% are title block marks
        edge_margin = max(20, int(math.hypot(w, h) * 0.03))

        # Find contours in the annotation mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: List[Tuple[int, int, int]] = []  # (cx, cy, r)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue

            # Circularity: 4π·area / perimeter²
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity < 0.55:
                continue

            # Solidity: area / convex hull area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = area / hull_area
            if solidity < 0.75:
                continue

            # Fit enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(radius))

            # Radius band check
            if r_i < median_r * 0.5 or r_i > median_r * 1.5:
                continue

            # Edge margin check
            if (cx_i < edge_margin or cx_i > w - edge_margin
                    or cy_i < edge_margin or cy_i > h - edge_margin):
                continue

            # Not already a detected bubble
            if any(math.hypot(cx_i - b.x, cy_i - b.y) < median_r * 1.3
                   for b in bubbles):
                continue

            candidates.append((cx_i, cy_i, r_i))

        # Multi-preprocessing OCR on each candidate
        for cx, cy, cr in candidates:
            result = self._recover_bubble_from_circle(image, cx, cy, cr)
            if result is None:
                # Try with CLAHE preprocessing
                margin = max(8, int(cr * 0.3))
                x1 = max(0, cx - cr - margin)
                y1 = max(0, cy - cr - margin)
                x2 = min(w, cx + cr + margin)
                y2 = min(h, cy + cr + margin)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                scale = 4 if cr <= 26 else 3
                up = cv2.resize(crop, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
                gray_c = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                enhanced = clahe.apply(gray_c)
                _, thresh = cv2.threshold(enhanced, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                for variant in [cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(255 - thresh, cv2.COLOR_GRAY2BGR)]:
                    try:
                        ocr_result = self.ocr(variant)
                    except Exception:
                        continue
                    items = (ocr_result[0] if isinstance(ocr_result, tuple)
                             and len(ocr_result) >= 1 else (ocr_result or []))
                    for item in items or []:
                        try:
                            ti = item[1]
                            text = str(ti[0] if isinstance(ti, (list, tuple)) else ti).strip()
                            conf = float(ti[1] if isinstance(ti, (list, tuple)) and len(ti) > 1 else 0.9)
                            norm = self._normalize_local_bubble_candidate(text)
                            if norm and norm not in seen and conf >= 0.6:
                                result = (norm, conf)
                                break
                        except Exception:
                            continue
                    if result:
                        break
                if result is None:
                    continue

            norm_text, conf = result
            if norm_text in seen:
                continue
            if self._recovered_id_collides_with_dimension(
                norm_text, cx, cy, cr, self._norm_tokens, set()
            ):
                continue
            # Reject if the recovered ID matches an existing bubble-type
            # OCR token that's near this blob.  This catches dimension
            # symbols (⌀56) whose circular annotation contour looks
            # like a bubble — the "56" is already in the token pool as
            # a bubble-type token, meaning it was OCR'd from the
            # drawing but isn't inside a real bubble circle.
            is_stray_dim_digit = False
            if hasattr(self, '_norm_tokens') and self._norm_tokens:
                for tok in self._norm_tokens:
                    if tok.token_type != "bubble":
                        continue
                    tok_norm = re.sub(r"[^0-9A-Z]", "", tok.text.upper())
                    if tok_norm != norm_text:
                        continue
                    # Token with same text must be nearby
                    if math.hypot(tok.cx - cx, tok.cy - cy) < median_r * 4:
                        # AND the token must NOT already be matched
                        # to a real detected bubble
                        matched = any(
                            math.hypot(tok.cx - b.x, tok.cy - b.y) < b.radius * 1.3
                            for b in bubbles
                        )
                        if not matched:
                            is_stray_dim_digit = True
                            break
            if is_stray_dim_digit:
                continue

            bubbles.append(BubbleResult(
                bubble_number=norm_text,
                x=cx, y=cy, radius=cr,
                confidence=conf,
                dimension="NO_DIMENSION",
                needs_review=True,
                review_reason="annotation_blob",
            ))
            seen.add(norm_text)

    def _remove_concatenated_ids(
        self,
        bubbles: List[BubbleResult],
    ) -> List[BubbleResult]:
        """
        Remove bubble IDs that appear to be formed by concatenating two or more
        already-detected single-token IDs.

        Example: if bubbles "4" and "5" are detected, then spurious "45" is removed.
        This catches OCR artifacts where adjacent circles are read as one number.
        Only applies to purely numeric IDs of length >= 2.
        """
        detected_ids = {b.bubble_number for b in bubbles}
        bubble_positions = {b.bubble_number: (b.x, b.y) for b in bubbles}
        to_remove: set = set()

        for b in bubbles:
            bid = b.bubble_number
            if not re.fullmatch(r"\d{2,}", bid):
                continue

            for split in range(1, len(bid)):
                part1 = bid[:split]
                part2 = bid[split:]
                if (
                    part1 in detected_ids
                    and part2 in detected_ids
                    and part1 != bid
                    and part2 != bid
                    and part1 != part2
                    and re.fullmatch(r"\d+", part1)
                    and re.fullmatch(r"\d+", part2)
                ):
                    # Only remove if the candidate is physically CLOSE
                    # to at least one of its parts — indicating a real
                    # OCR concatenation, not two separate bubbles.
                    p1 = bubble_positions.get(part1)
                    p2 = bubble_positions.get(part2)
                    close_to_part = False
                    if p1:
                        close_to_part = close_to_part or math.hypot(b.x - p1[0], b.y - p1[1]) < b.radius * 4
                    if p2:
                        close_to_part = close_to_part or math.hypot(b.x - p2[0], b.y - p2[1]) < b.radius * 4
                    if close_to_part:
                        to_remove.add(bid)
                        break

        if to_remove:
            bubbles = [b for b in bubbles if b.bubble_number not in to_remove]
        return bubbles

    def _recover_bubble_from_circle(
        self,
        image: np.ndarray,
        cx: int, cy: int, r: int,
    ) -> Optional[Tuple[str, float]]:
        """Try to OCR the number inside an unmatched circle."""
        # Use evidence score with a LOW threshold for recovery.
        # Recovery OCR validates by actually reading text inside —
        # if text is found, the circle is real regardless of color.
        # We just need to avoid wasting time on obviously wrong
        # circles (large drawing geometry with no contrast).
        score = self._compute_bubble_evidence(image, cx, cy, r)
        if score < 0.03:  # essentially zero evidence
            return None

        margin = max(8, int(r * 0.28))
        x1 = max(0, cx - r - margin)
        y1 = max(0, cy - r - margin)
        x2 = min(image.shape[1], cx + r + margin)
        y2 = min(image.shape[0], cy + r + margin)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if not hasattr(self, "_bubble_recovery_cache"):
            self._bubble_recovery_cache = {}
        cache_key = (
            int(round(x1 / 4.0)) * 4,
            int(round(y1 / 4.0)) * 4,
            int(round(x2 / 4.0)) * 4,
            int(round(y2 / 4.0)) * 4,
            int(round(r)),
        )
        if cache_key in self._bubble_recovery_cache:
            return self._bubble_recovery_cache[cache_key]
        if self._step6_recovery_budget_exceeded():
            self._bubble_recovery_cache[cache_key] = None
            return None

        scale = 4 if r <= 26 else 3
        upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        best: Optional[Tuple[float, str, float]] = None
        # Digit fragments collected across ALL variants — when OCR is
        # unstable on red-on-red glyphs it often returns single digits
        # of a 2-digit bubble on different variants (e.g. "14" → "1"
        # on scale-3 color, "4" on threshold). We merge spatially at
        # the end if none of the full-number reads passed.
        fragments: List[Tuple[float, str, float]] = []  # (x_px, digit, conf)

        # Run ALL variants and collect fragments + best full read in
        # a single pass so we can compare them at the end.
        for variant in [upscaled, cv2.cvtColor(255 - thresh, cv2.COLOR_GRAY2BGR)]:
            try:
                result = self.ocr(variant)
            except Exception:
                continue
            items = result[0] if (isinstance(result, tuple) and len(result) >= 1) else (result or [])
            if not items:
                continue
            for item in items:
                try:
                    text_info = item[1]
                    text = str(text_info[0] if isinstance(text_info, (list, tuple)) else text_info).strip()
                    conf = float(text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.9)
                    bbox = item[0]
                    bx = sum(p[0] for p in bbox) / len(bbox) / scale + x1
                    by = sum(p[1] for p in bbox) / len(bbox) / scale + y1
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    glyph_h = (max(ys) - min(ys)) / scale
                    d = math.dist((bx, by), (cx, cy))
                    if d > r * 0.8:
                        continue
                    if self._strict_photo_mode() and d > r * 0.48:
                        continue
                    if self._strict_photo_mode() and glyph_h < max(7.0, r * 0.32):
                        continue
                    # Track single-digit fragments separately — they may
                    # combine across variants into the real 2-digit ID.
                    if re.fullmatch(r"\d", text) and conf >= 0.70:
                        fragments.append((bx, text, conf))
                    norm = self._normalize_local_bubble_candidate(text)
                    if norm is None:
                        continue
                    if conf < 0.85:
                        continue
                    # Prefer higher confidence; on ties, prefer the
                    # LONGER normalized string — "14" must beat "1"
                    # when both come in at conf 0.90 from different
                    # variants of the same circle.
                    cand_key = (conf, len(norm))
                    best_key = (best[0], len(best[1])) if best else (-1.0, -1)
                    if cand_key > best_key:
                        best = (conf, norm, conf)
                except Exception:
                    continue

        # Fragment-merge upgrade: if the best full read is only a
        # single digit but we saw fragments at distinct x-positions,
        # those fragments likely ARE the two digits of a 2-digit
        # bubble that OCR couldn't read together. Prefer the merge.
        if fragments:
            fragments.sort(key=lambda f: f[0])
            clustered: List[Tuple[float, str, float]] = []
            for fx, ftext, fconf in fragments:
                if clustered and abs(fx - clustered[-1][0]) < r * 0.3:
                    # Same spatial position — keep higher-confidence read
                    if fconf > clustered[-1][2]:
                        clustered[-1] = (fx, ftext, fconf)
                else:
                    clustered.append((fx, ftext, fconf))
            if len(clustered) >= 2:
                combined = "".join(c[1] for c in clustered)
                norm = self._normalize_local_bubble_candidate(combined)
                if norm is not None:
                    avg_conf = sum(c[2] for c in clustered) / len(clustered)
                    # Prefer the merged result if either (a) no `best`
                    # was found, or (b) `best` is only 1 char and the
                    # merge yields something longer.
                    if best is None or (
                        len(best[1]) == 1 and len(norm) > len(best[1])
                    ):
                        best = (avg_conf, norm, avg_conf)

        recovered = (best[1], best[2]) if best else None
        self._bubble_recovery_cache[cache_key] = recovered
        return recovered

    def _recovered_id_collides_with_dimension(
        self,
        candidate_id: str,
        cx: int, cy: int, cr: int,
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
    ) -> bool:
        """
        Return True if a recovered bubble ID is likely a misread of a
        nearby dimension token.

        Example: circle at (252,516) reads "30" via recovery OCR, but
        there's a dimension token "3.0" at (220,515) — the candidate
        is actually reading the dimension text, not a real bubble number.
        """
        # Check both single-digit and multi-digit numeric IDs. The rule
        # below is what separates real bubbles from dimension misreads:
        # a dimension token always carries non-digit context around its
        # digits (parens, Ø, decimal point, units, etc.) — a bare bubble
        # number token does not.  So only flag a collision when the
        # nearby token's full text is STRICTLY LONGER than its extracted
        # digit string, i.e. carries non-digit context.
        if not re.fullmatch(r"\d{1,2}", candidate_id):
            return False

        search_radius = cr * 3.0
        for idx, t in enumerate(norm_tokens):
            if idx in used_token_ids:
                continue
            if t.token_type == "bubble":
                continue  # don't compare against other bubble tokens
            d = math.hypot(t.cx - cx, t.cy - cy)
            if d > search_radius:
                continue
            tok_digits = re.sub(r"[^0-9]", "", t.text)
            if tok_digits != candidate_id:
                continue
            # Unified rule — collision only when the token has non-digit
            # context (parens, Ø, decimal, tolerance, etc.).  A bare
            # digits-only token is another bubble candidate, not a
            # dimension, so do NOT flag it as a collision.
            if len(t.text) > len(tok_digits):
                return True
        return False

    def _normalize_local_bubble_candidate(self, text: str) -> Optional[str]:
        t = (text or "").strip().upper().replace(" ", "")
        if t in {"I", "L", "!", "|", "/", "\\"}:
            t = "1"
        t = re.sub(r"[^A-Z0-9]", "", t)
        if not t:
            return None
        if re.fullmatch(r"\d{1,2}[A-Z]?", t) and is_bubble_token(t):
            result = normalize_bubble_value(t)
            # "0" is never a valid bubble number — guard against the recovery
            # OCR path creating BubbleResult(bubble_number="0")
            if not result or result == "0":
                return None
            return result
        return None

    def _compute_bubble_evidence(
        self,
        image: np.ndarray,
        cx: int, cy: int, r: int,
    ) -> float:
        """Compute a multi-signal bubble evidence score in [0, 1].

        Combines four LOCAL, IMAGE-AGNOSTIC signals:

          1. CHROMA: does the rim have color (non-gray)?
             Balloons are typically colored (red, purple, blue).
             Score: mean chroma / 50, capped at 1.0.

          2. ANNOTATION MASK: does the rim overlap with the
             pre-extracted annotation layer?
             Score: fraction of rim pixels on the mask.

          3. CONTRAST: is the rim darker/lighter than the interior?
             Real balloons have a visible circle outline.
             Score: abs(rim_mean - interior_mean) / 80, capped at 1.0.

          4. CIRCULARITY: is there a circular gradient pattern?
             Sample rim at multiple radii — a real circle has
             consistent darkness at the rim radius.
             Score: fraction of rim pixels darker than background.

        Each signal is [0, 1]. Returns the average.
        No hardcoded threshold — caller decides what score is enough.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Sample rim pixels
        rim_bgr: list = []
        rim_gray: list = []
        ann_hits = 0
        total_samples = 0

        for ang_deg in range(0, 360, 5):
            ang = math.radians(ang_deg)
            px = int(round(cx + r * math.cos(ang)))
            py = int(round(cy + r * math.sin(ang)))
            if 0 <= px < w and 0 <= py < h:
                total_samples += 1
                rim_bgr.append(image[py, px].astype(np.float32))
                rim_gray.append(float(gray[py, px]))
                if (hasattr(self, '_annotation_layer')
                        and self._annotation_layer is not None
                        and self._annotation_layer.mask[py, px] > 0):
                    ann_hits += 1

        if total_samples == 0:
            return 0.0

        # Signal 1: chroma
        arr = np.array(rim_bgr)
        chroma_vals = arr.max(axis=1) - arr.min(axis=1)
        s_chroma = min(1.0, float(np.mean(chroma_vals)) / 50.0)

        # Signal 2: annotation mask overlap
        s_annotation = ann_hits / total_samples

        # Signal 3: contrast (rim vs interior)
        interior_gray: list = []
        for ang_deg in range(0, 360, 30):
            ang = math.radians(ang_deg)
            for frac in [0.3, 0.5, 0.7]:
                px = int(round(cx + r * frac * math.cos(ang)))
                py = int(round(cy + r * frac * math.sin(ang)))
                if 0 <= px < w and 0 <= py < h:
                    interior_gray.append(float(gray[py, px]))
        if interior_gray:
            contrast = abs(np.mean(rim_gray) - np.mean(interior_gray))
            s_contrast = min(1.0, contrast / 80.0)
        else:
            s_contrast = 0.0

        # Signal 4: rim darkness consistency
        bg_estimate = float(np.percentile(rim_gray, 90))
        dark_rim = sum(1 for g in rim_gray if g < bg_estimate - 15)
        s_dark_rim = dark_rim / total_samples

        return (s_chroma + s_annotation + s_contrast + s_dark_rim) / 4.0

    def _strict_photo_mode(self) -> bool:
        """True when quality signals suggest phone/screen-photo artifacts."""
        q = getattr(self, "_quality", None) or {}
        return bool(
            q.get("is_high_clutter")
            or q.get("is_low_resolution")
            or getattr(self, "_was_upscaled", False)
        )

    def _bubble_label_scale_ok(self, token: NormalizedToken, radius: float) -> bool:
        """Reject oversized circle fits around tiny numeric text on photo inputs."""
        if not self._strict_photo_mode():
            return True
        glyph_h = max(1.0, float(token.y2 - token.y1))
        return glyph_h >= max(7.0, float(radius) * 0.34)

    def _circle_too_clipped(
        self,
        image: np.ndarray,
        cx: int,
        cy: int,
        r: int,
    ) -> bool:
        """Reject large partial-arc Hough fits on noisy photo inputs."""
        if not self._strict_photo_mode():
            return False
        h, w = image.shape[:2]
        outside = 0
        total = 0
        for ang_deg in range(0, 360, 10):
            ang = math.radians(ang_deg)
            px = int(round(cx + r * math.cos(ang)))
            py = int(round(cy + r * math.sin(ang)))
            total += 1
            if not (0 <= px < w and 0 <= py < h):
                outside += 1
        return total > 0 and (outside / total) >= 0.18

    def _circle_has_annotation_tint(
        self,
        image: np.ndarray,
        cx: int, cy: int, r: int,
        strict: bool = False,
    ) -> bool:
        """Return True if the circle rim has annotation colour.

        Simple, robust check: mean BGR chroma >= 6.0 on rim pixels,
        OR grayscale gradient std >= 6.0 (for low-chroma drawings).
        Strict mode uses the annotation mask if available.
        """
        h, w = image.shape[:2]

        if strict and hasattr(self, '_annotation_layer') and self._annotation_layer is not None:
            mask = self._annotation_layer.mask
            ann_count = total = 0
            for ang_deg in range(0, 360, 5):
                ang = math.radians(ang_deg)
                px = int(round(cx + r * math.cos(ang)))
                py = int(round(cy + r * math.sin(ang)))
                if 0 <= px < w and 0 <= py < h:
                    total += 1
                    if mask[py, px] > 0:
                        ann_count += 1
            if total > 0:
                # In strict mode, annotation mask is definitive
                return (ann_count / total) >= 0.15

        # BGR chroma analysis. Hough's circle fit can be 2-4 px off
        # the actual rendered ring (which is itself only 2 px thick),
        # so sampling at exactly `r` sometimes lands inside the white
        # interior and reports zero chroma even on a clearly-colored
        # balloon. Sample at r, r-2, r-3, r+2, r+3 and take the BEST
        # mean chroma — this catches off-by-a-few Hough fits.
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        best_chroma = 0.0
        best_gray_std = 0.0
        for dr in (0, -2, 2, -3, 3):
            rr = r + dr
            if rr < 4:
                continue
            ring_pixels: list = []
            ring_gray: list = []
            for ang_deg in range(0, 360, 5):
                ang = math.radians(ang_deg)
                px = int(round(cx + rr * math.cos(ang)))
                py = int(round(cy + rr * math.sin(ang)))
                if 0 <= px < w and 0 <= py < h:
                    ring_pixels.append(image[py, px].astype(np.float32))
                    ring_gray.append(float(gray_full[py, px]))
            if not ring_pixels:
                continue
            arr = np.array(ring_pixels)
            chroma = arr.max(axis=1) - arr.min(axis=1)
            mean_chroma = float(np.mean(chroma))
            if mean_chroma > best_chroma:
                best_chroma = mean_chroma
            if ring_gray:
                gstd = float(np.std(ring_gray))
                if gstd > best_gray_std:
                    best_gray_std = gstd
            # Early-out: clearly colored ring at this radius
            if mean_chroma >= 6.0:
                return True
        # Fall back to grayscale gradient — works for low-chroma
        # (e.g. dark-grey) annotation rings.
        return best_gray_std >= 6.0

    # ── Callout Construction ───────────────────────────────────────

    @staticmethod
    def _is_noise_token(text: str) -> bool:
        """True for OCR artifacts that are definitely not engineering dimensions."""
        t = text.strip()
        # Single "0" — OCR artifact, not a real dimension
        if t == "0":
            return True
        # Pure repeated zeros like "000", "0000"
        if re.fullmatch(r"0{2,}", t):
            return True
        # "0,0" is the origin-cross annotation on engineering drawings (+X/-Y etc.)
        # It is NOT a dimension value and must not be matched to any bubble.
        if t == "0,0":
            return True
        # Footer/title-block OCR and email/URL fragments are never
        # engineering dimension callouts.
        if "@" in t or re.search(r"\b(?:COM|WWW|HTTP|MAIL)\b", t, re.I):
            return True
        return False

    def _build_callouts(
        self,
        norm_tokens: List[NormalizedToken],
        used_token_ids: set,
        bubbles: List[BubbleResult],
        scale_factor: float = 1.0,
    ) -> List[CalloutGroup]:
        bubble_ids = {b.bubble_number for b in bubbles}
        bubble_positions = {
            b.bubble_number: (float(b.x), float(b.y), float(b.radius))
            for b in bubbles
        }
        remaining = [
            self._reinterpret_unassigned_bubble_token(
                t, bubble_ids, bubble_positions, all_tokens=norm_tokens,
            )
            for idx, t in enumerate(norm_tokens)
            if idx not in used_token_ids
            and not self._is_noise_token(t.text)
        ]
        remaining = self._recover_vertical_digit_stack_callouts(
            remaining, bubbles, scale_factor,
        )

        groups = build_callout_groups(remaining, scale_factor=scale_factor)
        groups = self._dedup_callout_groups(groups)
        return groups

    def _recover_vertical_digit_stack_callouts(
        self,
        tokens: List[NormalizedToken],
        bubbles: List[BubbleResult],
        scale_factor: float,
    ) -> List[NormalizedToken]:
        """Add synthetic callout tokens for vertical numeric OCR fragments.

        Phone/photo OCR sometimes reads a rotated dimension like ``59.5`` as
        three separate black digit tokens in one vertical column (``5``, ``9``,
        ``5``), while missing the decimal point. Individual-token logic cannot
        safely reclassify those because a nearby bubble may share the same
        digit. This grouped recovery only uses non-annotation-colour tokens and
        only emits a synthetic dimension when the stack sits near a detected
        bubble, so table row numbers and notes do not become callouts.
        """
        if len(tokens) < 3 or not bubbles:
            return tokens

        digit_tokens = [
            t for t in tokens
            if not getattr(t, "is_maroon", False)
            and re.fullmatch(r"\d", (t.text or "").strip())
            and (t.x2 - t.x1) > 3
            and (t.y2 - t.y1) > 6
        ]
        if len(digit_tokens) < 3:
            return tokens

        max_x_gap = max(14.0, 18.0 * scale_factor)
        max_center_gap = max(34.0, 46.0 * scale_factor)
        min_center_gap = max(5.0, 5.0 * scale_factor)
        max_stack_span = max(70.0, 105.0 * scale_factor)
        max_bubble_dist = max(
            180.0 * scale_factor,
            max(float(b.radius) for b in bubbles) * 6.0,
        )

        ordered = sorted(digit_tokens, key=lambda t: (t.cx, t.cy))
        added: List[NormalizedToken] = []
        seen_keys: set = set()

        for i, seed in enumerate(ordered):
            column = [
                t for t in ordered
                if abs(t.cx - seed.cx) <= max_x_gap
            ]
            column.sort(key=lambda t: t.cy)
            for start in range(len(column) - 2):
                stack = column[start:start + 3]
                if seed not in stack:
                    continue
                gaps = [
                    stack[k + 1].cy - stack[k].cy
                    for k in range(len(stack) - 1)
                ]
                if any(g < min_center_gap or g > max_center_gap for g in gaps):
                    continue
                if stack[-1].y2 - stack[0].y1 > max_stack_span:
                    continue
                if max(t.x2 for t in stack) - min(t.x1 for t in stack) > max_x_gap * 2.6:
                    continue

                cx = sum(t.cx for t in stack) / len(stack)
                cy = sum(t.cy for t in stack) / len(stack)
                near_bubble = any(
                    math.hypot(cx - b.x, cy - b.y) <= max_bubble_dist
                    and math.hypot(cx - b.x, cy - b.y) > b.radius * 1.15
                    for b in bubbles
                )
                if not near_bubble:
                    continue

                digits = "".join(t.text.strip() for t in stack)
                # Three isolated vertical digits on engineering drawings most
                # often represent a one-decimal value whose dot was too small
                # for OCR, e.g. 595 -> 59.5. Keep this recovery narrow; longer
                # stacks and true integers are left to normal OCR/grouping.
                combined = f"{digits[:2]}.{digits[2]}"
                key = (
                    combined,
                    round(cx / max_x_gap),
                    round(cy / max_center_gap),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                xs = [t.x1 for t in stack] + [t.x2 for t in stack]
                ys = [t.y1 for t in stack] + [t.y2 for t in stack]
                added.append(NormalizedToken(
                    raw_text=combined,
                    text=combined,
                    cx=cx,
                    cy=cy,
                    conf=sum(t.conf for t in stack) / len(stack),
                    x1=min(xs),
                    y1=min(ys),
                    x2=max(xs),
                    y2=max(ys),
                    token_type="dimension",
                    semantic_type="vertical_numeric_stack",
                ))

        if added:
            logger.info("Recovered %d vertical digit-stack callout(s)", len(added))
            return tokens + added
        return tokens

    @staticmethod
    def _dedup_callout_groups(groups: List[CalloutGroup]) -> List[CalloutGroup]:
        """Remove near-duplicate callout groups (same text, nearby position).

        Prefers structured groups (chamfer, thread, diameter) over
        singletons, and longer text over shorter when positions overlap.
        """
        # Sort: prefer structured types, then longer text
        type_rank = {"chamfer": 0, "thread": 1, "diameter": 2,
                     "diameter_tolerance": 2, "numeric_pair": 3}
        groups_sorted = sorted(
            groups,
            key=lambda g: (type_rank.get(g.callout_type, 9), -len(g.text)),
        )
        kept: List[CalloutGroup] = []
        for g in groups_sorted:
            # Check if a kept group already covers this one
            is_dup = False
            g_text_norm = re.sub(r"[^A-Z0-9.]", "", g.text.upper())
            for k in kept:
                k_text_norm = re.sub(r"[^A-Z0-9.]", "", k.text.upper())
                pos_close = math.hypot(g.cx - k.cx, g.cy - k.cy) < 60
                if not pos_close:
                    continue
                # Exact same normalised text → dup
                if g_text_norm == k_text_norm:
                    is_dup = True
                    break
                # Short token near a longer structured callout is likely
                # a consumed prefix (e.g. "82" near "82 0.5x45")
                if (len(g_text_norm) <= 3
                        and len(k_text_norm) > 3
                        and k.callout_type in ("chamfer", "thread", "diameter")):
                    is_dup = True
                    break
                # Substring check — but DO NOT eliminate individual
                # numeric tokens that are substrings of stacked pairs.
                is_stacked = "/" in k.text or "/" in g.text
                if not is_stacked and (g_text_norm in k_text_norm
                                       or k_text_norm in g_text_norm):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(g)
        return kept

    def _suppress_phantom_digits(
        self,
        tokens: List[NormalizedToken],
        bubbles: List[BubbleResult],
        leader_segments,
    ) -> List[NormalizedToken]:
        """Suppress phantom single-digit OCR tokens.

        A phantom digit floats in empty space with no line geometry.
        A real dimension digit sits at a dimension line endpoint
        (arrowhead or tick mark).

        Uses HoughLinesP segment endpoints as the discriminator:
          - Real dimension: a line segment endpoint within 40px
          - Phantom: no line geometry nearby

        This is geometrically correct — dimension values in
        engineering drawings ALWAYS terminate a dimension line.
        """
        balloon_regions = [
            (float(b.x), float(b.y), float(b.radius))
            for b in bubbles
        ]

        # Collect all segment endpoints
        seg_endpoints: List[Tuple[float, float]] = []
        if leader_segments:
            for seg in leader_segments:
                seg_endpoints.append((float(seg.x1), float(seg.y1)))
                seg_endpoints.append((float(seg.x2), float(seg.y2)))

        ENDPOINT_RADIUS = 40.0

        out: List[NormalizedToken] = []
        for tok in tokens:
            t = tok.text.strip()

            if getattr(tok, "dual_use", False):
                out.append(tok)
                continue

            # Only evaluate single bare digits
            if not re.fullmatch(r"\d", t):
                out.append(tok)
                continue

            # (a) Inside or touching any balloon → real → keep
            if any(
                math.hypot(tok.cx - bx, tok.cy - by) <= br + 8
                for bx, by, br in balloon_regions
            ):
                out.append(tok)
                continue

            # (d) Any HoughLinesP endpoint within 40px → real → keep
            endpoint_nearby = any(
                math.hypot(tok.cx - ex, tok.cy - ey) < ENDPOINT_RADIUS
                for ex, ey in seg_endpoints
            )
            if endpoint_nearby:
                out.append(tok)
                continue

            # Also check segment midpoints within 25px (fallback for
            # short dimension lines where the endpoint is slightly off)
            if leader_segments:
                midpoint_nearby = any(
                    math.hypot(
                        tok.cx - (seg.x1 + seg.x2) / 2,
                        tok.cy - (seg.y1 + seg.y2) / 2,
                    ) < 25
                    for seg in leader_segments
                )
                if midpoint_nearby:
                    out.append(tok)
                    continue

            # All conditions met → suppress
            out.append(NormalizedToken(
                raw_text=tok.raw_text, text=tok.text,
                cx=tok.cx, cy=tok.cy, conf=tok.conf,
                x1=tok.x1, y1=tok.y1, x2=tok.x2, y2=tok.y2,
                token_type=tok.token_type,
                semantic_type="suppressed",
            ))

        return out

    def _reinterpret_unassigned_bubble_token(
        self,
        token: NormalizedToken,
        detected_bubble_ids: set,
        bubble_positions: Dict[str, Tuple[float, float, float]] = {},
        all_tokens: Optional[List[NormalizedToken]] = None,
    ) -> NormalizedToken:
        if token.token_type != "bubble":
            return token

        tok_id = token.text.strip()

        # Normalised form: strip parentheses (or partial parens from OCR
        # mis-segmentation). "(82)", "82)", "(82" all normalise to "82".
        # Without this, partial-paren reads of a real bubble ID get
        # reclassified as dimensions and leak into dim_groups.
        norm_id = tok_id
        if norm_id.startswith("("):
            norm_id = norm_id[1:]
        if norm_id.endswith(")"):
            norm_id = norm_id[:-1]
        norm_id = norm_id.strip()

        # Token doesn't match any detected bubble → dimension value
        if tok_id not in detected_bubble_ids and norm_id not in detected_bubble_ids:
            # Phantom-token filter: a bare-numeric bubble-typed token
            # sitting on non-annotation-coloured background is often an OCR
            # fragment of a longer dimension token or a
            # false-positive bubble-text read on a non-bubble part of
            # the drawing. Real bubble labels live on the maroon
            # balloon background; real dimensions normally classify
            # as token_type="dimension" directly.
            #
            # We only drop "bare" numeric reads (raw_text is pure
            # digits/parens, no Ø/Φ/R/M/±/° prefix). When the raw OCR
            # caught a dimension-indicator character, the token IS a
            # real dim that the normaliser stripped down to digits; those
            # must keep flowing into the callout pool.
            #
            # Keeping the token as token_type="bubble" means
            # build_callout_groups (which only accepts dimension /
            # keyword) will skip it.
            raw = (token.raw_text or "").strip()
            bare_numeric = bool(re.fullmatch(r"[()\d.]+", raw)) and bool(re.search(r"\d", raw))
            if (
                not getattr(token, "is_maroon", False)
                and bare_numeric
                and "(" not in raw and ")" not in raw
            ):
                # Refinement: only KEEP as bubble (filter from callouts)
                # when this token's bbox is INSIDE another dim token's
                # bbox — that's the true "OCR fragment of larger dim"
                # case. A standalone bare
                # numeric adjacent to but NOT enclosed by other dim
                # tokens can be a legitimate callout reference.
                is_fragment = False
                if all_tokens is not None:
                    for other in all_tokens:
                        if other is token:
                            continue
                        if other.token_type != "dimension":
                            continue
                        # Bbox containment with small slack
                        contains = (
                            other.x1 - 2 <= token.x1
                            and other.y1 - 2 <= token.y1
                            and other.x2 + 2 >= token.x2
                            and other.y2 + 2 >= token.y2
                        )
                        if contains:
                            is_fragment = True
                            break
                if is_fragment:
                    return token
            return NormalizedToken(
                raw_text=token.raw_text, text=token.text,
                cx=token.cx, cy=token.cy, conf=token.conf,
                x1=token.x1, y1=token.y1, x2=token.x2, y2=token.y2,
                token_type="dimension", semantic_type="numeric",
            )

        # Token matches a known bubble ID — check spatial distance.
        # If this token is far from its bubble's circle, it's a
        # standalone dimension value that happens to be the same
        # digit (e.g. "1" as a shaft diameter when bubble 1 exists).
        # Use the normalised id (paren-stripped) for the lookup so
        # partial-paren reads also map to their real bubble.
        lookup_id = tok_id if tok_id in bubble_positions else norm_id
        if lookup_id in bubble_positions:
            bx, by, br = bubble_positions[lookup_id]
            dist = math.hypot(token.cx - bx, token.cy - by)
            # Threshold depends on token length.
            # Single digits (1-9) are almost always bubble IDs — need
            # very large distance to be reclassified as dimensions.
            # Multi-digit tokens (10+) or tokens with letters are more
            # likely genuine dimensions.
            threshold = max(2.5 * br, 150.0)
            if dist > threshold:
                sem = "dual_use_digit" if len(tok_id) == 1 else "numeric"
                return NormalizedToken(
                    raw_text=token.raw_text, text=token.text,
                    cx=token.cx, cy=token.cy, conf=token.conf,
                    x1=token.x1, y1=token.y1, x2=token.x2, y2=token.y2,
                    token_type="dimension", semantic_type=sem,
                )

        # Token is close to its matching bubble → keep as bubble label
        return token

    # ── Steps 7-8: Seed detection + path tracing ───────────────────

    def _score_leader_trace_quality(
        self,
        bubble: BubbleResult,
        path: Optional[List[Tuple[int, int]]],
        callout_groups: List[CalloutGroup],
    ) -> TraceQuality:
        """Score whether a traced path is trustworthy enough for assignment."""
        br = max(1.0, float(getattr(bubble, "radius", 1) or 1))
        bx, by = float(bubble.x), float(bubble.y)
        if not path or len(path) < 2:
            return TraceQuality(
                score=0.0,
                path_points=0,
                endpoint_distance=0.0,
                target_distance=float("inf"),
                straightness=0.0,
                continuity=0.0,
                escape_score=0.0,
                target_score=0.0,
                touches_callout=False,
                reason="no_trace_path",
            )

        pts = [(float(x), float(y)) for x, y in path]
        ex, ey = pts[-1]
        endpoint_distance = math.hypot(ex - bx, ey - by)
        segment_lengths = [
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        ]
        total_length = max(1e-6, sum(segment_lengths))
        net_length = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        straightness = max(0.0, min(1.0, net_length / total_length))
        escape_score = max(0.0, min(1.0, (endpoint_distance / br - 1.05) / 2.8))

        direction_stability = straightness
        if len(pts) >= 4 and net_length > 1e-6:
            ux = (pts[-1][0] - pts[0][0]) / net_length
            uy = (pts[-1][1] - pts[0][1]) / net_length
            dots: List[float] = []
            for i in range(1, len(pts)):
                sl = segment_lengths[i - 1]
                if sl <= 1e-6:
                    continue
                sx = (pts[i][0] - pts[i - 1][0]) / sl
                sy = (pts[i][1] - pts[i - 1][1]) / sl
                dots.append(max(0.0, sx * ux + sy * uy))
            if dots:
                direction_stability = max(0.0, min(1.0, float(sum(dots) / len(dots))))

        continuity_hits = 0
        continuity_total = 0
        ann_mask = getattr(getattr(self, "_annotation_layer", None), "mask", None)
        gray = None
        if self.image is not None:
            gray = (
                cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                if len(self.image.shape) == 3 else self.image
            )
        for px_f, py_f in pts:
            px = int(round(px_f))
            py = int(round(py_f))
            if self.image is not None:
                h, w = self.image.shape[:2]
                if not (0 <= px < w and 0 <= py < h):
                    continue
            continuity_total += 1
            hit = False
            if ann_mask is not None and 0 <= py < ann_mask.shape[0] and 0 <= px < ann_mask.shape[1]:
                hit = bool(ann_mask[py, px] > 0)
            if not hit and gray is not None and 0 <= py < gray.shape[0] and 0 <= px < gray.shape[1]:
                hit = bool(gray[py, px] < 210)
            if hit:
                continuity_hits += 1
        continuity = (
            float(continuity_hits) / float(continuity_total)
            if continuity_total else 0.0
        )

        target_distance = float("inf")
        touches_callout = False
        for c in callout_groups:
            if not self._is_assignable_callout(
                c,
                allow_reference_only=True,
                allow_keyword_only=True,
            ):
                continue
            box = (float(c.x1), float(c.y1), float(c.x2), float(c.y2))
            target_distance = min(
                target_distance,
                self._point_to_box_distance(ex, ey, *box),
            )
            pad = max(4.0, br * 0.20)
            if any(
                self._point_in_expanded_box(float(px), float(py), *box, pad)
                for px, py in path[-12:]
            ):
                touches_callout = True

        target_limit = max(90.0, br * 3.5)
        target_score = (
            max(0.0, min(1.0, 1.0 - target_distance / target_limit))
            if math.isfinite(target_distance) else 0.0
        )
        if touches_callout:
            target_score = max(target_score, 0.92)

        score = (
            0.26 * escape_score
            + 0.18 * straightness
            + 0.16 * direction_stability
            + 0.18 * continuity
            + 0.22 * target_score
        )
        if len(path) < max(4, int(br * 0.35)):
            score *= 0.72
        if endpoint_distance <= br * 1.15:
            score *= 0.40

        reasons: List[str] = []
        if endpoint_distance <= br * 1.15:
            reasons.append("endpoint_near_balloon")
        if continuity < 0.35:
            reasons.append("low_ink_continuity")
        if straightness < 0.35:
            reasons.append("wandering_trace")
        if target_score < 0.25:
            reasons.append("no_near_callout")
        if touches_callout:
            reasons.append("touches_callout")
        if not reasons:
            reasons.append("usable_trace")

        return TraceQuality(
            score=max(0.0, min(1.0, float(score))),
            path_points=len(path),
            endpoint_distance=float(endpoint_distance),
            target_distance=float(target_distance),
            straightness=float(straightness),
            continuity=float(continuity),
            escape_score=float(escape_score),
            target_score=float(target_score),
            touches_callout=bool(touches_callout),
            reason="+".join(reasons),
        )

    def _trace_balloon_leader_paths(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
    ) -> Tuple[Dict[str, Any], Dict[int, Tuple[float, float]]]:
        """
        For each bubble:
          Step 7 — detect leader seed (HoughLinesP v2 → color fallback)
          Step 8 — trace leader path from seed

        Returns
        -------
        seed_traces : dict bubble_number → {"seed": ..., "path": ...}
        leader_dirs : dict bubble_index → (dir_x, dir_y)
        """
        if self.image is None:
            return {}, {}

        note_boxes = [(c.x1, c.y1, c.x2, c.y2) for c in callout_groups]
        seed_traces: Dict[str, Any] = {}
        seed_traces_by_index: Dict[int, Any] = {}
        leader_dirs: Dict[int, Tuple[float, float]] = {}

        has_annotation = (self._annotation_layer is not None
                          and self._annotation_layer.confidence > 0.15)

        # Separate trace store used ONLY by the diagnostic mask render.
        # Allows the visualization to use outward-corrected directions
        # (which produces visible leader stubs) WITHOUT changing the
        # production `_seed_traces` data the Hungarian + propagation
        # pipeline was tuned against.
        render_seed_traces: Dict[str, Any] = {}

        # Saturation-boosted maroon mask used ONLY for the render-only
        # re-trace below. The production annotation_layer.mask uses
        # strict thresholds and misses faint / anti-aliased stubs; the
        # boosted variant (sat × 2.0, sat_min=40) matches what the
        # mask renderer in v1_unified_view sees, so the trace can walk
        # ANY stub the user sees as light-grey in the rendered output.
        render_mask = None
        try:
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.int32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255)
            boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            hsv_b = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)
            m_red = cv2.inRange(hsv_b,
                                np.array([0, 35, 35]), np.array([12, 255, 255]))
            m_blue_purp = cv2.inRange(hsv_b,
                                      np.array([85, 30, 35]), np.array([180, 255, 255]))
            render_mask = cv2.morphologyEx(
                cv2.bitwise_or(m_red, m_blue_purp),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
        except Exception:
            render_mask = self._annotation_layer.mask if has_annotation else None

        for b_idx, b in enumerate(bubbles):
            geom = BalloonGeometry(cx=float(b.x), cy=float(b.y), radius=float(b.radius))

            # Step 7: detect leader seed at balloon boundary
            seed = detect_best_leader_seed(self.image, geom)
            # 3rd-tier seed fallback: when both the HoughLinesP-based and
            # color-based seed detectors return nothing, sweep the rim
            # of the boosted maroon mask for any outward annotation
            # trail. This catches drawings whose leaders are technically
            # annotation-coloured but with low saturation or thin enough
            # to fall under the strict seed detectors' thresholds.
            if seed is None and render_mask is not None:
                scan = _perimeter_scan_for_stub(
                    render_mask, (float(b.x), float(b.y)), float(b.radius),
                )
                if scan is not None:
                    (ex, ey), (dx, dy) = scan
                    seed = LeaderSeed(
                        contact_x=int(ex),
                        contact_y=int(ey),
                        dir_x=float(dx),
                        dir_y=float(dy),
                        confidence=0.30,
                        source="boosted_mask_perimeter",
                    )
            if seed is None:
                continue

            leader_dirs[b_idx] = (seed.dir_x, seed.dir_y)

            # Step 8: trace leader path.
            # Use annotation-color trace first (reliable when leaders
            # share the balloon color), fallback to general trace.
            path = None
            if has_annotation:
                path = trace_annotation_path(
                    annotation_mask=self._annotation_layer.mask,
                    start_x=seed.contact_x,
                    start_y=seed.contact_y,
                    direction=(seed.dir_x, seed.dir_y),
                    balloon_cx=float(b.x),
                    balloon_cy=float(b.y),
                    balloon_radius=float(b.radius),
                    callout_boxes=note_boxes,
                )

            if path is None or len(path) < 3:
                path = trace_leader_from_seed(
                    image=self.image,
                    balloon_geom=geom,
                    seed=seed,
                    boxes=note_boxes,
                    callouts=callout_groups,
                )

            # 3rd-tier perimeter-scan trace on the saturation-boosted
            # mask — ONLY fires when both prior trace tiers returned
            # nothing (or a path that never escaped the rim). Anything
            # else is left untouched: the seed-based trace already
            # captures the leader and replacing it can shift cost-
            # matrix signals downstream.
            prod_max = 0.0
            prod_endpoint_dist = 0.0
            if path:
                for px, py in path:
                    prod_max = max(prod_max, math.hypot(px - b.x, py - b.y))
                pe = path[-1]
                prod_endpoint_dist = math.hypot(pe[0] - b.x, pe[1] - b.y)
            radial_x = float(seed.contact_x) - float(b.x)
            radial_y = float(seed.contact_y) - float(b.y)
            weak_production_trace = (
                path is None
                or len(path) < 8
                or prod_max <= float(b.radius) * 1.55
                or prod_endpoint_dist <= float(b.radius) * 1.45
                or _is_rim_dir(seed.dir_x, seed.dir_y, radial_x, radial_y)
            )
            current_target_score = float("inf")
            if path:
                ex_cur, ey_cur = path[-1]
                for c in callout_groups:
                    if not self._is_assignable_callout(
                        c,
                        allow_reference_only=True,
                        allow_keyword_only=True,
                    ):
                        continue
                    current_target_score = min(
                        current_target_score,
                        self._point_to_box_distance(
                            float(ex_cur), float(ey_cur),
                            float(c.x1), float(c.y1), float(c.x2), float(c.y2),
                        ),
                    )
            if render_mask is not None:
                perimeter_path = _outward_rim_trace(
                    mask=render_mask,
                    contact=(int(seed.contact_x), int(seed.contact_y)),
                    radial=(seed.dir_x, seed.dir_y),
                    balloon_centre=(float(b.x), float(b.y)),
                    balloon_radius=float(b.radius),
                    max_steps=400,
                )
                perimeter_endpoint_dist = 0.0
                if perimeter_path:
                    ex_p, ey_p = perimeter_path[-1]
                    perimeter_endpoint_dist = math.hypot(ex_p - b.x, ey_p - b.y)
                perimeter_target_score = float("inf")
                if perimeter_path:
                    ex_per, ey_per = perimeter_path[-1]
                    for c in callout_groups:
                        if not self._is_assignable_callout(
                            c,
                            allow_reference_only=True,
                            allow_keyword_only=True,
                        ):
                            continue
                        perimeter_target_score = min(
                            perimeter_target_score,
                            self._point_to_box_distance(
                                float(ex_per), float(ey_per),
                                float(c.x1), float(c.y1), float(c.x2), float(c.y2),
                            ),
                        )
                perimeter_better_target = (
                    perimeter_target_score + max(8.0, float(b.radius) * 0.35)
                    < current_target_score
                )
                if (
                    perimeter_path
                    and len(perimeter_path) >= 3
                    and perimeter_endpoint_dist > float(b.radius) * 1.25
                    and (
                        weak_production_trace
                        or perimeter_better_target
                        or current_target_score > max(90.0, float(b.radius) * 3.0)
                    )
                ):
                    path = perimeter_path
                    ex_p, ey_p = perimeter_path[-1]
                    dx = ex_p - b.x
                    dy = ey_p - b.y
                    dn = math.hypot(dx, dy)
                    if dn > 1e-6:
                        leader_dirs[b_idx] = (dx / dn, dy / dn)

            final_target_score = float("inf")
            if path:
                ex_final, ey_final = path[-1]
                for c in callout_groups:
                    if not self._is_assignable_callout(
                        c,
                        allow_reference_only=True,
                        allow_keyword_only=True,
                    ):
                        continue
                    final_target_score = min(
                        final_target_score,
                        self._point_to_box_distance(
                            float(ex_final), float(ey_final),
                            float(c.x1), float(c.y1), float(c.x2), float(c.y2),
                        ),
                    )

            trace_quality = self._score_leader_trace_quality(b, path, callout_groups)
            trace_record = {
                "seed": seed,
                "path": path,
                "target_score": final_target_score,
                "quality": trace_quality,
                "quality_score": trace_quality.score,
                "quality_reason": trace_quality.reason,
                "bubble_center": (float(b.x), float(b.y)),
                "bubble_number": str(b.bubble_number),
            }
            seed_traces_by_index[b_idx] = trace_record

            trace_key = str(b.bubble_number)
            existing_trace = seed_traces.get(trace_key)
            existing_score = (
                float(existing_trace.get("target_score", float("inf")))
                if existing_trace else float("inf")
            )
            existing_len = len((existing_trace or {}).get("path") or [])
            new_len = len(path or [])
            if (
                existing_trace is None
                or final_target_score + 1e-6 < existing_score
                or (math.isinf(final_target_score) and new_len > existing_len)
            ):
                seed_traces[trace_key] = trace_record

            # ── Render-only re-trace with outward-corrected direction
            # The seed detector occasionally returns a direction tangent
            # to or pointing into the rim; the production BFS then walks
            # along the maroon rim instead of into the leader stub. For
            # the diagnostic mask render we want to SEE the stub, so
            # rerun the trace with the radial-outward direction whenever
            # the production trace doesn't leave the rim clearly.
            if render_mask is not None:
                radial_x = float(seed.contact_x) - float(b.x)
                radial_y = float(seed.contact_y) - float(b.y)
                radial_norm = math.hypot(radial_x, radial_y)
                prod_max = 0.0
                for px, py in (path or []):
                    dd = math.hypot(px - b.x, py - b.y)
                    if dd > prod_max:
                        prod_max = dd
                prod_endpoint_dist = 0.0
                if path:
                    pe = path[-1]
                    prod_endpoint_dist = math.hypot(pe[0] - b.x, pe[1] - b.y)
                weak_production = (
                    prod_max <= b.radius * 1.5
                    or prod_endpoint_dist <= b.radius * 1.5
                )

                if radial_norm > 1e-6 and (
                    weak_production
                    or _is_rim_dir(seed.dir_x, seed.dir_y, radial_x, radial_y)
                ):
                    rx = radial_x / radial_norm
                    ry = radial_y / radial_norm
                    try:
                        render_path = _outward_rim_trace(
                            mask=render_mask,
                            contact=(int(seed.contact_x), int(seed.contact_y)),
                            radial=(rx, ry),
                            balloon_centre=(float(b.x), float(b.y)),
                            balloon_radius=float(b.radius),
                            max_steps=400,
                        )
                        if render_path and len(render_path) >= 3:
                            import copy as _copy
                            render_seed = _copy.copy(seed)
                            try:
                                render_seed.dir_x = rx
                                render_seed.dir_y = ry
                            except Exception:
                                pass
                            render_seed_traces[b.bubble_number] = {
                                "seed": render_seed, "path": render_path,
                            }
                    except Exception as exc:
                        logger.debug("render re-trace failed for #%s: %s",
                                     b.bubble_number, exc)

        # Expose render traces as detector state (used only by the
        # diagnostic mask renderer; production code paths must continue
        # to read from `_seed_traces`).
        self._render_seed_traces = render_seed_traces
        self._seed_traces_by_index = seed_traces_by_index

        return seed_traces, leader_dirs

    # ── Step 9: Trace-first assignment ────────────────────────────

    def _assign_leader_first_candidates(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
    ) -> None:
        """Assign only callouts with direct traced-leader support.

        This is the first layer of the production assignment architecture:
        collect evidence-ranked candidates, claim strong physical links, and
        leave ambiguous bubbles for the existing global fallback. It uses
        instance-keyed traces so repeated physical bubble labels in separate
        drawing views do not share one trace by accident.
        """
        traces_by_index = getattr(self, "_seed_traces_by_index", {}) or {}
        if not bubbles or not callout_groups or not traces_by_index:
            return

        candidates: List[AssignmentCandidate] = []
        dense_page = len(bubbles) > int(self.cfg.dense_leader_trace_bubble_count)
        min_quality = (
            float(self.cfg.min_dense_leader_trace_quality)
            if dense_page else float(self.cfg.min_leader_trace_quality)
        )
        for bi, b in enumerate(bubbles):
            if b.dimension and b.dimension != "NO_DIMENSION":
                continue
            trace = traces_by_index.get(bi)
            path = (trace or {}).get("path") or []
            if trace is not None:
                trace["leader_first_quality_threshold"] = min_quality
            if len(path) < 2:
                if trace is not None:
                    trace["leader_first_status"] = "rejected_no_path"
                continue
            quality = (trace or {}).get("quality")
            quality_score = float(
                getattr(quality, "score", (trace or {}).get("quality_score", 0.0)) or 0.0
            )
            if quality_score < min_quality:
                if trace is not None:
                    trace["leader_first_status"] = "rejected_trace_quality"
                continue

            br = max(1.0, float(b.radius or 1))
            ex, ey = map(float, path[-1])
            endpoint_from_bubble = math.hypot(ex - float(b.x), ey - float(b.y))
            if endpoint_from_bubble <= br * 1.10:
                if trace is not None:
                    trace["leader_first_status"] = "rejected_endpoint_near_balloon"
                continue

            local_candidates: List[Dict[str, Any]] = []
            for ci, c in enumerate(callout_groups):
                text = (c.text or "").strip()
                if not text:
                    continue
                if self._is_weak_bare_integer_callout(c):
                    continue
                if self._is_table_description_assignment(text):
                    continue
                if not self._is_assignable_callout(
                    c,
                    allow_reference_only=True,
                    allow_keyword_only=True,
                ):
                    continue

                box = (float(c.x1), float(c.y1), float(c.x2), float(c.y2))
                path_touches = any(
                    self._point_in_expanded_box(
                        float(px), float(py), *box, max(4.0, br * 0.20)
                    )
                    for px, py in path
                )
                if not path_touches:
                    tx, ty = ex - float(b.x), ey - float(b.y)
                    tn = math.hypot(tx, ty)
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    vx, vy = cx - ex, cy - ey
                    vn = math.hypot(vx, vy)
                    forward = True
                    if tn > 1e-6 and vn > 1e-6:
                        forward = ((tx / tn) * (vx / vn) + (ty / tn) * (vy / vn)) >= 0.20
                    if not forward:
                        continue

                endpoint_dist = self._point_to_box_distance(ex, ey, *box)
                if not path_touches and endpoint_dist > max(48.0, br * 2.75):
                    continue

                if endpoint_dist > max(70.0, br * 3.2):
                    continue

                text_quality = self._score_text_quality(text)
                if text_quality <= -5.0:
                    continue

                score = 100.0
                score += min(25.0, max(-10.0, text_quality))
                score -= endpoint_dist * 0.35
                if path_touches:
                    score += 25.0
                if self._is_reference_only_callout(c):
                    score -= 8.0
                score += quality_score * 18.0

                if score < 75.0:
                    continue
                local_candidates.append({
                    "callout_index": ci,
                    "text": text,
                    "score": round(float(score), 2),
                    "endpoint_distance": round(float(endpoint_dist), 1),
                    "path_touches": bool(path_touches),
                })
                candidates.append(AssignmentCandidate(
                    bubble_index=bi,
                    callout_index=ci,
                    score=score,
                    endpoint_distance=endpoint_dist,
                    text_quality=text_quality,
                    reason="leader_first_trace",
                ))
            if trace is not None:
                if local_candidates:
                    local_candidates.sort(key=lambda item: (-item["score"], item["endpoint_distance"]))
                    trace["leader_first_status"] = "candidate"
                    trace["leader_first_candidates"] = local_candidates[:5]
                else:
                    trace["leader_first_status"] = "rejected_no_supported_callout"

        if not candidates:
            return

        candidates.sort(key=lambda c: (-c.score, c.endpoint_distance))
        claimed_bubbles: Set[int] = set()
        claimed_callouts: Set[int] = set()
        assigned = 0
        for cand in candidates:
            if cand.bubble_index in claimed_bubbles:
                continue
            if cand.callout_index in claimed_callouts:
                continue
            b = bubbles[cand.bubble_index]
            if b.dimension and b.dimension != "NO_DIMENSION":
                continue
            c = callout_groups[cand.callout_index]
            self._apply_assignment_candidate(
                b,
                c.text,
                float(max(0.72, min(0.95, cand.score / 140.0))),
                cand.reason,
            )
            trace = traces_by_index.get(cand.bubble_index)
            if trace is not None:
                trace["leader_first_status"] = "assigned"
                trace["leader_first_assigned_callout"] = cand.callout_index
            claimed_bubbles.add(cand.bubble_index)
            claimed_callouts.add(cand.callout_index)
            assigned += 1

        if assigned:
            logger.info("Leader-first candidate assignment claimed %d bubble(s)", assigned)

    @staticmethod
    def _apply_assignment_candidate(
        bubble: BubbleResult,
        text: str,
        confidence: float,
        reason: str,
        *,
        review_threshold: float = 0.86,
    ) -> None:
        """Apply one scored assignment through a single boundary."""
        bubble.dimension = text
        bubble.confidence = float(max(0.0, min(1.0, confidence)))
        bubble.needs_review = bubble.confidence < review_threshold
        bubble.review_reason = reason

    def _assign_using_traces(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
    ) -> None:
        """Assign dimensions to bubbles whose traced path reaches a callout box.

        Pass 0 (pre): global colour trace — follows the annotation colour of
          the bubble's leader line across the full image.  No direction needed.
          Most reliable when the drawing uses coloured annotations.
        Pass 1: path pixel enters a callout box (direction-based BFS).
        Pass 2: nearest callout to path endpoint (proximity fallback).

        Each callout is consumed by at most one bubble (first strong match wins).
        """
        if not callout_groups:
            return

        used_callout_indices: set = set()

        # ── Pass 0: global colour trace (direction-agnostic) ──────────
        if self.image is not None:
            for b in bubbles:
                if b.dimension and b.dimension != "NO_DIMENSION":
                    continue
                geom = BalloonGeometry(cx=float(b.x), cy=float(b.y), radius=float(b.radius))
                ci = color_trace_to_callout(
                    image=self.image,
                    balloon_geom=geom,
                    callouts=callout_groups,
                )
                if ci is not None and ci not in used_callout_indices:
                    self._apply_assignment_candidate(
                        b,
                        callout_groups[ci].text,
                        0.88,
                        "trace_color_assignment",
                    )
                    used_callout_indices.add(ci)

        if not self._seed_traces:
            return

        # Pass 1: path pixel enters a callout box (strong evidence).
        # Process bubbles whose trace directly enters a callout region first.
        for b_idx, b in enumerate(bubbles):
            if b.dimension and b.dimension != "NO_DIMENSION":
                continue
            trace_info = self._trace_for_bubble(b, b_idx)
            if not trace_info:
                continue
            path = trace_info.get("path")
            if not path:
                continue

            best_idx: Optional[int] = None
            best_step: int = 0
            for step, (px, py) in enumerate(path):
                for i, c in enumerate(callout_groups):
                    if i in used_callout_indices:
                        continue
                    if c.x1 <= px <= c.x2 and c.y1 <= py <= c.y2:
                        best_idx = i
                        best_step = step
                        break
                if best_idx is not None:
                    break

            if best_idx is not None:
                ctext = callout_groups[best_idx].text
                # Penalise very long traces: a real leader line should
                # reach its callout within a reasonable distance from the
                # bubble. Traces that wander > 6× the bubble radius
                # likely traversed unrelated geometry and hit a wrong
                # callout — mark for review and lower confidence.
                trace_distance = math.hypot(
                    path[best_step][0] - path[0][0],
                    path[best_step][1] - path[0][1],
                )
                max_reliable = b.radius * 6.0
                if self._score_text_quality(ctext) > -5:
                    self._apply_assignment_candidate(
                        b,
                        ctext,
                        0.90 if trace_distance <= max_reliable else 0.45,
                        "trace_box_assignment",
                    )
                    used_callout_indices.add(best_idx)

        # Pass 2: endpoint proximity fallback.
        # Build all (distance, bubble_idx, callout_idx) triples and assign
        # closest-first so the bubble with the nearest endpoint wins when two
        # traces end near the same callout (avoids first-in-list bias).
        proximity_claims: List[Tuple[float, int, int]] = []  # (dist, b_idx, ci)
        for b_idx, b in enumerate(bubbles):
            if b.dimension and b.dimension != "NO_DIMENSION":
                continue
            trace_info = self._trace_for_bubble(b, b_idx)
            if not trace_info:
                continue
            path = trace_info.get("path")
            if not path:
                continue
            end_x, end_y = path[-1]
            for i, c in enumerate(callout_groups):
                if i in used_callout_indices:
                    continue
                dx = max(c.x1 - end_x, 0.0, end_x - c.x2)
                dy = max(c.y1 - end_y, 0.0, end_y - c.y2)
                d = math.hypot(dx, dy)
                if d < 80.0:
                    proximity_claims.append((d, b_idx, i))

        proximity_claims.sort()  # closest endpoint wins
        for dist, b_idx, ci in proximity_claims:
            if ci in used_callout_indices:
                continue
            b = bubbles[b_idx]
            if b.dimension and b.dimension != "NO_DIMENSION":
                continue
            ctext = callout_groups[ci].text
            if self._score_text_quality(ctext) > -5:
                self._apply_assignment_candidate(
                    b,
                    ctext,
                    0.90,
                    "trace_endpoint_assignment",
                )
                used_callout_indices.add(ci)

    # ── Step 9: Global optimal assignment (Hungarian) ───────────────

    def _optimal_assign(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
        leader_directions: Dict[int, Tuple[float, float]],
        eff_assoc_dist: int,
        eff_assoc_cost: float = 360.0,
    ) -> None:
        """
        Assign dimensions using the Hungarian algorithm for globally
        optimal bubble-to-callout matching.

        The cost matrix uses three principled, normalised signals
        (each in [0, 1]) with no hand-tuned magic numbers:

          1. Normalised distance    — dist / image_diagonal
          2. Trace hit (binary)     — 0 if trace enters callout box, 1 if not
          3. Direction alignment    — (1 - cos(angle)) / 2

        The three signals are combined with equal weight.
        No text-quality bonuses, no special-case patterns.
        Text validation happens AFTER assignment, not during scoring.
        """
        from scipy.optimize import linear_sum_assignment

        if not bubbles or not callout_groups:
            return

        n_bubbles  = len(bubbles)
        n_callouts = len(callout_groups)
        h, w = (self.image.shape[:2] if self.image is not None
                else (1000, 1000))
        diag = math.hypot(w, h)

        INF = 1e9
        cost = np.full((n_bubbles, n_callouts), INF, dtype=np.float64)

        # Pre-compute trace hits using MULTIPLE trace algorithms:
        # 1. Original annotation-color BFS trace
        # 2. A* pathfinding with gap-closing (new)
        # 3. Ray-casting from endpoint (new)
        trace_hits: Dict[int, set] = {}   # bi → set of ci

        # Also run A* traces for bubbles where the original trace
        # didn't hit any callout box
        from trace_algorithms import astar_trace, ray_cast_to_callout
        astar_hits: Dict[int, set] = {}
        ray_hits: Dict[int, set] = {}

        callout_box_list = [(c.x1, c.y1, c.x2, c.y2) for c in callout_groups]
        callout_box_indexed = [(c.x1, c.y1, c.x2, c.y2, ci) for ci, c in enumerate(callout_groups)]

        if self._seed_traces and self.image is not None:
            gray_for_astar = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            for bi, b in enumerate(bubbles):
                info = self._trace_for_bubble(b, bi)
                if not info or not info.get("path"):
                    continue

                # Original trace hits — exact bbox entry.
                hits: set = set()
                for px, py in info["path"]:
                    for ci, c in enumerate(callout_groups):
                        if c.x1 <= px <= c.x2 and c.y1 <= py <= c.y2:
                            hits.add(ci)
                # Near-hits: real engineering drawings put a deliberate
                # gap between the leader-line arrowhead and the dim
                # text. The trace stops at the arrowhead but the
                # "physically correct" target sits a short distance
                # further along. We count a near-hit when:
                #
                #   - the trace endpoint is reasonably outside the
                #     balloon (i.e. an actual leader, not a rim loop)
                #   - a callout sits forward of the trace direction
                #     (cos_align ≥ 0.6 — narrow forward cone)
                #   - the gap from endpoint to callout bbox edge is
                #     within a generous-but-bounded radius
                #
                # The radius scales with how confident we are that this
                # is a real leader (longer paths + further endpoints
                # = stronger signal = wider near-hit acceptance).
                if not hits and len(info["path"]) >= 2:
                    ex, ey = info["path"][-1]
                    endpoint_dist = math.hypot(ex - float(b.x), ey - float(b.y))
                    path_len = len(info["path"])
                    # Trace direction (centre → endpoint).
                    tx = ex - float(b.x)
                    ty = ey - float(b.y)
                    tn = math.hypot(tx, ty)
                    if tn > 1e-6 and endpoint_dist > float(b.radius) * 1.0:
                        tx, ty = tx / tn, ty / tn
                        # Tier 1 — short gap, loose alignment.
                        near_r_close = max(40.0, float(b.radius) * 2.5)
                        # Tier 2 — wider gap, strict alignment, only
                        # for traces that genuinely escaped the balloon.
                        is_real_leader = (
                            path_len >= 30
                            and endpoint_dist > float(b.radius) * 2.0
                        )
                        near_r_far = (
                            max(80.0, float(b.radius) * 6.0)
                            if is_real_leader else 0.0
                        )
                        for ci, c in enumerate(callout_groups):
                            nx = max(c.x1, min(ex, c.x2))
                            ny = max(c.y1, min(ey, c.y2))
                            edge_dist = math.hypot(ex - nx, ey - ny)
                            dx = c.cx - ex
                            dy = c.cy - ey
                            dn = math.hypot(dx, dy)
                            cos_align = (tx * dx + ty * dy) / dn if dn > 1e-6 else 1.0
                            in_tier1 = edge_dist <= near_r_close and cos_align >= 0.3
                            in_tier2 = (
                                edge_dist <= near_r_far
                                and cos_align >= 0.6
                            )
                            if in_tier1 or in_tier2:
                                hits.add(ci)
                trace_hits[bi] = hits

                # A* trace: if original trace didn't hit any box,
                # try A* which can jump small gaps
                if not hits and info.get("seed"):
                    seed = info["seed"]
                    astar_path = astar_trace(
                        gray_for_astar,
                        start=(seed.contact_x, seed.contact_y),
                        direction=(seed.dir_x, seed.dir_y),
                        balloon_cx=float(b.x),
                        balloon_cy=float(b.y),
                        balloon_radius=float(b.radius),
                        callout_boxes=callout_box_list,
                        max_steps=300,
                    )
                    if astar_path:
                        a_hits: set = set()
                        for px, py in astar_path:
                            for ci, c in enumerate(callout_groups):
                                if c.x1 <= px <= c.x2 and c.y1 <= py <= c.y2:
                                    a_hits.add(ci)
                        astar_hits[bi] = a_hits

                # Ray-cast: project from trace endpoint along THREE
                # candidate directions and union the hits. Different
                # leaders produce different trajectory shapes, so
                # we try multiple direction estimates and use any
                # callout reached from any of them.
                #
                # 1) Last-segment direction (path[-2] → path[-1]):
                #    very local; best when the BFS extended far and
                #    the last step continues the leader faithfully.
                #
                # 2) Balloon-centre → endpoint:
                #    overall outward trajectory; best when the BFS
                #    stopped on a noisy step.
                #
                # 3) Path mid-point → endpoint (the path's later half):
                #    captures the leader direction when the trace
                #    started by walking around the rim and only later
                #    hooked outward — using the FULL centre-to-endpoint
                #    averages that initial rim-loop into the direction.
                if info.get("path"):
                    path = info["path"]
                    if len(path) >= 2:
                        ex, ey = path[-1]
                        hits_collected: set = set()

                        # 1) Last-segment direction
                        px, py = path[-2]
                        ray1_dx, ray1_dy = ex - px, ey - py
                        if math.hypot(ray1_dx, ray1_dy) > 0:
                            h1 = ray_cast_to_callout(
                                (ex, ey), (ray1_dx, ray1_dy),
                                callout_box_indexed,
                            )
                            if h1 is not None:
                                hits_collected.add(h1)

                        # 2) Centre → endpoint direction
                        ray2_dx = ex - float(b.x)
                        ray2_dy = ey - float(b.y)
                        if math.hypot(ray2_dx, ray2_dy) >= b.radius * 0.5:
                            h2 = ray_cast_to_callout(
                                (ex, ey), (ray2_dx, ray2_dy),
                                callout_box_indexed,
                            )
                            if h2 is not None:
                                hits_collected.add(h2)

                        # 3) Mid-of-path → endpoint direction
                        mid_idx = len(path) // 2
                        mx, my = path[mid_idx]
                        ray3_dx, ray3_dy = ex - mx, ey - my
                        if math.hypot(ray3_dx, ray3_dy) >= b.radius * 0.3:
                            h3 = ray_cast_to_callout(
                                (ex, ey), (ray3_dx, ray3_dy),
                                callout_box_indexed,
                            )
                            if h3 is not None:
                                hits_collected.add(h3)

                        if hits_collected:
                            ray_hits[bi] = hits_collected

        # Pre-compute: color trace result per bubble
        color_trace_map: Dict[int, Optional[int]] = {}
        if self.image is not None:
            from leader_seed_rules import (
                BalloonGeometry, color_trace_to_callout,
            )
            for bi, b in enumerate(bubbles):
                geom = BalloonGeometry(
                    cx=float(b.x), cy=float(b.y), radius=float(b.radius)
                )
                color_trace_map[bi] = color_trace_to_callout(
                    self.image, geom, callout_groups,
                )

        # Pre-compute: trace endpoint per bubble
        trace_endpoints: Dict[int, Tuple[int, int]] = {}
        trace_lengths: Dict[int, int] = {}
        if self._seed_traces:
            for bi, b in enumerate(bubbles):
                info = self._trace_for_bubble(b, bi)
                if info and info.get("path"):
                    path = info["path"]
                    trace_endpoints[bi] = path[-1]
                    trace_lengths[bi] = len(path)

        # Pre-compute: radial exit detection per bubble
        # Pure geometric algorithm — scans from circle center outward
        # to find where a dark line exits the balloon boundary.
        from leader_geometry import (
            find_leader_exits, find_nearest_callout_along_exit,
        )
        # Full geometric leader analysis per bubble:
        #   - Find exit directions (HoughLinesP crossing rim)
        #   - Follow leader line from exit (edge-based)
        #   - Check if path enters any callout box (intersection)
        #   - Search callouts in directional cone from exit
        exit_matches: Dict[int, set] = {}   # bi → set of ci from cone search

        if self.image is not None:
            gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            callout_positions = [(c.cx, c.cy, ci)
                                 for ci, c in enumerate(callout_groups)]
            for bi, b in enumerate(bubbles):
                bx, by, br = int(b.x), int(b.y), int(b.radius)
                exits = find_leader_exits(
                    gray_img, bx, by, br, max_exits=3,
                )
                # Take first exit that finds a callout
                best_ci: Optional[int] = None
                for ex in exits:
                    ci = find_nearest_callout_along_exit(
                        ex, callout_positions,
                        float(bx), float(by),
                    )
                    if ci is not None:
                        best_ci = ci
                        break

                # VALIDATION: only trust the cone match if the matched
                # callout is not absurdly far compared to the nearest.
                # A real leader doesn't skip over a nearby callout to
                # reach one 5× further away.
                if best_ci is not None:
                    cone_dist = math.hypot(
                        bx - callout_groups[best_ci].cx,
                        by - callout_groups[best_ci].cy,
                    )
                    nearest_dist = min(
                        math.hypot(bx - c.cx, by - c.cy)
                        for c in callout_groups
                        if re.search(r"\d", c.text)
                    ) if callout_groups else cone_dist
                    # If cone match is > 4× the nearest distance, it's
                    # likely a false exit following drawing geometry
                    if cone_dist > max(nearest_dist * 4, br * 8):
                        best_ci = None

                exit_matches[bi] = {best_ci} if best_ci is not None else set()

        # Pre-compute per-bubble: minimum edge-to-edge distance to any
        # callout with leader evidence. A non-leader callout that is
        # dramatically closer AND geometrically aligned with the bubble
        # (GDT-frame pattern: text sits directly beside the bubble, no
        # drawn leader line) should be allowed to compete.
        leader_min_dist: Dict[int, float] = {}
        for bi in range(len(bubbles)):
            leader_cis = (
                exit_matches.get(bi, set())
                | trace_hits.get(bi, set())
                | astar_hits.get(bi, set())
                | ray_hits.get(bi, set())
                | (
                    {color_trace_map[bi]}
                    if color_trace_map.get(bi) is not None
                    else set()
                )
            )
            if not leader_cis:
                continue
            bx_, by_ = float(bubbles[bi].x), float(bubbles[bi].y)
            dists = []
            for mi in leader_cis:
                if mi < 0 or mi >= len(callout_groups):
                    continue
                cm = callout_groups[mi]
                dx_ = max(cm.x1 - bx_, 0.0, bx_ - cm.x2)
                dy_ = max(cm.y1 - by_, 0.0, by_ - cm.y2)
                dists.append(math.hypot(dx_, dy_))
            if dists:
                leader_min_dist[bi] = min(dists)

        for bi, b in enumerate(bubbles):
            bx, by, br = float(b.x), float(b.y), float(b.radius)

            for ci, c in enumerate(callout_groups):
                if not self._is_assignable_callout(
                    c,
                    allow_reference_only=True,
                    allow_keyword_only=True,
                ):
                    continue

                # ══════════════════════════════════════════════════════
                # 5-SIGNAL DOMAIN SCORING
                #
                # Each signal is normalised to [0, 1] where 0 = best.
                # Weights reflect the PHYSICAL RELIABILITY hierarchy
                # of engineering drawings — NOT tuned to test images.
                # ══════════════════════════════════════════════════════

                # ── Distance ──────────────────────────────────────────
                dx_edge = max(c.x1 - bx, 0.0, bx - c.x2)
                dy_edge = max(c.y1 - by, 0.0, by - c.y2)
                dist = math.hypot(dx_edge, dy_edge)
                if dist > diag * 0.5:
                    continue

                # ── THREE SIMPLE SIGNALS ──────────────────────────────
                #
                # 1. LEADER EXIT CONE (0 or 1):
                #    Did the radial exit detector find a line from
                #    this balloon pointing toward this callout?
                #    Also counts: trace path entering callout box,
                #    A* path, ray-cast hit, or color trace.
                #
                # 2. CLOSE PROXIMITY (0 or 1):
                #    Is the callout within 2× balloon radius?
                #    If yes, this is almost certainly the right match
                #    regardless of leader lines.
                #
                # 3. NORMALISED DISTANCE (0 to ~0.5):
                #    How far is the callout from the balloon?
                #    Tiebreaker when signals 1 and 2 don't decide.

                # ── PATH A vs PATH B decision ─────────────────────────
                #
                # Check if this bubble has RELIABLE leader evidence
                # pointing to THIS specific callout.
                #
                # Leader evidence is reliable when:
                #   - Exit cone found this callout AND it's not
                #     absurdly far (validated above)
                #   - Trace path physically entered this callout box
                #   - Color trace matched this callout
                #
                leader_points_here = (
                    ci in exit_matches.get(bi, set())
                    or (bi in trace_hits and ci in trace_hits[bi])
                    or (bi in astar_hits and ci in astar_hits[bi])
                    or (bi in ray_hits and ci in ray_hits[bi])
                    or color_trace_map.get(bi) == ci
                )

                # Does this bubble have ANY leader match at all?
                has_any_leader = (
                    bool(exit_matches.get(bi))
                    or (bi in trace_hits and bool(trace_hits[bi]))
                    or (bi in astar_hits and bool(astar_hits[bi]))
                    or (bi in ray_hits and bool(ray_hits[bi]))
                    or color_trace_map.get(bi) is not None
                )

                norm_dist = dist / diag

                # Count how many independent trace algorithms agree that
                # this callout is the target.  A single algorithm can be
                # wrong (exit-cone misled by geometry, color trace on
                # noise).  Requiring 2+ algorithms to agree filters out
                # false leader evidence on images like edb where traces
                # operate on weak/absent annotation masks.
                evidence_count = sum([
                    ci in exit_matches.get(bi, set()),
                    bi in trace_hits and ci in trace_hits[bi],
                    bi in astar_hits and ci in astar_hits[bi],
                    bi in ray_hits and ci in ray_hits[bi],
                    color_trace_map.get(bi) == ci,
                ])
                strong_leader = evidence_count >= 2

                reference_local = (
                    self._is_reference_only_callout(c)
                    and dist <= max(float(br) * 1.25, 90.0)
                )
                allow_reference_only = reference_local or (
                    leader_points_here
                    and dist <= max(float(br) * 2.25, 180.0)
                )
                if not self._is_assignable_callout(
                    c,
                    allow_reference_only=allow_reference_only,
                    allow_keyword_only=True,
                ):
                    continue

                if reference_local:
                    cost[bi, ci] = norm_dist
                elif strong_leader:
                    # Multiple algorithms agree → high confidence
                    cost[bi, ci] = norm_dist
                elif leader_points_here:
                    # Single algorithm points here → moderate confidence
                    cost[bi, ci] = 0.5 + norm_dist
                elif has_any_leader:
                    # Leader exists but points elsewhere.
                    # GDT-frame override: if this callout is both MUCH
                    # closer than the leader-matched one AND sits in
                    # the bubble's horizontal or vertical band (|Δ| <
                    # 2× radius measured to the callout box edge), the
                    # leader evidence is likely a trace that wandered
                    # through part geometry. Let distance dominate.
                    leader_d = leader_min_dist.get(bi)
                    y_aligned = (c.y1 <= by + 2.0 * br) and (c.y2 >= by - 2.0 * br)
                    x_aligned = (c.x1 <= bx + 2.0 * br) and (c.x2 >= bx - 2.0 * br)
                    if (
                        leader_d is not None
                        and dist < leader_d * 0.5
                        and (y_aligned or x_aligned)
                    ):
                        cost[bi, ci] = 0.5 + norm_dist
                    else:
                        cost[bi, ci] = 5.0 + norm_dist
                else:
                    # No leader evidence → pure proximity
                    if dist < br * 2.0:
                        cost[bi, ci] = norm_dist
                    else:
                        cost[bi, ci] = 2.0 + norm_dist

        # ── Topology signal: connected-component bonus ────────────────
        # Use annotation-layer connected components to identify which
        # bubble-callout pairs share ink.  This is an ADDITIONAL signal
        # fed into the Hungarian cost matrix (not a locked assignment)
        # so the global solver can weigh topology against distance and
        # leader evidence.
        try:
            ann_mask = None
            if (hasattr(self, '_annotation_layer')
                    and self._annotation_layer is not None):
                ann_mask = self._annotation_layer.mask
            bubble_geoms = [(int(b.x), int(b.y), int(b.radius))
                            for b in bubbles]
            callout_boxes = [(c.x1, c.y1, c.x2, c.y2)
                             for c in callout_groups]
            skel_assignments = skeleton_assign(
                self.image, ann_mask,
                bubble_geoms, callout_boxes,
            )
            # Apply topology bonus: reduce cost for exclusive CC links
            TOPO_BONUS = 3.0
            for bi, ci in skel_assignments.items():
                if bi < n_bubbles and ci < n_callouts:
                    if cost[bi, ci] < INF * 0.5:
                        cost[bi, ci] = max(0.001, cost[bi, ci] - TOPO_BONUS)
        except Exception as e:
            logger.warning("Skeleton topology signal failed: %s", e)

        n_real_callouts = n_callouts
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = list(zip(row_ind, col_ind))

        # ── Non-crossing repair (FINAL AUTHORITY) ────────────────────
        # Engineering drawings do not have crossing leader lines.
        # Collect all crossing pairs, rank by distance improvement,
        # and apply the best non-cascading swaps.
        crossing_swaps: List[Tuple[float, int, int, int, int]] = []
        for ai in range(len(assignment)):
            bi, ci = assignment[ai]
            if ci >= n_real_callouts:
                continue
            for aj in range(ai + 1, len(assignment)):
                bj, cj = assignment[aj]
                if cj >= n_real_callouts:
                    continue
                if cost[bi, cj] >= INF * 0.5 or cost[bj, ci] >= INF * 0.5:
                    continue
                b_i, b_j = bubbles[bi], bubbles[bj]
                c_i, c_j = callout_groups[ci], callout_groups[cj]
                if not self._segments_cross(
                    b_i.x, b_i.y, c_i.cx, c_i.cy,
                    b_j.x, b_j.y, c_j.cx, c_j.cy,
                ):
                    continue  # not crossing
                if self._segments_cross(
                    b_i.x, b_i.y, c_j.cx, c_j.cy,
                    b_j.x, b_j.y, c_i.cx, c_i.cy,
                ):
                    continue  # swapped also crosses
                # Compute distance improvement (lower = better swap)
                cur_dist = (math.hypot(b_i.x - c_i.cx, b_i.y - c_i.cy)
                            + math.hypot(b_j.x - c_j.cx, b_j.y - c_j.cy))
                swap_dist = (math.hypot(b_i.x - c_j.cx, b_i.y - c_j.cy)
                             + math.hypot(b_j.x - c_i.cx, b_j.y - c_i.cy))
                cur_cost = float(cost[bi, ci] + cost[bj, cj])
                swap_cost = float(cost[bi, cj] + cost[bj, ci])
                if swap_cost > cur_cost + 2.5:
                    continue
                if swap_dist < cur_dist:
                    improvement = cur_dist - swap_dist
                    crossing_swaps.append((improvement, ai, aj, bi, bj))

        # Apply best swaps greedily (no bubble swapped twice)
        crossing_swaps.sort(reverse=True)  # best improvement first
        swapped_indices: Set[int] = set()
        swapped_bubble_ids: Set[int] = set()
        for _, ai, aj, bi, bj in crossing_swaps:
            if ai in swapped_indices or aj in swapped_indices:
                continue
            old_ci = assignment[ai][1]
            old_cj = assignment[aj][1]
            assignment[ai] = (bi, old_cj)
            assignment[aj] = (bj, old_ci)
            swapped_indices.add(ai)
            swapped_indices.add(aj)
            swapped_bubble_ids.add(bi)
            swapped_bubble_ids.add(bj)

        for bi, ci in assignment:
            if ci >= n_real_callouts:
                continue  # null assignment → leave as NO_DIMENSION
            b = bubbles[bi]
            c = callout_groups[ci]
            if cost[bi, ci] >= 2.25:
                continue
            dx_edge = max(c.x1 - b.x, 0.0, b.x - c.x2)
            dy_edge = max(c.y1 - b.y, 0.0, b.y - c.y2)
            bubble_edge_dist = math.hypot(dx_edge, dy_edge)
            endpoint_edge_dist = bubble_edge_dist
            if bi in trace_endpoints:
                ex, ey = trace_endpoints[bi]
                dx_end = max(c.x1 - ex, 0.0, ex - c.x2)
                dy_end = max(c.y1 - ey, 0.0, ey - c.y2)
                endpoint_edge_dist = math.hypot(dx_end, dy_end)
            physical_limit = max(180.0, float(b.radius) * 6.0)
            if min(bubble_edge_dist, endpoint_edge_dist) > physical_limit:
                continue
            if b.dimension and b.dimension != "NO_DIMENSION":
                continue
            if bi in swapped_bubble_ids:
                # Non-crossing repaired: give high confidence so rescue
                # doesn't overwrite with garbage from nearby OCR.
                self._apply_assignment_candidate(
                    b,
                    c.text,
                    0.75,
                    "non_crossing_repair",
                    review_threshold=0.75,
                )
            else:
                max_cost = 11.0
                # Cast through float() / bool() to avoid numpy scalar
                # types (np.float64, np.bool_) leaking into the
                # BubbleResult — these don't serialize through
                # JSONResponse and crash the /api/detect endpoint
                # with "Object of type bool_ is not JSON serializable".
                opt_conf = float(max(0.05, 1.0 - cost[bi, ci] / max_cost))
                self._apply_assignment_candidate(
                    b,
                    c.text,
                    opt_conf,
                    "optimal_assign",
                    review_threshold=0.50,
                )

    # ── Step 10: Fallback linker ───────────────────────────────────

    def _link_bubbles_to_callouts(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
        leader_segments,
        leader_directions: Dict[int, Tuple[float, float]],
        eff_assoc_dist: int,
        eff_assoc_cost: float,
    ) -> None:
        if not bubbles:
            return

        use_image = (
            self.cfg.enable_image_linking
            and len(bubbles) <= self.cfg.max_balloons_for_image_linking
        )

        # Only link unresolved bubbles
        unresolved = [
            (i, b) for i, b in enumerate(bubbles)
            if not (b.dimension or "").strip() or b.dimension == "NO_DIMENSION"
        ]
        if not unresolved:
            return

        orig_indices = [i for i, _ in unresolved]
        unres_bubbles = [b for _, b in unresolved]

        # Build leader_directions keyed by LOCAL index (0..len(unresolved))
        local_leader_dirs: Dict[int, Tuple[float, float]] = {
            local_i: leader_directions[orig_i]
            for local_i, orig_i in enumerate(orig_indices)
            if orig_i in leader_directions
        }

        balloon_nodes = [
            BalloonNode(b.bubble_number, float(b.x), float(b.y), float(b.radius))
            for b in unres_bubbles
        ]

        try:
            link_results = link_balloons_to_callouts(
                balloons=balloon_nodes,
                callouts=callout_groups,
                leader_segments=leader_segments,
                max_assoc_distance=float(eff_assoc_dist),
                max_accept_cost=float(eff_assoc_cost),
                leader_bonus_weight=float(self.cfg.leader_bonus_weight),
                image=(
                    self.image
                    if (use_image and self.cfg.enable_heavy_path_disambiguation)
                    else None
                ),
                grammar_tokens=self._norm_tokens,
                leader_directions=local_leader_dirs,
            )
        except Exception as e:
            logger.error("Linking failed: %s", e)
            return

        # Apply link results
        for local_i, res in enumerate(link_results):
            orig_i = orig_indices[local_i]
            b = bubbles[orig_i]
            if res.callout_index is not None and res.assigned_text:
                link_conf = 0.55 if res.needs_review else 0.72
                self._apply_assignment_candidate(
                    b,
                    res.assigned_text,
                    link_conf,
                    "linker_assign",
                    review_threshold=0.86 if res.needs_review else 0.50,
                )

        # Conflict resolution: enforce exclusive callout ownership
        # Build a combined result list covering ALL bubbles.
        # For already-resolved bubbles (assigned by step 9), look up their
        # callout_index by text so resolve_assignment_conflicts can detect
        # if the step-10 linker double-assigned the same callout.
        text_to_ci: Dict[str, int] = {
            cg.text: ci for ci, cg in enumerate(callout_groups)
        }
        all_results: List[LinkResult] = []
        local_map: Dict[int, int] = {orig_i: local_i for local_i, orig_i in enumerate(orig_indices)}
        for orig_i, b in enumerate(bubbles):
            if orig_i in local_map:
                all_results.append(link_results[local_map[orig_i]])
            else:
                # Already resolved bubble: carry through its callout_index
                # (score=0 means highest priority in conflict resolution)
                stub_ci = text_to_ci.get(b.dimension)
                all_results.append(LinkResult(orig_i, stub_ci, b.dimension, 0.0, False))

        all_results = resolve_assignment_conflicts(all_results, callout_groups)

        # Apply conflict resolution outcomes
        for orig_i, res in enumerate(all_results):
            if orig_i in local_map:
                b = bubbles[orig_i]
                if res.callout_index is None:
                    b.dimension    = "NO_DIMENSION"
                    b.needs_review = True

    # ── Targeted OCR at trace endpoints ──────────────────────────────

    def _likelihood_scan_bubbles(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
        image: np.ndarray,
        gray: np.ndarray,
        eff_assoc_dist: int,
    ) -> None:
        """
        Multi-evidence bubble likelihood scan.

        Slides a circular template across the annotation mask to find
        bubble-shaped structures that HoughCircles missed.  A candidate
        must have:
          - Annotation-tint pixels on the ring (≥20% coverage)
          - Clean interior (low edge density inside)
          - Not overlap any existing bubble

        Survivors get multi-preprocessing OCR (original, CLAHE, inverted).
        """
        if not bubbles:
            return
        if not hasattr(self, '_annotation_layer') or self._annotation_layer is None:
            return

        ann_mask = self._annotation_layer.mask
        if ann_mask is None or np.count_nonzero(ann_mask) < 50:
            return

        h, w = gray.shape[:2]
        radii = sorted(b.radius for b in bubbles)
        median_r = radii[len(radii) // 2]
        seen = {b.bubble_number for b in bubbles}
        existing_positions = [(b.x, b.y) for b in bubbles]

        # Scan step: check every grid position at ~radius intervals
        step = max(8, median_r // 2)
        candidates: List[Tuple[float, int, int]] = []
        full_edges = cv2.Canny(gray, 50, 150)

        for cy in range(median_r + 2, h - median_r - 2, step):
            for cx in range(median_r + 2, w - median_r - 2, step):
                # Quick reject: skip if no annotation pixels nearby
                local = ann_mask[max(0, cy - median_r):cy + median_r,
                                 max(0, cx - median_r):cx + median_r]
                if np.count_nonzero(local) < 8:
                    continue

                # Skip if too close to existing bubble
                if any(math.hypot(cx - bx, cy - by) < median_r * 1.5
                       for bx, by in existing_positions):
                    continue

                # Score: annotation tint on ring
                ann_count = 0
                total = 0
                for ang_deg in range(0, 360, 10):
                    ang = math.radians(ang_deg)
                    px = int(cx + median_r * math.cos(ang))
                    py = int(cy + median_r * math.sin(ang))
                    if 0 <= px < w and 0 <= py < h:
                        total += 1
                        if ann_mask[py, px] > 0:
                            ann_count += 1
                if total == 0:
                    continue
                ring_score = ann_count / total
                if ring_score < 0.20:
                    continue

                # Score: interior cleanliness (low edge density)
                crop_x1 = max(0, cx - median_r)
                crop_y1 = max(0, cy - median_r)
                crop_x2 = min(w, cx + median_r)
                crop_y2 = min(h, cy + median_r)
                edges = full_edges[crop_y1:crop_y2, crop_x1:crop_x2]
                interior_r = max(1, median_r - 4)
                mask_int = np.zeros_like(edges)
                cv2.circle(
                    mask_int,
                    (int(cx - crop_x1), int(cy - crop_y1)),
                    interior_r,
                    255,
                    -1,
                )
                edge_in_interior = np.count_nonzero(edges & mask_int)
                interior_area = max(1, np.count_nonzero(mask_int))
                edge_density = edge_in_interior / interior_area
                if edge_density > 0.15:
                    continue  # too much interior edge = drawing geometry

                # Combined score
                score = ring_score * (1.0 - edge_density)
                candidates.append((-score, cx, cy))

        # Sort by score (best first), try top candidates
        candidates.sort()
        # Cap tries aggressively — 20 × 3 variants = 60 OCR calls was
        # consuming ~1.5s per image.  10 tries with early exit is enough
        # in practice — high-scoring candidates tend to hit first.
        max_tries = min(10, len(candidates))
        found_this_pass = 0

        for _, cx, cy in candidates[:max_tries]:
            # Early exit: if we've already found 3 new bubbles on this
            # pass, remaining candidates are unlikely to be bubbles.
            if found_this_pass >= 3:
                break
            # Multi-preprocessing OCR: try original, CLAHE, inverted
            margin = max(8, int(median_r * 0.3))
            x1 = max(0, cx - median_r - margin)
            y1 = max(0, cy - median_r - margin)
            x2 = min(w, cx + median_r + margin)
            y2 = min(h, cy + median_r + margin)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            scale = 4 if median_r <= 26 else 3
            upscaled = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)

            # Single preprocessing variant (CLAHE) — previously tried 3
            # but empirically one good variant catches most real bubbles
            # and eliminating the fallback variants cuts OCR calls by 3×.
            gray_crop = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray_crop)

            variants = [cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)]

            best_result = None
            best_conf = 0.0
            for variant in variants:
                try:
                    result = self.ocr(variant)
                except Exception:
                    continue
                items = (result[0] if isinstance(result, tuple)
                         and len(result) >= 1 else (result or []))
                if not items:
                    continue
                for item in items:
                    try:
                        text_info = item[1]
                        text = str(text_info[0] if isinstance(text_info, (list, tuple))
                                   else text_info).strip()
                        conf = float(text_info[1] if isinstance(text_info, (list, tuple))
                                     and len(text_info) > 1 else 0.9)
                        norm = self._normalize_local_bubble_candidate(text)
                        if norm is None or norm in seen:
                            continue
                        # Text center must be near circle center
                        bbox = item[0]
                        bx = sum(p[0] for p in bbox) / len(bbox) / scale + x1
                        by = sum(p[1] for p in bbox) / len(bbox) / scale + y1
                        d = math.dist((bx, by), (cx, cy))
                        if d > median_r * 0.9:
                            continue
                        if conf > best_conf:
                            best_conf = conf
                            best_result = (norm, conf)
                    except Exception:
                        continue

            if best_result is None:
                continue
            norm_text, conf = best_result
            if conf < 0.6:
                continue

            # Dimension-collision check
            if self._recovered_id_collides_with_dimension(
                norm_text, cx, cy, median_r, self._norm_tokens, set()
            ):
                continue

            bubbles.append(BubbleResult(
                bubble_number=norm_text,
                x=cx, y=cy, radius=median_r,
                confidence=conf,
                dimension="NO_DIMENSION",
                needs_review=True,
                review_reason="likelihood_scan",
            ))
            seen.add(norm_text)
            existing_positions.append((cx, cy))
            found_this_pass += 1

    def _has_uncovered_annotation_circle_evidence(
        self,
        image: np.ndarray,
        bubbles: List[BubbleResult],
    ) -> bool:
        """Return True when color geometry suggests a missed balloon.

        This is a cheap preflight for the grid-based likelihood scan.  It does
        not promote detections; it only asks whether there are annotation-color
        blobs with plausible ring scale that are not already covered by known
        bubbles.
        """
        if image is None or image.size == 0 or not bubbles:
            return False
        try:
            mask = self._annotation_hsv_mask(image)
        except Exception:
            return True
        if int(np.count_nonzero(mask)) < 50:
            return False

        radii = sorted(max(6, int(round(float(b.radius or 0)))) for b in bubbles)
        median_r = radii[len(radii) // 2] if radii else 25
        min_area = max(18.0, median_r * median_r * 0.05)
        max_area = max(min_area + 1.0, median_r * median_r * 5.5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter <= 1.0:
                continue
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity < 0.42:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if (
                x <= median_r
                or y <= median_r
                or x + bw >= w - median_r
                or y + bh >= h - median_r
            ):
                continue
            span = max(bw, bh)
            if span < median_r * 0.70 or span > median_r * 4.0:
                continue
            if max(bw, bh) / max(1, min(bw, bh)) > 2.6:
                continue

            (cx_f, cy_f), radius_f = cv2.minEnclosingCircle(cnt)
            cx, cy = int(round(cx_f)), int(round(cy_f))
            rr = max(6, int(round(radius_f)))
            if rr < median_r * 0.45 or rr > median_r * 2.3:
                continue
            if any(
                math.hypot(cx - b.x, cy - b.y) < max(rr, b.radius) * 1.75
                for b in bubbles
            ):
                continue
            try:
                ring = float(self._annotation_ring_score(mask, cx, cy, rr))
                evidence = float(self._compute_bubble_evidence(image, cx, cy, rr))
            except Exception:
                continue
            if ring >= 0.018 and evidence >= 0.08:
                return True
        return False

    def _should_run_likelihood_scan(
        self,
        bubbles: List[BubbleResult],
        image: np.ndarray,
    ) -> bool:
        """Run expensive likelihood scan only when primary recovery is sparse."""
        if not bubbles:
            return True
        trusted = 0
        trust_mask = None
        try:
            trust_mask = self._annotation_hsv_mask(image)
        except Exception:
            trust_mask = None
        for b in bubbles:
            try:
                if self._has_trusted_colored_bubble_evidence(b):
                    trusted += 1
                    continue
                if trust_mask is not None:
                    ring = float(self._annotation_ring_score(
                        trust_mask, int(round(b.x)), int(round(b.y)),
                        max(6, int(round(float(b.radius or 0)))),
                    ))
                    evidence = float(self._compute_bubble_evidence(
                        image, int(round(b.x)), int(round(b.y)),
                        max(6, int(round(float(b.radius or 0)))),
                    ))
                    if ring >= 0.020 and evidence >= 0.10:
                        trusted += 1
            except Exception:
                continue
        if trusted < min(2, len(bubbles)):
            return True
        if len(bubbles) < 4:
            return self._has_uncovered_annotation_circle_evidence(image, bubbles)
        return trusted < 4

    def _should_run_annotation_ring_recovery(
        self,
        bubbles: List[BubbleResult],
        image: np.ndarray,
    ) -> bool:
        """Run ring-label OCR only when primary evidence suggests misses."""
        if len(bubbles) >= int(self.cfg.dense_leader_trace_bubble_count):
            return True
        return self._should_run_likelihood_scan(bubbles, image)

    def _reverse_discover_bubbles(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
        leader_segments,
        gray: np.ndarray,
        image: np.ndarray,
        eff_min_r: int, eff_max_r: int,
        eff_p2: int, eff_dist: int,
        use_pre_hungarian_filter: bool = False,
    ) -> None:
        """
        Reverse leader discovery: for each unassigned callout, search
        a wide area for annotation-tinted small circles (bubbles) that
        the global Hough pass missed.  Run targeted OCR inside each
        candidate and assign the callout's dimension if a valid bubble
        ID is read.

        This catches dense-region failures where Hough misses small bubbles
        but the dimension callout text was successfully OCR'd.
        """
        if not bubbles:
            return
        radii = sorted(b.radius for b in bubbles)
        median_r = radii[len(radii) // 2]
        seen = {b.bubble_number for b in bubbles}

        if use_pre_hungarian_filter:
            # Pre-trace mode: Hungarian hasn't run yet, so we can't use
            # "callout.text in assigned_dims". Instead, treat a callout
            # as "needing a bubble" when there's no existing bubble
            # within a reasonable search radius of the callout centre.
            search_distance = max(120.0, float(median_r) * 6.0)
            unassigned = [
                c for c in callout_groups
                if re.search(r"\d", c.text)
                and not any(
                    math.hypot(c.cx - b.x, c.cy - b.y) < search_distance
                    for b in bubbles
                )
            ]
        else:
            assigned_dims = {
                b.dimension for b in bubbles
                if b.dimension and b.dimension != "NO_DIMENSION"
            }
            unassigned = [
                c for c in callout_groups
                if c.text not in assigned_dims
                and re.search(r"\d", c.text)
            ]
        if not unassigned:
            return

        h, w = gray.shape[:2]
        nr_min = max(6, int(median_r * 0.7))
        nr_max = max(nr_min + 6, int(median_r * 1.4))
        # Search radius: how far from the callout to look for bubbles
        search_r = int(median_r * 12)

        ann_mask = None
        if hasattr(self, "_annotation_layer") and self._annotation_layer is not None:
            ann_mask = self._annotation_layer.mask

        def _annotation_context(cg: CalloutGroup) -> int:
            if ann_mask is None:
                return 0
            pad = int(max(24.0, median_r * 4.0))
            x1 = max(0, int(cg.x1) - pad)
            y1 = max(0, int(cg.y1) - pad)
            x2 = min(w, int(cg.x2) + pad)
            y2 = min(h, int(cg.y2) + pad)
            if x2 <= x1 or y2 <= y1:
                return 0
            return int(np.count_nonzero(ann_mask[y1:y2, x1:x2]))

        if ann_mask is not None and not self.cfg.enable_targeted_endpoint_ocr:
            min_context = max(6, int(median_r * 0.35))
            scored_unassigned = [
                (_annotation_context(c), c)
                for c in unassigned
            ]
            unassigned = [
                c for score, c in scored_unassigned
                if score >= min_context
            ]
            unassigned.sort(key=_annotation_context, reverse=True)
        else:
            unassigned = list(unassigned)
        if not self.cfg.enable_targeted_endpoint_ocr:
            unassigned = unassigned[:max(0, int(self.cfg.max_reverse_discovery_callouts))]

        ocr_attempts = 0
        max_reverse_ocr = (
            10**9 if self.cfg.enable_targeted_endpoint_ocr
            else self.cfg.max_reverse_discovery_ocr
        )

        for cg in unassigned:
            if ocr_attempts >= max_reverse_ocr:
                break
            cg_cx, cg_cy = int(cg.cx), int(cg.cy)

            # Scan a wide crop around the callout for small circles
            x1c = max(0, cg_cx - search_r)
            y1c = max(0, cg_cy - search_r)
            x2c = min(w, cg_cx + search_r)
            y2c = min(h, cg_cy + search_r)
            crop_gray = gray[y1c:y2c, x1c:x2c]
            if crop_gray.size == 0:
                continue

            blurred = cv2.GaussianBlur(crop_gray, (5, 5), 2)
            hough_result = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1.0, minDist=int(median_r * 1.2),
                param1=self.cfg.hough_param1,
                param2=max(12, eff_p2 - 6),
                minRadius=nr_min, maxRadius=nr_max,
            )
            if hough_result is None:
                continue

            # Rank candidates by distance to callout, prefer closer
            candidates = []
            for c in hough_result[0]:
                cx_abs = int(c[0]) + x1c
                cy_abs = int(c[1]) + y1c
                cr = int(c[2])
                d = math.dist((cx_abs, cy_abs), (cg_cx, cg_cy))
                # Skip if too close to the callout text itself
                if d < median_r * 1.5:
                    continue
                # Skip if overlaps an existing bubble
                if any(math.dist((cx_abs, cy_abs), (b.x, b.y)) < cr * 1.5 for b in bubbles):
                    continue
                # Must have annotation tint
                if not self._circle_has_annotation_tint(image, cx_abs, cy_abs, cr):
                    continue
                candidates.append((d, cx_abs, cy_abs, cr))

            candidates.sort()

            for d, cx, cy, cr in candidates[:3]:
                if ocr_attempts >= max_reverse_ocr:
                    break
                ocr_attempts += 1
                result = self._recover_bubble_from_circle(image, cx, cy, cr)
                if result is None:
                    continue
                norm_text, conf = result
                if norm_text in seen:
                    continue
                # Dimension-collision check (local)
                if self._recovered_id_collides_with_dimension(
                    norm_text, cx, cy, cr, self._norm_tokens, set()
                ):
                    continue
                # Global collision: reject if the recovered ID matches a
                # contiguous digit group inside any assigned dimension.
                # e.g. "60" from ".600" (digit group "600" contains "60").
                # Uses digit-group extraction to avoid false positives
                # like "22" in "52.2" where "522" substring-matches but
                # the actual digit groups are ["52", "2"].
                is_dim_fragment = False
                for eb in bubbles:
                    if eb.dimension and eb.dimension != "NO_DIMENSION":
                        digit_groups = re.findall(r"\d+", eb.dimension)
                        for dg in digit_groups:
                            if norm_text in dg and len(dg) > len(norm_text):
                                is_dim_fragment = True
                                break
                    if is_dim_fragment:
                        break
                if is_dim_fragment:
                    continue
                logger.info(
                    "Reverse-discovered bubble %r for callout %r at (%d,%d) r=%d",
                    norm_text, cg.text, cx, cy, cr,
                )
                # In pre-Hungarian mode we leave the dim unset so the
                # global Hungarian assigns it from a complete picture.
                # In post-Hungarian (legacy) mode the original behaviour
                # of immediate dim-assignment is preserved.
                bubbles.append(BubbleResult(
                    bubble_number=norm_text,
                    x=cx, y=cy, radius=cr,
                    confidence=conf,
                    dimension=(
                        "NO_DIMENSION" if use_pre_hungarian_filter else cg.text
                    ),
                    needs_review=True,
                    review_reason="reverse_leader_discovery",
                ))
                seen.add(norm_text)
                break  # assigned this callout

    def _targeted_endpoint_ocr(
        self,
        bubbles: List[BubbleResult],
        image: np.ndarray,
        callout_groups: List[CalloutGroup],
    ) -> None:
        """
        Micro-text OCR rescue for unresolved or very weak bubbles.

        Uses deterministic local RapidOCR passes over multiple crop
        positions and scales. This is safer than changing the global
        matcher because it only touches bubbles that are unresolved or
        already low-confidence.

        Performance guard: each request gets a hard budget on local
        OCR calls so a drawing with many unresolved bubbles doesn't
        blow up end-to-end latency. Cache + budget together keep a
        dense 12-bubble page under ~2s of rescue time.
        """
        # Budget for the whole request. Local rescue is intentionally
        # bounded, but it needs enough crops to cover multiple unresolved
        # blue/red balloons on one drawing.
        self._rescue_ocr_budget = max(0, int(self.cfg.max_targeted_endpoint_ocr))
        self._local_rescue_variant_budget = max(0, int(self.cfg.max_local_rescue_ocr_variants))
        h, w = image.shape[:2]
        bubble_ids_set = {bb.bubble_number for bb in bubbles}
        gray_for_exit = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Process bubbles most-likely-to-benefit first: NO_DIMENSION
        # before low-confidence. Once the budget is exhausted we bail
        # out without iterating further.
        def _rescue_priority(b):
            has_dim = bool(b.dimension) and b.dimension != "NO_DIMENSION"
            m = re.match(r"\d+", str(b.bubble_number or ""))
            numeric_id = int(m.group(0)) if m else 10**9
            return (has_dim, numeric_id, b.confidence)

        for b in sorted(bubbles, key=_rescue_priority):
            if self._rescue_ocr_budget <= 0:
                break
            has_dim = b.dimension and b.dimension != "NO_DIMENSION"

            # Broadened retry triggers (previously: needs_review AND
            # confidence < 0.35 only). The Hungarian-fallback path
            # ("optimal_assign") frequently produces confidently-wrong
            # assignments — the cost matrix saturates so every bubble
            # ends up at ~0.30 confidence whether the dim is right or
            # not. Retry whenever:
            #   - assignment came from the optimal-assign fallback
            #   - the assigned text is just 1-2 chars (likely OCR
            #     fragment promoted to a fake dimension)
            #   - the bubble is marked needs_review with conf < 0.55
            low_conf_assignment = bool(
                has_dim
                and b.needs_review
                and b.confidence < 0.55
            )
            suspicious_text = bool(
                has_dim
                and isinstance(b.dimension, str)
                and len(b.dimension.strip()) <= 2
            )
            implausible_text = bool(
                has_dim
                and isinstance(b.dimension, str)
                and not self._is_plausible_dimension(b.dimension)
            )
            optimal_assign_fallback = bool(
                has_dim
                and b.review_reason == "optimal_assign"
                and b.confidence < 0.55
                and implausible_text
            )
            low_conf_local_rescue = bool(
                has_dim
                and b.needs_review
                and b.confidence < 0.55
                and "local_rapidocr_rescue" in (getattr(b, "review_reason", "") or "")
            )
            should_retry = (
                not has_dim
                or (low_conf_assignment and implausible_text)
                or suspicious_text
                or optimal_assign_fallback
                or low_conf_local_rescue
            )
            if not should_retry:
                continue

            bx, by, br = int(b.x), int(b.y), int(b.radius)

            from leader_geometry import find_leader_exits
            exits = find_leader_exits(gray_for_exit, bx, by, br, max_exits=3)
            candidates, candidate_confidences = self._collect_local_dimension_candidates(
                image=image,
                bubble=b,
                exits=exits,
                bubble_ids_set=bubble_ids_set,
                callout_groups=callout_groups,
            )

            if not candidates:
                continue

            def _candidate_rank(item: Tuple[str, int]) -> Tuple[int, int, float, int]:
                text, votes = item
                has_decimal = 1 if re.search(r"\d+\.\d+", text) else 0
                conf = candidate_confidences.get(text, 0.0)
                return (votes, has_decimal, conf, len(text))

            best_text, best_votes = max(candidates.items(), key=_candidate_rank)
            best_confidence = candidate_confidences.get(best_text, 0.0)
            best_text = self._upgrade_integer_rescue_from_callouts(
                best_text, b, callout_groups,
            )
            merged_text = self._merge_local_rescue_prefix_with_existing_decimal(
                best_text,
                b.dimension if has_dim else "",
            )
            if merged_text != best_text:
                best_text = merged_text
                best_confidence = max(best_confidence, 0.80)
            best_confidence = max(
                best_confidence,
                candidate_confidences.get(best_text, 0.0),
            )
            single_decimal_rescue = bool(
                best_votes == 1
                and best_confidence <= 0.01
                and re.fullmatch(r"\d+\.\d+", best_text)
            )
            accept = (
                best_votes >= 2
                or (
                    best_votes == 1
                    and best_confidence > 0.85
                    and self._is_plausible_dimension(best_text)
                )
                or single_decimal_rescue
                or (
                    has_dim
                    and best_text != b.dimension
                    and best_confidence >= 0.80
                    and self._is_plausible_dimension(best_text)
                )
            )
            if accept:
                if has_dim and not low_conf_assignment and best_text == b.dimension:
                    continue
                # Don't assign a value that's already claimed by another
                # bubble — the rescue OCR may read a nearby bubble's
                # dimension (e.g. "4DIA" belonging to bubble 1 being
                # read for bubble 11 which sits nearby).
                already_used = any(
                    bb.dimension == best_text
                    and bb.bubble_number != b.bubble_number
                    for bb in bubbles
                    if bb.dimension and bb.dimension != "NO_DIMENSION"
                )
                if already_used:
                    # Try next-best candidate
                    for alt_text, alt_votes in candidates.most_common(5)[1:]:
                        if alt_votes < 1:
                            break
                        alt_taken = any(
                            bb.dimension == alt_text
                            and bb.bubble_number != b.bubble_number
                            for bb in bubbles
                            if bb.dimension and bb.dimension != "NO_DIMENSION"
                        )
                        if not alt_taken:
                            best_text = alt_text
                            best_confidence = candidate_confidences.get(alt_text, 0.0)
                            already_used = False
                            break
                if already_used:
                    continue
                self._apply_assignment_candidate(
                    b,
                    best_text,
                    max(0.70, best_confidence),
                    "local_rapidocr_rescue",
                    review_threshold=0.86,
                )

    def _upgrade_integer_rescue_from_callouts(
        self,
        text: str,
        bubble: BubbleResult,
        callout_groups: List[CalloutGroup],
    ) -> str:
        """Replace a local integer fragment with a nearby precise decimal."""
        if not re.fullmatch(r"\d{2,4}", text or ""):
            return text
        best_text = text
        best_dist = float("inf")
        prefix = re.escape(text)
        pattern = re.compile(rf"^{prefix}[\.,]\d+")
        for cg in callout_groups:
            ctext = (cg.text or "").strip()
            if not pattern.match(ctext):
                continue
            if not self._is_assignable_callout(cg, allow_reference_only=True):
                continue
            dx_edge = max(cg.x1 - bubble.x, 0.0, bubble.x - cg.x2)
            dy_edge = max(cg.y1 - bubble.y, 0.0, bubble.y - cg.y2)
            dist = math.hypot(dx_edge, dy_edge)
            if dist > max(240.0, float(bubble.radius) * 8.0):
                continue
            if dist < best_dist:
                best_dist = dist
                best_text = ctext
        return best_text

    @staticmethod
    def _merge_local_rescue_prefix_with_existing_decimal(
        rescue_text: str,
        existing_text: str,
    ) -> str:
        """Use local OCR prefix evidence to repair a weak decimal read."""
        rescue = re.sub(r"\s+", "", (rescue_text or "").strip().upper())
        existing = re.sub(r"\s+", "", (existing_text or "").strip().upper())
        if not rescue or not existing:
            return rescue_text

        rescue_m = re.fullmatch(r"([RØΦ]?)(\d{2,4})", rescue)
        existing_m = re.fullmatch(r"([RØΦ]?)(\d+)[\.,](\d+)(.*)", existing)
        if not rescue_m or not existing_m:
            return rescue_text

        rescue_prefix, rescue_digits = rescue_m.groups()
        existing_prefix, existing_digits, frac, suffix = existing_m.groups()
        if (rescue_prefix or "") != (existing_prefix or ""):
            return rescue_text
        if len(rescue_digits) != len(existing_digits):
            return rescue_text
        if rescue_digits == existing_digits:
            return rescue_text
        prefix = rescue_prefix or ""
        return f"{prefix}{rescue_digits}.{frac}{suffix}"

    def _refine_low_conf_local_rescue_decimals(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
    ) -> None:
        """Repair weak decimal assignments using local OCR prefix evidence."""
        if self.image is None or not bubbles:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        bubble_ids_set = {str(bb.bubble_number) for bb in bubbles}

        old_budget = getattr(self, "_rescue_ocr_budget", 0)
        old_variant_budget = getattr(self, "_local_rescue_variant_budget", 0)
        self._rescue_ocr_budget = max(old_budget, 4)
        self._local_rescue_variant_budget = max(old_variant_budget, 24)
        try:
            from leader_geometry import find_leader_exits
            for b in bubbles:
                dim = (b.dimension or "").strip()
                reason = getattr(b, "review_reason", "") or ""
                if not dim or dim == "NO_DIMENSION":
                    continue
                if "local_rapidocr_rescue" not in reason:
                    continue
                if not b.needs_review or float(getattr(b, "confidence", 0.0) or 0.0) >= 0.55:
                    continue
                if not re.search(r"\d+[.,]\d+", dim):
                    continue

                exits = find_leader_exits(
                    gray,
                    int(round(float(b.x))),
                    int(round(float(b.y))),
                    max(1, int(round(float(b.radius or 1)))),
                    max_exits=3,
                )
                candidates, candidate_confidences = self._collect_local_dimension_candidates(
                    image=self.image,
                    bubble=b,
                    exits=exits,
                    bubble_ids_set=bubble_ids_set,
                    callout_groups=callout_groups,
                )
                if not candidates:
                    continue
                for text, votes in candidates.most_common(8):
                    repaired = self._merge_local_rescue_prefix_with_existing_decimal(
                        text,
                        dim,
                    )
                    if repaired == text or repaired == dim:
                        continue
                    conf = float(candidate_confidences.get(text, 0.0) or 0.0)
                    if conf < 0.78 and votes < 2:
                        continue
                    if not self._is_plausible_dimension(repaired):
                        continue
                    self._apply_assignment_candidate(
                        b,
                        repaired,
                        max(float(getattr(b, "confidence", 0.0) or 0.0), conf, 0.72),
                        "local_rescue_prefix_repair",
                        review_threshold=0.86,
                    )
                    logger.info(
                        "Local rescue prefix repair: bubble #%s %r -> %r",
                        b.bubble_number,
                        dim,
                        repaired,
                    )
                    break
        finally:
            self._rescue_ocr_budget = old_budget
            self._local_rescue_variant_budget = old_variant_budget

    def _collect_local_dimension_candidates(
        self,
        image: np.ndarray,
        bubble: BubbleResult,
        exits,
        bubble_ids_set: set[str],
        callout_groups: List[CalloutGroup],
    ):
        """Collect local RapidOCR dimension candidates from nearby crops.

        Performance: caches OCR results by crop rectangle on the
        detector instance, so overlapping rescue calls across
        multiple bubbles don't re-OCR the same region. Without this
        cache the rescue pass runs 60-80 OCR calls (~5s) per image;
        with it, ~15-20 unique crops per image (~1.5s).
        """
        from collections import Counter as _Counter
        from ocr_rules import normalize_engineering_text

        # Per-run crop-result cache lives on self
        if not hasattr(self, "_rescue_crop_cache"):
            self._rescue_crop_cache = {}

        h, w = image.shape[:2]
        bx, by, br = int(bubble.x), int(bubble.y), max(1, int(bubble.radius))

        centers: List[Tuple[int, int]] = []

        # Top priority: the actual traced leader endpoint, if the
        # leader trace step produced one. This is the most reliable
        # crop location because it's where the leader line physically
        # terminates near a dimension — not a guess from direction.
        trace_info = self._trace_for_bubble(bubble)
        if trace_info and trace_info.get("path"):
            path = trace_info["path"]
            if path:
                end_x, end_y = path[-1]
                centers.append((int(end_x), int(end_y)))

        centers.append((bx, by))
        ring_dist = max(int(br * 2.25), 35)
        for ang_deg in range(0, 360, 90):
            ang = math.radians(ang_deg)
            centers.append((
                bx + int(ring_dist * math.cos(ang)),
                by + int(ring_dist * math.sin(ang)),
            ))

        for ex in exits or []:
            for mult in (2.2, 3.0):
                centers.append((
                    bx + int(br * mult * ex.direction_x),
                    by + int(br * mult * ex.direction_y),
                ))

        nearby_callouts = []
        for cg in callout_groups:
            if not self._is_plausible_dimension(cg.text):
                continue
            dx_edge = max(cg.x1 - bx, 0.0, bx - cg.x2)
            dy_edge = max(cg.y1 - by, 0.0, by - cg.y2)
            dist = math.hypot(dx_edge, dy_edge)
            nearby_callouts.append((dist, int(cg.cx), int(cg.cy)))
        nearby_callouts.sort(key=lambda item: item[0])
        for _, cx, cy in nearby_callouts[:2]:
            centers.append((cx, cy))

        seen_centers = set()
        unique_centers: List[Tuple[int, int]] = []
        for cx, cy in centers:
            cx = min(max(cx, 0), w - 1)
            cy = min(max(cy, 0), h - 1)
            # Dedup centres at a coarser 30-pixel grid so nearby
            # candidates share the same crop region.
            key = (int(round(cx / 30.0)), int(round(cy / 30.0)))
            if key in seen_centers:
                continue
            seen_centers.add(key)
            unique_centers.append((cx, cy))

        candidates: _Counter = _Counter()
        candidate_confidences: Dict[str, float] = {}
        scales = (4,)

        for cx, cy in unique_centers:
            pad = max(70, int(br * 2.2))
            x1 = max(0, cx - pad)
            y1 = max(0, cy - pad)
            x2 = min(w, cx + pad)
            y2 = min(h, cy + pad)
            if x2 <= x1 or y2 <= y1:
                continue

            for scale in scales:
                # Cache key: rounded crop rectangle + scale. Two
                # rescue candidates that produce the same crop share
                # the same cached OCR result.
                ck = (
                    int(round(x1 / 20.0)) * 20,
                    int(round(y1 / 20.0)) * 20,
                    int(round(x2 / 20.0)) * 20,
                    int(round(y2 / 20.0)) * 20,
                    scale,
                )
                cached = self._rescue_crop_cache.get(ck)
                if cached is None:
                    # Respect the per-request budget so a dense image
                    # can't explode to 10+ seconds of rescue OCR.
                    if getattr(self, "_rescue_ocr_budget", 0) <= 0:
                        self._rescue_crop_cache[ck] = []
                        continue
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        self._rescue_crop_cache[ck] = []
                        continue
                    try:
                        cached = self._run_ocr(crop, scale, single_scale=True)
                    except Exception:
                        cached = []
                    self._rescue_crop_cache[ck] = cached
                    self._rescue_ocr_budget -= 1

                crop_added = False
                for r in cached:
                    norm = normalize_engineering_text(r.text)
                    recovered_norms = self._local_rescue_text_candidates(norm)
                    if norm and norm not in recovered_norms:
                        recovered_norms.insert(0, norm)
                    for norm in recovered_norms:
                        if not norm or not re.search(r"\d", norm):
                            continue
                        if not self._is_plausible_dimension(norm):
                            continue
                        if norm == bubble.bubble_number:
                            continue
                        if norm in bubble_ids_set:
                            continue
                        candidates[norm] += 1
                        crop_added = True
                        candidate_confidences[norm] = max(
                            candidate_confidences.get(norm, 0.0),
                            float(getattr(r, "conf", 0.0)),
                        )

                # RapidOCR often returns confidence 0.0 on tiny local crops
                # even when it reads the text. Run a tiny geometry-scoped
                # last-resort pass when this crop found no candidates. Other
                # crops may see unrelated nearby dimensions; that should not
                # suppress rotated OCR at the actual leader/bubble crop.
                crop = image[y1:y2, x1:x2]
                if crop.size and not crop_added:
                    for raw_text, raw_conf in self._local_rescue_ocr_texts(crop):
                        norm0 = normalize_engineering_text(raw_text)
                        for norm in self._local_rescue_text_candidates(norm0):
                            if not norm or not re.search(r"\d", norm):
                                continue
                            if not self._is_plausible_dimension(norm):
                                continue
                            if norm == bubble.bubble_number or norm in bubble_ids_set:
                                continue
                            candidates[norm] += 1
                            crop_added = True
                            candidate_confidences[norm] = max(
                                candidate_confidences.get(norm, 0.0),
                                float(raw_conf),
                            )

        return candidates, candidate_confidences

    def _local_rescue_ocr_texts(self, crop: np.ndarray) -> List[Tuple[str, float]]:
        """Return raw OCR texts from local crop variants, accepting conf=0."""
        if crop.size == 0:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(gray)
        variants = [cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)]

        out: List[Tuple[str, float]] = []
        for variant in variants:
            for rot in (0, 90):
                img_v = variant
                if rot == 90:
                    img_v = cv2.rotate(variant, cv2.ROTATE_90_CLOCKWISE)
                for scale in (8,):
                    if getattr(self, "_local_rescue_variant_budget", 0) <= 0:
                        return out
                    try:
                        up = cv2.resize(
                            img_v, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                        result = self.ocr(up)
                        self._local_rescue_variant_budget -= 1
                    except Exception:
                        self._local_rescue_variant_budget -= 1
                        continue
                    items = result[0] if (
                        isinstance(result, tuple) and len(result) >= 1
                    ) else (result or [])
                    for item in items or []:
                        try:
                            text_info = item[1]
                            if isinstance(text_info, (list, tuple)):
                                text = str(text_info[0]).strip()
                                conf = float(text_info[1]) if len(text_info) > 1 else 0.0
                            else:
                                text = str(text_info).strip()
                                conf = float(item[2]) if len(item) > 2 else 0.0
                        except Exception:
                            continue
                        if text:
                            out.append((text, conf))
        return out

    @staticmethod
    def _local_rescue_text_candidates(text: str) -> List[str]:
        """Generate dimension-shaped candidates from noisy local OCR text."""
        t = (text or "").strip().upper()
        if not t:
            return []
        t = t.replace("O", "0")
        t = re.sub(r"[，,]", ".", t)
        cleaned = re.sub(r"[^0-9A-ZØΦ°±+./×X-]", "", t)
        if not cleaned:
            return []

        digits = re.sub(r"[^0-9]", "", cleaned)
        bare_digits = bool(re.fullmatch(r"\d+", cleaned))
        candidates: List[str] = []
        if not (bare_digits and len(digits) >= 4 and not digits.startswith("0")):
            candidates.append(cleaned)
        if "." not in cleaned and digits:
            prefix = ""
            if cleaned[:1] in {"R", "Ã˜", "Ø", "Î¦", "Φ"}:
                prefix = cleaned[:1]
            if len(digits) == 4:
                if prefix:
                    candidates.append(f"{prefix}{digits[:2]}.{digits[2:]}")
                candidates.append(f"{digits[:2]}.{digits[2:]}")
                candidates.append(f"{digits[:3]}.{digits[3]}")
            elif len(digits) == 5:
                if prefix:
                    candidates.append(f"{prefix}{digits[:3]}.{digits[3:]}")
                candidates.append(f"{digits[:3]}.{digits[3:]}")

        normalized: List[str] = []
        seen: set = set()
        for cand in candidates:
            cand = cand.strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)
            normalized.append(cand)
        return normalized

    # ── Post-processing: smart assign remaining NO_DIMENSION ────────

    def _smart_assign_unresolved(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
        eff_assoc_dist: int,
    ) -> None:
        """
        Last-resort pass: nearest unused callout for still-unresolved bubbles.
        Always marks assignments as needs_review=True.
        """
        if not callout_groups:
            return

        def _text_key(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").strip()).upper()

        claimed_callouts: set = set()
        for existing in bubbles:
            if not existing.dimension or existing.dimension == "NO_DIMENSION":
                continue
            existing_key = _text_key(existing.dimension)
            best_idx = None
            best_dist = float("inf")
            for ci, c in enumerate(callout_groups):
                if _text_key(c.text) != existing_key:
                    continue
                dx_edge = max(c.x1 - existing.x, 0.0, existing.x - c.x2)
                dy_edge = max(c.y1 - existing.y, 0.0, existing.y - c.y2)
                dist = math.hypot(dx_edge, dy_edge)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = ci
            if best_idx is not None and best_dist <= eff_assoc_dist * 1.25:
                claimed_callouts.add(best_idx)

        for b in bubbles:
            if b.dimension and b.dimension != "NO_DIMENSION":
                continue

            best_callout = None
            best_idx = None
            best_dist = float("inf")

            for ci, c in enumerate(callout_groups):
                if ci in claimed_callouts:
                    continue
                if not self._is_assignable_callout(c, allow_reference_only=False):
                    continue
                quality = self._score_text_quality(c.text)
                if quality <= -5:
                    continue
                dx_edge = max(c.x1 - b.x, 0.0, b.x - c.x2)
                dy_edge = max(c.y1 - b.y, 0.0, b.y - c.y2)
                dist = math.hypot(dx_edge, dy_edge)
                if dist > eff_assoc_dist:
                    continue
                conservative_limit = max(140.0, float(b.radius) * 5.0)
                if dist > conservative_limit:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_callout = c
                    best_idx = ci

            if best_callout is not None and best_dist < eff_assoc_dist:
                confidence = max(0.35, min(0.62, 0.62 - best_dist / 500.0))
                self._apply_assignment_candidate(
                    b,
                    best_callout.text,
                    confidence,
                    "smart_assign_fallback",
                    review_threshold=0.86,
                )
                if best_idx is not None:
                    claimed_callouts.add(best_idx)

    @staticmethod
    def _is_valid_dimension_text(text: str) -> bool:
        """Return True if text could be a valid engineering dimension.

        A valid dimension must contain at least one digit.
        """
        if not text or not text.strip():
            return False
        return bool(re.search(r"\d", text))

    @staticmethod
    def _is_plausible_dimension(text: str) -> bool:
        """Return True for known engineering-value syntax used in fallback OCR."""
        if not text or not text.strip():
            return False
        t = text.strip().upper()
        if re.fullmatch(r"(?:\u00b1|[+-])\s*\d+(?:\.\d+)?", t):
            return False
        patterns = (
            r"^\d+\.\d+$",
            r"^\.\d+$",
            r"^\d+(?:\.\d+)?(?:Â°|°)?\s*\(?REF\)?(?:\s*/\s*\d+(?:\.\d+)?(?:Â°|°)?\s*\(?REF\)?)?$",
            r"^[Ã˜Ø]\s*\d+(?:\.\d+)?(?:\s+\w+)?$",
            r"^(?:R\d+(?:\.\d+)?|\d+(?:\.\d+)?R)(?:\s+TYP)?$",
            r"^MJ?\d+[XxÃ—]\d+(?:\.\d+)?(?:\s+\d+[Hh]\d+[Hh])?$",
            r"^\d+(?:\.\d+)?[Â±/]\d+(?:\.\d+)?$",
            r"^\d+(?:\.\d+)?(?:[XxÃ—]\d+(?:\.\d+)?)?\s*[Â°]$",
        )
        return any(re.fullmatch(pattern, t) for pattern in patterns) or is_dimension_token(t)

    @staticmethod
    def _segments_cross(
        ax: float, ay: float, bx: float, by: float,
        cx: float, cy: float, dx: float, dy: float,
    ) -> bool:
        """Return True if segment AB crosses segment CD."""
        def ccw(px, py, qx, qy, rx, ry):
            return (qx - px) * (ry - py) - (qy - py) * (rx - px)
        d1 = ccw(ax, ay, bx, by, cx, cy)
        d2 = ccw(ax, ay, bx, by, dx, dy)
        d3 = ccw(cx, cy, dx, dy, ax, ay)
        d4 = ccw(cx, cy, dx, dy, bx, by)
        if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
            return True
        return False

    @staticmethod
    def _score_text_quality(text: str) -> float:
        """Score how likely text is a real engineering dimension.

        DOMAIN KNOWLEDGE (not image-specific):
        - Engineering symbols (Ø, ±, °) indicate real dimensions
        - Thread specs (MJ5x0.8), tolerance classes (4h6h) are structured
        - Very short or very long text is less likely to be a dimension
        - Must contain at least one digit

        Returns a score where higher = more likely a real dimension.
        """
        t = text.strip()
        if not re.search(r"\d", t):
            return -100.0
        bonus = 0.0
        if "Ø" in t: bonus += 15.0
        if "±" in t: bonus += 15.0
        if "°" in t: bonus += 8.0
        if re.search(r"\d/\d", t): bonus += 12.0
        if re.search(r"MJ?\d+[xX]\d+", t, re.I): bonus += 20.0
        if re.search(r"\d+[hH]\d+[hH]", t): bonus += 15.0
        if re.search(r"R\d|\d+R", t, re.I): bonus += 10.0
        if re.search(r"\d+\.\d+", t): bonus += 5.0
        if len(t) < 2: bonus -= 20.0
        word_count = len(re.findall(r"[A-Za-z]+", t))
        numeric_groups = len(re.findall(r"\d+(?:[.,]\d+)?", t))
        has_engineering_phrase = bool(
            re.search(
                r"\b(?:THRU|TYP|CHAMFER|PROFILE|DATUM|BOSS|RIBS|THREAD|DIA|MMC|"
                r"ALL\s+AROUND|OUTER|INNER|SURFACE)\b",
                t,
                re.I,
            )
        )
        if has_engineering_phrase:
            bonus += 12.0
        if len(t) > 25:
            bonus -= 10.0
        if len(t) > 45:
            bonus -= 20.0
        if word_count >= 6 and not has_engineering_phrase:
            bonus -= 35.0
        if numeric_groups >= 4 and not has_engineering_phrase:
            bonus -= 20.0
        if re.search(
            r"\b(?:NOTES?|UNLESS\w*|OTHERWISE|SPECIF\w*|DIMEN\w*|REMOVE|SHARP|REVISION|HISTORY|DATE|"
            r"APPROVED|MATERIAL|FINISH|SURFACE\s+FINISH|TOLERANCE\s+CHART|"
            r"HOLE\s+DIAMETERS?|SHAFT\s+DIAMETERS?|DRAWN|CHECKED|SHEET|TITLE|PART\s+NO)\b",
            t,
            re.I,
        ):
            bonus -= 35.0
        if re.fullmatch(r"[A-Z]{2,}[- ]?\d{3,}[A-Z0-9-]*", t, re.I):
            bonus -= 35.0
        return bonus

    # ── Post-processing: shared dimension propagation ───────────────

    @staticmethod
    def _is_coordinate_callout(group: CalloutGroup) -> bool:
        text = (group.text or "").strip().upper()
        if group.callout_type == "coordinate":
            return True
        if re.search(r"(?:^|[^A-Z])[+-]\s*[XY](?:[^A-Z]|$)", text):
            return True
        return any(
            getattr(tok, "semantic_type", "") == "coordinate"
            for tok in getattr(group, "tokens", []) or []
        )

    @staticmethod
    def _is_reference_only_callout(group: CalloutGroup) -> bool:
        text = (group.text or "").strip().upper()
        if "REF" not in text:
            return False
        if re.search(r"[ØΦ]|(?:^|\b)R\s*\d|(?:^|\b)M\d|DIA|THRU|TYP|[±/]", text):
            parts = [p.strip() for p in re.split(r"\s*/\s*", text) if p.strip()]
            if len(parts) <= 1:
                return False
            cleaned_parts = [re.sub(r"[^A-Z0-9.°]", "", p) for p in parts]
            return all(
                re.fullmatch(r"[A-Z]?\d+(?:\.\d+)?(?:°)?REF", p)
                for p in cleaned_parts
            )
        cleaned = re.sub(r"[^A-Z0-9.°]", "", text)
        return bool(re.fullmatch(r"[A-Z]?\d+(?:\.\d+)?(?:°)?REF", cleaned))

    @staticmethod
    def _is_weak_bare_integer_callout(group: CalloutGroup) -> bool:
        text = (group.text or "").strip()
        if not re.fullmatch(r"\d{1,2}", text):
            return False
        tokens = getattr(group, "tokens", []) or []
        if len(tokens) != 1:
            return False
        raw = (getattr(tokens[0], "raw_text", "") or "").strip()
        if not raw:
            return False
        # OCR fragments such as "14-" are not standalone dimensions; they
        # usually come from chopped vertical text or a nearby bubble label.
        # Clean integer reads like "29" remain assignable.
        return bool(re.search(r"[^0-9\s]", raw))

    def _is_assignable_callout(
        self,
        group: CalloutGroup,
        *,
        allow_reference_only: bool = False,
        allow_keyword_only: bool = False,
    ) -> bool:
        text = (group.text or "").strip()
        if not text:
            return False
        if not re.search(r"\d", text):
            if not allow_keyword_only or group.callout_type != "keyword":
                return False
            if self._is_noise_token(text):
                return False
            if not re.search(r"[A-Z]{3,}", text, re.IGNORECASE):
                return False
            return True
        if self._is_noise_token(text):
            return False
        if self._is_coordinate_callout(group):
            return False
        if self._is_reference_only_callout(group) and not allow_reference_only:
            return False
        if self._is_weak_bare_integer_callout(group):
            return False
        if self._score_text_quality(text) <= -25:
            return False
        return True

    def _propagate_shared_dimensions(self, bubbles: List[BubbleResult]) -> None:
        """
        Propagate dimension values between bubbles that share a leader
        line, trace endpoint, or have related IDs.

        Engineering drawing conventions handled:
          - Suffix IDs:  bubble "11A" is related to "11".
          - Shared trace endpoints: two bubbles whose traced paths
            end near the same region likely reference the same dimension.
          - Connected annotation lines: bubbles on the same annotation
            line (detected via pixel connectivity from their seeds)
            share the dimension.

        Fills in bubbles that still lack a dimension AND overrides
        weak fallback assignments when a better donor is available.
        """
        if not bubbles:
            return

        def _is_weak(b: BubbleResult) -> bool:
            """True if the bubble has no dimension or a low-quality fallback."""
            if not b.dimension or b.dimension == "NO_DIMENSION":
                return True
            reason = getattr(b, "review_reason", "")
            if reason == "smart_assign_fallback":
                return True
            # Also treat low-confidence trace assignments as weak
            if b.needs_review and b.confidence < 0.50:
                return True
            return False

        resolved   = {b.bubble_number: b for b in bubbles
                      if b.dimension and b.dimension != "NO_DIMENSION"
                      and not _is_weak(b)}
        upgradeable = [b for b in bubbles if _is_weak(b)]
        bubble_index_by_obj = {id(b): idx for idx, b in enumerate(bubbles)}

        def _trace_endpoint_for(b: BubbleResult) -> Tuple[float, float]:
            idx = bubble_index_by_obj.get(id(b))
            trace = self._trace_for_bubble(b, idx)
            path = (trace or {}).get("path") or []
            if path:
                ex, ey = path[-1]
                return (float(ex), float(ey))
            return (float(b.x), float(b.y))

        # ── Strategy 1: suffix-based ID matching ─────────────────────
        # Engineering convention: a lettered suffix balloon (e.g. "11A")
        # often references the same base feature as "11". Use that as a
        # repair only for weak/empty suffix assignments; do not overwrite a
        # strong independently assigned suffix dimension.
        for b in list(bubbles):
            base_id = re.sub(r'[A-Za-z]+$', '', b.bubble_number)
            if not base_id or base_id == b.bubble_number:
                continue
            if not _is_weak(b):
                continue
            if base_id not in resolved:
                continue
            donor = resolved[base_id]
            if b.dimension == donor.dimension:
                continue  # already inherited / coincidentally identical

            # Save the suffix's old dim so we can transfer it to whoever's leader
            # endpoint is actually closest to it — that bubble is the
            # rightful owner (Hungarian got confused by the suffix competing for
            # the same callout).
            old_dim = b.dimension if (b.dimension and b.dimension != "NO_DIMENSION") else None

            self._apply_assignment_candidate(
                b,
                donor.dimension,
                float(donor.confidence) * 0.92,
                f"shared_dim_suffix_{base_id}",
                review_threshold=0.99,
            )
            resolved[b.bubble_number] = b
            if b in upgradeable:
                upgradeable.remove(b)

            # Transfer the freed dim to the geographically-closest
            # dimensionless (or weak) sibling. Don't transfer to b's
            # base balloon (it already has the canonical dim) or to a
            # bubble with a strong existing dim.
            if old_dim:
                best_recipient: Optional[BubbleResult] = None
                best_dist = float("inf")
                # The "freed" dim was geographically near b — that's
                # where Hungarian originally placed it. Use b's leader
                # endpoint (or b's center) as the anchor for "where did
                # this dim live".
                anchor = _trace_endpoint_for(b)
                for other in bubbles:
                    if other is b or other.bubble_number == base_id:
                        continue
                    # Only consider weak/empty recipients
                    if other.dimension and other.dimension != "NO_DIMENSION":
                        # Allow override only if other's dim is a strict
                        # substring of the freed dim — a partial / fragment
                        # like "82)" should be replaced by the full
                        # "(82) 0.5x45°" it came from.
                        od_clean = re.sub(r"[\s()]", "", other.dimension or "")
                        fd_clean = re.sub(r"[\s()]", "", old_dim or "")
                        if not (od_clean and len(od_clean) >= 2
                                and od_clean in fd_clean
                                and len(od_clean) < len(fd_clean)):
                            continue
                    o_anchor = _trace_endpoint_for(other)
                    d = math.hypot(o_anchor[0] - anchor[0], o_anchor[1] - anchor[1])
                    if d < best_dist:
                        best_dist = d
                        best_recipient = other
                if best_recipient is not None and best_dist < 200.0:
                    self._apply_assignment_candidate(
                        best_recipient,
                        old_dim,
                        max(float(getattr(best_recipient, "confidence", 0.0) or 0.0), 0.55),
                        f"transferred_from_{b.bubble_number}",
                        review_threshold=0.99,
                    )
                    resolved[best_recipient.bubble_number] = best_recipient

        if not upgradeable or not resolved:
            return

        # ── Strategy 2: trace endpoint proximity ─────────────────────
        # If two bubbles' leader paths end within a small radius they
        # reference the same dimension region.
        if not self._seed_traces or not upgradeable:
            return

        endpoint_radius = 30.0  # px — tight to avoid false matches
        for b in list(upgradeable):
            trace = self._trace_for_bubble(b, bubble_index_by_obj.get(id(b)))
            path = (trace or {}).get("path") or []
            if not path:
                continue
            ex, ey = path[-1]
            best_donor: Optional[BubbleResult] = None
            best_dist = endpoint_radius
            for r_bid, r_bubble in resolved.items():
                r_trace = self._trace_for_bubble(
                    r_bubble,
                    bubble_index_by_obj.get(id(r_bubble)),
                )
                r_path = (r_trace or {}).get("path") or []
                if not r_path:
                    continue
                rx, ry = r_path[-1]
                d = math.hypot(ex - rx, ey - ry)
                if d < best_dist:
                    # Extra safety: the two bubbles should not be too
                    # close together (otherwise one trace endpoint is
                    # simply near the other bubble, not a shared target).
                    inter_bubble_dist = math.hypot(
                        b.x - r_bubble.x, b.y - r_bubble.y
                    )
                    if inter_bubble_dist < b.radius * 4:
                        continue  # bubbles too close — endpoints are
                                  # coincidentally nearby, not shared
                    best_dist  = d
                    best_donor = r_bubble
            if best_donor is not None:
                self._apply_assignment_candidate(
                    b,
                    best_donor.dimension,
                    float(best_donor.confidence) * 0.85,
                    f"shared_dim_trace_{best_donor.bubble_number}",
                    review_threshold=0.99,
                )
                resolved[b.bubble_number] = b
                upgradeable.remove(b)

        # ── Strategy 3: annotation-line connectivity ─────────────────
        # If an unresolved bubble is geometrically close to a resolved
        # bubble AND both lie on the same vertical/horizontal annotation
        # line (common for stacked equal-dimension callouts), propagate.
        if not upgradeable or self.image is None:
            return

        for b in list(upgradeable):
            # Check resolved bubbles within reasonable proximity
            for r_bid, r_bubble in resolved.items():
                dist = math.hypot(b.x - r_bubble.x, b.y - r_bubble.y)
                if dist > 350:
                    continue  # too far apart
                # Check if aligned vertically or horizontally
                dx = abs(b.x - r_bubble.x)
                dy = abs(b.y - r_bubble.y)
                if dx > 80 and dy > 80:
                    continue  # not aligned
                # Verify pixel connectivity along the alignment axis
                if self._check_annotation_line_connectivity(
                    b, r_bubble, max_gap=8
                ):
                    self._apply_assignment_candidate(
                        b,
                        r_bubble.dimension,
                        float(r_bubble.confidence) * 0.80,
                        f"shared_dim_line_{r_bid}",
                        review_threshold=0.99,
                    )
                    resolved[b.bubble_number] = b
                    upgradeable.remove(b)
                    break

    def _check_annotation_line_connectivity(
        self,
        bubble_a: BubbleResult,
        bubble_b: BubbleResult,
        max_gap: int = 8,
    ) -> bool:
        """
        Check whether two bubbles are connected via a continuous
        annotation line (dark pixels along the straight path between
        their centres).

        Uses adaptive thresholding on a narrow corridor between the
        bubble centres.  Returns True if ≥ 70% of sampled points along
        the corridor are dark (ink) pixels, allowing for small gaps.
        """
        if self.image is None:
            return False

        h, w = self.image.shape[:2]
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image

        # Sample points along the straight line between centres
        ax, ay = float(bubble_a.x), float(bubble_a.y)
        bx, by = float(bubble_b.x), float(bubble_b.y)
        dist = math.hypot(bx - ax, by - ay)
        if dist < 1:
            return False

        n_samples = max(10, int(dist / 2))
        dark_count = 0
        total_valid = 0

        # Skip points inside bubble radii
        ra = max(1.0, float(bubble_a.radius))
        rb = max(1.0, float(bubble_b.radius))

        # Otsu threshold for "dark pixel" classification
        thresh_val = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]

        for i in range(n_samples + 1):
            t = i / n_samples
            px = int(ax + t * (bx - ax))
            py = int(ay + t * (by - ay))
            if not (0 <= px < w and 0 <= py < h):
                continue
            # Skip points inside either bubble
            if math.hypot(px - ax, py - ay) < ra * 1.1:
                continue
            if math.hypot(px - bx, py - by) < rb * 1.1:
                continue
            total_valid += 1
            if gray[py, px] < thresh_val:
                dark_count += 1

        if total_valid < 5:
            return False
        return (dark_count / total_valid) >= 0.40

    # ── Confidence assessment ──────────────────────────────────────

    # ── VLM Verification ────────────────────────────────────────────

    def _vlm_verify_and_correct(
        self,
        bubbles: List[BubbleResult],
        image: np.ndarray,
    ) -> None:
        """
        Use a local vision-language model (via Ollama) to verify and
        correct dimension assignments.

        For each bubble, crops a region around the bubble (including its
        leader line context) and asks the VLM to read the dimension
        value the leader points to.

        Fusion logic:
          - If VLM and CV agree → high confidence, keep CV value
          - If VLM reads a valid value and CV has NO_DIMENSION → use VLM
          - If VLM reads a different valid value → prefer VLM (it can
            read tiny text and understand spatial relationships that
            OCR misses)
        """
        try:
            from ollama import Client
            client = Client(host=self.cfg.vlm_host)
            # Quick connectivity check
            client.list()
        except Exception:
            logger.info("VLM not available (Ollama not running); skipping verification")
            return

        import base64
        h, w = image.shape[:2]

        # For unresolved bubbles, use VLM with tight directional
        # micro-crops.  Scan a ring of small crops around the bubble
        # and ask the VLM to read the digit in each.  This catches
        # tiny dimension text that OCR misses.

        micro_prompt = (
            "What number or dimension value is written between the "
            "dimension lines in this engineering drawing crop? "
            "Answer with just the digit or value, nothing else."
        )

        for b in bubbles:
            cv_dim = b.dimension or ""
            cv_reason = getattr(b, 'review_reason', '')

            need_vlm = (
                not cv_dim or cv_dim == "NO_DIMENSION"
                or (cv_reason == 'micro_ocr_endpoint' and b.confidence < 0.6)
            )
            if not need_vlm:
                continue

            bx, by, br = int(b.x), int(b.y), int(b.radius)

            # Scan tight micro-crops in a ring around the bubble.
            # Each crop is a small window at 2-3× radius in 8 directions.
            # The VLM reads the digit/value in each crop.
            other_positions = [
                (ob.x, ob.y, ob.radius) for ob in bubbles
                if ob.bubble_number != b.bubble_number
            ]
            from collections import Counter as _VLMCounter
            vlm_answers: _VLMCounter = _VLMCounter()
            scan_dist = int(br * 2.5)
            crop_half = max(40, int(br * 0.7))  # tight crops

            for ang_deg in range(0, 360, 45):
                ang = math.radians(ang_deg)
                sx = bx + int(scan_dist * math.cos(ang))
                sy = by + int(scan_dist * math.sin(ang))
                # Skip if inside another bubble
                if any(math.hypot(sx - ox, sy - oy) < orad * 1.3
                       for ox, oy, orad in other_positions):
                    continue
                x1 = max(0, sx - crop_half)
                y1 = max(0, sy - crop_half)
                x2 = min(w, sx + crop_half)
                y2 = min(h, sy + crop_half)
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                _, buffer = cv2.imencode('.png', roi)
                img_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

                try:
                    response = client.chat(
                        model=self.cfg.vlm_model,
                        messages=[{
                            'role': 'user',
                            'content': micro_prompt,
                            'images': [img_b64],
                        }],
                        options={'temperature': 0.0},
                    )
                    raw = response['message']['content'].strip()
                    raw = raw.split('\n')[0].strip().strip('"\'')
                    import re as _re
                    raw = _re.sub(r'\s*(mm|MM|degrees?|DEG)\s*$', '', raw).strip()
                    if raw and len(raw) <= 5 and _re.search(r'\d', raw):
                        vlm_answers[raw] += 1
                except Exception:
                    continue

            if not vlm_answers:
                continue

            vlm_text, vote_count = vlm_answers.most_common(1)[0]
            if vote_count < 2:
                continue  # no consensus

            from ocr_rules import normalize_engineering_text
            vlm_norm = normalize_engineering_text(vlm_text)
            if not vlm_norm:
                continue

            cv_dim = b.dimension or ""
            cv_norm = normalize_engineering_text(cv_dim) if cv_dim and cv_dim != "NO_DIMENSION" else ""

            # Fusion logic
            if cv_norm and cv_norm.upper() == vlm_norm.upper():
                # Agreement — boost confidence
                b.confidence = min(0.98, b.confidence + 0.15)
                continue

            if not cv_norm or cv_dim == "NO_DIMENSION":
                # CV had nothing, VLM found something → use VLM
                self._apply_assignment_candidate(
                    b,
                    vlm_norm,
                    0.75,
                    "vlm_fill",
                    review_threshold=0.50,
                )
                continue

            # We only reach here for uncertain CV bubbles, so
            # prefer VLM if it produced a valid dimension.
            if self._is_valid_dimension_text(vlm_norm):
                self._apply_assignment_candidate(
                    b,
                    vlm_norm,
                    0.70,
                    "vlm_override",
                    review_threshold=0.50,
                )

    def _leader_attention_ocr(
        self,
        bubbles: List[BubbleResult],
        image: np.ndarray,
        callout_groups: List[CalloutGroup],
    ) -> None:
        """
        Targeted OCR along the leader exit direction for low-confidence
        or duplicate-value bubbles.

        For each weak bubble, compute the leader exit direction, define
        a search rectangle along that direction, and run high-res OCR
        on just that crop.  The small crop isolates the target dimension
        text from surrounding noise, catching small text that global
        OCR deprioritizes.
        """
        from leader_geometry import find_leader_exits
        from ocr_rules import normalize_engineering_text

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        assigned_dims = {}
        for b in bubbles:
            if b.dimension and b.dimension != "NO_DIMENSION":
                assigned_dims.setdefault(b.dimension, []).append(b.bubble_number)
        bubble_ids = {b.bubble_number for b in bubbles}

        for b in bubbles:
            # Only process weak assignments
            has_dim = b.dimension and b.dimension != "NO_DIMENSION"
            is_weak = has_dim and b.confidence < 0.50
            is_dup = (has_dim and len(assigned_dims.get(b.dimension, [])) > 1)
            if not (is_weak or is_dup):
                continue

            bx, by, br = int(b.x), int(b.y), max(1, int(b.radius))
            exits = find_leader_exits(gray, bx, by, br, max_exits=2)
            if not exits:
                continue

            ex = exits[0]
            dx, dy = ex.direction_x, ex.direction_y

            # Search rectangle along exit direction
            center_x = int(bx + dx * br * 2.5)
            center_y = int(by + dy * br * 2.5)
            pad_w = int(br * 3.0)
            pad_h = int(br * 1.5)

            x1 = max(0, center_x - pad_w)
            y1 = max(0, center_y - pad_h)
            x2 = min(w, center_x + pad_w)
            y2 = min(h, center_y + pad_h)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                continue

            # High-res OCR on the crop
            scale = 4
            up = cv2.resize(crop, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
            gray_c = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray_c)

            best_text = None
            best_conf = 0.0
            for variant in [up, cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)]:
                try:
                    result = self.ocr(variant)
                except Exception:
                    continue
                items = (result[0] if isinstance(result, tuple)
                         and len(result) >= 1 else (result or []))
                for item in items or []:
                    try:
                        ti = item[1]
                        text = str(ti[0] if isinstance(ti, (list, tuple)) else ti).strip()
                        conf = float(ti[1] if isinstance(ti, (list, tuple)) and len(ti) > 1 else 0.9)
                        norm = normalize_engineering_text(text)
                        if not norm or not re.search(r"\d", norm):
                            continue
                        # Skip bubble IDs
                        if norm in bubble_ids:
                            continue
                        # Skip already-assigned dimensions
                        if norm == b.dimension:
                            continue
                        if not self._is_plausible_dimension(norm):
                            continue
                        if conf > best_conf:
                            best_conf = conf
                            best_text = norm
                    except Exception:
                        continue

            if best_text and best_conf > 0.5:
                # Only accept if the new text isn't already used by a
                # non-weak bubble
                used_by_strong = any(
                    bb.dimension == best_text and bb.confidence >= 0.50
                    for bb in bubbles if bb.bubble_number != b.bubble_number
                )
                if not used_by_strong:
                    self._apply_assignment_candidate(
                        b,
                        best_text,
                        max(float(b.confidence), best_conf * 0.8),
                        "leader_attention_ocr",
                        review_threshold=0.86,
                    )

    @staticmethod
    def _normalize_assigned_dimensions(bubbles: List[BubbleResult]) -> None:
        """Fix common OCR artifacts in assigned dimension text.

        Rules applied (domain knowledge, not image-specific):
          0. Axis-origin labels — "0,0" / "+X" / "-Y" / "0.0+X" / etc.
             are coordinate-system labels at the drawing origin, not
             dimensions. Clear them so the inspector sees an empty
             cell rather than a misleading value.
          1. Leading-zero diameter: "063" → "Ø63", "063.1" → "Ø63.1"
             OCR reads the Ø symbol as digit "0". When a dimension
             starts with "0" followed by 2+ digits (not a decimal like
             "0.5"), it's almost always a diameter with misread prefix.
          2. Duplicate value in tolerance format:
             "184 ±0.1 / 184" → "184 ±0.1"
             OCR reads the same number twice from stacked tolerance text.
          3. Stray closing parens: ")0.5" → "0.5"
             OCR occasionally reads a bracket artifact before a number.
        """
        # Pre-compiled patterns that recognise axis-origin labels.
        # A dim text is rejected if (after stripping whitespace and
        # normalising commas to dots) it matches any of these — they
        # never represent real engineering dimensions.
        AXIS_LABEL_PATTERNS = (
            re.compile(r"^[+\-]?\s*0[.,]0\s*[+\-]?[xy]?$", re.IGNORECASE),
            re.compile(r"^[+\-][xy]$", re.IGNORECASE),
            re.compile(r"^[+\-]?\s*0[.,]0\s*[+\-][xy]$", re.IGNORECASE),
            re.compile(r"^[+\-]\s*[xy]\s*[+\-]?\s*[xy]?$", re.IGNORECASE),
        )

        for b in bubbles:
            dim = b.dimension
            if not dim or dim == "NO_DIMENSION":
                continue

            if "\u0105" in dim:
                dim = dim.replace("\u0105", "\u00b1")
                b.dimension = dim

            # Rule 0: axis-origin labels never belong to a real bubble.
            stripped = dim.strip()
            if any(p.fullmatch(stripped) for p in AXIS_LABEL_PATTERNS):
                logger.info("Cleared axis-label dim for bubble #%s: %r",
                            b.bubble_number, dim)
                b.dimension = ""
                b.needs_review = True
                continue

            # Rule 0: decimal comma → dot  ("0,001 A" → "0.001 A")
            # RapidOCR is locale-agnostic and frequently emits European
            # decimal commas on engineering drawings. Only replace when
            # the comma sits between digits (so list separators like
            # "A, B" are untouched).
            new_dim = re.sub(r"(\d),(\d)", r"\1.\2", dim)
            if new_dim != dim:
                b.dimension = new_dim
                dim = new_dim

            # Rule 0a.1: OCR can read the dot/zero in a tolerance as C/O.
            # "0.800±C01" -> "0.800±.001".
            def _fix_alpha_tolerance(match: re.Match) -> str:
                digits = match.group(2).zfill(3)
                return f"{match.group(1)}.{digits}"

            new_dim = re.sub(
                r"([±\u00b1]\s*)[CO]\s*(\d{1,3})(?=\s*$)",
                _fix_alpha_tolerance,
                dim,
                flags=re.IGNORECASE,
            )
            if new_dim != dim:
                b.dimension = new_dim
                dim = new_dim

            # Rule 0b: tolerance before nominal value.
            # OCR may read stacked tolerance text above/left of the
            # nominal dimension first: "±0.05 22.00" -> "22.00 ±0.05".
            m = re.match(
                r"^\s*([±\u00b1+\-]\s*\d+(?:\.\d+)?)\s+"
                r"(\d+(?:\.\d+)?(?:\s*[A-Za-z]\d+)?)\s*$",
                dim.strip(),
            )
            if m:
                dim = f"{m.group(2)} {m.group(1)}"
                b.dimension = dim

            # Rule 0c: duplicated slash separators in stacked tolerance
            # reads: "45.00h7 / /+0.025" -> "45.00h7 / +0.025".
            new_dim = re.sub(r"/\s*/\s*", "/ ", dim)
            if new_dim != dim:
                b.dimension = new_dim
                dim = new_dim

            # Rule 0d: OCR reads diameter as a leading zero and THRU as
            # "1hRU"/"IhRU". These are shape-level OCR confusions, not
            # image-specific values.
            m = re.match(
                r"^0(\d{1,3}(?:\.\d+)?)\s+[1IilTt][hH][rR][uU]\b(.*)$",
                dim.strip(),
            )
            if m:
                dim = f"Ø{m.group(1)} THRU{m.group(2)}"
                b.dimension = dim

            m = re.match(
                r"^0\s+(\d{1,3}(?:\.\d+)?)\s+THRU\b(.*)$",
                dim.strip(),
                flags=re.IGNORECASE,
            )
            if m:
                dim = f"Ø{m.group(1)} THRU{m.group(2)}"
                b.dimension = dim

            # Rule 0e: leading zero diameter with fit/tolerance suffix:
            # "045.00h7 / +0.025" -> "Ø45.00h7 / +0.025".
            m = re.match(
                r"^0(\d{2,}(?:\.\d+)?(?:\s*[A-Za-z]\d+)?"
                r"(?:\s*/\s*[+\-]?\d+(?:\.\d+)?)?)$",
                dim.strip(),
            )
            if m:
                dim = f"Ø{m.group(1)}"
                b.dimension = dim

            if BubbleDetector._score_text_quality(dim) <= -5:
                logger.info("Cleared note/table dim for bubble #%s: %r",
                            b.bubble_number, dim)
                b.dimension = ""
                b.needs_review = True
                continue

            # Rule 1: leading-zero → diameter prefix
            # "063" → "Ø63", "063.1" → "Ø63.1"
            m = re.match(r"^0(\d{2,}(?:\.\d+)?)$", dim.strip())
            if m:
                b.dimension = f"Ø{m.group(1)}"
                continue

            # Rule 2: duplicate value in tolerance
            # "184 ±0.1 / 184" → "184 ±0.1"
            m = re.match(
                r"^(\d+(?:\.\d+)?)\s*([±]\s*\d+(?:\.\d+)?)\s*/\s*\1$",
                dim.strip(),
            )
            if m:
                b.dimension = f"{m.group(1)} {m.group(2)}"
                continue

            # Rule 3: stray closing paren before a digit.
            # ")0.5" -> "0.5"
            dim = re.sub(r"(?<=\d)\)\s+(?=\d)", " ", dim)
            dim = re.sub(r"\)(\d)", r"\1", dim)
            if dim != b.dimension:
                b.dimension = dim
                continue

            # Rule 4: misplaced ° in chamfer — "0.5°45°" → "0.5x45°"
            # OCR sometimes reads "x" as "°" in chamfer specs
            m = re.match(r"^(\d+(?:\.\d+)?)°(\d+)°?$", dim.strip())
            if m:
                b.dimension = f"{m.group(1)}x{m.group(2)}°"

            m = re.match(r"^(\d+\.\d+)[A-Za-z]\d+$", dim.strip())
            if m:
                b.dimension = m.group(1)

    @staticmethod
    def _clear_axis_label_assignments(bubbles: List[BubbleResult]) -> None:
        """Clear coordinate-axis labels that late correction may assign."""
        axis_label_patterns = (
            re.compile(r"^[+\-]?\s*0[.,]0\s*[+\-]?[xy]?$", re.IGNORECASE),
            re.compile(r"^[+\-][xy]$", re.IGNORECASE),
            re.compile(r"^[+\-]?\s*0[.,]0\s*[+\-][xy]$", re.IGNORECASE),
            re.compile(r"^[+\-]\s*[xy]\s*[+\-]?\s*[xy]?$", re.IGNORECASE),
        )
        for b in bubbles:
            dim = (b.dimension or "").strip()
            if not dim or dim == "NO_DIMENSION":
                continue
            if not any(p.fullmatch(dim) for p in axis_label_patterns):
                continue
            logger.info(
                "Cleared final axis-label dim for bubble #%s: %r",
                b.bubble_number,
                b.dimension,
            )
            b.dimension = ""
            b.needs_review = True

    @staticmethod
    def _is_table_description_assignment(text: str) -> bool:
        cleaned = (text or "").strip().upper()
        if not cleaned:
            return False
        cleaned = cleaned.replace("0", "O")
        cleaned = re.sub(r"[^A-Z0-9#ØΦ]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return False
        patterns = (
            r"OVERALL LENGTH",
            r"BODY LENGTH",
            r"STEP LENGTH",
            r"SHOULDER LENGTH",
            r"FLANGE (?:OD|WIDTH|THICKNESS|STEP)",
            r"GROOVE (?:WIDTH|DEPTH|HEIGHT)",
            r"BASE (?:LENGTH|HEIGHT)",
            r"BORE(?:\s*(?:[#ØΦ]?\s*[A-Z0-9()]+|LENGTH|CIRCLE))*",
            r"BOSS(?:\s*[A-Z0-9()]+)?",
            r"OUTER DIAMETER",
            r"INTERNAL FILLET",
            r"FILLET",
            r"THREAD",
            r"CHAMFER",
            r"KEYWAY",
            r"KEYWAY WIDTH",
            r"SECTION",
            r"DETAIL [A-Z](?: WIDTH| DEPTH| RADIUS)?",
            r"POSITION(?: TOL(?:ERANCE)?)?",
            r"PERPENDICULARITY",
            r"HOLE SPACING",
            r"LINEAR",
            r"ANGULAR",
            r"RADII",
            r"HOLE DIAMETERS?",
            r"SHAFT DIAMETERS?",
            r"MACHINED SURFACES",
            r"CAST SURFACES",
        )
        return any(re.fullmatch(pattern, cleaned) for pattern in patterns)

    @staticmethod
    def _clear_table_description_assignments(bubbles: List[BubbleResult]) -> None:
        """Clear row/header descriptors from dimension-reference tables."""
        for b in bubbles:
            dim = (b.dimension or "").strip()
            if not dim or dim == "NO_DIMENSION":
                continue
            if not BubbleDetector._is_table_description_assignment(dim):
                continue
            logger.info(
                "Cleared table-description dim for bubble #%s: %r",
                b.bubble_number,
                dim,
            )
            b.dimension = ""
            b.needs_review = True

    def _repair_omitted_dimension_prefixes(
        self,
        bubbles: List[BubbleResult],
    ) -> None:
        """Restore leading decimal points that OCR dropped from callouts.

        This pass is intentionally evidence-based. A bare 3-digit assignment
        becomes `.ddd` only when a dot-like ink component is physically present
        immediately to the left of the matching OCR token.
        """
        if self.image is None or not bubbles:
            return
        tokens = getattr(self, "_norm_tokens", None) or []
        if not tokens:
            return

        for b in bubbles:
            dim = (b.dimension or "").strip()
            if not dim or dim == "NO_DIMENSION":
                continue

            m_tol = re.fullmatch(r"(\d{3})(\s*[\u00b1]\s*\.?\d+)", dim)
            if m_tol:
                b.dimension = f".{m_tol.group(1)}{m_tol.group(2)}"
                dim = b.dimension

            if not re.fullmatch(r"\d{3}", dim):
                continue

            tok = self._nearest_matching_dimension_token(b, dim, tokens)
            if tok is None:
                continue
            if self._token_has_left_decimal_dot(tok):
                logger.info(
                    "Restored omitted leading decimal for bubble #%s: %s -> .%s",
                    b.bubble_number,
                    dim,
                    dim,
                )
                b.dimension = f".{dim}"

    def _nearest_matching_dimension_token(
        self,
        bubble: BubbleResult,
        dim: str,
        tokens: List[NormalizedToken],
    ) -> Optional[NormalizedToken]:
        best = None
        best_dist = float("inf")
        for tok in tokens:
            text = (getattr(tok, "text", "") or "").strip()
            raw = (getattr(tok, "raw_text", "") or "").strip()
            if text != dim and raw != dim:
                continue
            if getattr(tok, "token_type", "") != "dimension":
                continue
            dist = math.hypot(float(tok.cx) - float(bubble.x),
                              float(tok.cy) - float(bubble.y))
            if dist < best_dist:
                best_dist = dist
                best = tok
        return best

    def _token_has_left_decimal_dot(self, tok: NormalizedToken) -> bool:
        if self.image is None:
            return False
        h_img, w_img = self.image.shape[:2]
        x1 = int(round(float(tok.x1)))
        y1 = int(round(float(tok.y1)))
        x2 = int(round(float(tok.x2)))
        y2 = int(round(float(tok.y2)))
        token_h = max(1, y2 - y1)
        strip_w = max(10, int(round(token_h * 0.65)))
        sx1 = max(0, x1 - strip_w)
        sx2 = max(0, min(w_img, x1 + 2))
        sy1 = max(0, y1 - int(token_h * 0.35))
        sy2 = min(h_img, y2 + int(token_h * 0.35))
        if sx2 <= sx1 or sy2 <= sy1:
            return False

        crop = self.image[sy1:sy2, sx1:sx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        nlabels, _labels, stats, cents = cv2.connectedComponentsWithStats(
            inv, connectivity=8
        )
        for i in range(1, nlabels):
            x, y, bw, bh, area = stats[i]
            if area < 2 or area > max(18, token_h * token_h * 0.08):
                continue
            if bw > max(5, token_h * 0.28) or bh > max(5, token_h * 0.35):
                continue
            abs_cx = sx1 + float(cents[i][0])
            abs_cy = sy1 + float(cents[i][1])
            close_to_token = (x1 - token_h * 0.35) <= abs_cx <= (x1 + 3)
            lower_half = (y1 + token_h * 0.45) <= abs_cy <= (y2 + token_h * 0.25)
            if close_to_token and lower_half:
                return True
        return False

    def _assign_keyword_note_callouts(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
    ) -> None:
        """Assign nonnumeric note callouts when leader geometry is explicit."""
        if not bubbles or not callout_groups:
            return

        keywords = [
            cg for cg in callout_groups
            if self._is_assignable_callout(
                cg,
                allow_reference_only=True,
                allow_keyword_only=True,
            )
            and cg.callout_type == "keyword"
            and not self._is_coordinate_callout(cg)
        ]
        if not keywords:
            return

        claimed = {
            (b.dimension or "").strip()
            for b in bubbles
            if (b.dimension or "").strip()
            and (b.dimension or "").strip() != "NO_DIMENSION"
        }

        for bi, b in enumerate(bubbles):
            dim = (b.dimension or "").strip()
            if dim and dim != "NO_DIMENSION":
                continue

            trace = self._trace_for_bubble(b, bi)
            path = (trace or {}).get("path") or []
            if not path:
                continue

            bx, by = float(b.x), float(b.y)
            br = max(1.0, float(b.radius or 1))
            ex, ey = map(float, path[-1])
            best: Optional[Tuple[float, CalloutGroup]] = None
            for cg in keywords:
                text = (cg.text or "").strip()
                if not text or text in claimed:
                    continue
                box = (float(cg.x1), float(cg.y1), float(cg.x2), float(cg.y2))
                endpoint_dist = self._point_to_box_distance(ex, ey, *box)
                bubble_dist = self._point_to_box_distance(bx, by, *box)
                supported = (
                    endpoint_dist <= max(28.0, br * 0.9)
                    or (
                        endpoint_dist <= max(80.0, br * 2.0)
                        and self._trace_supports_box(b, box)
                    )
                    or bubble_dist <= max(65.0, br * 2.0)
                )
                if not supported:
                    continue
                score = min(endpoint_dist, bubble_dist * 0.75)
                if best is None or score < best[0]:
                    best = (score, cg)

            if best is None:
                continue

            _, cg = best
            reason = getattr(b, "review_reason", "") or ""
            text = (cg.text or "").strip()
            self._apply_assignment_candidate(
                b,
                text,
                max(float(getattr(b, "confidence", 0.0) or 0.0), 0.62),
                ((reason + "+") if reason else "") + "keyword_note_assignment",
                review_threshold=0.86,
            )
            claimed.add(text)
            logger.info(
                "Keyword note assignment: bubble #%s <- %r",
                b.bubble_number,
                b.dimension,
            )

    def _assign_local_angle_callouts(self, bubbles: List[BubbleResult]) -> None:
        """OCR small local angle values for unresolved leader balloons."""
        if self.image is None or not bubbles:
            return

        for bi, b in enumerate(bubbles):
            dim = (b.dimension or "").strip()
            if dim and dim != "NO_DIMENSION":
                continue

            centers = self._local_angle_crop_centers(b, bi)
            if not centers:
                continue
            if not self._has_nearby_angle_context(b, centers):
                continue
            candidates: Dict[str, float] = {}
            for cx, cy in centers[:5]:
                for text, conf in self._ocr_local_angle_crop(b, int(cx), int(cy)):
                    cand = self._normalize_local_angle_candidate(text)
                    if cand is None:
                        continue
                    angle_value = cand[:-1] if cand.endswith("°") else cand
                    if angle_value == str(b.bubble_number).strip():
                        continue
                    candidates[cand] = max(candidates.get(cand, 0.0), float(conf))

            if not candidates:
                continue
            best_text, best_conf = sorted(
                candidates.items(),
                key=lambda item: (item[1], len(item[0])),
                reverse=True,
            )[0]
            reason = getattr(b, "review_reason", "") or ""
            self._apply_assignment_candidate(
                b,
                best_text,
                max(float(getattr(b, "confidence", 0.0) or 0.0), min(0.70, best_conf)),
                ((reason + "+") if reason else "") + "local_angle_rescue",
                review_threshold=0.86,
            )
            logger.info(
                "Local angle rescue: bubble #%s <- %r",
                b.bubble_number,
                best_text,
            )

    def _has_nearby_angle_context(
        self,
        bubble: BubbleResult,
        centers: List[Tuple[int, int]],
    ) -> bool:
        """Cheaply gate local angle OCR to areas with degree evidence."""
        label = str(bubble.bubble_number or "").strip()
        br = max(1.0, float(bubble.radius or 1))
        max_dist = max(96.0, br * 3.5)

        def _has_degree_text(text: str) -> bool:
            t = (text or "").strip().upper()
            return bool(re.search(r"[°º˚]|Â°|Ã‚|\bDEG(?:REE)?S?\b", t))

        def _near_box(x1: float, y1: float, x2: float, y2: float) -> bool:
            if self._trace_supports_box(bubble, (x1, y1, x2, y2)):
                return True
            for cx, cy in centers[:3]:
                if self._point_to_box_distance(float(cx), float(cy), x1, y1, x2, y2) <= max_dist:
                    return True
            return False

        for cg in getattr(self, "_callout_groups", []) or []:
            text = getattr(cg, "text", "") or ""
            kind = (getattr(cg, "callout_type", "") or "").lower()
            if kind != "angle" and not _has_degree_text(text):
                continue
            if text.strip() == label:
                continue
            if _near_box(float(cg.x1), float(cg.y1), float(cg.x2), float(cg.y2)):
                return True

        for tok in getattr(self, "_norm_tokens", []) or []:
            text = getattr(tok, "text", "") or getattr(tok, "raw_text", "") or ""
            if not _has_degree_text(text):
                continue
            if normalize_bubble_value(text) == label:
                continue
            if _near_box(float(tok.x1), float(tok.y1), float(tok.x2), float(tok.y2)):
                return True

        return False

    def _assign_local_plain_numeric_callouts(self, bubbles: List[BubbleResult]) -> None:
        """Assign plain numeric values only when leader geometry supports them."""
        if self.image is None or not bubbles:
            return

        claimed = {
            (b.dimension or "").strip()
            for b in bubbles
            if (b.dimension or "").strip()
            and (b.dimension or "").strip() != "NO_DIMENSION"
        }

        for bi, b in enumerate(bubbles):
            dim = (b.dimension or "").strip()
            if dim and dim != "NO_DIMENSION":
                continue
            if not self._has_trusted_colored_bubble_evidence(b):
                continue

            label = str(b.bubble_number or "").strip()
            br = max(1.0, float(b.radius or 1))
            trace = self._trace_for_bubble(b, bi)
            path = (trace or {}).get("path") or []
            if not path:
                continue
            ex, ey = map(float, path[-1])

            best: Optional[Tuple[float, str, float]] = None
            for tok in getattr(self, "_norm_tokens", []) or []:
                text = (getattr(tok, "text", "") or getattr(tok, "raw_text", "") or "").strip()
                if not re.fullmatch(r"\d{1,3}", text):
                    continue
                if text == label or text in claimed:
                    continue
                if getattr(tok, "is_maroon", False):
                    continue
                box = (float(tok.x1), float(tok.y1), float(tok.x2), float(tok.y2))
                endpoint_dist = self._point_to_box_distance(ex, ey, *box)
                bubble_dist = self._point_to_box_distance(float(b.x), float(b.y), *box)
                supported = (
                    endpoint_dist <= max(85.0, br * 2.4)
                    or self._trace_supports_box(b, box)
                    or bubble_dist <= max(135.0, br * 4.0)
                )
                if not supported:
                    continue
                score = min(endpoint_dist, bubble_dist * 0.65)
                if best is None or score < best[0]:
                    best = (score, text, float(getattr(tok, "conf", 0.75) or 0.75))

            if best is None:
                local_votes: Dict[str, Tuple[int, float]] = {}
                for cx, cy in self._local_angle_crop_centers(b, bi)[:5]:
                    for raw_text, conf in self._ocr_local_angle_crop(b, int(cx), int(cy)):
                        cand = self._normalize_local_plain_numeric_candidate(raw_text)
                        if cand is None or cand == label or cand in claimed:
                            continue
                        count, prev_conf = local_votes.get(cand, (0, 0.0))
                        local_votes[cand] = (count + 1, max(prev_conf, float(conf)))
                repeated = [
                    (count, conf, text)
                    for text, (count, conf) in local_votes.items()
                    if count >= 2
                ]
                if repeated:
                    repeated.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    count, conf, text = repeated[0]
                    best = (999.0 - count, text, conf)

            if best is None:
                continue

            _score, text, conf = best
            reason = getattr(b, "review_reason", "") or ""
            self._apply_assignment_candidate(
                b,
                text,
                max(
                    float(getattr(b, "confidence", 0.0) or 0.0),
                    min(0.68, conf),
                ),
                ((reason + "+") if reason else "") + "plain_numeric_leader_rescue",
                review_threshold=0.86,
            )
            claimed.add(text)
            logger.info(
                "Plain numeric leader rescue: bubble #%s <- %r",
                b.bubble_number,
                b.dimension,
            )

    def _assign_traced_endpoint_ocr_callouts(
        self,
        bubbles: List[BubbleResult],
        *,
        strict_rich_only: bool = False,
    ) -> None:
        """Read dimension text at traced leader endpoints for unresolved bubbles."""
        if self.image is None or not bubbles:
            return

        from collections import Counter
        from ocr_rules import normalize_engineering_text

        bubble_ids = {str(b.bubble_number or "").strip() for b in bubbles}
        claimed = {
            (b.dimension or "").strip()
            for b in bubbles
            if (b.dimension or "").strip()
            and (b.dimension or "").strip() != "NO_DIMENSION"
        }

        def _endpoint_candidate_allowed(text: str) -> bool:
            t = (text or "").strip().upper()
            if not t:
                return False
            letters = re.sub(r"[^A-Z]", "", t)
            if not letters:
                return True
            engineering_words = (
                "DIA", "MAJOR", "MINOR", "MIN", "MAX", "REF", "TYP",
                "THRU", "THREAD", "TAPER", "MARK", "ETCH", "DETAIL",
                "ALL", "AROUND", "PROFILE",
            )
            if any(word in t for word in engineering_words):
                return True
            if re.search(r"\bM\d", t) or re.search(r"\bR\d", t):
                return True
            return False

        def _rich_endpoint_dimension(text: str) -> bool:
            t = (text or "").strip().upper()
            if not t or self._is_table_description_assignment(t):
                return False
            if re.fullmatch(r"\d{1,3}", t):
                return False
            return bool(
                re.search(r"\d+[.,]\d+", t)
                or re.search(r"[ØΦÂ±±°º˚×X/]", t)
                or re.search(r"\b(?:DIA|REF|TYP|THRU|M\d|R\d)\b", t)
            )

        for bi, b in enumerate(bubbles):
            dim = (b.dimension or "").strip()
            if dim and dim != "NO_DIMENSION":
                continue

            trace = self._trace_for_bubble(b, bi)
            path = (trace or {}).get("path") or []
            if not path:
                continue
            quality = (trace or {}).get("quality")
            quality_score = float(
                getattr(quality, "score", (trace or {}).get("quality_score", 0.0)) or 0.0
            )
            if quality_score < 0.45:
                if trace is not None:
                    trace["endpoint_ocr_status"] = "skipped_low_trace_quality"
                continue
            if strict_rich_only and quality_score < max(
                0.78,
                float(self.cfg.min_dense_leader_trace_quality),
            ):
                if trace is not None:
                    trace["endpoint_ocr_status"] = "skipped_strict_trace_quality"
                continue
            if strict_rich_only and not (
                self._has_trusted_colored_bubble_evidence(b)
                or self._has_matching_maroon_label_token(b)
            ):
                if trace is not None:
                    trace["endpoint_ocr_status"] = "skipped_strict_bubble_evidence"
                continue
            ex, ey = path[-1]
            br = max(1.0, float(b.radius or 1))
            if math.hypot(float(ex) - float(b.x), float(ey) - float(b.y)) <= br * 1.15:
                if trace is not None:
                    trace["endpoint_ocr_status"] = "skipped_endpoint_near_balloon"
                continue

            votes: Counter[str] = Counter()
            confidences: Dict[str, float] = {}

            def _collect_from_center(cx: int, cy: int) -> None:
                for raw_text, conf in self._ocr_local_angle_crop(b, cx, cy):
                    norm0 = normalize_engineering_text(raw_text)
                    candidates = self._local_rescue_text_candidates(norm0)
                    if norm0 and norm0 not in candidates:
                        candidates.insert(0, norm0)
                    for cand in candidates:
                        cand = (cand or "").strip()
                        if not cand or not re.search(r"\d", cand):
                            continue
                        if cand == str(b.bubble_number).strip() or cand in bubble_ids:
                            continue
                        if cand in claimed:
                            continue
                        if not _endpoint_candidate_allowed(cand):
                            continue
                        if not self._is_plausible_dimension(cand):
                            continue
                        if strict_rich_only and not _rich_endpoint_dimension(cand):
                            continue
                        votes[cand] += 1
                        confidences[cand] = max(confidences.get(cand, 0.0), float(conf))

            _collect_from_center(int(ex), int(ey))
            if not any(count >= 2 for count in votes.values()):
                for cx, cy in self._local_angle_crop_centers(b, bi)[:2]:
                    _collect_from_center(int(cx), int(cy))
                    if any(count >= 2 for count in votes.values()):
                        break

            if not votes:
                if trace is not None:
                    trace["endpoint_ocr_status"] = "no_votes"
                continue

            def _rank(item: Tuple[str, int]) -> Tuple[int, int, float, int]:
                text, count = item
                has_decimal = 1 if re.search(r"\d+\.\d+", text) else 0
                return (count, has_decimal, confidences.get(text, 0.0), len(text))

            best_text, best_votes = max(votes.items(), key=_rank)
            best_conf = confidences.get(best_text, 0.0)
            required_conf = 0.90 if strict_rich_only else 0.85
            if best_votes < 2 and best_conf < required_conf:
                if trace is not None:
                    trace["endpoint_ocr_status"] = "weak_votes"
                continue

            reason = getattr(b, "review_reason", "") or ""
            self._apply_assignment_candidate(
                b,
                best_text,
                max(float(getattr(b, "confidence", 0.0) or 0.0), min(0.74, best_conf)),
                ((reason + "+") if reason else "")
                + ("leader_endpoint_ocr_strict" if strict_rich_only else "leader_endpoint_ocr"),
                review_threshold=0.86,
            )
            if trace is not None:
                trace["endpoint_ocr_status"] = "assigned"
                trace["endpoint_ocr_text"] = best_text
                trace["endpoint_ocr_votes"] = int(best_votes)
            claimed.add(best_text)
            logger.info(
                "Leader endpoint OCR: bubble #%s <- %r",
                b.bubble_number,
                b.dimension,
            )

    def _assign_coordinate_leader_callouts(
        self,
        bubbles: List[BubbleResult],
    ) -> None:
        """Assign coordinate labels only when a leader explicitly targets them."""
        tokens = getattr(self, "_norm_tokens", []) or []
        coordinate_candidates: List[Tuple[str, float, float, float, float]] = []

        def _is_coordinate_label(text: str) -> bool:
            t = re.sub(r"\s+", "", (text or "").strip().upper())
            return bool(
                re.search(r"\d", t)
                or re.fullmatch(r"[+-]?[XY]", t)
                or re.fullmatch(r"[XY][+-]?", t)
            )

        def _normalize_axis_label(text: str) -> Optional[str]:
            t = re.sub(r"\s+", "", (text or "").strip().upper())
            t = t.replace("−", "-").replace("–", "-").replace("—", "-")
            m = re.fullmatch(r"(?:\d{1,3}:)?([+-][XY])", t)
            if m:
                return m.group(1)
            return None

        seen_coordinate_boxes: Set[Tuple[str, int, int, int, int]] = set()
        for t in tokens:
            text = getattr(t, "text", "") or getattr(t, "raw_text", "") or ""
            if getattr(t, "semantic_type", "") != "coordinate":
                continue
            if not _is_coordinate_label(text):
                continue
            key = (
                text.strip(),
                int(round(float(t.x1))),
                int(round(float(t.y1))),
                int(round(float(t.x2))),
                int(round(float(t.y2))),
            )
            if key in seen_coordinate_boxes:
                continue
            seen_coordinate_boxes.add(key)
            coordinate_candidates.append((
                text.strip(),
                float(t.x1),
                float(t.y1),
                float(t.x2),
                float(t.y2),
            ))

        for cg in getattr(self, "_callout_groups", []) or []:
            text = getattr(cg, "text", "") or ""
            if (getattr(cg, "callout_type", "") or "").lower() != "coordinate":
                continue
            if not _is_coordinate_label(text):
                continue
            key = (
                text.strip(),
                int(round(float(cg.x1))),
                int(round(float(cg.y1))),
                int(round(float(cg.x2))),
                int(round(float(cg.y2))),
            )
            if key in seen_coordinate_boxes:
                continue
            seen_coordinate_boxes.add(key)
            coordinate_candidates.append((
                text.strip(),
                float(cg.x1),
                float(cg.y1),
                float(cg.x2),
                float(cg.y2),
            ))

        if not coordinate_candidates:
            return

        for bi, b in enumerate(bubbles):
            dim = (b.dimension or "").strip()
            has_dim = bool(dim and dim != "NO_DIMENSION")
            trace = self._trace_for_bubble(b, bi)
            path = (trace or {}).get("path") or []
            if not path:
                continue
            ex, ey = map(float, path[-1])
            br = max(1.0, float(b.radius or 1))
            best = None
            best_dist = float("inf")
            for text, x1, y1, x2, y2 in coordinate_candidates:
                dist = self._point_to_box_distance(
                    ex, ey,
                    x1, y1, x2, y2,
                )
                if dist > max(36.0, br * 1.25):
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best = text
            if best is None:
                continue
            if has_dim:
                weak_numeric_fragment = bool(
                    getattr(b, "needs_review", False)
                    and re.fullmatch(r"\d{1,2}", dim)
                    and best_dist <= max(30.0, br * 1.1)
                )
                if not weak_numeric_fragment:
                    continue

            reason = getattr(b, "review_reason", "") or ""
            self._apply_assignment_candidate(
                b,
                best,
                max(float(getattr(b, "confidence", 0.0) or 0.0), 0.60),
                ((reason + "+") if reason else "") + "coordinate_leader_assignment",
                review_threshold=0.86,
            )
            logger.info(
                "Coordinate leader assignment: bubble #%s <- %r",
                b.bubble_number,
                b.dimension,
            )

    def _correct_weak_assignments_from_trace_endpoint(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
    ) -> None:
        """Prefer the callout physically touched by the traced leader endpoint."""
        if not bubbles or not callout_groups:
            return

        risky_reasons = (
            "optimal_assign",
            "smart_assign_fallback",
            "proximity_fallback",
            "plain_numeric_leader_rescue",
            "low_confidence",
            "cleared_garbage_dim",
        )
        claimed = {
            (b.dimension or "").strip()
            for b in bubbles
            if (b.dimension or "").strip()
            and (b.dimension or "").strip() != "NO_DIMENSION"
        }
        bubble_ids = {
            str(bb.bubble_number or "").strip()
            for bb in bubbles
            if str(bb.bubble_number or "").strip()
        }

        def _rich_dimension_text(text: str) -> bool:
            t = (text or "").strip().upper()
            if not t:
                return False
            return bool(
                re.search(r"\d+[.,]\d+", t)
                or re.search(r"[Ã˜Î¦Â±Â°ÂºËšÃ—X/]", t)
                or re.search(r"\b(?:DIA|REF|TYP|THRU|M\d|R\d)\b", t)
            )

        def _candidate_rank_distance(text: str, endpoint_dist: float) -> float:
            t = (text or "").strip()
            rank = float(endpoint_dist)
            if _rich_dimension_text(t):
                rank -= 22.0
            if re.fullmatch(r"\d{1,3}", t):
                rank += 55.0
                if t in bubble_ids:
                    rank += 65.0
            return rank

        for bi, b in enumerate(bubbles):
            dim = (b.dimension or "").strip()
            if not dim or dim == "NO_DIMENSION":
                continue
            reason = getattr(b, "review_reason", "") or ""
            weak = (
                bool(getattr(b, "needs_review", False))
                or float(getattr(b, "confidence", 0.0) or 0.0) < 0.65
                or any(r in reason for r in risky_reasons)
            )
            if not weak:
                continue

            trace = self._trace_for_bubble(b, bi)
            path = (trace or {}).get("path") or []
            if not path:
                continue
            ex, ey = map(float, path[-1])
            br = max(1.0, float(b.radius or 1))
            endpoint_dist_from_bubble = math.hypot(ex - float(b.x), ey - float(b.y))
            if endpoint_dist_from_bubble <= br * 1.10:
                continue
            endpoint_trace_usable = endpoint_dist_from_bubble <= max(180.0, br * 6.0)

            current_boxes: List[Tuple[float, float, float, float]] = []
            candidates: List[Tuple[float, float, str, float]] = []
            for cg in callout_groups:
                text = (getattr(cg, "text", "") or "").strip()
                if not text:
                    continue
                if not self._is_assignable_callout(
                    cg,
                    allow_reference_only=True,
                    allow_keyword_only=False,
                ):
                    continue
                if not re.search(r"\d|[ØΦ±°º˚×x/]", text):
                    continue
                box = (float(cg.x1), float(cg.y1), float(cg.x2), float(cg.y2))
                endpoint_dist = self._point_to_box_distance(ex, ey, *box)
                if self._assignment_text_matches(dim, text):
                    current_boxes.append(box)
                    continue
                if text in claimed:
                    continue
                if text == str(b.bubble_number).strip():
                    continue
                if re.fullmatch(r"\d{1,3}", text) and text in bubble_ids:
                    continue
                if endpoint_trace_usable:
                    path_touches = any(
                        self._point_in_expanded_box(float(px), float(py), *box, max(4.0, br * 0.20))
                        for px, py in path[-8:]
                    )
                    gap_limit = (
                        max(90.0, br * 3.5)
                        if _rich_dimension_text(text)
                        else max(30.0, br * 1.15)
                    )
                    if not path_touches and endpoint_dist > gap_limit:
                        continue
                    quality = max(0.45, min(0.76, 0.76 - endpoint_dist / 260.0))
                    candidates.append((
                        _candidate_rank_distance(text, endpoint_dist),
                        endpoint_dist,
                        text,
                        quality,
                    ))
                else:
                    bubble_dist = self._point_to_box_distance(
                        float(b.x), float(b.y), *box
                    )
                    if bubble_dist > max(85.0, br * 3.2):
                        continue
                    quality = max(0.42, min(0.68, 0.68 - bubble_dist / 300.0))
                    candidates.append((bubble_dist, bubble_dist, text, quality))

            for tok in getattr(self, "_norm_tokens", []) or []:
                text = (getattr(tok, "text", "") or getattr(tok, "raw_text", "") or "").strip()
                if not text:
                    continue
                if getattr(tok, "semantic_type", "") == "suppressed":
                    continue
                if not re.search(r"\d|[ØΦ±°º˚×x/]", text):
                    continue
                if not self._is_plausible_dimension(text):
                    continue
                box = (float(tok.x1), float(tok.y1), float(tok.x2), float(tok.y2))
                endpoint_dist = self._point_to_box_distance(ex, ey, *box)
                if self._assignment_text_matches(dim, text):
                    current_boxes.append(box)
                    continue
                if text in claimed:
                    continue
                if text == str(b.bubble_number).strip():
                    continue
                if re.fullmatch(r"\d{1,3}", text) and text in bubble_ids:
                    continue
                if endpoint_trace_usable:
                    path_touches = any(
                        self._point_in_expanded_box(float(px), float(py), *box, max(4.0, br * 0.20))
                        for px, py in path[-8:]
                    )
                    gap_limit = (
                        max(90.0, br * 3.5)
                        if _rich_dimension_text(text)
                        else max(30.0, br * 1.15)
                    )
                    if not path_touches and endpoint_dist > gap_limit:
                        continue
                    quality = max(0.45, min(0.76, 0.76 - endpoint_dist / 260.0))
                    candidates.append((
                        _candidate_rank_distance(text, endpoint_dist),
                        endpoint_dist,
                        text,
                        quality,
                    ))
                else:
                    bubble_dist = self._point_to_box_distance(
                        float(b.x), float(b.y), *box
                    )
                    if bubble_dist > max(85.0, br * 3.2):
                        continue
                    quality = max(0.42, min(0.68, 0.68 - bubble_dist / 300.0))
                    candidates.append((bubble_dist, bubble_dist, text, quality))

            if not candidates:
                continue

            if endpoint_trace_usable:
                current_best = min(
                    (self._point_to_box_distance(ex, ey, *box) for box in current_boxes),
                    default=float("inf"),
                )
            else:
                current_best = min(
                    (
                        self._point_to_box_distance(float(b.x), float(b.y), *box)
                        for box in current_boxes
                    ),
                    default=float("inf"),
                )
            current_rank = (
                _candidate_rank_distance(dim, current_best)
                if math.isfinite(current_best) else float("inf")
            )
            candidates.sort(key=lambda item: (item[0], item[1], -len(item[2])))
            _rank_dist, best_dist, best_text, best_quality = candidates[0]
            if re.fullmatch(r"\d{1,3}", best_text) and _rank_dist > max(95.0, br * 4.0):
                continue
            if current_rank <= _rank_dist + max(8.0, br * 0.30):
                continue

            logger.info(
                "Endpoint correction: bubble #%s from %r to %r "
                "(endpoint_dist=%.1f current_dist=%.1f rank=%.1f current_rank=%.1f)",
                b.bubble_number,
                dim,
                best_text,
                best_dist,
                current_best,
                _rank_dist,
                current_rank,
            )
            claimed.discard(dim)
            self._apply_assignment_candidate(
                b,
                best_text,
                max(
                    float(getattr(b, "confidence", 0.0) or 0.0),
                    best_quality,
                ),
                ((reason + "+") if reason else "") + "trace_endpoint_correction",
                review_threshold=0.86,
            )
            claimed.add(best_text)

    def _local_angle_crop_centers(
        self,
        bubble: BubbleResult,
        bubble_index: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        bx, by = int(bubble.x), int(bubble.y)
        br = max(1, int(bubble.radius or 1))
        centers: List[Tuple[int, int]] = []
        trace = self._trace_for_bubble(bubble, bubble_index)
        path = (trace or {}).get("path") or []
        if path:
            ex, ey = path[-1]
            centers.append((int(ex), int(ey)))
            vx, vy = int(ex) - bx, int(ey) - by
            centers.append((int(ex + vx * 0.9), int(ey + vy * 0.9)))

        # Generic short-leader angle placements: right / down-right / down.
        centers.extend([
            (int(bx + br * 2.3), by),
            (int(bx + br * 2.3), int(by + br * 0.8)),
            (bx, int(by + br * 2.3)),
        ])

        h, w = self.image.shape[:2] if self.image is not None else (0, 0)
        unique: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for cx, cy in centers:
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            key = (int(round(cx / 20.0)), int(round(cy / 20.0)))
            if key in seen:
                continue
            seen.add(key)
            unique.append((cx, cy))
        return unique

    def _ocr_local_angle_crop(
        self,
        bubble: BubbleResult,
        cx: int,
        cy: int,
    ) -> List[Tuple[str, float]]:
        if self.image is None:
            return []
        br = max(1, int(bubble.radius or 1))
        h, w = self.image.shape[:2]
        pad_x = max(48, int(br * 2.4))
        pad_y = max(28, int(br * 1.2))
        x1 = max(0, cx - pad_x)
        y1 = max(0, cy - pad_y)
        x2 = min(w, cx + pad_x)
        y2 = min(h, cy + pad_y)
        if x2 <= x1 or y2 <= y1:
            return []
        crop = self.image[y1:y2, x1:x2]
        out: List[Tuple[str, float]] = []
        if not hasattr(self, "_local_angle_crop_cache"):
            self._local_angle_crop_cache = {}
        for scale in (2, 3):
            ck = (
                int(round(x1 / 12.0)) * 12,
                int(round(y1 / 12.0)) * 12,
                int(round(x2 / 12.0)) * 12,
                int(round(y2 / 12.0)) * 12,
                scale,
            )
            cached = self._local_angle_crop_cache.get(ck)
            if cached is not None:
                out.extend(cached)
                continue
            scale_out: List[Tuple[str, float]] = []
            try:
                up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                result = self.ocr(up)
            except Exception:
                self._local_angle_crop_cache[ck] = []
                continue
            items = result[0] if (
                isinstance(result, tuple) and len(result) >= 1
            ) else (result or [])
            for item in items or []:
                try:
                    text_info = item[1]
                    text = str(
                        text_info[0]
                        if isinstance(text_info, (list, tuple))
                        else text_info
                    ).strip()
                    conf = float(
                        text_info[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) > 1
                        else 0.9
                    )
                except Exception:
                    continue
                if text:
                    scale_out.append((text, conf))
            self._local_angle_crop_cache[ck] = scale_out
            out.extend(scale_out)
        return out

    def _ocr_rotated_local_leader_crop(
        self,
        bubble: BubbleResult,
        cx: int,
        cy: int,
    ) -> List[Tuple[str, float]]:
        """OCR a local leader endpoint crop after 90-degree rotation."""
        if self.image is None:
            return []
        br = max(1, int(round(float(bubble.radius or 1))))
        h, w = self.image.shape[:2]
        pad_x = max(52, int(br * 2.8))
        pad_y = max(90, int(br * 4.0))
        x1 = max(0, int(cx) - pad_x)
        y1 = max(0, int(cy) - pad_y)
        x2 = min(w, int(cx) + pad_x)
        y2 = min(h, int(cy) + pad_y)
        if x2 <= x1 or y2 <= y1:
            return []

        if not hasattr(self, "_local_rotated_leader_ocr_cache"):
            self._local_rotated_leader_ocr_cache = {}
        out: List[Tuple[str, float]] = []
        crop = self.image[y1:y2, x1:x2]
        variants = [
            cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        for rot_idx, variant in enumerate(variants):
            for scale in (2, 3):
                ck = (
                    int(round(x1 / 16.0)) * 16,
                    int(round(y1 / 16.0)) * 16,
                    int(round(x2 / 16.0)) * 16,
                    int(round(y2 / 16.0)) * 16,
                    rot_idx,
                    scale,
                )
                cached = self._local_rotated_leader_ocr_cache.get(ck)
                if cached is not None:
                    out.extend(cached)
                    continue
                scale_out: List[Tuple[str, float]] = []
                try:
                    up = cv2.resize(
                        variant,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    result = self.ocr(up)
                except Exception:
                    self._local_rotated_leader_ocr_cache[ck] = []
                    continue
                items = result[0] if (
                    isinstance(result, tuple) and len(result) >= 1
                ) else (result or [])
                for item in items or []:
                    try:
                        text_info = item[1]
                        text = str(
                            text_info[0]
                            if isinstance(text_info, (list, tuple))
                            else text_info
                        ).strip()
                        conf = float(
                            text_info[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) > 1
                            else 0.9
                        )
                    except Exception:
                        continue
                    if text:
                        scale_out.append((text, conf))
                self._local_rotated_leader_ocr_cache[ck] = scale_out
                out.extend(scale_out)
        return out

    def _assign_targeted_rotated_leader_ocr(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
    ) -> None:
        """Recover vertical dimensions from rotated OCR around leader endpoints."""
        if self.image is None or not bubbles:
            return
        from collections import Counter
        from ocr_rules import normalize_engineering_text

        claimed = {
            (b.dimension or "").strip()
            for b in bubbles
            if (b.dimension or "").strip()
            and (b.dimension or "").strip() != "NO_DIMENSION"
        }
        bubble_ids = {str(b.bubble_number or "").strip() for b in bubbles}

        for bi, b in enumerate(bubbles):
            dim = (b.dimension or "").strip()
            reason = getattr(b, "review_reason", "") or ""
            weak_dim = (
                not dim
                or dim == "NO_DIMENSION"
                or (
                    b.needs_review
                    and float(getattr(b, "confidence", 0.0) or 0.0) < 0.70
                    and (
                        re.fullmatch(r"\d{1,3}", dim or "") is not None
                        or "plain_numeric_leader_rescue" in reason
                    )
                )
            )
            if not weak_dim:
                continue
            trace = self._trace_for_bubble(b, bi)
            quality = float((trace or {}).get("quality_score") or 0.0)
            path = (trace or {}).get("path") or []
            if not path or quality < 0.58:
                continue
            ex, ey = path[-1]
            if not self._has_vertical_text_evidence_near(
                int(ex),
                int(ey),
                max(1, int(round(float(b.radius or 1)))),
            ) and not re.fullmatch(r"\d{1,3}", dim or ""):
                continue

            votes: Counter = Counter()
            confidences: Dict[str, float] = {}
            for cx, cy in [(int(ex), int(ey)), *self._local_angle_crop_centers(b, bi)[:2]]:
                for raw_text, conf in self._ocr_rotated_local_leader_crop(b, int(cx), int(cy)):
                    norm0 = normalize_engineering_text(raw_text)
                    for cand in self._local_rescue_text_candidates(norm0):
                        if not cand or cand in bubble_ids or cand in claimed:
                            continue
                        if not self._is_plausible_dimension(cand):
                            continue
                        if re.fullmatch(r"\d{1,2}", cand):
                            continue
                        votes[cand] += 1
                        confidences[cand] = max(confidences.get(cand, 0.0), float(conf))
            if not votes:
                continue

            best_text = self._decimal_cluster_consensus(
                votes,
                confidences,
            )

            def _rank(item: Tuple[str, int]) -> Tuple[int, int, float, int]:
                text, count = item
                decimal = 1 if re.search(r"\d+\.\d+", text) else 0
                return (count, decimal, confidences.get(text, 0.0), len(text))

            if best_text is None:
                best_text, best_votes = max(votes.items(), key=_rank)
            else:
                best_votes = votes.get(best_text, 1)
            best_conf = confidences.get(best_text, 0.0)
            if best_votes < 2 and best_conf < 0.82:
                continue
            self._apply_assignment_candidate(
                b,
                best_text,
                max(float(getattr(b, "confidence", 0.0) or 0.0), min(0.74, best_conf)),
                ((reason + "+") if reason else "") + "targeted_rotated_leader_ocr",
                review_threshold=0.86,
            )
            claimed.add(best_text)
            logger.info(
                "Targeted rotated leader OCR: bubble #%s <- %r",
                b.bubble_number,
                best_text,
            )

    def _has_vertical_text_evidence_near(self, cx: int, cy: int, radius: int) -> bool:
        """Cheap ink-shape gate for local rotated OCR near a leader endpoint."""
        if self.image is None:
            return False
        h, w = self.image.shape[:2]
        pad_x = max(36, int(radius * 1.8))
        pad_y = max(70, int(radius * 3.2))
        x1 = max(0, int(cx) - pad_x)
        y1 = max(0, int(cy) - pad_y)
        x2 = min(w, int(cx) + pad_x)
        y2 = min(h, int(cy) + pad_y)
        if x2 <= x1 or y2 <= y1:
            return False
        crop = self.image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        merged = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < 8:
                continue
            if bh < max(18, radius * 0.9):
                continue
            if bh / max(1, bw) >= 1.8:
                return True
        return False

    @staticmethod
    def _decimal_cluster_consensus(
        votes: Dict[str, int],
        confidences: Dict[str, float],
    ) -> Optional[str]:
        """Average tight decimal OCR variants such as 25.94/25.95/25.96."""
        clusters: Dict[Tuple[str, int], List[Tuple[str, float, int, float]]] = {}
        for text, count in votes.items():
            t = (text or "").strip()
            m = re.fullmatch(r"([RØΦ]?)(\d+)\.(\d+)(.*)", t)
            if not m:
                continue
            prefix, whole, frac, suffix = m.groups()
            if suffix:
                continue
            try:
                value = float(f"{whole}.{frac}")
            except ValueError:
                continue
            clusters.setdefault((prefix or "", len(frac)), []).append(
                (t, value, int(count), float(confidences.get(t, 0.0) or 0.0))
            )

        best: Optional[Tuple[int, float, str]] = None
        for (prefix, precision), items in clusters.items():
            if len(items) < 2:
                continue
            values = [value for _text, value, _count, _conf in items]
            if max(values) - min(values) > max(0.021, 2.1 * (10 ** -precision)):
                continue
            total_votes = sum(count for _text, _value, count, _conf in items)
            if total_votes < 3:
                continue
            weighted = sum(value * count for _text, value, count, _conf in items) / total_votes
            rounded = round(weighted, precision)
            candidate = f"{prefix}{rounded:.{precision}f}"
            score = (total_votes, max(conf for _text, _value, _count, conf in items))
            packed = (score[0], score[1], candidate)
            if best is None or packed > best:
                best = packed
        return best[2] if best is not None else None

    @staticmethod
    def _normalize_local_angle_candidate(text: str) -> Optional[str]:
        t = (text or "").strip().upper()
        if not t:
            return None
        has_degree_marker = bool(re.search(r"[°º˚]|Â°|Ã‚", t))
        if not has_degree_marker:
            return None
        t = t.replace("O", "0")
        t = t.replace("\u00ba", "\u00b0").replace("\u02da", "\u00b0")
        if re.search(r"[A-Z]", t):
            return None
        t = re.sub(r"[^0-9.,°Â]", "", t)
        t = t.replace(",", ".").replace("Â°", "°")
        if re.fullmatch(r"\d(?:\.\d)?°?", t):
            value = t[:-1] if t.endswith("°") else t
            try:
                numeric_value = float(value)
            except ValueError:
                return None
            if numeric_value <= 0.0:
                return None
            return f"{value}°"
        return None

    @staticmethod
    def _normalize_local_plain_numeric_candidate(text: str) -> Optional[str]:
        t = (text or "").strip().upper()
        if not t:
            return None
        t = t.replace("O", "0")
        if re.search(r"[A-Z]", t):
            return None
        if re.search(r"[./,Â°Ã‚°º°+\-]", t):
            return None
        t = re.sub(r"[^0-9]", "", t)
        if re.fullmatch(r"\d{1,3}", t) and t != "0":
            return t
        return None

    def _suppress_suspicious_bubbles(
        self,
        bubbles: List[BubbleResult],
    ) -> List[BubbleResult]:
        """Drop bubbles that match known false-positive patterns
        produced by the discovery pipeline on noisy inputs.

        Only inspects bubbles with `needs_review = True` — confident
        bubbles are always kept regardless of their dimension text,
        because the cost of dropping a real bubble is higher than the
        cost of leaving one false positive for the inspector to delete.

        Two rules:
          1. <b>Garbage alphabetic dimension</b> — after removing known
             engineering tokens (Ø, R, M, REF, TYP, MIN, MAX, THRU,
             ALL, DIA, mm, ±, °), if 3+ consecutive letters remain the
             dim is OCR garbage.
          2. <b>Label-is-fragment-of-dim</b> — bubble number is a
             substring of its dim text and the dim is a straight
             decimal extension of the label (e.g. label '59',
             dim '59.5'). Indicates the bubble label was OCR'd from
             the leading digits of the dim itself.
        """
        if not bubbles:
            return bubbles

        # Tokens we recognise as legitimate inside a dimension. After
        # stripping these, anything left over that's still mostly
        # alphabetic is garbage.
        ENG_TOKENS = (
            "REF", "TYP", "MIN", "MAX", "THRU", "ALL", "DIA",
            "CHAMFER", "MM", "EQ", "SP",
        )
        ENG_PREFIXES = ("Ø", "R", "M", "X", "°", "±")

        def is_alphabetic_garbage(dim: str) -> bool:
            if not dim:
                return False
            up = dim.upper()
            for tok in ENG_TOKENS:
                up = up.replace(tok, " ")
            for ch in ENG_PREFIXES:
                up = up.replace(ch.upper(), " ")
            # Remove digits, decimal points, slashes, parens, spaces
            up = re.sub(r"[\d./()\-+\s,]", "", up)
            # Anything left of length >= 3 is junk
            return len(up) >= 3

        def is_label_fragment_of_dim(label: str, dim: str) -> bool:
            if not label or not dim:
                return False
            label = label.strip()
            dim = dim.strip()
            if len(label) < 2 or label == dim:
                return False
            # Direct substring + decimal-expansion match: catches
            # label="59" inside dim="59.5".
            if label in dim and (
                re.fullmatch(rf"{re.escape(label)}\.\d+", dim) or
                re.fullmatch(rf"{re.escape(label)}\d{{0,1}}", dim)
            ):
                return True
            # Decimal-dropped match: catches label="15" picked up
            # from dim="1.500" — OCR drops the dot from a single
            # decimal number and the concatenated digits get
            # classified as a bubble number. Restricted to dims
            # shaped exactly "<int>.<frac>" so we don't break real
            # bubbles whose label happens to be a digit-prefix of a
            # complex dim (e.g. label="10" + dim="10.0 / 9.8" is a
            # legitimate tolerance-pair callout, not OCR garbage).
            m = re.fullmatch(r"(\d+)\.(\d+)", dim)
            if not m:
                return False
            int_part, dec_part = m.group(1), m.group(2)
            return (
                len(label) > len(int_part)
                and (int_part + dec_part).startswith(label)
            )

        # Pre-compute the set of dim texts that came out of the
        # callout grouper. The grouper applies structural validation
        # (thread / diameter / chamfer / tolerance / radius / angle
        # patterns, plus vertical-stack and paren-grouped notes) at
        # build time. ANY text that survived that grouping is by
        # definition structural engineering content — the garbage
        # filter must not second-guess it.
        #
        # We exempt all callout-grouper outputs, regardless of
        # callout_type, EXCEPT bare singletons whose semantic_type is
        # explicitly "numeric" (single OCR token, no merge, no
        # structural validation — those go through the filter as
        # before).
        structured_texts: set = set()
        for cg in getattr(self, "_callout_groups", None) or []:
            text = (cg.text or "").strip()
            if not text:
                continue
            kind = cg.callout_type or ""
            # Multi-token groups (chamfer / thread / diameter / stacked /
            # tolerance / keyword) come from positive structural matches
            # and are always exempt. A "numeric" type with only one
            # token IS just a bare reading — keep it under the filter
            # so genuine OCR garbage that happens to merge with a digit
            # still gets cleared.
            n_tokens = len(getattr(cg, "tokens", []) or [])
            if n_tokens > 1 or kind not in ("numeric", "keyword"):
                structured_texts.add(text)
            else:
                # Single-token "numeric" callouts: exempt only if the
                # text contains both a digit AND at least one engineering
                # symbol (Ø/R/M/±/°/% or a slash separator). That's
                # enough to call it structural without trusting the
                # alphabetic residue.
                if re.search(r"\d", text) and re.search(
                    r"[ØΦ±°%×x/]|\bR\d|\bM\d", text
                ):
                    structured_texts.add(text)

        def _is_structured(dim_text: str) -> bool:
            d = (dim_text or "").strip()
            if not d:
                return False
            return d in structured_texts

        def _supported_edge_recovery(bubble: BubbleResult) -> bool:
            if self.image is None:
                return False
            h, w = self.image.shape[:2]
            cx = int(round(float(bubble.x)))
            cy = int(round(float(bubble.y)))
            r = max(1, int(round(float(bubble.radius or 1))))
            edge_clipped = (
                cx - r <= 2 or cy - r <= 2
                or cx + r >= w - 3 or cy + r >= h - 3
            )
            if not edge_clipped:
                return False
            try:
                evidence = float(self._compute_bubble_evidence(self.image, cx, cy, r))
            except Exception:
                evidence = 0.0
            return (
                evidence >= 0.08
                and self._circle_has_annotation_tint(
                    self.image, cx, cy, r, strict=False
                )
            )

        kept = []
        for b in bubbles:
            if not b.needs_review:
                kept.append(b)
                continue
            dim = b.dimension or ""
            # When a bubble has a maroon-coloured rim on the ORIGINAL
            # image, the magenta annotation circle is physically there.
            # That's strong evidence the bubble itself is real — only
            # the dim assignment is wrong. In that case CLEAR the bad
            # dim and KEEP the bubble (so the inspector sees a numbered
            # bubble awaiting a manual dim) rather than dropping it.
            has_real_rim = (
                not self._low_chroma_in_original(b)
                or _supported_edge_recovery(b)
                or self._has_trusted_colored_bubble_evidence(b)
            )
            # Skip the alphabetic-garbage filter for dim texts that
            # came from a typed callout group — those passed structural
            # validation (thread/diameter/chamfer pattern match) and
            # can't be garbage even if they contain "junk-looking"
            # letter clusters like the 'J' in 'MJ5x0.8' or the 'h' in
            # the '4h6h' tolerance class.
            if is_alphabetic_garbage(dim) and not _is_structured(dim):
                if has_real_rim:
                    logger.info("Clear bubble #%s dim (garbage=%r), keeping bubble",
                                b.bubble_number, dim)
                    b.dimension = "NO_DIMENSION"
                    b.review_reason = (b.review_reason or "") + "+cleared_garbage_dim"
                    kept.append(b)
                    continue
                logger.info("Suppress bubble #%s: alphabetic garbage dim=%r",
                            b.bubble_number, dim)
                continue
            if is_label_fragment_of_dim(b.bubble_number, dim):
                if has_real_rim:
                    logger.info("Clear bubble #%s dim (label-fragment=%r), keeping bubble",
                                b.bubble_number, dim)
                    b.dimension = "NO_DIMENSION"
                    b.review_reason = (b.review_reason or "") + "+cleared_fragment_dim"
                    kept.append(b)
                    continue
                logger.info("Suppress bubble #%s: label is fragment of dim=%r",
                            b.bubble_number, dim)
                continue
            if (
                self._low_chroma_in_original(b)
                and not _supported_edge_recovery(b)
                and not self._has_trusted_colored_bubble_evidence(b)
            ):
                logger.info("Suppress bubble #%s: location is B&W on the "
                            "original image (dim=%r)", b.bubble_number, dim)
                continue
            kept.append(b)
        return kept

    def _suppress_ui_table_artifact_bubbles(
        self,
        bubbles: List[BubbleResult],
    ) -> List[BubbleResult]:
        """Remove bubbles that come from application UI/result-grid overlays.

        Some inputs are screenshots of the inspection application rather than
        raw drawings. Result grids contain colored row values and headers that
        look like annotation text, but they are not drawing balloons. Detect
        those UI bands from column-header vocabulary and suppress only bubbles
        whose centres fall inside the band.
        """
        tokens = getattr(self, "_norm_tokens", []) or []
        if not tokens or self.image is None:
            return bubbles

        h, w = self.image.shape[:2]
        ui_terms = {
            "NUMBER", "FEATURE", "MIN", "MAX", "NOMINAL", "MEAN",
            "TOLERANCE", "TYPE", "SHOW", "SHOWLI", "DELETE",
        }
        ui_action_terms = {"SHOW", "SHOWLI", "DELETE"}
        panel_terms = {
            "NEXT", "NUMBER", "SHOW", "LINE", "FONT", "SIZE", "SET",
            "COLOR", "ZOOM", "PAGE", "DEFAULT", "TOL",
        }
        panel_anchor_terms = {"NEXT", "FONT", "ZOOM", "PAGE", "DEFAULT", "COLOR"}

        header_tokens = []
        panel_tokens = []
        for t in tokens:
            raw = (getattr(t, "text", "") or "").upper()
            words = set(re.findall(r"[A-Z]+", raw))
            if not words:
                continue
            matched = {
                term for term in ui_terms
                if any(word.startswith(term) or term.startswith(word) for word in words)
            }
            if matched:
                header_tokens.append((float(t.cy), float(t.x1), float(t.x2), matched))
            panel_matched = {
                term for term in panel_terms
                if any(word.startswith(term) or term.startswith(word) for word in words)
            }
            if panel_matched:
                panel_tokens.append((
                    float(t.cx), float(t.cy),
                    float(t.x1), float(t.y1), float(t.x2), float(t.y2),
                    panel_matched,
                ))

        bands: List[Tuple[float, float]] = []
        if len(header_tokens) >= 4:
            header_tokens.sort(key=lambda item: item[0])
            for cy, _x1, _x2, _matched in header_tokens:
                cluster = [
                    item for item in header_tokens
                    if abs(item[0] - cy) <= 22.0
                ]
                matched_terms = set().union(*(item[3] for item in cluster))
                if len(matched_terms) < 4 or not (matched_terms & ui_action_terms):
                    continue
                span_x1 = min(item[1] for item in cluster)
                span_x2 = max(item[2] for item in cluster)
                if (span_x2 - span_x1) < w * 0.35:
                    continue
                band_top = max(0.0, min(item[0] for item in cluster) - 35.0)
                bands.append((band_top, float(h)))

        panel_regions: List[Tuple[float, float, float, float]] = []
        if len(panel_tokens) >= 4:
            for seed in panel_tokens:
                sx, sy = seed[0], seed[1]
                cluster = [
                    item for item in panel_tokens
                    if abs(item[0] - sx) <= max(220.0, w * 0.18)
                    and abs(item[1] - sy) <= max(260.0, h * 0.30)
                ]
                if len(cluster) < 4:
                    continue
                matched_terms = set().union(*(item[6] for item in cluster))
                if len(matched_terms) < 4 or not (matched_terms & panel_anchor_terms):
                    continue
                x1 = min(item[2] for item in cluster)
                y1 = min(item[3] for item in cluster)
                x2 = max(item[4] for item in cluster)
                y2 = max(item[5] for item in cluster)
                span_x = x2 - x1
                span_y = y2 - y1
                if span_x > w * 0.45 or span_y < 45:
                    continue
                region = (
                    max(0.0, x1 - 25.0),
                    max(0.0, y1 - 25.0),
                    min(float(w), x2 + max(90.0, w * 0.08)),
                    min(float(h), y2 + 35.0),
                )
                if not any(
                    abs(region[0] - ex[0]) < 8.0
                    and abs(region[1] - ex[1]) < 8.0
                    and abs(region[2] - ex[2]) < 8.0
                    and abs(region[3] - ex[3]) < 8.0
                    for ex in panel_regions
                ):
                    panel_regions.append(region)

        if not bands and not panel_regions:
            return bubbles

        kept: List[BubbleResult] = []
        for b in bubbles:
            if any(y1 <= float(b.y) <= y2 for y1, y2 in bands):
                logger.info(
                    "Suppress bubble #%s inside UI/table overlay band at y=%s",
                    b.bubble_number,
                    b.y,
                )
                continue
            if any(
                x1 <= float(b.x) <= x2 and y1 <= float(b.y) <= y2
                for x1, y1, x2, y2 in panel_regions
            ):
                logger.info(
                    "Suppress bubble #%s inside UI/control overlay panel at (%s,%s)",
                    b.bubble_number,
                    b.x,
                    b.y,
                )
                continue
            kept.append(b)
        return kept

    def _suppress_tiny_text_artifact_bubbles(
        self,
        bubbles: List[BubbleResult],
    ) -> List[BubbleResult]:
        """Remove tiny OCR-text candidates that lack circular rim evidence.

        Pixel size alone is not enough because screenshots can shrink genuine
        balloons. The production rule is therefore conditional: if a candidate
        is text-sized, it must show a real circular annotation/dark rim around
        the number. Otherwise it is just OCR text promoted to a bubble.
        """
        if self.image is None or not bubbles:
            return bubbles

        radii = [float(max(1, b.radius)) for b in bubbles]
        median_r = float(np.median(radii)) if radii else 0.0
        tiny_abs = 10.0
        tiny_rel = median_r * 0.45 if median_r >= 18.0 else 0.0
        tiny_limit = max(tiny_abs, tiny_rel)

        kept: List[BubbleResult] = []
        for bi, b in enumerate(bubbles):
            r = float(max(1, b.radius))
            if r > tiny_limit:
                kept.append(b)
                continue

            evidence = self._compute_bubble_evidence(self.image, int(b.x), int(b.y), int(r))
            has_trace = False
            trace = self._trace_for_bubble(b, bi)
            if trace and trace.get("path"):
                path = trace["path"]
                if len(path) >= max(8, int(r * 1.2)):
                    ex, ey = path[-1]
                    has_trace = math.hypot(ex - b.x, ey - b.y) >= r * 0.75

            # Keep tiny candidates only when the circle itself is visible, or
            # when a short but real leader trace supports it. This preserves
            # genuinely small zoomed-out balloons while removing bare number
            # text from tables/UI overlays.
            if evidence >= 0.16 or (evidence >= 0.10 and has_trace):
                kept.append(b)
                continue

            logger.info(
                "Suppress bubble #%s as tiny text artifact r=%.1f evidence=%.3f",
                b.bubble_number,
                r,
                evidence,
            )
        return kept

    def _suppress_non_circular_bubble_artifacts(
        self,
        bubbles: List[BubbleResult],
    ) -> List[BubbleResult]:
        """Remove digit candidates that do not have a circular annotation rim.

        Assignment quality is downstream from bubble validity: a row value or
        UI dropdown may sit near dimension-looking text, but it is still not a
        balloon unless its perimeter contains colored circular annotation ink.
        This final gate uses only local shape/color evidence. It does not know
        filenames, expected labels, or expected dimensions.
        """
        if self.image is None or not bubbles:
            return bubbles

        mask = self._annotation_hsv_mask(self.image)
        if int(np.count_nonzero(mask)) < 50:
            return bubbles

        kept: List[BubbleResult] = []
        for bi, b in enumerate(bubbles):
            r = max(1, int(round(float(b.radius or 1))))
            cx, cy = int(round(float(b.x))), int(round(float(b.y)))
            ring_score = self._annotation_ring_score(mask, cx, cy, r)
            evidence = self._compute_bubble_evidence(self.image, cx, cy, r)
            h, w = self.image.shape[:2]
            edge_clipped = (
                cx - r <= 2 or cy - r <= 2
                or cx + r >= w - 3 or cy + r >= h - 3
            )
            has_dim = bool(b.dimension) and b.dimension != "NO_DIMENSION"
            reason = getattr(b, "review_reason", "") or ""

            trace = self._trace_for_bubble(b, bi)
            has_leader_trace = False
            if trace and trace.get("path"):
                path = trace["path"]
                if len(path) >= max(8, int(r * 1.4)):
                    ex, ey = path[-1]
                    has_leader_trace = math.hypot(ex - cx, ey - cy) >= r * 1.2

            passes = (
                ring_score >= 0.10
                or (r <= 10 and ring_score >= 0.07 and evidence >= 0.12)
                or (has_leader_trace and ring_score >= 0.06 and evidence >= 0.14)
                or (
                    ring_score >= 0.05
                    and evidence >= 0.12
                    and self._circle_has_annotation_tint(self.image, cx, cy, r, strict=False)
                )
                or (
                    evidence >= 0.20
                    and self._circle_has_annotation_tint(self.image, cx, cy, r, strict=False)
                    and re.fullmatch(r"\d{1,2}[A-Za-z]?", str(b.bubble_number or "").strip())
                )
                or (
                    self._has_matching_maroon_label_token(b)
                    and evidence >= 0.25
                    and self._circle_has_annotation_tint(self.image, cx, cy, r, strict=False)
                )
                or (
                    edge_clipped
                    and has_dim
                    and evidence >= 0.08
                    and self._circle_has_annotation_tint(self.image, cx, cy, r, strict=False)
                )
                or (
                    edge_clipped
                    and "edge_blob_recovery" in reason
                    and evidence >= 0.08
                    and self._circle_has_annotation_tint(self.image, cx, cy, r, strict=False)
                )
                or (
                    (has_dim or self._has_matching_maroon_label_token(b))
                    and self._has_trusted_colored_bubble_evidence(b)
                )
            )
            if passes:
                kept.append(b)
                continue

            logger.info(
                "Suppress bubble #%s: insufficient circular annotation rim "
                "(ring=%.3f evidence=%.3f r=%s)",
                b.bubble_number,
                ring_score,
                evidence,
                r,
            )

        return kept

    def _has_matching_maroon_label_token(self, bubble: BubbleResult) -> bool:
        label = str(bubble.bubble_number or "").strip()
        if not label:
            return False
        bx, by = float(bubble.x), float(bubble.y)
        br = max(1.0, float(bubble.radius or 1))
        for tok in getattr(self, "_norm_tokens", []) or []:
            if not getattr(tok, "is_maroon", False):
                continue
            text = (getattr(tok, "text", "") or getattr(tok, "raw_text", "") or "").strip()
            if not is_bubble_token(text):
                continue
            norm = normalize_bubble_value(text)
            if norm != label:
                continue
            if math.hypot(float(tok.cx) - bx, float(tok.cy) - by) <= max(45.0, br * 1.25):
                return True
        return False

    def _has_trusted_colored_bubble_evidence(self, bubble: BubbleResult) -> bool:
        """Return True when local colored-ring evidence supports a bubble.

        Recovery contours can include a leader segment, so their fitted radius
        may be larger than the visible balloon.  This helper scans nearby
        radii, including the matching maroon label centre when OCR found it,
        before deciding that a candidate lacks a real annotation rim.
        """
        if self.image is None:
            return False
        label = str(bubble.bubble_number or "").strip()
        if not label:
            return False

        mask = self._annotation_hsv_mask(self.image)
        if int(np.count_nonzero(mask)) < 50:
            return False

        bx, by = float(bubble.x), float(bubble.y)
        br = max(8.0, float(bubble.radius or 1))
        centers: List[Tuple[int, int]] = [(int(round(bx)), int(round(by)))]
        has_matching_label = False
        for tok in getattr(self, "_norm_tokens", []) or []:
            if not getattr(tok, "is_maroon", False):
                continue
            text = (getattr(tok, "text", "") or getattr(tok, "raw_text", "") or "").strip()
            if not is_bubble_token(text):
                continue
            if normalize_bubble_value(text) != label:
                continue
            if math.hypot(float(tok.cx) - bx, float(tok.cy) - by) > max(55.0, br * 1.35):
                continue
            centers.append((int(round(float(tok.cx))), int(round(float(tok.cy)))))
            has_matching_label = True

        radii = {
            max(6, int(round(br * factor)))
            for factor in (0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.08)
        }
        best_ring = 0.0
        best_evidence = 0.0
        for cx, cy in centers:
            for rr in sorted(radii):
                ring = float(self._annotation_ring_score(mask, cx, cy, rr))
                evidence = float(self._compute_bubble_evidence(self.image, cx, cy, rr))
                best_ring = max(best_ring, ring)
                best_evidence = max(best_evidence, evidence)
                if (
                    ring >= 0.045
                    and evidence >= 0.10
                    and self._circle_has_annotation_tint(self.image, cx, cy, rr, strict=False)
                ):
                    return True
                if (
                    has_matching_label
                    and ring >= 0.020
                    and evidence >= 0.08
                    and self._circle_has_annotation_tint(self.image, cx, cy, rr, strict=False)
                ):
                    return True

        return bool(has_matching_label and best_ring >= 0.015 and best_evidence >= 0.12)

    def _suppress_overlapping_bubble_candidates(
        self,
        bubbles: List[BubbleResult],
    ) -> List[BubbleResult]:
        """Remove weaker candidates that claim the same physical balloon.

        Multiple OCR/discovery paths can create two bubble results over the
        same rim: one from the true center digit and another from nearby
        dimension text or a leader fragment. Real balloon circles do not
        overlap this tightly, so we keep the candidate with stronger local
        circular evidence and suppress the rest.
        """
        if self.image is None or len(bubbles) < 2:
            return bubbles

        try:
            mask = self._annotation_hsv_mask(self.image)
        except Exception:
            mask = None

        evidence_cache: Dict[int, Tuple[float, float, float]] = {}

        def candidate_score(idx: int) -> Tuple[float, float, float]:
            if idx in evidence_cache:
                return evidence_cache[idx]
            b = bubbles[idx]
            cx = int(round(float(b.x)))
            cy = int(round(float(b.y)))
            r = max(1, int(round(float(b.radius or 1))))
            try:
                evidence = float(self._compute_bubble_evidence(self.image, cx, cy, r))
            except Exception:
                evidence = 0.0
            try:
                ring = (
                    float(self._annotation_ring_score(mask, cx, cy, r))
                    if mask is not None else 0.0
                )
            except Exception:
                ring = 0.0
            label = str(b.bubble_number or "").strip()
            label_bias = -0.015 * max(0, len(label) - 1)
            score = ring * 1.5 + evidence + label_bias
            evidence_cache[idx] = (score, ring, evidence)
            return evidence_cache[idx]

        suppressed: Set[int] = set()
        n = len(bubbles)
        for i in range(n):
            if i in suppressed:
                continue
            bi = bubbles[i]
            for j in range(i + 1, n):
                if j in suppressed:
                    continue
                bj = bubbles[j]
                ri = max(1.0, float(bi.radius or 1))
                rj = max(1.0, float(bj.radius or 1))
                center_dist = math.hypot(float(bi.x) - float(bj.x),
                                         float(bi.y) - float(bj.y))
                if center_dist > min(ri, rj) * 1.9:
                    continue
                overlap_ratio = (ri + rj - center_dist) / max(ri, rj)
                if overlap_ratio < 0.25:
                    continue

                score_i, ring_i, evidence_i = candidate_score(i)
                score_j, ring_j, evidence_j = candidate_score(j)
                if abs(score_i - score_j) < 0.06:
                    li = len(str(bi.bubble_number or "").strip())
                    lj = len(str(bj.bubble_number or "").strip())
                    loser = j if li <= lj else i
                else:
                    loser = j if score_i >= score_j else i

                winner = i if loser == j else j
                self._transfer_overlap_assignment_if_supported(
                    bubbles[winner], bubbles[loser]
                )
                suppressed.add(loser)
                lb = bubbles[loser]
                logger.info(
                    "Suppress bubble #%s: overlaps stronger candidate "
                    "(ring=%.3f evidence=%.3f)",
                    lb.bubble_number,
                    ring_j if loser == j else ring_i,
                    evidence_j if loser == j else evidence_i,
                )
                if loser == i:
                    break

        if not suppressed:
            return bubbles
        return [b for idx, b in enumerate(bubbles) if idx not in suppressed]

    def _transfer_overlap_assignment_if_supported(
        self,
        winner: BubbleResult,
        loser: BubbleResult,
    ) -> None:
        """Move a valid dimension from a weaker duplicate to the kept bubble."""
        winner_dim = (winner.dimension or "").strip()
        if winner_dim and winner_dim != "NO_DIMENSION":
            return
        loser_dim = (loser.dimension or "").strip()
        if not loser_dim or loser_dim == "NO_DIMENSION":
            return

        boxes: List[Tuple[float, float, float, float]] = []
        for cg in getattr(self, "_callout_groups", []) or []:
            if not self._assignment_text_matches(loser_dim, getattr(cg, "text", "")):
                continue
            if not self._is_assignable_callout(cg, allow_reference_only=True):
                continue
            boxes.append((float(cg.x1), float(cg.y1), float(cg.x2), float(cg.y2)))

        for tok in getattr(self, "_norm_tokens", []) or []:
            text = getattr(tok, "text", "") or getattr(tok, "raw_text", "")
            if not self._assignment_text_matches(loser_dim, text):
                continue
            if getattr(tok, "semantic_type", "") == "suppressed":
                continue
            boxes.append((float(tok.x1), float(tok.y1), float(tok.x2), float(tok.y2)))

        if boxes and not self._assignment_has_geometry_support(winner, boxes):
            return

        reason = getattr(winner, "review_reason", "") or ""
        self._apply_assignment_candidate(
            winner,
            loser_dim,
            min(
                0.70,
                max(float(getattr(winner, "confidence", 0.0) or 0.0),
                    float(getattr(loser, "confidence", 0.0) or 0.0) * 0.9),
            ),
            ((reason + "+") if reason else "") + "overlap_assignment_transfer",
            review_threshold=0.99,
        )
        logger.info(
            "Transferred dimension %r from overlapping bubble #%s to #%s",
            loser_dim,
            loser.bubble_number,
            winner.bubble_number,
        )

    # Check chroma along the bubble's rim on the *original* (pre-
    # normalisation) image. Old version sampled a 30x30 box around
    # the centre — but a balloon's stroke is a thin maroon ring
    # with a WHITE interior, so the box averaged mostly-white
    # pixels and incorrectly reported low chroma on real maroon
    # balloons. New version samples 24 angular points on the rim
    # (with ±2 px radial offsets) and takes the MAX. If any rim
    # sample is chromatic, the bubble has annotation ink at that
    # location and is treated as real.
    def _low_chroma_in_original(self, b: BubbleResult,
                                threshold: float = 8.0) -> bool:
        orig = getattr(self, "_original_image", None)
        if orig is None or orig.ndim != 3:
            return False
        h, w = orig.shape[:2]
        cx, cy = int(b.x), int(b.y)
        r = max(10, int(b.radius) if b.radius else 25)
        max_chroma = 0.0
        for ang_deg in range(0, 360, 15):
            ang = math.radians(ang_deg)
            cos_a, sin_a = math.cos(ang), math.sin(ang)
            for dr in (-2, 0, 2):
                rr_ = r + dr
                px = cx + int(rr_ * cos_a)
                py = cy + int(rr_ * sin_a)
                if not (0 <= px < w and 0 <= py < h):
                    continue
                b_ = float(orig[py, px, 0])
                g_ = float(orig[py, px, 1])
                r_ = float(orig[py, px, 2])
                chroma = (abs(r_ - g_) + abs(g_ - b_) + abs(r_ - b_)) / 3.0
                if chroma > max_chroma:
                    max_chroma = chroma
        return max_chroma < threshold

    def _suppress_low_chroma_unmatched_pre_rescue(
        self, bubbles: List[BubbleResult]
    ) -> List[BubbleResult]:
        """Drop unmatched bubbles whose centre sits on near-B&W pixels
        in the original image. Runs BEFORE targeted rescue so spurious
        Hough circles in price-table / B&W regions can't claim a
        rescued dimension that rightfully belongs to a real bubble
        nearby. Gated to screenshot-normalisation mode — clean CAD
        exports don't suffer the moiré-induced false circles this
        rule targets, and a tighter chroma threshold would risk
        suppressing genuine bubbles whose ring chroma is muted on a
        clean print. Threshold of 11 was chosen because the new.jpeg
        real bubbles measured 13–25 on the original-image chroma
        scale, while the spurious #59 sat at 9.5 — leaving ~2 points
        of headroom against the weakest real bubble.
        """
        if not bubbles:
            return bubbles
        kept: List[BubbleResult] = []
        for b in bubbles:
            has_dim = b.dimension and b.dimension != "NO_DIMENSION"
            if has_dim:
                kept.append(b)
                continue
            if self._low_chroma_in_original(b, window=30, threshold=11.0):
                logger.info(
                    "Pre-rescue suppress bubble #%s: unmatched and low "
                    "chroma on original image (likely spurious circle)",
                    b.bubble_number,
                )
                continue
            kept.append(b)
        return kept

    @staticmethod
    def _assignment_match_key(text: str, *, loose: bool = False) -> str:
        """Normalize dimension text for assignment/evidence comparison."""
        s = (text or "").upper()
        replacements = {
            "\u00d8": "DIA",
            "\u03a6": "DIA",
            "\u2300": "DIA",
            "\u0172": "DIA",
            "Ã˜": "DIA",
            "\u00b0": "DEG",
            "\u00ba": "DEG",
            "\u02da": "DEG",
            "Â°": "DEG",
            "\u00b1": "PM",
            "Â±": "PM",
            "\u00d7": "X",
            "Ã—": "X",
        }
        for src, dst in replacements.items():
            s = s.replace(src, dst)
        s = re.sub(r"\s+", "", s)
        if loose:
            s = s.replace("DIA", "").replace("DEG", "")
            s = re.sub(r"[^A-Z0-9]", "", s)
        else:
            s = re.sub(r"[^A-Z0-9.+/\\-]", "", s)
        return s

    @staticmethod
    def _assignment_text_matches(expected: str, observed: str) -> bool:
        """Return True when two OCR strings can represent the same value."""
        a = BubbleDetector._assignment_match_key(expected)
        b = BubbleDetector._assignment_match_key(observed)
        if not a or not b:
            return False
        if a == b:
            return True
        if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
            return True

        la = BubbleDetector._assignment_match_key(expected, loose=True)
        lb = BubbleDetector._assignment_match_key(observed, loose=True)
        if not la or not lb:
            return False
        if la == lb:
            return True
        if len(la) >= 3 and len(lb) >= 3 and (la in lb or lb in la):
            return True

        da = re.sub(r"\D", "", la)
        db = re.sub(r"\D", "", lb)
        return bool(da and da == db and len(da) >= 3)

    @staticmethod
    def _point_to_box_distance(
        x: float,
        y: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        dx = max(float(x1) - x, 0.0, x - float(x2))
        dy = max(float(y1) - y, 0.0, y - float(y2))
        return math.hypot(dx, dy)

    @staticmethod
    def _point_in_expanded_box(
        x: float,
        y: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        pad: float,
    ) -> bool:
        return (
            float(x1) - pad <= x <= float(x2) + pad
            and float(y1) - pad <= y <= float(y2) + pad
        )

    def _trace_for_bubble(
        self,
        bubble: BubbleResult,
        bubble_index: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the physical trace for one bubble instance.

        Prefer index-keyed traces; fall back to the legacy label-keyed map for
        older call sites that do not yet pass an index.
        """
        if bubble_index is not None:
            trace = (getattr(self, "_seed_traces_by_index", {}) or {}).get(bubble_index)
            if trace is not None:
                return trace
        traces = getattr(self, "_seed_traces", {}) or {}
        return (
            traces.get(str(bubble.bubble_number))
            or traces.get(bubble.bubble_number)
        )

    def _trace_supports_box(
        self,
        bubble: BubbleResult,
        box: Tuple[float, float, float, float],
    ) -> bool:
        """Check whether a traced leader physically supports a box target."""
        trace = self._trace_for_bubble(bubble)
        if not trace or not trace.get("path"):
            return False
        path = trace["path"]
        if len(path) < 2:
            return False

        bx, by = float(bubble.x), float(bubble.y)
        br = max(1.0, float(bubble.radius or 1))
        x1, y1, x2, y2 = box
        pad = max(4.0, br * 0.20)

        for px, py in path:
            if self._point_in_expanded_box(float(px), float(py), x1, y1, x2, y2, pad):
                return True

        ex, ey = map(float, path[-1])
        endpoint_dist = math.hypot(ex - bx, ey - by)
        if endpoint_dist < max(8.0, br * 1.05):
            return False

        edge_dist = self._point_to_box_distance(ex, ey, x1, y1, x2, y2)
        near_limit = max(48.0, br * 2.75)
        if edge_dist > near_limit:
            return False

        tx, ty = ex - bx, ey - by
        tn = math.hypot(tx, ty)
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        vx, vy = cx - ex, cy - ey
        vn = math.hypot(vx, vy)
        if tn <= 1e-6 or vn <= 1e-6:
            return True
        return ((tx / tn) * (vx / vn) + (ty / tn) * (vy / vn)) >= 0.20

    def _box_has_local_support(
        self,
        bubble: BubbleResult,
        box: Tuple[float, float, float, float],
    ) -> bool:
        """Accept very local, axis-aligned callouts even without trace data."""
        bx, by = float(bubble.x), float(bubble.y)
        br = max(1.0, float(bubble.radius or 1))
        x1, y1, x2, y2 = box
        dist = self._point_to_box_distance(bx, by, x1, y1, x2, y2)
        if dist > max(55.0, br * 1.75):
            return False
        y_band = float(y1) <= by + br * 1.7 and float(y2) >= by - br * 1.7
        x_band = float(x1) <= bx + br * 1.7 and float(x2) >= bx - br * 1.7
        return bool(x_band or y_band)

    def _assignment_has_geometry_support(
        self,
        bubble: BubbleResult,
        boxes: List[Tuple[float, float, float, float]],
    ) -> bool:
        for box in boxes:
            if self._trace_supports_box(bubble, box):
                return True
            if self._box_has_local_support(bubble, box):
                return True
        return False

    def _validate_assignment_geometry(
        self,
        bubbles: List[BubbleResult],
        callout_groups: List[CalloutGroup],
    ) -> None:
        """Clear weak dimension links that are not backed by drawing geometry.

        This is deliberately output-boundary validation. Earlier assignment
        passes may use OCR rescue, proximity, or propagation so a missing value
        can be found, but the final published value must still be supported by
        either a traced leader endpoint or a very local callout box. If not, we
        emit NO_DIMENSION instead of a misleading assignment.
        """
        if not bubbles:
            return

        tokens = getattr(self, "_norm_tokens", None) or []
        risky_reasons = (
            "smart_assign_fallback",
            "local_rapidocr_rescue",
            "leader_attention_ocr",
            "shared_dim",
            "low_confidence",
            "optimal_assign",
        )

        for bi, b in enumerate(bubbles):
            dim = (b.dimension or "").strip()
            if not dim or dim == "NO_DIMENSION":
                continue

            reason = getattr(b, "review_reason", "") or ""
            weak_assignment = (
                bool(getattr(b, "needs_review", False))
                or float(getattr(b, "confidence", 0.0) or 0.0) < 0.80
                or any(r in reason for r in risky_reasons)
            )
            if not weak_assignment:
                continue

            boxes: List[Tuple[float, float, float, float]] = []
            for cg in callout_groups or []:
                if not self._assignment_text_matches(dim, getattr(cg, "text", "")):
                    continue
                if not self._is_assignable_callout(
                    cg,
                    allow_reference_only=True,
                    allow_keyword_only=True,
                ):
                    continue
                boxes.append((float(cg.x1), float(cg.y1), float(cg.x2), float(cg.y2)))

            for tok in tokens:
                text = getattr(tok, "text", "") or getattr(tok, "raw_text", "")
                if not self._assignment_text_matches(dim, text):
                    continue
                if getattr(tok, "semantic_type", "") == "suppressed":
                    continue
                boxes.append((float(tok.x1), float(tok.y1), float(tok.x2), float(tok.y2)))

            if boxes and self._assignment_has_geometry_support(b, boxes):
                continue

            if "local_angle_rescue" in reason and not boxes:
                continue

            if (
                "coordinate_leader_assignment" in reason
                and re.fullmatch(r"[+-][XY]", dim.upper())
            ):
                trace = self._trace_for_bubble(b, bi)
                path = (trace or {}).get("path") or []
                endpoint_axis_supported = False
                if path:
                    ex, ey = path[-1]
                    for raw_text, _conf in self._ocr_local_angle_crop(b, int(ex), int(ey)):
                        t = re.sub(r"\s+", "", (raw_text or "").strip().upper())
                        t = t.replace("−", "-").replace("–", "-").replace("—", "-")
                        m = re.fullmatch(r"(?:\d{1,3}:)?([+-][XY])", t)
                        if m and m.group(1) == dim.upper():
                            endpoint_axis_supported = True
                            break
                if endpoint_axis_supported:
                    b.needs_review = True
                    continue

            if (
                boxes
                and "local_rapidocr_rescue" in reason
                and self._assignment_has_bounded_local_rescue_support(b, boxes)
            ):
                continue

            if (
                boxes
                and self._strict_photo_mode()
                and self._has_matching_maroon_label_token(b)
                and self._assignment_has_bounded_local_rescue_support(b, boxes)
            ):
                continue

            if (
                boxes
                and self._has_trusted_colored_bubble_evidence(b)
                and self._assignment_has_bounded_local_rescue_support(b, boxes)
            ):
                continue

            # If the value was created by direct local OCR and no matching OCR
            # box exists, keep it review-marked. It may have come from a cropped
            # recognizer result that is not present in the full-page OCR graph.
            if not boxes and "local_rapidocr_rescue" in reason:
                b.needs_review = True
                continue

            if "plain_numeric_leader_rescue" in reason:
                b.needs_review = True
                continue

            replacement = self._find_supported_assignment_replacement(
                b, callout_groups, exclude_text=dim
            )
            if replacement is not None:
                new_text, new_score = replacement
                logger.info(
                    "Reassigned bubble #%s from unsupported %r to supported %r",
                    b.bubble_number,
                    dim,
                    new_text,
                )
                self._apply_assignment_candidate(
                    b,
                    new_text,
                    max(
                        float(getattr(b, "confidence", 0.0) or 0.0),
                        min(0.72, new_score),
                    ),
                    ((reason + "+") if reason else "") + "geometry_reassignment",
                    review_threshold=0.99,
                )
                continue

            logger.info(
                "Cleared unsupported assignment for bubble #%s: %r "
                "(reason=%s, boxes=%d)",
                b.bubble_number,
                dim,
                reason or "unknown",
                len(boxes),
            )
            b.dimension = "NO_DIMENSION"
            b.confidence = min(float(getattr(b, "confidence", 0.0) or 0.0), 0.20)
            b.needs_review = True
            b.review_reason = (
                (reason + "+") if reason else ""
            ) + "cleared_unsupported_assignment"

    def _assignment_has_bounded_local_rescue_support(
        self,
        bubble: BubbleResult,
        boxes: List[Tuple[float, float, float, float]],
    ) -> bool:
        bx, by = float(bubble.x), float(bubble.y)
        br = max(1.0, float(bubble.radius or 1))
        limit = max(150.0, br * 6.5)
        for box in boxes:
            if self._point_to_box_distance(bx, by, *box) <= limit:
                return True
        return False

    def _find_supported_assignment_replacement(
        self,
        bubble: BubbleResult,
        callout_groups: List[CalloutGroup],
        *,
        exclude_text: str,
    ) -> Optional[Tuple[str, float]]:
        """Find a geometry-supported callout to replace a rejected value."""
        best: Optional[Tuple[float, str, float]] = None
        for cg in callout_groups or []:
            text = (getattr(cg, "text", "") or "").strip()
            if not text or self._assignment_text_matches(exclude_text, text):
                continue
            if not self._is_assignable_callout(
                cg,
                allow_reference_only=True,
                allow_keyword_only=True,
            ):
                continue
            box = (float(cg.x1), float(cg.y1), float(cg.x2), float(cg.y2))
            if not self._assignment_has_geometry_support(bubble, [box]):
                continue

            trace = self._trace_for_bubble(bubble)
            if trace and trace.get("path"):
                ex, ey = trace["path"][-1]
                dist = self._point_to_box_distance(float(ex), float(ey), *box)
            else:
                dist = self._point_to_box_distance(float(bubble.x), float(bubble.y), *box)
            quality = max(0.35, min(0.72, 0.72 - dist / 500.0))
            candidate = (dist, text, quality)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            return None
        return best[1], best[2]

    def _proximity_fallback_for_empty_dims(
        self,
        bubbles: List[BubbleResult],
    ) -> None:
        """For bubbles that ended the pipeline with no dimension (or a
        clearly-bogus one cleared by the axis-label rule), look up the
        closest dimension-shaped OCR token within ~3× the bubble's
        radius and adopt it. Marked needs_review=True so the inspector
        knows to verify.

        Operates on the normalised token list cached by Step 4
        (`self._norm_tokens`). Each bubble only borrows a token if it
        hasn't already been assigned to another bubble in this pass.
        Tokens are accepted if they match a basic dimension shape:
        decimal value, integer, Ø/R/M prefix, or degree symbol.
        """
        tokens = getattr(self, "_norm_tokens", None) or []
        if not tokens or not bubbles:
            return

        # Strict acceptance pattern for the proximity fallback. On
        # noisy screenshot inputs, pure integer tokens (`29`, `1111`,
        # row labels in price tables) are a major source of false
        # matches, so we require ONE of: a decimal point, an explicit
        # engineering prefix (Ø, R, M), a degree symbol, or a
        # (REF)/(TYP) annotation. Plain integers without context get
        # ignored — better to leave the dim empty and let the
        # inspector fill it in than to guess from a price-table row.
        dim_pat = re.compile(
            r"""^
            (?:
                [ØRM]?\s*\d+[.,]\d+[°"']?(?:\s*\(REF\)|\s*\(TYP\))?  # decimal value
              | [ØRM]\s*\d+(?:[.,]\d+)?                              # explicit prefix
              | \d+\s*°(?:\s*\(REF\))?                               # degree
              | \d+(?:[.,]\d+)?\s*\((?:REF|TYP)\)                    # n(REF) / n(TYP)
            )
            $""",
            re.IGNORECASE | re.VERBOSE,
        )

        empty_or_cleared = [
            b for b in bubbles
            if not (b.dimension or "").strip()
        ]
        if not empty_or_cleared:
            return

        already_assigned = {
            (b.dimension or "").strip().lower()
            for b in bubbles if (b.dimension or "").strip()
        }

        for b in empty_or_cleared:
            # Distance budget for the fallback. Two constraints:
            #   - Floor of 200 px so small-radius balloons don't lose
            #     their true dim (e.g. bubble #14 r=43, true dim 157 px).
            #   - Ceiling of 280 px so large-radius balloons don't
            #     start matching dims from the opposite side of the
            #     drawing (e.g. bubble #6 r=96 found `M16` at 410 px
            #     from the price-table without this cap).
            # 280 px ≈ the longest leader line you typically see in a
            # standard CAD drawing.
            budget = min(280.0, max(200.0, 5.0 * float(b.radius or 30)))
            best_token = None
            best_dist = budget
            for t in tokens:
                text = (t.text or "").strip()
                if not text or not dim_pat.match(text):
                    continue
                if text.lower() in already_assigned:
                    continue
                dx = float(t.cx) - float(b.x)
                dy = float(t.cy) - float(b.y)
                d = math.hypot(dx, dy)
                if d < best_dist:
                    best_dist = d
                    best_token = t
            if best_token is not None:
                box = (
                    float(best_token.x1), float(best_token.y1),
                    float(best_token.x2), float(best_token.y2),
                )
                if not (
                    self._assignment_has_geometry_support(b, [box])
                    or self._box_has_local_support(b, box)
                ):
                    continue
                text = best_token.text.strip()
                reason = str(getattr(b, "review_reason", "") or "")
                self._apply_assignment_candidate(
                    b,
                    text,
                    max(float(getattr(b, "confidence", 0.0) or 0.0), 0.45),
                    ((reason + "+") if reason else "") + "proximity_fallback",
                    review_threshold=0.86,
                )
                already_assigned.add(text.lower())
                logger.info(
                    "Proximity fallback: bubble #%s ← %r (d=%.0fpx)",
                    b.bubble_number, b.dimension, best_dist,
                )

    def _clear_weak_orphan_dims(
        self,
        bubbles: List[BubbleResult],
    ) -> None:
        """Clear single/double-digit dim assignments on `needs_review`
        bubbles when the digits don't appear in any *nearby* OCR token.

        On noisy screenshot inputs the standard assignment path
        sometimes attaches a stray single-digit token from elsewhere on
        the drawing to a bubble whose real dim text was unreadable
        (OCR-shredded). Example failure: bubble #6 reading `4` while
        the real dim `9°(REF)` was visible right next to it but
        OCR'd as `9TrEr` / `L9'(REF)` — no clean candidate exists, so
        showing `4` actively misleads the inspector. Better to clear
        it and let them enter manually.

        A bubble is considered "orphaned" if its dim value doesn't
        appear as a substring of any token within 5× radius. Bubbles
        whose dim is genuinely visible nearby (e.g. #13 → '29' with a
        '29' token close by) are kept.
        """
        tokens = getattr(self, "_norm_tokens", None) or []
        if not tokens or not bubbles:
            return
        weak_dim_re = re.compile(r"^\d{1,2}$")
        for b in bubbles:
            if not b.needs_review:
                continue
            dim = (b.dimension or "").strip()
            if not dim or not weak_dim_re.fullmatch(dim):
                continue
            # Look for the dim as a *standalone numeric value* — not
            # just a digit-substring of a longer number. `4` matches
            # `4` and `4mm` and `R4`, but NOT `27.9`, `0.4`, or `14`.
            # The lookarounds exclude adjacent digit / decimal-point.
            standalone = re.compile(
                rf"(?<![0-9.]){re.escape(dim)}(?![0-9.])"
            )
            # Tighter budget here than the recovery fallback: a real
            # dim sits at the *end* of a leader line, which is typically
            # 2–3× radius. If the closest standalone match is 4×+ away,
            # the dim was almost certainly assigned from somewhere
            # unrelated (e.g. the price-table on the right side of the
            # drawing).
            budget = max(150.0, 3.0 * float(b.radius or 30))
            found_nearby = False
            for t in tokens:
                text = t.text or ""
                if not standalone.search(text):
                    continue
                d = math.hypot(float(t.cx) - float(b.x),
                               float(t.cy) - float(b.y))
                if d <= budget:
                    found_nearby = True
                    break
            if not found_nearby:
                logger.info(
                    "Cleared orphan weak dim for bubble #%s: %r "
                    "(not visible as a standalone value in any nearby "
                    "token within %.0fpx)",
                    b.bubble_number, dim, budget,
                )
                b.dimension = ""

    def _assess_confidence(self, b: BubbleResult) -> None:
        original_conf = float(b.confidence)
        reason = str(b.review_reason or "")
        conf = original_conf
        if b.candidate_count == 0:
            conf *= 0.3
        elif b.candidate_count == 1:
            conf *= 0.7
        elif b.candidate_count > 5:
            conf *= 0.8

        dim = b.dimension or ""
        if dim == "NO_DIMENSION":
            conf *= 0.1
        elif len(dim) < 2:
            conf *= 0.5
        elif "Ø" in dim:
            conf = min(conf * 1.1, 1.0)

        if "leader_first_trace" in reason and dim and dim != "NO_DIMENSION":
            conf = max(conf, original_conf)

        b.confidence = min(conf, 1.0)
        if b.confidence < 0.90 and not b.needs_review:
            b.needs_review  = True
            b.review_reason = b.review_reason or "low_confidence"

    # ── Annotation ────────────────────────────────────────────────

    def _annotate(
        self,
        img: np.ndarray,
        bubbles: List[BubbleResult],
        circles: List[Tuple[int, int, int]],
        callout_groups: List[CalloutGroup],
    ) -> np.ndarray:
        if not self.cfg.enable_annotation:
            return img.copy()

        out = img.copy()

        # Draw seed traces
        for bi, b in enumerate(bubbles):
            trace_info = self._trace_for_bubble(b, bi)
            if not trace_info:
                continue
            seed = trace_info.get("seed")
            path = trace_info.get("path")
            if seed is not None:
                cv2.circle(out, (seed.contact_x, seed.contact_y), 4, (0, 0, 255), -1)
            if path:
                for i in range(1, len(path)):
                    cv2.line(out, path[i - 1], path[i], (0, 165, 255), 1)

        # Draw bubbles
        for b in bubbles:
            color = (0, 200, 0) if b.dimension and b.dimension != "NO_DIMENSION" else (0, 180, 255)
            cv2.circle(out, (b.x, b.y), b.radius, color, 3)

            label = b.bubble_number
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lx = b.x - tw // 2
            ly = b.y - b.radius - 12
            cv2.rectangle(out, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 5), (255, 255, 255), -1)
            cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 0, 120), 2)

            if b.dimension and b.dimension != "NO_DIMENSION":
                dim = b.dimension[:42]
                # Unicode-capable rendering so Ø, °, ×, ±, ⊥, ⊕ etc.
                # appear as their actual glyphs instead of "?".
                font_px = 13
                dw, dh = _measure_unicode_text(dim, font_px)
                dx = b.x - dw // 2
                dy = b.y + b.radius + 16
                cv2.rectangle(out, (dx - 2, dy - dh - 2), (dx + dw + 2, dy + 4), (200, 255, 200), -1)
                _put_text_unicode(out, dim, (dx, dy), font_px, (0, 80, 0))

        return out
