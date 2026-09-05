"""Symbol safety and evidence-gated recovery regressions (no model download)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diameter_ocr import diameter_candidate, reread_diameter
from diameter_ocr import _has_diameter_stroke
import cv2
from ocr_rules import OCRToken, normalize_engineering_text, normalize_ocr_tokens
from pipeline_v2.dimension_ocr import normalize_dimension_text, detect_dimension_text
from pipeline_v2.associator import associate
from pipeline_v2.contracts import BalloonCandidate, BalloonOcrResult, LeaderTrace


@pytest.mark.parametrize("text,expected", [
    ("Ø6", "Ø6"), ("⌀6", "Ø6"), ("φ3.5", "Ø3.5"), ("Φ6", "Ø6"),
    ("2-⌀6", "2-Ø6"), ("2-ø3.5", "2-Ø3.5"), ("2-O/6", "2-Ø6"),
    ("0.6", "0.6"), ("2-0.5", "2-0.5"), ("20", "20"),
])
def test_symbol_normalization(text, expected):
    assert normalize_engineering_text(text) == expected
    assert normalize_dimension_text(text) == expected
    assert normalize_engineering_text(expected) == expected


@pytest.mark.parametrize("text", ["≤0.5", "⊥0.1", "中文 ⌀6"])
def test_unicode_not_discarded(text):
    normalized = normalize_engineering_text(text)
    assert text[0] in normalized


@pytest.mark.parametrize("text,expected", [
    ("2-06", "2-Ø6"), ("2-03.5", "2-Ø3.5"), (" 4 – O 12,5 ", "4-Ø12.5"),
    ("2-0.5", None), ("0.6", None), ("06", "Ø6"), ("-03.5", "-Ø3.5"), ("PART2-06", None),
    ("2-006", None), ("2-Ø6", None), ("2-06 H7", None),
])
def test_proposals_are_narrow(text, expected):
    assert diameter_candidate(text) == expected


IMAGE = np.full((160, 260, 3), 255, dtype=np.uint8)
BOX = [[20, 30], [160, 90], [150, 115], [10, 55]]


@pytest.mark.parametrize("source,observed,expected", [
    ("2-06", "2-⌀6", "2-Ø6"), ("2-03.5", "2-φ3.5", "2-Ø3.5"),
    ("2-06", "2-06", "2-06"), ("2-06", "2-Ø8", "2-06"),
    ("2-06", "3-Ø6", "2-06"), ("2-06", "Ø6", "2-06"),
])
def test_retry_requires_matching_image_evidence(source, observed, expected):
    crops = []
    def ocr(crop, **kwargs):
        crops.append(crop)
        return [[BOX, observed, 0.95]], None
    result, conf = reread_diameter(IMAGE, BOX, source, 0.9, ocr)
    assert result == expected
    assert conf <= 0.9
    assert 1 <= len(crops) <= 2
    assert crops[0].shape[1] > crops[0].shape[0]


def test_retry_failure_and_low_confidence_keep_original():
    def failing(_):
        raise RuntimeError("engine unavailable")
    for ocr in (failing, lambda _: ([[BOX, "2-Ø6", 0.4]], None)):
        assert reread_diameter(IMAGE, BOX, "2-06", 0.9, ocr) == ("2-06", 0.9)
    assert reread_diameter(IMAGE, BOX, "2-0.5", 0.9, failing) == ("2-0.5", 0.9)


def test_retry_rejects_invalid_geometry_without_calling_engine():
    def unexpected(_):
        pytest.fail("Invalid geometry should not invoke OCR")
    for box in ([], [[0, 0]] * 4, [[float("nan"), 0]] * 4):
        assert reread_diameter(IMAGE, box, "2-06", 0.9, unexpected) == ("2-06", 0.9)


def test_v2_retry_budget(monkeypatch):
    from pipeline_v2 import dimension_ocr
    calls = []
    def ocr(crop, **kwargs):
        calls.append(crop)
        return ([[BOX, "2-06", 0.95]] * 20 if len(calls) == 1 else []), None
    monkeypatch.setattr(dimension_ocr, "get_ocr", lambda: ocr)
    candidates = detect_dimension_text(IMAGE)
    assert len(calls) == 1 + 12 * 2
    assert len(candidates) == 20
    assert all(c.review_required for c in candidates)


def test_original_read_survives_normalization():
    token = OCRToken("2-Ø6", 40, 40, 0.9, 10, 10, 70, 70, original_text="2-06")
    result = normalize_ocr_tokens([token], 200, 200)[0]
    assert result.raw_text == "2-06"
    assert result.text == "2-Ø6"


def test_recognition_only_retry_recovers_split_detection():
    calls = []
    def ocr(crop, **kwargs):
        calls.append(kwargs)
        if kwargs.get("use_det") is False:
            return [["2-⌀3.5", 0.92]], [0.01]
        return None, None
    assert reread_diameter(IMAGE, BOX, "2-03.5", 0.95, ocr) == ("2-Ø3.5", 0.92)
    assert calls == [{}, {"use_det": False, "use_cls": False}]


@pytest.mark.parametrize("symbol,expected", [("0", False), ("8", False), ("diameter", True)])
def test_pixel_verifier_distinguishes_zero_eight_and_diameter(symbol, expected):
    crop = np.full((110, 240, 3), 255, dtype=np.uint8)
    cv2.putText(crop, "2-", (5, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    cv2.putText(crop, "6", (165, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    if symbol == "diameter":
        cv2.ellipse(crop, (138, 52), (17, 21), 0, 0, 360, (0, 0, 0), 4)
        cv2.line(crop, (126, 76), (150, 28), (0, 0, 0), 4)
    else:
        cv2.putText(crop, symbol, (121, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    cv2.line(crop, (5, 92), (230, 92), (0, 0, 0), 3)
    assert _has_diameter_stroke(crop, "2-06") == expected


def test_default_drawing_real_ocr_dimensions():
    """Exercise the actual auto-annotation path, including rotated OCR/dedup."""
    from auto_annotate import _extract_dimensions
    drawing = Path(__file__).resolve().parent.parent / "web/public/demo-engineering-drawing.png"
    values = [g.text for g in _extract_dimensions(cv2.imread(str(drawing)))]
    assert "2-Ø6" in values
    assert "2-Ø3.5" in values
    assert "2.06" in values
    assert "2-06" not in values and "2-03.5" not in values
    assert len(values) == 10


def test_v2_recovery_and_review_propagation(monkeypatch):
    from pipeline_v2 import dimension_ocr
    calls = []
    def ocr(crop, **kwargs):
        calls.append(crop)
        return [[BOX, "2-06" if len(calls) == 1 else "2-⌀6", 0.95]], None
    monkeypatch.setattr(dimension_ocr, "get_ocr", lambda: ocr)
    candidate = detect_dimension_text(IMAGE)[0]
    assert candidate.text == "2-Ø6"
    assert candidate.raw_text == "2-06"
    assert not candidate.review_required

    monkeypatch.setattr(dimension_ocr, "get_ocr", lambda: lambda _, **kwargs: ([[BOX, "2-06", 0.95]], None))
    candidate = detect_dimension_text(IMAGE)[0]
    assert candidate.review_required
    assignments = associate(
        [BalloonCandidate("b", (0, 0, 10, 10), (5, 5), 5, 70, 1, 1, 0.99)],
        [BalloonOcrResult("b", "6", 0.99, "ok")],
        [LeaderTrace("b", [], (40, 50), (5, 5), 1, 0.99, "found")],
        [candidate],
    )
    assert assignments[0].review_required
    assert "ambiguous_diameter_symbol" in assignments[0].review_reason
