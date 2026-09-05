from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class ImageEnhancementResult:
    image_shape: Tuple[int, int]
    enhanced_shape: Tuple[int, int]
    scale_factor: float
    quality_report: Dict[str, Any]
    ops: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnnotationSegmentationResult:
    dominant_hue: Optional[int]
    dominant_sat: Optional[int]
    dominant_val: Optional[int]
    pixel_count: int
    confidence: float
    color_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BalloonCandidate:
    candidate_id: str
    bbox: BBox
    center: Tuple[float, float]
    radius: float
    contour_area: float
    circularity: float
    aspect_ratio: float
    confidence: float


@dataclass
class BalloonOcrResult:
    balloon_candidate_id: str
    text: str
    confidence: float
    status: str
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeaderTrace:
    balloon_candidate_id: str
    polyline: List[Point]
    target_endpoint: Optional[Point]
    source_endpoint: Optional[Point]
    component_id: Optional[int]
    confidence: float
    status: str
    method: str = "none"
    fallback_stage: int = 0
    debug_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextCandidate:
    candidate_id: str
    text: str
    bbox: BBox
    center: Tuple[float, float]
    confidence: float
    kind: str
    suppressed: bool = False
    suppression_reason: str = ""
    raw_text: str = ""
    review_required: bool = False


@dataclass
class Assignment:
    balloon_candidate_id: str
    balloon_id: str
    dimension_candidate_id: Optional[str]
    dimension_text: str
    dimension_bbox: Optional[BBox]
    confidence: float
    review_required: bool
    review_reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineV2Result:
    image: str
    enhancement: ImageEnhancementResult
    segmentation: AnnotationSegmentationResult
    balloons: List[BalloonCandidate]
    balloon_ocr: List[BalloonOcrResult]
    leaders: List[LeaderTrace]
    text_candidates: List[TextCandidate]
    assignments: List[Assignment]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
