from __future__ import annotations

import math
from typing import Dict, List, Optional

from .contracts import Assignment, BalloonCandidate, BalloonOcrResult, LeaderTrace, TextCandidate


def _bbox_distance(point, bbox) -> float:
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(math.hypot(dx, dy))


def associate(
    balloons: List[BalloonCandidate],
    balloon_ocr: List[BalloonOcrResult],
    leaders: List[LeaderTrace],
    text_candidates: List[TextCandidate],
) -> List[Assignment]:
    ocr_by_balloon: Dict[str, BalloonOcrResult] = {o.balloon_candidate_id: o for o in balloon_ocr}
    leader_by_balloon: Dict[str, LeaderTrace] = {l.balloon_candidate_id: l for l in leaders}
    assignments: List[Assignment] = []
    claimed: set[str] = set()

    for balloon in balloons:
        ocr = ocr_by_balloon.get(balloon.candidate_id)
        leader = leader_by_balloon.get(balloon.candidate_id)
        endpoint = leader.target_endpoint if leader and leader.status == "found" else None
        best: Optional[TextCandidate] = None
        best_score = float("inf")
        best_dist = float("inf")

        if endpoint is not None:
            for cand in text_candidates:
                if cand.candidate_id in claimed:
                    continue
                dist = _bbox_distance(endpoint, cand.bbox)
                table_penalty = 250.0 if cand.suppressed else 0.0
                ocr_bonus = -80.0 * cand.confidence
                score = dist + table_penalty + ocr_bonus
                if score < best_score:
                    best = cand
                    best_score = score
                    best_dist = dist

        if best is not None:
            claimed.add(best.candidate_id)
            leader_conf = leader.confidence if leader else 0.0
            balloon_id_conf = ocr.confidence if ocr else 0.0
            distance_conf = max(0.0, min(1.0, 1.0 - best_dist / 220.0))
            confidence = float(0.25 * balloon.confidence + 0.20 * balloon_id_conf + 0.25 * leader_conf + 0.20 * best.confidence + 0.10 * distance_conf)
            review = confidence < 0.65 or best.suppressed or best.review_required
            reason = "low_confidence" if confidence < 0.65 else ""
            if best.suppressed:
                reason = "dimension_inside_suppressed_region"
            if best.review_required:
                reason = ";".join(filter(None, (reason, "ambiguous_diameter_symbol")))
            assignments.append(Assignment(
                balloon_candidate_id=balloon.candidate_id,
                balloon_id=ocr.text if ocr and ocr.text else balloon.candidate_id,
                dimension_candidate_id=best.candidate_id,
                dimension_text=best.text,
                dimension_bbox=best.bbox,
                confidence=confidence,
                review_required=review,
                review_reason=reason,
                evidence={
                    "endpoint": endpoint,
                    "endpoint_to_text_distance": round(best_dist, 2),
                    "association_score": round(best_score, 2),
                    "raw_ocr_text": best.raw_text or best.text,
                },
            ))
        else:
            assignments.append(Assignment(
                balloon_candidate_id=balloon.candidate_id,
                balloon_id=ocr.text if ocr and ocr.text else balloon.candidate_id,
                dimension_candidate_id=None,
                dimension_text="NO_DIMENSION",
                dimension_bbox=None,
                confidence=0.0,
                review_required=True,
                review_reason="no_leader_or_dimension_candidate",
            ))

    return assignments
