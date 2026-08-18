"""Formal landmark evidence gate for action analysis.

This module is intentionally independent from the display stabilizer.  It
never predicts or fills a landmark: it only describes whether a measurement
may be used by formal features and state transitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any

from hyrox.geometry import coerce_point
from hyrox.landmark_names import HYROX_CORE_LANDMARKS


FORBIDDEN_FORMAL_ORIGINS = frozenset({"predicted", "held", "synthetic", "rejected"})


def _lookup(landmarks: Sequence[object] | Mapping[str | int, object], name: str) -> object | None:
    if isinstance(landmarks, Mapping):
        return landmarks.get(name)
    for point in landmarks:
        if getattr(point, "name", None) == name:
            return point
    return None


def _confidence(point: object | None) -> float:
    resolved = coerce_point(point, min_visibility=0.0, min_presence=0.0)
    if resolved is None:
        return 0.0
    return max(0.0, min(1.0, min(resolved.visibility, resolved.presence)))


@dataclass(frozen=True, slots=True)
class FormalQualityFrame:
    landmarks: Mapping[str, Mapping[str, object]]
    identity_continuity: Mapping[str, object]
    evidence_quality: float
    endpoint_evidence_allowed: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "landmarks": {name: dict(value) for name, value in self.landmarks.items()},
            "identity_continuity": dict(self.identity_continuity),
            "evidence_quality": self.evidence_quality,
            "endpoint_evidence_allowed": self.endpoint_evidence_allowed,
            "reason_codes": list(self.reason_codes),
            "formal_prediction_allowed": False,
        }


class FormalLandmarkQualityGate:
    """Classify measurements and detect conservative identity discontinuities."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.20,
        max_identity_gap_ms: int = 900,
        max_center_jump: float = 0.30,
        max_scale_ratio: float = 2.50,
        max_isolated_landmark_jump: float = 0.22,
    ) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_identity_gap_ms = max(1, int(max_identity_gap_ms))
        self.max_center_jump = max(0.01, float(max_center_jump))
        self.max_scale_ratio = max(1.01, float(max_scale_ratio))
        self.max_isolated_landmark_jump = max(0.01, float(max_isolated_landmark_jump))
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_ms: int | None = None
        self._last_center: tuple[float, float] | None = None
        self._last_scale: float | None = None
        self._last_points: dict[str, tuple[float, float]] = {}

    def evaluate(
        self,
        landmarks: Sequence[object] | Mapping[str | int, object] | None,
        *,
        timestamp_ms: int | None,
        metadata: Mapping[str, object] | None = None,
    ) -> FormalQualityFrame:
        metadata = metadata or {}
        held_names = {
            str(name)
            for name in (metadata.get("occlusion_guarded_keypoints") or ())
        }
        whole_pose_held = bool(metadata.get("stabilized_hold"))
        center, scale = self._body_extent(landmarks)
        center_motion = (
            None
            if center is None or self._last_center is None
            else hypot(center[0] - self._last_center[0], center[1] - self._last_center[1])
        )
        qualities: dict[str, dict[str, object]] = {}
        accepted_scores: list[float] = []
        forbidden_count = 0
        for name in HYROX_CORE_LANDMARKS:
            point = None if landmarks is None else _lookup(landmarks, name)
            confidence = _confidence(point)
            reasons: list[str] = []
            origin = "observed"
            if whole_pose_held:
                origin = "held"
                reasons.append("POSE_STALE")
            elif name in held_names:
                origin = "held"
                reasons.append("OCCLUDED")
            elif point is None:
                origin = "rejected"
                reasons.append("LANDMARK_MISSING")
            elif confidence < self.min_confidence:
                origin = "rejected"
                reasons.append("OCCLUDED")
            resolved = coerce_point(point, min_visibility=0.0, min_presence=0.0)
            previous = self._last_points.get(name)
            if (
                origin == "observed"
                and resolved is not None
                and previous is not None
                and max(abs(resolved.x), abs(resolved.y), abs(previous[0]), abs(previous[1])) <= 1.5
                and hypot(resolved.x - previous[0], resolved.y - previous[1])
                > self.max_isolated_landmark_jump
                and (center_motion is None or center_motion < self.max_center_jump * 0.35)
            ):
                origin = "rejected"
                reasons.append("LANDMARK_JUMP")
            observable = origin not in FORBIDDEN_FORMAL_ORIGINS
            if observable:
                accepted_scores.append(confidence)
            else:
                forbidden_count += 1
            qualities[name] = {
                "observable": observable,
                "origin": origin,
                "confidence": confidence,
                "reason_codes": reasons,
            }

        for name, quality in qualities.items():
            if not bool(quality["observable"]):
                continue
            point = None if landmarks is None else _lookup(landmarks, name)
            resolved = coerce_point(point, min_visibility=0.0, min_presence=0.0)
            if resolved is not None:
                self._last_points[name] = (resolved.x, resolved.y)

        identity_reasons: list[str] = []
        if timestamp_ms is not None and self._last_timestamp_ms is not None:
            gap = int(timestamp_ms) - self._last_timestamp_ms
            if gap <= 0 or gap > self.max_identity_gap_ms:
                identity_reasons.append("IDENTITY_TIMESTAMP_GAP")
        if center is not None and self._last_center is not None:
            if hypot(center[0] - self._last_center[0], center[1] - self._last_center[1]) > self.max_center_jump:
                identity_reasons.append("IDENTITY_CENTER_JUMP")
        if scale is not None and self._last_scale is not None and min(scale, self._last_scale) > 1e-6:
            ratio = max(scale, self._last_scale) / min(scale, self._last_scale)
            if ratio > self.max_scale_ratio:
                identity_reasons.append("IDENTITY_SCALE_JUMP")

        status = "DISCONTINUOUS" if identity_reasons else "CONTINUOUS"
        if timestamp_ms is not None:
            self._last_timestamp_ms = int(timestamp_ms)
        if center is not None:
            self._last_center = center
        if scale is not None:
            self._last_scale = scale
        reason_codes = list(identity_reasons)
        if whole_pose_held:
            reason_codes.append("PREDICTED_EVIDENCE_FORBIDDEN")
        evidence_quality = sum(accepted_scores) / len(HYROX_CORE_LANDMARKS)
        endpoint_allowed = not whole_pose_held and forbidden_count < max(1, len(HYROX_CORE_LANDMARKS) // 2)
        return FormalQualityFrame(
            landmarks=qualities,
            identity_continuity={"status": status, "reason_codes": identity_reasons},
            evidence_quality=max(0.0, min(1.0, evidence_quality)),
            endpoint_evidence_allowed=endpoint_allowed,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
        )

    @staticmethod
    def _body_extent(
        landmarks: Sequence[object] | Mapping[str | int, object] | None,
    ) -> tuple[tuple[float, float] | None, float | None]:
        if landmarks is None:
            return None, None
        points = [
            coerce_point(_lookup(landmarks, name), min_visibility=0.2, min_presence=0.2)
            for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        ]
        valid = [point for point in points if point is not None and isfinite(point.x) and isfinite(point.y)]
        if len(valid) < 2:
            return None, None
        center = (
            sum(point.x for point in valid) / len(valid),
            sum(point.y for point in valid) / len(valid),
        )
        xs = [point.x for point in valid]
        ys = [point.y for point in valid]
        return center, max(max(xs) - min(xs), max(ys) - min(ys))


__all__ = [
    "FORBIDDEN_FORMAL_ORIGINS",
    "FormalLandmarkQualityGate",
    "FormalQualityFrame",
]
