from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from statistics import median
from typing import Literal

from .geometry import PosePoint, coerce_point


FloorReferenceStatus = Literal["READY", "UNSURE"]
FloorReferenceSource = Literal["auto", "manual", "none"]

FOOT_NAMES = ("left_heel", "right_heel", "left_foot_index", "right_foot_index")


def _safe_float(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if isfinite(resolved) else None


@dataclass(frozen=True)
class FloorLine:
    point1: PosePoint
    point2: PosePoint

    def __post_init__(self) -> None:
        p1 = self.point1
        p2 = self.point2
        if p2.x < p1.x:
            object.__setattr__(self, "point1", p2)
            object.__setattr__(self, "point2", p1)
            p1, p2 = p2, p1
        if hypot(p2.x - p1.x, p2.y - p1.y) <= 1e-6:
            raise ValueError("floor line points must be distinct")
        if abs(p2.x - p1.x) <= 0.05:
            raise ValueError("floor line points must be separated horizontally")

    @classmethod
    def horizontal(cls, y: float) -> FloorLine:
        resolved = float(y)
        return cls(PosePoint(0.0, resolved), PosePoint(1.0, resolved))

    def y_at(self, x: float) -> float:
        dx = self.point2.x - self.point1.x
        ratio = (float(x) - self.point1.x) / dx
        return self.point1.y + ratio * (self.point2.y - self.point1.y)

    def as_dict(self) -> dict[str, float]:
        return {
            "x1": self.point1.x,
            "y1": self.point1.y,
            "x2": self.point2.x,
            "y2": self.point2.y,
        }


def signed_distance_to_floor(point: object | None, floor_line: FloorLine | None) -> float | None:
    """Return perpendicular image distance, positive on the body side of the floor."""
    resolved = coerce_point(point, min_visibility=0.0, min_presence=0.0)
    if resolved is None or floor_line is None:
        return None
    p1 = floor_line.point1
    p2 = floor_line.point2
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    length = hypot(dx, dy)
    cross = dx * (resolved.y - p1.y) - dy * (resolved.x - p1.x)
    return -cross / length


def normalized_height_to_floor(
    point: object | None,
    body_height: float | None,
    floor_line: FloorLine | None,
) -> float | None:
    distance = signed_distance_to_floor(point, floor_line)
    resolved_height = _safe_float(body_height)
    if distance is None or resolved_height is None or resolved_height <= 1e-6:
        return None
    return distance / resolved_height


@dataclass(frozen=True)
class FloorReferenceResult:
    status: FloorReferenceStatus
    source: FloorReferenceSource
    confidence: float
    reason_code: str | None
    line: FloorLine | None
    body_height: float | None
    body_height_source: str
    sample_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source": self.source,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "line": None if self.line is None else self.line.as_dict(),
            "body_height": self.body_height,
            "body_height_source": self.body_height_source,
            "sample_count": self.sample_count,
        }


class LocalFloorReference:
    """Estimate a local floor line from stable standing feet or two manual points."""

    def __init__(
        self,
        *,
        window_ms: int = 500,
        min_samples: int = 5,
        min_landmark_confidence: float = 0.60,
        stable_foot_y_tolerance: float = 0.012,
        contradiction_tolerance: float = 0.05,
        missing_timeout_ms: int = 750,
    ) -> None:
        self.window_ms = max(100, int(window_ms))
        self.min_samples = max(3, int(min_samples))
        self.min_landmark_confidence = max(0.0, min(1.0, float(min_landmark_confidence)))
        self.stable_foot_y_tolerance = max(0.001, float(stable_foot_y_tolerance))
        self.contradiction_tolerance = max(0.01, float(contradiction_tolerance))
        self.missing_timeout_ms = max(self.window_ms, int(missing_timeout_ms))
        self.reset()

    def reset(self) -> None:
        self._samples: deque[tuple[int, float, float | None]] = deque()
        self._auto_line: FloorLine | None = None
        self._manual_line: FloorLine | None = None
        self._calibrated_body_height: float | None = None
        self._last_foot_timestamp_ms: int | None = None
        self._started_timestamp_ms: int | None = None
        self._previous_floor_candidate: float | None = None
        self._contradiction_frames = 0
        self._pending_reason: str | None = None
        self.last_result = FloorReferenceResult(
            status="UNSURE",
            source="none",
            confidence=0.0,
            reason_code="FLOOR_NOT_CALIBRATED",
            line=None,
            body_height=None,
            body_height_source="none",
            sample_count=0,
        )

    @property
    def line(self) -> FloorLine | None:
        return self._manual_line or self._auto_line

    def set_manual_line(
        self,
        point1: object | None,
        point2: object | None,
    ) -> None:
        if point1 is None or point2 is None:
            self._manual_line = None
            return
        resolved1 = coerce_point(point1, min_visibility=0.0, min_presence=0.0)
        resolved2 = coerce_point(point2, min_visibility=0.0, min_presence=0.0)
        if resolved1 is None or resolved2 is None:
            raise ValueError("manual floor points must contain finite x/y coordinates")
        candidate = FloorLine(resolved1, resolved2)
        slope = abs((candidate.point2.y - candidate.point1.y) / (candidate.point2.x - candidate.point1.x))
        if slope > 1.0:
            raise ValueError("manual floor line is too steep")
        self._manual_line = candidate
        self._pending_reason = None

    def _timestamp(self, timestamp_ms: int | None, frame_index: int) -> int:
        return int(timestamp_ms) if timestamp_ms is not None else int(frame_index * 1000 / 30)

    def _foot_points(self, features: Mapping[str, object]) -> list[PosePoint]:
        points: list[PosePoint] = []
        for name in FOOT_NAMES:
            x = _safe_float(features.get(f"{name}_x"))
            y = _safe_float(features.get(f"{name}_y"))
            confidence = _safe_float(features.get(f"{name}_confidence"))
            if (
                x is None
                or y is None
                or confidence is None
                or confidence < self.min_landmark_confidence
            ):
                continue
            points.append(
                PosePoint(
                    x=x,
                    y=y,
                    visibility=confidence,
                    presence=confidence,
                )
            )
        return points

    def _standing(self, features: Mapping[str, object]) -> bool:
        knee = _safe_float(features.get("min_knee_angle"))
        hip = _safe_float(features.get("min_hip_angle"))
        return knee is not None and hip is not None and knee >= 160.0 and hip >= 155.0

    def _body_height(self, features: Mapping[str, object]) -> tuple[float | None, str]:
        if self._calibrated_body_height is not None:
            return self._calibrated_body_height, "standing_calibration"
        current = _safe_float(features.get("body_box_height_norm"))
        if current is not None and current > 0.10:
            return current, "current_body_box"
        current = _safe_float(features.get("body_height_norm"))
        if current is not None and current > 0.10:
            return current, "current_body_box"
        skeleton = _safe_float(features.get("skeleton_height_estimate_norm"))
        if skeleton is not None and skeleton > 0.10:
            return skeleton, "skeleton_segments"
        return None, "none"

    def _full_body_visible(
        self,
        features: Mapping[str, object],
        feet: Sequence[PosePoint],
    ) -> bool:
        lower_score = _safe_float(features.get("lower_body_visible_score")) or 0.0
        body_height = _safe_float(features.get("body_height_norm"))
        return (
            len(feet) >= 2
            and lower_score >= self.min_landmark_confidence
            and body_height is not None
            and body_height >= 0.20
            and all(0.01 <= point.x <= 0.99 and 0.02 <= point.y <= 0.995 for point in feet)
        )

    def _trim_samples(self, timestamp_ms: int) -> None:
        cutoff = timestamp_ms - self.window_ms
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _sample_confidence(self) -> float:
        if not self._samples:
            return 0.0
        values = [sample[1] for sample in self._samples]
        center = median(values)
        deviation = median(abs(value - center) for value in values)
        count_score = min(1.0, len(values) / max(self.min_samples, 1))
        stability_score = max(0.0, 1.0 - deviation / self.stable_foot_y_tolerance)
        return min(0.94, 0.55 + 0.25 * count_score + 0.14 * stability_score)

    def update(
        self,
        features: Mapping[str, object],
        *,
        timestamp_ms: int | None,
        frame_index: int,
    ) -> FloorReferenceResult:
        current_timestamp = self._timestamp(timestamp_ms, frame_index)
        if self._started_timestamp_ms is None:
            self._started_timestamp_ms = current_timestamp
        feet = self._foot_points(features)
        full_body_visible = self._full_body_visible(features, feet)
        standing = self._standing(features)
        floor_candidate = max((point.y for point in feet), default=None)

        if feet:
            self._last_foot_timestamp_ms = current_timestamp
        foot_missing_too_long = (
            (
                self._last_foot_timestamp_ms is None
                and current_timestamp - self._started_timestamp_ms > self.missing_timeout_ms
            )
            or (
                self._last_foot_timestamp_ms is not None
                and current_timestamp - self._last_foot_timestamp_ms > self.missing_timeout_ms
            )
        )

        stable_feet = False
        if floor_candidate is not None:
            stable_feet = (
                self._previous_floor_candidate is None
                or abs(floor_candidate - self._previous_floor_candidate)
                <= self.stable_foot_y_tolerance
            )
            self._previous_floor_candidate = floor_candidate

        if standing and stable_feet and full_body_visible and floor_candidate is not None:
            body_height = _safe_float(features.get("body_box_height_norm"))
            if body_height is None:
                body_height = _safe_float(features.get("body_height_norm"))
            self._samples.append((current_timestamp, floor_candidate, body_height))
            self._trim_samples(current_timestamp)
            if len(self._samples) >= self.min_samples:
                floor_y = median(sample[1] for sample in self._samples)
                body_heights = [
                    sample[2] for sample in self._samples if sample[2] is not None
                ]
                if self._auto_line is None:
                    self._auto_line = FloorLine.horizontal(floor_y)
                    if body_heights:
                        self._calibrated_body_height = median(body_heights)
                else:
                    contradiction = abs(floor_y - self._auto_line.y_at(0.5))
                    if contradiction > self.contradiction_tolerance:
                        self._contradiction_frames += 1
                    else:
                        self._contradiction_frames = 0
                        previous_y = self._auto_line.y_at(0.5)
                        self._auto_line = FloorLine.horizontal(
                            0.9 * previous_y + 0.1 * floor_y
                        )
                    if self._contradiction_frames >= 3:
                        self._auto_line = None
                        self._calibrated_body_height = None
                        newest = self._samples[-1]
                        self._samples.clear()
                        self._samples.append(newest)
                        self._contradiction_frames = 0
                        self._pending_reason = "CAMERA_MOVED"

        line = self.line
        source: FloorReferenceSource = (
            "manual" if self._manual_line is not None else "auto" if self._auto_line is not None else "none"
        )
        reason: str | None = self._pending_reason
        status: FloorReferenceStatus = "READY"
        if line is None:
            status = "UNSURE"
            reason = reason or "FLOOR_NOT_CALIBRATED"
        elif foot_missing_too_long:
            status = "UNSURE"
            reason = "FOOT_LANDMARKS_MISSING"
        elif not full_body_visible:
            status = "UNSURE"
            reason = "BODY_NOT_FULLY_IN_FRAME"
        elif self._manual_line is not None and standing and floor_candidate is not None:
            distances = [abs(signed_distance_to_floor(point, line) or 0.0) for point in feet]
            if distances and median(distances) > self.contradiction_tolerance:
                status = "UNSURE"
                reason = "FLOOR_FOOT_CONTRADICTION"

        body_height, body_height_source = self._body_height(features)
        confidence = 0.0
        if line is not None:
            confidence = 0.95 if source == "manual" else self._sample_confidence()
            if status == "UNSURE":
                confidence = min(confidence, 0.49)
        result = FloorReferenceResult(
            status=status,
            source=source,
            confidence=confidence,
            reason_code=reason,
            line=line,
            body_height=body_height,
            body_height_source=body_height_source,
            sample_count=len(self._samples),
        )
        self.last_result = result
        if status == "READY":
            self._pending_reason = None
        return result

    def enrich_features(
        self,
        features: dict[str, object],
        *,
        timestamp_ms: int | None,
        frame_index: int,
    ) -> FloorReferenceResult:
        result = self.update(features, timestamp_ms=timestamp_ms, frame_index=frame_index)
        line = result.line
        features.update(
            {
                "floor_reference_status": result.status,
                "floor_reference_source": result.source,
                "floor_reference_confidence": result.confidence,
                "floor_reference_reason": result.reason_code,
                "floor_y": None if line is None else line.y_at(0.5),
                "floor_line_x1": None if line is None else line.point1.x,
                "floor_line_y1": None if line is None else line.point1.y,
                "floor_line_x2": None if line is None else line.point2.x,
                "floor_line_y2": None if line is None else line.point2.y,
                "body_height_reference": result.body_height,
                "body_height_reference_source": result.body_height_source,
            }
        )
        body_height = result.body_height
        for name in ("hip_center", "knee_center", *FOOT_NAMES):
            x = _safe_float(features.get(f"{name}_x"))
            y = _safe_float(features.get(f"{name}_y"))
            point = None if x is None or y is None else PosePoint(x, y)
            features[f"{name}_height_to_floor"] = signed_distance_to_floor(point, line)
            features[f"{name}_height_to_floor_body_ratio"] = normalized_height_to_floor(
                point,
                body_height,
                line,
            )
        return result


__all__ = [
    "FOOT_NAMES",
    "FloorLine",
    "FloorReferenceResult",
    "LocalFloorReference",
    "normalized_height_to_floor",
    "signed_distance_to_floor",
]
