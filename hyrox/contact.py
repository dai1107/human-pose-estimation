from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import acos, degrees, hypot, isfinite
from typing import Literal

from .floor_reference import FloorLine, signed_distance_to_floor
from .geometry import PosePoint


ContactStatus = Literal["CONTACT", "NO_CONTACT", "UNSURE", "NOT_OBSERVABLE"]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if isfinite(resolved) else None


@dataclass(frozen=True)
class ContactResult:
    status: ContactStatus
    confidence: float
    surface_height_ratio: float | None
    hold_ms: int
    evidence_frames: list[int]
    surface_point_x: float | None = None
    surface_point_y: float | None = None
    surface_side: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "confidence": round(_clamp(self.confidence), 4),
            "surface_height_ratio": self.surface_height_ratio,
            "hold_ms": max(0, int(self.hold_ms)),
            "evidence_frames": list(self.evidence_frames),
            "surface_point": (
                None
                if self.surface_point_x is None or self.surface_point_y is None
                else {
                    "x": self.surface_point_x,
                    "y": self.surface_point_y,
                }
            ),
            "surface_side": self.surface_side,
        }


@dataclass(frozen=True)
class KneeContactConfig:
    surface_radius_shank_ratio: float = 0.10
    enter_height_body_ratio: float = 0.015
    exit_height_body_ratio: float = 0.035
    max_vertical_speed_body_per_second: float = 0.15
    min_hold_frames: Mapping[str, int] = field(
        default_factory=lambda: {"high": 2, "medium": 3, "low": 4}
    )
    min_landmark_confidence: float = 0.60
    confirm_confidence: float = 0.72


@dataclass(frozen=True)
class ChestContactConfig:
    shoulder_weight: float = 0.65
    hip_weight: float = 0.35
    surface_offset_torso_ratio: float = 0.20
    enter_height_body_ratio: float = 0.020
    exit_height_body_ratio: float = 0.045
    shoulder_height_body_ratio_max: float = 0.080
    hip_height_body_ratio_max: float = 0.160
    torso_to_floor_angle_deg_max: float = 25.0
    min_hold_frames: Mapping[str, int] = field(
        default_factory=lambda: {"high": 2, "medium": 3, "low": 4}
    )
    min_landmark_confidence: float = 0.60
    confirm_confidence: float = 0.72
    max_vertical_speed_body_per_second: float = 0.15


@dataclass(frozen=True)
class SegmentationContactConfig:
    enabled: bool = True
    floor_band_body_ratio: float = 0.012
    minimum_overlap_ratio: float = 0.08
    mask_threshold: float = 0.50


@dataclass(frozen=True)
class _Observation:
    frame_index: int
    timestamp_ms: int
    height_ratio: float
    vertical_speed: float | None


class _TemporalContactState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.contact = False
        self.candidate_start_ms: int | None = None
        self.candidate_frames: list[int] = []
        self.observations: deque[_Observation] = deque(maxlen=3)
        self.last_result = ContactResult("UNSURE", 0.0, None, 0, [])

    def observe(
        self,
        *,
        height_ratio: float,
        frame_index: int,
        timestamp_ms: int | None,
        enter_ratio: float,
        exit_ratio: float,
    ) -> tuple[int, int, list[int], bool, float | None]:
        timestamp = (
            int(timestamp_ms)
            if timestamp_ms is not None
            else (
                self.observations[-1].timestamp_ms + 33
                if self.observations
                else max(0, int(frame_index) * 33)
            )
        )
        speed = None
        if self.observations:
            previous = self.observations[-1]
            elapsed = timestamp - previous.timestamp_ms
            if elapsed > 0:
                speed = (height_ratio - previous.height_ratio) * 1000.0 / elapsed
        observation = _Observation(int(frame_index), timestamp, height_ratio, speed)
        self.observations.append(observation)

        in_enter_range = height_ratio <= enter_ratio
        in_hysteresis_range = height_ratio <= exit_ratio
        if in_enter_range:
            if self.candidate_start_ms is None:
                self.candidate_start_ms = timestamp
            if frame_index not in self.candidate_frames:
                self.candidate_frames.append(int(frame_index))
        elif not in_hysteresis_range:
            self.contact = False
            self.candidate_start_ms = None
            self.candidate_frames.clear()

        local_minimum = False
        if len(self.observations) == 3:
            before, middle, after = self.observations
            local_minimum = (
                before.height_ratio > middle.height_ratio
                and after.height_ratio > middle.height_ratio
                and middle.height_ratio <= enter_ratio
                and (middle.vertical_speed is None or middle.vertical_speed < 0.0)
                and (after.vertical_speed is None or after.vertical_speed > 0.0)
            )
            if local_minimum:
                if self.candidate_start_ms is None:
                    self.candidate_start_ms = middle.timestamp_ms
                if middle.frame_index not in self.candidate_frames:
                    self.candidate_frames.append(middle.frame_index)

        hold_ms = (
            0
            if self.candidate_start_ms is None
            else max(0, timestamp - self.candidate_start_ms)
        )
        return (
            len(self.candidate_frames) + (1 if local_minimum else 0),
            hold_ms,
            self.candidate_frames[-12:],
            local_minimum,
            speed,
        )


def _floor_line(features: Mapping[str, object]) -> FloorLine | None:
    values = [
        _safe_float(features.get(name))
        for name in ("floor_line_x1", "floor_line_y1", "floor_line_x2", "floor_line_y2")
    ]
    if any(value is None for value in values):
        return None
    try:
        return FloorLine(
            PosePoint(values[0], values[1]),  # type: ignore[arg-type]
            PosePoint(values[2], values[3]),  # type: ignore[arg-type]
        )
    except ValueError:
        return None


def _point(
    features: Mapping[str, object],
    name: str,
    min_confidence: float,
) -> PosePoint | None:
    x = _safe_float(features.get(f"{name}_x"))
    y = _safe_float(features.get(f"{name}_y"))
    confidence = _safe_float(features.get(f"{name}_confidence"))
    if (
        x is None
        or y is None
        or confidence is None
        or confidence < min_confidence
    ):
        return None
    return PosePoint(x, y, visibility=confidence, presence=confidence)


def _distance_score(height_ratio: float, enter_ratio: float, exit_ratio: float) -> float:
    if height_ratio <= enter_ratio:
        return 1.0
    if height_ratio >= exit_ratio:
        return 0.0
    return 1.0 - (height_ratio - enter_ratio) / max(1e-6, exit_ratio - enter_ratio)


def _surface_point_toward_floor(
    point: PosePoint,
    floor: FloorLine,
    offset: float,
) -> PosePoint:
    """Move a joint/torso center toward the floor along the floor normal."""
    dx = floor.point2.x - floor.point1.x
    dy = floor.point2.y - floor.point1.y
    length = hypot(dx, dy)
    return PosePoint(
        point.x - dy * float(offset) / length,
        point.y + dx * float(offset) / length,
    )


def _phase_score(kind: str, phase: str) -> float:
    normalized = str(phase).strip().lower()
    if normalized in {"unknown", "no_pose", "low_visibility", "idle", "reset"}:
        return 0.0
    if kind == "chest":
        if normalized in {"chest_down", "prone", "bottom"}:
            return 1.0
        if normalized in {"hands_down", "descent"}:
            return 0.70
        return 0.25
    if normalized in {"bottom", "kneeling", "knee_down"}:
        return 1.0
    if normalized in {"descent", "ascent"}:
        return 0.75
    if normalized in {"stand", "landing", "flight_or_move"}:
        return 0.0
    return 0.35


def _torso_floor_angle(
    shoulder_mid: PosePoint,
    hip_mid: PosePoint,
    floor_line: FloorLine,
) -> float | None:
    torso = (hip_mid.x - shoulder_mid.x, hip_mid.y - shoulder_mid.y)
    floor = (
        floor_line.point2.x - floor_line.point1.x,
        floor_line.point2.y - floor_line.point1.y,
    )
    torso_length = hypot(*torso)
    floor_length = hypot(*floor)
    if torso_length <= 1e-8 or floor_length <= 1e-8:
        return None
    cosine = abs((torso[0] * floor[0] + torso[1] * floor[1]) / (torso_length * floor_length))
    return degrees(acos(_clamp(cosine, -1.0, 1.0)))


def segmentation_floor_overlap_ratio(
    mask: object | None,
    *,
    chest_center: PosePoint,
    torso_length: float,
    body_height: float,
    floor_line: FloorLine,
    config: SegmentationContactConfig | None = None,
) -> float | None:
    """Return chest-ROI silhouette overlap with a narrow local floor band."""
    if mask is None:
        return None
    try:
        import numpy as np

        values = np.asarray(mask)
    except (ImportError, TypeError, ValueError):
        return None
    if values.ndim == 3:
        values = np.squeeze(values)
    if values.ndim != 2 or values.size == 0:
        return None
    cfg = config or SegmentationContactConfig()
    height, width = values.shape
    if height <= 1 or width <= 1:
        return None

    center_x = chest_center.x * (width - 1)
    center_y = chest_center.y * (height - 1)
    radius_x = max(2.0, torso_length * width * 0.55)
    radius_y = max(2.0, torso_length * height * 0.38)
    x0 = max(0, int(center_x - radius_x))
    x1 = min(width, int(center_x + radius_x) + 1)
    y0 = max(0, int(center_y - radius_y))
    y1 = min(height, int(center_y + radius_y) + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    ys, xs = np.mgrid[y0:y1, x0:x1]
    normalized_x = xs / max(1, width - 1)
    normalized_y = ys / max(1, height - 1)
    dx = floor_line.point2.x - floor_line.point1.x
    dy = floor_line.point2.y - floor_line.point1.y
    line_length = hypot(dx, dy)
    signed = -(
        dx * (normalized_y - floor_line.point1.y)
        - dy * (normalized_x - floor_line.point1.x)
    ) / max(line_length, 1e-8)
    floor_band = abs(signed) <= max(1e-5, cfg.floor_band_body_ratio * body_height)
    silhouette = values[y0:y1, x0:x1] >= cfg.mask_threshold
    roi_count = int(silhouette.sum())
    if roi_count <= 0:
        return 0.0
    return _clamp(float((silhouette & floor_band).sum()) / roi_count)


class KneeContactDetector:
    def __init__(
        self,
        config: KneeContactConfig | None = None,
        *,
        sensitivity: str = "medium",
    ) -> None:
        self.config = config or KneeContactConfig()
        self.sensitivity = sensitivity if sensitivity in {"high", "medium", "low"} else "medium"
        self._state = _TemporalContactState()

    def reset(self) -> None:
        self._state.reset()

    def update(
        self,
        features: Mapping[str, object],
        *,
        phase: str,
        frame_index: int,
        timestamp_ms: int | None,
        side: Literal["left", "right"] | None = None,
    ) -> ContactResult:
        cfg = self.config
        floor = _floor_line(features)
        body_height = _safe_float(features.get("body_height_reference"))
        if features.get("floor_reference_status") != "READY" or floor is None or not body_height or body_height <= 0:
            result = ContactResult("UNSURE", 0.0, None, 0, [])
            self._state.last_result = result
            return result

        candidates: list[tuple[float, float, PosePoint, str]] = []
        sides = (side,) if side is not None else ("left", "right")
        for candidate_side in sides:
            knee = _point(
                features,
                f"{candidate_side}_knee",
                cfg.min_landmark_confidence,
            )
            ankle = _point(
                features,
                f"{candidate_side}_ankle",
                cfg.min_landmark_confidence,
            )
            if knee is None or ankle is None:
                continue
            joint_height = signed_distance_to_floor(knee, floor)
            if joint_height is None:
                continue
            shank_length = hypot(knee.x - ankle.x, knee.y - ankle.y)
            surface_height_ratio = (
                joint_height - cfg.surface_radius_shank_ratio * shank_length
            ) / body_height
            surface_point = _surface_point_toward_floor(
                knee,
                floor,
                cfg.surface_radius_shank_ratio * shank_length,
            )
            confidence = min(knee.visibility, ankle.visibility)
            candidates.append(
                (
                    surface_height_ratio,
                    confidence,
                    surface_point,
                    candidate_side,
                )
            )
        if not candidates:
            result = ContactResult("NOT_OBSERVABLE", 0.0, None, 0, [])
            self._state.last_result = result
            return result

        height_ratio, landmark_confidence, surface_point, surface_side = min(
            candidates,
            key=lambda item: item[0],
        )
        held_frames, hold_ms, evidence, local_minimum, speed = self._state.observe(
            height_ratio=height_ratio,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            enter_ratio=cfg.enter_height_body_ratio,
            exit_ratio=cfg.exit_height_body_ratio,
        )
        phase_score = _phase_score("knee", phase)
        distance_score = _distance_score(
            height_ratio,
            cfg.enter_height_body_ratio,
            cfg.exit_height_body_ratio,
        )
        speed_score = (
            0.5
            if speed is None
            else _clamp(1.0 - abs(speed) / max(cfg.max_vertical_speed_body_per_second, 1e-6))
        )
        duration_score = _clamp(
            held_frames / max(1, int(cfg.min_hold_frames[self.sensitivity]))
        )
        confidence = _clamp(
            0.40 * distance_score
            + 0.15 * phase_score
            + 0.15 * speed_score
            + 0.15 * duration_score
            + 0.15 * landmark_confidence
        )
        motion_ok = (
            speed is None
            or abs(speed) <= cfg.max_vertical_speed_body_per_second
            or local_minimum
        )
        enough_hold = held_frames >= max(1, int(cfg.min_hold_frames[self.sensitivity]))
        if height_ratio > cfg.exit_height_body_ratio:
            status: ContactStatus = "NO_CONTACT"
        elif self._state.contact and phase_score >= 0.70:
            status = "CONTACT"
        elif (
            (
                height_ratio <= cfg.enter_height_body_ratio
                or local_minimum
                or self._state.contact
            )
            and enough_hold
            and motion_ok
            and phase_score >= 0.70
            and confidence >= cfg.confirm_confidence
        ):
            status = "CONTACT"
            self._state.contact = True
        else:
            status = "UNSURE"
        result = ContactResult(
            status,
            confidence,
            height_ratio,
            hold_ms,
            evidence,
            surface_point.x,
            surface_point.y,
            surface_side,
        )
        self._state.last_result = result
        return result


class ChestContactDetector:
    """Detect a chest-to-floor proxy; this is not a nipple-line measurement."""

    def __init__(
        self,
        config: ChestContactConfig | None = None,
        segmentation_config: SegmentationContactConfig | None = None,
        *,
        sensitivity: str = "medium",
    ) -> None:
        self.config = config or ChestContactConfig()
        self.segmentation_config = segmentation_config or SegmentationContactConfig()
        self.sensitivity = sensitivity if sensitivity in {"high", "medium", "low"} else "medium"
        self._state = _TemporalContactState()

    def reset(self) -> None:
        self._state.reset()

    def update(
        self,
        features: Mapping[str, object],
        *,
        phase: str,
        frame_index: int,
        timestamp_ms: int | None,
    ) -> ContactResult:
        cfg = self.config
        floor = _floor_line(features)
        body_height = _safe_float(features.get("body_height_reference"))
        if features.get("floor_reference_status") != "READY" or floor is None or not body_height or body_height <= 0:
            result = ContactResult("UNSURE", 0.0, None, 0, [])
            self._state.last_result = result
            return result

        points = {
            name: _point(features, name, cfg.min_landmark_confidence)
            for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        }
        if any(point is None for point in points.values()):
            result = ContactResult("NOT_OBSERVABLE", 0.0, None, 0, [])
            self._state.last_result = result
            return result
        left_shoulder = points["left_shoulder"]
        right_shoulder = points["right_shoulder"]
        left_hip = points["left_hip"]
        right_hip = points["right_hip"]
        assert left_shoulder and right_shoulder and left_hip and right_hip
        shoulder_mid = PosePoint(
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
        )
        hip_mid = PosePoint(
            (left_hip.x + right_hip.x) / 2.0,
            (left_hip.y + right_hip.y) / 2.0,
        )
        chest_center = PosePoint(
            cfg.shoulder_weight * shoulder_mid.x + cfg.hip_weight * hip_mid.x,
            cfg.shoulder_weight * shoulder_mid.y + cfg.hip_weight * hip_mid.y,
        )
        torso_length = hypot(
            shoulder_mid.x - hip_mid.x,
            shoulder_mid.y - hip_mid.y,
        )
        center_height = signed_distance_to_floor(chest_center, floor)
        shoulder_height = signed_distance_to_floor(shoulder_mid, floor)
        hip_height = signed_distance_to_floor(hip_mid, floor)
        if center_height is None or shoulder_height is None or hip_height is None:
            result = ContactResult("UNSURE", 0.0, None, 0, [])
            self._state.last_result = result
            return result
        height_ratio = (
            center_height - cfg.surface_offset_torso_ratio * torso_length
        ) / body_height
        surface_point = _surface_point_toward_floor(
            chest_center,
            floor,
            cfg.surface_offset_torso_ratio * torso_length,
        )
        shoulder_ratio = shoulder_height / body_height
        hip_ratio = hip_height / body_height
        torso_angle = _torso_floor_angle(shoulder_mid, hip_mid, floor)

        held_frames, hold_ms, evidence, local_minimum, speed = self._state.observe(
            height_ratio=height_ratio,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            enter_ratio=cfg.enter_height_body_ratio,
            exit_ratio=cfg.exit_height_body_ratio,
        )
        distance_score = _distance_score(
            height_ratio,
            cfg.enter_height_body_ratio,
            cfg.exit_height_body_ratio,
        )
        overlap = None
        if self.segmentation_config.enabled:
            overlap = segmentation_floor_overlap_ratio(
                features.get("_segmentation_mask"),
                chest_center=chest_center,
                torso_length=torso_length,
                body_height=body_height,
                floor_line=floor,
                config=self.segmentation_config,
            )
        segmentation_score = (
            0.0
            if overlap is None
            else _clamp(
                overlap / max(self.segmentation_config.minimum_overlap_ratio, 1e-6)
            )
        )
        parallel_score = (
            0.0
            if torso_angle is None
            else _clamp(1.0 - torso_angle / max(cfg.torso_to_floor_angle_deg_max, 1e-6))
        )
        speed_score = (
            0.5
            if speed is None
            else _clamp(1.0 - abs(speed) / max(cfg.max_vertical_speed_body_per_second, 1e-6))
        )
        duration_score = _clamp(
            held_frames / max(1, int(cfg.min_hold_frames[self.sensitivity]))
        )
        confidence = (
            0.40 * distance_score
            + 0.25 * segmentation_score
            + 0.15 * parallel_score
            + 0.10 * speed_score
            + 0.10 * duration_score
        )
        landmark_confidence = min(
            left_shoulder.visibility,
            right_shoulder.visibility,
            left_hip.visibility,
            right_hip.visibility,
        )
        confidence *= 0.85 + 0.15 * landmark_confidence
        phase_score = _phase_score("chest", phase)
        confidence *= 0.70 + 0.30 * phase_score
        if overlap is None:
            confidence = min(confidence, 0.74)
        confidence = _clamp(confidence)

        geometry_ok = (
            shoulder_ratio <= cfg.shoulder_height_body_ratio_max
            and hip_ratio <= cfg.hip_height_body_ratio_max
            and torso_angle is not None
            and torso_angle <= cfg.torso_to_floor_angle_deg_max
        )
        motion_ok = (
            speed is None
            or abs(speed) <= cfg.max_vertical_speed_body_per_second
            or local_minimum
        )
        enough_hold = held_frames >= max(1, int(cfg.min_hold_frames[self.sensitivity]))
        if height_ratio > cfg.exit_height_body_ratio:
            status: ContactStatus = "NO_CONTACT"
        elif self._state.contact and geometry_ok and phase_score >= 0.70:
            status = "CONTACT"
        elif (
            (
                height_ratio <= cfg.enter_height_body_ratio
                or local_minimum
                or self._state.contact
            )
            and geometry_ok
            and enough_hold
            and motion_ok
            and phase_score >= 0.70
            and confidence >= cfg.confirm_confidence
        ):
            status = "CONTACT"
            self._state.contact = True
        else:
            status = "UNSURE"
        result = ContactResult(
            status,
            confidence,
            height_ratio,
            hold_ms,
            evidence,
            surface_point.x,
            surface_point.y,
            "chest",
        )
        self._state.last_result = result
        return result


class ContactDetectorSuite:
    def __init__(self, *, sensitivity: str = "medium") -> None:
        self.sensitivity = sensitivity if sensitivity in {"high", "medium", "low"} else "medium"
        self.knee = KneeContactDetector(sensitivity=self.sensitivity)
        self.chest = ChestContactDetector(sensitivity=self.sensitivity)
        self.last_results = {
            "knee": ContactResult("UNSURE", 0.0, None, 0, []),
            "chest_proxy": ContactResult("UNSURE", 0.0, None, 0, []),
        }

    def reset(self) -> None:
        self.knee.reset()
        self.chest.reset()

    def update(
        self,
        features: Mapping[str, object],
        *,
        phase: str,
        frame_index: int,
        timestamp_ms: int | None,
    ) -> dict[str, ContactResult]:
        self.last_results = {
            "knee": self.knee.update(
                features,
                phase=phase,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            ),
            "chest_proxy": self.chest.update(
                features,
                phase=phase,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            ),
        }
        return self.last_results

    def as_dict(self) -> dict[str, dict[str, object]]:
        return {
            name: result.as_dict()
            for name, result in self.last_results.items()
        }


__all__ = [
    "ChestContactConfig",
    "ChestContactDetector",
    "ContactDetectorSuite",
    "ContactResult",
    "ContactStatus",
    "KneeContactConfig",
    "KneeContactDetector",
    "SegmentationContactConfig",
    "segmentation_floor_overlap_ratio",
]
