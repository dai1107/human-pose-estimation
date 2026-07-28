"""Body-relative 3D shadow evidence.

The measurements in this module are deliberately body-relative.  MediaPipe
world landmarks are not a calibrated camera/world coordinate system, so none
of the values below are interpreted as metric floor distance or contact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import acos, degrees, isfinite
from statistics import median
from typing import Any

import numpy as np


FOOT_POINTS = {
    "left": ("left_ankle", "left_heel", "left_foot_index"),
    "right": ("right_ankle", "right_heel", "right_foot_index"),
}


@dataclass(frozen=True, slots=True)
class ShadowEvidence3DConfig:
    """Conservative thresholds used only by the shadow experiment."""

    min_visibility: float = 0.70
    min_presence: float = 0.70
    depth_order_gap_body_ratio: float = 0.12
    stationary_speed_body_per_s: float = 0.18
    stationary_dwell_frames: int = 3
    synchronous_event_ms: float = 120.0
    conflict_event_ms: float = 220.0
    prone_horizontal_score_min: float = 0.68
    angle_assist_enabled: bool = True
    body_assist_enabled: bool = True
    max_2d_3d_difference_deg: float = 45.0
    angle_conflict_min_frames: int = 3
    angle_conflict_min_ratio: float = 0.35
    angle_support_min_frames: int = 3
    angle_support_min_ratio: float = 0.50
    confidence_boost: float = 0.04
    conflict_confidence_cap: float = 0.49

    def as_dict(self) -> dict[str, float | int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


class BodyRelative3DTracker:
    """Extract temporal body-relative 3D features without inferring contact."""

    def __init__(self, config: ShadowEvidence3DConfig | None = None) -> None:
        self.config = config or ShadowEvidence3DConfig()
        self.reset()

    def reset(self) -> None:
        self._previous_timestamp_ms: float | None = None
        self._previous_hip_center: np.ndarray | None = None
        self._previous_relative: dict[str, np.ndarray] = {}
        self._stationary_frames = {"left": 0, "right": 0}
        self._moving = {"left": False, "right": False}
        self._takeoff_ms: dict[str, float | None] = {"left": None, "right": None}
        self._landing_ms: dict[str, float | None] = {"left": None, "right": None}
        self._takeoff_count = {"left": 0, "right": 0}
        self._landing_count = {"left": 0, "right": 0}

    def update(
        self,
        image_points: Sequence[object],
        world_points: Sequence[object],
        *,
        timestamp_ms: float | int | None,
        camera_view: str = "unknown",
    ) -> dict[str, Any]:
        image = _point_map(image_points)
        world = _point_map(world_points)
        arrays = {
            name: value
            for name, point in world.items()
            if (value := _xyz(point)) is not None
        }
        reasons: set[str] = set()
        required = (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            reasons.add("body_relative_world_landmarks_missing")
        quality_points = [image.get(name) or world.get(name) for name in required]
        visibility = min(
            (_quality(point, "visibility") for point in quality_points),
            default=0.0,
        )
        presence = min(
            (_quality(point, "presence") for point in quality_points),
            default=0.0,
        )
        if visibility < self.config.min_visibility:
            reasons.add("body_relative_low_visibility")
        if presence < self.config.min_presence:
            reasons.add("body_relative_low_presence")

        hip_center = _midpoint(arrays, "left_hip", "right_hip")
        shoulder_center = _midpoint(arrays, "left_shoulder", "right_shoulder")
        scale = _body_scale(arrays, hip_center, shoulder_center)
        if hip_center is None or scale is None:
            reasons.add("body_relative_scale_unavailable")
        reliable = not reasons
        relative = (
            {
                name: (point - hip_center) / scale
                for name, point in arrays.items()
            }
            if hip_center is not None and scale is not None
            else {}
        )

        timestamp = _finite_float(timestamp_ms)
        dt_seconds = None
        if timestamp is not None and self._previous_timestamp_ms is not None:
            delta_ms = timestamp - self._previous_timestamp_ms
            if 0.0 < delta_ms <= 500.0:
                dt_seconds = delta_ms / 1000.0
            else:
                reasons.add("body_relative_timestamp_gap")

        height: dict[str, float | None] = {}
        height_delta: dict[str, float | None] = {}
        for side in ("left", "right"):
            for joint in ("knee", "ankle", "heel", "foot_index"):
                name = f"{side}_{joint}"
                point = relative.get(name)
                previous = self._previous_relative.get(name)
                # MediaPipe world y is downward-positive.  Subtracting the hip
                # removes whole-body vertical translation.
                height[name] = None if point is None else float(point[1])
                height_delta[name] = (
                    None
                    if point is None or previous is None
                    else float(point[1] - previous[1])
                )

        foot_centers = {
            side: _mean_available(relative, FOOT_POINTS[side])
            for side in ("left", "right")
        }
        speeds: dict[str, float | None] = {}
        for side, center in foot_centers.items():
            previous = _mean_available(self._previous_relative, FOOT_POINTS[side])
            speed = (
                None
                if center is None or previous is None or dt_seconds is None
                else float(np.linalg.norm(center - previous) / dt_seconds)
            )
            speeds[side] = speed
            stationary = (
                speed is not None
                and speed <= self.config.stationary_speed_body_per_s
            )
            was_moving = self._moving[side]
            if stationary:
                self._stationary_frames[side] += 1
                if (
                    was_moving
                    and self._stationary_frames[side]
                    == self.config.stationary_dwell_frames
                ):
                    self._landing_ms[side] = timestamp
                    self._landing_count[side] += 1
                if self._stationary_frames[side] >= self.config.stationary_dwell_frames:
                    self._moving[side] = False
            elif speed is not None:
                if not was_moving and self._stationary_frames[side] >= (
                    self.config.stationary_dwell_frames
                ):
                    self._takeoff_ms[side] = timestamp
                    self._takeoff_count[side] += 1
                self._moving[side] = True
                self._stationary_frames[side] = 0

        left_depth = (
            None if foot_centers["left"] is None else float(foot_centers["left"][2])
        )
        right_depth = (
            None if foot_centers["right"] is None else float(foot_centers["right"][2])
        )
        depth_gap = (
            None
            if left_depth is None or right_depth is None
            else left_depth - right_depth
        )
        nearer_side = None
        if (
            reliable
            and depth_gap is not None
            and abs(depth_gap) >= self.config.depth_order_gap_body_ratio
        ):
            nearer_side = "left" if depth_gap < 0.0 else "right"
        view = str(camera_view).strip().lower().replace("-", "_")
        leading_hint = nearer_side if view in {"front", "front_left", "front_right"} else None
        trailing_hint = (
            None
            if leading_hint is None
            else "right" if leading_hint == "left" else "left"
        )

        torso_vector = (
            None
            if shoulder_center is None or hip_center is None or scale is None
            else (shoulder_center - hip_center) / scale
        )
        torso_norm = (
            None if torso_vector is None else float(np.linalg.norm(torso_vector))
        )
        prone_score = (
            None
            if torso_vector is None or torso_norm is None or torso_norm <= 1e-8
            else 1.0 - abs(float(torso_vector[1])) / torso_norm
        )
        shoulder_hip_twist = _segment_angle(
            arrays.get("left_shoulder"),
            arrays.get("right_shoulder"),
            arrays.get("left_hip"),
            arrays.get("right_hip"),
        )
        shoulder_width = _distance(arrays, "left_shoulder", "right_shoulder")
        hip_width = _distance(arrays, "left_hip", "right_hip")
        width_ratio = (
            None
            if shoulder_width is None or hip_width is None or hip_width <= 1e-8
            else shoulder_width / hip_width
        )
        takeoff_delta = _time_delta(self._takeoff_ms)
        landing_delta = _time_delta(self._landing_ms)

        payload = {
            "schema_version": 1,
            "coordinate_semantics": (
                "body_relative_mediapipe_world_not_metric_or_floor_referenced"
            ),
            "reliable": reliable,
            "quality_reasons": sorted(reasons),
            "minimum_visibility": visibility,
            "minimum_presence": presence,
            "body_scale": scale,
            "hip_compensated_height_downward_positive": height,
            "hip_compensated_height_delta": height_delta,
            "leg_depth_order": {
                "left_foot_z_body_ratio": left_depth,
                "right_foot_z_body_ratio": right_depth,
                "left_minus_right_body_ratio": depth_gap,
                "nearer_side": nearer_side,
                "leading_side_hint": leading_hint,
                "trailing_side_hint": trailing_hint,
                "confidence": (
                    0.0
                    if depth_gap is None
                    else min(1.0, abs(depth_gap) / max(
                        self.config.depth_order_gap_body_ratio * 2.0,
                        1e-8,
                    ))
                ),
            },
            "foot_motion": {
                "left_speed_body_per_s": speeds["left"],
                "right_speed_body_per_s": speeds["right"],
                "left_stationary_dwell_frames": self._stationary_frames["left"],
                "right_stationary_dwell_frames": self._stationary_frames["right"],
                "left_takeoff_ms": self._takeoff_ms["left"],
                "right_takeoff_ms": self._takeoff_ms["right"],
                "left_takeoff_count": self._takeoff_count["left"],
                "right_takeoff_count": self._takeoff_count["right"],
                "takeoff_time_difference_ms": takeoff_delta,
                "left_landing_ms": self._landing_ms["left"],
                "right_landing_ms": self._landing_ms["right"],
                "left_landing_count": self._landing_count["left"],
                "right_landing_count": self._landing_count["right"],
                "landing_time_difference_ms": landing_delta,
            },
            "torso_spatial": {
                "shoulder_hip_vector_body_ratio": (
                    None if torso_vector is None else torso_vector.tolist()
                ),
                "prone_horizontal_score": prone_score,
                "prone_hint": (
                    bool(prone_score >= self.config.prone_horizontal_score_min)
                    if prone_score is not None
                    else None
                ),
                "shoulder_hip_depth_offset_body_ratio": (
                    None if torso_vector is None else float(torso_vector[2])
                ),
                "shoulder_hip_twist_deg": shoulder_hip_twist,
                "shoulder_to_hip_width_ratio": width_ratio,
            },
            "thresholds": self.config.as_dict(),
            # Explicitly state the safety boundary in every frame artifact.
            "contact_inference_allowed": False,
            "validity_promotion_allowed": False,
        }
        if reliable and timestamp is not None:
            self._previous_timestamp_ms = timestamp
            self._previous_hip_center = hip_center
            self._previous_relative = relative
        return payload


def _point_map(points: Sequence[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for point in points:
        name = (
            point.get("name")
            if isinstance(point, Mapping)
            else getattr(point, "name", None)
        )
        if name:
            result[str(name)] = point
    return result


def _value(point: object, name: str) -> object:
    return point.get(name) if isinstance(point, Mapping) else getattr(point, name, None)


def _xyz(point: object | None) -> np.ndarray | None:
    if point is None:
        return None
    try:
        value = np.asarray(
            [float(_value(point, axis)) for axis in ("x", "y", "z")],
            dtype=float,
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value.shape == (3,) and np.all(np.isfinite(value)) else None


def _quality(point: object | None, name: str) -> float:
    if point is None:
        return 0.0
    raw = _value(point, name)
    if raw is None:
        raw = _value(point, "confidence")
    value = _finite_float(raw)
    return 0.0 if value is None else max(0.0, min(1.0, value))


def _finite_float(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if isfinite(resolved) else None


def _midpoint(
    points: Mapping[str, np.ndarray], first: str, second: str
) -> np.ndarray | None:
    point_a = points.get(first)
    point_b = points.get(second)
    return None if point_a is None or point_b is None else (point_a + point_b) / 2.0


def _distance(
    points: Mapping[str, np.ndarray], first: str, second: str
) -> float | None:
    point_a = points.get(first)
    point_b = points.get(second)
    return (
        None
        if point_a is None or point_b is None
        else float(np.linalg.norm(point_a - point_b))
    )


def _body_scale(
    points: Mapping[str, np.ndarray],
    hip_center: np.ndarray | None,
    shoulder_center: np.ndarray | None,
) -> float | None:
    values: list[float] = []
    if hip_center is not None and shoulder_center is not None:
        values.append(float(np.linalg.norm(shoulder_center - hip_center)))
    for side in ("left", "right"):
        for first, second in (("hip", "knee"), ("knee", "ankle")):
            distance = _distance(points, f"{side}_{first}", f"{side}_{second}")
            if distance is not None:
                values.append(distance)
    valid = [value for value in values if isfinite(value) and value > 1e-6]
    return float(median(valid)) if valid else None


def _mean_available(
    points: Mapping[str, np.ndarray], names: Sequence[str]
) -> np.ndarray | None:
    values = [points[name] for name in names if name in points]
    return None if not values else np.mean(values, axis=0)


def _time_delta(values: Mapping[str, float | None]) -> float | None:
    left = values.get("left")
    right = values.get("right")
    return None if left is None or right is None else abs(float(left) - float(right))


def _segment_angle(
    first_a: np.ndarray | None,
    first_b: np.ndarray | None,
    second_a: np.ndarray | None,
    second_b: np.ndarray | None,
) -> float | None:
    if any(value is None for value in (first_a, first_b, second_a, second_b)):
        return None
    first = first_b - first_a
    second = second_b - second_a
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norms <= 1e-8:
        return None
    cosine = float(np.clip(np.dot(first, second) / norms, -1.0, 1.0))
    return degrees(acos(cosine))


__all__ = [
    "BodyRelative3DTracker",
    "ShadowEvidence3DConfig",
]
