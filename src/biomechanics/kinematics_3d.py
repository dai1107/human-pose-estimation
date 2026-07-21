"""MediaPipe world-landmark kinematics for shadow and confidence-only assist modes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.product_pose import ThreeDKinematicsConfig, ThreeDQualityConfig


ANGLE_DEFINITIONS_3D: Mapping[str, tuple[str, str, str]] = {
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
}

IDENTITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)


@dataclass(frozen=True, slots=True)
class AngleMeasurement:
    angle_2d: float | None
    angle_3d: float | None
    selected_angle: float | None
    selected_source: str
    confidence: float
    three_d_reliable: bool
    difference_deg: float | None = None
    quality_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ThreeDKinematicsResult:
    enabled: bool
    decision_mode: str
    assist_status: str
    assist_confidence_boost: float
    assist_conflict_confidence_cap: float
    three_d_available: bool
    world_landmark_count: int
    three_d_reliable: bool
    three_d_reliable_ratio: float
    three_d_conflict_ratio: float
    measurements: Mapping[str, AngleMeasurement]
    quality_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        angles_2d: dict[str, float | None] = {}
        angles_3d: dict[str, float | None] = {}
        differences: dict[str, float | None] = {}
        reliability: dict[str, bool] = {}
        measurements: dict[str, dict[str, Any]] = {}
        flattened: dict[str, Any] = {}
        for name, measurement in self.measurements.items():
            angle_2d_name = f"{name}_2d"
            angle_3d_name = f"{name}_3d"
            difference_name = f"{name}_2d_3d_difference_deg"
            reliable_name = f"{name}_3d_reliable"
            angles_2d[angle_2d_name] = measurement.angle_2d
            angles_3d[angle_3d_name] = measurement.angle_3d
            differences[difference_name] = measurement.difference_deg
            reliability[reliable_name] = measurement.three_d_reliable
            measurements[name] = {
                "angle_2d": measurement.angle_2d,
                "angle_3d": measurement.angle_3d,
                "selected_angle": measurement.selected_angle,
                "selected_source": measurement.selected_source,
                "confidence": measurement.confidence,
                "three_d_reliable": measurement.three_d_reliable,
                "difference_deg": measurement.difference_deg,
                "quality_reasons": list(measurement.quality_reasons),
            }
            flattened.update(
                {
                    angle_2d_name: measurement.angle_2d,
                    angle_3d_name: measurement.angle_3d,
                    difference_name: measurement.difference_deg,
                    reliable_name: measurement.three_d_reliable,
                }
            )
        return {
            "enabled": self.enabled,
            "decision_mode": self.decision_mode,
            "assist_status": self.assist_status,
            "assist_confidence_boost": self.assist_confidence_boost,
            "assist_conflict_confidence_cap": self.assist_conflict_confidence_cap,
            "three_d_available": self.three_d_available,
            "world_landmark_count": self.world_landmark_count,
            "three_d_reliable": self.three_d_reliable,
            "three_d_reliable_ratio": self.three_d_reliable_ratio,
            "three_d_conflict_ratio": self.three_d_conflict_ratio,
            "angles_2d": angles_2d,
            "angles_3d": angles_3d,
            "angle_differences_deg": differences,
            "angle_reliability": reliability,
            "measurements": measurements,
            "quality_reasons": list(self.quality_reasons),
            **flattened,
        }


def calculate_angle_3d(
    point_a: np.ndarray,
    vertex: np.ndarray,
    point_b: np.ndarray,
) -> float | None:
    arrays = tuple(np.asarray(point, dtype=float) for point in (point_a, vertex, point_b))
    if any(array.shape != (3,) or not np.all(np.isfinite(array)) for array in arrays):
        return None
    vector_a = arrays[0] - arrays[1]
    vector_b = arrays[2] - arrays[1]
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return None
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    value = float(np.degrees(np.arccos(cosine)))
    return value if isfinite(value) else None


def calculate_angle_2d(
    point_a: np.ndarray,
    vertex: np.ndarray,
    point_b: np.ndarray,
) -> float | None:
    arrays = tuple(np.asarray(point, dtype=float) for point in (point_a, vertex, point_b))
    if any(array.shape != (2,) or not np.all(np.isfinite(array)) for array in arrays):
        return None
    vector_a = arrays[0] - arrays[1]
    vector_b = arrays[2] - arrays[1]
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return None
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    value = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return value if isfinite(value) else None


class ThreeDKinematicsTracker:
    """Evaluate world landmarks while keeping every selected angle strictly 2D."""

    def __init__(
        self,
        kinematics_config: ThreeDKinematicsConfig | None = None,
        quality_config: ThreeDQualityConfig | None = None,
        *,
        max_pose_age_ms: float = 150.0,
    ) -> None:
        self.kinematics_config = kinematics_config or ThreeDKinematicsConfig()
        self.quality_config = quality_config or ThreeDQualityConfig()
        self.max_pose_age_ms = max(0.0, float(max_pose_age_ms))
        self._previous_timestamp_ns: int | None = None
        self._previous_world: dict[str, np.ndarray] = {}
        self._previous_bone_lengths: dict[tuple[str, str], float] = {}
        self._previous_angles: dict[str, float] = {}

    def reset(self) -> None:
        self._previous_timestamp_ns = None
        self._previous_world.clear()
        self._previous_bone_lengths.clear()
        self._previous_angles.clear()

    def update(
        self,
        result: PoseResult,
        *,
        capture_timestamp_ns: int | None = None,
        pose_age_ms: float = 0.0,
    ) -> ThreeDKinematicsResult:
        image_points = _point_map(result.keypoints)
        raw_world = result.extra.get("world_keypoints")
        world_points = _point_map(raw_world if isinstance(raw_world, (list, tuple)) else ())
        timestamp_ns = _resolve_timestamp_ns(result.timestamp_ms, capture_timestamp_ns)
        world_arrays = {
            name: array
            for name, point in world_points.items()
            if (array := _xyz(point)) is not None
        }
        world_available = bool(world_arrays)

        gap_exceeded = False
        dt_seconds: float | None = None
        if timestamp_ns is not None and self._previous_timestamp_ns is not None:
            gap_ns = timestamp_ns - self._previous_timestamp_ns
            threshold_ns = int(self.quality_config.max_gap_ms_before_reset * 1_000_000.0)
            gap_exceeded = gap_ns <= 0 or gap_ns > threshold_ns
            if not gap_exceeded:
                dt_seconds = gap_ns / 1_000_000_000.0
        if gap_exceeded:
            self.reset()

        pose_too_old = pose_age_ms > self.max_pose_age_ms
        identity_swapped = self._identity_swapped(world_arrays)
        bone_lengths = _bone_lengths(world_arrays)
        body_scale = _body_scale(bone_lengths, self._previous_bone_lengths)
        global_reasons: set[str] = set()
        if not self.kinematics_config.enabled:
            global_reasons.add("three_d_disabled")
        if not world_available:
            global_reasons.add("world_landmarks_missing")
        if pose_too_old:
            global_reasons.add("pose_too_old")
        if gap_exceeded:
            global_reasons.add("world_gap_exceeded")
        if identity_swapped:
            global_reasons.add("left_right_identity_swap")

        measurements: dict[str, AngleMeasurement] = {}
        severe_temporal_failure = False
        for name, definition in ANGLE_DEFINITIONS_3D.items():
            reasons = set(global_reasons)
            image_triplet = tuple(image_points.get(point_name) for point_name in definition)
            world_triplet = tuple(world_arrays.get(point_name) for point_name in definition)
            angle_2d = _angle_2d_from_points(image_triplet)
            angle_3d = _angle_3d_from_arrays(world_triplet)
            confidence = _triplet_confidence(image_triplet)

            if any(point is None for point in image_triplet):
                reasons.add("image_joint_missing")
            if any(point is None for point in world_triplet):
                reasons.add("world_joint_missing")
            if angle_3d is None:
                reasons.add("invalid_world_geometry")
            visibility, presence = _triplet_quality(image_triplet)
            if visibility < self.quality_config.min_visibility:
                reasons.add("low_visibility")
            if presence < self.quality_config.min_presence:
                reasons.add("low_presence")

            for segment in ((definition[0], definition[1]), (definition[1], definition[2])):
                current_length = bone_lengths.get(segment)
                previous_length = self._previous_bone_lengths.get(segment)
                if current_length is None or current_length <= 1e-8:
                    reasons.add("invalid_bone_length")
                elif previous_length is not None and previous_length > 1e-8:
                    change_ratio = abs(current_length - previous_length) / previous_length
                    if change_ratio > self.quality_config.max_bone_length_change_ratio:
                        reasons.add("bone_length_jump")
                        severe_temporal_failure = True

            if body_scale is not None and self._previous_world:
                for point_name in definition:
                    current = world_arrays.get(point_name)
                    previous = self._previous_world.get(point_name)
                    if current is not None and previous is not None:
                        z_change = abs(float(current[2] - previous[2])) / body_scale
                        if z_change > self.quality_config.max_z_change_body_scale:
                            reasons.add("z_jump")
                            severe_temporal_failure = True

            previous_angle = self._previous_angles.get(name)
            if angle_3d is not None and previous_angle is not None:
                angle_delta = abs(angle_3d - previous_angle)
                if angle_delta > self.quality_config.max_angle_delta_deg:
                    reasons.add("angle_jump")
                if (
                    dt_seconds is not None
                    and angle_delta / dt_seconds
                    > self.quality_config.max_angular_velocity_deg_s
                ):
                    reasons.add("angular_velocity_exceeded")

            difference = (
                abs(angle_2d - angle_3d)
                if angle_2d is not None and angle_3d is not None
                else None
            )
            if (
                difference is not None
                and difference > self.quality_config.max_2d_3d_difference_deg
            ):
                reasons.add("two_d_three_d_conflict")

            reliable = not reasons
            selected_source = (
                "2d_assist"
                if self.kinematics_config.decision_mode == "assist"
                else "2d_shadow"
            )
            measurements[name] = AngleMeasurement(
                angle_2d=angle_2d,
                angle_3d=angle_3d,
                selected_angle=angle_2d,
                selected_source=selected_source if angle_2d is not None else "none",
                confidence=confidence,
                three_d_reliable=reliable,
                difference_deg=difference,
                quality_reasons=tuple(sorted(reasons)),
            )

        reliable_count = sum(
            1 for measurement in measurements.values() if measurement.three_d_reliable
        )
        reliable_ratio = reliable_count / len(measurements) if measurements else 0.0
        conflict_count = sum(
            "two_d_three_d_conflict" in measurement.quality_reasons
            for measurement in measurements.values()
        )
        conflict_ratio = conflict_count / len(measurements) if measurements else 0.0
        if self.kinematics_config.decision_mode != "assist":
            assist_status = "shadow"
        elif not self.kinematics_config.enabled:
            assist_status = "disabled"
        elif conflict_count:
            assist_status = "conflict"
        elif reliable_count:
            assist_status = "supporting"
        else:
            assist_status = "fallback_2d"
        all_reasons = set(global_reasons)
        for measurement in measurements.values():
            all_reasons.update(measurement.quality_reasons)

        if (
            self.kinematics_config.enabled
            and world_available
            and not pose_too_old
            and not identity_swapped
            and not severe_temporal_failure
        ):
            self._previous_timestamp_ns = timestamp_ns
            self._previous_world = world_arrays
            self._previous_bone_lengths = bone_lengths
            self._previous_angles = {
                name: measurement.angle_3d
                for name, measurement in measurements.items()
                if measurement.angle_3d is not None
            }

        return ThreeDKinematicsResult(
            enabled=self.kinematics_config.enabled,
            decision_mode=self.kinematics_config.decision_mode,
            assist_status=assist_status,
            assist_confidence_boost=self.kinematics_config.assist_confidence_boost,
            assist_conflict_confidence_cap=(
                self.kinematics_config.assist_conflict_confidence_cap
            ),
            three_d_available=world_available,
            world_landmark_count=len(world_arrays),
            three_d_reliable=bool(measurements) and reliable_count == len(measurements),
            three_d_reliable_ratio=reliable_ratio,
            three_d_conflict_ratio=conflict_ratio,
            measurements=measurements,
            quality_reasons=tuple(sorted(all_reasons)),
        )

    def attach(
        self,
        result: PoseResult,
        *,
        capture_timestamp_ns: int | None = None,
        pose_age_ms: float = 0.0,
    ) -> tuple[PoseResult, ThreeDKinematicsResult]:
        kinematics = self.update(
            result,
            capture_timestamp_ns=capture_timestamp_ns,
            pose_age_ms=pose_age_ms,
        )
        extra = dict(result.extra)
        extra["three_d_kinematics"] = kinematics.as_dict()
        return replace(result, extra=extra), kinematics

    def _identity_swapped(self, current: Mapping[str, np.ndarray]) -> bool:
        same_cost = 0.0
        swapped_cost = 0.0
        pair_count = 0
        for left_name, right_name in IDENTITY_PAIRS:
            current_left = current.get(left_name)
            current_right = current.get(right_name)
            previous_left = self._previous_world.get(left_name)
            previous_right = self._previous_world.get(right_name)
            if any(
                point is None
                for point in (current_left, current_right, previous_left, previous_right)
            ):
                continue
            same_cost += float(np.linalg.norm(current_left - previous_left))
            same_cost += float(np.linalg.norm(current_right - previous_right))
            swapped_cost += float(np.linalg.norm(current_left - previous_right))
            swapped_cost += float(np.linalg.norm(current_right - previous_left))
            pair_count += 1
        return (
            pair_count >= 2
            and swapped_cost + 1e-8
            < same_cost * self.quality_config.identity_swap_cost_ratio
        )


def summarize_three_d_records(records: Iterable[object]) -> dict[str, Any]:
    resolved: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for record in records:
        if isinstance(record, Mapping):
            shadow = record.get("three_d_kinematics")
            context = record
        else:
            shadow = getattr(record, "three_d_kinematics", None)
            context = {
                "camera_view": getattr(record, "camera_view", None),
            }
        if isinstance(shadow, Mapping):
            resolved.append((shadow, context))

    summary = _summarize_shadow_items([shadow for shadow, _ in resolved])
    for field_name in ("action", "camera_view", "phase"):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for shadow, context in resolved:
            label = context.get(field_name)
            if label not in (None, ""):
                grouped.setdefault(str(label), []).append(shadow)
        summary[f"by_{field_name}"] = {
            label: _summarize_shadow_items(items)
            for label, items in sorted(grouped.items())
        }
    return summary


def _summarize_shadow_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = sum(bool(item.get("three_d_available")) for item in items)
    reliable = sum(bool(item.get("three_d_reliable")) for item in items)
    reliable_ratios = [
        float(item.get("three_d_reliable_ratio", 0.0))
        for item in items
        if isinstance(item.get("three_d_reliable_ratio"), (int, float))
    ]
    differences: dict[str, list[float]] = {}
    reason_counts: Counter[str] = Counter()
    decision_mode_counts: Counter[str] = Counter()
    assist_status_counts: Counter[str] = Counter()
    conflict_ratios: list[float] = []
    for item in items:
        decision_mode_counts.update((str(item.get("decision_mode", "unknown")),))
        assist_status_counts.update((str(item.get("assist_status", "unknown")),))
        conflict_ratio = item.get("three_d_conflict_ratio")
        if isinstance(conflict_ratio, (int, float)) and isfinite(float(conflict_ratio)):
            conflict_ratios.append(float(conflict_ratio))
        raw_differences = item.get("angle_differences_deg")
        if isinstance(raw_differences, Mapping):
            for name, value in raw_differences.items():
                if isinstance(value, (int, float)) and isfinite(float(value)):
                    differences.setdefault(str(name), []).append(float(value))
        reasons = item.get("quality_reasons")
        if isinstance(reasons, (list, tuple)):
            reason_counts.update(str(reason) for reason in reasons)
    return {
        "frame_count": len(items),
        "world_landmarks_availability_ratio": available / len(items) if items else 0.0,
        "fully_reliable_frame_ratio": reliable / len(items) if items else 0.0,
        "mean_reliable_angle_ratio": (
            float(np.mean(reliable_ratios)) if reliable_ratios else 0.0
        ),
        "mean_conflict_angle_ratio": (
            float(np.mean(conflict_ratios)) if conflict_ratios else 0.0
        ),
        "decision_modes": dict(sorted(decision_mode_counts.items())),
        "assist_statuses": dict(sorted(assist_status_counts.items())),
        "angle_difference_deg": {
            name: {
                "count": len(values),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
            for name, values in sorted(differences.items())
        },
        "failure_reasons": dict(sorted(reason_counts.items())),
    }


def _point_map(points: Sequence[object]) -> dict[str, object]:
    return {
        str(name): point
        for point in points
        if (name := getattr(point, "name", None))
    }


def _xyz(point: object | None) -> np.ndarray | None:
    if point is None:
        return None
    try:
        array = np.array(
            [float(getattr(point, axis)) for axis in ("x", "y", "z")],
            dtype=float,
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return array if np.all(np.isfinite(array)) else None


def _xy(point: object | None) -> np.ndarray | None:
    if point is None:
        return None
    try:
        array = np.array([float(getattr(point, "x")), float(getattr(point, "y"))])
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return array if np.all(np.isfinite(array)) else None


def _angle_2d_from_points(points: Sequence[object | None]) -> float | None:
    arrays = tuple(_xy(point) for point in points)
    if any(array is None for array in arrays):
        return None
    return calculate_angle_2d(arrays[0], arrays[1], arrays[2])


def _angle_3d_from_arrays(points: Sequence[np.ndarray | None]) -> float | None:
    if any(point is None for point in points):
        return None
    return calculate_angle_3d(points[0], points[1], points[2])


def _quality_value(point: object | None, name: str) -> float:
    if point is None:
        return 0.0
    fallback = getattr(point, "confidence", 0.0)
    raw = getattr(point, name, None)
    raw = fallback if raw is None else raw
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(1.0, value)) if isfinite(value) else 0.0


def _triplet_quality(points: Sequence[object | None]) -> tuple[float, float]:
    return (
        min((_quality_value(point, "visibility") for point in points), default=0.0),
        min((_quality_value(point, "presence") for point in points), default=0.0),
    )


def _triplet_confidence(points: Sequence[object | None]) -> float:
    visibility, presence = _triplet_quality(points)
    return min(visibility, presence)


def _bone_lengths(world: Mapping[str, np.ndarray]) -> dict[tuple[str, str], float]:
    lengths: dict[tuple[str, str], float] = {}
    for first, middle, third in ANGLE_DEFINITIONS_3D.values():
        for segment in ((first, middle), (middle, third)):
            point_a = world.get(segment[0])
            point_b = world.get(segment[1])
            if point_a is not None and point_b is not None:
                lengths[segment] = float(np.linalg.norm(point_a - point_b))
    return lengths


def _body_scale(
    current: Mapping[tuple[str, str], float],
    previous: Mapping[tuple[str, str], float],
) -> float | None:
    values = [value for value in current.values() if isfinite(value) and value > 1e-8]
    if not values:
        values = [value for value in previous.values() if isfinite(value) and value > 1e-8]
    return float(np.median(values)) if values else None


def _resolve_timestamp_ns(timestamp_ms: int | None, timestamp_ns: int | None) -> int | None:
    if timestamp_ns is not None:
        return int(timestamp_ns)
    return None if timestamp_ms is None else int(timestamp_ms) * 1_000_000


__all__ = [
    "ANGLE_DEFINITIONS_3D",
    "AngleMeasurement",
    "ThreeDKinematicsResult",
    "ThreeDKinematicsTracker",
    "calculate_angle_2d",
    "calculate_angle_3d",
    "summarize_three_d_records",
]
