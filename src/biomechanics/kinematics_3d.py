"""MediaPipe world-landmark kinematics for shadow and confidence-only assist modes."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.product_pose import ThreeDKinematicsConfig, ThreeDQualityConfig
from src.biomechanics.body_coordinates import build_body_coordinate_system
from src.biomechanics.ground_estimation import GroundEstimator, GroundEstimatorConfig
from src.biomechanics.joint_metrics import (
    ANGLE_DEFINITIONS,
    JointMetric,
    calculate_angle_2d,
    calculate_angle_3d,
    select_joint_metric,
)
from src.biomechanics.shadow_evidence_3d import (
    BodyRelative3DTracker,
    ShadowEvidence3DConfig,
)


ANGLE_DEFINITIONS_3D = ANGLE_DEFINITIONS

IDENTITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)


AngleMeasurement = JointMetric


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
    measurements: Mapping[str, JointMetric]
    quality_reasons: tuple[str, ...]
    body_relative: Mapping[str, Any]
    body_coordinate_system: Mapping[str, Any]
    reliability: Mapping[str, Any]
    foot_contact_evidence: Mapping[str, Any]
    ground_estimation: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        angles_2d: dict[str, float | None] = {}
        angles_3d: dict[str, float | None] = {}
        differences: dict[str, float | None] = {}
        reliability: dict[str, bool] = {}
        measurements: dict[str, dict[str, Any]] = {}
        flattened: dict[str, Any] = {}
        canonical_angles = self.body_coordinate_system.get("canonical_3d_angles", {})
        if not isinstance(canonical_angles, Mapping):
            canonical_angles = {}
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
                "raw_2d": measurement.raw_2d,
                "smooth_2d": measurement.smooth_2d,
                "raw_3d": measurement.raw_3d,
                "smooth_3d": measurement.smooth_3d,
                "selected_value": measurement.selected_value,
                "source": measurement.source,
                "observable": measurement.observable,
                "legacy_angle": measurement.selected_angle,
                "canonical_3d_angle": canonical_angles.get(name),
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
                    f"{name}_legacy_angle": measurement.selected_angle,
                    f"{name}_canonical_3d_angle": canonical_angles.get(name),
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
            "body_relative": dict(self.body_relative),
            "body_coordinate_system": dict(self.body_coordinate_system),
            "canonical_3d_angles": dict(canonical_angles),
            "reliability": dict(self.reliability),
            "foot_contact_evidence": dict(self.foot_contact_evidence),
            "ground_estimation": dict(self.ground_estimation),
            "ground_confidence": self.ground_estimation.get(
                "ground_confidence", 0.0
            ),
            "contact_evidence": dict(
                self.ground_estimation.get("contact_evidence", {})
            ) if isinstance(
                self.ground_estimation.get("contact_evidence"), Mapping
            ) else {},
            "quality_reasons": list(self.quality_reasons),
            **flattened,
        }


class ThreeDKinematicsTracker:
    """Build unified joint metrics and lightweight temporal 3D reliability."""

    def __init__(
        self,
        kinematics_config: ThreeDKinematicsConfig | None = None,
        quality_config: ThreeDQualityConfig | None = None,
        *,
        max_pose_age_ms: float = 150.0,
        shadow_evidence_config: ShadowEvidence3DConfig | None = None,
    ) -> None:
        self.kinematics_config = kinematics_config or ThreeDKinematicsConfig()
        self.quality_config = quality_config or ThreeDQualityConfig()
        self.max_pose_age_ms = max(0.0, float(max_pose_age_ms))
        self._previous_timestamp_ns: int | None = None
        self._previous_world: dict[str, np.ndarray] = {}
        self._previous_bone_lengths: dict[tuple[str, str], float] = {}
        self._bone_length_history: dict[tuple[str, str], deque[float]] = {}
        self._previous_angles: dict[str, float] = {}
        self._foot_centers: dict[str, np.ndarray] = {}
        self._foot_stable_frames = {"left": 0, "right": 0}
        self._ground_estimator = GroundEstimator(
            GroundEstimatorConfig(
                history_size=self.quality_config.ground_history_size,
                minimum_samples=self.quality_config.ground_minimum_samples,
                minimum_contact_confidence=(
                    self.quality_config.ground_minimum_contact_confidence
                ),
                maximum_image_deviation=(
                    self.quality_config.ground_maximum_image_deviation
                ),
                maximum_world_vertical_deviation_m=(
                    self.quality_config.ground_maximum_world_vertical_deviation_m
                ),
            )
        )
        self._body_relative = BodyRelative3DTracker(shadow_evidence_config)

    def reset(self) -> None:
        self._previous_timestamp_ns = None
        self._previous_world.clear()
        self._previous_bone_lengths.clear()
        self._bone_length_history.clear()
        self._previous_angles.clear()
        self._foot_centers.clear()
        self._foot_stable_frames = {"left": 0, "right": 0}
        self._ground_estimator.reset()
        self._body_relative.reset()

    def update(
        self,
        result: PoseResult,
        *,
        capture_timestamp_ns: int | None = None,
        pose_age_ms: float = 0.0,
        image_width: int | float | None = None,
        image_height: int | float | None = None,
        raw_result: PoseResult | None = None,
        camera_view: str | None = None,
    ) -> ThreeDKinematicsResult:
        image_points = _point_map(result.keypoints)
        raw_world = result.extra.get("world_keypoints")
        world_points = _point_map(raw_world if isinstance(raw_world, (list, tuple)) else ())
        source_raw_result = raw_result or result
        raw_image_points = _point_map(source_raw_result.keypoints)
        source_raw_world = source_raw_result.extra.get("world_keypoints")
        raw_world_points = _point_map(
            source_raw_world if isinstance(source_raw_world, (list, tuple)) else ()
        )
        timestamp_ns = _resolve_timestamp_ns(result.timestamp_ms, capture_timestamp_ns)
        world_arrays = {
            name: array
            for name, point in world_points.items()
            if (array := _xyz(point)) is not None
        }
        raw_world_arrays = {
            name: array
            for name, point in raw_world_points.items()
            if (array := _xyz(point)) is not None
        }
        world_available = bool(world_arrays)
        resolved_camera_view = str(
            camera_view
            if camera_view is not None
            else result.extra.get("camera_view", "unknown")
        )
        body_relative = self._body_relative.update(
            result.keypoints,
            raw_world if isinstance(raw_world, (list, tuple)) else (),
            timestamp_ms=result.timestamp_ms,
            camera_view=resolved_camera_view,
        )
        body_coordinate_system = build_body_coordinate_system(
            raw_world if isinstance(raw_world, (list, tuple)) else (),
            quality_points=result.keypoints,
            minimum_quality=self.quality_config.min_visibility,
        ).as_dict()

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
        reliability_world = raw_world_arrays or world_arrays
        identity_swapped = self._identity_swapped(reliability_world)
        bone_lengths = _bone_lengths(reliability_world)
        body_scale = _body_scale(bone_lengths, self._previous_bone_lengths)
        historical_bone_lengths = {
            segment: float(np.median(tuple(values)))
            for segment, values in self._bone_length_history.items()
            if values
        }
        bilateral_mismatches = _bilateral_bone_mismatches(
            bone_lengths,
            self.quality_config.max_left_right_bone_ratio,
        )
        landmark_speeds, isolated_velocity_joints = _landmark_velocity_outliers(
            reliability_world,
            self._previous_world,
            dt_seconds=dt_seconds,
            max_speed_m_s=self.quality_config.max_landmark_speed_m_s,
            isolated_ratio=self.quality_config.isolated_velocity_ratio,
        )
        foot_contact_evidence = self._foot_contact_evidence(
            image_points=image_points,
            world=reliability_world,
            dt_seconds=dt_seconds,
            body_scale=body_scale,
        )
        ground_estimation = self._ground_estimator.update(
            result.keypoints,
            raw_world if isinstance(raw_world, (list, tuple)) else (),
            foot_contact_evidence,
        )
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

        measurements: dict[str, JointMetric] = {}
        severe_temporal_failure = False
        for name, definition in ANGLE_DEFINITIONS_3D.items():
            reasons = set(global_reasons)
            image_triplet = tuple(image_points.get(point_name) for point_name in definition)
            world_triplet = tuple(world_arrays.get(point_name) for point_name in definition)
            raw_image_triplet = tuple(
                raw_image_points.get(point_name) for point_name in definition
            )
            raw_world_triplet = tuple(
                raw_world_arrays.get(point_name) for point_name in definition
            )
            smooth_2d = _angle_2d_from_points(
                image_triplet,
                image_width=image_width,
                image_height=image_height,
            )
            raw_2d = _angle_2d_from_points(
                raw_image_triplet,
                image_width=image_width,
                image_height=image_height,
            )
            smooth_3d = _angle_3d_from_arrays(world_triplet)
            raw_3d = _angle_3d_from_arrays(raw_world_triplet)
            confidence = _triplet_confidence(image_triplet)

            if any(point is None for point in image_triplet):
                reasons.add("image_joint_missing")
            if any(point is None for point in world_triplet):
                reasons.add("world_joint_missing")
            if smooth_3d is None:
                reasons.add("invalid_world_geometry")
            visibility, presence = _triplet_quality(image_triplet)
            if visibility < self.quality_config.min_visibility:
                reasons.add("low_visibility")
            if presence < self.quality_config.min_presence:
                reasons.add("low_presence")

            for segment in ((definition[0], definition[1]), (definition[1], definition[2])):
                segment_key = _segment_key(*segment)
                current_length = bone_lengths.get(segment_key)
                historical_length = historical_bone_lengths.get(segment_key)
                if current_length is None or current_length <= 1e-8:
                    reasons.add("invalid_bone_length")
                elif historical_length is not None and historical_length > 1e-8:
                    change_ratio = abs(current_length - historical_length) / historical_length
                    if change_ratio > self.quality_config.max_bone_length_change_ratio:
                        reasons.add("bone_length_jump")
                        severe_temporal_failure = True
                if segment_key in bilateral_mismatches:
                    reasons.add("left_right_bone_mismatch")

            if any(point_name in isolated_velocity_joints for point_name in definition):
                reasons.add("isolated_landmark_velocity")
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
            if smooth_3d is not None and previous_angle is not None:
                angle_delta = abs(smooth_3d - previous_angle)
                if angle_delta > self.quality_config.max_angle_delta_deg:
                    reasons.add("angle_jump")
                if (
                    dt_seconds is not None
                    and angle_delta / dt_seconds
                    > self.quality_config.max_angular_velocity_deg_s
                ):
                    reasons.add("angular_velocity_exceeded")

            difference = (
                abs(smooth_2d - smooth_3d)
                if smooth_2d is not None and smooth_3d is not None
                else None
            )
            if (
                difference is not None
                and difference > self.quality_config.max_2d_3d_difference_deg
            ):
                reasons.add("two_d_three_d_conflict")

            reliable = not reasons
            measurements[name] = select_joint_metric(
                name=name,
                raw_2d=raw_2d,
                smooth_2d=smooth_2d,
                raw_3d=raw_3d,
                smooth_3d=smooth_3d,
                three_d_reliable=reliable,
                confidence=confidence,
                camera_view=resolved_camera_view,
                decision_mode=self.kinematics_config.decision_mode,
                quality_reasons=reasons,
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
            self._previous_world = dict(reliability_world)
            self._previous_bone_lengths = {
                segment: length
                for segment, length in bone_lengths.items()
                if segment not in bilateral_mismatches
            }
            for segment, length in bone_lengths.items():
                if segment in bilateral_mismatches:
                    continue
                history = self._bone_length_history.setdefault(
                    segment,
                    deque(maxlen=self.quality_config.bone_length_history_size),
                )
                history.append(length)
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
            body_relative=body_relative,
            body_coordinate_system=body_coordinate_system,
            reliability={
                "schema_version": 1,
                "bone_length_history_samples": {
                    _segment_label(segment): len(values)
                    for segment, values in sorted(self._bone_length_history.items())
                },
                "bone_length_historical_median_m": {
                    _segment_label(segment): value
                    for segment, value in sorted(historical_bone_lengths.items())
                },
                "left_right_mismatch_segments": [
                    _segment_label(segment) for segment in sorted(bilateral_mismatches)
                ],
                "landmark_speed_m_s": dict(sorted(landmark_speeds.items())),
                "isolated_velocity_joints": sorted(isolated_velocity_joints),
                "confidence_only": True,
                "position_correction_applied": False,
            },
            foot_contact_evidence=foot_contact_evidence,
            ground_estimation=ground_estimation,
        )

    def attach(
        self,
        result: PoseResult,
        *,
        capture_timestamp_ns: int | None = None,
        pose_age_ms: float = 0.0,
        image_width: int | float | None = None,
        image_height: int | float | None = None,
        raw_result: PoseResult | None = None,
        camera_view: str | None = None,
    ) -> tuple[PoseResult, ThreeDKinematicsResult]:
        kinematics = self.update(
            result,
            capture_timestamp_ns=capture_timestamp_ns,
            pose_age_ms=pose_age_ms,
            image_width=image_width,
            image_height=image_height,
            raw_result=raw_result,
            camera_view=camera_view,
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

    def _foot_contact_evidence(
        self,
        *,
        image_points: Mapping[str, object],
        world: Mapping[str, np.ndarray],
        dt_seconds: float | None,
        body_scale: float | None,
    ) -> dict[str, Any]:
        """Estimate confidence-only foot contact evidence.

        This is deliberately not a ground-plane or rule decision.  It combines
        ankle/heel/toe vertical coherence, velocity and stable-frame dwell.
        """

        evidence: dict[str, Any] = {
            "schema_version": 1,
            "evidence_only": True,
            "formal_rule_replacement_allowed": False,
        }
        scale = body_scale if body_scale is not None and body_scale > 1e-8 else None
        for side in ("left", "right"):
            names = (f"{side}_ankle", f"{side}_heel", f"{side}_foot_index")
            points = [world.get(name) for name in names]
            valid = [point for point in points if point is not None]
            center = np.mean(valid, axis=0) if len(valid) == len(names) else None
            previous = self._foot_centers.get(side)
            speed_m_s = (
                None
                if center is None or previous is None or dt_seconds is None
                else float(np.linalg.norm(center - previous) / dt_seconds)
            )
            support_points = [world.get(name) for name in names[1:]]
            vertical_spread = (
                None
                if center is None
                or scale is None
                or any(point is None for point in support_points)
                else float(
                    abs(support_points[0][1] - support_points[1][1]) / scale
                )
            )
            quality = min(
                (_quality_value(image_points.get(name), "visibility") for name in names),
                default=0.0,
            )
            stable = (
                speed_m_s is not None
                and speed_m_s <= self.quality_config.foot_stationary_speed_m_s
                and vertical_spread is not None
                and vertical_spread
                <= self.quality_config.foot_vertical_spread_body_ratio
                and quality >= self.quality_config.min_visibility
            )
            self._foot_stable_frames[side] = (
                self._foot_stable_frames[side] + 1 if stable else 0
            )
            dwell_score = min(
                1.0,
                self._foot_stable_frames[side]
                / max(1, self.quality_config.foot_contact_stable_frames),
            )
            speed_score = (
                0.0
                if speed_m_s is None
                else max(
                    0.0,
                    1.0
                    - speed_m_s
                    / max(self.quality_config.foot_stationary_speed_m_s, 1e-8),
                )
            )
            spread_score = (
                0.0
                if vertical_spread is None
                else max(
                    0.0,
                    1.0
                    - vertical_spread
                    / max(
                        self.quality_config.foot_vertical_spread_body_ratio,
                        1e-8,
                    ),
                )
            )
            confidence = max(
                0.0,
                min(1.0, quality * dwell_score * speed_score * spread_score),
            )
            evidence[side] = {
                "observable": center is not None and scale is not None,
                "foot_contact_confidence": confidence,
                "stable_frames": self._foot_stable_frames[side],
                "vertical_spread_body_ratio": vertical_spread,
                "velocity_m_s": speed_m_s,
                "minimum_visibility": quality,
                "likely_contact": confidence >= 0.60,
            }
            if center is not None:
                self._foot_centers[side] = center
        return evidence


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
    body_coordinate_available = 0
    body_coordinate_reliable = 0
    ground_ready = 0
    ground_confidences: list[float] = []
    ground_contact_statuses: Counter[str] = Counter()
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
        body_coordinates = item.get("body_coordinate_system")
        if isinstance(body_coordinates, Mapping):
            body_coordinate_available += int(bool(body_coordinates.get("available")))
            body_coordinate_reliable += int(bool(body_coordinates.get("reliable")))
        ground = item.get("ground_estimation")
        if isinstance(ground, Mapping):
            ground_ready += int(str(ground.get("status")) == "READY")
            confidence = ground.get("ground_confidence")
            if isinstance(confidence, (int, float)) and isfinite(float(confidence)):
                ground_confidences.append(float(confidence))
            contacts = ground.get("contact_evidence")
            if isinstance(contacts, Mapping):
                for side, evidence in contacts.items():
                    if isinstance(evidence, Mapping):
                        ground_contact_statuses.update(
                            (f"{side}:{evidence.get('status', 'UNSURE')}",)
                        )
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
        "body_canonical_available_ratio": (
            body_coordinate_available / len(items) if items else 0.0
        ),
        "body_canonical_reliable_ratio": (
            body_coordinate_reliable / len(items) if items else 0.0
        ),
        "ground_ready_ratio": ground_ready / len(items) if items else 0.0,
        "mean_ground_confidence": (
            float(np.mean(ground_confidences)) if ground_confidences else 0.0
        ),
        "ground_contact_evidence_statuses": dict(
            sorted(ground_contact_statuses.items())
        ),
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


def _angle_2d_from_points(
    points: Sequence[object | None],
    *,
    image_width: int | float | None = None,
    image_height: int | float | None = None,
) -> float | None:
    arrays = tuple(_xy(point) for point in points)
    if any(array is None for array in arrays):
        return None
    return calculate_angle_2d(
        arrays[0],
        arrays[1],
        arrays[2],
        image_width,
        image_height,
    )


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
                lengths[_segment_key(*segment)] = float(np.linalg.norm(point_a - point_b))
    return lengths


def _segment_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))  # type: ignore[return-value]


def _segment_label(segment: tuple[str, str]) -> str:
    return f"{segment[0]}__{segment[1]}"


def _bilateral_bone_mismatches(
    lengths: Mapping[tuple[str, str], float],
    maximum_ratio: float,
) -> set[tuple[str, str]]:
    grouped: dict[tuple[str, str], dict[str, tuple[tuple[str, str], float]]] = {}
    for segment, length in lengths.items():
        sides = {
            name.split("_", 1)[0]
            for name in segment
            if name.startswith(("left_", "right_"))
        }
        if len(sides) != 1:
            continue
        side = next(iter(sides))
        signature = tuple(sorted(name.split("_", 1)[1] for name in segment))
        grouped.setdefault(signature, {})[side] = (segment, length)
    mismatches: set[tuple[str, str]] = set()
    for values in grouped.values():
        left = values.get("left")
        right = values.get("right")
        if left is None or right is None:
            continue
        denominator = max((left[1] + right[1]) / 2.0, 1e-8)
        if abs(left[1] - right[1]) / denominator > maximum_ratio:
            mismatches.update((left[0], right[0]))
    return mismatches


_LANDMARK_NEIGHBORS: Mapping[str, tuple[str, ...]] = {
    "left_shoulder": ("left_elbow", "left_hip", "right_shoulder"),
    "right_shoulder": ("right_elbow", "right_hip", "left_shoulder"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_wrist": ("left_elbow",),
    "right_wrist": ("right_elbow",),
    "left_hip": ("left_shoulder", "left_knee", "right_hip"),
    "right_hip": ("right_shoulder", "right_knee", "left_hip"),
    "left_knee": ("left_hip", "left_ankle"),
    "right_knee": ("right_hip", "right_ankle"),
    "left_ankle": ("left_knee", "left_heel", "left_foot_index"),
    "right_ankle": ("right_knee", "right_heel", "right_foot_index"),
    "left_heel": ("left_ankle", "left_foot_index"),
    "right_heel": ("right_ankle", "right_foot_index"),
    "left_foot_index": ("left_ankle", "left_heel"),
    "right_foot_index": ("right_ankle", "right_heel"),
}


def _landmark_velocity_outliers(
    current: Mapping[str, np.ndarray],
    previous: Mapping[str, np.ndarray],
    *,
    dt_seconds: float | None,
    max_speed_m_s: float,
    isolated_ratio: float,
) -> tuple[dict[str, float], set[str]]:
    if dt_seconds is None or dt_seconds <= 0.0:
        return {}, set()
    speeds = {
        name: float(np.linalg.norm(point - previous[name]) / dt_seconds)
        for name, point in current.items()
        if name in previous
    }
    outliers: set[str] = set()
    for name, speed in speeds.items():
        if speed <= max_speed_m_s:
            continue
        neighbor_speeds = [
            speeds[neighbor]
            for neighbor in _LANDMARK_NEIGHBORS.get(name, ())
            if neighbor in speeds
        ]
        neighbor_reference = float(np.median(neighbor_speeds)) if neighbor_speeds else 0.0
        if speed > max(max_speed_m_s, neighbor_reference * isolated_ratio):
            outliers.add(name)
    return speeds, outliers


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
