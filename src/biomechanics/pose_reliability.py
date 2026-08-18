"""Unified reliability evidence and validation metric selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any


REASON_OCCLUDED = "OCCLUDED"
REASON_CAMERA_MOVING = "CAMERA_MOVING"
REASON_BONE_LENGTH_OUTLIER = "BONE_LENGTH_OUTLIER"
REASON_LANDMARK_JUMP = "LANDMARK_JUMP"
REASON_GROUND_UNCERTAIN = "GROUND_UNCERTAIN"
REASON_CONTACT_UNCERTAIN = "CONTACT_UNCERTAIN"
REASON_WORLD_MISSING = "WORLD_LANDMARKS_MISSING"
REASON_VIEW_LIMITED = "VIEW_LIMITED"
REASON_POSE_STALE = "POSE_STALE"
REASON_BODY_FRAME_UNCERTAIN = "BODY_FRAME_UNCERTAIN"


_REASON_MAP: Mapping[str, str] = {
    "three_d_disabled": REASON_WORLD_MISSING,
    "low_visibility": REASON_OCCLUDED,
    "low_presence": REASON_OCCLUDED,
    "image_joint_missing": REASON_OCCLUDED,
    "world_joint_missing": REASON_WORLD_MISSING,
    "world_landmarks_missing": REASON_WORLD_MISSING,
    "invalid_world_geometry": REASON_WORLD_MISSING,
    "invalid_bone_length": REASON_BONE_LENGTH_OUTLIER,
    "bone_length_jump": REASON_BONE_LENGTH_OUTLIER,
    "left_right_bone_mismatch": REASON_BONE_LENGTH_OUTLIER,
    "isolated_landmark_velocity": REASON_LANDMARK_JUMP,
    "z_jump": REASON_LANDMARK_JUMP,
    "angle_jump": REASON_LANDMARK_JUMP,
    "angular_velocity_exceeded": REASON_LANDMARK_JUMP,
    "left_right_identity_swap": REASON_LANDMARK_JUMP,
    "world_gap_exceeded": REASON_POSE_STALE,
    "contact_foot_drift": REASON_CONTACT_UNCERTAIN,
    "pose_too_old": REASON_POSE_STALE,
    "camera_view_limited": REASON_VIEW_LIMITED,
    "two_d_three_d_conflict": REASON_VIEW_LIMITED,
}

_REASON_CODES = frozenset(_REASON_MAP.values())


@dataclass(frozen=True, slots=True)
class PoseReliability:
    global_confidence: float
    joint_confidence: Mapping[str, float]
    reasons: tuple[str, ...]
    joint_reasons: Mapping[str, tuple[str, ...]]
    evidence: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "global_confidence": self.global_confidence,
            "joint_confidence": dict(self.joint_confidence),
            "reasons": list(self.reasons),
            "joint_reasons": {
                name: list(reasons) for name, reasons in self.joint_reasons.items()
            },
            "evidence": dict(self.evidence),
            "camera_motion_affects_local_joint_angles": False,
            "formal_threshold_replacement_allowed": False,
        }


def build_pose_reliability(
    measurements: Mapping[str, object],
    *,
    body_coordinate_system: Mapping[str, Any],
    ground_estimation: Mapping[str, Any],
    foot_contact_evidence: Mapping[str, Any],
    camera_motion: Mapping[str, Any],
    reliability: Mapping[str, Any],
) -> PoseReliability:
    joint_confidence: dict[str, float] = {}
    joint_reasons: dict[str, tuple[str, ...]] = {}
    global_reasons: set[str] = set()
    local_values: list[float] = []
    for metric_name, metric in measurements.items():
        joint_name = str(metric_name).removesuffix("_angle")
        raw_reasons = getattr(metric, "quality_reasons", ())
        reasons = {
            normalized
            for reason in raw_reasons
            if (normalized := _REASON_MAP.get(str(reason))) in _REASON_CODES
        }
        confidence = _clamp(getattr(metric, "confidence", 0.0))
        if REASON_OCCLUDED in reasons:
            confidence *= 0.35
        if REASON_WORLD_MISSING in reasons:
            confidence *= 0.70
        if REASON_BONE_LENGTH_OUTLIER in reasons:
            confidence *= 0.45
        if REASON_LANDMARK_JUMP in reasons:
            confidence *= 0.35
        if REASON_CONTACT_UNCERTAIN in reasons:
            confidence *= 0.60
        if REASON_POSE_STALE in reasons:
            confidence = 0.0
        joint_confidence[joint_name] = _clamp(confidence)
        joint_reasons[joint_name] = tuple(sorted(reasons))
        local_values.append(joint_confidence[joint_name])
        global_reasons.update(reasons)

    body_confidence = _clamp(body_coordinate_system.get("confidence", 0.0))
    if not bool(body_coordinate_system.get("reliable")):
        global_reasons.add(REASON_BODY_FRAME_UNCERTAIN)
    ground_confidence = _clamp(
        ground_estimation.get(
            "camera_adjusted_ground_confidence",
            ground_estimation.get("ground_confidence", 0.0),
        )
    )
    if ground_estimation.get("status") != "READY" or ground_confidence < 0.50:
        global_reasons.add(REASON_GROUND_UNCERTAIN)
    contact_values = [
        _clamp(value.get("foot_contact_confidence", 0.0))
        for side, value in foot_contact_evidence.items()
        if side in {"left", "right"} and isinstance(value, Mapping)
    ]
    contact_confidence = max(contact_values, default=0.0)
    if not contact_values or max(contact_values, default=0.0) < 0.35:
        global_reasons.add(REASON_CONTACT_UNCERTAIN)
    camera_score = _clamp(camera_motion.get("camera_motion_score", 0.0))
    if str(camera_motion.get("state")) == "camera_unstable" or camera_score >= 0.42:
        global_reasons.add(REASON_CAMERA_MOVING)
    global_position = _clamp(reliability.get("global_position_reliability", 1.0))
    local_joint_confidence = sum(local_values) / len(local_values) if local_values else 0.0
    # Local joint evidence dominates. Ground/contact/camera only affect the
    # global-position component, consistent with the camera-motion contract.
    context_confidence = (
        0.35 * body_confidence
        + 0.25 * ground_confidence
        + 0.20 * contact_confidence
        + 0.20 * global_position
    )
    global_confidence = _clamp(0.70 * local_joint_confidence + 0.30 * context_confidence)
    return PoseReliability(
        global_confidence=global_confidence,
        joint_confidence=joint_confidence,
        reasons=tuple(sorted(global_reasons)),
        joint_reasons=joint_reasons,
        evidence={
            "local_joint_confidence": local_joint_confidence,
            "body_coordinate_confidence": body_confidence,
            "ground_confidence": ground_confidence,
            "contact_confidence": contact_confidence,
            "camera_motion_score": camera_score,
            "global_position_reliability": global_position,
        },
    )


def select_metric_candidates(
    measurements: Mapping[str, object],
    biomech_metrics: Mapping[str, Any],
    pose_reliability: PoseReliability,
    *,
    minimum_joint_confidence: float = 0.50,
) -> dict[str, dict[str, Any]]:
    """Select validation candidates without changing formal legacy angles."""

    biomech_by_joint = {
        "left_knee_angle": "left_knee_flexion_proxy",
        "right_knee_angle": "right_knee_flexion_proxy",
        "left_hip_angle": "left_hip_flexion_proxy",
        "right_hip_angle": "right_hip_flexion_proxy",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, measurement in measurements.items():
        joint = str(name).removesuffix("_angle")
        confidence = _clamp(pose_reliability.joint_confidence.get(joint, 0.0))
        angle_3d = _finite(getattr(measurement, "angle_3d", None))
        angle_2d = _finite(getattr(measurement, "angle_2d", None))
        three_d_reliable = bool(getattr(measurement, "three_d_reliable", False))
        observable = bool(getattr(measurement, "observable", False))
        if confidence >= minimum_joint_confidence and three_d_reliable and angle_3d is not None:
            selected, source, status = angle_3d, "3D", "OBSERVABLE"
        elif observable and angle_2d is not None:
            selected, source, status = angle_2d, "2D", "OBSERVABLE"
        else:
            selected, source, status = None, "UNSURE", "UNSURE"
        biomech_name = biomech_by_joint.get(str(name))
        biomech = biomech_metrics.get(biomech_name) if biomech_name else None
        biomech_candidate = (
            dict(biomech)
            if isinstance(biomech, Mapping)
            and bool(biomech.get("observable"))
            and confidence >= minimum_joint_confidence
            else None
        )
        result[str(name)] = {
            "selected_value": selected,
            "selected_source": source,
            "status": status,
            "joint_confidence": confidence,
            "reasons": list(pose_reliability.joint_reasons.get(joint, ())),
            "three_d_candidate": angle_3d if three_d_reliable else None,
            "two_d_candidate": angle_2d if observable else None,
            "biomech_candidate": biomech_candidate,
            "formal_value": _finite(getattr(measurement, "selected_angle", None)),
            "formal_source": getattr(measurement, "selected_source", "none"),
            "validation_only": True,
            "formal_threshold_replacement_allowed": False,
        }
    return result


def _clamp(value: object) -> float:
    number = _finite(value)
    return 0.0 if number is None else max(0.0, min(1.0, number))


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


__all__ = [
    "PoseReliability",
    "build_pose_reliability",
    "select_metric_candidates",
]
