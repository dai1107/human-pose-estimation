from __future__ import annotations

"""Quality-gated WHAM evidence for offline HYROX review.

The fusion is deliberately post-hoc.  It enriches advanced reports and the
upload playback skeleton, while the established MediaPipe HYROX rule result
remains the formal result.
"""

import math
import statistics
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.biomechanics.joint_metrics import ANGLE_DEFINITIONS, calculate_angle_3d

from ..alignment import MotionAlignmentResult
from ..base import Offline3DResult


ACTION_ANGLE_KEYS: dict[str, tuple[str, ...]] = {
    "lunge": ("left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle"),
    "wall_ball": ("left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle", "left_elbow_angle", "right_elbow_angle"),
    "farmers_carry": ("left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle"),
    "rowing": ("left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle", "left_elbow_angle", "right_elbow_angle"),
    "skierg": ("left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle", "left_shoulder_angle", "right_shoulder_angle"),
    "burpee_broad_jump": ("left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle"),
    "sled_push": ("left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle", "left_shoulder_angle", "right_shoulder_angle"),
    "sled_pull": ("left_knee_angle", "right_knee_angle", "left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle"),
}

WHAM_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot_index"),
)


@dataclass(slots=True)
class WhamHyroxFusionResult:
    status: str
    action: str
    frame_count: int = 0
    matched_frame_count: int = 0
    reliable_frame_count: int = 0
    confirmed_frame_count: int = 0
    conflict_frame_count: int = 0
    wham_only_frame_count: int = 0
    angle_differences_deg: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        coverage = self.matched_frame_count / self.frame_count if self.frame_count else 0.0
        reliable_coverage = self.reliable_frame_count / self.frame_count if self.frame_count else 0.0
        return {
            "schema_version": "wham_hyrox_fusion_v1",
            "status": self.status,
            "action": self.action,
            "role": "quality_gated_hyrox_review_assist",
            "formal_rule_replacement_allowed": False,
            "formal_result_source": "MediaPipe HYROX rule engine",
            "frame_count": self.frame_count,
            "matched_frame_count": self.matched_frame_count,
            "reliable_frame_count": self.reliable_frame_count,
            "confirmed_frame_count": self.confirmed_frame_count,
            "conflict_frame_count": self.conflict_frame_count,
            "wham_only_frame_count": self.wham_only_frame_count,
            "coverage_ratio": round(coverage, 6),
            "reliable_coverage_ratio": round(reliable_coverage, 6),
            "median_angle_difference_deg": (
                round(statistics.median(self.angle_differences_deg), 3)
                if self.angle_differences_deg
                else None
            ),
            "p95_angle_difference_deg": _percentile(self.angle_differences_deg, 0.95),
            "skeleton_overlay_enabled": self.reliable_frame_count > 0,
            "assistance": [
                "WHAM 3D joint angles and SMPL skeleton overlay",
                "MediaPipe/WHAM agreement and conflict audit",
                "world-root pelvis motion and speed when available",
                "occlusion review when MediaPipe 3D angles are unavailable",
            ],
            "warnings": list(self.warnings),
            "is_ground_truth": False,
        }


def apply_wham_hyrox_assist(
    frames: Sequence[MutableMapping[str, Any]],
    *,
    alignment: MotionAlignmentResult,
    wham_result: Offline3DResult,
    action: str,
    minimum_confidence: float = 0.30,
    confirmation_tolerance_deg: float = 20.0,
    conflict_tolerance_deg: float = 35.0,
) -> WhamHyroxFusionResult:
    result = WhamHyroxFusionResult(
        status="UNAVAILABLE",
        action=action,
        frame_count=len(frames),
    )
    if wham_result.status != "COMPLETED" or alignment.status != "COMPLETED":
        result.warnings.append(
            f"WHAM/HYROX fusion unavailable: wham={wham_result.status}, alignment={alignment.status}"
        )
        return result

    by_index = {
        int(frame["frame_index"]): frame
        for frame in frames
        if isinstance(frame.get("frame_index"), (int, float))
    }
    metadata = wham_result.metadata
    width = _positive_number(metadata.get("source_width"))
    height = _positive_number(metadata.get("source_height"))
    focal = _positive_number(metadata.get("focal_length_px"))
    previous_timestamp: float | None = None
    previous_trajectory: tuple[float, float, float] | None = None

    for motion in alignment.frames:
        frame_index = motion.mediapipe.get("frame_index")
        target = by_index.get(int(frame_index)) if isinstance(frame_index, (int, float)) else None
        if target is None:
            continue
        wham = motion.wham
        if not isinstance(wham, Mapping):
            target["wham_assist"] = _unavailable_frame("timestamp_unmatched")
            continue
        result.matched_frame_count += 1
        confidence = _confidence(wham.get("confidence"))
        joints = _joints(wham.get("joints_3d"))
        relevant_keys = ACTION_ANGLE_KEYS.get(action, tuple(ANGLE_DEFINITIONS))
        wham_angles = _angles(joints, relevant_keys)
        mediapipe_angles = _mediapipe_angles(motion.mediapipe, relevant_keys)
        differences = {
            key: abs(wham_angles[key] - mediapipe_angles[key])
            for key in wham_angles.keys() & mediapipe_angles.keys()
        }
        result.angle_differences_deg.extend(differences.values())
        required_angle_count = max(1, min(2, len(relevant_keys)))
        reliable = confidence >= minimum_confidence and len(wham_angles) >= required_angle_count
        reasons: list[str] = []
        if confidence < minimum_confidence:
            reasons.append("wham_confidence_below_threshold")
        if len(wham_angles) < required_angle_count:
            reasons.append("wham_action_joints_incomplete")
        if motion.alignment_method == "UNMATCHED":
            reliable = False
            reasons.append("timestamp_unmatched")

        if not reliable:
            review_status = "LOW_CONFIDENCE"
        elif not mediapipe_angles:
            review_status = "WHAM_ONLY"
            result.wham_only_frame_count += 1
        elif any(value > conflict_tolerance_deg for value in differences.values()):
            review_status = "CONFLICT"
            result.conflict_frame_count += 1
        elif differences and all(
            value <= confirmation_tolerance_deg for value in differences.values()
        ):
            review_status = "CONFIRMED"
            result.confirmed_frame_count += 1
        else:
            review_status = "SUPPLEMENTAL"

        trajectory = _point3(wham.get("global_trajectory"))
        pelvis_speed = None
        if trajectory is not None and previous_trajectory is not None and previous_timestamp is not None:
            elapsed = (motion.source_timestamp_ms - previous_timestamp) / 1000.0
            if 0.0 < elapsed <= 0.5:
                pelvis_speed = math.dist(trajectory, previous_trajectory) / elapsed
        if trajectory is not None:
            previous_trajectory = trajectory
            previous_timestamp = motion.source_timestamp_ms

        projected = _projected_joints(joints, width=width, height=height, focal=focal)
        if reliable:
            result.reliable_frame_count += 1
        target["wham_assist"] = {
            "status": review_status,
            "reliable": reliable,
            "confidence": round(confidence, 4),
            "alignment_method": motion.alignment_method,
            "quality_reasons": reasons,
            "angles_3d": {key: round(value, 3) for key, value in wham_angles.items()},
            "mediapipe_angles_3d": {
                key: round(value, 3) for key, value in mediapipe_angles.items()
            },
            "angle_differences_deg": {
                key: round(value, 3) for key, value in differences.items()
            },
            "pelvis_speed_mps": None if pelvis_speed is None else round(pelvis_speed, 4),
            "global_trajectory": list(trajectory) if trajectory is not None else [],
            "projected_keypoints": projected if reliable else [],
            "connections": [list(pair) for pair in WHAM_CONNECTIONS],
            "display_role": "WHAM 3D reference skeleton",
            "formal_rule_replacement_allowed": False,
        }

    result.status = "COMPLETED" if result.matched_frame_count else "UNAVAILABLE"
    if result.reliable_frame_count == 0:
        result.warnings.append("WHAM returned no frames that passed HYROX assist quality gates")
    return result


def _unavailable_frame(reason: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "reliable": False,
        "confidence": 0.0,
        "quality_reasons": [reason],
        "projected_keypoints": [],
        "connections": [],
        "formal_rule_replacement_allowed": False,
    }


def _joints(value: Any) -> dict[str, tuple[float, float, float]]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, tuple[float, float, float]] = {}
    for name, point in value.items():
        resolved = _point3(point)
        if resolved is not None:
            output[str(name)] = resolved
    return output


def _angles(
    joints: Mapping[str, tuple[float, float, float]], keys: Sequence[str]
) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in keys:
        definition = ANGLE_DEFINITIONS.get(key)
        if definition is None:
            continue
        value = calculate_angle_3d(*(joints.get(name) for name in definition))
        if value is not None and math.isfinite(float(value)):
            output[key] = float(value)
    return output


def _mediapipe_angles(
    payload: Mapping[str, Any], keys: Sequence[str]
) -> dict[str, float]:
    candidates = payload.get("canonical_3d_angles")
    if not isinstance(candidates, Mapping):
        candidates = {}
    selected = payload.get("selected_rule_angles")
    if not isinstance(selected, Mapping):
        selected = {}
    output: dict[str, float] = {}
    for key in keys:
        raw = candidates.get(key, selected.get(key))
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            output[key] = value
    return output


def _projected_joints(
    joints: Mapping[str, tuple[float, float, float]],
    *,
    width: float | None,
    height: float | None,
    focal: float | None,
) -> list[dict[str, Any]]:
    if width is None or height is None:
        return []
    resolved_focal = focal or math.sqrt(width * width + height * height)
    output: list[dict[str, Any]] = []
    for name, (x, y, z) in joints.items():
        if name not in {item for pair in WHAM_CONNECTIONS for item in pair} or z <= 1e-6:
            continue
        normalized_x = (resolved_focal * x / z + width / 2.0) / width
        normalized_y = (resolved_focal * y / z + height / 2.0) / height
        if not all(math.isfinite(value) for value in (normalized_x, normalized_y)):
            continue
        output.append(
            {
                "name": name,
                "x": round(normalized_x, 6),
                "y": round(normalized_y, 6),
                "visibility": 1.0,
            }
        )
    return output


def _point3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    try:
        point = tuple(float(value[index]) for index in range(3))
    except (TypeError, ValueError, OverflowError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def _confidence(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(1.0, resolved)) if math.isfinite(resolved) else 0.0


def _positive_number(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if math.isfinite(resolved) and resolved > 0 else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


__all__ = ["WhamHyroxFusionResult", "apply_wham_hyrox_assist"]
