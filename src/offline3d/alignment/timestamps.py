from __future__ import annotations

import bisect
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..base import Offline3DFrame, Offline3DResult
from .coordinates import coordinate_relationship
from .joints import joint_overlap, landmark_list_to_map


AlignmentMethod = Literal["EXACT", "LINEAR_INTERPOLATION", "UNMATCHED"]


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    exact_tolerance_ms: float = 0.5
    maximum_interpolation_span_ms: float = 120.0


@dataclass(slots=True)
class MotionFrame:
    source_timestamp_ms: float
    mediapipe: dict[str, Any]
    wham: dict[str, Any] | None
    alignment_method: AlignmentMethod
    alignment: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_timestamp_ms": round(float(self.source_timestamp_ms), 6),
            "mediapipe": dict(self.mediapipe),
            "wham": None if self.wham is None else dict(self.wham),
            "alignment_method": self.alignment_method,
            "alignment": dict(self.alignment),
        }


@dataclass(slots=True)
class MotionAlignmentResult:
    status: str
    frames: list[MotionFrame] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    coordinate_relationship: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def unavailable(cls, reason: str) -> "MotionAlignmentResult":
        return cls(status="UNAVAILABLE", warnings=[reason])

    def as_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": "motion_alignment_v1",
            "status": self.status,
            "alignment_key": "source_timestamp_ms",
            "frame_index_alignment_allowed": False,
            "mediapipe_interpolation_allowed": False,
            "wham_interpolation_allowed": True,
            "statistics": dict(self.statistics),
            "coordinate_relationship": dict(self.coordinate_relationship),
            "warnings": list(self.warnings),
        }
        if include_frames:
            payload["motion_frames"] = [frame.as_dict() for frame in self.frames]
        return payload


def align_mediapipe_wham(
    mediapipe_frames: Sequence[Mapping[str, Any]],
    wham_result: Offline3DResult,
    config: AlignmentConfig | None = None,
) -> MotionAlignmentResult:
    """Align reconstruction samples to MediaPipe observations by source time.

    MediaPipe observations are never interpolated. WHAM values may be linearly
    interpolated only between two bounded reconstruction samples, and every
    interpolated field is labelled in the returned MotionFrame.
    """

    resolved = config or AlignmentConfig()
    if wham_result.status != "COMPLETED" or not wham_result.frames:
        return MotionAlignmentResult.unavailable(
            f"WHAM result is not alignable: status={wham_result.status}"
        )
    ordered_wham = sorted(wham_result.frames, key=lambda frame: frame.timestamp_ms)
    wham_times = [float(frame.timestamp_ms) for frame in ordered_wham]
    if any(right <= left for left, right in zip(wham_times, wham_times[1:])):
        return MotionAlignmentResult.unavailable(
            "WHAM timestamps must be strictly increasing"
        )

    motion_frames: list[MotionFrame] = []
    exact_count = 0
    interpolated_count = 0
    unmatched_count = 0
    temporal_distances: list[float] = []
    input_timestamps: list[float] = []
    for source_position, media_frame in enumerate(mediapipe_frames):
        raw_timestamp = media_frame.get("timestamp_ms")
        try:
            timestamp_ms = float(raw_timestamp)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(timestamp_ms):
            continue
        input_timestamps.append(timestamp_ms)
        wham_payload, method, audit = _sample_wham(
            timestamp_ms,
            ordered_wham,
            wham_times,
            resolved,
        )
        if method == "EXACT":
            exact_count += 1
        elif method == "LINEAR_INTERPOLATION":
            interpolated_count += 1
        else:
            unmatched_count += 1
        if audit.get("maximum_source_distance_ms") is not None:
            temporal_distances.append(float(audit["maximum_source_distance_ms"]))
        mediapipe = _mediapipe_payload(media_frame, source_position)
        overlap = joint_overlap(
            landmark_list_to_map(mediapipe.get("canonical_3d")),
            (wham_payload or {}).get("joints_3d", {}),
        )
        audit["joint_compatibility"] = overlap
        motion_frames.append(
            MotionFrame(
                source_timestamp_ms=timestamp_ms,
                mediapipe=mediapipe,
                wham=wham_payload,
                alignment_method=method,
                alignment=audit,
            )
        )

    input_monotonic = all(
        right > left for left, right in zip(input_timestamps, input_timestamps[1:])
    )
    warnings: list[str] = []
    if not input_monotonic:
        warnings.append("MediaPipe source timestamps are not strictly increasing")
    total = len(motion_frames)
    return MotionAlignmentResult(
        status="COMPLETED",
        frames=motion_frames,
        statistics={
            "mediapipe_frame_count": total,
            "wham_frame_count": len(ordered_wham),
            "exact_count": exact_count,
            "interpolated_count": interpolated_count,
            "unmatched_count": unmatched_count,
            "matched_ratio": (
                (exact_count + interpolated_count) / total if total else 0.0
            ),
            "maximum_source_distance_ms": max(temporal_distances, default=None),
            "median_source_distance_ms": (
                statistics.median(temporal_distances) if temporal_distances else None
            ),
            "mediapipe_source_timestamps_strictly_increasing": input_monotonic,
        },
        coordinate_relationship=coordinate_relationship(
            "mediapipe_image/world/body_canonical",
            wham_result.coordinate_system,
        ),
        warnings=warnings,
    )


def _mediapipe_payload(frame: Mapping[str, Any], source_position: int) -> dict[str, Any]:
    three_d = frame.get("three_d_kinematics")
    three_d = dict(three_d) if isinstance(three_d, Mapping) else {}
    body = three_d.get("body_coordinate_system")
    body = dict(body) if isinstance(body, Mapping) else {}
    return {
        "frame_index": frame.get("frame_index", source_position),
        "analysis_timestamp_ms": frame.get("analysis_timestamp_ms"),
        "raw_landmarks_2d": list(frame.get("raw_keypoints") or []),
        "landmarks_2d": list(frame.get("keypoints") or []),
        "raw_landmarks_3d": list(frame.get("raw_world_keypoints") or []),
        "landmarks_3d": list(frame.get("world_keypoints") or []),
        "canonical_3d": list(body.get("canonical_landmarks") or []),
        "selected_rule_angles": _selected_rule_angles(
            frame.get("angle_observations")
        ),
        "angle_observations": list(frame.get("angle_observations") or []),
        "angle_source_policy": dict(frame.get("angle_sources") or {}),
        "canonical_3d_angles": dict(three_d.get("canonical_3d_angles") or {}),
        "observation_interpolated": False,
    }


def _selected_rule_angles(value: Any) -> dict[str, float]:
    if not isinstance(value, list):
        return {}
    angles: dict[str, float] = {}
    for observation in value:
        if not isinstance(observation, Mapping):
            continue
        raw = observation.get("rule_angle_deg")
        try:
            angle = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(angle):
            continue
        side = str(observation.get("side", "center"))
        joint = str(observation.get("joint_name", "unknown"))
        angles[f"{side}_{joint}"] = angle
    return angles


def _sample_wham(
    timestamp_ms: float,
    frames: Sequence[Offline3DFrame],
    times: Sequence[float],
    config: AlignmentConfig,
) -> tuple[dict[str, Any] | None, AlignmentMethod, dict[str, Any]]:
    insertion = bisect.bisect_left(times, timestamp_ms)
    candidates = []
    if insertion < len(times):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    if candidates:
        nearest = min(candidates, key=lambda index: abs(times[index] - timestamp_ms))
        distance = abs(times[nearest] - timestamp_ms)
        if distance <= config.exact_tolerance_ms:
            payload = frames[nearest].as_dict()
            payload["sample_interpolated"] = False
            return payload, "EXACT", {
                "source_timestamps_ms": [times[nearest]],
                "maximum_source_distance_ms": distance,
                "interpolation_fraction": 0.0,
            }
    if insertion <= 0 or insertion >= len(times):
        return None, "UNMATCHED", {
            "reason": "target timestamp is outside the WHAM timeline",
            "maximum_source_distance_ms": None,
        }
    left = frames[insertion - 1]
    right = frames[insertion]
    span = float(right.timestamp_ms - left.timestamp_ms)
    if span <= 0 or span > config.maximum_interpolation_span_ms:
        return None, "UNMATCHED", {
            "reason": "WHAM interpolation span exceeds configured maximum",
            "source_timestamps_ms": [left.timestamp_ms, right.timestamp_ms],
            "maximum_source_distance_ms": max(
                timestamp_ms - left.timestamp_ms, right.timestamp_ms - timestamp_ms
            ),
        }
    fraction = (timestamp_ms - left.timestamp_ms) / span
    payload = _interpolate_frame(left, right, timestamp_ms, fraction)
    return payload, "LINEAR_INTERPOLATION", {
        "source_timestamps_ms": [left.timestamp_ms, right.timestamp_ms],
        "maximum_source_distance_ms": max(
            timestamp_ms - left.timestamp_ms, right.timestamp_ms - timestamp_ms
        ),
        "interpolation_fraction": fraction,
        "interpolated_fields": [
            "joints_3d", "smpl_pose", "body_orientation", "body_translation",
            "camera_motion", "global_trajectory", "confidence",
        ],
    }


def _interpolate_frame(
    left: Offline3DFrame,
    right: Offline3DFrame,
    timestamp_ms: float,
    fraction: float,
) -> dict[str, Any]:
    shared_joints = set(left.joints_3d) & set(right.joints_3d)
    joints = {
        name: _interpolate_numeric(left.joints_3d[name], right.joints_3d[name], fraction)
        for name in sorted(shared_joints)
    }
    confidence = (
        None
        if left.confidence is None or right.confidence is None
        else float(left.confidence + (right.confidence - left.confidence) * fraction)
    )
    return {
        "timestamp_ms": timestamp_ms,
        "frame_index": None,
        "joints_3d": joints,
        "smpl_pose": _interpolate_numeric(left.smpl_pose, right.smpl_pose, fraction),
        "body_orientation": _interpolate_numeric(
            left.body_orientation, right.body_orientation, fraction
        ),
        "body_translation": _interpolate_numeric(
            left.body_translation, right.body_translation, fraction
        ),
        "camera_motion": _interpolate_numeric(
            left.camera_motion, right.camera_motion, fraction
        ),
        "global_trajectory": _interpolate_numeric(
            left.global_trajectory, right.global_trajectory, fraction
        ),
        "confidence": confidence,
        "sample_interpolated": True,
        "extra": {},
    }


def _interpolate_numeric(left: Any, right: Any, fraction: float) -> Any:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) + (float(right) - float(left)) * fraction
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
        and len(left) == len(right)
    ):
        return [
            _interpolate_numeric(first, second, fraction)
            for first, second in zip(left, right)
        ]
    return None
