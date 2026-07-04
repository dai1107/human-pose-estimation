from __future__ import annotations

from collections import defaultdict
from math import hypot, isfinite
from pathlib import Path
from typing import Any

from src.reference.session_loader import SessionData, parse_number
from src.biomechanics.landmarks import landmark_at, midpoint
from src.biomechanics.types import KinematicFrame, LandmarkPoint, PoseFrame

from .schema import SquatCalibration, SquatFrameMeasurement, SquatRep, SquatRepMetrics, finite_or_none
from .symmetry import hip_symmetry_proxy, knee_symmetry_proxy


def _finite_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and isfinite(float(value))]


def _min(values: list[float | None]) -> float | None:
    finite = _finite_values(values)
    return min(finite) if finite else None


def _range(values: list[float | None]) -> float | None:
    finite = _finite_values(values)
    return max(finite) - min(finite) if finite else None


def _pose_valid_ratio(frames: list[SquatFrameMeasurement]) -> float:
    return sum(1 for frame in frames if frame.pose_detected) / len(frames) if frames else 0.0


def _quality_level(frames: list[SquatFrameMeasurement]) -> str:
    valid_ratio = _pose_valid_ratio(frames)
    visibility_values = _finite_values([frame.visibility_mean for frame in frames])
    visibility = sum(visibility_values) / len(visibility_values) if visibility_values else 0.0
    if valid_ratio >= 0.9 and visibility >= 0.75:
        return "GOOD"
    if valid_ratio >= 0.75 and visibility >= 0.55:
        return "WARNING"
    return "LOW"


def compute_rep_metrics(rep: SquatRep, calibration: SquatCalibration) -> SquatRepMetrics:
    frames = rep.frames
    scale = calibration.baseline_body_scale or 1.0
    baseline_y = calibration.baseline_pelvis_y
    pelvis_displacements: list[float | None] = []
    if baseline_y is not None and scale > 1e-9:
        pelvis_displacements = [
            (frame.pelvis_y - baseline_y) / scale if frame.pelvis_y is not None else None
            for frame in frames
        ]
    finite_displacements = _finite_values(pelvis_displacements)
    pelvis_vertical = max(finite_displacements) if finite_displacements else None

    pelvis_x_values = _finite_values([frame.pelvis_x for frame in frames])
    pelvis_lateral = (max(pelvis_x_values) - min(pelvis_x_values)) / scale if pelvis_x_values and scale > 1e-9 else None
    trunk_offsets: list[float] = []
    for frame in frames:
        if frame.shoulder_x is not None and frame.pelvis_x is not None and scale > 1e-9:
            trunk_offsets.append((frame.shoulder_x - frame.pelvis_x) / scale)
    trunk_lateral = max(trunk_offsets) - min(trunk_offsets) if trunk_offsets else None

    knee_mean, knee_peak = knee_symmetry_proxy(frames)
    hip_mean, hip_peak = hip_symmetry_proxy(frames)
    total = rep.end_timestamp_ms - rep.start_timestamp_ms
    descent = rep.bottom_timestamp_ms - rep.start_timestamp_ms
    ascent = rep.end_timestamp_ms - rep.bottom_timestamp_ms

    return SquatRepMetrics(
        rep_index=rep.rep_index,
        start_timestamp_ms=rep.start_timestamp_ms,
        bottom_timestamp_ms=rep.bottom_timestamp_ms,
        end_timestamp_ms=rep.end_timestamp_ms,
        total_duration_ms=total,
        descent_duration_ms=descent,
        bottom_duration_ms=0,
        ascent_duration_ms=ascent,
        left_knee_min_angle=_min([frame.left_knee_angle for frame in frames]),
        right_knee_min_angle=_min([frame.right_knee_angle for frame in frames]),
        left_hip_min_angle=_min([frame.left_hip_angle for frame in frames]),
        right_hip_min_angle=_min([frame.right_hip_angle for frame in frames]),
        left_knee_angle_range=_range([frame.left_knee_angle for frame in frames]),
        right_knee_angle_range=_range([frame.right_knee_angle for frame in frames]),
        left_hip_angle_range=_range([frame.left_hip_angle for frame in frames]),
        right_hip_angle_range=_range([frame.right_hip_angle for frame in frames]),
        trunk_tilt_range=_range([frame.trunk_tilt_proxy for frame in frames]),
        pelvis_vertical_displacement_normalized=pelvis_vertical,
        left_right_knee_difference_mean=knee_mean,
        left_right_knee_difference_peak=knee_peak,
        left_right_hip_difference_mean=hip_mean,
        left_right_hip_difference_peak=hip_peak,
        pelvis_lateral_drift_proxy=pelvis_lateral,
        trunk_lateral_drift_proxy=trunk_lateral,
        pose_valid_ratio=_pose_valid_ratio(frames),
        data_quality_level=_quality_level(frames),
    )


def compute_all_rep_metrics(reps: list[SquatRep], calibration: SquatCalibration) -> list[SquatRepMetrics]:
    return [compute_rep_metrics(rep, calibration) for rep in reps]


def _landmark_point(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    x = finite_or_none(row.get("smoothed_x"))
    y = finite_or_none(row.get("smoothed_y"))
    z = finite_or_none(row.get("smoothed_z"))
    if x is None:
        x = finite_or_none(row.get("image_x"))
    if y is None:
        y = finite_or_none(row.get("image_y"))
    if z is None:
        z = finite_or_none(row.get("image_z"))
    return x, y, z


def _distance(a: tuple[float | None, float | None, float | None] | None, b: tuple[float | None, float | None, float | None] | None) -> float | None:
    if a is None or b is None:
        return None
    if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return None
    az = a[2] or 0.0
    bz = b[2] or 0.0
    return hypot(hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])), az - bz)


def _midpoint(a: tuple[float | None, float | None, float | None] | None, b: tuple[float | None, float | None, float | None] | None) -> tuple[float | None, float | None, float | None] | None:
    if a is None or b is None:
        return None
    if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return None
    return ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0, ((a[2] or 0.0) + (b[2] or 0.0)) / 2.0)


def build_squat_frames_from_session(session: SessionData) -> list[SquatFrameMeasurement]:
    landmarks_by_frame: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in session.landmarks:
        frame_index = int(parse_number(row.get("frame_index")))
        landmarks_by_frame[frame_index][str(row.get("landmark_name", ""))] = row

    frames: list[SquatFrameMeasurement] = []
    for row in session.kinematics:
        frame_index = int(parse_number(row.get("frame_index")))
        landmark_rows = landmarks_by_frame.get(frame_index, {})
        points = {name: _landmark_point(value) for name, value in landmark_rows.items()}
        pelvis = _midpoint(points.get("left_hip"), points.get("right_hip"))
        shoulder = _midpoint(points.get("left_shoulder"), points.get("right_shoulder"))
        shoulder_width = _distance(points.get("left_shoulder"), points.get("right_shoulder"))
        hip_width = _distance(points.get("left_hip"), points.get("right_hip"))
        torso = _distance(pelvis, shoulder)
        scale = next((value for value in (shoulder_width, hip_width, torso) if value is not None and value > 1e-9), None)

        pose_raw = row.get("pose_detected")
        pose_detected = str(pose_raw).strip().lower() in {"1", "true", "yes"} if pose_raw is not None else True
        visibility_mean = finite_or_none(row.get("visibility_mean"))
        if visibility_mean is None:
            vis_values = _finite_values([finite_or_none(value.get("visibility")) for value in landmark_rows.values()])
            visibility_mean = sum(vis_values) / len(vis_values) if vis_values else None
        frames.append(
            SquatFrameMeasurement(
                frame_index=frame_index,
                timestamp_ms=int(parse_number(row.get("timestamp_ms"))),
                pose_detected=pose_detected,
                pelvis_x=pelvis[0] if pelvis else None,
                pelvis_y=pelvis[1] if pelvis else None,
                shoulder_x=shoulder[0] if shoulder else None,
                shoulder_y=shoulder[1] if shoulder else None,
                body_scale=scale,
                left_knee_angle=finite_or_none(row.get("left_knee_angle")),
                right_knee_angle=finite_or_none(row.get("right_knee_angle")),
                left_hip_angle=finite_or_none(row.get("left_hip_angle")),
                right_hip_angle=finite_or_none(row.get("right_hip_angle")),
                trunk_tilt_proxy=finite_or_none(row.get("trunk_tilt_proxy")),
                left_ankle_x=points.get("left_ankle", (None, None, None))[0],
                right_ankle_x=points.get("right_ankle", (None, None, None))[0],
                visibility_mean=visibility_mean,
                missing_ratio=finite_or_none(row.get("missing_ratio")),
                source=dict(row),
            )
        )
    return frames


def rep_kinematics_rows(rep: SquatRep) -> list[dict[str, Any]]:
    return [dict(frame.source) for frame in rep.frames if frame.source]


def _point_from_landmark(point: LandmarkPoint | None) -> tuple[float | None, float | None, float | None] | None:
    if point is None or not point.is_finite():
        return None
    return (point.x, point.y, point.z)


def build_squat_frame_from_pose(pose_frame: PoseFrame, kinematic_frame: KinematicFrame | None) -> SquatFrameMeasurement:
    landmarks = pose_frame.smoothed_landmarks or pose_frame.image_landmarks
    pelvis_point = midpoint(landmarks, "left_hip", "right_hip") if landmarks else None
    shoulder_point = midpoint(landmarks, "left_shoulder", "right_shoulder") if landmarks else None
    pelvis = _point_from_landmark(pelvis_point)
    shoulder = _point_from_landmark(shoulder_point)
    left_shoulder = _point_from_landmark(landmark_at(landmarks, "left_shoulder")) if landmarks else None
    right_shoulder = _point_from_landmark(landmark_at(landmarks, "right_shoulder")) if landmarks else None
    left_hip = _point_from_landmark(landmark_at(landmarks, "left_hip")) if landmarks else None
    right_hip = _point_from_landmark(landmark_at(landmarks, "right_hip")) if landmarks else None
    scale = next(
        (value for value in (_distance(left_shoulder, right_shoulder), _distance(left_hip, right_hip), _distance(pelvis, shoulder)) if value is not None and value > 1e-9),
        None,
    )
    quality = kinematic_frame.quality if kinematic_frame is not None else {}
    return SquatFrameMeasurement(
        frame_index=pose_frame.frame_index,
        timestamp_ms=pose_frame.timestamp_ms,
        pose_detected=pose_frame.pose_detected,
        pelvis_x=pelvis[0] if pelvis else None,
        pelvis_y=pelvis[1] if pelvis else None,
        shoulder_x=shoulder[0] if shoulder else None,
        shoulder_y=shoulder[1] if shoulder else None,
        body_scale=scale,
        left_knee_angle=finite_or_none(getattr(kinematic_frame, "left_knee_angle", None)),
        right_knee_angle=finite_or_none(getattr(kinematic_frame, "right_knee_angle", None)),
        left_hip_angle=finite_or_none(getattr(kinematic_frame, "left_hip_angle", None)),
        right_hip_angle=finite_or_none(getattr(kinematic_frame, "right_hip_angle", None)),
        trunk_tilt_proxy=finite_or_none(getattr(kinematic_frame, "trunk_tilt_proxy", None)),
        left_ankle_x=landmark_at(landmarks, "left_ankle").x if landmarks and landmark_at(landmarks, "left_ankle") else None,
        right_ankle_x=landmark_at(landmarks, "right_ankle").x if landmarks and landmark_at(landmarks, "right_ankle") else None,
        visibility_mean=finite_or_none(quality.get("visibility_mean")),
        missing_ratio=finite_or_none(quality.get("missing_ratio")),
        source={
            "frame_index": pose_frame.frame_index,
            "timestamp_ms": pose_frame.timestamp_ms,
            "left_knee_angle": getattr(kinematic_frame, "left_knee_angle", ""),
            "right_knee_angle": getattr(kinematic_frame, "right_knee_angle", ""),
            "left_hip_angle": getattr(kinematic_frame, "left_hip_angle", ""),
            "right_hip_angle": getattr(kinematic_frame, "right_hip_angle", ""),
            "trunk_tilt_proxy": getattr(kinematic_frame, "trunk_tilt_proxy", ""),
            "pose_detected": int(pose_frame.pose_detected),
        },
    )
