from __future__ import annotations

from math import isfinite, nan

import numpy as np

from .angles import ANGLE_DEFINITIONS, compute_body_centers, compute_joint_angles
from .landmarks import coerce_landmarks, landmark_at, mean_visibility, missing_ratio, point_array
from .segments import compute_segment_vectors
from .stability import StabilityTracker, compute_motion_energy_proxy, compute_symmetry
from .types import KinematicFrame, LandmarkPoint, PoseFrame


def dt_seconds(current_timestamp_ms: int, previous_timestamp_ms: int, max_dt_ms: int = 1000) -> float | None:
    delta_ms = int(current_timestamp_ms) - int(previous_timestamp_ms)
    if delta_ms <= 0 or delta_ms > max_dt_ms:
        return None
    return delta_ms / 1000.0


def linear_velocity(
    current: LandmarkPoint | None,
    previous: LandmarkPoint | None,
    current_timestamp_ms: int,
    previous_timestamp_ms: int,
    max_dt_ms: int = 1000,
) -> tuple[float, float, float, float]:
    dt = dt_seconds(current_timestamp_ms, previous_timestamp_ms, max_dt_ms=max_dt_ms)
    current_array = point_array(current)
    previous_array = point_array(previous)
    if dt is None or current_array is None or previous_array is None:
        return (nan, nan, nan, nan)
    vector = (current_array - previous_array) / dt
    speed = float(np.linalg.norm(vector))
    return (float(vector[0]), float(vector[1]), float(vector[2]), speed)


def angular_velocity(
    current_angle: float,
    previous_angle: float,
    current_timestamp_ms: int,
    previous_timestamp_ms: int,
    max_dt_ms: int = 1000,
) -> float:
    dt = dt_seconds(current_timestamp_ms, previous_timestamp_ms, max_dt_ms=max_dt_ms)
    if dt is None or not isfinite(current_angle) or not isfinite(previous_angle):
        return nan
    return (float(current_angle) - float(previous_angle)) / dt


def _nan_kinematic_frame(frame: PoseFrame, quality: dict[str, float] | None = None) -> KinematicFrame:
    return KinematicFrame(
        frame_index=frame.frame_index,
        timestamp_ms=frame.timestamp_ms,
        pose_detected=frame.pose_detected,
        left_elbow_angle=nan,
        right_elbow_angle=nan,
        left_knee_angle=nan,
        right_knee_angle=nan,
        left_hip_angle=nan,
        right_hip_angle=nan,
        left_shoulder_angle=nan,
        right_shoulder_angle=nan,
        trunk_tilt_proxy=nan,
        left_wrist_speed=nan,
        right_wrist_speed=nan,
        left_ankle_speed=nan,
        right_ankle_speed=nan,
        pelvis_speed=nan,
        shoulder_center_speed=nan,
        left_elbow_angular_velocity=nan,
        right_elbow_angular_velocity=nan,
        left_knee_angular_velocity=nan,
        right_knee_angular_velocity=nan,
        left_hip_angular_velocity=nan,
        right_hip_angular_velocity=nan,
        motion_energy_proxy=nan,
        quality=quality or {"visibility_mean": nan, "missing_ratio": 1.0},
    )


class KinematicsProcessor:
    def __init__(self, max_dt_ms: int = 1000, stability_window_ms: int = 1000) -> None:
        self.max_dt_ms = max_dt_ms
        self._previous_landmarks: list[LandmarkPoint] | None = None
        self._previous_centers: dict[str, LandmarkPoint | None] | None = None
        self._previous_angles: dict[str, float] | None = None
        self._previous_timestamp_ms: int | None = None
        self._stability = StabilityTracker(window_ms=stability_window_ms)

    def reset(self) -> None:
        self._previous_landmarks = None
        self._previous_centers = None
        self._previous_angles = None
        self._previous_timestamp_ms = None
        self._stability = StabilityTracker(window_ms=self._stability.window_ms)

    def process(self, frame: PoseFrame) -> KinematicFrame:
        landmarks = coerce_landmarks(frame.smoothed_landmarks or frame.image_landmarks)
        quality = {
            "visibility_mean": mean_visibility(landmarks),
            "missing_ratio": missing_ratio(landmarks),
        }

        if not frame.pose_detected or not landmarks:
            return _nan_kinematic_frame(frame, quality=quality)

        angles = compute_joint_angles(landmarks)
        segments = compute_segment_vectors(landmarks)
        centers = compute_body_centers(landmarks)

        previous_ts = self._previous_timestamp_ms
        speeds = {
            "left_wrist_speed": nan,
            "right_wrist_speed": nan,
            "left_ankle_speed": nan,
            "right_ankle_speed": nan,
            "pelvis_speed": nan,
            "shoulder_center_speed": nan,
        }

        if self._previous_landmarks is not None and previous_ts is not None:
            point_pairs = {
                "left_wrist_speed": ("left_wrist", landmark_at(self._previous_landmarks, "left_wrist")),
                "right_wrist_speed": ("right_wrist", landmark_at(self._previous_landmarks, "right_wrist")),
                "left_ankle_speed": ("left_ankle", landmark_at(self._previous_landmarks, "left_ankle")),
                "right_ankle_speed": ("right_ankle", landmark_at(self._previous_landmarks, "right_ankle")),
            }
            for speed_name, (landmark_name, previous_point) in point_pairs.items():
                _, _, _, speeds[speed_name] = linear_velocity(
                    landmark_at(landmarks, landmark_name),
                    previous_point,
                    frame.timestamp_ms,
                    previous_ts,
                    max_dt_ms=self.max_dt_ms,
                )

            if self._previous_centers is not None:
                _, _, _, speeds["pelvis_speed"] = linear_velocity(
                    centers["pelvis_center"],
                    self._previous_centers.get("pelvis_center"),
                    frame.timestamp_ms,
                    previous_ts,
                    max_dt_ms=self.max_dt_ms,
                )
                _, _, _, speeds["shoulder_center_speed"] = linear_velocity(
                    centers["shoulder_center"],
                    self._previous_centers.get("shoulder_center"),
                    frame.timestamp_ms,
                    previous_ts,
                    max_dt_ms=self.max_dt_ms,
                )

        angle_velocities: dict[str, float] = {
            f"{name.removesuffix('_angle')}_angular_velocity": nan
            for name in ANGLE_DEFINITIONS
            if name.endswith("_angle")
        }
        if self._previous_angles is not None and previous_ts is not None:
            for angle_name, current_angle in angles.items():
                if not angle_name.endswith("_angle"):
                    continue
                velocity_name = f"{angle_name.removesuffix('_angle')}_angular_velocity"
                angle_velocities[velocity_name] = angular_velocity(
                    current_angle,
                    self._previous_angles.get(angle_name, nan),
                    frame.timestamp_ms,
                    previous_ts,
                    max_dt_ms=self.max_dt_ms,
                )

        motion_energy = compute_motion_energy_proxy(speeds)
        symmetry = compute_symmetry(angles, speeds)
        stability = self._stability.update(
            frame.timestamp_ms,
            centers["pelvis_center"],
            segments.get("trunk").unit if segments.get("trunk") and segments["trunk"].valid else None,
        )

        self._previous_landmarks = landmarks
        self._previous_centers = centers
        self._previous_angles = angles
        self._previous_timestamp_ms = frame.timestamp_ms

        return KinematicFrame(
            frame_index=frame.frame_index,
            timestamp_ms=frame.timestamp_ms,
            pose_detected=True,
            left_elbow_angle=angles.get("left_elbow_angle", nan),
            right_elbow_angle=angles.get("right_elbow_angle", nan),
            left_knee_angle=angles.get("left_knee_angle", nan),
            right_knee_angle=angles.get("right_knee_angle", nan),
            left_hip_angle=angles.get("left_hip_angle", nan),
            right_hip_angle=angles.get("right_hip_angle", nan),
            left_shoulder_angle=angles.get("left_shoulder_angle", nan),
            right_shoulder_angle=angles.get("right_shoulder_angle", nan),
            trunk_tilt_proxy=angles.get("trunk_tilt_proxy", nan),
            left_wrist_speed=speeds["left_wrist_speed"],
            right_wrist_speed=speeds["right_wrist_speed"],
            left_ankle_speed=speeds["left_ankle_speed"],
            right_ankle_speed=speeds["right_ankle_speed"],
            pelvis_speed=speeds["pelvis_speed"],
            shoulder_center_speed=speeds["shoulder_center_speed"],
            left_elbow_angular_velocity=angle_velocities.get("left_elbow_angular_velocity", nan),
            right_elbow_angular_velocity=angle_velocities.get("right_elbow_angular_velocity", nan),
            left_knee_angular_velocity=angle_velocities.get("left_knee_angular_velocity", nan),
            right_knee_angular_velocity=angle_velocities.get("right_knee_angular_velocity", nan),
            left_hip_angular_velocity=angle_velocities.get("left_hip_angular_velocity", nan),
            right_hip_angular_velocity=angle_velocities.get("right_hip_angular_velocity", nan),
            motion_energy_proxy=motion_energy,
            segment_vectors=segments,
            angular_velocities=angle_velocities,
            stability=stability,
            symmetry=symmetry,
            quality=quality,
        )

