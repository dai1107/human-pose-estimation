from __future__ import annotations

import csv
import json
from pathlib import Path


KINEMATIC_COLUMNS = [
    "frame_index",
    "timestamp_ms",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "trunk_tilt_proxy",
    "left_wrist_speed",
    "right_wrist_speed",
    "left_ankle_speed",
    "right_ankle_speed",
    "pelvis_speed",
    "shoulder_center_speed",
    "left_elbow_angular_velocity",
    "right_elbow_angular_velocity",
    "left_knee_angular_velocity",
    "right_knee_angular_velocity",
    "left_hip_angular_velocity",
    "right_hip_angular_velocity",
    "motion_energy_proxy",
    "pose_detected",
    "visibility_mean",
    "missing_ratio",
]


def make_session(tmp_path: Path, session_id: str = "session_a", frame_count: int = 24, time_step_ms: int = 50, offset: float = 0.0, valid_ratio: float = 1.0) -> Path:
    session_dir = tmp_path / "sessions" / session_id
    session_dir.mkdir(parents=True)
    metadata = {
        "session_id": session_id,
        "started_at": "2026-07-04T12:00:00+08:00",
        "ended_at": "2026-07-04T12:00:02+08:00",
        "camera_index": 0,
        "actual_resolution": {"width": 1280, "height": 720},
        "average_fps": 20.0,
        "mirror": True,
        "smoothing": 0.65,
        "model_name": "pose_landmarker_full.task",
        "landmark_frame_count": frame_count,
        "pose_detected_frame_count": int(frame_count * valid_ratio),
        "pose_lost_frame_count": frame_count - int(frame_count * valid_ratio),
    }
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    valid_frames = int(frame_count * valid_ratio)
    with (session_dir / "kinematics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=KINEMATIC_COLUMNS)
        writer.writeheader()
        for index in range(frame_count):
            t = 1000 + index * time_step_ms
            pose_detected = 1 if index < valid_frames else 0
            row = {
                "frame_index": index,
                "timestamp_ms": t,
                "left_elbow_angle": 80 + index + offset,
                "right_elbow_angle": 82 + index + offset,
                "left_knee_angle": 100 + 0.5 * index + offset,
                "right_knee_angle": 101 + 0.5 * index + offset,
                "left_hip_angle": 120 - 0.25 * index + offset,
                "right_hip_angle": 121 - 0.25 * index + offset,
                "left_shoulder_angle": 60 + 0.2 * index + offset,
                "right_shoulder_angle": 62 + 0.2 * index + offset,
                "trunk_tilt_proxy": 5 + 0.1 * index,
                "left_wrist_speed": abs(index - frame_count / 2) * 0.1 + offset,
                "right_wrist_speed": abs(index - frame_count / 3) * 0.12 + offset,
                "left_ankle_speed": index * 0.03 + offset,
                "right_ankle_speed": index * 0.035 + offset,
                "pelvis_speed": index * 0.04 + offset,
                "shoulder_center_speed": index * 0.045 + offset,
                "left_elbow_angular_velocity": 2 + offset,
                "right_elbow_angular_velocity": 3 + offset,
                "left_knee_angular_velocity": 1 + offset,
                "right_knee_angular_velocity": 1.5 + offset,
                "left_hip_angular_velocity": 0.5 + offset,
                "right_hip_angular_velocity": 0.7 + offset,
                "motion_energy_proxy": 1 + index * 0.1 + offset,
                "pose_detected": pose_detected,
                "visibility_mean": 0.9 if pose_detected else 0.1,
                "missing_ratio": 0.05 if pose_detected else 0.9,
            }
            writer.writerow(row)

    landmark_columns = [
        "frame_index",
        "timestamp_ms",
        "landmark_name",
        "image_x",
        "image_y",
        "image_z",
        "world_x",
        "world_y",
        "world_z",
        "smoothed_x",
        "smoothed_y",
        "smoothed_z",
        "visibility",
        "presence",
    ]
    with (session_dir / "landmarks.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=landmark_columns)
        writer.writeheader()
        for index in range(frame_count):
            t = 1000 + index * time_step_ms
            visible = 0.9 if index < valid_frames else 0.1
            for landmark_name in ("left_hip", "right_hip", "left_wrist", "right_wrist"):
                writer.writerow(
                    {
                        "frame_index": index,
                        "timestamp_ms": t,
                        "landmark_name": landmark_name,
                        "image_x": 0.5,
                        "image_y": 0.5,
                        "image_z": 0.0,
                        "world_x": 0.0,
                        "world_y": 0.0,
                        "world_z": 0.0,
                        "smoothed_x": 0.0,
                        "smoothed_y": 0.0,
                        "smoothed_z": 0.0,
                        "visibility": visible,
                        "presence": visible,
                    }
                )
    return session_dir

