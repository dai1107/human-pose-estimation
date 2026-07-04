from __future__ import annotations

import csv
import json
from pathlib import Path

from src.fitness.squat.schema import SquatFrameMeasurement


def make_squat_measurements(missing_at: set[int] | None = None) -> list[SquatFrameMeasurement]:
    missing_at = missing_at or set()
    values = [
        (0, 0.00, 170, 170),
        (50, 0.00, 170, 170),
        (100, 0.00, 170, 170),
        (150, 0.00, 170, 170),
        (200, 0.00, 170, 170),
        (250, 0.10, 158, 158),
        (350, 0.22, 135, 135),
        (450, 0.34, 100, 100),
        (550, 0.35, 92, 92),
        (650, 0.30, 105, 105),
        (750, 0.16, 135, 135),
        (850, 0.04, 168, 168),
    ]
    frames: list[SquatFrameMeasurement] = []
    for index, (timestamp, pelvis_y, knee, hip) in enumerate(values):
        pose_detected = index not in missing_at
        frames.append(
            SquatFrameMeasurement(
                frame_index=index,
                timestamp_ms=timestamp,
                pose_detected=pose_detected,
                pelvis_x=0.0,
                pelvis_y=pelvis_y,
                shoulder_x=0.0,
                shoulder_y=pelvis_y - 1.0,
                body_scale=1.0,
                left_knee_angle=float(knee),
                right_knee_angle=float(knee),
                left_hip_angle=float(hip),
                right_hip_angle=float(hip),
                trunk_tilt_proxy=10.0 + pelvis_y,
                visibility_mean=0.9 if pose_detected else 0.0,
                missing_ratio=0.0 if pose_detected else 1.0,
                source={
                    "frame_index": index,
                    "timestamp_ms": timestamp,
                    "left_knee_angle": knee,
                    "right_knee_angle": knee,
                    "left_hip_angle": hip,
                    "right_hip_angle": hip,
                    "trunk_tilt_proxy": 10.0 + pelvis_y,
                    "pose_detected": int(pose_detected),
                    "visibility_mean": 0.9 if pose_detected else 0.0,
                    "missing_ratio": 0.0 if pose_detected else 1.0,
                },
            )
        )
    return frames


def make_squat_session(tmp_path: Path, session_id: str = "squat_session") -> Path:
    frames = make_squat_measurements()
    session_dir = tmp_path / "sessions" / session_id
    session_dir.mkdir(parents=True)
    metadata = {
        "session_id": session_id,
        "camera_index": 0,
        "actual_resolution": {"width": 1280, "height": 720},
        "average_fps": 20.0,
        "mirror": True,
        "smoothing": 0.65,
        "model_name": "pose_landmarker_full.task",
    }
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    kin_columns = [
        "frame_index",
        "timestamp_ms",
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_shoulder_angle",
        "right_shoulder_angle",
        "trunk_tilt_proxy",
        "left_wrist_speed",
        "right_wrist_speed",
        "left_ankle_speed",
        "right_ankle_speed",
        "pelvis_speed",
        "motion_energy_proxy",
        "pose_detected",
        "visibility_mean",
        "missing_ratio",
    ]
    with (session_dir / "kinematics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=kin_columns)
        writer.writeheader()
        for frame in frames:
            row = dict(frame.source)
            row.update(
                {
                    "left_elbow_angle": 150,
                    "right_elbow_angle": 150,
                    "left_shoulder_angle": 50,
                    "right_shoulder_angle": 50,
                    "left_wrist_speed": 0.1,
                    "right_wrist_speed": 0.1,
                    "left_ankle_speed": 0.05,
                    "right_ankle_speed": 0.05,
                    "pelvis_speed": 0.2,
                    "motion_energy_proxy": 1.0,
                }
            )
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
        for frame in frames:
            y = frame.pelvis_y or 0.0
            points = {
                "left_hip": (-0.2, y, 0.0),
                "right_hip": (0.2, y, 0.0),
                "left_shoulder": (-0.2, y - 1.0, 0.0),
                "right_shoulder": (0.2, y - 1.0, 0.0),
                "left_knee": (-0.2, y + 0.55, 0.0),
                "right_knee": (0.2, y + 0.55, 0.0),
                "left_ankle": (-0.2, y + 1.1, 0.0),
                "right_ankle": (0.2, y + 1.1, 0.0),
            }
            for name, (x_value, y_value, z_value) in points.items():
                writer.writerow(
                    {
                        "frame_index": frame.frame_index,
                        "timestamp_ms": frame.timestamp_ms,
                        "landmark_name": name,
                        "image_x": x_value,
                        "image_y": y_value,
                        "image_z": z_value,
                        "world_x": x_value,
                        "world_y": y_value,
                        "world_z": z_value,
                        "smoothed_x": x_value,
                        "smoothed_y": y_value,
                        "smoothed_z": z_value,
                        "visibility": frame.visibility_mean,
                        "presence": frame.visibility_mean,
                    }
                )
    return session_dir

