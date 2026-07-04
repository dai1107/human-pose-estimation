from __future__ import annotations

import csv
import json
from pathlib import Path


KIN_COLUMNS = [
    "frame_index",
    "timestamp_ms",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_elbow_angular_velocity",
    "right_elbow_angular_velocity",
    "left_wrist_speed",
    "right_wrist_speed",
    "left_ankle_speed",
    "right_ankle_speed",
    "pelvis_speed",
    "shoulder_center_speed",
    "trunk_tilt_proxy",
    "motion_energy_proxy",
    "pose_detected",
    "visibility_mean",
    "missing_ratio",
]


def make_shot_session(tmp_path: Path, session_id: str = "shot_session", low_visibility: bool = False) -> Path:
    session_dir = tmp_path / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text(json.dumps({"session_id": session_id, "mirror": True}, indent=2), encoding="utf-8")
    rows = []
    timestamps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    knee = [172, 168, 150, 138, 150, 162, 170, 174, 174, 174]
    hip = [170, 166, 148, 140, 152, 164, 171, 174, 174, 174]
    elbow = [70, 72, 76, 82, 98, 120, 148, 168, 172, 170]
    elbow_v = [0, 20, 40, 60, 120, 220, 300, 180, 40, -10]
    wrist_speed = [0.05, 0.08, 0.12, 0.25, 0.55, 0.95, 1.25, 0.75, 0.35, 0.1]
    pelvis_y = [0.0, 0.0, 0.08, 0.16, 0.10, 0.03, -0.02, -0.02, 0.0, 0.0]
    wrist_y = [-0.45, -0.50, -0.55, -0.70, -0.95, -1.18, -1.32, -1.35, -1.28, -1.10]
    visibility = 0.35 if low_visibility else 0.9
    with (session_dir / "kinematics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=KIN_COLUMNS)
        writer.writeheader()
        for i, ts in enumerate(timestamps):
            row = {
                "frame_index": i,
                "timestamp_ms": ts,
                "left_knee_angle": knee[i],
                "right_knee_angle": knee[i],
                "left_hip_angle": hip[i],
                "right_hip_angle": hip[i],
                "left_shoulder_angle": 80 + i * 3,
                "right_shoulder_angle": 80 + i * 3,
                "left_elbow_angle": elbow[i],
                "right_elbow_angle": elbow[i],
                "left_elbow_angular_velocity": elbow_v[i],
                "right_elbow_angular_velocity": elbow_v[i],
                "left_wrist_speed": wrist_speed[i],
                "right_wrist_speed": wrist_speed[i],
                "left_ankle_speed": 0.1,
                "right_ankle_speed": 0.1,
                "pelvis_speed": abs(pelvis_y[i] - pelvis_y[i - 1]) * 10 if i else 0,
                "shoulder_center_speed": 0.2 + i * 0.02,
                "trunk_tilt_proxy": 8 + i,
                "motion_energy_proxy": 0.5 + wrist_speed[i],
                "pose_detected": 1,
                "visibility_mean": visibility,
                "missing_ratio": 0.05 if not low_visibility else 0.5,
            }
            rows.append(row)
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
        for i, ts in enumerate(timestamps):
            py = pelvis_y[i]
            points = {
                "left_hip": (-0.2, py, 0.0),
                "right_hip": (0.2, py, 0.0),
                "left_shoulder": (-0.2, py - 1.0, 0.0),
                "right_shoulder": (0.2, py - 1.0, 0.0),
                "left_elbow": (-0.2, wrist_y[i] + 0.25, 0.0),
                "right_elbow": (0.2, wrist_y[i] + 0.25, 0.0),
                "left_wrist": (-0.18, wrist_y[i], 0.0),
                "right_wrist": (0.22, wrist_y[i], 0.0),
                "left_knee": (-0.2, py + 0.55, 0.0),
                "right_knee": (0.2, py + 0.55, 0.0),
                "left_ankle": (-0.2, py + 1.1, 0.0),
                "right_ankle": (0.2, py + 1.1, 0.0),
            }
            for name, (x, y, z) in points.items():
                writer.writerow(
                    {
                        "frame_index": i,
                        "timestamp_ms": ts,
                        "landmark_name": name,
                        "image_x": x,
                        "image_y": y,
                        "image_z": z,
                        "world_x": x,
                        "world_y": y,
                        "world_z": z,
                        "smoothed_x": x,
                        "smoothed_y": y,
                        "smoothed_z": z,
                        "visibility": visibility,
                        "presence": visibility,
                    }
                )
    return session_dir

