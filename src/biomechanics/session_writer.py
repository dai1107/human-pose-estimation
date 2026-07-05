from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

from src.utils.time_utils import make_session_id, now_iso

from .hand_landmarks import SUPPLEMENTAL_FINGER_JOINTS, empty_hand_landmarks, hand_landmark_name
from .landmarks import LANDMARK_NAMES, empty_landmarks
from .report import write_report_outputs
from .types import KinematicFrame, LandmarkPoint, PoseFrame


@dataclass(frozen=True)
class SessionConfig:
    camera_index: int
    width: int
    height: int
    mirror: bool
    smoothing: float
    model_name: str
    plot_on_save: bool = True
    landmark_profile: str = "full"
    hands_enabled: bool = False
    hand_model_name: str | None = None


def _number(value: float | int | None) -> str:
    if value is None:
        return ""
    value = float(value)
    return f"{value:.8g}" if isfinite(value) else ""


def _landmark_at(points: list[LandmarkPoint], index: int) -> LandmarkPoint:
    if 0 <= index < len(points):
        return points[index]
    return empty_landmarks(1)[0]


def _hand_landmark_at(points: list[LandmarkPoint], index: int) -> LandmarkPoint:
    if 0 <= index < len(points):
        return points[index]
    return empty_hand_landmarks(1)[0]


def _ordered_hand_items(hand_map: dict[str, list[LandmarkPoint]]) -> list[tuple[str, list[LandmarkPoint]]]:
    side_order = {"left": 0, "right": 1}
    return sorted(hand_map.items(), key=lambda item: (side_order.get(item[0], 2), item[0]))


class SessionWriter:
    def __init__(self, save_dir: Path | str = "outputs") -> None:
        self.save_dir = Path(save_dir)
        self.session_dir: Path | None = None
        self.session_id: str | None = None
        self.config: SessionConfig | None = None
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.pose_frames: list[PoseFrame] = []
        self.kinematic_frames: list[KinematicFrame] = []

    @property
    def is_active(self) -> bool:
        return self.session_dir is not None

    def start(self, config: SessionConfig, session_id: str | None = None) -> str:
        if self.is_active:
            raise RuntimeError("session is already active")
        base_session_id = session_id or make_session_id()
        session_dir = self.save_dir / "sessions" / base_session_id
        if session_dir.exists():
            suffix = 1
            while (self.save_dir / "sessions" / f"{base_session_id}_{suffix}").exists():
                suffix += 1
            base_session_id = f"{base_session_id}_{suffix}"
            session_dir = self.save_dir / "sessions" / base_session_id
        self.session_id = base_session_id
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.started_at = now_iso()
        self.ended_at = None
        self.pose_frames = []
        self.kinematic_frames = []
        return self.session_id

    def add_frame(self, pose_frame: PoseFrame, kinematic_frame: KinematicFrame) -> None:
        if not self.is_active:
            return
        self.pose_frames.append(pose_frame)
        self.kinematic_frames.append(kinematic_frame)

    def stop(self, final_mirror: bool | None = None) -> Path | None:
        if not self.is_active or self.session_dir is None or self.config is None:
            return None
        self.ended_at = now_iso()
        session_dir = self.session_dir
        self._write_landmarks_csv(session_dir / "landmarks.csv")
        self._write_kinematics_csv(session_dir / "kinematics.csv")
        write_report_outputs(session_dir, self.pose_frames, self.kinematic_frames, plot_on_save=self.config.plot_on_save)
        self._write_metadata(session_dir / "metadata.json", final_mirror=final_mirror)

        self.session_dir = None
        self.config = None
        return session_dir

    def _metadata_payload(self, final_mirror: bool | None) -> dict[str, Any]:
        assert self.config is not None
        total = len(self.pose_frames)
        detected = sum(1 for frame in self.pose_frames if frame.pose_detected)
        hands_detected = sum(1 for frame in self.pose_frames if frame.hands_detected)
        fps_values = [frame.fps for frame in self.pose_frames if isfinite(frame.fps) and frame.fps > 0]
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "camera_index": self.config.camera_index,
            "actual_resolution": {"width": self.config.width, "height": self.config.height},
            "average_fps": sum(fps_values) / len(fps_values) if fps_values else 0.0,
            "mirror": self.config.mirror,
            "mirror_at_end": final_mirror if final_mirror is not None else self.config.mirror,
            "smoothing": self.config.smoothing,
            "model_name": self.config.model_name,
            "landmark_profile": self.config.landmark_profile,
            "hands_enabled": self.config.hands_enabled,
            "hand_model_name": self.config.hand_model_name,
            "landmark_frame_count": total,
            "pose_detected_frame_count": detected,
            "pose_lost_frame_count": total - detected,
            "hands_detected_frame_count": hands_detected,
        }

    def _write_metadata(self, path: Path, final_mirror: bool | None) -> None:
        path.write_text(json.dumps(self._metadata_payload(final_mirror), indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_landmarks_csv(self, path: Path) -> None:
        columns = [
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
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for frame in self.pose_frames:
                for index, name in enumerate(LANDMARK_NAMES):
                    image = _landmark_at(frame.image_landmarks, index)
                    world = _landmark_at(frame.world_landmarks, index)
                    smoothed = _landmark_at(frame.smoothed_landmarks, index)
                    writer.writerow(
                        {
                            "frame_index": frame.frame_index,
                            "timestamp_ms": frame.timestamp_ms,
                            "landmark_name": name,
                            "image_x": _number(image.x),
                            "image_y": _number(image.y),
                            "image_z": _number(image.z),
                            "world_x": _number(world.x),
                            "world_y": _number(world.y),
                            "world_z": _number(world.z),
                            "smoothed_x": _number(smoothed.x),
                            "smoothed_y": _number(smoothed.y),
                            "smoothed_z": _number(smoothed.z),
                            "visibility": _number(image.visibility),
                            "presence": _number(image.presence),
                        }
                    )
                for side, image_points in _ordered_hand_items(frame.hand_landmarks):
                    world_points = frame.hand_world_landmarks.get(side, [])
                    smoothed_points = frame.smoothed_hand_landmarks.get(side, [])
                    for _, index in SUPPLEMENTAL_FINGER_JOINTS:
                        image = _hand_landmark_at(image_points, index)
                        world = _hand_landmark_at(world_points, index)
                        smoothed = _hand_landmark_at(smoothed_points, index)
                        writer.writerow(
                            {
                                "frame_index": frame.frame_index,
                                "timestamp_ms": frame.timestamp_ms,
                                "landmark_name": hand_landmark_name(side, index),
                                "image_x": _number(image.x),
                                "image_y": _number(image.y),
                                "image_z": _number(image.z),
                                "world_x": _number(world.x),
                                "world_y": _number(world.y),
                                "world_z": _number(world.z),
                                "smoothed_x": _number(smoothed.x),
                                "smoothed_y": _number(smoothed.y),
                                "smoothed_z": _number(smoothed.z),
                                "visibility": _number(image.visibility),
                                "presence": _number(image.presence),
                            }
                        )

    def _write_kinematics_csv(self, path: Path) -> None:
        columns = [
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
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for frame in self.kinematic_frames:
                row = {
                    "frame_index": frame.frame_index,
                    "timestamp_ms": frame.timestamp_ms,
                    "pose_detected": int(frame.pose_detected),
                    "visibility_mean": _number(frame.quality.get("visibility_mean")),
                    "missing_ratio": _number(frame.quality.get("missing_ratio")),
                }
                for column in columns:
                    if column not in row and hasattr(frame, column):
                        row[column] = _number(getattr(frame, column))
                writer.writerow(row)
