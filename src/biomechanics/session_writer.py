from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

from src.utils.time_utils import make_session_id, now_iso
from src.version import __version__
from src.output_schema import (
    OUTPUT_SCHEMA_VERSION,
    artifact_metadata,
    versioned_csv_columns,
    versioned_csv_row,
)

from .hand_landmarks import SUPPLEMENTAL_FINGER_JOINTS, empty_hand_landmarks, hand_landmark_name
from .kinematics_3d import ANGLE_DEFINITIONS_3D, summarize_three_d_records
from .landmarks import LANDMARK_NAMES, empty_landmarks
from .report import write_report_outputs
from .types import KinematicFrame, LandmarkPoint, PoseFrame


POSE_LANDMARK_PROFILES: dict[str, frozenset[int]] = {
    "full": frozenset(range(len(LANDMARK_NAMES))),
    "no-face": frozenset(range(11, len(LANDMARK_NAMES))),
    "upper-body": frozenset({11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}),
    "lower-body": frozenset({23, 24, 25, 26, 27, 28, 29, 30, 31, 32}),
    "shot": frozenset({11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}),
}


class SessionWriteError(RuntimeError):
    error_code = "OUT002"

    def __init__(
        self,
        session_dir: Path,
        cause: Exception,
        recovered_files: list[str],
    ) -> None:
        self.session_dir = session_dir
        self.cause = cause
        self.recovered_files = recovered_files
        recovered = ", ".join(recovered_files) if recovered_files else "none"
        super().__init__(
            f"session save incomplete at {session_dir}: {cause}; "
            f"recoverable files: {recovered}"
        )


@dataclass(frozen=True)
class SessionConfig:
    camera_index: int
    width: int
    height: int
    mirror: bool
    smoothing: float
    model_name: str
    plot_on_save: bool = True
    landmark_profile: str = "no-face"
    hands_enabled: bool = False
    hand_model_name: str | None = None
    program_version: str = __version__
    schema_version: int = OUTPUT_SCHEMA_VERSION
    camera_view: str = "unknown"


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
        try:
            self._write_metadata(
                session_dir / "metadata.json",
                final_mirror=final_mirror,
                write_status="partial",
            )
            self._write_landmarks_csv(session_dir / "landmarks.csv")
            self._write_kinematics_csv(session_dir / "kinematics.csv")
            self._write_three_d_kinematics_csv(session_dir / "kinematics_3d.csv")
            write_report_outputs(
                session_dir,
                self.pose_frames,
                self.kinematic_frames,
                plot_on_save=self.config.plot_on_save,
            )
            self._write_metadata(
                session_dir / "metadata.json",
                final_mirror=final_mirror,
                write_status="complete",
            )
        except Exception as exc:
            recovered_files: list[str] = []
            try:
                candidates = list(session_dir.iterdir())
            except OSError:
                candidates = []
            for candidate in candidates:
                if candidate.name.endswith(".tmp"):
                    continue
                try:
                    if candidate.is_file() and candidate.stat().st_size > 0:
                        recovered_files.append(candidate.name)
                except OSError:
                    continue
            recovered_files.sort()
            try:
                self._write_metadata(
                    session_dir / "metadata.json",
                    final_mirror=final_mirror,
                    write_status="partial",
                    recovery_error=f"{type(exc).__name__}: {exc}",
                    recovered_files=recovered_files,
                )
            except Exception:
                pass
            raise SessionWriteError(session_dir, exc, recovered_files) from exc
        finally:
            self.session_dir = None
            self.config = None
            self.pose_frames = []
            self.kinematic_frames = []
        return session_dir

    def _metadata_payload(
        self,
        final_mirror: bool | None,
        *,
        write_status: str = "complete",
        recovery_error: str | None = None,
        recovered_files: list[str] | None = None,
    ) -> dict[str, Any]:
        assert self.config is not None
        total = len(self.pose_frames)
        detected = sum(1 for frame in self.pose_frames if frame.pose_detected)
        hands_detected = sum(1 for frame in self.pose_frames if frame.hands_detected)
        fps_values = [frame.fps for frame in self.pose_frames if isfinite(frame.fps) and frame.fps > 0]
        three_d_summary = summarize_three_d_records(self.pose_frames)
        return {
            **artifact_metadata(
                "pose_session",
                schema_version=self.config.schema_version,
            ),
            "program_version": self.config.program_version,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "camera_index": self.config.camera_index,
            "camera_view": self.config.camera_view,
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
            "three_d_kinematics": three_d_summary,
            "write_status": write_status,
            "recovery_error": recovery_error,
            "recovered_files": list(recovered_files or []),
        }

    def _write_metadata(
        self,
        path: Path,
        final_mirror: bool | None,
        *,
        write_status: str = "complete",
        recovery_error: str | None = None,
        recovered_files: list[str] | None = None,
    ) -> None:
        payload = self._metadata_payload(
            final_mirror,
            write_status=write_status,
            recovery_error=recovery_error,
            recovered_files=recovered_files,
        )
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def _pose_landmark_items(self) -> list[tuple[int, str]]:
        assert self.config is not None
        indices = POSE_LANDMARK_PROFILES.get(self.config.landmark_profile, POSE_LANDMARK_PROFILES["no-face"])
        return [(index, LANDMARK_NAMES[index]) for index in sorted(indices)]

    def _write_landmarks_csv(self, path: Path) -> None:
        columns = versioned_csv_columns([
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
        ])
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            pose_items = self._pose_landmark_items()
            for frame in self.pose_frames:
                for index, name in pose_items:
                    image = _landmark_at(frame.image_landmarks, index)
                    world = _landmark_at(frame.world_landmarks, index)
                    smoothed = _landmark_at(frame.smoothed_landmarks, index)
                    writer.writerow(
                        versioned_csv_row({
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
                        })
                    )
                for side, image_points in _ordered_hand_items(frame.hand_landmarks):
                    world_points = frame.hand_world_landmarks.get(side, [])
                    smoothed_points = frame.smoothed_hand_landmarks.get(side, [])
                    for _, index in SUPPLEMENTAL_FINGER_JOINTS:
                        image = _hand_landmark_at(image_points, index)
                        world = _hand_landmark_at(world_points, index)
                        smoothed = _hand_landmark_at(smoothed_points, index)
                        writer.writerow(
                            versioned_csv_row({
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
                            })
                        )

    def _write_kinematics_csv(self, path: Path) -> None:
        columns = versioned_csv_columns([
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
        ])
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
                writer.writerow(versioned_csv_row(row))

    def _write_three_d_kinematics_csv(self, path: Path) -> None:
        angle_columns = [
            field
            for angle_name in ANGLE_DEFINITIONS_3D
            for field in (
                f"{angle_name}_2d",
                f"{angle_name}_3d",
                f"{angle_name}_2d_3d_difference_deg",
                f"{angle_name}_3d_reliable",
            )
        ]
        columns = versioned_csv_columns(
            [
                "frame_index",
                "timestamp_ms",
                "decision_mode",
                "assist_status",
                "assist_confidence_boost",
                "assist_conflict_confidence_cap",
                "three_d_available",
                "world_landmark_count",
                "three_d_reliable",
                "three_d_reliable_ratio",
                "three_d_conflict_ratio",
                "quality_reasons",
                *angle_columns,
            ]
        )
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for frame in self.pose_frames:
                kinematics = frame.three_d_kinematics
                row: dict[str, object] = {
                    "frame_index": frame.frame_index,
                    "timestamp_ms": frame.timestamp_ms,
                    "decision_mode": kinematics.get("decision_mode", "shadow"),
                    "assist_status": kinematics.get("assist_status", "shadow"),
                    "assist_confidence_boost": _number(
                        kinematics.get("assist_confidence_boost")
                    ),
                    "assist_conflict_confidence_cap": _number(
                        kinematics.get("assist_conflict_confidence_cap")
                    ),
                    "three_d_available": int(bool(kinematics.get("three_d_available"))),
                    "world_landmark_count": kinematics.get("world_landmark_count", 0),
                    "three_d_reliable": int(bool(kinematics.get("three_d_reliable"))),
                    "three_d_reliable_ratio": _number(
                        kinematics.get("three_d_reliable_ratio")
                    ),
                    "three_d_conflict_ratio": _number(
                        kinematics.get("three_d_conflict_ratio")
                    ),
                    "quality_reasons": ";".join(
                        str(reason) for reason in kinematics.get("quality_reasons", [])
                    ),
                }
                for column in angle_columns:
                    value = kinematics.get(column)
                    row[column] = int(value) if isinstance(value, bool) else _number(value)
                writer.writerow(versioned_csv_row(row))
