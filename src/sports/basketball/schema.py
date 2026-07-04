from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any


SHOT_TYPES = {"set_shot", "jump_shot"}
SHOOTING_SIDES = {"right", "left"}
CAMERA_VIEWS = {"side", "front_left", "front_right", "front", "unknown"}
SHOT_PHASES = ("IDLE", "SETUP", "DIP", "RISE", "ARM_EXTENSION", "RELEASE_PROXY", "FOLLOW_THROUGH", "RECOVERY")


@dataclass(frozen=True)
class ShotFrame:
    frame_index: int
    timestamp_ms: int
    pose_detected: bool
    pelvis_x: float | None = None
    pelvis_y: float | None = None
    shoulder_x: float | None = None
    shoulder_y: float | None = None
    body_scale: float | None = None
    shooting_side: str = "right"
    shooting_knee_angle: float | None = None
    non_shooting_knee_angle: float | None = None
    shooting_hip_angle: float | None = None
    non_shooting_hip_angle: float | None = None
    shooting_shoulder_angle: float | None = None
    shooting_elbow_angle: float | None = None
    shooting_elbow_angular_velocity: float | None = None
    shooting_wrist_speed: float | None = None
    shooting_ankle_speed: float | None = None
    pelvis_speed: float | None = None
    shoulder_center_speed: float | None = None
    trunk_tilt_proxy: float | None = None
    motion_energy_proxy: float | None = None
    shooting_wrist_x: float | None = None
    shooting_wrist_y: float | None = None
    shooting_elbow_x: float | None = None
    shooting_elbow_y: float | None = None
    shooting_shoulder_x: float | None = None
    shooting_shoulder_y: float | None = None
    visibility_mean: float | None = None
    missing_ratio: float | None = None
    source: dict[str, Any] = field(default_factory=dict)

    def usable(self, minimum_visibility: float = 0.2) -> bool:
        if not self.pose_detected:
            return False
        if self.visibility_mean is not None and self.visibility_mean < minimum_visibility:
            return False
        return True

    def wrist_relative_shoulder_height(self) -> float | None:
        if self.shooting_wrist_y is None or self.shooting_shoulder_y is None or not self.body_scale:
            return None
        return (self.shooting_shoulder_y - self.shooting_wrist_y) / self.body_scale

    def wrist_relative_elbow_height(self) -> float | None:
        if self.shooting_wrist_y is None or self.shooting_elbow_y is None or not self.body_scale:
            return None
        return (self.shooting_elbow_y - self.shooting_wrist_y) / self.body_scale


@dataclass(frozen=True)
class ShotClip:
    session_id: str
    start_ms: int
    end_ms: int
    start_frame: int | None
    end_frame: int | None
    frames: list[ShotFrame]
    kinematics_rows: list[dict[str, Any]]
    landmark_rows: list[dict[str, Any]]

    def to_range_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_ms": self.end_ms - self.start_ms,
        }


@dataclass(frozen=True)
class ShotEvent:
    event: str
    timestamp_ms: int | None
    normalized_time_percent: float | None
    confidence: float
    source_signal: str
    data_quality: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseProxy:
    release_proxy_time: int | None
    release_proxy_confidence: float
    release_proxy_reason: str
    release_source: str = "auto"
    automatic_time: int | None = None
    automatic_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseResult:
    phase_by_frame: list[dict[str, Any]]
    phase_timestamps: dict[str, int | None]
    uncertain: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def unique_output_dir(root: str | Path, base_id: str) -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    candidate = path / base_id
    if not candidate.exists():
        return candidate
    suffix = 1
    while (path / f"{base_id}_{suffix}").exists():
        suffix += 1
    return path / f"{base_id}_{suffix}"

