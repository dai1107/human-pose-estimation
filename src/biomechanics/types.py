from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0

    def is_finite(self) -> bool:
        return isfinite(self.x) and isfinite(self.y) and isfinite(self.z)

    def is_usable(self, min_visibility: float = 0.2, min_presence: float = 0.2) -> bool:
        return self.is_finite() and self.visibility >= min_visibility and self.presence >= min_presence

    def xyz(self) -> Vector3:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class NormalizationResult:
    landmarks: list[LandmarkPoint]
    success: bool
    origin: Vector3 | None
    scale: float | None
    scale_method: str | None
    message: str = ""


@dataclass(frozen=True)
class SegmentVector:
    name: str
    start: str
    end: str
    vector: Vector3
    unit: Vector3
    valid: bool


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    timestamp_ms: int
    pose_detected: bool
    image_landmarks: list[LandmarkPoint] = field(default_factory=list)
    world_landmarks: list[LandmarkPoint] = field(default_factory=list)
    smoothed_landmarks: list[LandmarkPoint] = field(default_factory=list)
    normalized_landmarks: list[LandmarkPoint] = field(default_factory=list)
    hands_detected: bool = False
    hand_landmarks: dict[str, list[LandmarkPoint]] = field(default_factory=dict)
    hand_world_landmarks: dict[str, list[LandmarkPoint]] = field(default_factory=dict)
    smoothed_hand_landmarks: dict[str, list[LandmarkPoint]] = field(default_factory=dict)
    normalization_success: bool = False
    normalization_message: str = ""
    mirror: bool = True
    camera_width: int = 0
    camera_height: int = 0
    fps: float = 0.0


@dataclass(frozen=True)
class KinematicFrame:
    frame_index: int
    timestamp_ms: int
    pose_detected: bool
    left_elbow_angle: float
    right_elbow_angle: float
    left_knee_angle: float
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float
    left_shoulder_angle: float
    right_shoulder_angle: float
    trunk_tilt_proxy: float
    left_wrist_speed: float
    right_wrist_speed: float
    left_ankle_speed: float
    right_ankle_speed: float
    pelvis_speed: float
    shoulder_center_speed: float
    left_elbow_angular_velocity: float
    right_elbow_angular_velocity: float
    left_knee_angular_velocity: float
    right_knee_angular_velocity: float
    left_hip_angular_velocity: float
    right_hip_angular_velocity: float
    motion_energy_proxy: float
    segment_vectors: dict[str, SegmentVector] = field(default_factory=dict)
    angular_velocities: dict[str, float] = field(default_factory=dict)
    stability: dict[str, float] = field(default_factory=dict)
    symmetry: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
