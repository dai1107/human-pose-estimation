from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any


CAMERA_VIEWS = {"side", "front", "front_left", "front_right", "unknown"}
SQUAT_STATES = {"IDLE", "READY", "DESCENT", "BOTTOM", "ASCENT", "COMPLETE", "PAUSED"}


DEFAULT_SQUAT_CONFIG: dict[str, Any] = {
    "analysis_name": "squat_basic_v1",
    "data_quality": {
        "minimum_pose_valid_ratio": 0.75,
        "minimum_landmark_visibility": 0.65,
    },
    "rep_detection": {
        "min_rep_duration_ms": 600,
        "min_descent_duration_ms": 180,
        "min_ascent_duration_ms": 180,
        "bottom_hold_max_ms": 1200,
        "baseline_return_tolerance": 0.15,
        "descent_start_displacement": 0.08,
        "bottom_velocity_threshold": 0.15,
        "min_knee_flexion_delta": 8.0,
        "min_hip_flexion_delta": 6.0,
        "min_bottom_displacement": 0.16,
        "stable_ready_ms": 180,
        "lost_reset_ms": 350,
    },
    "metrics": {
        "enable_side_view_metrics": True,
        "enable_front_view_metrics": True,
        "enable_reference_comparison": True,
    },
    "interpretation": {
        "mode": "descriptive_only",
        "allow_standard_verdict": False,
        "evidence_source": None,
        "evidence_note": "Do not enable until externally verified.",
    },
}


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "null":
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_squat_config(path: str | Path | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_SQUAT_CONFIG)
    config["data_quality"] = dict(DEFAULT_SQUAT_CONFIG["data_quality"])
    config["rep_detection"] = dict(DEFAULT_SQUAT_CONFIG["rep_detection"])
    config["metrics"] = dict(DEFAULT_SQUAT_CONFIG["metrics"])
    config["interpretation"] = dict(DEFAULT_SQUAT_CONFIG["interpretation"])
    config_path = Path(path) if path is not None else Path("configs/squat_basic_v1.yaml")
    if not config_path.exists():
        return config

    parsed: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        if not line.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            if value.strip():
                parsed[key] = _parse_scalar(value)
                current_section = None
            else:
                parsed[key] = {}
                current_section = key
        elif current_section:
            key, value = line.strip().split(":", 1)
            parsed[current_section][key.strip()] = _parse_scalar(value)
    return _merge_dict(config, parsed)


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


@dataclass(frozen=True)
class SquatFrameMeasurement:
    frame_index: int
    timestamp_ms: int
    pose_detected: bool
    pelvis_x: float | None = None
    pelvis_y: float | None = None
    shoulder_x: float | None = None
    shoulder_y: float | None = None
    body_scale: float | None = None
    left_knee_angle: float | None = None
    right_knee_angle: float | None = None
    left_hip_angle: float | None = None
    right_hip_angle: float | None = None
    trunk_tilt_proxy: float | None = None
    left_ankle_x: float | None = None
    right_ankle_x: float | None = None
    visibility_mean: float | None = None
    missing_ratio: float | None = None
    source: dict[str, Any] = field(default_factory=dict)

    def usable(self, min_visibility: float = 0.2) -> bool:
        if not self.pose_detected:
            return False
        if self.pelvis_y is None or self.body_scale is None or self.body_scale <= 1e-9:
            return False
        if self.visibility_mean is not None and self.visibility_mean < min_visibility:
            return False
        return True

    def mean_knee_angle(self) -> float | None:
        values = [value for value in (self.left_knee_angle, self.right_knee_angle) if value is not None]
        return sum(values) / len(values) if values else None

    def mean_hip_angle(self) -> float | None:
        values = [value for value in (self.left_hip_angle, self.right_hip_angle) if value is not None]
        return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class SquatCalibration:
    status: str
    baseline_pelvis_y: float | None
    baseline_pelvis_x: float | None
    baseline_body_scale: float | None
    baseline_left_knee_angle: float | None
    baseline_right_knee_angle: float | None
    baseline_left_hip_angle: float | None
    baseline_right_hip_angle: float | None
    baseline_trunk_tilt_proxy: float | None
    visibility_mean: float
    stability_proxy: float | None
    camera_view: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    recommended_position: str = "Keep hips, knees, ankles, and shoulders visible."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return self.status in {"PASS", "WARNING"}


@dataclass(frozen=True)
class SquatRep:
    rep_index: int
    start_timestamp_ms: int
    bottom_timestamp_ms: int
    end_timestamp_ms: int
    start_frame_index: int
    bottom_frame_index: int
    end_frame_index: int
    frames: list[SquatFrameMeasurement]


@dataclass(frozen=True)
class SquatRepMetrics:
    rep_index: int
    start_timestamp_ms: int
    bottom_timestamp_ms: int
    end_timestamp_ms: int
    total_duration_ms: int
    descent_duration_ms: int
    bottom_duration_ms: int
    ascent_duration_ms: int
    left_knee_min_angle: float | None
    right_knee_min_angle: float | None
    left_hip_min_angle: float | None
    right_hip_min_angle: float | None
    left_knee_angle_range: float | None
    right_knee_angle_range: float | None
    left_hip_angle_range: float | None
    right_hip_angle_range: float | None
    trunk_tilt_range: float | None
    pelvis_vertical_displacement_normalized: float | None
    left_right_knee_difference_mean: float | None
    left_right_knee_difference_peak: float | None
    left_right_hip_difference_mean: float | None
    left_right_hip_difference_peak: float | None
    pelvis_lateral_drift_proxy: float | None
    trunk_lateral_drift_proxy: float | None
    pose_valid_ratio: float
    data_quality_level: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

