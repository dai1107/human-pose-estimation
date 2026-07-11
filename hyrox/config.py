from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_LUNGE_CONFIG: dict[str, Any] = {
    "action_name": "lunge",
    "config_name": "lunge_default",
    "visibility_min": 0.45,
    "stand_knee_angle_min": 150.0,
    "stand_hip_angle_min": 145.0,
    "full_extension_knee_angle_min": 165.0,
    "full_extension_hip_angle_min": 160.0,
    "bottom_knee_angle_max": 115.0,
    "deep_knee_angle_max": 100.0,
    "torso_lean_warn": 20.0,
    "motion_tolerance": 3.0,
    "hip_motion_tolerance": 0.004,
    "hip_drop_min": 0.035,
    "stable_frames": 3,
    "rep_cooldown_ms": 400,
}

DEFAULT_WALL_BALL_CONFIG: dict[str, Any] = {
    "action_name": "wall_ball",
    "config_name": "wall_ball_default",
    "visibility_min": 0.45,
    "stand_knee_angle_min": 150.0,
    "stand_hip_angle_min": 145.0,
    "bottom_knee_angle_max": 110.0,
    "hip_below_knee_margin": -0.05,
    "throw_knee_angle_min": 150.0,
    "throw_hip_angle_min": 145.0,
    "throw_elbow_angle_min": 125.0,
    "wrist_above_shoulder_min": 0.03,
    "full_extension_knee_angle_min": 165.0,
    "full_extension_hip_angle_min": 160.0,
    "knee_cave_ratio_max": 0.72,
    "minimum_frontal_ankle_width": 0.08,
    "motion_tolerance": 3.0,
    "hip_motion_tolerance": 0.004,
    "stable_frames": 3,
    "rep_cooldown_ms": 400,
}

DEFAULT_FARMERS_CARRY_CONFIG: dict[str, Any] = {
    "action_name": "farmers_carry",
    "config_name": "farmers_carry_default",
    "visibility_min": 0.55,
    "shoulder_tilt_warn": 0.08,
    "hip_tilt_warn": 0.08,
    "torso_lean_warn": 25.0,
    "arms_down_margin": 0.05,
    "stable_frames": 3,
    "cooldown_ms": 350,
    "rest_timeout_ms": 1200,
}

DEFAULT_ROWING_CONFIG: dict[str, Any] = {
    "action_name": "rowing",
    "config_name": "rowing_default",
    "visibility_min": 0.55,
    "catch_knee_angle_max": 105.0,
    "finish_knee_angle_min": 145.0,
    "finish_torso_lean_max": 35.0,
    "too_much_back_lean": 45.0,
    "early_arm_pull_elbow_angle": 120.0,
    "stable_frames": 3,
    "stroke_cooldown_ms": 500,
    "min_phase_duration_ms": 120,
}

DEFAULT_SKIERG_CONFIG: dict[str, Any] = {
    "action_name": "skierg",
    "config_name": "skierg_default",
    "visibility_min": 0.50,
    "top_wrist_above_shoulder_margin": 0.03,
    "bottom_wrist_below_chest_margin": 0.05,
    "hip_hinge_torso_angle_min": 15.0,
    "too_much_squat_knee_angle_max": 110.0,
    "wrist_asymmetry_warn": 0.08,
    "stable_frames": 3,
    "pull_cooldown_ms": 350,
    "min_phase_duration_ms": 100,
}

DEFAULT_BURPEE_BROAD_JUMP_CONFIG: dict[str, Any] = {
    "action_name": "burpee_broad_jump",
    "config_name": "burpee_broad_jump_default",
    "visibility_min": 0.55,
    "chest_down_body_height_max": 0.35,
    "bottom_torso_horizontal_min": 60.0,
    "jump_forward_min_delta_x": 0.08,
    "feet_stagger_warn": 0.08,
    "extra_step_window_ms": 500,
    "stable_frames": 3,
    "rep_cooldown_ms": 800,
    "min_phase_duration_ms": 100,
}

DEFAULT_SLED_PUSH_CONFIG: dict[str, Any] = {
    "action_name": "sled_push",
    "config_name": "sled_push_default",
    "visibility_min": 0.55,
    "drive_torso_angle_min": 25.0,
    "drive_torso_angle_max": 65.0,
    "too_upright_angle": 20.0,
    "too_low_angle": 70.0,
    "leg_drive_knee_extension_min": 20.0,
    "short_step_ankle_delta_min": 0.04,
    "stable_frames": 3,
    "step_cooldown_ms": 250,
}

DEFAULT_SLED_PULL_CONFIG: dict[str, Any] = {
    "action_name": "sled_pull",
    "config_name": "sled_pull_default",
    "visibility_min": 0.55,
    "not_standing_knee_angle_max": 95.0,
    "not_standing_body_center_y_max": 0.75,
    "over_lean_back_angle": 35.0,
    "pull_elbow_delta_min": 25.0,
    "hip_knee_drive_delta_min": 8.0,
    "wrist_asymmetry_warn": 0.08,
    "stable_frames": 3,
    "pull_cooldown_ms": 350,
}

DEFAULT_HYROX_CONFIG_PATHS: dict[str, str] = {
    "lunge": "configs/hyrox/lunge.yaml",
    "wall_ball": "configs/hyrox/wall_ball.yaml",
    "farmers_carry": "configs/hyrox/farmers_carry.yaml",
    "rowing": "configs/hyrox/rowing.yaml",
    "skierg": "configs/hyrox/skierg.yaml",
    "burpee_broad_jump": "configs/hyrox/burpee_broad_jump.yaml",
    "sled_push": "configs/hyrox/sled_push.yaml",
    "sled_pull": "configs/hyrox/sled_pull.yaml",
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


def _load_flat_config(
    defaults: dict[str, Any],
    path: str | Path | None,
    default_path: str,
) -> dict[str, Any]:
    config = dict(defaults)
    config_path = Path(path) if path is not None else PROJECT_ROOT / default_path
    if not config_path.exists():
        return config

    parsed: dict[str, Any] = {}
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return config
    parent_key: str | None = None
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            parent_key = key if not value else None
        if indent > 0 and parent_key:
            nested = parsed.setdefault(parent_key, {})
            if isinstance(nested, dict) and value:
                nested[key] = _parse_scalar(value)
        elif value:
            parsed[key] = _parse_scalar(value)
    if "config_name" not in parsed:
        parsed["config_name"] = config_path.stem
    config.update(parsed)
    return config


def load_lunge_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(DEFAULT_LUNGE_CONFIG, path, "configs/hyrox/lunge.yaml")


def load_wall_ball_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(DEFAULT_WALL_BALL_CONFIG, path, "configs/hyrox/wall_ball.yaml")


def load_farmers_carry_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_FARMERS_CARRY_CONFIG,
        path,
        "configs/hyrox/farmers_carry.yaml",
    )


def load_rowing_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(DEFAULT_ROWING_CONFIG, path, "configs/hyrox/rowing.yaml")


def load_skierg_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(DEFAULT_SKIERG_CONFIG, path, "configs/hyrox/skierg.yaml")


def load_burpee_broad_jump_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_BURPEE_BROAD_JUMP_CONFIG,
        path,
        "configs/hyrox/burpee_broad_jump.yaml",
    )


def load_sled_push_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(DEFAULT_SLED_PUSH_CONFIG, path, "configs/hyrox/sled_push.yaml")


def load_sled_pull_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(DEFAULT_SLED_PULL_CONFIG, path, "configs/hyrox/sled_pull.yaml")


def resolve_hyrox_config_path(action: str, configured_path: str | None = None) -> str | None:
    if configured_path:
        return configured_path
    return DEFAULT_HYROX_CONFIG_PATHS.get(action)


__all__ = [
    "DEFAULT_LUNGE_CONFIG",
    "DEFAULT_WALL_BALL_CONFIG",
    "DEFAULT_FARMERS_CARRY_CONFIG",
    "DEFAULT_ROWING_CONFIG",
    "DEFAULT_SKIERG_CONFIG",
    "DEFAULT_BURPEE_BROAD_JUMP_CONFIG",
    "DEFAULT_SLED_PUSH_CONFIG",
    "DEFAULT_SLED_PULL_CONFIG",
    "DEFAULT_HYROX_CONFIG_PATHS",
    "load_lunge_config",
    "load_wall_ball_config",
    "load_farmers_carry_config",
    "load_rowing_config",
    "load_skierg_config",
    "load_burpee_broad_jump_config",
    "load_sled_push_config",
    "load_sled_pull_config",
    "resolve_hyrox_config_path",
]
