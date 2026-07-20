from __future__ import annotations

from math import isfinite
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.configuration import (
    ConfigValidationError,
    load_simple_yaml,
    reject_unknown_fields,
)
from src.paths import installation_root, resolve_asset


PROJECT_ROOT = installation_root()


DEFAULT_OBSERVABILITY_CONFIG: dict[str, Any] = {
    "config_name": "observability_default",
    "required_landmark_confidence": 0.60,
    "rep_mean_confidence": 0.65,
    "decisive_rule_confidence": 0.72,
}

DEFAULT_LUNGE_CONFIG: dict[str, Any] = {
    "action_name": "lunge",
    "config_name": "lunge_default",
    "visibility_min": 0.45,
    "stand_knee_angle_min": 150.0,
    "stand_hip_angle_min": 145.0,
    "full_extension_knee_angle_min": 165.0,
    "full_extension_hip_angle_min": 165.0,
    "full_extension_hold_frames_high": 1,
    "full_extension_hold_frames_medium": 2,
    "full_extension_hold_frames_low": 3,
    "bottom_knee_angle_max": 115.0,
    "deep_knee_angle_max": 100.0,
    "torso_lean_warn": 20.0,
    "motion_tolerance": 3.0,
    "hip_motion_tolerance": 0.004,
    "hip_drop_min": 0.035,
    "stable_frames": 2,
    "rep_cooldown_ms": 400,
    "knee_surface_radius_shank_ratio": 0.25,
    "knee_contact_enter_height_body_ratio": 0.060,
    "knee_contact_exit_height_body_ratio": 0.090,
    "side_extension_tolerance_deg": 3.0,
}

DEFAULT_WALL_BALL_CONFIG: dict[str, Any] = {
    "action_name": "wall_ball",
    "config_name": "wall_ball_default",
    "visibility_min": 0.45,
    "stand_knee_angle_min": 150.0,
    "stand_hip_angle_min": 145.0,
    "tall_start_knee_angle_min": 165.0,
    "tall_start_hip_angle_min": 165.0,
    "tall_start_trunk_from_vertical_max_deg": 25.0,
    "bottom_knee_angle_max": 110.0,
    "hip_below_knee_margin": 0.01,
    "throw_knee_angle_min": 150.0,
    "throw_hip_angle_min": 145.0,
    "throw_elbow_angle_min": 125.0,
    "wrist_above_shoulder_min": 0.03,
    "full_extension_knee_angle_min": 165.0,
    "full_extension_hip_angle_min": 165.0,
    "wrist_peak_time_diff_ms_pass": 120,
    "wrist_peak_time_diff_ms_unsure": 220,
    "both_wrists_above_shoulders_required": True,
    "throw_wrist_rise_body_ratio_min": 0.12,
    "throw_wrist_chest_band_body_ratio": 0.25,
    "throw_wrist_midline_body_ratio_max": 0.60,
    "knee_cave_ratio_max": 0.72,
    "minimum_frontal_ankle_width": 0.08,
    "motion_tolerance": 3.0,
    "hip_motion_tolerance": 0.004,
    "stable_frames": 2,
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
    "arm_position_elbow_angle_min_deg": 155.0,
    "wrist_below_hip_margin_body_ratio": 0.03,
    "wrist_lateral_from_hip_max_shoulder_width_ratio": 0.80,
    "arm_position_min_violation_ms": 300,
    "stable_frames": 2,
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
    "standing_violation_knee_angle_min_deg": 160.0,
    "standing_violation_hip_angle_min_deg": 155.0,
    "standing_violation_trunk_from_vertical_max_deg": 30.0,
    "standing_violation_hip_vertical_rise_body_ratio_min": 0.18,
    "standing_violation_min_hold_ms": 300,
    "stable_frames": 2,
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
    "stable_frames": 2,
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
    "stable_frames": 2,
    "rep_cooldown_ms": 800,
    "min_phase_duration_ms": 100,
    "hand_placement_pass_foot_length_ratio": 1.25,
    "hand_placement_unsure_foot_length_ratio": 1.45,
    "forward_jump_min_com_displacement_leg_ratio": 0.20,
    "forward_jump_min_both_feet_displacement_leg_ratio": 0.15,
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
    "stable_frames": 2,
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
    "kneeling_min_violation_ms": 150,
    "seated_hip_drop_body_ratio_min": 0.18,
    "seated_knee_angle_max": 130.0,
    "seated_trunk_forward_max_deg": 30.0,
    "seated_hip_vertical_speed_body_ratio_max": 0.05,
    "seated_min_violation_ms": 250,
    "stable_frames": 2,
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

ACTION_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    "lunge": DEFAULT_LUNGE_CONFIG,
    "wall_ball": DEFAULT_WALL_BALL_CONFIG,
    "farmers_carry": DEFAULT_FARMERS_CARRY_CONFIG,
    "rowing": DEFAULT_ROWING_CONFIG,
    "skierg": DEFAULT_SKIERG_CONFIG,
    "burpee_broad_jump": DEFAULT_BURPEE_BROAD_JUMP_CONFIG,
    "sled_push": DEFAULT_SLED_PUSH_CONFIG,
    "sled_pull": DEFAULT_SLED_PULL_CONFIG,
}

_FEEDBACK_LIMIT_FIELDS = {"max_messages", "low_visibility_exclusive"}
_AUXILIARY_SCHEMAS: dict[str, dict[str, dict[str, type]]] = {
    "contact": {
        "knee_contact": {
            "surface_radius_shank_ratio": float,
            "enter_height_body_ratio": float,
            "exit_height_body_ratio": float,
            "max_vertical_speed_body_per_second": float,
            "min_landmark_confidence": float,
            "confirm_confidence": float,
            "min_hold_frames_high": int,
            "min_hold_frames_medium": int,
            "min_hold_frames_low": int,
        },
        "chest_contact": {
            "shoulder_weight": float,
            "hip_weight": float,
            "surface_offset_torso_ratio": float,
            "enter_height_body_ratio": float,
            "exit_height_body_ratio": float,
            "shoulder_height_body_ratio_max": float,
            "hip_height_body_ratio_max": float,
            "torso_to_floor_angle_deg_max": float,
            "min_hold_frames_high": int,
            "min_hold_frames_medium": int,
            "min_hold_frames_low": int,
        },
        "segmentation_contact": {
            "enabled": bool,
            "floor_band_body_ratio": float,
            "minimum_overlap_ratio": float,
        },
    },
    "foot_events": {
        "foot_support": {
            "grounded_height_body_ratio": float,
            "airborne_height_body_ratio": float,
            "min_vertical_speed_body_per_second": float,
            "min_landmark_confidence": float,
            "transition_frames_high": int,
            "transition_frames_medium": int,
            "transition_frames_low": int,
        },
        "foot_sync": {
            "pass_ms": int,
            "unsure_ms": int,
        },
        "foot_stagger": {
            "pass_foot_length_ratio": float,
            "unsure_foot_length_ratio": float,
        },
        "step_event": {
            "min_horizontal_displacement_leg_ratio": float,
            "min_airborne_ms": int,
            "min_grounded_ms": int,
        },
    },
}


def _validated_scalar(
    key: str,
    value: Any,
    expected: Any,
    *,
    path: str | Path | None,
) -> Any:
    if isinstance(expected, bool):
        if type(value) is not bool:
            raise ConfigValidationError(
                "expected boolean true/false",
                path=path,
                key=key,
            )
        return value
    if isinstance(expected, int):
        if type(value) is not int:
            raise ConfigValidationError(
                "expected integer",
                path=path,
                key=key,
            )
        resolved: Any = value
    elif isinstance(expected, float):
        if type(value) not in {int, float}:
            raise ConfigValidationError(
                "expected number",
                path=path,
                key=key,
            )
        resolved = float(value)
    elif isinstance(expected, str):
        if not isinstance(value, str) or not value.strip():
            raise ConfigValidationError(
                "expected non-empty text",
                path=path,
                key=key,
            )
        return value.strip()
    else:
        resolved = value

    if isinstance(resolved, (int, float)):
        if not isfinite(float(resolved)):
            raise ConfigValidationError(
                "number must be finite",
                path=path,
                key=key,
            )
        if (
            key.endswith("_ms")
            or "frames" in key
            or "ratio" in key
            or "confidence" in key
            or "visibility" in key
            or "margin" in key
            or "tolerance" in key
            or "delta" in key
        ) and resolved < 0:
            raise ConfigValidationError(
                "value must be non-negative",
                path=path,
                key=key,
            )
        if ("visibility" in key or "confidence" in key) and resolved > 1:
            raise ConfigValidationError(
                "confidence/visibility must be between 0 and 1",
                path=path,
                key=key,
            )
        if "frames" in key and resolved < 1:
            raise ConfigValidationError(
                "frame count must be at least 1",
                path=path,
                key=key,
            )
        if (
            "angle" in key
            or key.endswith("_deg")
            or "torso_lean" in key
            or "back_lean" in key
            or "over_lean" in key
        ) and not 0 <= resolved <= 180:
            raise ConfigValidationError(
                "angle must be between 0 and 180 degrees",
                path=path,
                key=key,
            )
    return resolved


def _validate_feedback_limits(
    value: Any,
    *,
    path: str | Path | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            "expected a nested mapping",
            path=path,
            key="feedback_limits",
        )
    reject_unknown_fields(
        value,
        _FEEDBACK_LIMIT_FIELDS,
        path=path,
        prefix="feedback_limits.",
    )
    validated: dict[str, Any] = {}
    if "max_messages" in value:
        maximum = _validated_scalar(
            "feedback_limits.max_messages",
            value["max_messages"],
            2,
            path=path,
        )
        if maximum > 10:
            raise ConfigValidationError(
                "must be between 1 and 10",
                path=path,
                key="feedback_limits.max_messages",
            )
        validated["max_messages"] = maximum
    if "low_visibility_exclusive" in value:
        validated["low_visibility_exclusive"] = _validated_scalar(
            "feedback_limits.low_visibility_exclusive",
            value["low_visibility_exclusive"],
            True,
            path=path,
        )
    return validated


def validate_action_config(
    action_name: str,
    values: Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        defaults = ACTION_CONFIG_DEFAULTS[action_name]
    except KeyError as exc:
        raise ConfigValidationError(
            f"unknown HYROX action {action_name!r}",
            path=path,
            key="action_name",
        ) from exc
    allowed = set(defaults) | {"feedback_limits"}
    reject_unknown_fields(values, allowed, path=path)
    validated: dict[str, Any] = {}
    for key, value in values.items():
        if key == "feedback_limits":
            validated[key] = _validate_feedback_limits(value, path=path)
        else:
            validated[key] = _validated_scalar(
                key,
                value,
                defaults[key],
                path=path,
            )
    configured_action = validated.get("action_name")
    if configured_action is not None and configured_action != action_name:
        raise ConfigValidationError(
            f"expected {action_name!r}, got {configured_action!r}",
            path=path,
            key="action_name",
        )
    merged = dict(defaults)
    merged.update(validated)
    _validate_relations(action_name, merged, path=path)
    return merged


def _validate_relations(
    action_name: str,
    values: Mapping[str, Any],
    *,
    path: str | Path | None,
) -> None:
    relation_groups = (
        (
            "full_extension_hold_frames_high",
            "full_extension_hold_frames_medium",
            "full_extension_hold_frames_low",
        ),
        ("wrist_peak_time_diff_ms_pass", "wrist_peak_time_diff_ms_unsure"),
        ("drive_torso_angle_min", "drive_torso_angle_max"),
        (
            "hand_placement_pass_foot_length_ratio",
            "hand_placement_unsure_foot_length_ratio",
        ),
    )
    for fields in relation_groups:
        if all(field in values for field in fields):
            numbers = [float(values[field]) for field in fields]
            if any(left > right for left, right in zip(numbers, numbers[1:])):
                raise ConfigValidationError(
                    f"expected {' <= '.join(fields)}",
                    path=path,
                    key=fields[-1],
                )
    if action_name == "lunge":
        enter = float(values["knee_contact_enter_height_body_ratio"])
        exit_value = float(values["knee_contact_exit_height_body_ratio"])
        if enter >= exit_value:
            raise ConfigValidationError(
                "contact enter threshold must be smaller than exit threshold",
                path=path,
                key="knee_contact_exit_height_body_ratio",
            )


def validate_observability_config(
    values: Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    reject_unknown_fields(values, set(DEFAULT_OBSERVABILITY_CONFIG), path=path)
    validated = {
        key: _validated_scalar(
            key,
            value,
            DEFAULT_OBSERVABILITY_CONFIG[key],
            path=path,
        )
        for key, value in values.items()
    }
    merged = dict(DEFAULT_OBSERVABILITY_CONFIG)
    merged.update(validated)
    return merged


def validate_auxiliary_config(
    name: str,
    path: str | Path,
) -> dict[str, Any]:
    try:
        schema = _AUXILIARY_SCHEMAS[name]
    except KeyError as exc:
        raise ConfigValidationError(
            f"unknown auxiliary configuration {name!r}",
            path=path,
        ) from exc
    parsed = load_simple_yaml(path)
    reject_unknown_fields(parsed, set(schema), path=path)
    for section, fields in schema.items():
        if section not in parsed:
            raise ConfigValidationError(
                "required section is missing",
                path=path,
                key=section,
            )
        section_values = parsed[section]
        if not isinstance(section_values, Mapping):
            raise ConfigValidationError(
                "expected a nested mapping",
                path=path,
                key=section,
            )
        reject_unknown_fields(
            section_values,
            set(fields),
            path=path,
            prefix=f"{section}.",
        )
        for key, expected_type in fields.items():
            if key not in section_values:
                raise ConfigValidationError(
                    "required field is missing",
                    path=path,
                    key=f"{section}.{key}",
                )
            expected = (
                False
                if expected_type is bool
                else 1
                if expected_type is int
                else 1.0
            )
            _validated_scalar(
                f"{section}.{key}",
                section_values[key],
                expected,
                path=path,
            )
    return parsed


def _load_flat_config(
    defaults: dict[str, Any],
    path: str | Path | None,
    default_path: str,
    *,
    action_name: str | None = None,
) -> dict[str, Any]:
    config_path = Path(path) if path is not None else resolve_asset(default_path)
    if not config_path.exists():
        return dict(defaults)

    parsed = load_simple_yaml(config_path)
    if "config_name" not in parsed:
        parsed["config_name"] = config_path.stem
    if action_name is None:
        return validate_observability_config(parsed, path=config_path)
    return validate_action_config(action_name, parsed, path=config_path)


def load_lunge_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_LUNGE_CONFIG,
        path,
        "configs/hyrox/lunge.yaml",
        action_name="lunge",
    )


def load_observability_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_OBSERVABILITY_CONFIG,
        path,
        "configs/hyrox/observability.yaml",
    )


def load_wall_ball_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_WALL_BALL_CONFIG,
        path,
        "configs/hyrox/wall_ball.yaml",
        action_name="wall_ball",
    )


def load_farmers_carry_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_FARMERS_CARRY_CONFIG,
        path,
        "configs/hyrox/farmers_carry.yaml",
        action_name="farmers_carry",
    )


def load_rowing_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_ROWING_CONFIG,
        path,
        "configs/hyrox/rowing.yaml",
        action_name="rowing",
    )


def load_skierg_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_SKIERG_CONFIG,
        path,
        "configs/hyrox/skierg.yaml",
        action_name="skierg",
    )


def load_burpee_broad_jump_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_BURPEE_BROAD_JUMP_CONFIG,
        path,
        "configs/hyrox/burpee_broad_jump.yaml",
        action_name="burpee_broad_jump",
    )


def load_sled_push_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_SLED_PUSH_CONFIG,
        path,
        "configs/hyrox/sled_push.yaml",
        action_name="sled_push",
    )


def load_sled_pull_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_flat_config(
        DEFAULT_SLED_PULL_CONFIG,
        path,
        "configs/hyrox/sled_pull.yaml",
        action_name="sled_pull",
    )


def resolve_hyrox_config_path(action: str, configured_path: str | None = None) -> str | None:
    if configured_path:
        return configured_path
    return DEFAULT_HYROX_CONFIG_PATHS.get(action)


__all__ = [
    "DEFAULT_OBSERVABILITY_CONFIG",
    "DEFAULT_LUNGE_CONFIG",
    "DEFAULT_WALL_BALL_CONFIG",
    "DEFAULT_FARMERS_CARRY_CONFIG",
    "DEFAULT_ROWING_CONFIG",
    "DEFAULT_SKIERG_CONFIG",
    "DEFAULT_BURPEE_BROAD_JUMP_CONFIG",
    "DEFAULT_SLED_PUSH_CONFIG",
    "DEFAULT_SLED_PULL_CONFIG",
    "DEFAULT_HYROX_CONFIG_PATHS",
    "ACTION_CONFIG_DEFAULTS",
    "ConfigValidationError",
    "load_observability_config",
    "load_lunge_config",
    "load_wall_ball_config",
    "load_farmers_carry_config",
    "load_rowing_config",
    "load_skierg_config",
    "load_burpee_broad_jump_config",
    "load_sled_push_config",
    "load_sled_pull_config",
    "resolve_hyrox_config_path",
    "validate_action_config",
    "validate_auxiliary_config",
    "validate_observability_config",
]
