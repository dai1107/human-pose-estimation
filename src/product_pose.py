"""Strict configuration for the formal product pose backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from src.backends.catalog import PRODUCT_BACKEND
from src.configuration import ConfigValidationError, load_simple_yaml, reject_unknown_fields
from src.paths import installation_root


DEFAULT_PRODUCT_POSE_CONFIG = installation_root() / "configs" / "product_pose.yaml"


@dataclass(frozen=True, slots=True)
class RealtimeLatencyConfig:
    latest_frame_only: bool = True
    camera_buffer_size: int = 1
    warning_pose_age_ms: float = 80.0
    max_pose_age_ms: float = 150.0
    max_frame_gap: int = 5
    hide_pose_after_ms: float = 300.0


@dataclass(frozen=True, slots=True)
class WebRealtimeConfig:
    max_requests_in_flight: int = 1
    inference_long_edge: int = 640
    jpeg_quality: float = 0.65


@dataclass(frozen=True, slots=True)
class OneEuroProfileConfig:
    min_cutoff: float
    beta: float
    d_cutoff: float


@dataclass(frozen=True, slots=True)
class RealtimeSmoothingConfig:
    mode: str = "adaptive_one_euro"
    profile: str = "responsive"
    max_gap_ms_before_reset: float = 250.0
    stable: OneEuroProfileConfig = field(
        default_factory=lambda: OneEuroProfileConfig(0.8, 0.025, 1.0)
    )
    balanced: OneEuroProfileConfig = field(
        default_factory=lambda: OneEuroProfileConfig(1.2, 0.05, 1.0)
    )
    responsive: OneEuroProfileConfig = field(
        default_factory=lambda: OneEuroProfileConfig(1.7, 0.08, 1.0)
    )
    fast_joint_min_cutoff_scale: float = 1.25
    fast_joint_beta_scale: float = 1.50
    stable_joint_min_cutoff_scale: float = 0.75
    stable_joint_beta_scale: float = 0.50
    world_min_cutoff_scale: float = 0.90
    world_beta_scale: float = 0.85

    @property
    def profiles(self) -> Mapping[str, OneEuroProfileConfig]:
        return MappingProxyType(
            {
                "stable": self.stable,
                "balanced": self.balanced,
                "responsive": self.responsive,
            }
        )


@dataclass(frozen=True, slots=True)
class ThreeDKinematicsConfig:
    enabled: bool = True
    decision_mode: str = "shadow"
    assist_confidence_boost: float = 0.05
    assist_conflict_confidence_cap: float = 0.49


@dataclass(frozen=True, slots=True)
class ThreeDQualityConfig:
    min_visibility: float = 0.70
    min_presence: float = 0.70
    max_bone_length_change_ratio: float = 0.20
    max_angle_delta_deg: float = 35.0
    max_angular_velocity_deg_s: float = 720.0
    max_2d_3d_difference_deg: float = 25.0
    max_z_change_body_scale: float = 0.35
    identity_swap_cost_ratio: float = 0.75
    max_gap_ms_before_reset: float = 250.0


@dataclass(frozen=True, slots=True)
class ProductPoseConfig:
    backend: str = PRODUCT_BACKEND
    allow_experimental_backends: bool = False
    realtime_latency: RealtimeLatencyConfig = field(default_factory=RealtimeLatencyConfig)
    realtime_smoothing: RealtimeSmoothingConfig = field(default_factory=RealtimeSmoothingConfig)
    three_d_kinematics: ThreeDKinematicsConfig = field(default_factory=ThreeDKinematicsConfig)
    three_d_quality: ThreeDQualityConfig = field(default_factory=ThreeDQualityConfig)
    web_realtime: WebRealtimeConfig = field(default_factory=WebRealtimeConfig)


def load_product_pose_config(
    path: str | Path = DEFAULT_PRODUCT_POSE_CONFIG,
) -> ProductPoseConfig:
    config_path = Path(path)
    values = load_simple_yaml(config_path)
    reject_unknown_fields(
        values,
        {
            "product_pose",
            "realtime_latency",
            "realtime_smoothing",
            "three_d_kinematics",
            "three_d_quality",
            "web_realtime",
        },
        path=config_path,
    )
    section = values.get("product_pose")
    if not isinstance(section, dict):
        raise ConfigValidationError(
            "product_pose must be a mapping",
            path=config_path,
            key="product_pose",
        )
    reject_unknown_fields(
        section,
        {"backend", "allow_experimental_backends"},
        path=config_path,
        prefix="product_pose.",
    )
    backend = str(section.get("backend", "")).strip().lower()
    if backend != PRODUCT_BACKEND:
        raise ConfigValidationError(
            f"formal product backend must be {PRODUCT_BACKEND!r}",
            path=config_path,
            key="product_pose.backend",
        )
    allow_experimental = section.get("allow_experimental_backends", False)
    if not isinstance(allow_experimental, bool):
        raise ConfigValidationError(
            "must be true or false",
            path=config_path,
            key="product_pose.allow_experimental_backends",
        )
    realtime_latency = _load_realtime_latency(values.get("realtime_latency"), path=config_path)
    realtime_smoothing = _load_realtime_smoothing(
        values.get("realtime_smoothing"),
        path=config_path,
    )
    three_d_kinematics = _load_three_d_kinematics(
        values.get("three_d_kinematics"),
        path=config_path,
    )
    three_d_quality = _load_three_d_quality(
        values.get("three_d_quality"),
        path=config_path,
    )
    web_realtime = _load_web_realtime(values.get("web_realtime"), path=config_path)
    return ProductPoseConfig(
        backend=backend,
        allow_experimental_backends=allow_experimental,
        realtime_latency=realtime_latency,
        realtime_smoothing=realtime_smoothing,
        three_d_kinematics=three_d_kinematics,
        three_d_quality=three_d_quality,
        web_realtime=web_realtime,
    )


def _load_realtime_latency(value: object, *, path: Path) -> RealtimeLatencyConfig:
    if value is None:
        return RealtimeLatencyConfig()
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "realtime_latency must be a mapping",
            path=path,
            key="realtime_latency",
        )
    allowed = {
        "latest_frame_only",
        "camera_buffer_size",
        "warning_pose_age_ms",
        "max_pose_age_ms",
        "max_frame_gap",
        "hide_pose_after_ms",
    }
    reject_unknown_fields(value, allowed, path=path, prefix="realtime_latency.")

    latest_frame_only = value.get("latest_frame_only", True)
    if not isinstance(latest_frame_only, bool):
        raise ConfigValidationError("must be true or false", path=path, key="realtime_latency.latest_frame_only")
    camera_buffer_size = value.get("camera_buffer_size", 1)
    try:
        parsed_camera_buffer_size = int(camera_buffer_size)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError("must be exactly 1", path=path, key="realtime_latency.camera_buffer_size") from exc
    if isinstance(camera_buffer_size, bool) or parsed_camera_buffer_size != 1:
        raise ConfigValidationError("must be exactly 1", path=path, key="realtime_latency.camera_buffer_size")

    def positive_number(name: str, default: float) -> float:
        raw = value.get(name, default)
        if isinstance(raw, bool):
            raise ConfigValidationError("must be a positive number", path=path, key=f"realtime_latency.{name}")
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError("must be a positive number", path=path, key=f"realtime_latency.{name}") from exc
        if parsed <= 0:
            raise ConfigValidationError("must be a positive number", path=path, key=f"realtime_latency.{name}")
        return parsed

    max_frame_gap = value.get("max_frame_gap", 5)
    if isinstance(max_frame_gap, bool):
        raise ConfigValidationError("must be a non-negative integer", path=path, key="realtime_latency.max_frame_gap")
    try:
        max_frame_gap = int(max_frame_gap)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError("must be a non-negative integer", path=path, key="realtime_latency.max_frame_gap") from exc
    if max_frame_gap < 0:
        raise ConfigValidationError("must be a non-negative integer", path=path, key="realtime_latency.max_frame_gap")

    warning_pose_age_ms = positive_number("warning_pose_age_ms", 80.0)
    max_pose_age_ms = positive_number("max_pose_age_ms", 150.0)
    hide_pose_after_ms = positive_number("hide_pose_after_ms", 300.0)
    if warning_pose_age_ms > max_pose_age_ms:
        raise ConfigValidationError(
            "must be <= realtime_latency.max_pose_age_ms",
            path=path,
            key="realtime_latency.warning_pose_age_ms",
        )
    if hide_pose_after_ms < max_pose_age_ms:
        raise ConfigValidationError(
            "must be >= realtime_latency.max_pose_age_ms",
            path=path,
            key="realtime_latency.hide_pose_after_ms",
        )
    return RealtimeLatencyConfig(
        latest_frame_only=latest_frame_only,
        camera_buffer_size=1,
        warning_pose_age_ms=warning_pose_age_ms,
        max_pose_age_ms=max_pose_age_ms,
        max_frame_gap=max_frame_gap,
        hide_pose_after_ms=hide_pose_after_ms,
    )


def _load_web_realtime(value: object, *, path: Path) -> WebRealtimeConfig:
    if value is None:
        return WebRealtimeConfig()
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "web_realtime must be a mapping",
            path=path,
            key="web_realtime",
        )
    reject_unknown_fields(
        value,
        {"max_requests_in_flight", "inference_long_edge", "jpeg_quality"},
        path=path,
        prefix="web_realtime.",
    )
    max_requests = value.get("max_requests_in_flight", 1)
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests != 1:
        raise ConfigValidationError(
            "must be exactly 1",
            path=path,
            key="web_realtime.max_requests_in_flight",
        )
    inference_long_edge = value.get("inference_long_edge", 640)
    if (
        isinstance(inference_long_edge, bool)
        or not isinstance(inference_long_edge, int)
        or not 128 <= inference_long_edge <= 1280
    ):
        raise ConfigValidationError(
            "must be an integer from 128 to 1280",
            path=path,
            key="web_realtime.inference_long_edge",
        )
    jpeg_quality = value.get("jpeg_quality", 0.65)
    if isinstance(jpeg_quality, bool):
        raise ConfigValidationError(
            "must be a number greater than 0 and at most 1",
            path=path,
            key="web_realtime.jpeg_quality",
        )
    try:
        jpeg_quality = float(jpeg_quality)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            "must be a number greater than 0 and at most 1",
            path=path,
            key="web_realtime.jpeg_quality",
        ) from exc
    if not 0.0 < jpeg_quality <= 1.0:
        raise ConfigValidationError(
            "must be a number greater than 0 and at most 1",
            path=path,
            key="web_realtime.jpeg_quality",
        )
    return WebRealtimeConfig(
        max_requests_in_flight=1,
        inference_long_edge=inference_long_edge,
        jpeg_quality=jpeg_quality,
    )


def _load_realtime_smoothing(value: object, *, path: Path) -> RealtimeSmoothingConfig:
    defaults = RealtimeSmoothingConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "realtime_smoothing must be a mapping",
            path=path,
            key="realtime_smoothing",
        )
    profile_fields = {
        f"{profile}_{field_name}"
        for profile in ("stable", "balanced", "responsive")
        for field_name in ("min_cutoff", "beta", "d_cutoff")
    }
    scale_fields = {
        "fast_joint_min_cutoff_scale",
        "fast_joint_beta_scale",
        "stable_joint_min_cutoff_scale",
        "stable_joint_beta_scale",
        "world_min_cutoff_scale",
        "world_beta_scale",
    }
    reject_unknown_fields(
        value,
        {
            "mode",
            "profile",
            "max_gap_ms_before_reset",
            *profile_fields,
            *scale_fields,
        },
        path=path,
        prefix="realtime_smoothing.",
    )
    mode = str(value.get("mode", defaults.mode)).strip().lower()
    if mode != "adaptive_one_euro":
        raise ConfigValidationError(
            "must be 'adaptive_one_euro'",
            path=path,
            key="realtime_smoothing.mode",
        )
    profile = str(value.get("profile", defaults.profile)).strip().lower()
    if profile not in {"stable", "balanced", "responsive"}:
        raise ConfigValidationError(
            "must be stable, balanced, or responsive",
            path=path,
            key="realtime_smoothing.profile",
        )

    max_gap_ms = _config_number(
        value,
        "max_gap_ms_before_reset",
        defaults.max_gap_ms_before_reset,
        path=path,
        positive=True,
    )

    def load_profile(name: str, fallback: OneEuroProfileConfig) -> OneEuroProfileConfig:
        return OneEuroProfileConfig(
            min_cutoff=_config_number(
                value,
                f"{name}_min_cutoff",
                fallback.min_cutoff,
                path=path,
                positive=True,
            ),
            beta=_config_number(
                value,
                f"{name}_beta",
                fallback.beta,
                path=path,
                non_negative=True,
            ),
            d_cutoff=_config_number(
                value,
                f"{name}_d_cutoff",
                fallback.d_cutoff,
                path=path,
                positive=True,
            ),
        )

    scales = {
        name: _config_number(
            value,
            name,
            getattr(defaults, name),
            path=path,
            positive=True,
        )
        for name in scale_fields
    }
    return RealtimeSmoothingConfig(
        mode=mode,
        profile=profile,
        max_gap_ms_before_reset=max_gap_ms,
        stable=load_profile("stable", defaults.stable),
        balanced=load_profile("balanced", defaults.balanced),
        responsive=load_profile("responsive", defaults.responsive),
        **scales,
    )


def _load_three_d_kinematics(value: object, *, path: Path) -> ThreeDKinematicsConfig:
    defaults = ThreeDKinematicsConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "three_d_kinematics must be a mapping",
            path=path,
            key="three_d_kinematics",
        )
    reject_unknown_fields(
        value,
        {
            "enabled",
            "decision_mode",
            "assist_confidence_boost",
            "assist_conflict_confidence_cap",
        },
        path=path,
        prefix="three_d_kinematics.",
    )
    enabled = value.get("enabled", defaults.enabled)
    if not isinstance(enabled, bool):
        raise ConfigValidationError(
            "must be true or false",
            path=path,
            key="three_d_kinematics.enabled",
        )
    decision_mode = str(value.get("decision_mode", defaults.decision_mode)).strip().lower()
    if decision_mode not in {"shadow", "assist"}:
        raise ConfigValidationError(
            "must be 'shadow' or 'assist'; rule-specific mode is not enabled",
            path=path,
            key="three_d_kinematics.decision_mode",
        )
    confidence_values = {
        name: _config_number(
            value,
            name,
            getattr(defaults, name),
            path=path,
            non_negative=True,
            prefix="three_d_kinematics",
        )
        for name in (
            "assist_confidence_boost",
            "assist_conflict_confidence_cap",
        )
    }
    for name, parsed in confidence_values.items():
        if parsed > 1.0:
            raise ConfigValidationError(
                "must be between 0 and 1",
                path=path,
                key=f"three_d_kinematics.{name}",
            )
    return ThreeDKinematicsConfig(
        enabled=enabled,
        decision_mode=decision_mode,
        **confidence_values,
    )


def _load_three_d_quality(value: object, *, path: Path) -> ThreeDQualityConfig:
    defaults = ThreeDQualityConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "three_d_quality must be a mapping",
            path=path,
            key="three_d_quality",
        )
    fields = {
        "min_visibility",
        "min_presence",
        "max_bone_length_change_ratio",
        "max_angle_delta_deg",
        "max_angular_velocity_deg_s",
        "max_2d_3d_difference_deg",
        "max_z_change_body_scale",
        "identity_swap_cost_ratio",
        "max_gap_ms_before_reset",
    }
    reject_unknown_fields(value, fields, path=path, prefix="three_d_quality.")
    parsed = {
        name: _config_number(
            value,
            name,
            getattr(defaults, name),
            path=path,
            positive=True,
            prefix="three_d_quality",
        )
        for name in fields
    }
    for name in ("min_visibility", "min_presence", "identity_swap_cost_ratio"):
        if parsed[name] > 1.0:
            raise ConfigValidationError(
                "must be greater than 0 and at most 1",
                path=path,
                key=f"three_d_quality.{name}",
            )
    return ThreeDQualityConfig(**parsed)


def _config_number(
    values: dict[str, object],
    name: str,
    default: float,
    *,
    path: Path,
    positive: bool = False,
    non_negative: bool = False,
    prefix: str = "realtime_smoothing",
) -> float:
    raw = values.get(name, default)
    if isinstance(raw, bool):
        parsed = float("nan")
    else:
        try:
            parsed = float(raw)
        except (TypeError, ValueError, OverflowError):
            parsed = float("nan")
    valid = isfinite(parsed)
    if positive:
        valid = valid and parsed > 0
    if non_negative:
        valid = valid and parsed >= 0
    if not valid:
        qualifier = "a positive finite number" if positive else "a non-negative finite number"
        raise ConfigValidationError(
            f"must be {qualifier}",
            path=path,
            key=f"{prefix}.{name}",
        )
    return parsed


__all__ = [
    "DEFAULT_PRODUCT_POSE_CONFIG",
    "OneEuroProfileConfig",
    "ProductPoseConfig",
    "RealtimeLatencyConfig",
    "RealtimeSmoothingConfig",
    "ThreeDKinematicsConfig",
    "ThreeDQualityConfig",
    "WebRealtimeConfig",
    "load_product_pose_config",
]
