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
class RenderingConfig:
    angle_text_fps: float = 12.0
    metrics_fps: float = 5.0
    stats_fps: float = 3.0
    timing_sample_capacity: int = 240


@dataclass(frozen=True, slots=True)
class CameraConfig:
    preferred_width: int = 640
    preferred_height: int = 480
    preferred_fps: float = 60.0
    fallback_fps: float = 30.0
    diagnostic_sample_fps: float = 5.0
    low_light_luma: float = 55.0
    fps_warning_ratio: float = 0.80
    interval_anomaly_ratio: float = 1.80
    duplicate_warning_ratio: float = 0.20


@dataclass(frozen=True, slots=True)
class LocalFirstArchitectureConfig:
    web_pipeline: str = "local_browser"
    desktop_pipeline: str = "local_device"
    server_pose_fallback: bool = True
    neural_prediction_enabled: bool = False


@dataclass(frozen=True, slots=True)
class OneEuroProfileConfig:
    min_cutoff: float
    beta: float
    d_cutoff: float


@dataclass(frozen=True, slots=True)
class RealtimeSmoothingConfig:
    mode: str = "adaptive_one_euro"
    profile: str = "responsive"
    prediction_enabled: bool = False
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
class DisplaySmoothingConfig:
    mode: str = "adaptive_one_euro"
    profile: str = "ultra_responsive"
    prediction_enabled: bool = True
    max_gap_ms_before_reset: float = 250.0
    min_cutoff: float = 2.2
    beta: float = 0.12
    d_cutoff: float = 1.0
    raw_blend_enabled: bool = True
    max_raw_weight: float = 0.45
    minimum_visibility: float = 0.70
    slow_speed: float = 0.15
    fast_speed: float = 1.20
    extremity_raw_weight_scale: float = 1.0
    core_raw_weight_scale: float = 0.35
    face_raw_weight_scale: float = 0.0
    world_speed_scale: float = 1.25


@dataclass(frozen=True, slots=True)
class DisplayPredictionConfig:
    enabled: bool = True
    mode: str = "constant_velocity"
    max_horizon_ms: float = 45.0
    maximum_body_scale_displacement: float = 0.06
    minimum_visibility: float = 0.70
    velocity_decay: float = 0.85
    disable_after_gap_ms: float = 100.0
    reversal_strength: float = 0.25
    core_prediction_scale: float = 0.45
    face_prediction_scale: float = 0.0
    support_foot_horizontal_scale: float = 0.0


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
    realtime_model: str = "auto"
    analysis_model: str = "full"
    realtime_latency: RealtimeLatencyConfig = field(default_factory=RealtimeLatencyConfig)
    analysis_smoothing: RealtimeSmoothingConfig = field(default_factory=RealtimeSmoothingConfig)
    display_smoothing: DisplaySmoothingConfig = field(default_factory=DisplaySmoothingConfig)
    display_prediction: DisplayPredictionConfig = field(default_factory=DisplayPredictionConfig)
    three_d_kinematics: ThreeDKinematicsConfig = field(default_factory=ThreeDKinematicsConfig)
    three_d_quality: ThreeDQualityConfig = field(default_factory=ThreeDQualityConfig)
    web_realtime: WebRealtimeConfig = field(default_factory=WebRealtimeConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    local_first: LocalFirstArchitectureConfig = field(
        default_factory=LocalFirstArchitectureConfig
    )

    @property
    def realtime_smoothing(self) -> RealtimeSmoothingConfig:
        """Backward-compatible alias for the authoritative analysis stream."""

        return self.analysis_smoothing


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
            "analysis_smoothing",
            "display_smoothing",
            "display_prediction",
            "realtime_smoothing",
            "three_d_kinematics",
            "three_d_quality",
            "web_realtime",
            "rendering",
            "camera",
            "local_first",
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
        {"backend", "allow_experimental_backends", "realtime_model", "analysis_model"},
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
    realtime_model = str(section.get("realtime_model", "auto")).strip().lower()
    if realtime_model not in {"auto", "lite", "full"}:
        raise ConfigValidationError(
            "must be auto, lite, or full",
            path=config_path,
            key="product_pose.realtime_model",
        )
    analysis_model = str(section.get("analysis_model", "full")).strip().lower()
    if analysis_model != "full":
        raise ConfigValidationError(
            "formal analysis model must be 'full'",
            path=config_path,
            key="product_pose.analysis_model",
        )
    realtime_latency = _load_realtime_latency(values.get("realtime_latency"), path=config_path)
    if values.get("analysis_smoothing") is not None and values.get("realtime_smoothing") is not None:
        raise ConfigValidationError(
            "use analysis_smoothing; do not also define legacy realtime_smoothing",
            path=config_path,
            key="analysis_smoothing",
        )
    analysis_section_name = (
        "analysis_smoothing"
        if values.get("analysis_smoothing") is not None
        else "realtime_smoothing"
    )
    analysis_smoothing = _load_realtime_smoothing(
        values.get(analysis_section_name),
        path=config_path,
        section_name=analysis_section_name,
    )
    display_smoothing = _load_display_smoothing(
        values.get("display_smoothing"),
        path=config_path,
    )
    display_prediction = _load_display_prediction(
        values.get("display_prediction"),
        path=config_path,
    )
    if display_smoothing.prediction_enabled != display_prediction.enabled:
        raise ConfigValidationError(
            "must match display_prediction.enabled",
            path=config_path,
            key="display_smoothing.prediction_enabled",
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
    rendering = _load_rendering(values.get("rendering"), path=config_path)
    camera = _load_camera(values.get("camera"), path=config_path)
    local_first = _load_local_first(values.get("local_first"), path=config_path)
    return ProductPoseConfig(
        backend=backend,
        allow_experimental_backends=allow_experimental,
        realtime_model=realtime_model,
        analysis_model=analysis_model,
        realtime_latency=realtime_latency,
        analysis_smoothing=analysis_smoothing,
        display_smoothing=display_smoothing,
        display_prediction=display_prediction,
        three_d_kinematics=three_d_kinematics,
        three_d_quality=three_d_quality,
        web_realtime=web_realtime,
        rendering=rendering,
        camera=camera,
        local_first=local_first,
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


def _load_rendering(value: object, *, path: Path) -> RenderingConfig:
    defaults = RenderingConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "rendering must be a mapping",
            path=path,
            key="rendering",
        )
    fields = {
        "angle_text_fps",
        "metrics_fps",
        "stats_fps",
        "timing_sample_capacity",
    }
    reject_unknown_fields(value, fields, path=path, prefix="rendering.")
    parsed = {
        name: _config_number(
            value,
            name,
            getattr(defaults, name),
            path=path,
            positive=True,
            prefix="rendering",
        )
        for name in ("angle_text_fps", "metrics_fps", "stats_fps")
    }
    for name, maximum in (
        ("angle_text_fps", 30.0),
        ("metrics_fps", 10.0),
        ("stats_fps", 10.0),
    ):
        if parsed[name] > maximum:
            raise ConfigValidationError(
                f"must be no greater than {maximum:g}",
                path=path,
                key=f"rendering.{name}",
            )
    capacity = value.get("timing_sample_capacity", defaults.timing_sample_capacity)
    if isinstance(capacity, bool) or not isinstance(capacity, int) or not 30 <= capacity <= 2000:
        raise ConfigValidationError(
            "must be an integer between 30 and 2000",
            path=path,
            key="rendering.timing_sample_capacity",
        )
    return RenderingConfig(timing_sample_capacity=capacity, **parsed)


def _load_camera(value: object, *, path: Path) -> CameraConfig:
    defaults = CameraConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError("camera must be a mapping", path=path, key="camera")
    fields = {
        "preferred_width",
        "preferred_height",
        "preferred_fps",
        "fallback_fps",
        "diagnostic_sample_fps",
        "low_light_luma",
        "fps_warning_ratio",
        "interval_anomaly_ratio",
        "duplicate_warning_ratio",
    }
    reject_unknown_fields(value, fields, path=path, prefix="camera.")
    dimensions: dict[str, int] = {}
    for name in ("preferred_width", "preferred_height"):
        raw = value.get(name, getattr(defaults, name))
        if isinstance(raw, bool) or not isinstance(raw, int) or not 160 <= raw <= 3840:
            raise ConfigValidationError(
                "must be an integer between 160 and 3840",
                path=path,
                key=f"camera.{name}",
            )
        dimensions[name] = raw
    parsed = {
        name: _config_number(
            value,
            name,
            getattr(defaults, name),
            path=path,
            positive=True,
            prefix="camera",
        )
        for name in (
            "preferred_fps",
            "fallback_fps",
            "diagnostic_sample_fps",
            "low_light_luma",
            "fps_warning_ratio",
            "interval_anomaly_ratio",
            "duplicate_warning_ratio",
        )
    }
    if parsed["fallback_fps"] > parsed["preferred_fps"]:
        raise ConfigValidationError(
            "must be no greater than camera.preferred_fps",
            path=path,
            key="camera.fallback_fps",
        )
    for name in ("fps_warning_ratio", "duplicate_warning_ratio"):
        if parsed[name] > 1.0:
            raise ConfigValidationError(
                "must be greater than 0 and at most 1",
                path=path,
                key=f"camera.{name}",
            )
    if parsed["low_light_luma"] > 255:
        raise ConfigValidationError(
            "must be no greater than 255",
            path=path,
            key="camera.low_light_luma",
        )
    return CameraConfig(**dimensions, **parsed)


def _load_local_first(value: object, *, path: Path) -> LocalFirstArchitectureConfig:
    defaults = LocalFirstArchitectureConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "local_first must be a mapping",
            path=path,
            key="local_first",
        )
    fields = {
        "web_pipeline",
        "desktop_pipeline",
        "server_pose_fallback",
        "neural_prediction_enabled",
    }
    reject_unknown_fields(value, fields, path=path, prefix="local_first.")
    web_pipeline = str(value.get("web_pipeline", defaults.web_pipeline)).strip().lower()
    desktop_pipeline = str(value.get("desktop_pipeline", defaults.desktop_pipeline)).strip().lower()
    if web_pipeline != "local_browser":
        raise ConfigValidationError(
            "must be 'local_browser'",
            path=path,
            key="local_first.web_pipeline",
        )
    if desktop_pipeline != "local_device":
        raise ConfigValidationError(
            "must be 'local_device'",
            path=path,
            key="local_first.desktop_pipeline",
        )
    server_pose_fallback = value.get("server_pose_fallback", defaults.server_pose_fallback)
    neural_prediction_enabled = value.get(
        "neural_prediction_enabled",
        defaults.neural_prediction_enabled,
    )
    if not isinstance(server_pose_fallback, bool):
        raise ConfigValidationError(
            "must be true or false",
            path=path,
            key="local_first.server_pose_fallback",
        )
    if neural_prediction_enabled is not False:
        raise ConfigValidationError(
            "neural prediction is outside the product architecture",
            path=path,
            key="local_first.neural_prediction_enabled",
        )
    return LocalFirstArchitectureConfig(
        web_pipeline=web_pipeline,
        desktop_pipeline=desktop_pipeline,
        server_pose_fallback=server_pose_fallback,
        neural_prediction_enabled=False,
    )


def _load_realtime_smoothing(
    value: object,
    *,
    path: Path,
    section_name: str = "analysis_smoothing",
) -> RealtimeSmoothingConfig:
    defaults = RealtimeSmoothingConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError(
            f"{section_name} must be a mapping",
            path=path,
            key=section_name,
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
            "prediction_enabled",
            "max_gap_ms_before_reset",
            *profile_fields,
            *scale_fields,
        },
        path=path,
        prefix=f"{section_name}.",
    )
    mode = str(value.get("mode", defaults.mode)).strip().lower()
    if mode != "adaptive_one_euro":
        raise ConfigValidationError(
            "must be 'adaptive_one_euro'",
            path=path,
            key=f"{section_name}.mode",
        )
    profile = str(value.get("profile", defaults.profile)).strip().lower()
    if profile not in {"stable", "balanced", "responsive"}:
        raise ConfigValidationError(
            "must be stable, balanced, or responsive",
            path=path,
            key=f"{section_name}.profile",
        )
    prediction_enabled = value.get("prediction_enabled", False)
    if prediction_enabled is not False:
        raise ConfigValidationError(
            "authoritative analysis prediction must be false",
            path=path,
            key=f"{section_name}.prediction_enabled",
        )

    max_gap_ms = _config_number(
        value,
        "max_gap_ms_before_reset",
        defaults.max_gap_ms_before_reset,
        path=path,
        positive=True,
        prefix=section_name,
    )

    def load_profile(name: str, fallback: OneEuroProfileConfig) -> OneEuroProfileConfig:
        return OneEuroProfileConfig(
            min_cutoff=_config_number(
                value,
                f"{name}_min_cutoff",
                fallback.min_cutoff,
                path=path,
                positive=True,
                prefix=section_name,
            ),
            beta=_config_number(
                value,
                f"{name}_beta",
                fallback.beta,
                path=path,
                non_negative=True,
                prefix=section_name,
            ),
            d_cutoff=_config_number(
                value,
                f"{name}_d_cutoff",
                fallback.d_cutoff,
                path=path,
                positive=True,
                prefix=section_name,
            ),
        )

    scales = {
        name: _config_number(
            value,
            name,
            getattr(defaults, name),
            path=path,
            positive=True,
            prefix=section_name,
        )
        for name in scale_fields
    }
    return RealtimeSmoothingConfig(
        mode=mode,
        profile=profile,
        prediction_enabled=False,
        max_gap_ms_before_reset=max_gap_ms,
        stable=load_profile("stable", defaults.stable),
        balanced=load_profile("balanced", defaults.balanced),
        responsive=load_profile("responsive", defaults.responsive),
        **scales,
    )


def _load_display_smoothing(value: object, *, path: Path) -> DisplaySmoothingConfig:
    defaults = DisplaySmoothingConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "display_smoothing must be a mapping",
            path=path,
            key="display_smoothing",
        )
    fields = {
        "mode",
        "profile",
        "prediction_enabled",
        "max_gap_ms_before_reset",
        "min_cutoff",
        "beta",
        "d_cutoff",
        "raw_blend_enabled",
        "max_raw_weight",
        "minimum_visibility",
        "slow_speed",
        "fast_speed",
        "extremity_raw_weight_scale",
        "core_raw_weight_scale",
        "face_raw_weight_scale",
        "world_speed_scale",
    }
    reject_unknown_fields(value, fields, path=path, prefix="display_smoothing.")
    mode = str(value.get("mode", defaults.mode)).strip().lower()
    if mode != "adaptive_one_euro":
        raise ConfigValidationError(
            "must be 'adaptive_one_euro'",
            path=path,
            key="display_smoothing.mode",
        )
    profile = str(value.get("profile", defaults.profile)).strip().lower()
    if profile != "ultra_responsive":
        raise ConfigValidationError(
            "must be 'ultra_responsive'",
            path=path,
            key="display_smoothing.profile",
        )
    prediction_enabled = value.get("prediction_enabled", defaults.prediction_enabled)
    if not isinstance(prediction_enabled, bool):
        raise ConfigValidationError(
            "must be true or false",
            path=path,
            key="display_smoothing.prediction_enabled",
        )
    raw_blend_enabled = value.get("raw_blend_enabled", defaults.raw_blend_enabled)
    if not isinstance(raw_blend_enabled, bool):
        raise ConfigValidationError(
            "must be true or false",
            path=path,
            key="display_smoothing.raw_blend_enabled",
        )
    numeric_fields = {
        "max_gap_ms_before_reset": (defaults.max_gap_ms_before_reset, True, False),
        "min_cutoff": (defaults.min_cutoff, True, False),
        "beta": (defaults.beta, False, True),
        "d_cutoff": (defaults.d_cutoff, True, False),
        "max_raw_weight": (defaults.max_raw_weight, False, True),
        "minimum_visibility": (defaults.minimum_visibility, False, True),
        "slow_speed": (defaults.slow_speed, False, True),
        "fast_speed": (defaults.fast_speed, True, False),
        "extremity_raw_weight_scale": (defaults.extremity_raw_weight_scale, False, True),
        "core_raw_weight_scale": (defaults.core_raw_weight_scale, False, True),
        "face_raw_weight_scale": (defaults.face_raw_weight_scale, False, True),
        "world_speed_scale": (defaults.world_speed_scale, True, False),
    }
    parsed = {
        name: _config_number(
            value,
            name,
            default,
            path=path,
            positive=positive,
            non_negative=non_negative,
            prefix="display_smoothing",
        )
        for name, (default, positive, non_negative) in numeric_fields.items()
    }
    for name in (
        "minimum_visibility",
        "extremity_raw_weight_scale",
        "core_raw_weight_scale",
        "face_raw_weight_scale",
    ):
        if parsed[name] > 1.0:
            raise ConfigValidationError(
                "must be between 0 and 1",
                path=path,
                key=f"display_smoothing.{name}",
            )
    if parsed["max_raw_weight"] > 0.45:
        raise ConfigValidationError(
            "must be between 0 and 0.45",
            path=path,
            key="display_smoothing.max_raw_weight",
        )
    if parsed["fast_speed"] <= parsed["slow_speed"]:
        raise ConfigValidationError(
            "must be greater than display_smoothing.slow_speed",
            path=path,
            key="display_smoothing.fast_speed",
        )
    return DisplaySmoothingConfig(
        mode=mode,
        profile=profile,
        prediction_enabled=prediction_enabled,
        raw_blend_enabled=raw_blend_enabled,
        **parsed,
    )


def _load_display_prediction(value: object, *, path: Path) -> DisplayPredictionConfig:
    defaults = DisplayPredictionConfig()
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ConfigValidationError(
            "display_prediction must be a mapping",
            path=path,
            key="display_prediction",
        )
    fields = {
        "enabled",
        "mode",
        "max_horizon_ms",
        "maximum_body_scale_displacement",
        "minimum_visibility",
        "velocity_decay",
        "disable_after_gap_ms",
        "reversal_strength",
        "core_prediction_scale",
        "face_prediction_scale",
        "support_foot_horizontal_scale",
    }
    reject_unknown_fields(value, fields, path=path, prefix="display_prediction.")
    enabled = value.get("enabled", defaults.enabled)
    if not isinstance(enabled, bool):
        raise ConfigValidationError(
            "must be true or false",
            path=path,
            key="display_prediction.enabled",
        )
    mode = str(value.get("mode", defaults.mode)).strip().lower()
    if mode != "constant_velocity":
        raise ConfigValidationError(
            "must be 'constant_velocity'",
            path=path,
            key="display_prediction.mode",
        )
    numeric_fields = {
        "max_horizon_ms": (defaults.max_horizon_ms, True, False),
        "maximum_body_scale_displacement": (
            defaults.maximum_body_scale_displacement,
            True,
            False,
        ),
        "minimum_visibility": (defaults.minimum_visibility, False, True),
        "velocity_decay": (defaults.velocity_decay, False, True),
        "disable_after_gap_ms": (defaults.disable_after_gap_ms, True, False),
        "reversal_strength": (defaults.reversal_strength, False, True),
        "core_prediction_scale": (defaults.core_prediction_scale, False, True),
        "face_prediction_scale": (defaults.face_prediction_scale, False, True),
        "support_foot_horizontal_scale": (
            defaults.support_foot_horizontal_scale,
            False,
            True,
        ),
    }
    parsed = {
        name: _config_number(
            value,
            name,
            default,
            path=path,
            positive=positive,
            non_negative=non_negative,
            prefix="display_prediction",
        )
        for name, (default, positive, non_negative) in numeric_fields.items()
    }
    for name in (
        "minimum_visibility",
        "velocity_decay",
        "reversal_strength",
        "core_prediction_scale",
        "face_prediction_scale",
        "support_foot_horizontal_scale",
    ):
        if parsed[name] > 1.0:
            raise ConfigValidationError(
                "must be between 0 and 1",
                path=path,
                key=f"display_prediction.{name}",
            )
    if parsed["max_horizon_ms"] > 60.0:
        raise ConfigValidationError(
            "must be no greater than 60",
            path=path,
            key="display_prediction.max_horizon_ms",
        )
    if parsed["maximum_body_scale_displacement"] > 0.20:
        raise ConfigValidationError(
            "must be no greater than 0.20",
            path=path,
            key="display_prediction.maximum_body_scale_displacement",
        )
    return DisplayPredictionConfig(enabled=enabled, mode=mode, **parsed)


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
    "CameraConfig",
    "DisplaySmoothingConfig",
    "DisplayPredictionConfig",
    "OneEuroProfileConfig",
    "LocalFirstArchitectureConfig",
    "ProductPoseConfig",
    "RealtimeLatencyConfig",
    "RenderingConfig",
    "RealtimeSmoothingConfig",
    "ThreeDKinematicsConfig",
    "ThreeDQualityConfig",
    "WebRealtimeConfig",
    "load_product_pose_config",
]
