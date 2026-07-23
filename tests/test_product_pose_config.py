from __future__ import annotations

from pathlib import Path

import pytest

from src.configuration import ConfigValidationError
from src.product_pose import load_product_pose_config


def test_product_pose_config_defaults_to_mediapipe_and_disables_experiments() -> None:
    config = load_product_pose_config(Path("configs/product_pose.yaml"))

    assert config.backend == "mediapipe"
    assert config.allow_experimental_backends is False
    assert config.realtime_model == "auto"
    assert config.analysis_model == "full"
    assert config.analysis_smoothing.profile == "responsive"
    assert config.analysis_smoothing.prediction_enabled is False
    assert config.realtime_smoothing is config.analysis_smoothing
    assert config.display_smoothing.profile == "ultra_responsive"
    assert config.display_smoothing.min_cutoff == pytest.approx(2.2)
    assert config.display_smoothing.beta == pytest.approx(0.12)
    assert config.display_smoothing.raw_blend_enabled is True
    assert config.display_smoothing.max_raw_weight == pytest.approx(0.45)
    assert config.display_smoothing.prediction_enabled is True
    assert config.display_prediction.enabled is True
    assert config.display_prediction.max_horizon_ms == pytest.approx(45)
    assert config.display_prediction.maximum_body_scale_displacement == pytest.approx(0.06)
    assert config.display_prediction.disable_after_gap_ms == pytest.approx(100)
    assert config.rendering.angle_text_fps == pytest.approx(12)
    assert config.rendering.metrics_fps == pytest.approx(5)
    assert config.rendering.stats_fps == pytest.approx(3)
    assert config.rendering.timing_sample_capacity == 240
    assert config.realtime_latency.latest_frame_only is True
    assert config.realtime_latency.camera_buffer_size == 1
    assert config.realtime_latency.max_pose_age_ms == 150
    assert config.realtime_latency.max_frame_gap == 5
    assert config.web_realtime.max_requests_in_flight == 1
    assert config.web_realtime.inference_long_edge == 640
    assert config.web_realtime.jpeg_quality == pytest.approx(0.65)
    assert config.camera.preferred_width == 640
    assert config.camera.preferred_height == 480
    assert config.camera.preferred_fps == pytest.approx(60)
    assert config.camera.fallback_fps == pytest.approx(30)
    assert config.local_first.web_pipeline == "local_browser"
    assert config.local_first.desktop_pipeline == "local_device"
    assert config.local_first.server_pose_fallback is True
    assert config.local_first.neural_prediction_enabled is False


def test_product_pose_config_rejects_an_experimental_product_backend(
    tmp_path: Path,
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n  backend: yolo-pose\n  allow_experimental_backends: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="formal product backend"):
        load_product_pose_config(path)


def test_product_pose_config_rejects_realtime_buffers_larger_than_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "realtime_latency:\n"
        "  camera_buffer_size: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="exactly 1"):
        load_product_pose_config(path)


def test_product_pose_config_rejects_multiple_web_requests_in_flight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "web_realtime:\n"
        "  max_requests_in_flight: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="exactly 1"):
        load_product_pose_config(path)


@pytest.mark.parametrize("model", ["auto", "lite", "full"])
def test_product_pose_config_accepts_realtime_model_tiers(tmp_path: Path, model: str) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        f"  realtime_model: {model}\n"
        "  analysis_model: full\n",
        encoding="utf-8",
    )

    assert load_product_pose_config(path).realtime_model == model


def test_product_pose_config_rejects_non_full_analysis_model(tmp_path: Path) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "  realtime_model: auto\n"
        "  analysis_model: lite\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="formal analysis model"):
        load_product_pose_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("profile", "responsive"),
        ("max_raw_weight", "0.8"),
        ("minimum_visibility", "1.2"),
        ("fast_speed", "0.1"),
    ),
)
def test_product_pose_config_rejects_unsafe_display_smoothing(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "display_smoothing:\n"
        f"  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError):
        load_product_pose_config(path)


def test_product_pose_config_rejects_prediction_in_analysis_stream(tmp_path: Path) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "analysis_smoothing:\n"
        "  prediction_enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="analysis prediction"):
        load_product_pose_config(path)


def test_product_pose_config_rejects_neural_prediction_in_local_first_architecture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "local_first:\n"
        "  neural_prediction_enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="neural prediction"):
        load_product_pose_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mode", "neural"),
        ("max_horizon_ms", "61"),
        ("maximum_body_scale_displacement", "0.21"),
        ("minimum_visibility", "1.1"),
        ("velocity_decay", "-0.1"),
    ),
)
def test_product_pose_config_rejects_unsafe_display_prediction(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "display_prediction:\n"
        f"  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError):
        load_product_pose_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("angle_text_fps", "31"),
        ("metrics_fps", "11"),
        ("stats_fps", "0"),
        ("timing_sample_capacity", "29"),
        ("timing_sample_capacity", "240.5"),
    ),
)
def test_product_pose_config_rejects_unsafe_rendering(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "rendering:\n"
        f"  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError):
        load_product_pose_config(path)
