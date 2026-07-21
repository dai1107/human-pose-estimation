from __future__ import annotations

from pathlib import Path

import pytest

from src.configuration import ConfigValidationError
from src.product_pose import load_product_pose_config


def test_product_pose_config_defaults_to_mediapipe_and_disables_experiments() -> None:
    config = load_product_pose_config(Path("configs/product_pose.yaml"))

    assert config.backend == "mediapipe"
    assert config.allow_experimental_backends is False
    assert config.realtime_latency.latest_frame_only is True
    assert config.realtime_latency.camera_buffer_size == 1
    assert config.realtime_latency.max_pose_age_ms == 150
    assert config.realtime_latency.max_frame_gap == 5
    assert config.web_realtime.max_requests_in_flight == 1
    assert config.web_realtime.inference_long_edge == 640
    assert config.web_realtime.jpeg_quality == pytest.approx(0.65)


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
