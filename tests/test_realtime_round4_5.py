from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.backends.mediapipe_backend import MediaPipeBackend, MediaPipeLiveStreamBackend
from src.configuration import ConfigValidationError
from src.product_pose import DisplaySmoothingConfig, load_product_pose_config
from src.realtime.backend_runtime import create_display_smoother, create_runtime_smoother
from src.realtime.cli import parse_args
from src.realtime.inference_resolution import InferenceResolutionController
from src.utils.draw_utils import draw_landmark_lag_debug


def _pose(x: float, timestamp_ms: int) -> PoseResult:
    points = [
        Keypoint(
            name="left_wrist",
            x=x,
            y=0.5,
            confidence=1.0,
            visibility=1.0,
            presence=1.0,
            source_model="mediapipe",
        ),
        Keypoint(
            name="left_elbow",
            x=max(0.0, x - 0.1),
            y=0.5,
            confidence=1.0,
            visibility=1.0,
            presence=1.0,
            source_model="mediapipe",
        ),
    ]
    return PoseResult(
        keypoints=points,
        connections=((0, 1),),
        model_name="mediapipe",
        num_keypoints=2,
        success=True,
        inference_time_ms=10.0,
        timestamp_ms=timestamp_ms,
    )


def test_display_and_analysis_filters_are_independent_and_display_is_more_responsive() -> None:
    config = load_product_pose_config(Path("configs/product_pose.yaml"))
    analysis = create_runtime_smoother(parse_args([]), config.analysis_smoothing)
    display = create_display_smoother(config.display_smoothing)

    for value, timestamp_ms in ((0.2, 0), (0.2, 50), (0.9, 100)):
        raw = _pose(value, timestamp_ms)
        analysis_result = analysis.smooth_result(
            raw, capture_timestamp_ns=timestamp_ms * 1_000_000
        )
        display_result = display.smooth_result(
            raw, capture_timestamp_ns=timestamp_ms * 1_000_000
        )

    assert analysis_result.keypoints[0].x < display_result.keypoints[0].x < 0.9
    assert analysis_result.extra["smoothing_profile"] == "responsive"
    assert display_result.extra["smoothing_profile"] == "ultra_responsive"
    blend = display_result.extra["display_raw_blend"]
    assert 0.0 < blend["max_raw_weight"] <= config.display_smoothing.max_raw_weight
    assert "display_raw_blend" not in analysis_result.extra


def test_display_raw_blend_is_disabled_by_config() -> None:
    display = create_display_smoother(
        DisplaySmoothingConfig(raw_blend_enabled=False, prediction_enabled=True)
    )
    display.smooth_result(_pose(0.1, 0), capture_timestamp_ns=0)
    result = display.smooth_result(_pose(0.9, 50), capture_timestamp_ns=50_000_000)

    assert "display_raw_blend" not in result.extra


def test_display_filter_holds_a_brief_low_confidence_landmark_without_moving_it() -> None:
    display = create_display_smoother(DisplaySmoothingConfig(raw_blend_enabled=False))
    reliable = display.smooth_result(_pose(0.4, 0), capture_timestamp_ns=0)
    low_points = [
        Keypoint(
            name=point.name,
            x=0.9,
            y=point.y,
            confidence=0.1,
            visibility=0.1,
            presence=0.1,
            source_model=point.source_model,
        )
        for point in _pose(0.9, 50).keypoints
    ]
    low = PoseResult(
        keypoints=low_points,
        connections=((0, 1),),
        model_name="mediapipe",
        num_keypoints=2,
        success=True,
        inference_time_ms=10.0,
        timestamp_ms=50,
    )

    held = display.smooth_result(low, capture_timestamp_ns=50_000_000)
    expired = display.smooth_result(low, capture_timestamp_ns=271_000_000)

    assert held.keypoints[0].x == pytest.approx(reliable.keypoints[0].x)
    assert held.keypoints[0].confidence > 0.2
    assert expired.keypoints[0].confidence == 0.0


def test_display_filter_holds_short_whole_pose_misses() -> None:
    display = create_display_smoother(DisplaySmoothingConfig(pose_hold_frames=2))
    display.smooth_result(_pose(0.4, 0), capture_timestamp_ns=0)
    missed = PoseResult(
        keypoints=[],
        connections=(),
        model_name="mediapipe",
        num_keypoints=0,
        success=False,
        inference_time_ms=10.0,
        timestamp_ms=33,
    )

    first = display.smooth_result(missed, capture_timestamp_ns=33_000_000)
    second = display.smooth_result(missed, capture_timestamp_ns=66_000_000)
    third = display.smooth_result(missed, capture_timestamp_ns=99_000_000)

    assert first.success and first.extra["stabilized_hold"]
    assert second.success and second.extra["hold_frames"] == 2
    assert third.success is False


def test_display_filter_deadband_suppresses_subpixel_static_jitter() -> None:
    display = create_display_smoother(
        DisplaySmoothingConfig(raw_blend_enabled=False, jitter_deadband=0.0025)
    )
    first = display.smooth_result(_pose(0.4, 0), capture_timestamp_ns=0)
    jittered = display.smooth_result(
        _pose(0.401, 33),
        capture_timestamp_ns=33_000_000,
    )

    assert jittered.keypoints[0].x == pytest.approx(first.keypoints[0].x)


def test_landmark_lag_debug_draws_raw_and_filtered_overlay() -> None:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    draw_landmark_lag_debug(frame, _pose(0.25, 0), _pose(0.75, 0))

    assert np.count_nonzero(frame) > 0
    assert np.count_nonzero(frame[:, :, 0]) > 0  # raw magenta contains blue
    assert np.count_nonzero(frame[:, :, 1]) > 0  # display green


def test_inference_resolution_preserves_display_frame_and_aspect_ratio() -> None:
    controller = InferenceResolutionController(640, adaptive=False)
    display = np.zeros((720, 1280, 3), dtype=np.uint8)
    prepared = controller.prepare(display)

    assert display.shape == (720, 1280, 3)
    assert prepared.image.shape == (360, 640, 3)
    assert prepared.width == 640
    assert prepared.height == 360
    assert prepared.image is not display
    assert prepared.resize_ms >= 0.0


def test_inference_resolution_never_upscales_and_adaptively_steps_down() -> None:
    controller = InferenceResolutionController(
        640,
        adaptive=True,
        max_inference_p95_ms=20.0,
        sample_window=4,
        min_samples=3,
    )
    small = np.zeros((240, 320, 3), dtype=np.uint8)
    assert controller.prepare(small).image is small

    assert controller.observe(30.0) is False
    assert controller.observe(35.0) is False
    assert controller.observe(40.0) is True
    assert controller.current_width == 512
    assert controller.downgrade_count == 1


def test_round5_product_config_and_backend_safe_defaults() -> None:
    config = load_product_pose_config(Path("configs/product_pose.yaml"))

    assert config.pose.inference_width == 640
    assert config.pose.adaptive_resolution is True
    assert inspect.signature(MediaPipeBackend).parameters[
        "output_segmentation_masks"
    ].default is False
    assert inspect.signature(MediaPipeLiveStreamBackend).parameters[
        "output_segmentation_masks"
    ].default is False
    assert parse_args(["--landmark-lag-debug"]).landmark_lag_debug is True


@pytest.mark.parametrize(
    "pose_yaml",
    (
        "  inference_width: 100\n",
        "  inference_width: wide\n",
        "  adaptive_resolution: yes\n",
        "  unknown: 1\n",
    ),
)
def test_pose_inference_config_rejects_invalid_values(
    tmp_path: Path, pose_yaml: str
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "pose:\n"
        f"{pose_yaml}",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError):
        load_product_pose_config(path)
