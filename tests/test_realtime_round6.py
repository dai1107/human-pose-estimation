from __future__ import annotations

from math import isfinite
from pathlib import Path

import pytest

from src.backends.base import Keypoint, PoseResult
from src.configuration import ConfigValidationError
from src.product_pose import load_product_pose_config
from src.realtime.backend_runtime import create_runtime_smoother
from src.realtime.cli import parse_args
from src.utils.smoothing import KeypointSmoother, OneEuroValueFilter


def _pose(
    image_x: float,
    timestamp_ms: int,
    *,
    name: str = "left_knee",
    world_x: float | None = None,
) -> PoseResult:
    extra: dict[str, object] = {}
    if world_x is not None:
        extra["world_keypoints"] = [
            Keypoint(
                name=name,
                x=world_x,
                y=0.0,
                z=0.0,
                confidence=1.0,
                source_model="mediapipe-world",
            )
        ]
    return PoseResult(
        keypoints=[
            Keypoint(
                name=name,
                x=image_x,
                y=0.0,
                z=0.0,
                confidence=1.0,
                source_model="mediapipe",
            )
        ],
        connections=(),
        model_name="mediapipe",
        num_keypoints=1,
        success=True,
        inference_time_ms=1.0,
        timestamp_ms=timestamp_ms,
        extra=extra,
    )


def test_one_euro_uses_observed_dt_instead_of_a_fixed_frame_rate() -> None:
    short_dt = OneEuroValueFilter(
        min_cutoff=1.0,
        beta=0.0,
        d_cutoff=1.0,
        max_gap_ms_before_reset=500.0,
    )
    long_dt = OneEuroValueFilter(
        min_cutoff=1.0,
        beta=0.0,
        d_cutoff=1.0,
        max_gap_ms_before_reset=500.0,
    )

    short_dt.apply(0.0, timestamp_ns=1_000_000_000)
    short_value = short_dt.apply(1.0, timestamp_ns=1_010_000_000)
    long_dt.apply(0.0, timestamp_ns=1_000_000_000)
    long_value = long_dt.apply(1.0, timestamp_ns=1_100_000_000)

    assert short_dt.last_dt_seconds == pytest.approx(0.010)
    assert long_dt.last_dt_seconds == pytest.approx(0.100)
    assert long_value > short_value


def test_one_euro_without_capture_time_does_not_invent_30_fps() -> None:
    value_filter = OneEuroValueFilter()

    assert value_filter.apply(0.0) == 0.0
    assert value_filter.apply(1.0) == 1.0
    assert value_filter.last_dt_seconds is None


def test_stable_balanced_and_responsive_profiles_have_increasing_response() -> None:
    outputs: dict[str, float] = {}
    for profile in ("stable", "balanced", "responsive"):
        smoother = KeypointSmoother(profile=profile, occlusion_guard=False)
        smoother.smooth_result(_pose(0.0, 0))
        outputs[profile] = smoother.smooth_result(_pose(1.0, 33)).keypoints[0].x

    assert outputs["stable"] < outputs["balanced"] < outputs["responsive"]


def test_fast_joint_group_responds_faster_than_stable_joint_group() -> None:
    smoother = KeypointSmoother(profile="balanced", occlusion_guard=False)
    first = PoseResult(
        keypoints=[
            Keypoint("left_wrist", 0.0, 0.0, confidence=1.0, source_model="mediapipe"),
            Keypoint("left_shoulder", 0.0, 0.0, confidence=1.0, source_model="mediapipe"),
        ],
        connections=(),
        model_name="mediapipe",
        num_keypoints=2,
        success=True,
        inference_time_ms=1.0,
        timestamp_ms=0,
    )
    second = PoseResult(
        keypoints=[
            Keypoint("left_wrist", 1.0, 0.0, confidence=1.0, source_model="mediapipe"),
            Keypoint("left_shoulder", 1.0, 0.0, confidence=1.0, source_model="mediapipe"),
        ],
        connections=(),
        model_name="mediapipe",
        num_keypoints=2,
        success=True,
        inference_time_ms=1.0,
        timestamp_ms=33,
    )

    smoother.smooth_result(first)
    moved = smoother.smooth_result(second)

    assert moved.keypoints[0].x > moved.keypoints[1].x


def test_long_observation_gap_resets_filter_to_current_pose() -> None:
    smoother = KeypointSmoother(max_gap_ms_before_reset=250.0, occlusion_guard=False)
    smoother.smooth_result(_pose(0.0, 0))
    before_gap = smoother.smooth_result(_pose(1.0, 100))
    after_gap = smoother.smooth_result(_pose(0.25, 400))

    assert before_gap.keypoints[0].x < 1.0
    assert after_gap.keypoints[0].x == pytest.approx(0.25)
    assert after_gap.extra["smoothing_gap_reset_spaces"] == ("image",)


def test_image_and_world_filter_states_reset_independently() -> None:
    smoother = KeypointSmoother(max_gap_ms_before_reset=250.0, occlusion_guard=False)
    smoother.smooth_result(_pose(0.0, 0, world_x=0.0))
    smoother.smooth_result(_pose(1.0, 100, world_x=1.0))
    smoother.smooth_result(_pose(0.4, 200))
    smoother.smooth_result(_pose(0.6, 300))
    result = smoother.smooth_result(_pose(1.0, 400, world_x=0.25))

    world_points = result.extra["world_keypoints"]
    assert isinstance(world_points, list)
    assert result.keypoints[0].x < 1.0
    assert world_points[0].x == pytest.approx(0.25)
    assert result.extra["smoothing_gap_reset_spaces"] == ("world",)
    assert result.extra["world_landmarks_smoothed"] is True


def test_finite_inputs_do_not_generate_nan() -> None:
    smoother = KeypointSmoother(profile="responsive", occlusion_guard=False)
    for frame_id, value in enumerate((0.0, 1.0, -0.5, 0.75, 0.25)):
        result = smoother.smooth_result(
            _pose(value, frame_id * 41, world_x=value * 0.5)
        )
        world_points = result.extra["world_keypoints"]
        assert all(
            isfinite(axis)
            for axis in (
                result.keypoints[0].x,
                result.keypoints[0].y,
                result.keypoints[0].z,
                result.keypoints[0].confidence,
            )
        )
        assert isinstance(world_points, list)
        assert all(isfinite(axis) for axis in (world_points[0].x, world_points[0].y, world_points[0].z))


def test_product_smoothing_config_drives_runtime_defaults() -> None:
    config = load_product_pose_config(Path("configs/product_pose.yaml"))
    smoother = create_runtime_smoother(parse_args([]), config.realtime_smoothing)

    assert config.realtime_smoothing.profile == "responsive"
    assert config.realtime_smoothing.max_gap_ms_before_reset == 250.0
    assert set(config.realtime_smoothing.profiles) == {
        "stable",
        "balanced",
        "responsive",
    }
    assert smoother.profile == "responsive"
    assert smoother.one_euro_min_cutoff == pytest.approx(1.7)
    assert smoother.one_euro_beta == pytest.approx(0.08)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("profile", "turbo"),
        ("max_gap_ms_before_reset", "0"),
        ("stable_min_cutoff", "nan"),
        ("fast_joint_beta_scale", "-1"),
    ),
)
def test_product_smoothing_config_rejects_invalid_values(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path = tmp_path / "product_pose.yaml"
    path.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "realtime_smoothing:\n"
        f"  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError):
        load_product_pose_config(path)
