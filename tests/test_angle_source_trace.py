from __future__ import annotations

import pytest

from src.backends.base import Keypoint, PoseResult
from src.utils.keypoint_schema import MEDIAPIPE_CONNECTIONS
from webui.angle_trace import angle_source_summary, trace_angle_sources


def _point(
    name: str,
    x: float,
    y: float,
    z: float = 0.0,
) -> Keypoint:
    return Keypoint(
        name=name,
        x=x,
        y=y,
        z=z,
        confidence=0.9,
        visibility=0.95,
        presence=0.9,
        source_model="test",
    )


def _result(
    image_points: list[Keypoint],
    world_points: list[Keypoint],
) -> PoseResult:
    return PoseResult(
        keypoints=image_points,
        connections=MEDIAPIPE_CONNECTIONS,
        model_name="test",
        num_keypoints=len(image_points),
        success=True,
        inference_time_ms=0.0,
        extra={"world_keypoints": world_points},
    )


def test_angle_trace_names_display_rule_and_skeleton_coordinate_sources() -> None:
    raw_image = [
        _point("left_hip", 0.5, 0.25),
        _point("left_knee", 0.5, 0.5),
        _point("left_ankle", 0.75, 0.5),
    ]
    smooth_image = [
        _point("left_hip", 0.5, 0.25),
        _point("left_knee", 0.5, 0.5),
        _point("left_ankle", 0.70, 0.5),
    ]
    raw_world = [
        _point("left_hip", 0.0, 1.0, 0.0),
        _point("left_knee", 0.0, 0.0, 0.0),
        _point("left_ankle", 1.0, 0.0, 0.0),
    ]
    smooth_world = [
        _point("left_hip", 0.0, 1.0, 0.0),
        _point("left_knee", 0.0, 0.0, 0.0),
        _point("left_ankle", 1.0, 0.0, 0.0),
    ]
    observations = trace_angle_sources(
        frame_index=7,
        timestamp_ms=233.333,
        raw_result=_result(raw_image, raw_world),
        smoothed_result=_result(smooth_image, smooth_world),
        image_width=200,
        image_height=100,
        rule_features={"left_knee_angle": 90.0},
        assessment={
            "angles": [
                {
                    "key": "left_knee_angle",
                    "value": 90.0,
                    "display_angle_source": "world_landmarks_smoothed",
                }
            ]
        },
    )
    knee = next(
        item
        for item in observations
        if item["side"] == "left" and item["joint_name"] == "knee"
    )

    assert knee["angle_2d_raw_deg"] == 90.0
    assert knee["angle_2d_smoothed_deg"] == 90.0
    assert knee["angle_3d_raw_deg"] == 90.0
    assert knee["angle_3d_smoothed_deg"] == 90.0
    assert knee["display_angle_deg"] == 90.0
    assert knee["rule_angle_deg"] == 90.0
    assert knee["drawn_landmarks_angle_deg"] == 90.0
    assert knee["display_angle_source"] == "world_landmarks_smoothed"
    assert knee["rule_angle_source"] == "image_landmarks_analysis_smoothed"
    assert (
        knee["drawn_landmarks_source"]
        == "image_landmarks_analysis_smoothed"
    )
    assert knee["geometry_valid"] is True

    summary = angle_source_summary(observations)
    assert summary == {
        "display_angle_source": ["world_landmarks_smoothed"],
        "rule_angle_source": "image_landmarks_analysis_smoothed",
        "screen_skeleton_source": "image_landmarks_analysis_smoothed",
        "displayed_angle_count": 1,
        "display_skeleton_mismatch_count": 0,
        "display_skeleton_tolerance_deg": 0.5,
    }


def test_angle_trace_exports_center_torso_raw_smoothed_3d_and_rule_sources() -> None:
    image = [
        _point("left_shoulder", 0.4, 0.2),
        _point("right_shoulder", 0.6, 0.2),
        _point("left_hip", 0.45, 0.6),
        _point("right_hip", 0.65, 0.6),
    ]
    world = [
        _point("left_shoulder", -0.2, -0.4, 0.0),
        _point("right_shoulder", 0.2, -0.4, 0.0),
        _point("left_hip", -0.1, 0.0, 0.0),
        _point("right_hip", 0.3, 0.0, 0.0),
    ]

    observations = trace_angle_sources(
        frame_index=2,
        timestamp_ms=66.667,
        raw_result=_result(image, world),
        smoothed_result=_result(image, world),
        image_width=200,
        image_height=100,
        rule_features={"torso_angle": -14.036243},
        assessment={"angles": []},
    )
    torso = next(
        item
        for item in observations
        if item["joint_name"] == "torso"
    )

    assert torso["side"] == "center"
    assert torso["angle_2d_raw_deg"] == pytest.approx(-14.0362)
    assert torso["angle_2d_smoothed_deg"] == pytest.approx(-14.0362)
    assert torso["angle_3d_raw_deg"] == pytest.approx(14.0362)
    assert torso["angle_3d_smoothed_deg"] == pytest.approx(14.0362)
    assert torso["rule_angle_deg"] == pytest.approx(-14.0362)
    assert torso["display_angle_source"] == "not_displayed"
