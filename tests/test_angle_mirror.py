from __future__ import annotations

import pytest

from hyrox.features import extract_basic_pose_features
from hyrox.geometry import calculate_angle_2d
from src.backends.base import Keypoint


def test_horizontal_mirror_preserves_angle_value() -> None:
    points = ((0.2, 0.3), (0.4, 0.6), (0.7, 0.8))
    mirrored = tuple((1.0 - x, y) for x, y in points)

    assert calculate_angle_2d(*mirrored, 1920, 1080) == pytest.approx(
        calculate_angle_2d(*points, 1920, 1080)
    )


def test_mirror_preserves_left_and_right_field_identity() -> None:
    points = [
        Keypoint("left_hip", 0.20, 0.20, confidence=0.95),
        Keypoint("left_knee", 0.20, 0.50, confidence=0.95),
        Keypoint("left_ankle", 0.40, 0.50, confidence=0.95),
        Keypoint("right_hip", 0.70, 0.20, confidence=0.95),
        Keypoint("right_knee", 0.70, 0.50, confidence=0.95),
        Keypoint("right_ankle", 0.70, 0.80, confidence=0.95),
    ]
    mirrored = [
        Keypoint(
            point.name,
            1.0 - point.x,
            point.y,
            confidence=point.confidence,
        )
        for point in points
    ]

    original_features = extract_basic_pose_features(points, 1280, 720)
    mirrored_features = extract_basic_pose_features(mirrored, 1280, 720)

    assert original_features["left_knee_angle"] == pytest.approx(90.0)
    assert original_features["right_knee_angle"] == pytest.approx(180.0)
    assert mirrored_features["left_knee_angle"] == pytest.approx(
        original_features["left_knee_angle"]
    )
    assert mirrored_features["right_knee_angle"] == pytest.approx(
        original_features["right_knee_angle"]
    )
