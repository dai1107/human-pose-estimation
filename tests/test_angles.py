from __future__ import annotations

import math

import pytest

from src.biomechanics.angles import calculate_joint_angle, compute_joint_angles
from src.biomechanics.landmarks import empty_landmarks
from src.biomechanics.types import LandmarkPoint


def test_calculate_joint_angle_90_degrees() -> None:
    angle = calculate_joint_angle(
        LandmarkPoint(1.0, 0.0, 0.0),
        LandmarkPoint(0.0, 0.0, 0.0),
        LandmarkPoint(0.0, 1.0, 0.0),
    )
    assert angle == pytest.approx(90.0)


def test_calculate_joint_angle_180_degrees() -> None:
    angle = calculate_joint_angle(
        LandmarkPoint(-1.0, 0.0, 0.0),
        LandmarkPoint(0.0, 0.0, 0.0),
        LandmarkPoint(1.0, 0.0, 0.0),
    )
    assert angle == pytest.approx(180.0)


def test_calculate_joint_angle_0_degrees() -> None:
    angle = calculate_joint_angle(
        LandmarkPoint(2.0, 0.0, 0.0),
        LandmarkPoint(1.0, 0.0, 0.0),
        LandmarkPoint(3.0, 0.0, 0.0),
    )
    assert angle == pytest.approx(0.0)


def test_missing_points_return_nan_without_exception() -> None:
    landmarks = empty_landmarks()
    angles = compute_joint_angles(landmarks)
    assert math.isnan(angles["left_elbow_angle"])
    assert math.isnan(calculate_joint_angle(None, None, None))

