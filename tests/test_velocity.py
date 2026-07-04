from __future__ import annotations

import math

import pytest

from src.biomechanics.types import LandmarkPoint
from src.biomechanics.velocity import angular_velocity, linear_velocity


def test_linear_velocity_uses_real_timestamp_delta() -> None:
    _, _, _, speed = linear_velocity(
        LandmarkPoint(0.1, 0.0, 0.0),
        LandmarkPoint(0.0, 0.0, 0.0),
        current_timestamp_ms=1100,
        previous_timestamp_ms=1000,
    )
    assert speed == pytest.approx(1.0)


def test_angular_velocity_uses_real_timestamp_delta() -> None:
    omega = angular_velocity(
        current_angle=120.0,
        previous_angle=90.0,
        current_timestamp_ms=2000,
        previous_timestamp_ms=1000,
    )
    assert omega == pytest.approx(30.0)


def test_bad_timestamp_or_missing_point_returns_nan() -> None:
    _, _, _, speed = linear_velocity(
        LandmarkPoint(1.0, 0.0, 0.0),
        None,
        current_timestamp_ms=1000,
        previous_timestamp_ms=1000,
    )
    assert math.isnan(speed)
    assert math.isnan(angular_velocity(10.0, 5.0, 1000, 1000))

