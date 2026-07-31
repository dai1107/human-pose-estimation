from __future__ import annotations

import math

import pytest

from hyrox.geometry import calculate_angle_2d, calculate_angle_3d


@pytest.mark.parametrize(
    "points",
    (
        ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
        ((0.0, 0.0), (1.0, 1.0), (1.0, 1.0)),
        ((math.nan, 0.0), (0.0, 0.0), (1.0, 1.0)),
        ((math.inf, 0.0), (0.0, 0.0), (1.0, 1.0)),
        (None, (0.0, 0.0), (1.0, 1.0)),
    ),
)
def test_invalid_2d_geometry_returns_none_not_zero(
    points: tuple[object, object, object],
) -> None:
    assert calculate_angle_2d(*points) is None


def test_invalid_frame_dimensions_return_none() -> None:
    points = ((0.0, 1.0), (0.0, 0.0), (1.0, 0.0))

    assert calculate_angle_2d(*points, 0, 1080) is None
    assert calculate_angle_2d(*points, 1920, math.nan) is None
    assert calculate_angle_2d(*points, 1920, None) is None


def test_low_visibility_geometry_remains_calculable_for_explicit_quality_tagging() -> None:
    points = (
        {"x": 0.0, "y": 1.0, "visibility": 0.1, "presence": 0.1},
        {"x": 0.0, "y": 0.0, "visibility": 0.1, "presence": 0.1},
        {"x": 1.0, "y": 0.0, "visibility": 0.1, "presence": 0.1},
    )

    assert calculate_angle_2d(*points) == pytest.approx(90.0)


def test_invalid_3d_geometry_returns_none() -> None:
    assert calculate_angle_3d(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ) is None
    assert calculate_angle_3d(
        (0.0, 0.0, math.nan),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ) is None
