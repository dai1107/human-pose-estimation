from __future__ import annotations

import math

import pytest

from hyrox.geometry import calculate_angle_2d, calculate_angle_3d


@pytest.mark.parametrize(
    ("degrees", "expected"),
    (
        (180.0, 180.0),
        (90.0, 90.0),
        (60.0, 60.0),
        (45.0, 45.0),
    ),
)
def test_fixed_synthetic_angles_are_in_degrees_with_sub_tenth_degree_error(
    degrees: float,
    expected: float,
) -> None:
    radians = math.radians(degrees)
    first = (1.0, 0.0)
    vertex = (0.0, 0.0)
    second = (math.cos(radians), math.sin(radians))

    actual = calculate_angle_2d(first, vertex, second)

    assert actual == pytest.approx(expected, abs=0.1)


def test_translation_and_uniform_scale_do_not_change_angle() -> None:
    points = ((1.0, 0.0), (0.0, 0.0), (0.5, math.sqrt(3.0) / 2.0))
    translated = tuple((x + 17.0, y - 9.0) for x, y in points)
    scaled = tuple((x * 5.0, y * 5.0) for x, y in points)

    baseline = calculate_angle_2d(*points)

    assert baseline == pytest.approx(60.0)
    assert calculate_angle_2d(*translated) == pytest.approx(baseline)
    assert calculate_angle_2d(*scaled) == pytest.approx(baseline)


def test_b_is_always_the_vertex_and_endpoint_order_is_symmetric() -> None:
    a = (0.0, 0.0)
    b = (1.0, 0.0)
    c = (2.0, 1.0)

    at_b = calculate_angle_2d(a, b, c)

    assert at_b == pytest.approx(135.0)
    assert calculate_angle_2d(c, b, a) == pytest.approx(at_b)
    assert calculate_angle_2d(b, a, c) == pytest.approx(
        math.degrees(math.atan2(1.0, 2.0))
    )


def test_3d_angle_uses_all_three_axes_and_returns_degrees() -> None:
    assert calculate_angle_3d(
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ) == pytest.approx(90.0)
