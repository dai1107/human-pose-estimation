from __future__ import annotations

import math

import pytest

from hyrox.features import extract_basic_pose_features
from hyrox.geometry import calculate_angle_2d
from src.backends.base import Keypoint


def _normalized_sixty_degree_triplet(
    width: int,
    height: int,
) -> tuple[tuple[float, float], ...]:
    vertex_x = width * 0.5
    vertex_y = height * 0.5
    length = min(width, height) * 0.25
    return (
        (vertex_x / width, (vertex_y - length) / height),
        (vertex_x / width, vertex_y / height),
        (
            (vertex_x + length * math.cos(math.radians(30.0))) / width,
            (vertex_y - length * math.sin(math.radians(30.0))) / height,
        ),
    )


@pytest.mark.parametrize("width,height", ((1920, 1080), (1080, 1920)))
def test_normalized_2d_angle_is_corrected_to_pixel_aspect_ratio(
    width: int,
    height: int,
) -> None:
    points = _normalized_sixty_degree_triplet(width, height)

    corrected = calculate_angle_2d(*points, width, height)
    uncorrected = calculate_angle_2d(*points)

    assert corrected == pytest.approx(60.0, abs=0.1)
    assert abs(float(uncorrected) - 60.0) > 5.0


def test_formal_rule_features_use_projected_2d_not_image_landmark_z() -> None:
    base = [
        Keypoint("left_hip", 0.5, 0.25, 0.0, 0.95),
        Keypoint("left_knee", 0.5, 0.5, 0.0, 0.95),
        Keypoint("left_ankle", 0.75, 0.5, 0.0, 0.95),
    ]
    changed_depth = [
        Keypoint(
            point.name,
            point.x,
            point.y,
            4.0 if point.name == "left_ankle" else point.z,
            point.confidence,
        )
        for point in base
    ]

    first = extract_basic_pose_features(base, 1920, 1080)
    second = extract_basic_pose_features(changed_depth, 1920, 1080)

    assert first["left_knee_angle"] == pytest.approx(90.0)
    assert second["left_knee_angle"] == pytest.approx(
        first["left_knee_angle"]
    )
