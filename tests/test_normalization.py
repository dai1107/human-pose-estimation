from __future__ import annotations

import pytest

from src.biomechanics.landmarks import LANDMARK_INDEX, empty_landmarks
from src.biomechanics.normalization import normalize_landmarks
from src.biomechanics.types import LandmarkPoint


def _pose() -> list[LandmarkPoint]:
    points = empty_landmarks()
    points[LANDMARK_INDEX["left_hip"]] = LandmarkPoint(-0.5, 0.0, 0.0)
    points[LANDMARK_INDEX["right_hip"]] = LandmarkPoint(0.5, 0.0, 0.0)
    points[LANDMARK_INDEX["left_shoulder"]] = LandmarkPoint(-0.5, -1.0, 0.0)
    points[LANDMARK_INDEX["right_shoulder"]] = LandmarkPoint(0.5, -1.0, 0.0)
    points[LANDMARK_INDEX["left_wrist"]] = LandmarkPoint(-1.0, -1.5, 0.0)
    return points


def test_normalization_is_translation_and_scale_invariant() -> None:
    base = _pose()
    transformed = [
        LandmarkPoint(point.x * 3.0 + 10.0, point.y * 3.0 - 7.0, point.z * 3.0, point.visibility, point.presence)
        for point in base
    ]

    base_norm = normalize_landmarks(base)
    transformed_norm = normalize_landmarks(transformed)

    assert base_norm.success
    assert transformed_norm.success
    left_wrist_index = LANDMARK_INDEX["left_wrist"]
    assert transformed_norm.landmarks[left_wrist_index].x == pytest.approx(base_norm.landmarks[left_wrist_index].x)
    assert transformed_norm.landmarks[left_wrist_index].y == pytest.approx(base_norm.landmarks[left_wrist_index].y)


def test_normalization_failure_has_clear_status() -> None:
    result = normalize_landmarks(empty_landmarks())
    assert not result.success
    assert result.message

