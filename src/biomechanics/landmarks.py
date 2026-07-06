from __future__ import annotations

from math import nan
from typing import Iterable, Sequence

import numpy as np

from .types import LandmarkPoint


LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

LANDMARK_INDEX: dict[str, int] = {name: index for index, name in enumerate(LANDMARK_NAMES)}


def empty_landmarks(count: int = 33) -> list[LandmarkPoint]:
    return [LandmarkPoint(nan, nan, nan, 0.0, 0.0) for _ in range(count)]


def _float_attribute(point: object, name: str, default: float) -> float:
    value = getattr(point, name, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_landmark(point: object | None) -> LandmarkPoint:
    if point is None:
        return LandmarkPoint(nan, nan, nan, 0.0, 0.0)
    return LandmarkPoint(
        x=_float_attribute(point, "x", nan),
        y=_float_attribute(point, "y", nan),
        z=_float_attribute(point, "z", 0.0),
        visibility=_float_attribute(point, "visibility", 1.0),
        presence=_float_attribute(point, "presence", 1.0),
    )


def coerce_landmarks(points: Sequence[object] | None, expected_count: int = 33) -> list[LandmarkPoint]:
    if points is None:
        return []
    landmarks = [coerce_landmark(point) for point in points]
    if len(landmarks) < expected_count:
        landmarks.extend(empty_landmarks(expected_count - len(landmarks)))
    return landmarks[:expected_count]


def landmark_name(index: int) -> str:
    if 0 <= index < len(LANDMARK_NAMES):
        return LANDMARK_NAMES[index]
    return f"landmark_{index}"


def landmark_at(landmarks: Sequence[LandmarkPoint], name_or_index: str | int) -> LandmarkPoint | None:
    index = LANDMARK_INDEX[name_or_index] if isinstance(name_or_index, str) else name_or_index
    if index < 0 or index >= len(landmarks):
        return None
    point = landmarks[index]
    if not point.is_finite():
        return None
    return point


def usable_landmark(
    landmarks: Sequence[LandmarkPoint],
    name_or_index: str | int,
    min_visibility: float = 0.2,
    min_presence: float = 0.2,
) -> LandmarkPoint | None:
    point = landmark_at(landmarks, name_or_index)
    if point is None or not point.is_usable(min_visibility, min_presence):
        return None
    return point


def point_array(point: LandmarkPoint | None) -> np.ndarray | None:
    if point is None or not point.is_finite():
        return None
    return np.array([point.x, point.y, point.z], dtype=float)


def midpoint(
    landmarks: Sequence[LandmarkPoint],
    first: str | int,
    second: str | int,
    min_visibility: float = 0.2,
) -> LandmarkPoint | None:
    point_a = usable_landmark(landmarks, first, min_visibility=min_visibility)
    point_b = usable_landmark(landmarks, second, min_visibility=min_visibility)
    if point_a is None or point_b is None:
        return None
    return LandmarkPoint(
        x=(point_a.x + point_b.x) / 2.0,
        y=(point_a.y + point_b.y) / 2.0,
        z=(point_a.z + point_b.z) / 2.0,
        visibility=min(point_a.visibility, point_b.visibility),
        presence=min(point_a.presence, point_b.presence),
    )


def mean_visibility(landmarks: Iterable[LandmarkPoint]) -> float:
    values = [point.visibility for point in landmarks if point.is_finite()]
    return float(np.mean(values)) if values else nan


def missing_ratio(landmarks: Sequence[LandmarkPoint], min_visibility: float = 0.2) -> float:
    if not landmarks:
        return 1.0
    missing = sum(1 for point in landmarks if not point.is_usable(min_visibility=min_visibility))
    return missing / len(landmarks)
