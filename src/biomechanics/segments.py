from __future__ import annotations

from typing import Sequence

import numpy as np

from .landmarks import coerce_landmarks, point_array, usable_landmark
from .types import LandmarkPoint, SegmentVector


SEGMENT_DEFINITIONS: dict[str, tuple[str, str]] = {
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_lower_leg": ("left_knee", "left_ankle"),
    "right_lower_leg": ("right_knee", "right_ankle"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
}


def _invalid_segment(name: str, start: str, end: str) -> SegmentVector:
    nan_vector = (float("nan"), float("nan"), float("nan"))
    return SegmentVector(name=name, start=start, end=end, vector=nan_vector, unit=nan_vector, valid=False)


def segment_between(name: str, start_name: str, end_name: str, start: LandmarkPoint | None, end: LandmarkPoint | None) -> SegmentVector:
    start_array = point_array(start)
    end_array = point_array(end)
    if start_array is None or end_array is None:
        return _invalid_segment(name, start_name, end_name)
    vector = end_array - start_array
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return _invalid_segment(name, start_name, end_name)
    unit = vector / norm
    return SegmentVector(
        name=name,
        start=start_name,
        end=end_name,
        vector=(float(vector[0]), float(vector[1]), float(vector[2])),
        unit=(float(unit[0]), float(unit[1]), float(unit[2])),
        valid=True,
    )


def compute_segment_vectors(landmarks: Sequence[object] | None) -> dict[str, SegmentVector]:
    points = coerce_landmarks(landmarks)
    segments: dict[str, SegmentVector] = {}
    for name, (start_name, end_name) in SEGMENT_DEFINITIONS.items():
        segments[name] = segment_between(
            name,
            start_name,
            end_name,
            usable_landmark(points, start_name),
            usable_landmark(points, end_name),
        )

    from .angles import compute_body_centers

    centers = compute_body_centers(points)
    segments["trunk"] = segment_between(
        "trunk",
        "pelvis_center",
        "shoulder_center",
        centers["pelvis_center"],
        centers["shoulder_center"],
    )
    return segments

