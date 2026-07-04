from __future__ import annotations

from math import degrees, isfinite, nan
from typing import Sequence

import numpy as np

from .landmarks import coerce_landmarks, midpoint, point_array, usable_landmark
from .types import LandmarkPoint


ANGLE_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
}


def calculate_joint_angle(
    point_a: LandmarkPoint | object | None,
    point_b: LandmarkPoint | object | None,
    point_c: LandmarkPoint | object | None,
) -> float:
    a = point_array(point_a if isinstance(point_a, LandmarkPoint) else None)
    b = point_array(point_b if isinstance(point_b, LandmarkPoint) else None)
    c = point_array(point_c if isinstance(point_c, LandmarkPoint) else None)

    if not isinstance(point_a, LandmarkPoint) and point_a is not None:
        a = point_array(LandmarkPoint(float(getattr(point_a, "x", nan)), float(getattr(point_a, "y", nan)), float(getattr(point_a, "z", 0.0))))
    if not isinstance(point_b, LandmarkPoint) and point_b is not None:
        b = point_array(LandmarkPoint(float(getattr(point_b, "x", nan)), float(getattr(point_b, "y", nan)), float(getattr(point_b, "z", 0.0))))
    if not isinstance(point_c, LandmarkPoint) and point_c is not None:
        c = point_array(LandmarkPoint(float(getattr(point_c, "x", nan)), float(getattr(point_c, "y", nan)), float(getattr(point_c, "z", 0.0))))

    if a is None or b is None or c is None:
        return nan
    vector_ab = a - b
    vector_cb = c - b
    norm_ab = float(np.linalg.norm(vector_ab))
    norm_cb = float(np.linalg.norm(vector_cb))
    if norm_ab <= 1e-12 or norm_cb <= 1e-12:
        return nan
    cosine = float(np.dot(vector_ab, vector_cb) / (norm_ab * norm_cb))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle = degrees(float(np.arccos(cosine)))
    return angle if isfinite(angle) else nan


def compute_body_centers(landmarks: Sequence[object] | None) -> dict[str, LandmarkPoint | None]:
    points = coerce_landmarks(landmarks)
    pelvis_center = midpoint(points, "left_hip", "right_hip")
    shoulder_center = midpoint(points, "left_shoulder", "right_shoulder")
    return {"pelvis_center": pelvis_center, "shoulder_center": shoulder_center}


def compute_trunk_tilt_proxy(landmarks: Sequence[object] | None) -> float:
    centers = compute_body_centers(landmarks)
    pelvis_center = centers["pelvis_center"]
    shoulder_center = centers["shoulder_center"]
    pelvis = point_array(pelvis_center)
    shoulder = point_array(shoulder_center)
    if pelvis is None or shoulder is None:
        return nan
    trunk_xy = (shoulder - pelvis)[:2]
    norm = float(np.linalg.norm(trunk_xy))
    if norm <= 1e-12:
        return nan
    image_vertical_up = np.array([0.0, -1.0], dtype=float)
    cosine = float(np.dot(trunk_xy, image_vertical_up) / norm)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return degrees(float(np.arccos(cosine)))


def compute_joint_angles(landmarks: Sequence[object] | None) -> dict[str, float]:
    points = coerce_landmarks(landmarks)
    angles: dict[str, float] = {}
    for angle_name, (first, middle, third) in ANGLE_DEFINITIONS.items():
        point_a = usable_landmark(points, first)
        point_b = usable_landmark(points, middle)
        point_c = usable_landmark(points, third)
        angles[angle_name] = calculate_joint_angle(point_a, point_b, point_c)
    angles["trunk_tilt_proxy"] = compute_trunk_tilt_proxy(points)
    return angles

