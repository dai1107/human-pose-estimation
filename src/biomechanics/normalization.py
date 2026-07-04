from __future__ import annotations

from math import isfinite
from typing import Sequence

import numpy as np

from .landmarks import coerce_landmarks, midpoint, usable_landmark
from .types import LandmarkPoint, NormalizationResult


def _distance(point_a: LandmarkPoint | None, point_b: LandmarkPoint | None) -> float:
    if point_a is None or point_b is None:
        return float("nan")
    vector = np.array(point_a.xyz(), dtype=float) - np.array(point_b.xyz(), dtype=float)
    return float(np.linalg.norm(vector))


def _valid_scale(value: float) -> bool:
    return isfinite(value) and value > 1e-9


def normalize_landmarks(landmarks: Sequence[object] | None) -> NormalizationResult:
    points = coerce_landmarks(landmarks)
    pelvis_center = midpoint(points, "left_hip", "right_hip")
    if pelvis_center is None:
        return NormalizationResult([], False, None, None, None, "pelvis center unavailable")

    left_shoulder = usable_landmark(points, "left_shoulder")
    right_shoulder = usable_landmark(points, "right_shoulder")
    left_hip = usable_landmark(points, "left_hip")
    right_hip = usable_landmark(points, "right_hip")
    shoulder_center = midpoint(points, "left_shoulder", "right_shoulder")

    scale_candidates = (
        ("shoulder_width", _distance(left_shoulder, right_shoulder)),
        ("hip_width", _distance(left_hip, right_hip)),
        ("torso_length", _distance(pelvis_center, shoulder_center)),
    )
    scale_method = None
    scale = None
    for candidate_name, candidate_scale in scale_candidates:
        if _valid_scale(candidate_scale):
            scale_method = candidate_name
            scale = candidate_scale
            break

    if scale is None or scale_method is None:
        return NormalizationResult([], False, pelvis_center.xyz(), None, None, "normalization scale unavailable")

    origin = np.array(pelvis_center.xyz(), dtype=float)
    normalized: list[LandmarkPoint] = []
    for point in points:
        if not point.is_finite():
            normalized.append(point)
            continue
        coords = (np.array(point.xyz(), dtype=float) - origin) / scale
        normalized.append(
            LandmarkPoint(
                x=float(coords[0]),
                y=float(coords[1]),
                z=float(coords[2]),
                visibility=point.visibility,
                presence=point.presence,
            )
        )

    return NormalizationResult(
        landmarks=normalized,
        success=True,
        origin=pelvis_center.xyz(),
        scale=scale,
        scale_method=scale_method,
        message="ok",
    )

