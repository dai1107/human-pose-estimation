from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import atan2, degrees, isfinite
from typing import Any

from .geometry import PosePoint, angle_3pts, coerce_point, midpoint
from .landmark_names import HYROX_CORE_LANDMARKS, LANDMARK_INDEX


FeatureValue = float | None


def _empty_features() -> dict[str, FeatureValue]:
    return {
        "left_knee_angle": None,
        "right_knee_angle": None,
        "left_hip_angle": None,
        "right_hip_angle": None,
        "torso_angle": None,
        "shoulder_tilt": None,
        "hip_tilt": None,
        "min_knee_angle": None,
        "min_hip_angle": None,
        "hip_center_y": None,
        "knee_center_y": None,
        "visible_score": 0.0,
    }


def _lookup_landmark(
    landmarks: Sequence[object] | Mapping[str | int, object] | None,
    name_or_index: str | int,
) -> object | None:
    if landmarks is None:
        return None
    if isinstance(landmarks, Mapping):
        if name_or_index in landmarks:
            return landmarks[name_or_index]
        if isinstance(name_or_index, str):
            index = LANDMARK_INDEX.get(name_or_index)
            if index is not None:
                return landmarks.get(index)
        return None

    if isinstance(name_or_index, str):
        for point in landmarks:
            if getattr(point, "name", None) == name_or_index:
                return point
    index = LANDMARK_INDEX[name_or_index] if isinstance(name_or_index, str) else int(name_or_index)
    if index < 0 or index >= len(landmarks):
        return None
    return landmarks[index]


def _scale_axis(value: float, scale: float) -> float:
    if not isfinite(value):
        return value
    if -1.5 <= value <= 1.5:
        return value * scale
    return value


def _scaled_point(
    landmarks: Sequence[object] | Mapping[str | int, object] | None,
    name_or_index: str | int,
    image_width: int,
    image_height: int,
) -> PosePoint | None:
    point = coerce_point(
        _lookup_landmark(landmarks, name_or_index),
        min_visibility=0.2,
        min_presence=0.2,
    )
    if point is None:
        return None

    width = max(1.0, float(image_width))
    height = max(1.0, float(image_height))
    depth_scale = (width + height) / 2.0
    return PosePoint(
        x=_scale_axis(point.x, width),
        y=_scale_axis(point.y, height),
        z=_scale_axis(point.z, depth_scale),
        visibility=point.visibility,
        presence=point.presence,
    )


def _line_angle_degrees(start: PosePoint | None, end: PosePoint | None) -> FeatureValue:
    if start is None or end is None:
        return None
    dx = end.x - start.x
    dy = end.y - start.y
    if abs(dx) <= 1e-8 and abs(dy) <= 1e-8:
        return None
    return degrees(atan2(dy, dx))


def _torso_angle_degrees(hip_center: PosePoint | None, shoulder_center: PosePoint | None) -> FeatureValue:
    if hip_center is None or shoulder_center is None:
        return None
    dx = shoulder_center.x - hip_center.x
    dy = shoulder_center.y - hip_center.y
    if abs(dx) <= 1e-8 and abs(dy) <= 1e-8:
        return None
    return degrees(atan2(dx, -dy))


def _min_value(*values: FeatureValue) -> FeatureValue:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return min(filtered)


def _visibility_score(landmarks: Sequence[object] | Mapping[str | int, object] | None) -> float:
    scores: list[float] = []
    for name in HYROX_CORE_LANDMARKS:
        point = coerce_point(
            _lookup_landmark(landmarks, name),
            min_visibility=0.0,
            min_presence=0.0,
        )
        if point is None:
            scores.append(0.0)
            continue
        score = min(point.visibility, point.presence)
        scores.append(max(0.0, min(1.0, float(score))))
    return sum(scores) / len(scores) if scores else 0.0


def extract_basic_pose_features(
    landmarks: Sequence[object] | Mapping[str | int, object] | None,
    image_width: int,
    image_height: int,
) -> dict[str, FeatureValue]:
    features = _empty_features()
    if landmarks is None:
        return features

    left_shoulder = _scaled_point(landmarks, "left_shoulder", image_width, image_height)
    right_shoulder = _scaled_point(landmarks, "right_shoulder", image_width, image_height)
    left_hip = _scaled_point(landmarks, "left_hip", image_width, image_height)
    right_hip = _scaled_point(landmarks, "right_hip", image_width, image_height)
    left_knee = _scaled_point(landmarks, "left_knee", image_width, image_height)
    right_knee = _scaled_point(landmarks, "right_knee", image_width, image_height)
    left_ankle = _scaled_point(landmarks, "left_ankle", image_width, image_height)
    right_ankle = _scaled_point(landmarks, "right_ankle", image_width, image_height)

    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)

    features["left_knee_angle"] = angle_3pts(left_hip, left_knee, left_ankle)
    features["right_knee_angle"] = angle_3pts(right_hip, right_knee, right_ankle)
    features["left_hip_angle"] = angle_3pts(left_shoulder, left_hip, left_knee)
    features["right_hip_angle"] = angle_3pts(right_shoulder, right_hip, right_knee)
    features["torso_angle"] = _torso_angle_degrees(hip_center, shoulder_center)
    features["shoulder_tilt"] = _line_angle_degrees(left_shoulder, right_shoulder)
    features["hip_tilt"] = _line_angle_degrees(left_hip, right_hip)
    features["min_knee_angle"] = _min_value(features["left_knee_angle"], features["right_knee_angle"])
    features["min_hip_angle"] = _min_value(features["left_hip_angle"], features["right_hip_angle"])
    features["hip_center_y"] = None if hip_center is None else hip_center.y / max(1.0, float(image_height))
    knee_center = midpoint(left_knee, right_knee)
    features["knee_center_y"] = None if knee_center is None else knee_center.y / max(1.0, float(image_height))
    features["visible_score"] = _visibility_score(landmarks)
    return features


__all__ = ["extract_basic_pose_features"]
