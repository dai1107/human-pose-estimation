from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import atan2, degrees, hypot, isfinite
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
        "left_elbow_angle": None,
        "right_elbow_angle": None,
        "left_shoulder_angle": None,
        "right_shoulder_angle": None,
        "torso_angle": None,
        "shoulder_tilt": None,
        "hip_tilt": None,
        "body_center_x": None,
        "body_center_y": None,
        "body_height_norm": None,
        "left_wrist_y": None,
        "right_wrist_y": None,
        "left_ankle_y": None,
        "right_ankle_y": None,
        "left_wrist_to_hip_y": None,
        "right_wrist_to_hip_y": None,
        "left_wrist_to_shoulder_y": None,
        "right_wrist_to_shoulder_y": None,
        "wrist_distance_norm": None,
        "ankle_distance_norm": None,
        "min_knee_angle": None,
        "min_hip_angle": None,
        "max_hip_angle": None,
        "hip_center_y": None,
        "knee_center_y": None,
        "shoulder_center_y": None,
        "wrist_center_y": None,
        "hip_knee_depth": None,
        "wrist_above_shoulder": None,
        "hip_width": None,
        "knee_width": None,
        "ankle_width": None,
        "min_elbow_angle": None,
        "max_elbow_angle": None,
        "left_wrist_above_shoulder": None,
        "right_wrist_above_shoulder": None,
        "visible_score": 0.0,
        "upper_body_visible_score": 0.0,
        "lower_body_visible_score": 0.0,
        "left_side_visible_score": 0.0,
        "right_side_visible_score": 0.0,
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


def _max_value(*values: FeatureValue) -> FeatureValue:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _visibility_score(
    landmarks: Sequence[object] | Mapping[str | int, object] | None,
    names: Sequence[str] = HYROX_CORE_LANDMARKS,
) -> float:
    scores: list[float] = []
    for name in names:
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


def _normalized_y(point: PosePoint | None, height: float) -> FeatureValue:
    return None if point is None else point.y / height


def _vertical_delta(start: PosePoint | None, end: PosePoint | None, height: float) -> FeatureValue:
    """Return signed screen-y delta: positive means ``end`` is lower."""
    if start is None or end is None:
        return None
    return (end.y - start.y) / height


def _normalized_distance(point_a: PosePoint | None, point_b: PosePoint | None, width: float, height: float) -> FeatureValue:
    if point_a is None or point_b is None:
        return None
    return hypot((point_b.x - point_a.x) / width, (point_b.y - point_a.y) / height)


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
    left_elbow = _scaled_point(landmarks, "left_elbow", image_width, image_height)
    right_elbow = _scaled_point(landmarks, "right_elbow", image_width, image_height)
    left_wrist = _scaled_point(landmarks, "left_wrist", image_width, image_height)
    right_wrist = _scaled_point(landmarks, "right_wrist", image_width, image_height)

    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    body_center = midpoint(shoulder_center, hip_center)
    width = max(1.0, float(image_width))
    height = max(1.0, float(image_height))

    features["left_knee_angle"] = angle_3pts(left_hip, left_knee, left_ankle)
    features["right_knee_angle"] = angle_3pts(right_hip, right_knee, right_ankle)
    features["left_hip_angle"] = angle_3pts(left_shoulder, left_hip, left_knee)
    features["right_hip_angle"] = angle_3pts(right_shoulder, right_hip, right_knee)
    features["left_elbow_angle"] = angle_3pts(left_shoulder, left_elbow, left_wrist)
    features["right_elbow_angle"] = angle_3pts(right_shoulder, right_elbow, right_wrist)
    features["left_shoulder_angle"] = angle_3pts(left_elbow, left_shoulder, left_hip)
    features["right_shoulder_angle"] = angle_3pts(right_elbow, right_shoulder, right_hip)
    features["torso_angle"] = _torso_angle_degrees(hip_center, shoulder_center)
    features["shoulder_tilt"] = _vertical_delta(left_shoulder, right_shoulder, height)
    features["hip_tilt"] = _vertical_delta(left_hip, right_hip, height)
    features["body_center_x"] = None if body_center is None else body_center.x / width
    features["body_center_y"] = _normalized_y(body_center, height)
    if shoulder_center is not None and left_ankle is not None and right_ankle is not None:
        ankle_center = midpoint(left_ankle, right_ankle)
        features["body_height_norm"] = (
            None if ankle_center is None else abs(ankle_center.y - shoulder_center.y) / height
        )
    features["left_wrist_y"] = _normalized_y(left_wrist, height)
    features["right_wrist_y"] = _normalized_y(right_wrist, height)
    features["left_ankle_y"] = _normalized_y(left_ankle, height)
    features["right_ankle_y"] = _normalized_y(right_ankle, height)
    features["left_wrist_to_hip_y"] = _vertical_delta(left_hip, left_wrist, height)
    features["right_wrist_to_hip_y"] = _vertical_delta(right_hip, right_wrist, height)
    features["left_wrist_to_shoulder_y"] = _vertical_delta(left_shoulder, left_wrist, height)
    features["right_wrist_to_shoulder_y"] = _vertical_delta(right_shoulder, right_wrist, height)
    features["wrist_distance_norm"] = _normalized_distance(left_wrist, right_wrist, width, height)
    features["ankle_distance_norm"] = _normalized_distance(left_ankle, right_ankle, width, height)
    features["min_knee_angle"] = _min_value(features["left_knee_angle"], features["right_knee_angle"])
    features["min_hip_angle"] = _min_value(features["left_hip_angle"], features["right_hip_angle"])
    features["max_hip_angle"] = _max_value(features["left_hip_angle"], features["right_hip_angle"])
    features["min_elbow_angle"] = _min_value(features["left_elbow_angle"], features["right_elbow_angle"])
    features["max_elbow_angle"] = _max_value(features["left_elbow_angle"], features["right_elbow_angle"])
    features["hip_center_y"] = _normalized_y(hip_center, height)
    knee_center = midpoint(left_knee, right_knee)
    features["knee_center_y"] = None if knee_center is None else knee_center.y / max(1.0, float(image_height))
    wrist_center = midpoint(left_wrist, right_wrist)
    features["shoulder_center_y"] = None if shoulder_center is None else shoulder_center.y / max(1.0, float(image_height))
    features["wrist_center_y"] = None if wrist_center is None else wrist_center.y / max(1.0, float(image_height))
    if features["hip_center_y"] is not None and features["knee_center_y"] is not None:
        features["hip_knee_depth"] = features["hip_center_y"] - features["knee_center_y"]
    if features["shoulder_center_y"] is not None and features["wrist_center_y"] is not None:
        features["wrist_above_shoulder"] = features["shoulder_center_y"] - features["wrist_center_y"]
    if left_shoulder is not None and left_wrist is not None:
        features["left_wrist_above_shoulder"] = (left_shoulder.y - left_wrist.y) / max(1.0, float(image_height))
    if right_shoulder is not None and right_wrist is not None:
        features["right_wrist_above_shoulder"] = (right_shoulder.y - right_wrist.y) / max(1.0, float(image_height))
    if left_hip is not None and right_hip is not None:
        features["hip_width"] = abs(right_hip.x - left_hip.x) / width
    if left_knee is not None and right_knee is not None:
        features["knee_width"] = abs(right_knee.x - left_knee.x) / width
    if left_ankle is not None and right_ankle is not None:
        features["ankle_width"] = abs(right_ankle.x - left_ankle.x) / width
    features["visible_score"] = _visibility_score(landmarks)
    features["upper_body_visible_score"] = _visibility_score(
        landmarks,
        ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"),
    )
    features["lower_body_visible_score"] = _visibility_score(
        landmarks,
        ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"),
    )
    features["left_side_visible_score"] = _visibility_score(
        landmarks,
        ("left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle"),
    )
    features["right_side_visible_score"] = _visibility_score(
        landmarks,
        ("right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle"),
    )
    return features


__all__ = ["extract_basic_pose_features"]
