from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import atan2, degrees, isfinite
from typing import Any

import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.biomechanics.kinematics_3d import (
    ANGLE_DEFINITIONS_3D,
    calculate_angle_2d,
    calculate_angle_3d,
)


@dataclass(frozen=True)
class AngleObservation:
    frame_index: int
    timestamp_ms: float
    joint_name: str
    side: str
    angle_2d_raw_deg: float | None
    angle_2d_smoothed_deg: float | None
    angle_3d_raw_deg: float | None
    angle_3d_smoothed_deg: float | None
    display_angle_deg: float | None
    rule_angle_deg: float | None
    drawn_landmarks_angle_deg: float | None
    display_angle_source: str
    rule_angle_source: str
    drawn_landmarks_source: str
    landmark_visibility: float
    geometry_valid: bool
    display_drawn_difference_deg: float | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, float) and isfinite(value):
                payload[key] = round(value, 4)
        return payload


def trace_angle_sources(
    *,
    frame_index: int,
    timestamp_ms: float,
    raw_result: PoseResult,
    smoothed_result: PoseResult,
    image_width: int,
    image_height: int,
    rule_features: Mapping[str, Any] | None,
    assessment: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    raw_image = _point_map(raw_result.keypoints)
    smooth_image = _point_map(smoothed_result.keypoints)
    raw_world = _world_point_map(raw_result)
    smooth_world = _world_point_map(smoothed_result)
    display_by_key = {
        str(item.get("key")): item
        for item in (assessment or {}).get("angles", ())
        if isinstance(item, Mapping)
    }
    observations: list[dict[str, Any]] = []
    for key, names in ANGLE_DEFINITIONS_3D.items():
        raw_2d = _image_angle(
            raw_image,
            names,
            image_width=image_width,
            image_height=image_height,
        )
        smooth_2d = _image_angle(
            smooth_image,
            names,
            image_width=image_width,
            image_height=image_height,
        )
        raw_3d = _world_angle(raw_world, names)
        smooth_3d = _world_angle(smooth_world, names)
        display_item = display_by_key.get(key)
        display_angle = _finite_number(
            display_item.get("value") if display_item is not None else None
        )
        rule_angle = _finite_number(
            (rule_features or {}).get(key)
        )
        visibility = _triplet_visibility(smooth_image, names)
        side, joint = _split_key(key)
        difference = (
            abs(display_angle - smooth_2d)
            if display_angle is not None and smooth_2d is not None
            else None
        )
        observation = AngleObservation(
            frame_index=int(frame_index),
            timestamp_ms=float(timestamp_ms),
            joint_name=joint,
            side=side,
            angle_2d_raw_deg=raw_2d,
            angle_2d_smoothed_deg=smooth_2d,
            angle_3d_raw_deg=raw_3d,
            angle_3d_smoothed_deg=smooth_3d,
            display_angle_deg=display_angle,
            rule_angle_deg=rule_angle,
            drawn_landmarks_angle_deg=smooth_2d,
            display_angle_source=(
                str(display_item.get("display_angle_source"))
                if display_item is not None
                else "not_displayed"
            ),
            rule_angle_source="image_landmarks_analysis_smoothed",
            drawn_landmarks_source="image_landmarks_analysis_smoothed",
            landmark_visibility=visibility,
            geometry_valid=bool(
                smooth_2d is not None and rule_angle is not None
            ),
            display_drawn_difference_deg=difference,
        )
        observations.append(observation.as_dict())
    torso_raw_2d = _torso_angle_2d(
        raw_image,
        image_width=image_width,
        image_height=image_height,
    )
    torso_smooth_2d = _torso_angle_2d(
        smooth_image,
        image_width=image_width,
        image_height=image_height,
    )
    torso_raw_3d = _torso_angle_3d(raw_world)
    torso_smooth_3d = _torso_angle_3d(smooth_world)
    torso_rule = _finite_number((rule_features or {}).get("torso_angle"))
    torso_observation = AngleObservation(
        frame_index=int(frame_index),
        timestamp_ms=float(timestamp_ms),
        joint_name="torso",
        side="center",
        angle_2d_raw_deg=torso_raw_2d,
        angle_2d_smoothed_deg=torso_smooth_2d,
        angle_3d_raw_deg=torso_raw_3d,
        angle_3d_smoothed_deg=torso_smooth_3d,
        display_angle_deg=None,
        rule_angle_deg=torso_rule,
        drawn_landmarks_angle_deg=torso_smooth_2d,
        display_angle_source="not_displayed",
        rule_angle_source="image_landmarks_analysis_smoothed",
        drawn_landmarks_source="image_landmarks_analysis_smoothed",
        landmark_visibility=_torso_visibility(smooth_image),
        geometry_valid=bool(
            torso_smooth_2d is not None and torso_rule is not None
        ),
        display_drawn_difference_deg=None,
    )
    observations.append(torso_observation.as_dict())
    return observations


def angle_source_summary(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    displayed = [
        item
        for item in observations
        if item.get("display_angle_deg") is not None
    ]
    mismatches = [
        item
        for item in displayed
        if (
            _finite_number(item.get("display_drawn_difference_deg"))
            or 0.0
        )
        > 0.5
    ]
    return {
        "display_angle_source": sorted(
            {
                str(item.get("display_angle_source"))
                for item in displayed
            }
        ),
        "rule_angle_source": "image_landmarks_analysis_smoothed",
        "screen_skeleton_source": "image_landmarks_analysis_smoothed",
        "displayed_angle_count": len(displayed),
        "display_skeleton_mismatch_count": len(mismatches),
        "display_skeleton_tolerance_deg": 0.5,
    }


def _point_map(points: Sequence[Keypoint]) -> dict[str, Keypoint]:
    return {point.name: point for point in points}


def _world_point_map(result: PoseResult) -> dict[str, Keypoint]:
    value = result.extra.get("world_keypoints")
    return _point_map(value if isinstance(value, (list, tuple)) else ())


def _image_angle(
    points: Mapping[str, Keypoint],
    names: tuple[str, str, str],
    *,
    image_width: int,
    image_height: int,
) -> float | None:
    triplet = [points.get(name) for name in names]
    if any(point is None for point in triplet):
        return None
    arrays = []
    for point in triplet:
        assert point is not None
        array = np.asarray(
            (
                _pixel_coordinate(point.x, image_width),
                _pixel_coordinate(point.y, image_height),
            ),
            dtype=float,
        )
        arrays.append(array)
    return calculate_angle_2d(arrays[0], arrays[1], arrays[2])


def _world_angle(
    points: Mapping[str, Keypoint],
    names: tuple[str, str, str],
) -> float | None:
    triplet = [points.get(name) for name in names]
    if any(point is None for point in triplet):
        return None
    arrays = []
    for point in triplet:
        assert point is not None
        arrays.append(np.asarray((point.x, point.y, point.z), dtype=float))
    return calculate_angle_3d(arrays[0], arrays[1], arrays[2])


def _pixel_coordinate(value: float, size: int) -> float:
    return float(value) * float(size) if -1.5 <= value <= 1.5 else float(value)


def _triplet_visibility(
    points: Mapping[str, Keypoint],
    names: tuple[str, str, str],
) -> float:
    values: list[float] = []
    for name in names:
        point = points.get(name)
        if point is None:
            return 0.0
        visibility = (
            point.confidence
            if point.visibility is None
            else float(point.visibility)
        )
        presence = (
            point.confidence
            if point.presence is None
            else float(point.presence)
        )
        values.append(min(visibility, presence))
    return max(0.0, min(1.0, min(values)))


def _torso_angle_2d(
    points: Mapping[str, Keypoint],
    *,
    image_width: int,
    image_height: int,
) -> float | None:
    shoulders = _midpoint(
        points.get("left_shoulder"),
        points.get("right_shoulder"),
    )
    hips = _midpoint(points.get("left_hip"), points.get("right_hip"))
    if shoulders is None or hips is None:
        return None
    dx = (shoulders[0] - hips[0]) * image_width
    dy = (shoulders[1] - hips[1]) * image_height
    if abs(dx) <= 1e-8 and abs(dy) <= 1e-8:
        return None
    return degrees(atan2(dx, -dy))


def _torso_angle_3d(
    points: Mapping[str, Keypoint],
) -> float | None:
    shoulders = _midpoint_3d(
        points.get("left_shoulder"),
        points.get("right_shoulder"),
    )
    hips = _midpoint_3d(
        points.get("left_hip"),
        points.get("right_hip"),
    )
    if shoulders is None or hips is None:
        return None
    vertical_reference = hips + np.asarray((0.0, -1.0, 0.0))
    return calculate_angle_3d(shoulders, hips, vertical_reference)


def _midpoint(
    first: Keypoint | None,
    second: Keypoint | None,
) -> tuple[float, float] | None:
    if first is None or second is None:
        return None
    values = (first.x, first.y, second.x, second.y)
    if not all(isfinite(float(value)) for value in values):
        return None
    return (
        (float(first.x) + float(second.x)) / 2.0,
        (float(first.y) + float(second.y)) / 2.0,
    )


def _midpoint_3d(
    first: Keypoint | None,
    second: Keypoint | None,
) -> np.ndarray | None:
    if first is None or second is None:
        return None
    values = (first.x, first.y, first.z, second.x, second.y, second.z)
    if not all(isfinite(float(value)) for value in values):
        return None
    return np.asarray(
        (
            (float(first.x) + float(second.x)) / 2.0,
            (float(first.y) + float(second.y)) / 2.0,
            (float(first.z) + float(second.z)) / 2.0,
        ),
        dtype=float,
    )


def _torso_visibility(points: Mapping[str, Keypoint]) -> float:
    values: list[float] = []
    for name in (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ):
        point = points.get(name)
        if point is None:
            return 0.0
        values.append(
            min(
                point.confidence
                if point.visibility is None
                else float(point.visibility),
                point.confidence
                if point.presence is None
                else float(point.presence),
            )
        )
    return max(0.0, min(1.0, min(values)))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and isfinite(float(value)):
        return float(value)
    return None


def _split_key(key: str) -> tuple[str, str]:
    if key.startswith("left_"):
        return "left", key.removeprefix("left_").removesuffix("_angle")
    if key.startswith("right_"):
        return "right", key.removeprefix("right_").removesuffix("_angle")
    return "center", key.removesuffix("_angle")


__all__ = [
    "AngleObservation",
    "angle_source_summary",
    "trace_angle_sources",
]
