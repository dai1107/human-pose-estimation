"""Lightweight segment-coordinate biomechanics proxies for realtime validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import acos, degrees, isfinite
from typing import Any

import numpy as np


SEGMENT_ENDPOINTS: Mapping[str, tuple[str, str]] = {
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shank": ("left_knee", "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
}


@dataclass(frozen=True, slots=True)
class BiomechMetric:
    legacy_angle: float | None
    canonical_3d_angle: float | None
    biomech_angle: float | None
    confidence: float
    observable: bool
    definition: str
    quality_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "legacy_angle": self.legacy_angle,
            "canonical_3d_angle": self.canonical_3d_angle,
            "biomech_angle": self.biomech_angle,
            "confidence": self.confidence,
            "observable": self.observable,
            "angle_definition": self.definition,
            "quality_reasons": list(self.quality_reasons),
            "validation_only": True,
            "formal_threshold_replacement_allowed": False,
        }


def build_biomechanical_representation(
    world_points: Sequence[object],
    *,
    quality_points: Sequence[object],
    body_coordinate_system: Mapping[str, Any],
    local_ground_frame: Mapping[str, Any],
    measurements: Mapping[str, object],
    minimum_quality: float = 0.60,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    points = _point_map(world_points)
    arrays = {name: array for name, point in points.items() if (array := _xyz(point)) is not None}
    quality = _point_map(quality_points)
    reference_axes = _reference_axes(local_ground_frame, body_coordinate_system)
    segment_coordinates: dict[str, Any] = {}
    pelvis = _pelvis_coordinate(arrays, quality, reference_axes, minimum_quality)
    torso = _torso_coordinate(arrays, quality, reference_axes, minimum_quality)
    segment_coordinates["pelvis"] = pelvis
    segment_coordinates["torso"] = torso
    for name, endpoints in SEGMENT_ENDPOINTS.items():
        segment_coordinates[name] = _limb_coordinate(
            name,
            endpoints,
            arrays,
            quality,
            reference_axes,
            minimum_quality,
        )
    relative_rotations: dict[str, Any] = {}
    for name, parent_name, child_name in (
        ("left_hip", "torso", "left_thigh"),
        ("right_hip", "torso", "right_thigh"),
        ("left_knee", "left_thigh", "left_shank"),
        ("right_knee", "right_thigh", "right_shank"),
        ("left_elbow", "left_upper_arm", "left_forearm"),
        ("right_elbow", "right_upper_arm", "right_forearm"),
        ("trunk_to_pelvis", "pelvis", "torso"),
    ):
        relative_rotations[name] = _relative_rotation(
            segment_coordinates[parent_name],
            segment_coordinates[child_name],
        )
    segment_payload = {
        "schema_version": 1,
        "coordinate_convention": (
            "segment_y=proximal_to_distal_for_limbs;"
            "segment_x=reference_left_right_projected_normal_to_y;z=cross(x,y)"
        ),
        "segments": segment_coordinates,
        "relative_rotations": relative_rotations,
        "validation_only": True,
        "formal_threshold_replacement_allowed": False,
    }
    metrics = _build_metrics(
        segment_coordinates,
        relative_rotations,
        measurements,
        body_coordinate_system,
        local_ground_frame,
        minimum_quality,
    )
    return segment_payload, {name: metric.as_dict() for name, metric in metrics.items()}


def _build_metrics(
    segments: Mapping[str, Mapping[str, Any]],
    relative: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, object],
    body: Mapping[str, Any],
    ground: Mapping[str, Any],
    minimum_quality: float,
) -> dict[str, BiomechMetric]:
    result: dict[str, BiomechMetric] = {}
    canonical = body.get("canonical_3d_angles")
    canonical = canonical if isinstance(canonical, Mapping) else {}
    for side in ("left", "right"):
        knee_name = f"{side}_knee_angle"
        knee_rotation = relative[f"{side}_knee"]
        knee_value = _rotation_magnitude(knee_rotation)
        result[f"{side}_knee_flexion_proxy"] = _metric(
            measurements.get(knee_name),
            canonical.get(knee_name),
            knee_value,
            _confidence(knee_rotation),
            minimum_quality,
            "angle between thigh and shank longitudinal axes; 0 deg=extended",
        )
        hip_name = f"{side}_hip_angle"
        torso_y = _segment_axis(segments["torso"], "y")
        thigh_y = _segment_axis(segments[f"{side}_thigh"], "y")
        hip_value = _supplement_angle(torso_y, thigh_y)
        hip_confidence = min(
            _confidence(segments["torso"]),
            _confidence(segments[f"{side}_thigh"]),
        )
        result[f"{side}_hip_flexion_proxy"] = _metric(
            measurements.get(hip_name),
            canonical.get(hip_name),
            hip_value,
            hip_confidence,
            minimum_quality,
            "supplement of torso-up versus thigh proximal-to-distal angle; 0 deg=neutral extension",
        )
    ground_y = _ground_axis(ground, "y_estimated_vertical")
    torso_y = _segment_axis(segments["torso"], "y")
    trunk_value = _angle(torso_y, ground_y)
    trunk_confidence = min(_confidence(segments["torso"]), _confidence(ground))
    result["trunk_flexion"] = _metric(
        None,
        None,
        trunk_value,
        trunk_confidence,
        minimum_quality,
        "unsigned angle between torso longitudinal axis and estimated local vertical",
    )
    ground_x = _ground_axis(ground, "x_body_left_to_right")
    pelvis_x = _segment_axis(segments["pelvis"], "x")
    result["pelvis_orientation"] = _metric(
        None,
        None,
        _angle(pelvis_x, ground_x),
        min(_confidence(segments["pelvis"]), _confidence(ground)),
        minimum_quality,
        "unsigned pelvis left-right axis orientation relative to local ground x",
    )
    trunk_rotation = relative["trunk_to_pelvis"]
    result["trunk_rotation"] = _metric(
        None,
        None,
        _yaw_deg(trunk_rotation),
        _confidence(trunk_rotation),
        minimum_quality,
        "relative torso-to-pelvis rotation about pelvis local vertical",
    )
    return result


def _metric(
    measurement: object,
    canonical: object,
    biomech: float | None,
    confidence: float,
    minimum_quality: float,
    definition: str,
) -> BiomechMetric:
    legacy = getattr(measurement, "selected_angle", None)
    legacy = _finite_or_none(legacy)
    canonical_value = _finite_or_none(canonical)
    biomech_value = _finite_or_none(biomech)
    reasons = []
    if biomech_value is None:
        reasons.append("biomech_geometry_unavailable")
    if confidence < minimum_quality:
        reasons.append("biomech_confidence_low")
    observable = biomech_value is not None and confidence >= minimum_quality
    return BiomechMetric(
        legacy_angle=legacy,
        canonical_3d_angle=canonical_value,
        biomech_angle=biomech_value,
        confidence=max(0.0, min(1.0, confidence)),
        observable=observable,
        definition=definition,
        quality_reasons=tuple(reasons),
    )


def _pelvis_coordinate(arrays, quality, reference, minimum):
    left, right = arrays.get("left_hip"), arrays.get("right_hip")
    origin = None if left is None or right is None else (left + right) / 2.0
    x_axis = None if left is None or right is None else _unit(right - left)
    y_axis = reference.get("y")
    return _coordinate("pelvis", origin, x_axis, y_axis, quality, ("left_hip", "right_hip"), minimum)


def _torso_coordinate(arrays, quality, reference, minimum):
    hips = _center(arrays, "left_hip", "right_hip")
    shoulders = _center(arrays, "left_shoulder", "right_shoulder")
    y_axis = None if hips is None or shoulders is None else _unit(shoulders - hips)
    x_axis = None
    if arrays.get("left_shoulder") is not None and arrays.get("right_shoulder") is not None:
        x_axis = _unit(arrays["right_shoulder"] - arrays["left_shoulder"])
    return _coordinate("torso", hips, x_axis, y_axis, quality, ("left_hip", "right_hip", "left_shoulder", "right_shoulder"), minimum)


def _limb_coordinate(name, endpoints, arrays, quality, reference, minimum):
    start, end = (arrays.get(item) for item in endpoints)
    y_axis = None if start is None or end is None else _unit(end - start)
    x_reference = reference.get("x")
    x_axis = None if y_axis is None or x_reference is None else _unit(x_reference - y_axis * float(np.dot(x_reference, y_axis)))
    return _coordinate(name, start, x_axis, y_axis, quality, endpoints, minimum)


def _coordinate(name, origin, x_axis, y_axis, quality, names, minimum):
    reasons = []
    z_axis = None
    if x_axis is not None and y_axis is not None:
        z_axis = _unit(np.cross(x_axis, y_axis))
        if z_axis is not None:
            x_axis = _unit(np.cross(y_axis, z_axis))
    if origin is None or x_axis is None or y_axis is None or z_axis is None:
        reasons.append("segment_coordinate_degenerate")
    point_quality = min((_quality(quality.get(item)) for item in names), default=0.0)
    if point_quality < minimum:
        reasons.append("segment_quality_low")
    available = not any(value is None for value in (origin, x_axis, y_axis, z_axis))
    return {
        "name": name,
        "available": available,
        "reliable": available and not reasons,
        "confidence": point_quality if available else 0.0,
        "origin_world_relative": _tuple(origin),
        "axes_world_relative": {"x": _tuple(x_axis), "y": _tuple(y_axis), "z": _tuple(z_axis)} if available else {},
        "quality_reasons": reasons,
    }


def _relative_rotation(parent, child):
    parent_matrix = _matrix(parent)
    child_matrix = _matrix(child)
    confidence = min(_confidence(parent), _confidence(child))
    if parent_matrix is None or child_matrix is None:
        return {"available": False, "confidence": 0.0, "matrix": [], "rotation_magnitude_deg": None, "euler_xyz_deg": {}}
    rotation = parent_matrix @ child_matrix.T
    magnitude = degrees(acos(max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))))
    euler = _euler_xyz(rotation)
    return {
        "available": True,
        "confidence": confidence,
        "matrix": [[float(value) for value in row] for row in rotation],
        "rotation_magnitude_deg": magnitude,
        "euler_xyz_deg": euler,
    }


def _reference_axes(ground, body):
    axes = ground.get("axes_world_relative") if bool(ground.get("reliable")) else None
    if isinstance(axes, Mapping):
        x_axis = _array(axes.get("x_body_left_to_right"))
        y_axis = _array(axes.get("y_estimated_vertical"))
    else:
        body_axes = body.get("axes_world")
        body_axes = body_axes if isinstance(body_axes, Mapping) else {}
        x_axis = _array(body_axes.get("x_left_to_right"))
        y_axis = _array(body_axes.get("y_up"))
    return {"x": _unit(x_axis) if x_axis is not None else None, "y": _unit(y_axis) if y_axis is not None else None}


def _matrix(segment):
    axes = segment.get("axes_world_relative") if isinstance(segment, Mapping) else None
    if not isinstance(axes, Mapping):
        return None
    values = [_array(axes.get(name)) for name in ("x", "y", "z")]
    return None if any(value is None for value in values) else np.stack(values, axis=0)


def _segment_axis(segment, name):
    axes = segment.get("axes_world_relative") if isinstance(segment, Mapping) else None
    return _array(axes.get(name)) if isinstance(axes, Mapping) else None


def _ground_axis(ground, name):
    axes = ground.get("axes_world_relative") if isinstance(ground, Mapping) else None
    return _array(axes.get(name)) if isinstance(axes, Mapping) else None


def _rotation_magnitude(rotation):
    return _finite_or_none(rotation.get("rotation_magnitude_deg")) if isinstance(rotation, Mapping) else None


def _yaw_deg(rotation):
    euler = rotation.get("euler_xyz_deg") if isinstance(rotation, Mapping) else None
    return _finite_or_none(euler.get("y")) if isinstance(euler, Mapping) else None


def _euler_xyz(matrix):
    sy = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(matrix[2, 1], matrix[2, 2])
        y = np.arctan2(-matrix[2, 0], sy)
        z = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x = np.arctan2(-matrix[1, 2], matrix[1, 1])
        y = np.arctan2(-matrix[2, 0], sy)
        z = 0.0
    return {"x": float(np.degrees(x)), "y": float(np.degrees(y)), "z": float(np.degrees(z))}


def _angle(first, second):
    first, second = _unit_or_none(first), _unit_or_none(second)
    if first is None or second is None:
        return None
    return degrees(acos(max(-1.0, min(1.0, float(np.dot(first, second))))))


def _supplement_angle(first, second):
    value = _angle(first, second)
    return None if value is None else 180.0 - value


def _unit_or_none(value):
    return None if value is None else _unit(value)


def _unit(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-8 else None


def _array(value):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result.shape == (3,) and np.all(np.isfinite(result)) else None


def _point_map(points):
    return {str(name): point for point in points if (name := (point.get("name") if isinstance(point, Mapping) else getattr(point, "name", None)))}


def _xyz(point):
    try:
        result = np.asarray([float(point.get(axis) if isinstance(point, Mapping) else getattr(point, axis)) for axis in ("x", "y", "z")], dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return result if np.all(np.isfinite(result)) else None


def _quality(point):
    if point is None:
        return 0.0
    values = []
    for name in ("confidence", "visibility", "presence"):
        raw = point.get(name) if isinstance(point, Mapping) else getattr(point, name, None)
        value = _finite_or_none(raw)
        if value is not None:
            values.append(value)
    return max(0.0, min(1.0, min(values, default=0.0)))


def _center(arrays, left, right):
    return None if arrays.get(left) is None or arrays.get(right) is None else (arrays[left] + arrays[right]) / 2.0


def _confidence(value):
    return max(0.0, min(1.0, _finite_or_none(value.get("confidence")) or 0.0)) if isinstance(value, Mapping) else 0.0


def _finite_or_none(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _tuple(value):
    return None if value is None else tuple(float(item) for item in value)


__all__ = ["BiomechMetric", "SEGMENT_ENDPOINTS", "build_biomechanical_representation"]
