"""Ground-aligned local coordinates derived from auxiliary realtime evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import numpy as np


SUPPORT_NAMES = (
    "left_heel",
    "left_foot_index",
    "left_ankle",
    "right_heel",
    "right_foot_index",
    "right_ankle",
)


def build_local_ground_frame(
    world_points: Sequence[object],
    *,
    body_coordinate_system: Mapping[str, Any],
    ground_estimation: Mapping[str, Any],
    foot_contact_evidence: Mapping[str, Any],
    minimum_confidence: float = 0.50,
) -> dict[str, Any]:
    """Fuse body left-right orientation with an estimated local support plane.

    MediaPipe world landmarks are body-relative.  The result is therefore
    deliberately named ``local_ground_frame`` and never presented as a true
    calibrated world/camera coordinate system.
    """

    points = _point_map(world_points)
    arrays = {
        name: value
        for name, point in points.items()
        if (value := _xyz(point)) is not None
    }
    reasons: set[str] = set()
    body_axes = body_coordinate_system.get("axes_world")
    body_x = _axis(body_axes, "x_left_to_right")
    body_y = _axis(body_axes, "y_up")
    if not bool(body_coordinate_system.get("reliable")) or body_x is None or body_y is None:
        reasons.add("body_coordinate_unreliable")

    contact_confidences = {
        side: _contact_confidence(foot_contact_evidence.get(side))
        for side in ("left", "right")
    }
    active_sides = [
        side for side, confidence in contact_confidences.items() if confidence >= 0.55
    ]
    support_names = [
        name
        for name in SUPPORT_NAMES
        if name.split("_", 1)[0] in active_sides and name in arrays
    ]
    # Both sides give enough horizontal spread for a meaningful support plane.
    if len(active_sides) < 2 or len(support_names) < 4:
        reasons.add("insufficient_contact_support")
    support = np.asarray([arrays[name] for name in support_names], dtype=float)
    origin: np.ndarray | None = None
    normal: np.ndarray | None = None
    residual: float | None = None
    spread_score = 0.0
    if len(support) >= 4:
        origin = np.mean(support, axis=0)
        centered = support - origin
        try:
            _, singular, vh = np.linalg.svd(centered, full_matrices=False)
            normal = _unit(vh[-1])
        except np.linalg.LinAlgError:
            normal = None
            singular = np.asarray([], dtype=float)
        if normal is None:
            reasons.add("support_plane_degenerate")
        else:
            if body_y is not None and float(np.dot(normal, body_y)) < 0.0:
                normal = -normal
            residual = float(np.sqrt(np.mean((centered @ normal) ** 2)))
            total_spread = float(np.linalg.norm(centered, axis=1).max(initial=0.0))
            spread_score = min(1.0, total_spread / 0.20)
            if len(singular) < 2 or float(singular[1]) <= 1e-5:
                reasons.add("support_plane_degenerate")

    x_axis: np.ndarray | None = None
    z_axis: np.ndarray | None = None
    if normal is not None and body_x is not None:
        x_axis = _unit(body_x - normal * float(np.dot(body_x, normal)))
        if x_axis is None:
            reasons.add("ground_left_right_axis_degenerate")
        else:
            z_axis = _unit(np.cross(x_axis, normal))
            if z_axis is None:
                reasons.add("ground_forward_axis_degenerate")
            else:
                x_axis = _unit(np.cross(normal, z_axis))

    ground_confidence = _number(
        ground_estimation.get(
            "camera_adjusted_ground_confidence",
            ground_estimation.get("ground_confidence", 0.0),
        ),
        0.0,
    )
    body_confidence = _number(body_coordinate_system.get("confidence"), 0.0)
    contact_confidence = min(contact_confidences.values(), default=0.0)
    residual_score = 0.0 if residual is None else max(0.0, 1.0 - residual / 0.04)
    confidence = max(
        0.0,
        min(
            1.0,
            body_confidence
            * ground_confidence
            * contact_confidence
            * residual_score
            * spread_score,
        ),
    )
    if ground_estimation.get("status") != "READY":
        reasons.add("ground_estimator_unready")
    if confidence < minimum_confidence:
        reasons.add("local_ground_confidence_low")
    available = all(value is not None for value in (origin, x_axis, normal, z_axis))
    reliable = available and not reasons and confidence >= minimum_confidence

    local_landmarks: list[dict[str, Any]] = []
    if available:
        assert origin is not None and x_axis is not None and normal is not None and z_axis is not None
        rotation = np.stack((x_axis, normal, z_axis), axis=0)
        for name, point in arrays.items():
            local = rotation @ (point - origin)
            local_landmarks.append(
                {
                    "name": name,
                    "x": float(local[0]),
                    "y": float(local[1]),
                    "z": float(local[2]),
                    "confidence": _point_quality(points[name]),
                }
            )
    return {
        "schema_version": 1,
        "name": "local_ground_frame",
        "coordinate_semantics": (
            "origin=current_contact_support_centroid;"
            "x=body_left_right_projected_on_support_plane;"
            "y=estimated_local_support_normal;z=forward"
        ),
        "available": available,
        "reliable": reliable,
        "confidence": confidence,
        "origin_world_relative": _tuple(origin),
        "axes_world_relative": {
            "x_body_left_to_right": _tuple(x_axis),
            "y_estimated_vertical": _tuple(normal),
            "z_forward": _tuple(z_axis),
        }
        if available
        else {},
        "support_landmarks": support_names,
        "active_contact_sides": active_sides,
        "support_plane_residual_m": residual,
        "landmarks": local_landmarks,
        "quality_reasons": sorted(reasons),
        "true_world_coordinate": False,
        "camera_extrinsics_recovered": False,
        "formal_threshold_replacement_allowed": False,
    }


def _point_map(points: Sequence[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for point in points:
        name = point.get("name") if isinstance(point, Mapping) else getattr(point, "name", None)
        if name:
            result[str(name)] = point
    return result


def _xyz(point: object) -> np.ndarray | None:
    try:
        values = np.asarray(
            [float(point.get(axis) if isinstance(point, Mapping) else getattr(point, axis)) for axis in ("x", "y", "z")],
            dtype=float,
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return values if values.shape == (3,) and np.all(np.isfinite(values)) else None


def _axis(axes: object, name: str) -> np.ndarray | None:
    if not isinstance(axes, Mapping):
        return None
    try:
        value = np.asarray(axes.get(name), dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    return _unit(value)


def _unit(value: np.ndarray) -> np.ndarray | None:
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-8 else None


def _contact_confidence(value: object) -> float:
    return max(
        0.0,
        min(
            1.0,
            _number(
                value.get("foot_contact_confidence", 0.0)
                if isinstance(value, Mapping)
                else 0.0,
                0.0,
            ),
        ),
    )


def _point_quality(point: object) -> float:
    values = []
    for name in ("confidence", "visibility", "presence"):
        raw = point.get(name) if isinstance(point, Mapping) else getattr(point, name, None)
        if raw is not None:
            values.append(_number(raw, 0.0))
    return max(0.0, min(1.0, min(values, default=0.0)))


def _number(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if isfinite(result) else default


def _tuple(value: np.ndarray | None) -> tuple[float, float, float] | None:
    return None if value is None else tuple(float(item) for item in value)


__all__ = ["build_local_ground_frame"]
