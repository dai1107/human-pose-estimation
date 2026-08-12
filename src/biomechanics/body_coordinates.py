"""Body-local orthonormal coordinates for MediaPipe world landmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .joint_metrics import ANGLE_DEFINITIONS, calculate_angle_3d


@dataclass(frozen=True, slots=True)
class BodyCoordinateResult:
    available: bool
    reliable: bool
    confidence: float
    origin_world: tuple[float, float, float] | None
    axes_world: Mapping[str, tuple[float, float, float]]
    canonical_landmarks: tuple[Mapping[str, Any], ...]
    canonical_3d_angles: Mapping[str, float | None]
    orthogonality_error: float | None
    quality_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "coordinate_semantics": (
                "origin=hip_midpoint;x=left_hip_to_right_hip;"
                "y=hip_midpoint_to_shoulder_midpoint;z=cross(x,y)"
            ),
            "available": self.available,
            "reliable": self.reliable,
            "confidence": self.confidence,
            "origin_world": self.origin_world,
            "axes_world": dict(self.axes_world),
            "canonical_landmarks": [dict(point) for point in self.canonical_landmarks],
            "canonical_3d_angles": dict(self.canonical_3d_angles),
            "orthogonality_error": self.orthogonality_error,
            "quality_reasons": list(self.quality_reasons),
            "formal_threshold_replacement_allowed": False,
        }


def build_body_coordinate_system(
    world_points: Sequence[object],
    *,
    quality_points: Sequence[object] | None = None,
    minimum_quality: float = 0.70,
) -> BodyCoordinateResult:
    """Rotate world landmarks into a hip-centred body coordinate system."""

    world = _point_map(world_points)
    quality = _point_map(quality_points or world_points)
    required = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    arrays = {name: _xyz(world.get(name)) for name in world}
    reasons: set[str] = set()
    if any(arrays.get(name) is None for name in required):
        reasons.add("body_coordinate_landmarks_missing")
        return _unavailable(reasons)

    left_hip = arrays["left_hip"]
    right_hip = arrays["right_hip"]
    left_shoulder = arrays["left_shoulder"]
    right_shoulder = arrays["right_shoulder"]
    assert all(
        point is not None
        for point in (left_hip, right_hip, left_shoulder, right_shoulder)
    )
    origin = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    x_axis = _unit(right_hip - left_hip)
    provisional_y = shoulder_center - origin
    if x_axis is None:
        reasons.add("body_left_right_axis_degenerate")
        return _unavailable(reasons)
    # Gram-Schmidt prevents hip/shoulder estimation noise from producing a
    # skewed coordinate frame.
    y_axis = _unit(provisional_y - x_axis * float(np.dot(provisional_y, x_axis)))
    if y_axis is None:
        reasons.add("body_up_axis_degenerate")
        return _unavailable(reasons)
    z_axis = _unit(np.cross(x_axis, y_axis))
    if z_axis is None:
        reasons.add("body_forward_axis_degenerate")
        return _unavailable(reasons)
    y_axis = _unit(np.cross(z_axis, x_axis))
    assert y_axis is not None

    required_quality = min(
        (_quality(quality.get(name)) for name in required),
        default=0.0,
    )
    if required_quality < minimum_quality:
        reasons.add("body_coordinate_low_quality")
    raw_x = right_hip - left_hip
    raw_y = provisional_y
    geometry_score = _axis_separation_score(raw_x, raw_y)
    confidence = max(0.0, min(1.0, required_quality * geometry_score))
    matrix = np.stack((x_axis, y_axis, z_axis), axis=0)
    orthogonality_error = float(
        np.max(np.abs(matrix @ matrix.T - np.eye(3, dtype=float)))
    )
    if orthogonality_error > 1e-5:
        reasons.add("body_coordinate_not_orthogonal")

    canonical_arrays: dict[str, np.ndarray] = {}
    canonical_points: list[Mapping[str, Any]] = []
    for name, world_point in world.items():
        array = arrays.get(name)
        if array is None:
            continue
        canonical = matrix @ (array - origin)
        canonical_arrays[name] = canonical
        canonical_points.append(
            {
                "name": name,
                "x": float(canonical[0]),
                "y": float(canonical[1]),
                "z": float(canonical[2]),
                "confidence": _quality(quality.get(name)),
            }
        )
    canonical_angles = {
        name: calculate_angle_3d(
            canonical_arrays.get(definition[0]),
            canonical_arrays.get(definition[1]),
            canonical_arrays.get(definition[2]),
        )
        for name, definition in ANGLE_DEFINITIONS.items()
    }
    reliable = not reasons and confidence >= minimum_quality
    return BodyCoordinateResult(
        available=True,
        reliable=reliable,
        confidence=confidence,
        origin_world=tuple(float(value) for value in origin),
        axes_world={
            "x_left_to_right": tuple(float(value) for value in x_axis),
            "y_up": tuple(float(value) for value in y_axis),
            "z_forward": tuple(float(value) for value in z_axis),
        },
        canonical_landmarks=tuple(canonical_points),
        canonical_3d_angles=canonical_angles,
        orthogonality_error=orthogonality_error,
        quality_reasons=tuple(sorted(reasons)),
    )


def _unavailable(reasons: set[str]) -> BodyCoordinateResult:
    return BodyCoordinateResult(
        available=False,
        reliable=False,
        confidence=0.0,
        origin_world=None,
        axes_world={},
        canonical_landmarks=(),
        canonical_3d_angles={},
        orthogonality_error=None,
        quality_reasons=tuple(sorted(reasons)),
    )


def _point_map(points: Sequence[object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for point in points:
        name = point.get("name") if isinstance(point, Mapping) else getattr(point, "name", None)
        if name:
            result[str(name)] = point
    return result


def _value(point: object, name: str) -> object:
    return point.get(name) if isinstance(point, Mapping) else getattr(point, name, None)


def _xyz(point: object | None) -> np.ndarray | None:
    if point is None:
        return None
    try:
        values = np.asarray(
            [float(_value(point, axis)) for axis in ("x", "y", "z")],
            dtype=float,
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return values if values.shape == (3,) and np.all(np.isfinite(values)) else None


def _quality(point: object | None) -> float:
    if point is None:
        return 0.0
    confidence = _value(point, "confidence")
    visibility = _value(point, "visibility")
    presence = _value(point, "presence")
    values: list[float] = []
    for raw in (
        confidence,
        confidence if visibility is None else visibility,
        confidence if presence is None else presence,
    ):
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        values.append(value if isfinite(value) else 0.0)
    return max(0.0, min(1.0, min(values, default=0.0)))


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return None if not isfinite(norm) or norm <= 1e-8 else vector / norm


def _axis_separation_score(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-8:
        return 0.0
    cosine = max(-1.0, min(1.0, float(np.dot(first, second) / denominator)))
    return float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))


__all__ = ["BodyCoordinateResult", "build_body_coordinate_system"]
