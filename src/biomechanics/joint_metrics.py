"""Unified 2D/3D joint-angle metrics.

This module is the single implementation point for ordinary three-point joint
angles.  Product rules can keep consuming their established 2D feature stream
while validation and future rule promotion use :class:`JointMetric`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, isfinite, sqrt
from typing import Literal, Mapping, Sequence


JointMetricSource = Literal["3D", "2D", "UNAVAILABLE"]


ANGLE_DEFINITIONS: Mapping[str, tuple[str, str, str]] = {
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_ankle_angle": ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle_angle": ("right_knee", "right_ankle", "right_foot_index"),
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
}


_SAGITTAL_ANGLE_TOKENS = ("knee", "hip", "ankle", "elbow")
_SIDE_VIEWS = {
    "side",
    "left_side",
    "right_side",
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
    "oblique",
}


@dataclass(frozen=True, slots=True)
class JointMetric:
    """Raw/smoothed 2D and 3D values plus a reliability-aware selection."""

    raw_2d: float | None
    smooth_2d: float | None
    raw_3d: float | None
    smooth_3d: float | None
    selected_value: float | None
    source: JointMetricSource
    confidence: float
    observable: bool
    three_d_reliable: bool
    difference_deg: float | None = None
    quality_reasons: tuple[str, ...] = ()
    legacy_selected_source: str = "none"

    @property
    def angle_2d(self) -> float | None:
        """Backward-compatible name used by existing reports."""

        return self.smooth_2d

    @property
    def angle_3d(self) -> float | None:
        """Backward-compatible name used by existing reports."""

        return self.smooth_3d

    @property
    def selected_angle(self) -> float | None:
        """Formal legacy selection, intentionally still 2D until validation."""

        return self.smooth_2d

    @property
    def selected_source(self) -> str:
        """Formal legacy source, intentionally still 2D until validation."""

        return self.legacy_selected_source


def calculate_angle_2d(
    a: object | None,
    b: object | None,
    c: object | None,
    frame_width: int | float | None = None,
    frame_height: int | float | None = None,
) -> float | None:
    """Return the projected A-B-C angle, correcting normalized aspect ratio."""

    points = tuple(_coordinates(point, dimensions=2) for point in (a, b, c))
    if any(point is None for point in points):
        return None
    width = _positive_dimension(frame_width)
    height = _positive_dimension(frame_height)
    if (frame_width is not None or frame_height is not None) and (
        width is None or height is None
    ):
        return None
    normalized = (
        width is not None
        and height is not None
        and all(-1.5 <= value <= 1.5 for point in points if point for value in point)
    )
    scale_x = width if normalized and width is not None else 1.0
    scale_y = height if normalized and height is not None else 1.0
    point_a, vertex, point_c = points
    assert point_a is not None and vertex is not None and point_c is not None
    return vector_angle(
        ((point_a[0] - vertex[0]) * scale_x, (point_a[1] - vertex[1]) * scale_y),
        ((point_c[0] - vertex[0]) * scale_x, (point_c[1] - vertex[1]) * scale_y),
    )


def calculate_angle_3d(
    a: object | None,
    b: object | None,
    c: object | None,
) -> float | None:
    """Return the spatial A-B-C angle."""

    points = tuple(_coordinates(point, dimensions=3) for point in (a, b, c))
    if any(point is None for point in points):
        return None
    point_a, vertex, point_c = points
    assert point_a is not None and vertex is not None and point_c is not None
    return vector_angle(
        tuple(left - middle for left, middle in zip(point_a, vertex, strict=True)),
        tuple(right - middle for right, middle in zip(point_c, vertex, strict=True)),
    )


def select_joint_metric(
    *,
    name: str,
    raw_2d: float | None,
    smooth_2d: float | None,
    raw_3d: float | None,
    smooth_3d: float | None,
    three_d_reliable: bool,
    confidence: float,
    camera_view: str,
    decision_mode: str,
    quality_reasons: Sequence[str] = (),
) -> JointMetric:
    """Select reliable 3D, view-appropriate 2D, or unavailable.

    ``shadow`` remains an explicit compatibility mode.  In ``assist`` mode the
    unified metric prefers reliable 3D, but its legacy ``selected_angle``
    property stays 2D so existing HYROX thresholds are not silently changed.
    """

    reasons = set(str(reason) for reason in quality_reasons)
    two_d_observable = is_2d_angle_observable(name, camera_view)
    use_three_d = (
        decision_mode != "shadow"
        and three_d_reliable
        and smooth_3d is not None
    )
    if use_three_d:
        selected = smooth_3d
        source: JointMetricSource = "3D"
    elif smooth_2d is not None and two_d_observable:
        selected = smooth_2d
        source = "2D"
    else:
        selected = None
        source = "UNAVAILABLE"
        if smooth_2d is not None and not two_d_observable:
            reasons.add("camera_view_limited")
    resolved_confidence = max(0.0, min(1.0, float(confidence)))
    if source == "UNAVAILABLE":
        resolved_confidence = 0.0
    elif source == "2D" and not three_d_reliable:
        resolved_confidence *= 0.90
    difference = (
        abs(smooth_2d - smooth_3d)
        if smooth_2d is not None and smooth_3d is not None
        else None
    )
    legacy_source = (
        "2d_assist" if decision_mode == "assist" else "2d_shadow"
    ) if smooth_2d is not None else "none"
    return JointMetric(
        raw_2d=raw_2d,
        smooth_2d=smooth_2d,
        raw_3d=raw_3d,
        smooth_3d=smooth_3d,
        selected_value=selected,
        source=source,
        confidence=resolved_confidence,
        observable=source != "UNAVAILABLE",
        three_d_reliable=three_d_reliable,
        difference_deg=difference,
        quality_reasons=tuple(sorted(reasons)),
        legacy_selected_source=legacy_source,
    )


def is_2d_angle_observable(name: str, camera_view: str) -> bool:
    """Return whether a projected angle is appropriate for the declared view."""

    view = str(camera_view or "unknown").strip().lower().replace("-", "_")
    if view in {"", "unknown", "auto"}:
        return True
    if not any(token in name for token in _SAGITTAL_ANGLE_TOKENS):
        return True
    return view in _SIDE_VIEWS


def vector_angle(first: Sequence[float], second: Sequence[float]) -> float | None:
    if (
        len(first) != len(second)
        or not first
        or any(not isfinite(float(value)) for value in (*first, *second))
    ):
        return None
    norm_first = sqrt(sum(float(value) ** 2 for value in first))
    norm_second = sqrt(sum(float(value) ** 2 for value in second))
    if norm_first <= 1e-8 or norm_second <= 1e-8:
        return None
    dot = sum(
        float(left) * float(right)
        for left, right in zip(first, second, strict=True)
    )
    cosine = max(-1.0, min(1.0, dot / (norm_first * norm_second)))
    value = degrees(acos(cosine))
    return value if isfinite(value) else None


def _coordinates(point: object | None, *, dimensions: int) -> tuple[float, ...] | None:
    if point is None:
        return None
    try:
        if isinstance(point, Mapping):
            values = tuple(float(point.get(axis, 0.0)) for axis in ("x", "y", "z")[:dimensions])
        elif (
            isinstance(point, Sequence)
            or (hasattr(point, "__len__") and hasattr(point, "__getitem__"))
        ) and not isinstance(point, (str, bytes, bytearray)):
            values = tuple(float(point[index]) for index in range(dimensions))
        else:
            values = tuple(float(getattr(point, axis)) for axis in ("x", "y", "z")[:dimensions])
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None
    return values if all(isfinite(value) for value in values) else None


def _positive_dimension(value: int | float | None) -> float | None:
    try:
        numeric = float(value) if value is not None else 0.0
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if isfinite(numeric) and numeric > 0.0 else None


__all__ = [
    "ANGLE_DEFINITIONS",
    "JointMetric",
    "JointMetricSource",
    "calculate_angle_2d",
    "calculate_angle_3d",
    "is_2d_angle_observable",
    "select_joint_metric",
    "vector_angle",
]
