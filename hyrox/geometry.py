from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import acos, degrees, isfinite, sqrt
from typing import Any


DEFAULT_MIN_VISIBILITY = 0.2
DEFAULT_MIN_PRESENCE = 0.2


@dataclass(frozen=True)
class PosePoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0

    def is_usable(
        self,
        min_visibility: float = DEFAULT_MIN_VISIBILITY,
        min_presence: float = DEFAULT_MIN_PRESENCE,
    ) -> bool:
        return (
            isfinite(self.x)
            and isfinite(self.y)
            and isfinite(self.z)
            and self.visibility >= min_visibility
            and self.presence >= min_presence
        )


def _read_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sequence_value(point: Sequence[Any], index: int, default: float) -> float:
    if len(point) <= index:
        return default
    return _read_float(point[index], default)


def coerce_point(
    point: object | None,
    *,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    min_presence: float = DEFAULT_MIN_PRESENCE,
) -> PosePoint | None:
    if point is None:
        return None

    if isinstance(point, PosePoint):
        normalized = point
    elif isinstance(point, Mapping):
        confidence = _read_float(point.get("confidence"), 1.0)
        normalized = PosePoint(
            x=_read_float(point.get("x"), float("nan")),
            y=_read_float(point.get("y"), float("nan")),
            z=_read_float(point.get("z"), 0.0),
            visibility=_read_float(point.get("visibility"), confidence),
            presence=_read_float(point.get("presence"), confidence),
        )
    elif (
        isinstance(point, Sequence)
        or (
            hasattr(point, "__len__")
            and hasattr(point, "__getitem__")
        )
    ) and not isinstance(point, (str, bytes, bytearray)):
        normalized = PosePoint(
            x=_sequence_value(point, 0, float("nan")),
            y=_sequence_value(point, 1, float("nan")),
            z=_sequence_value(point, 2, 0.0),
            visibility=_sequence_value(point, 3, 1.0),
            presence=_sequence_value(point, 4, 1.0),
        )
    else:
        confidence = _read_float(getattr(point, "confidence", None), 1.0)
        normalized = PosePoint(
            x=_read_float(getattr(point, "x", None), float("nan")),
            y=_read_float(getattr(point, "y", None), float("nan")),
            z=_read_float(getattr(point, "z", None), 0.0),
            visibility=_read_float(getattr(point, "visibility", None), confidence),
            presence=_read_float(getattr(point, "presence", None), confidence),
        )

    if not normalized.is_usable(min_visibility=min_visibility, min_presence=min_presence):
        return None
    return normalized


def angle_3pts(
    a: object | None,
    b: object | None,
    c: object | None,
    *,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    min_presence: float = DEFAULT_MIN_PRESENCE,
) -> float | None:
    point_a = coerce_point(a, min_visibility=min_visibility, min_presence=min_presence)
    point_b = coerce_point(b, min_visibility=min_visibility, min_presence=min_presence)
    point_c = coerce_point(c, min_visibility=min_visibility, min_presence=min_presence)
    if point_a is None or point_b is None or point_c is None:
        return None

    ba = (
        point_a.x - point_b.x,
        point_a.y - point_b.y,
        point_a.z - point_b.z,
    )
    bc = (
        point_c.x - point_b.x,
        point_c.y - point_b.y,
        point_c.z - point_b.z,
    )
    norm_ba = sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2])
    norm_bc = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2])
    if norm_ba <= 1e-8 or norm_bc <= 1e-8:
        return None

    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    cosine = max(-1.0, min(1.0, dot / (norm_ba * norm_bc)))
    return degrees(acos(cosine))


def calculate_angle_2d(
    a: object | None,
    b: object | None,
    c: object | None,
    frame_width: int | float | None = None,
    frame_height: int | float | None = None,
) -> float | None:
    """Return the A-B-C projected angle in degrees, with B as the vertex.

    When positive frame dimensions are supplied and all coordinates look
    normalized, x/y are converted to pixels before the angle is calculated.
    Low visibility does not change the geometry result; callers should carry
    visibility separately when deciding whether the observation is reliable.
    """

    points = tuple(
        coerce_point(
            point,
            min_visibility=0.0,
            min_presence=0.0,
        )
        for point in (a, b, c)
    )
    if any(point is None for point in points):
        return None
    point_a, vertex, point_c = points
    assert point_a is not None and vertex is not None and point_c is not None
    width = _positive_dimension(frame_width)
    height = _positive_dimension(frame_height)
    if (
        frame_width is not None or frame_height is not None
    ) and (width is None or height is None):
        return None
    normalized = (
        width is not None
        and height is not None
        and all(
            -1.5 <= value <= 1.5
            for point in points
            if point is not None
            for value in (point.x, point.y)
        )
    )
    scale_x = width if normalized and width is not None else 1.0
    scale_y = height if normalized and height is not None else 1.0
    return _vector_angle_degrees(
        (
            (point_a.x - vertex.x) * scale_x,
            (point_a.y - vertex.y) * scale_y,
        ),
        (
            (point_c.x - vertex.x) * scale_x,
            (point_c.y - vertex.y) * scale_y,
        ),
    )


def calculate_angle_3d(
    a: object | None,
    b: object | None,
    c: object | None,
) -> float | None:
    """Return the A-B-C spatial angle in degrees, with B as the vertex."""

    points = tuple(
        coerce_point(
            point,
            min_visibility=0.0,
            min_presence=0.0,
        )
        for point in (a, b, c)
    )
    if any(point is None for point in points):
        return None
    point_a, vertex, point_c = points
    assert point_a is not None and vertex is not None and point_c is not None
    return _vector_angle_degrees(
        (
            point_a.x - vertex.x,
            point_a.y - vertex.y,
            point_a.z - vertex.z,
        ),
        (
            point_c.x - vertex.x,
            point_c.y - vertex.y,
            point_c.z - vertex.z,
        ),
    )


def _positive_dimension(value: int | float | None) -> float | None:
    try:
        numeric = float(value) if value is not None else 0.0
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if isfinite(numeric) and numeric > 0.0 else None


def _vector_angle_degrees(
    first: Sequence[float],
    second: Sequence[float],
) -> float | None:
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
    cosine = max(
        -1.0,
        min(1.0, dot / (norm_first * norm_second)),
    )
    value = degrees(acos(cosine))
    return value if isfinite(value) else None


def safe_distance(
    p1: object | None,
    p2: object | None,
    *,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    min_presence: float = DEFAULT_MIN_PRESENCE,
) -> float | None:
    point_a = coerce_point(p1, min_visibility=min_visibility, min_presence=min_presence)
    point_b = coerce_point(p2, min_visibility=min_visibility, min_presence=min_presence)
    if point_a is None or point_b is None:
        return None
    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    dz = point_a.z - point_b.z
    return sqrt(dx * dx + dy * dy + dz * dz)


def midpoint(
    p1: object | None,
    p2: object | None,
    *,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    min_presence: float = DEFAULT_MIN_PRESENCE,
) -> PosePoint | None:
    point_a = coerce_point(p1, min_visibility=min_visibility, min_presence=min_presence)
    point_b = coerce_point(p2, min_visibility=min_visibility, min_presence=min_presence)
    if point_a is None or point_b is None:
        return None
    return PosePoint(
        x=(point_a.x + point_b.x) / 2.0,
        y=(point_a.y + point_b.y) / 2.0,
        z=(point_a.z + point_b.z) / 2.0,
        visibility=min(point_a.visibility, point_b.visibility),
        presence=min(point_a.presence, point_b.presence),
    )


__all__ = [
    "PosePoint",
    "angle_3pts",
    "calculate_angle_2d",
    "calculate_angle_3d",
    "coerce_point",
    "midpoint",
    "safe_distance",
]
