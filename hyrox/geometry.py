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
    elif isinstance(point, Sequence) and not isinstance(point, (str, bytes, bytearray)):
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
    "coerce_point",
    "midpoint",
    "safe_distance",
]
