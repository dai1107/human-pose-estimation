from __future__ import annotations

from math import isfinite

from .schema import SquatFrameMeasurement


def _finite(value: float | None) -> bool:
    return value is not None and isfinite(float(value))


def mean_abs_pair_difference(frames: list[SquatFrameMeasurement], left_attr: str, right_attr: str) -> tuple[float | None, float | None]:
    values: list[float] = []
    for frame in frames:
        left = getattr(frame, left_attr)
        right = getattr(frame, right_attr)
        if _finite(left) and _finite(right):
            values.append(abs(float(left) - float(right)))
    if not values:
        return None, None
    return sum(values) / len(values), max(values)


def knee_symmetry_proxy(frames: list[SquatFrameMeasurement]) -> tuple[float | None, float | None]:
    return mean_abs_pair_difference(frames, "left_knee_angle", "right_knee_angle")


def hip_symmetry_proxy(frames: list[SquatFrameMeasurement]) -> tuple[float | None, float | None]:
    return mean_abs_pair_difference(frames, "left_hip_angle", "right_hip_angle")

