from __future__ import annotations

from typing import Sequence

from .landmarks import coerce_landmarks
from .types import LandmarkPoint


def exponential_smooth_landmarks(
    current_landmarks: Sequence[object] | None,
    previous_landmarks: Sequence[LandmarkPoint] | None,
    alpha: float = 0.65,
) -> list[LandmarkPoint]:
    current = coerce_landmarks(current_landmarks)
    alpha = max(0.0, min(1.0, float(alpha)))
    if previous_landmarks is None or alpha <= 0.0 or len(previous_landmarks) != len(current):
        return current

    keep = 1.0 - alpha
    smoothed: list[LandmarkPoint] = []
    for old, new in zip(previous_landmarks, current):
        if not old.is_finite() or not new.is_finite():
            smoothed.append(new)
            continue
        smoothed.append(
            LandmarkPoint(
                x=old.x * keep + new.x * alpha,
                y=old.y * keep + new.y * alpha,
                z=old.z * keep + new.z * alpha,
                visibility=new.visibility,
                presence=new.presence,
            )
        )
    return smoothed

