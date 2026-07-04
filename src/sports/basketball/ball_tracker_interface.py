from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BallDetection:
    ball_detected: bool
    center_x: float | None
    center_y: float | None
    radius_or_bbox: float | tuple[float, float, float, float] | None
    confidence: float
    timestamp_ms: int


class BallTracker(Protocol):
    def detect(self, frame: Any, timestamp_ms: int) -> BallDetection:
        ...

