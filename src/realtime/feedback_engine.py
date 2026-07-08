from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from src.backends.base import PoseResult


@dataclass(frozen=True)
class FeedbackState:
    person_lost: bool
    low_confidence: bool
    keypoints_unstable: bool
    angle_available: bool
    message: str


class FeedbackEngine:
    def __init__(
        self,
        abnormal_frames_required: int = 5,
        min_confidence: float = 0.35,
        jitter_threshold: float = 0.06,
    ) -> None:
        self.abnormal_frames_required = max(1, int(abnormal_frames_required))
        self.min_confidence = float(min_confidence)
        self.jitter_threshold = float(jitter_threshold)
        self._person_lost_frames = 0
        self._low_conf_frames = 0
        self._unstable_frames = 0
        self._previous_xy: dict[str, tuple[float, float]] = {}
        self._last_message = "Ready"

    def update(self, result: PoseResult, angles: dict[str, float | None]) -> FeedbackState:
        person_lost_now = not result.success
        low_confidence_now = result.success and self._mean_confidence(result) < self.min_confidence
        unstable_now = result.success and self._mean_displacement(result) > self.jitter_threshold
        angle_available = any(value is not None for value in angles.values())

        self._person_lost_frames = self._next_count(self._person_lost_frames, person_lost_now)
        self._low_conf_frames = self._next_count(self._low_conf_frames, low_confidence_now)
        self._unstable_frames = self._next_count(self._unstable_frames, unstable_now)

        person_lost = self._person_lost_frames >= self.abnormal_frames_required
        low_confidence = self._low_conf_frames >= self.abnormal_frames_required
        keypoints_unstable = self._unstable_frames >= self.abnormal_frames_required

        if result.success:
            self._previous_xy = {
                point.name: (point.x, point.y)
                for point in result.keypoints
                if point.confidence >= self.min_confidence
            }

        if person_lost:
            self._last_message = "No person detected"
        elif low_confidence:
            self._last_message = "Low keypoint confidence"
        elif keypoints_unstable:
            self._last_message = "Keypoints unstable"
        elif not angle_available:
            self._last_message = "Angles unavailable"
        else:
            self._last_message = "Tracking stable"

        return FeedbackState(
            person_lost=person_lost,
            low_confidence=low_confidence,
            keypoints_unstable=keypoints_unstable,
            angle_available=angle_available,
            message=self._last_message,
        )

    def reset(self) -> None:
        self._person_lost_frames = 0
        self._low_conf_frames = 0
        self._unstable_frames = 0
        self._previous_xy.clear()
        self._last_message = "Ready"

    def _mean_confidence(self, result: PoseResult) -> float:
        values = [point.confidence for point in result.keypoints]
        return mean(values) if values else 0.0

    def _mean_displacement(self, result: PoseResult) -> float:
        distances: list[float] = []
        for point in result.keypoints:
            previous = self._previous_xy.get(point.name)
            if previous is None or point.confidence < self.min_confidence:
                continue
            dx = point.x - previous[0]
            dy = point.y - previous[1]
            distances.append((dx * dx + dy * dy) ** 0.5)
        return mean(distances) if distances else 0.0

    def _next_count(self, previous: int, active: bool) -> int:
        return previous + 1 if active else 0
