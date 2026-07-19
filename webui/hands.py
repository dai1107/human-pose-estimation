from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from pathlib import Path

import numpy as np

from src.biomechanics.hand_landmarks import (
    SUPPLEMENTAL_FINGER_CONNECTIONS,
    SUPPLEMENTAL_FINGER_DISPLAY_INDICES,
    hand_landmark_name,
)
from src.biomechanics.types import LandmarkPoint
from src.runtime_hand import HandDetection, MediaPipeHandTracker
from src.utils.draw_utils import draw_hand_landmarks


DEFAULT_WEB_HAND_DETECT_FPS = 10.0
DEFAULT_WEB_HAND_HOLD_MS = 350


class WebHandOverlay:
    """Lazy, rate-limited five-finger tracking shared by both web pipelines."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        detect_fps: float = DEFAULT_WEB_HAND_DETECT_FPS,
        hold_ms: int = DEFAULT_WEB_HAND_HOLD_MS,
        tracker_factory: Callable[..., MediaPipeHandTracker] = MediaPipeHandTracker,
    ) -> None:
        self.model_path = Path(model_path)
        self.detect_interval_ms = (
            0 if detect_fps <= 0 else max(1, int(round(1000.0 / float(detect_fps))))
        )
        self.hold_ms = max(0, int(hold_ms))
        self._tracker_factory = tracker_factory
        self._tracker: MediaPipeHandTracker | None = None
        self._detections: dict[str, HandDetection] = {}
        self._last_detection_ms = -1
        self._last_nonempty_ms = -1
        self.error: str | None = None

    def _ensure_tracker(self) -> MediaPipeHandTracker | None:
        if self._tracker is not None:
            return self._tracker
        if self.error is not None:
            return None
        try:
            self._tracker = self._tracker_factory(
                self.model_path,
                detect_width=416,
                max_hands=2,
            )
        except Exception as exc:
            self.error = str(exc)
            return None
        return self._tracker

    def update(
        self,
        frame: np.ndarray,
        *,
        timestamp_ms: int,
        enabled: bool,
    ) -> dict[str, HandDetection]:
        timestamp_ms = int(timestamp_ms)
        if not enabled:
            self._detections = {}
            self._last_detection_ms = -1
            self._last_nonempty_ms = -1
            return {}
        tracker = self._ensure_tracker()
        if tracker is None:
            return {}
        if (
            self._last_detection_ms >= 0
            and self.detect_interval_ms > 0
            and timestamp_ms - self._last_detection_ms < self.detect_interval_ms
        ):
            return dict(self._detections)
        self._last_detection_ms = timestamp_ms
        try:
            detections = tracker.detect(frame, timestamp_ms=timestamp_ms)
        except Exception as exc:
            self.error = str(exc)
            self._detections = {}
            return {}
        if detections:
            self._detections = dict(detections)
            self._last_nonempty_ms = timestamp_ms
        elif (
            self._last_nonempty_ms < 0
            or timestamp_ms - self._last_nonempty_ms > self.hold_ms
        ):
            self._detections = {}
        return dict(self._detections)

    def close(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
        self._tracker = None
        self._detections = {}


def hand_overlay_visible(profile: str, show_fingers: bool) -> bool:
    return bool(show_fingers and str(profile) != "lower-body")


def serialize_hand_overlay(
    detections: Mapping[str, HandDetection],
) -> tuple[list[dict[str, object]], list[list[str]]]:
    points: list[dict[str, object]] = []
    connections: list[list[str]] = []
    available: dict[str, set[int]] = {}
    for side, detection in sorted(detections.items()):
        side_key = str(side).strip().lower() or "unknown"
        side_indices: set[int] = set()
        for index in sorted(SUPPLEMENTAL_FINGER_DISPLAY_INDICES):
            if index >= len(detection.landmarks):
                continue
            point = detection.landmarks[index]
            confidence = min(float(point.visibility), float(point.presence))
            if (
                not point.is_usable(0.05, 0.05)
                or not isfinite(float(point.x))
                or not isfinite(float(point.y))
            ):
                continue
            side_indices.add(index)
            points.append(
                {
                    "name": hand_landmark_name(side_key, index),
                    "x": round(float(point.x), 6),
                    "y": round(float(point.y), 6),
                    "z": round(float(point.z), 6)
                    if isfinite(float(point.z))
                    else 0.0,
                    "visibility": round(max(0.0, min(1.0, confidence)), 4),
                }
            )
        available[side_key] = side_indices
    for side, indices in sorted(available.items()):
        for start, end in SUPPLEMENTAL_FINGER_CONNECTIONS:
            if start in indices and end in indices:
                connections.append(
                    [
                        hand_landmark_name(side, start),
                        hand_landmark_name(side, end),
                    ]
                )
    return points, connections


def rtmw_hand_detections(
    extra: Mapping[str, object],
    *,
    min_confidence: float = 0.30,
) -> dict[str, HandDetection]:
    """Adapt RTMW's 21-point hands to the existing web hand overlay."""
    raw_hands = extra.get("rtmw_hand_keypoints")
    if not isinstance(raw_hands, Mapping):
        return {}
    detections: dict[str, HandDetection] = {}
    for side in ("left", "right"):
        raw_points = raw_hands.get(side)
        if not isinstance(raw_points, Sequence) or len(raw_points) < 21:
            continue
        landmarks: list[LandmarkPoint] = []
        confidences: list[float] = []
        for raw_point in list(raw_points)[:21]:
            try:
                x = float(getattr(raw_point, "x"))
                y = float(getattr(raw_point, "y"))
                z = float(getattr(raw_point, "z", 0.0))
                confidence = max(
                    0.0,
                    min(1.0, float(getattr(raw_point, "confidence", 0.0))),
                )
            except (TypeError, ValueError, OverflowError):
                x, y, z, confidence = float("nan"), float("nan"), 0.0, 0.0
            display_confidence = (
                confidence
                if isfinite(x) and isfinite(y) and confidence >= min_confidence
                else 0.0
            )
            landmarks.append(
                LandmarkPoint(
                    x=x,
                    y=y,
                    z=z,
                    visibility=display_confidence,
                    presence=display_confidence,
                )
            )
            if isfinite(x) and isfinite(y) and confidence >= min_confidence:
                confidences.append(confidence)
        if len(confidences) < 5:
            continue
        detections[side] = make_hand_detection(
            side,
            landmarks,
            score=sum(confidences) / len(confidences),
        )
    return detections


def draw_hand_overlay(
    frame: np.ndarray,
    detections: Mapping[str, HandDetection],
) -> None:
    draw_hand_landmarks(
        frame,
        {
            side: detection.landmarks
            for side, detection in detections.items()
        },
    )


def make_hand_detection(
    side: str,
    landmarks: Sequence[LandmarkPoint],
    *,
    score: float = 1.0,
) -> HandDetection:
    """Small public constructor used by deterministic web overlay tests."""
    return HandDetection(
        side=side,
        score=score,
        landmarks=list(landmarks),
        world_landmarks=[],
    )


__all__ = [
    "DEFAULT_WEB_HAND_DETECT_FPS",
    "DEFAULT_WEB_HAND_HOLD_MS",
    "WebHandOverlay",
    "draw_hand_overlay",
    "hand_overlay_visible",
    "make_hand_detection",
    "rtmw_hand_detections",
    "serialize_hand_overlay",
]
