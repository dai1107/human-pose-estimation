from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.biomechanics.hand_landmarks import coerce_hand_landmarks
from src.biomechanics.landmarks import coerce_landmark
from src.biomechanics.types import LandmarkPoint

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ModuleNotFoundError:
    mp = None
    mp_python = None
    vision = None


DEFAULT_HAND_DETECT_WIDTH = 416
DEFAULT_MAX_HAND_DETECT_FPS = 18.0


@dataclass(frozen=True)
class HandDetection:
    side: str
    score: float
    landmarks: list[LandmarkPoint]
    world_landmarks: list[LandmarkPoint]


def prepare_detection_frame(frame: np.ndarray, detect_width: int) -> np.ndarray:
    if detect_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= detect_width:
        return frame
    detect_height = max(1, int(round(height * (detect_width / width))))
    return cv2.resize(frame, (detect_width, detect_height), interpolation=cv2.INTER_AREA)


def frame_to_mp_image(frame: np.ndarray, detect_width: int) -> "mp.Image":
    detection_frame = prepare_detection_frame(frame, detect_width)
    rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))


def infer_hand_side(landmarks: Sequence[object], fallback_index: int) -> str:
    xs = [point.x for point in (coerce_landmark(point) for point in landmarks) if isfinite(point.x)]
    if xs:
        return "left" if sum(xs) / len(xs) < 0.5 else "right"
    return "left" if fallback_index == 0 else "right"


def hand_side_from_handedness(
    result: object,
    hand_index: int,
    landmarks: Sequence[object],
) -> tuple[str, float]:
    handedness = getattr(result, "handedness", None) or []
    if hand_index < len(handedness) and handedness[hand_index]:
        category = handedness[hand_index][0]
        raw_name = str(getattr(category, "category_name", "") or getattr(category, "display_name", "")).strip().lower()
        raw_score = getattr(category, "score", 0.5)
        try:
            score = float(raw_score) if raw_score is not None else 0.5
        except (TypeError, ValueError):
            score = 0.5
        if raw_name in {"left", "right"}:
            return raw_name, score
    return infer_hand_side(landmarks, hand_index), 0.5


def extract_hand_detections(result: object | None) -> dict[str, HandDetection]:
    if result is None or not getattr(result, "hand_landmarks", None):
        return {}
    world_landmarks = getattr(result, "hand_world_landmarks", None) or []
    detections: dict[str, HandDetection] = {}
    for index, landmarks in enumerate(result.hand_landmarks):
        side, score = hand_side_from_handedness(result, index, landmarks)
        world = world_landmarks[index] if index < len(world_landmarks) else None
        candidate = HandDetection(
            side=side,
            score=score,
            landmarks=coerce_hand_landmarks(landmarks),
            world_landmarks=coerce_hand_landmarks(world) if world is not None else [],
        )
        existing = detections.get(side)
        if existing is not None and existing.score >= candidate.score:
            continue
        detections[side] = candidate
    return detections


class MediaPipeHandTracker:
    def __init__(
        self,
        model_path: str | Path = "models/hand_landmarker.task",
        *,
        detect_width: int = DEFAULT_HAND_DETECT_WIDTH,
        max_hands: int = 2,
    ) -> None:
        if mp is None or mp_python is None or vision is None:
            raise RuntimeError(
                "mediapipe is not installed. Install dependencies with 'python -m pip install -r requirements.txt'."
            )
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Hand Landmarker model file not found: {self.model_path}")
        self.detect_width = max(0, int(detect_width))
        self.max_hands = max(1, int(max_hands))
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame: np.ndarray, *, timestamp_ms: int | None = None) -> dict[str, HandDetection]:
        timestamp_ms = self._next_timestamp(timestamp_ms)
        image = frame_to_mp_image(frame, self.detect_width)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return extract_hand_detections(result)

    def close(self) -> None:
        self._landmarker.close()

    def _next_timestamp(self, timestamp_ms: int | None) -> int:
        if timestamp_ms is None:
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms
