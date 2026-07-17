from __future__ import annotations

import time
from math import nan
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.utils.keypoint_schema import MEDIAPIPE_33_NAMES, MEDIAPIPE_CONNECTIONS

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ModuleNotFoundError:
    mp = None
    mp_python = None
    vision = None


class MediaPipeBackend:
    model_name = "mediapipe"

    def __init__(
        self,
        model_path: str | Path = "models/pose_landmarker_full.task",
        *,
        output_segmentation_masks: bool = True,
    ) -> None:
        if mp is None or mp_python is None or vision is None:
            raise RuntimeError(
                "mediapipe is not installed. Install dependencies with 'python -m pip install -r requirements.txt'."
            )
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model file not found: {self.model_path}")
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=bool(output_segmentation_masks),
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        timestamp_ms = self._next_timestamp(timestamp_ms)
        started = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        inference_time_ms = (time.perf_counter() - started) * 1000.0
        extra = {"raw_result": result}
        segmentation_masks = getattr(result, "segmentation_masks", None)
        if segmentation_masks:
            try:
                extra["segmentation_mask"] = np.asarray(
                    segmentation_masks[0].numpy_view(),
                    dtype=np.float32,
                ).copy()
            except (AttributeError, IndexError, TypeError, ValueError):
                pass

        if not result.pose_landmarks:
            return PoseResult(
                keypoints=[],
                connections=MEDIAPIPE_CONNECTIONS,
                model_name=self.model_name,
                num_keypoints=0,
                success=False,
                inference_time_ms=inference_time_ms,
                timestamp_ms=timestamp_ms,
                extra=extra,
            )

        keypoints = self._to_keypoints(result.pose_landmarks[0])
        return PoseResult(
            keypoints=keypoints,
            connections=MEDIAPIPE_CONNECTIONS,
            model_name=self.model_name,
            num_keypoints=len(keypoints),
            success=True,
            inference_time_ms=inference_time_ms,
            bbox=self._bbox(keypoints),
            timestamp_ms=timestamp_ms,
            extra=extra,
        )

    def close(self) -> None:
        self._landmarker.close()

    def _next_timestamp(self, timestamp_ms: int | None) -> int:
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic_ns() / 1_000_000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _to_keypoints(self, landmarks: Sequence[object]) -> list[Keypoint]:
        keypoints: list[Keypoint] = []
        for index, name in enumerate(MEDIAPIPE_33_NAMES):
            landmark = landmarks[index] if index < len(landmarks) else None
            if landmark is None:
                keypoints.append(Keypoint(name=name, x=nan, y=nan, z=nan, confidence=0.0, source_model=self.model_name))
                continue
            visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
            presence = float(getattr(landmark, "presence", 1.0) or 0.0)
            keypoints.append(
                Keypoint(
                    name=name,
                    x=float(getattr(landmark, "x", nan)),
                    y=float(getattr(landmark, "y", nan)),
                    z=float(getattr(landmark, "z", 0.0)),
                    confidence=min(visibility, presence),
                    source_model=self.model_name,
                )
            )
        return keypoints

    def _bbox(self, keypoints: Sequence[Keypoint]) -> tuple[float, float, float, float] | None:
        usable = [point for point in keypoints if point.confidence >= 0.2 and np.isfinite(point.x) and np.isfinite(point.y)]
        if not usable:
            return None
        xs = [point.x for point in usable]
        ys = [point.y for point in usable]
        return min(xs), min(ys), max(xs), max(ys)
