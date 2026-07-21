from __future__ import annotations

import threading
import time
from collections.abc import Callable
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


def _to_keypoints(landmarks: Sequence[object], *, model_name: str) -> list[Keypoint]:
    keypoints: list[Keypoint] = []
    for index, name in enumerate(MEDIAPIPE_33_NAMES):
        landmark = landmarks[index] if index < len(landmarks) else None
        if landmark is None:
            keypoints.append(Keypoint(name=name, x=nan, y=nan, z=nan, confidence=0.0, source_model=model_name))
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
                source_model=model_name,
                visibility=visibility,
                presence=presence,
            )
        )
    return keypoints


def _bbox(keypoints: Sequence[Keypoint]) -> tuple[float, float, float, float] | None:
    usable = [point for point in keypoints if point.confidence >= 0.2 and np.isfinite(point.x) and np.isfinite(point.y)]
    if not usable:
        return None
    xs = [point.x for point in usable]
    ys = [point.y for point in usable]
    return min(xs), min(ys), max(xs), max(ys)


def _convert_landmarker_result(
    result: object,
    *,
    model_name: str,
    timestamp_ms: int,
    inference_time_ms: float,
) -> PoseResult:
    pose_landmarks = getattr(result, "pose_landmarks", None) or ()
    pose_candidates = [
        _to_keypoints(landmarks, model_name=model_name)
        for landmarks in pose_landmarks
    ]
    pose_world_landmarks = getattr(result, "pose_world_landmarks", None) or ()
    world_pose_candidates = [
        _to_keypoints(landmarks, model_name=f"{model_name}-world")
        for landmarks in pose_world_landmarks
    ]
    extra = {
        "raw_result": result,
        "pose_candidates": pose_candidates,
        "world_pose_candidates": world_pose_candidates,
        "world_keypoints": world_pose_candidates[0] if world_pose_candidates else [],
        "world_landmarks_available": bool(world_pose_candidates),
    }
    segmentation_masks = getattr(result, "segmentation_masks", None)
    if segmentation_masks:
        try:
            extra["segmentation_mask"] = np.asarray(
                segmentation_masks[0].numpy_view(),
                dtype=np.float32,
            ).copy()
        except (AttributeError, IndexError, TypeError, ValueError):
            pass

    if not pose_candidates:
        return PoseResult(
            keypoints=[],
            connections=MEDIAPIPE_CONNECTIONS,
            model_name=model_name,
            num_keypoints=0,
            success=False,
            inference_time_ms=float(inference_time_ms),
            timestamp_ms=int(timestamp_ms),
            extra=extra,
        )

    keypoints = pose_candidates[0]
    return PoseResult(
        keypoints=keypoints,
        connections=MEDIAPIPE_CONNECTIONS,
        model_name=model_name,
        num_keypoints=len(keypoints),
        success=True,
        inference_time_ms=float(inference_time_ms),
        bbox=_bbox(keypoints),
        timestamp_ms=int(timestamp_ms),
        extra=extra,
    )


class MediaPipeBackend:
    model_name = "mediapipe"
    support_tier = "product"

    def __init__(
        self,
        model_path: str | Path = "models/pose_landmarker_full.task",
        *,
        output_segmentation_masks: bool = True,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None or mp_python is None or vision is None:
            raise RuntimeError(
                "mediapipe is not installed. Install dependencies with 'python -m pip install -r requirements.txt'."
            )
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model file not found: {self.model_path}")
        self.num_poses = max(1, int(num_poses))
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.num_poses,
            min_pose_detection_confidence=float(min_pose_detection_confidence),
            min_pose_presence_confidence=float(min_pose_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
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
        return _convert_landmarker_result(
            result,
            model_name=self.model_name,
            timestamp_ms=timestamp_ms,
            inference_time_ms=inference_time_ms,
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
        return _to_keypoints(landmarks, model_name=self.model_name)

    def _bbox(self, keypoints: Sequence[Keypoint]) -> tuple[float, float, float, float] | None:
        return _bbox(keypoints)


class MediaPipeLiveStreamBackend:
    """MediaPipe LIVE_STREAM adapter used only by the realtime camera scheduler."""

    model_name = "mediapipe"
    support_tier = "product"

    def __init__(
        self,
        model_path: str | Path = "models/pose_landmarker_full.task",
        *,
        output_segmentation_masks: bool = True,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None or mp_python is None or vision is None:
            raise RuntimeError(
                "mediapipe is not installed. Install dependencies with 'python -m pip install -r requirements.txt'."
            )
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model file not found: {self.model_path}")
        self.num_poses = max(1, int(num_poses))
        self._lock = threading.RLock()
        self._closed = False
        self._last_timestamp_ms = -1
        self._result_callback: Callable[[PoseResult, int], None] | None = None
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=self.num_poses,
            min_pose_detection_confidence=float(min_pose_detection_confidence),
            min_pose_presence_confidence=float(min_pose_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            output_segmentation_masks=bool(output_segmentation_masks),
            result_callback=self._sdk_result_callback,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def set_result_callback(
        self,
        callback: Callable[[PoseResult, int], None] | None,
    ) -> None:
        with self._lock:
            if not self._closed:
                self._result_callback = callback

    def detect_async(self, frame: np.ndarray, timestamp_ms: int) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("MediaPipe live stream backend is closed")
            timestamp_ms = int(timestamp_ms)
            if timestamp_ms <= self._last_timestamp_ms:
                raise ValueError("MediaPipe LIVE_STREAM timestamps must be strictly increasing")
            self._last_timestamp_ms = timestamp_ms
            landmarker = self._landmarker
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        landmarker.detect_async(image, timestamp_ms)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._result_callback = None
            landmarker = self._landmarker
        landmarker.close()

    def _sdk_result_callback(self, result: object, _output_image: object, timestamp_ms: int) -> None:
        with self._lock:
            if self._closed:
                return
            callback = self._result_callback
        if callback is None:
            return
        pose = _convert_landmarker_result(
            result,
            model_name=self.model_name,
            timestamp_ms=int(timestamp_ms),
            inference_time_ms=0.0,
        )
        with self._lock:
            if self._closed or callback is not self._result_callback:
                return
        callback(pose, int(timestamp_ms))
