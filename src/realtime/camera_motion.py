"""Lightweight background camera-motion estimation for realtime streams."""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any

import cv2
import numpy as np


CAMERA_STATIC = "camera_static"
CAMERA_SMALL_MOTION = "camera_small_motion"
CAMERA_UNSTABLE = "camera_unstable"


@dataclass(frozen=True, slots=True)
class CameraMotionConfig:
    analysis_width: int = 192
    sample_fps: float = 5.0
    maximum_features: int = 120
    minimum_tracks: int = 8
    static_score: float = 0.12
    unstable_score: float = 0.42


class CameraMotionEstimator:
    """Estimate global background motion using sparse flow plus affine RANSAC."""

    def __init__(self, config: CameraMotionConfig | None = None) -> None:
        self.config = config or CameraMotionConfig()
        self._previous_gray: np.ndarray | None = None
        self._previous_timestamp_ms: float | None = None
        self._last_result = _unavailable("warming_up")

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_timestamp_ms = None
        self._last_result = _unavailable("warming_up")

    def update(
        self,
        frame: np.ndarray,
        *,
        timestamp_ms: int | float,
        person_keypoints: Sequence[object] = (),
    ) -> dict[str, Any]:
        timestamp = _finite(timestamp_ms, 0.0)
        minimum_interval = 1000.0 / max(0.1, float(self.config.sample_fps))
        if (
            self._previous_timestamp_ms is not None
            and timestamp - self._previous_timestamp_ms < minimum_interval
        ):
            return dict(self._last_result)
        gray, scale = _prepare_gray(frame, self.config.analysis_width)
        if gray is None:
            self._last_result = _unavailable("invalid_frame")
            return dict(self._last_result)
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            self._previous_gray = gray
            self._previous_timestamp_ms = timestamp
            self._last_result = _unavailable("warming_up")
            return dict(self._last_result)

        mask = np.full(gray.shape, 255, dtype=np.uint8)
        bbox = _normalized_bbox(person_keypoints)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            height, width = gray.shape
            margin = 0.08
            cv2.rectangle(
                mask,
                (max(0, int((x1 - margin) * width)), max(0, int((y1 - margin) * height))),
                (min(width - 1, int((x2 + margin) * width)), min(height - 1, int((y2 + margin) * height))),
                0,
                thickness=-1,
            )
        previous_points = cv2.goodFeaturesToTrack(
            self._previous_gray,
            maxCorners=max(16, int(self.config.maximum_features)),
            qualityLevel=0.01,
            minDistance=5,
            mask=mask,
            blockSize=5,
        )
        result = _unavailable("insufficient_background_features")
        if previous_points is not None and len(previous_points) >= self.config.minimum_tracks:
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                previous_points,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if current_points is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                old = previous_points.reshape(-1, 2)[valid]
                new = current_points.reshape(-1, 2)[valid]
                if len(old) >= self.config.minimum_tracks:
                    transform, inliers = cv2.estimateAffinePartial2D(
                        old,
                        new,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.0,
                        maxIters=100,
                        confidence=0.95,
                    )
                    result = self._result_from_transform(
                        transform,
                        inliers,
                        track_count=len(old),
                        shape=gray.shape,
                        scale=scale,
                    )
        self._previous_gray = gray
        self._previous_timestamp_ms = timestamp
        self._last_result = result
        return dict(result)

    def _result_from_transform(
        self,
        transform: np.ndarray | None,
        inliers: np.ndarray | None,
        *,
        track_count: int,
        shape: tuple[int, int],
        scale: float,
    ) -> dict[str, Any]:
        if transform is None:
            return _unavailable("affine_estimation_failed")
        a, b, tx = (float(value) for value in transform[0])
        c, d, ty = (float(value) for value in transform[1])
        rotation_rad = float(np.arctan2(c, a))
        zoom = max(1e-8, hypot(a, c))
        diagonal = max(1.0, hypot(*shape))
        translation_ratio = hypot(tx, ty) / diagonal
        rotation_ratio = abs(rotation_rad) / np.deg2rad(4.0)
        zoom_ratio = abs(zoom - 1.0) / 0.04
        raw_score = translation_ratio / 0.025 + rotation_ratio + zoom_ratio
        score = max(0.0, min(1.0, raw_score / 3.0))
        state = (
            CAMERA_STATIC
            if score < self.config.static_score
            else CAMERA_UNSTABLE
            if score >= self.config.unstable_score
            else CAMERA_SMALL_MOTION
        )
        inlier_ratio = (
            float(np.mean(inliers.reshape(-1).astype(bool)))
            if inliers is not None and len(inliers)
            else 0.0
        )
        return {
            "schema_version": 1,
            "available": True,
            "method": "background_sparse_optical_flow_affine_ransac",
            "camera_motion_score": score,
            "state": state,
            "translation_normalized": (tx / diagonal, ty / diagonal),
            "translation_pixels_source": (tx / max(scale, 1e-8), ty / max(scale, 1e-8)),
            "rotation_deg": float(np.rad2deg(rotation_rad)),
            "scale_change": zoom - 1.0,
            "tracked_background_points": int(track_count),
            "inlier_ratio": inlier_ratio,
            "modifies_body_3d": False,
            "formal_rule_replacement_allowed": False,
        }


class LatestCameraMotionWorker:
    """Run camera motion off the render loop with one replaceable pending job."""

    def __init__(self, estimator: CameraMotionEstimator | None = None) -> None:
        self.estimator = estimator or CameraMotionEstimator()
        self._jobs: queue.Queue[tuple[np.ndarray, float, tuple[object, ...]] | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest = _unavailable("warming_up")
        self._closed = False
        self._last_submit_timestamp_ms: float | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera-motion")
        self._thread.start()

    def submit(
        self,
        frame: np.ndarray,
        *,
        timestamp_ms: int | float,
        person_keypoints: Sequence[object] = (),
    ) -> None:
        timestamp = _finite(timestamp_ms, 0.0)
        minimum_interval = 1000.0 / max(0.1, float(self.estimator.config.sample_fps))
        if (
            self._closed
            or frame is None
            or (
                self._last_submit_timestamp_ms is not None
                and timestamp - self._last_submit_timestamp_ms < minimum_interval
            )
        ):
            return
        self._last_submit_timestamp_ms = timestamp
        job = (frame.copy(), timestamp, tuple(person_keypoints))
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def reset(self) -> None:
        self.estimator.reset()
        self._last_submit_timestamp_ms = None
        with self._lock:
            self._latest = _unavailable("warming_up")
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
            self._jobs.put_nowait(None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            frame, timestamp, keypoints = job
            result = self.estimator.update(
                frame,
                timestamp_ms=timestamp,
                person_keypoints=keypoints,
            )
            with self._lock:
                self._latest = result


def normalize_camera_motion(value: Mapping[str, object] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _unavailable("not_provided")
    score = max(0.0, min(1.0, _finite(value.get("camera_motion_score"), 0.0)))
    state = str(value.get("state", CAMERA_STATIC))
    if state not in {CAMERA_STATIC, CAMERA_SMALL_MOTION, CAMERA_UNSTABLE}:
        state = CAMERA_UNSTABLE if score >= 0.42 else CAMERA_SMALL_MOTION if score >= 0.12 else CAMERA_STATIC
    return {
        **{str(key): item for key, item in value.items()},
        "schema_version": 1,
        "available": bool(value.get("available", True)),
        "camera_motion_score": score,
        "state": state,
        "modifies_body_3d": False,
        "formal_rule_replacement_allowed": False,
    }


def _prepare_gray(frame: np.ndarray, width: int) -> tuple[np.ndarray | None, float]:
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
        return None, 1.0
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    source_width = gray.shape[1]
    target_width = max(64, int(width))
    scale = min(1.0, target_width / max(1, source_width))
    if scale < 1.0:
        gray = cv2.resize(gray, (target_width, max(1, int(round(gray.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    return gray, scale


def _normalized_bbox(points: Sequence[object]) -> tuple[float, float, float, float] | None:
    xy: list[tuple[float, float]] = []
    for point in points:
        x = _value(point, "x")
        y = _value(point, "y")
        try:
            x_value, y_value = float(x), float(y)
        except (TypeError, ValueError, OverflowError):
            continue
        if isfinite(x_value) and isfinite(y_value):
            xy.append((x_value, y_value))
    if len(xy) < 4:
        return None
    xs, ys = zip(*xy)
    return max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))


def _value(item: object, name: str) -> object:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if isfinite(number) else default


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "available": False,
        "method": "background_sparse_optical_flow_affine_ransac",
        "camera_motion_score": 0.0,
        "state": CAMERA_STATIC,
        "reason": reason,
        "modifies_body_3d": False,
        "formal_rule_replacement_allowed": False,
    }


__all__ = [
    "CAMERA_SMALL_MOTION",
    "CAMERA_STATIC",
    "CAMERA_UNSTABLE",
    "CameraMotionConfig",
    "CameraMotionEstimator",
    "LatestCameraMotionWorker",
    "normalize_camera_motion",
]
