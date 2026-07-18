from __future__ import annotations

import time
from math import hypot, isfinite, nan
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import cv2
import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.backends.yolo_pose_backend import TargetSelect, YoloPoseBackend
from src.biomechanics.hand_landmarks import hand_landmark_name
from src.utils.keypoint_schema import (
    COCO_17_NAMES,
    MEDIAPIPE_33_NAMES,
    MEDIAPIPE_CONNECTIONS,
)


DEFAULT_RTMW_MODEL = (
    "models/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"
)
RTMW_INPUT_SIZE = (192, 256)
IDENTITY_MATCH_NAMES: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO133_BODY_INDEX = {
    name: index
    for index, name in enumerate(COCO_17_NAMES)
}
COCO133_SUPPLEMENTAL_INDEX = {
    "left_heel": 19,
    "right_heel": 22,
    "left_thumb": 95,
    "right_thumb": 116,
    "left_index": 99,
    "right_index": 120,
    "left_pinky": 111,
    "right_pinky": 132,
}
COCO133_FOOT_INDEX_PAIRS = {
    "left_foot_index": (17, 18),
    "right_foot_index": (20, 21),
}
COCO133_HAND_RANGES = {
    "left": range(91, 112),
    "right": range(112, 133),
}


class YoloRtmwWholeBodyBackend:
    """Lock one person with YOLO and run RTMW WholeBody on that target box."""

    model_name = "yolo-rtmw-wholebody"

    def __init__(
        self,
        rtmw_model_path: str | Path = DEFAULT_RTMW_MODEL,
        yolo_model_path: str | Path = "yolo11n-pose.pt",
        *,
        target_select: TargetSelect = "tracking",
        yolo_device: str = "",
        rtmw_device: str = "auto",
        model_input_size: tuple[int, int] = RTMW_INPUT_SIZE,
        min_match_points: int = 6,
        max_match_distance: float = 0.20,
        match_confidence: float = 0.20,
        yolo_backend: Any | None = None,
        session: Any | None = None,
    ) -> None:
        if min_match_points < 1:
            raise ValueError("min_match_points must be at least 1")
        if max_match_distance <= 0.0:
            raise ValueError("max_match_distance must be positive")
        self.rtmw_model_path = Path(rtmw_model_path)
        self.model_input_size = (
            max(1, int(model_input_size[0])),
            max(1, int(model_input_size[1])),
        )
        self.min_match_points = int(min_match_points)
        self.max_match_distance = float(max_match_distance)
        self.match_confidence = float(match_confidence)
        self.yolo_backend = yolo_backend or YoloPoseBackend(
            str(yolo_model_path),
            target_select=target_select,
            device=yolo_device,
        )
        self._session = session or self._create_session(rtmw_device)
        self._input_name = self._session.get_inputs()[0].name
        providers = (
            self._session.get_providers()
            if hasattr(self._session, "get_providers")
            else ["injected"]
        )
        self.provider = str(providers[0]) if providers else "unknown"

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        yolo_result = self.yolo_backend.detect(frame, timestamp_ms=timestamp_ms)
        if not yolo_result.success or not yolo_result.keypoints:
            return self._missing_result(yolo_result)

        height, width = frame.shape[:2]
        bbox_pixels = self._bbox_pixels(yolo_result, width, height)
        if bbox_pixels is None:
            return self._fallback_result(
                yolo_result,
                rtmw_error="YOLO target did not provide a usable bounding box",
            )

        started = time.perf_counter()
        try:
            coordinates, scores = self._infer(frame, bbox_pixels)
            rtmw_time_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            rtmw_time_ms = (time.perf_counter() - started) * 1000.0
            return self._fallback_result(
                yolo_result,
                rtmw_time_ms=rtmw_time_ms,
                rtmw_error=f"{type(exc).__name__}: {exc}",
            )

        rtmw_body = self._body_keypoints(coordinates, scores, width, height)
        match_distance, match_points = self._identity_match(
            yolo_result.keypoints,
            rtmw_body,
        )
        identity_matched = (
            match_distance is not None
            and match_points >= self.min_match_points
            and match_distance <= self.max_match_distance
        )
        if not identity_matched:
            return self._fallback_result(
                yolo_result,
                rtmw_time_ms=rtmw_time_ms,
                match_distance=match_distance,
                match_points=match_points,
            )

        keypoints = self._mediapipe33_keypoints(
            coordinates,
            scores,
            width,
            height,
        )
        extra = {
            **dict(yolo_result.extra),
            "identity_matched": True,
            "identity_match_distance": match_distance,
            "identity_match_points": match_points,
            "rtmw_wholebody_available": True,
            "rtmw_keypoint_count": int(coordinates.shape[0]),
            "rtmw_provider": self.provider,
            "rtmw_inference_time_ms": rtmw_time_ms,
            "rtmw_hand_keypoints": self._hand_keypoints(
                coordinates,
                scores,
                width,
                height,
            ),
        }
        return PoseResult(
            keypoints=keypoints,
            connections=MEDIAPIPE_CONNECTIONS,
            model_name=self.model_name,
            num_keypoints=len(keypoints),
            success=True,
            inference_time_ms=yolo_result.inference_time_ms + rtmw_time_ms,
            bbox=yolo_result.bbox,
            timestamp_ms=yolo_result.timestamp_ms,
            extra=extra,
        )

    def close(self) -> None:
        try:
            self.yolo_backend.close()
        finally:
            self._session = None

    def _create_session(self, requested_device: str) -> Any:
        if not self.rtmw_model_path.exists():
            raise FileNotFoundError(
                f"RTMW WholeBody model not found: {self.rtmw_model_path}"
            )
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "onnxruntime is required for RTMW WholeBody"
            ) from exc

        available = set(ort.get_available_providers())
        requested = str(requested_device or "auto").strip().lower()
        wants_cuda = requested in {"auto", "cuda", "gpu", "0"} or requested.startswith(
            "cuda:"
        )
        if wants_cuda and "CUDAExecutionProvider" in available and hasattr(
            ort,
            "preload_dlls",
        ):
            try:
                # ONNX Runtime can locate CUDA/cuDNN wheels installed in the
                # Python environment even when their bin folders are not in
                # the process-wide Windows PATH.
                cudnn_bin = (
                    Path(ort.__file__).resolve().parents[1]
                    / "nvidia"
                    / "cudnn"
                    / "bin"
                )
                if cudnn_bin.is_dir():
                    ort.preload_dlls(
                        cuda=False,
                        cudnn=True,
                        msvc=False,
                        directory=str(cudnn_bin),
                    )
                else:
                    ort.preload_dlls()
            except Exception:
                # In auto mode ORT still has a safe CPU provider below.
                pass
        providers: list[Any] = []
        if wants_cuda and "CUDAExecutionProvider" in available:
            device_id = 0
            if requested.startswith("cuda:"):
                try:
                    device_id = max(0, int(requested.split(":", 1)[1]))
                except ValueError:
                    device_id = 0
            providers.append(
                ("CUDAExecutionProvider", {"device_id": device_id})
            )
        providers.append("CPUExecutionProvider")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        return ort.InferenceSession(
            str(self.rtmw_model_path),
            sess_options=options,
            providers=providers,
        )

    def _infer(
        self,
        frame: np.ndarray,
        bbox_pixels: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        tensor, center, scale = self._preprocess(frame, bbox_pixels)
        outputs = self._session.run(None, {self._input_name: tensor})
        simcc = [
            np.asarray(output, dtype=np.float32)
            for output in outputs
            if np.asarray(output).ndim == 3
        ]
        if len(simcc) < 2:
            raise RuntimeError("RTMW model did not return SimCC x/y outputs")
        simcc_x, simcc_y = sorted(simcc[:2], key=lambda item: item.shape[-1])
        if simcc_x.shape[:2] != simcc_y.shape[:2]:
            raise RuntimeError("RTMW SimCC output shapes are incompatible")

        x_locs = np.argmax(simcc_x, axis=-1).astype(np.float32)
        y_locs = np.argmax(simcc_y, axis=-1).astype(np.float32)
        scores = (
            np.max(simcc_x, axis=-1) + np.max(simcc_y, axis=-1)
        ) * 0.5
        coordinates = np.stack((x_locs, y_locs), axis=-1) / 2.0
        coordinates = (
            coordinates
            / np.asarray(self.model_input_size, dtype=np.float32)
            * scale
            + center
            - scale / 2.0
        )
        coordinates[scores <= 0.0] = -1.0
        return coordinates[0], scores[0]

    def _preprocess(
        self,
        frame: np.ndarray,
        bbox_pixels: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = bbox_pixels
        center = np.asarray(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
            dtype=np.float32,
        )
        scale = np.asarray(
            [max(1.0, x2 - x1), max(1.0, y2 - y1)],
            dtype=np.float32,
        ) * 1.25
        input_width, input_height = self.model_input_size
        aspect_ratio = input_width / input_height
        if scale[0] > scale[1] * aspect_ratio:
            scale[1] = scale[0] / aspect_ratio
        else:
            scale[0] = scale[1] * aspect_ratio

        source_direction = np.asarray([0.0, -0.5 * scale[0]], dtype=np.float32)
        destination_direction = np.asarray(
            [0.0, -0.5 * input_width],
            dtype=np.float32,
        )
        source = np.zeros((3, 2), dtype=np.float32)
        destination = np.zeros((3, 2), dtype=np.float32)
        source[0] = center
        source[1] = center + source_direction
        source[2] = self._third_point(source[0], source[1])
        destination[0] = (input_width * 0.5, input_height * 0.5)
        destination[1] = destination[0] + destination_direction
        destination[2] = self._third_point(destination[0], destination[1])
        transform = cv2.getAffineTransform(source, destination)
        resized = cv2.warpAffine(
            frame,
            transform,
            (input_width, input_height),
            flags=cv2.INTER_LINEAR,
        )
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (
            resized.astype(np.float32)
            - np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
        ) / np.asarray((58.395, 57.12, 57.375), dtype=np.float32)
        tensor = np.ascontiguousarray(
            normalized.transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        return tensor, center, scale

    @staticmethod
    def _third_point(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        direction = first - second
        return second + np.asarray(
            [-direction[1], direction[0]],
            dtype=np.float32,
        )

    def _body_keypoints(
        self,
        coordinates: np.ndarray,
        scores: np.ndarray,
        width: int,
        height: int,
    ) -> list[Keypoint]:
        return [
            self._point_from_index(
                name,
                index,
                coordinates,
                scores,
                width,
                height,
            )
            for name, index in COCO133_BODY_INDEX.items()
        ]

    def _mediapipe33_keypoints(
        self,
        coordinates: np.ndarray,
        scores: np.ndarray,
        width: int,
        height: int,
    ) -> list[Keypoint]:
        resolved: dict[str, Keypoint] = {
            point.name: point
            for point in self._body_keypoints(
                coordinates,
                scores,
                width,
                height,
            )
        }
        for name, index in COCO133_SUPPLEMENTAL_INDEX.items():
            resolved[name] = self._point_from_index(
                name,
                index,
                coordinates,
                scores,
                width,
                height,
            )
        for name, pair in COCO133_FOOT_INDEX_PAIRS.items():
            resolved[name] = self._average_points(
                name,
                pair,
                coordinates,
                scores,
                width,
                height,
            )
        return [
            resolved.get(name) or self._missing_keypoint(name)
            for name in MEDIAPIPE_33_NAMES
        ]

    def _hand_keypoints(
        self,
        coordinates: np.ndarray,
        scores: np.ndarray,
        width: int,
        height: int,
    ) -> dict[str, tuple[Keypoint, ...]]:
        hands: dict[str, tuple[Keypoint, ...]] = {}
        for side, indices in COCO133_HAND_RANGES.items():
            points = tuple(
                self._point_from_index(
                    hand_landmark_name(side, hand_index),
                    wholebody_index,
                    coordinates,
                    scores,
                    width,
                    height,
                )
                for hand_index, wholebody_index in enumerate(indices)
            )
            if any(self._usable(point) for point in points):
                hands[side] = points
        return hands

    def _point_from_index(
        self,
        name: str,
        index: int,
        coordinates: np.ndarray,
        scores: np.ndarray,
        width: int,
        height: int,
    ) -> Keypoint:
        if index >= len(coordinates) or index >= len(scores):
            return self._missing_keypoint(name)
        x_pixel, y_pixel = coordinates[index][:2]
        confidence = float(scores[index])
        if (
            not isfinite(float(x_pixel))
            or not isfinite(float(y_pixel))
            or not isfinite(confidence)
            or width <= 0
            or height <= 0
        ):
            return self._missing_keypoint(name)
        return Keypoint(
            name=name,
            x=max(0.0, min(1.0, float(x_pixel) / width)),
            y=max(0.0, min(1.0, float(y_pixel) / height)),
            z=0.0,
            confidence=max(0.0, min(1.0, confidence)),
            source_model="rtmw-wholebody",
        )

    def _average_points(
        self,
        name: str,
        indices: tuple[int, int],
        coordinates: np.ndarray,
        scores: np.ndarray,
        width: int,
        height: int,
    ) -> Keypoint:
        points = [
            self._point_from_index(
                name,
                index,
                coordinates,
                scores,
                width,
                height,
            )
            for index in indices
        ]
        usable = [point for point in points if self._usable(point)]
        if not usable:
            return self._missing_keypoint(name)
        return Keypoint(
            name=name,
            x=sum(point.x for point in usable) / len(usable),
            y=sum(point.y for point in usable) / len(usable),
            z=0.0,
            confidence=min(point.confidence for point in usable),
            source_model="rtmw-wholebody",
        )

    def _identity_match(
        self,
        yolo_keypoints: Sequence[Keypoint],
        rtmw_keypoints: Sequence[Keypoint],
    ) -> tuple[float | None, int]:
        yolo_by_name = {point.name: point for point in yolo_keypoints}
        rtmw_by_name = {point.name: point for point in rtmw_keypoints}
        distances = [
            hypot(
                yolo_by_name[name].x - rtmw_by_name[name].x,
                yolo_by_name[name].y - rtmw_by_name[name].y,
            )
            for name in IDENTITY_MATCH_NAMES
            if name in yolo_by_name
            and name in rtmw_by_name
            and self._usable(yolo_by_name[name])
            and self._usable(rtmw_by_name[name])
        ]
        if len(distances) < self.min_match_points:
            return None, len(distances)
        return float(median(distances)), len(distances)

    def _bbox_pixels(
        self,
        result: PoseResult,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float] | None:
        pixels = result.extra.get("bbox_pixels")
        if isinstance(pixels, (list, tuple)) and len(pixels) >= 4:
            values = tuple(float(value) for value in pixels[:4])
            if all(isfinite(value) for value in values):
                return values
        if result.bbox is None:
            return None
        x1, y1, x2, y2 = result.bbox
        return x1 * width, y1 * height, x2 * width, y2 * height

    def _missing_result(self, yolo_result: PoseResult) -> PoseResult:
        return PoseResult(
            keypoints=[],
            connections=MEDIAPIPE_CONNECTIONS,
            model_name=self.model_name,
            num_keypoints=0,
            success=False,
            inference_time_ms=yolo_result.inference_time_ms,
            bbox=yolo_result.bbox,
            timestamp_ms=yolo_result.timestamp_ms,
            extra={
                **dict(yolo_result.extra),
                "identity_matched": False,
                "identity_match_distance": None,
                "identity_match_points": 0,
                "rtmw_wholebody_available": False,
                "rtmw_provider": self.provider,
            },
        )

    def _fallback_result(
        self,
        yolo_result: PoseResult,
        *,
        rtmw_time_ms: float = 0.0,
        rtmw_error: str | None = None,
        match_distance: float | None = None,
        match_points: int = 0,
    ) -> PoseResult:
        yolo_by_name = {point.name: point for point in yolo_result.keypoints}
        keypoints = [
            (
                yolo_by_name.get(name)
                if name in COCO_17_NAMES
                else None
            )
            or self._missing_keypoint(name)
            for name in MEDIAPIPE_33_NAMES
        ]
        extra = {
            **dict(yolo_result.extra),
            "identity_matched": False,
            "identity_match_distance": match_distance,
            "identity_match_points": match_points,
            "rtmw_wholebody_available": False,
            "rtmw_provider": self.provider,
            "rtmw_inference_time_ms": rtmw_time_ms,
            "rtmw_hand_keypoints": {},
        }
        if rtmw_error:
            extra["rtmw_error"] = rtmw_error
        return PoseResult(
            keypoints=keypoints,
            connections=MEDIAPIPE_CONNECTIONS,
            model_name=self.model_name,
            num_keypoints=len(keypoints),
            success=True,
            inference_time_ms=yolo_result.inference_time_ms + rtmw_time_ms,
            bbox=yolo_result.bbox,
            timestamp_ms=yolo_result.timestamp_ms,
            extra=extra,
        )

    def _usable(self, point: Keypoint) -> bool:
        return (
            point.confidence >= self.match_confidence
            and isfinite(point.x)
            and isfinite(point.y)
        )

    def _missing_keypoint(self, name: str) -> Keypoint:
        return Keypoint(
            name=name,
            x=nan,
            y=nan,
            z=nan,
            confidence=0.0,
            source_model=self.model_name,
        )


__all__ = [
    "DEFAULT_RTMW_MODEL",
    "RTMW_INPUT_SIZE",
    "YoloRtmwWholeBodyBackend",
]
