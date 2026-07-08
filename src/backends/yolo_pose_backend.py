from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite, nan
from typing import Iterable, Literal

import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.utils.keypoint_schema import COCO_17_NAMES, COCO_CONNECTIONS
from src.utils.roi import BBox, clamp_bbox, normalize_bbox
from src.utils.ultralytics_config import ensure_ultralytics_config_dir


TargetSelect = Literal["confidence", "area"]


@dataclass(frozen=True)
class YoloPoseCandidate:
    bbox_pixels: BBox
    confidence: float
    keypoint_xy_pixels: np.ndarray
    keypoint_confidence: np.ndarray

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_pixels
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class YoloPoseBackend:
    model_name = "yolo-pose"

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        target_select: TargetSelect = "confidence",
        device: str = "",
    ) -> None:
        ensure_ultralytics_config_dir()
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 ultralytics，无法使用 --backend yolo-pose。请运行 pip install ultralytics，或改用 --backend mediapipe。"
            ) from exc

        if target_select not in {"confidence", "area"}:
            raise ValueError("--target-select must be confidence or area")
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.target_select: TargetSelect = target_select
        self.device = device.strip()

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        started = time.perf_counter()
        predict_kwargs = {"classes": [0], "verbose": False}
        if self.device:
            predict_kwargs["device"] = self.device
        results = self.model.predict(frame, **predict_kwargs)
        inference_time_ms = (time.perf_counter() - started) * 1000.0
        return self._to_pose_result(results, frame.shape, inference_time_ms, timestamp_ms)

    def close(self) -> None:
        return None

    def _to_pose_result(
        self,
        results: object,
        frame_shape: tuple[int, ...],
        inference_time_ms: float,
        timestamp_ms: int | None,
    ) -> PoseResult:
        height, width = frame_shape[:2]
        candidates = self._collect_candidates(results, width, height)
        selected = self._select_candidate(candidates)
        if selected is None:
            return PoseResult(
                keypoints=[],
                connections=COCO_CONNECTIONS,
                model_name=self.model_name,
                num_keypoints=0,
                success=False,
                inference_time_ms=inference_time_ms,
                timestamp_ms=timestamp_ms,
                extra={"raw_result": results},
            )

        keypoints = self._to_keypoints(selected, width, height)
        return PoseResult(
            keypoints=keypoints,
            connections=COCO_CONNECTIONS,
            model_name=self.model_name,
            num_keypoints=len(COCO_17_NAMES),
            success=True,
            inference_time_ms=inference_time_ms,
            bbox=normalize_bbox(selected.bbox_pixels, width, height),
            timestamp_ms=timestamp_ms,
            extra={
                "raw_result": results,
                "bbox_pixels": selected.bbox_pixels,
                "target_confidence": selected.confidence,
                "target_area": selected.area,
                "device": getattr(self, "device", "") or "auto",
            },
        )

    def _collect_candidates(self, results: object, width: int, height: int) -> list[YoloPoseCandidate]:
        candidates: list[YoloPoseCandidate] = []
        for result in _iter_results(results):
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None:
                continue
            keypoint_xy = _as_numpy(getattr(keypoints, "xy", None))
            if keypoint_xy is None:
                continue
            keypoint_xy = _ensure_keypoint_xy_batch(keypoint_xy)
            if keypoint_xy is None:
                continue

            keypoint_confidence = _as_numpy(getattr(keypoints, "conf", None))
            keypoint_confidence = _ensure_keypoint_confidence_batch(keypoint_confidence, keypoint_xy.shape)

            boxes = getattr(result, "boxes", None)
            box_xyxy = _as_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else None
            box_confidence = _as_numpy(getattr(boxes, "conf", None)) if boxes is not None else None
            box_xyxy = _ensure_box_batch(box_xyxy)
            box_confidence = _ensure_box_confidence_batch(box_confidence, keypoint_xy.shape[0])

            for index in range(keypoint_xy.shape[0]):
                xy = keypoint_xy[index]
                confidences = keypoint_confidence[index]
                bbox = self._candidate_bbox(index, xy, confidences, box_xyxy, width, height)
                if bbox is None:
                    continue
                confidence = float(box_confidence[index]) if index < len(box_confidence) else float(np.nanmean(confidences))
                if not isfinite(confidence):
                    confidence = 0.0
                candidate = YoloPoseCandidate(
                    bbox_pixels=bbox,
                    confidence=confidence,
                    keypoint_xy_pixels=xy,
                    keypoint_confidence=confidences,
                )
                if candidate.area > 1.0:
                    candidates.append(candidate)
        return candidates

    def _select_candidate(self, candidates: list[YoloPoseCandidate]) -> YoloPoseCandidate | None:
        if not candidates:
            return None
        if self.target_select == "area":
            return max(candidates, key=lambda item: item.area)
        return max(candidates, key=lambda item: item.confidence)

    def _candidate_bbox(
        self,
        index: int,
        xy: np.ndarray,
        confidences: np.ndarray,
        box_xyxy: np.ndarray | None,
        width: int,
        height: int,
    ) -> BBox | None:
        if box_xyxy is not None and index < len(box_xyxy):
            raw = box_xyxy[index]
            if len(raw) >= 4 and all(isfinite(float(value)) for value in raw[:4]):
                return clamp_bbox(tuple(float(value) for value in raw[:4]), width, height)

        valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1]) & (confidences >= 0.2)
        if not np.any(valid):
            return None
        xs = xy[valid, 0]
        ys = xy[valid, 1]
        return clamp_bbox((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), width, height)

    def _to_keypoints(self, candidate: YoloPoseCandidate, width: int, height: int) -> list[Keypoint]:
        keypoints: list[Keypoint] = []
        for index, name in enumerate(COCO_17_NAMES):
            if index >= len(candidate.keypoint_xy_pixels):
                keypoints.append(Keypoint(name=name, x=nan, y=nan, z=nan, confidence=0.0, source_model=self.model_name))
                continue

            x_pixel, y_pixel = candidate.keypoint_xy_pixels[index][:2]
            confidence = float(candidate.keypoint_confidence[index]) if index < len(candidate.keypoint_confidence) else 0.0
            x = _normalize_pixel(float(x_pixel), width)
            y = _normalize_pixel(float(y_pixel), height)
            if not isfinite(confidence):
                confidence = 0.0
            keypoints.append(
                Keypoint(
                    name=name,
                    x=x,
                    y=y,
                    z=0.0,
                    confidence=max(0.0, min(1.0, confidence)),
                    source_model=self.model_name,
                )
            )
        return keypoints


def _iter_results(results: object) -> Iterable[object]:
    if results is None:
        return ()
    if isinstance(results, (list, tuple)):
        return results
    return (results,)


def _as_numpy(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=float)
    return array


def _ensure_keypoint_xy_batch(array: np.ndarray) -> np.ndarray | None:
    if array.ndim == 2 and array.shape[-1] >= 2:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[-1] < 2:
        return None
    return array[:, :, :2]


def _ensure_keypoint_confidence_batch(array: np.ndarray | None, xy_shape: tuple[int, ...]) -> np.ndarray:
    person_count, keypoint_count = xy_shape[:2]
    if array is None:
        return np.ones((person_count, keypoint_count), dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        return np.ones((person_count, keypoint_count), dtype=float)
    padded = np.zeros((person_count, keypoint_count), dtype=float)
    rows = min(person_count, array.shape[0])
    cols = min(keypoint_count, array.shape[1])
    padded[:rows, :cols] = array[:rows, :cols]
    return padded


def _ensure_box_batch(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    if array.ndim == 1 and array.shape[0] >= 4:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 4:
        return None
    return array[:, :4]


def _ensure_box_confidence_batch(array: np.ndarray | None, person_count: int) -> np.ndarray:
    if array is None:
        return np.zeros(person_count, dtype=float)
    array = np.asarray(array, dtype=float).reshape(-1)
    if len(array) >= person_count:
        return array[:person_count]
    padded = np.zeros(person_count, dtype=float)
    padded[: len(array)] = array
    return padded


def _normalize_pixel(value: float, size: int) -> float:
    if size <= 0 or not isfinite(value):
        return nan
    return min(1.0, max(0.0, value / float(size)))
