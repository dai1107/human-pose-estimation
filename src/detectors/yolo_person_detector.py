from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.utils.roi import clamp_bbox, expand_bbox, smooth_bbox
from src.utils.ultralytics_config import ensure_ultralytics_config_dir


TargetSelect = Literal["tracking", "confidence", "area"]


@dataclass(frozen=True)
class PersonDetection:
    bbox: tuple[float, float, float, float] | None
    success: bool
    reused: bool
    inference_time_ms: float
    lost_count: int = 0


class YoloPersonDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        every_n: int = 5,
        bbox_expand: float = 1.25,
        bbox_smoothing: float = 0.6,
        target_select: TargetSelect = "tracking",
        max_reuse_frames: int | None = None,
        device: str = "",
    ) -> None:
        ensure_ultralytics_config_dir()
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 ultralytics，请先运行 pip install ultralytics，或使用 --person-detector none。"
            ) from exc

        if target_select not in {"tracking", "confidence", "area"}:
            raise ValueError("--target-select must be tracking, confidence or area")
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.every_n = max(1, int(every_n))
        self.bbox_expand = max(1.0, float(bbox_expand))
        self.bbox_smoothing = max(0.0, min(1.0, float(bbox_smoothing)))
        self.target_select: TargetSelect = target_select
        self.device = device.strip()
        self.max_reuse_frames = max_reuse_frames if max_reuse_frames is not None else self.every_n * 3
        self._frame_index = 0
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._lost_count = 0

    def update(self, frame: np.ndarray, force: bool = False) -> PersonDetection:
        self._frame_index += 1
        should_detect = force or self._last_bbox is None or self._lost_count > 0 or (self._frame_index - 1) % self.every_n == 0
        if not should_detect and self._last_bbox is not None:
            return PersonDetection(
                bbox=self._last_bbox,
                success=True,
                reused=True,
                inference_time_ms=0.0,
                lost_count=self._lost_count,
            )

        started = time.perf_counter()
        predict_kwargs = {"classes": [0], "verbose": False}
        if self.device:
            predict_kwargs["device"] = self.device
        results = self.model.predict(frame, **predict_kwargs)
        inference_time_ms = (time.perf_counter() - started) * 1000.0
        bbox = self._select_bbox(results, frame.shape[1], frame.shape[0])

        if bbox is None:
            self._lost_count += 1
            if self._last_bbox is not None and self._lost_count <= self.max_reuse_frames:
                return PersonDetection(
                    bbox=self._last_bbox,
                    success=True,
                    reused=True,
                    inference_time_ms=inference_time_ms,
                    lost_count=self._lost_count,
                )
            self._last_bbox = None
            return PersonDetection(
                bbox=None,
                success=False,
                reused=False,
                inference_time_ms=inference_time_ms,
                lost_count=self._lost_count,
            )

        self._lost_count = 0
        expanded = clamp_bbox(expand_bbox(bbox, self.bbox_expand), frame.shape[1], frame.shape[0])
        if self._last_bbox is not None:
            expanded = smooth_bbox(self._last_bbox, expanded, self.bbox_smoothing)
        self._last_bbox = expanded
        return PersonDetection(
            bbox=expanded,
            success=True,
            reused=False,
            inference_time_ms=inference_time_ms,
            lost_count=0,
        )

    def reset(self) -> None:
        self._frame_index = 0
        self._last_bbox = None
        self._lost_count = 0

    def _select_bbox(self, results: object, width: int, height: int) -> tuple[float, float, float, float] | None:
        candidates: list[tuple[float, float, tuple[float, float, float, float]]] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            conf = getattr(boxes, "conf", None)
            if xyxy is None or conf is None:
                continue
            xyxy_array = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
            conf_array = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
            for raw_box, raw_confidence in zip(xyxy_array, conf_array):
                x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
                bbox = clamp_bbox((x1, y1, x2, y2), width, height)
                area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                if area <= 1.0:
                    continue
                candidates.append((float(raw_confidence), area, bbox))
        if not candidates:
            return None
        if self.target_select == "area":
            return max(candidates, key=lambda item: item[1])[2]
        if self.target_select == "confidence":
            return max(candidates, key=lambda item: item[0])[2]
        if self._last_bbox is not None:
            compatible = [
                item
                for item in candidates
                if _bbox_iou(self._last_bbox, item[2]) >= 0.08
                or _relative_center_distance(self._last_bbox, item[2]) <= 0.45
            ]
            if not compatible:
                return None
            return max(
                compatible,
                key=lambda item: (
                    0.75 * _bbox_iou(self._last_bbox, item[2])
                    + 0.15 * (1.0 - min(1.0, _relative_center_distance(self._last_bbox, item[2])))
                    + 0.10 * item[0]
                ),
            )[2]
        max_area = max(item[1] for item in candidates)
        return max(
            candidates,
            key=lambda item: _initial_athlete_score(item[0], item[1], item[2], width, height, max_area),
        )[2]


def _bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _relative_center_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    dx = (first[0] + first[2] - second[0] - second[2]) / 2.0
    dy = (first[1] + first[3] - second[1] - second[3]) / 2.0
    scale = max(1.0, ((first[2] - first[0]) ** 2 + (first[3] - first[1]) ** 2) ** 0.5)
    return (dx * dx + dy * dy) ** 0.5 / scale


def _initial_athlete_score(
    confidence: float,
    area: float,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    max_area: float,
) -> float:
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    dx = abs(center_x - width / 2.0) / max(1.0, width / 2.0)
    dy = abs(center_y - height / 2.0) / max(1.0, height / 2.0)
    center_score = 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5 / 2**0.5)
    return 0.35 * (area / max(1.0, max_area)) + 0.45 * center_score + 0.20 * confidence
