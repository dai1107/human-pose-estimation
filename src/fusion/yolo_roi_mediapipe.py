from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backends.base import PoseBackend, PoseResult
from src.detectors.yolo_person_detector import YoloPersonDetector
from src.utils.roi import crop_roi, restore_result_from_roi


@dataclass(frozen=True)
class FusionFrameStats:
    roi_enabled: bool = False
    roi_success: bool = False
    yolo_detection_time_ms: float = 0.0
    bbox_reused: bool = False
    bbox_lost: bool = False
    fallback_to_full_frame: bool = False
    roi_bbox_pixels: tuple[float, float, float, float] | None = None


class YoloRoiMediaPipeFusion:
    def __init__(self, backend: PoseBackend, detector: YoloPersonDetector) -> None:
        self.backend = backend
        self.detector = detector

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> tuple[PoseResult, FusionFrameStats]:
        detection = self.detector.update(frame)
        if detection.bbox is None:
            result = self.backend.detect(frame, timestamp_ms=timestamp_ms)
            return result, FusionFrameStats(
                roi_enabled=False,
                roi_success=False,
                yolo_detection_time_ms=detection.inference_time_ms,
                bbox_reused=detection.reused,
                bbox_lost=True,
                fallback_to_full_frame=True,
            )

        roi, roi_bbox = crop_roi(frame, detection.bbox)
        if roi_bbox is None:
            result = self.backend.detect(frame, timestamp_ms=timestamp_ms)
            return result, FusionFrameStats(
                roi_enabled=False,
                roi_success=False,
                yolo_detection_time_ms=detection.inference_time_ms,
                bbox_reused=detection.reused,
                bbox_lost=False,
                fallback_to_full_frame=True,
            )

        roi_result = self.backend.detect(roi, timestamp_ms=timestamp_ms)
        if roi_result.success:
            restored = restore_result_from_roi(roi_result, roi_bbox, roi.shape, frame.shape)
            return restored, FusionFrameStats(
                roi_enabled=True,
                roi_success=True,
                yolo_detection_time_ms=detection.inference_time_ms,
                bbox_reused=detection.reused,
                bbox_lost=False,
                fallback_to_full_frame=False,
                roi_bbox_pixels=roi_bbox,
            )

        fallback = self.backend.detect(frame, timestamp_ms=timestamp_ms)
        return fallback, FusionFrameStats(
            roi_enabled=True,
            roi_success=False,
            yolo_detection_time_ms=detection.inference_time_ms,
            bbox_reused=detection.reused,
            bbox_lost=False,
            fallback_to_full_frame=True,
            roi_bbox_pixels=roi_bbox,
        )
