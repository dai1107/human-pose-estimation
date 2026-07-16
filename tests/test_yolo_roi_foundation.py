from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.backends.base import Keypoint, PoseResult
from src.detectors.yolo_person_detector import PersonDetection, YoloPersonDetector
from src.detectors.yolo_person_detector import ensure_ultralytics_config_dir
from src.fusion.yolo_roi_mediapipe import YoloRoiMediaPipeFusion
from src.utils.metrics import RealtimeMetrics
from src.utils.roi import crop_roi, restore_keypoints_from_roi


class FakeBackend:
    model_name = "fake"

    def detect(self, frame, timestamp_ms=None):
        return PoseResult(
            keypoints=[Keypoint("left_knee", 0.5, 0.5, confidence=1.0)],
            connections=(),
            model_name="fake",
            num_keypoints=1,
            success=True,
            inference_time_ms=3.0,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        return None


class FakeDetector:
    def __init__(self, bbox):
        self.bbox = bbox

    def update(self, frame, force=False):
        del frame, force
        return PersonDetection(
            bbox=self.bbox,
            success=self.bbox is not None,
            reused=False,
            inference_time_ms=7.0,
            lost_count=0 if self.bbox is not None else 1,
        )


def test_restore_keypoints_from_roi_maps_normalized_points_to_original_frame() -> None:
    points = [Keypoint("left_knee", 0.5, 0.5, confidence=1.0)]
    restored = restore_keypoints_from_roi(points, (20.0, 10.0, 60.0, 50.0), (40, 40, 3), (100, 100, 3))

    assert restored[0].x == 0.4
    assert restored[0].y == 0.3


def test_crop_roi_returns_cropped_image_and_pixel_bbox() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    roi, bbox = crop_roi(frame, (10.2, 20.2, 60.2, 70.2))

    assert roi.shape[:2] == (50, 50)
    assert bbox == (10.0, 20.0, 60.0, 70.0)


def test_yolo_roi_mediapipe_fusion_restores_result_coordinates() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fusion = YoloRoiMediaPipeFusion(FakeBackend(), FakeDetector((20.0, 10.0, 60.0, 50.0)))

    result, stats = fusion.detect(frame, timestamp_ms=100)

    assert stats.roi_success
    assert result.keypoints[0].x == 0.4
    assert result.keypoints[0].y == 0.3


def test_yolo_roi_mediapipe_fusion_falls_back_without_bbox() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fusion = YoloRoiMediaPipeFusion(FakeBackend(), FakeDetector(None))

    result, stats = fusion.detect(frame, timestamp_ms=100)

    assert result.success
    assert stats.fallback_to_full_frame
    assert stats.bbox_lost


def test_yolo_person_detector_tracking_prefers_continuity_over_new_confidence_peak() -> None:
    detector = YoloPersonDetector.__new__(YoloPersonDetector)
    detector.target_select = "tracking"
    detector._last_bbox = None
    first = [
        SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=np.array([[5.0, 5.0, 25.0, 70.0], [30.0, 5.0, 75.0, 95.0]]),
                conf=np.array([0.98, 0.72]),
            )
        )
    ]

    selected = detector._select_bbox(first, 100, 100)
    detector._last_bbox = selected
    next_frame = [
        SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=np.array([[4.0, 5.0, 26.0, 71.0], [32.0, 6.0, 77.0, 96.0]]),
                conf=np.array([0.999, 0.60]),
            )
        )
    ]

    tracked = detector._select_bbox(next_frame, 100, 100)

    assert selected == (30.0, 5.0, 75.0, 95.0)
    assert tracked == (32.0, 6.0, 77.0, 96.0)


def test_metrics_csv_includes_roi_fields(tmp_path) -> None:
    metrics = RealtimeMetrics(
        backend="mediapipe",
        smoothing="one-euro",
        person_detector="yolo",
        fusion="yolo-roi-mediapipe",
        detector_every_n=5,
    )
    result = PoseResult(
        keypoints=[Keypoint("left_knee", 0.0, 0.0, confidence=1.0)],
        connections=(),
        model_name="fake",
        num_keypoints=1,
        success=True,
        inference_time_ms=3.0,
    )
    metrics.update(
        result,
        {"left_knee_angle": 90.0},
        frame_started=1.0,
        frame_finished=1.01,
        roi_enabled=True,
        roi_success=True,
        yolo_detection_time_ms=7.0,
        bbox_reused=True,
    )

    path = tmp_path / "metrics.csv"
    metrics.write_csv(path)
    text = path.read_text(encoding="utf-8")

    assert "roi_success_rate" in text
    assert "avg_yolo_detection_time_ms" in text
    assert "yolo-roi-mediapipe" in text


def test_ultralytics_config_dir_uses_workspace_cache() -> None:
    path = ensure_ultralytics_config_dir()

    assert path.name == "ultralytics"
    assert path.parent.name == ".cache"
