from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.backends.yolo_pose_backend import YoloPoseBackend
from src.utils.keypoint_schema import COCO_17_NAMES
from src.utils.metrics import RealtimeMetrics


def _backend(target_select: str = "confidence") -> YoloPoseBackend:
    backend = YoloPoseBackend.__new__(YoloPoseBackend)
    backend.target_select = target_select
    backend._tracked_bbox = None
    backend._track_lost_frames = 0
    return backend


def _fake_result() -> SimpleNamespace:
    xy = np.zeros((2, 17, 2), dtype=float)
    xy[0, :, 0] = 20.0
    xy[0, :, 1] = 30.0
    xy[1, :, 0] = 60.0
    xy[1, :, 1] = 70.0
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.array([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 100.0, 90.0]], dtype=float),
            conf=np.array([0.95, 0.50], dtype=float),
        ),
        keypoints=SimpleNamespace(
            xy=xy,
            conf=np.full((2, 17), 0.8, dtype=float),
        ),
    )


def test_yolo_pose_result_uses_coco17_and_confidence_selection() -> None:
    result = _backend("confidence")._to_pose_result([_fake_result()], (100, 100, 3), 12.5, 123)

    assert result.success
    assert result.model_name == "yolo-pose"
    assert result.num_keypoints == 17
    assert [point.name for point in result.keypoints] == list(COCO_17_NAMES)
    assert result.bbox == pytest.approx((0.1, 0.2, 0.5, 0.8))
    assert result.keypoints[0].x == pytest.approx(0.2)
    assert result.keypoints[0].y == pytest.approx(0.3)
    assert result.inference_time_ms == 12.5
    assert result.timestamp_ms == 123


def test_yolo_pose_can_select_largest_area_person() -> None:
    result = _backend("area")._to_pose_result([_fake_result()], (100, 100, 3), 7.0, None)

    assert result.success
    assert result.bbox == pytest.approx((0.0, 0.0, 0.99, 0.9))
    assert result.keypoints[0].x == pytest.approx(0.6)
    assert result.keypoints[0].y == pytest.approx(0.7)


def test_yolo_pose_tracking_locks_onto_same_athlete_across_frames() -> None:
    backend = _backend("tracking")
    first = backend._to_pose_result([_fake_result()], (100, 100, 3), 7.0, None)
    changed = _fake_result()
    changed.boxes.conf = np.array([0.99, 0.35], dtype=float)
    changed.keypoints.xy[0, :, 0] = 15.0
    changed.keypoints.xy[1, :, 0] = 62.0

    second = backend._to_pose_result([changed], (100, 100, 3), 7.0, None)

    assert first.keypoints[0].x == pytest.approx(0.6)
    assert second.keypoints[0].x == pytest.approx(0.62)
    assert second.extra["target_tracking"] is True


def test_yolo_pose_tracking_does_not_immediately_switch_to_distant_bystander() -> None:
    backend = _backend("tracking")
    first = _fake_result()
    first.boxes.xyxy = np.array([[35.0, 10.0, 75.0, 95.0]], dtype=float)
    first.boxes.conf = np.array([0.80], dtype=float)
    first.keypoints.xy = first.keypoints.xy[1:2]
    first.keypoints.conf = first.keypoints.conf[1:2]
    backend._to_pose_result([first], (100, 100, 3), 7.0, None)

    bystander = _fake_result()
    bystander.boxes.xyxy = np.array([[0.0, 0.0, 18.0, 40.0]], dtype=float)
    bystander.boxes.conf = np.array([0.99], dtype=float)
    bystander.keypoints.xy = bystander.keypoints.xy[:1]
    bystander.keypoints.conf = bystander.keypoints.conf[:1]
    missing = backend._to_pose_result([bystander], (100, 100, 3), 7.0, None)

    assert not missing.success
    assert backend._track_lost_frames == 1


def test_yolo_pose_returns_unsuccessful_result_without_person() -> None:
    result = _backend()._to_pose_result([], (100, 100, 3), 2.0, None)

    assert not result.success
    assert result.num_keypoints == 0
    assert result.keypoints == []


def test_yolo_pose_missing_ultralytics_message_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics":
            raise ModuleNotFoundError("ultralytics")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="未安装 ultralytics"):
        YoloPoseBackend("missing.pt")


def test_metrics_include_keypoint_confidence_and_missing_rates(tmp_path) -> None:
    def pose(left_shoulder_confidence: float) -> PoseResult:
        keypoints = [
            Keypoint(name, 0.5, 0.5, confidence=1.0, source_model="yolo-pose")
            for name in (
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            )
        ]
        keypoints.append(Keypoint("left_shoulder", 0.5, 0.5, confidence=left_shoulder_confidence, source_model="yolo-pose"))
        return PoseResult(
            keypoints=keypoints,
            connections=(),
            model_name="yolo-pose",
            num_keypoints=17,
            success=True,
            inference_time_ms=4.0,
        )

    metrics = RealtimeMetrics(backend="yolo-pose", smoothing="one-euro")
    metrics.update(pose(1.0), {}, frame_started=1.0, frame_finished=1.01)
    metrics.update(pose(0.0), {}, frame_started=1.02, frame_finished=1.03)
    snapshot = metrics.snapshot()

    assert snapshot.num_keypoints == 17
    assert snapshot.missing_rate_shoulder == pytest.approx(0.25)
    assert snapshot.missing_rate_hip == 0.0

    path = tmp_path / "metrics.csv"
    metrics.write_csv(path)
    text = path.read_text(encoding="utf-8")

    assert "avg_keypoint_confidence" in text
    assert "missing_rate_shoulder" in text
    assert "yolo-pose" in text
