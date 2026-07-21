from __future__ import annotations

import pytest

from src.backends.base import Keypoint, PoseResult
from src.realtime.feedback_engine import FeedbackEngine
from src.utils.angle_utils import angle, knee_angle
from src.utils.metrics import RealtimeMetrics
from src.utils.smoothing import KeypointSmoother


def _pose(points: list[Keypoint], success: bool = True, timestamp_ms: int = 0) -> PoseResult:
    return PoseResult(
        keypoints=points,
        connections=(),
        model_name="test",
        num_keypoints=len(points),
        success=success,
        inference_time_ms=5.0,
        timestamp_ms=timestamp_ms,
    )


def test_ema_smoothing_does_not_let_low_confidence_pollute_state() -> None:
    smoother = KeypointSmoother(mode="ema", ema_alpha=0.5, min_confidence=0.2)
    first = _pose([Keypoint("left_knee", 0.0, 0.0, confidence=1.0)], timestamp_ms=0)
    second = _pose([Keypoint("left_knee", 1.0, 1.0, confidence=0.05)], timestamp_ms=33)
    third = _pose([Keypoint("left_knee", 1.0, 1.0, confidence=1.0)], timestamp_ms=66)

    smoother.smooth_result(first)
    low_conf = smoother.smooth_result(second)
    recovered = smoother.smooth_result(third)

    assert low_conf.keypoints[0].x == 1.0
    assert recovered.keypoints[0].x == pytest.approx(0.5)


def test_smoother_holds_last_pose_for_short_detection_drop() -> None:
    smoother = KeypointSmoother(mode="ema", ema_alpha=0.5, max_missing_frames=2)
    first = _pose([Keypoint("left_hip", 0.5, 0.5, confidence=1.0, source_model="mediapipe")], timestamp_ms=0)
    missing = _pose([], success=False, timestamp_ms=33)

    smoother.smooth_result(first)
    held = smoother.smooth_result(missing)

    assert held.success
    assert held.extra["stabilized_hold"] is True
    assert held.extra["hold_frames"] == 1
    assert held.keypoints[0].name == "left_hip"
    assert held.keypoints[0].x == pytest.approx(0.5)

    smoother.smooth_result(missing)
    expired = smoother.smooth_result(missing)

    assert not expired.success
    assert expired.keypoints == []


def test_smoother_suppresses_body_jump_when_hand_overlaps_joint() -> None:
    smoother = KeypointSmoother(mode="ema", ema_alpha=0.8)
    first = _pose(
        [
            Keypoint("left_hip", 0.50, 0.50, confidence=1.0, source_model="mediapipe"),
            Keypoint("left_wrist", 0.20, 0.20, confidence=1.0, source_model="mediapipe"),
        ],
        timestamp_ms=0,
    )
    occluded = _pose(
        [
            Keypoint("left_hip", 0.80, 0.50, confidence=1.0, source_model="mediapipe"),
            Keypoint("left_wrist", 0.80, 0.50, confidence=1.0, source_model="mediapipe"),
        ],
        timestamp_ms=33,
    )

    smoother.smooth_result(first)
    smoothed = smoother.smooth_result(occluded)

    assert smoothed.keypoints[0].x == pytest.approx(0.5)
    assert smoothed.extra["occlusion_guarded_keypoints"] == ("left_hip",)


def test_smoother_allows_same_jump_when_occlusion_guard_disabled() -> None:
    smoother = KeypointSmoother(mode="ema", ema_alpha=0.8, occlusion_guard=False)
    first = _pose(
        [
            Keypoint("left_hip", 0.50, 0.50, confidence=1.0, source_model="mediapipe"),
            Keypoint("left_wrist", 0.20, 0.20, confidence=1.0, source_model="mediapipe"),
        ],
        timestamp_ms=0,
    )
    moved = _pose(
        [
            Keypoint("left_hip", 0.80, 0.50, confidence=1.0, source_model="mediapipe"),
            Keypoint("left_wrist", 0.80, 0.50, confidence=1.0, source_model="mediapipe"),
        ],
        timestamp_ms=33,
    )

    smoother.smooth_result(first)
    smoothed = smoother.smooth_result(moved)

    assert smoothed.keypoints[0].x > 0.7
    assert "occlusion_guarded_keypoints" not in smoothed.extra


def test_angle_utils_return_none_for_low_confidence_points() -> None:
    assert angle(
        Keypoint("a", 0.0, 0.0, confidence=1.0),
        Keypoint("b", 1.0, 0.0, confidence=0.1),
        Keypoint("c", 1.0, 1.0, confidence=1.0),
    ) is None


def test_knee_angle_for_named_keypoints() -> None:
    points = [
        Keypoint("left_hip", 0.0, 0.0, confidence=1.0),
        Keypoint("left_knee", 1.0, 0.0, confidence=1.0),
        Keypoint("left_ankle", 1.0, 1.0, confidence=1.0),
    ]
    assert knee_angle(points, "left") == pytest.approx(90.0)


def test_feedback_engine_debounces_person_lost() -> None:
    engine = FeedbackEngine(abnormal_frames_required=3)
    missing = _pose([], success=False)

    assert not engine.update(missing, {}).person_lost
    assert not engine.update(missing, {}).person_lost
    state = engine.update(missing, {})

    assert state.person_lost
    assert state.message == "No person detected"


def test_metrics_can_write_csv(tmp_path) -> None:
    metrics = RealtimeMetrics(backend="mediapipe", smoothing="one-euro")
    pose = _pose([Keypoint("left_knee", 0.0, 0.0, confidence=1.0)], timestamp_ms=0)

    metrics.update(pose, {"left_knee_angle": 90.0}, frame_started=1.0, frame_finished=1.01)
    metrics.update(pose, {"left_knee_angle": 91.0}, frame_started=1.02, frame_finished=1.03)
    snapshot = metrics.snapshot()
    path = tmp_path / "metrics.csv"
    metrics.write_csv(path)

    text = path.read_text(encoding="utf-8")
    assert "success_rate" in text
    assert "mediapipe" in text
    assert "p50_inference_time_ms" in text
    assert "p95_end_to_end_latency_ms" in text
    assert snapshot.p50_end_to_end_latency_ms == pytest.approx(10.0)
    assert snapshot.p95_end_to_end_latency_ms == pytest.approx(10.0)


def test_metrics_records_runtime_backend_switch_history() -> None:
    metrics = RealtimeMetrics(backend="mediapipe", smoothing="one-euro", backend_device="cpu")

    metrics.set_backend("yolo-pose", "0")
    metrics.set_backend("mediapipe", "cpu")

    assert metrics.backend == "mediapipe->yolo-pose->mediapipe"
    assert metrics.backend_device == "cpu->0->cpu"


def test_metrics_counts_stability_guards() -> None:
    metrics = RealtimeMetrics(backend="mediapipe", smoothing="one-euro")
    pose = _pose(
        [Keypoint("left_hip", 0.0, 0.0, confidence=1.0)],
        timestamp_ms=0,
    )
    pose = PoseResult(
        keypoints=pose.keypoints,
        connections=pose.connections,
        model_name=pose.model_name,
        num_keypoints=pose.num_keypoints,
        success=pose.success,
        inference_time_ms=pose.inference_time_ms,
        timestamp_ms=pose.timestamp_ms,
        extra={"stabilized_hold": True, "occlusion_guarded_keypoints": ("left_hip", "right_hip")},
    )

    snapshot = metrics.update(pose, {}, frame_started=1.0, frame_finished=1.01)

    assert snapshot.stabilized_hold_count == 1
    assert snapshot.occlusion_guard_count == 2


def test_metrics_exposes_realtime_drop_counts() -> None:
    metrics = RealtimeMetrics(backend="mediapipe", smoothing="one-euro")

    metrics.set_realtime_drop_counts(busy=4, stale=2, camera_overwrite=7)
    snapshot = metrics.snapshot()

    assert snapshot.pose_busy_drop_count == 4
    assert snapshot.pose_stale_drop_count == 2
    assert snapshot.camera_overwrite_drop_count == 7
