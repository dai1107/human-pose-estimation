from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import src.backends.mediapipe_backend as mediapipe_module
import src.realtime.capture as capture_module
from src.backends.base import PoseResult
from src.backends.mediapipe_backend import MediaPipeLiveStreamBackend
from src.realtime.capture import open_capture
from src.realtime.latest_frame import LatestFrameCamera
from src.realtime.scheduler import LatestOnlyMediaPipeScheduler, PoseAgeGate
from src.realtime.types import CapturedFrame, TimedPoseResult


def _frame(frame_id: int, timestamp_ns: int) -> CapturedFrame:
    image = np.full((2, 3, 3), frame_id, dtype=np.uint8)
    return CapturedFrame(
        frame_id=frame_id,
        capture_timestamp_ns=timestamp_ns,
        image=image,
        source="camera:0",
        width=3,
        height=2,
    )


def _pose(timestamp_ms: int = 0) -> PoseResult:
    return PoseResult(
        keypoints=[],
        connections=(),
        model_name="mediapipe",
        num_keypoints=0,
        success=False,
        inference_time_ms=0.0,
        timestamp_ms=timestamp_ms,
    )


class FiniteCapture:
    def __init__(self, images: list[np.ndarray]) -> None:
        self.images = list(images)
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.images:
            return True, self.images.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


class BlockingCapture:
    def __init__(self) -> None:
        self.released = threading.Event()

    def read(self) -> tuple[bool, None]:
        self.released.wait(1.0)
        return False, None

    def release(self) -> None:
        self.released.set()


class FakeAsyncBackend:
    model_name = "mediapipe"

    def __init__(self) -> None:
        self.callback: Any = None
        self.submissions: list[tuple[object, int]] = []
        self.closed = False

    def set_result_callback(self, callback: Any) -> None:
        self.callback = callback

    def detect_async(self, image: object, timestamp_ms: int) -> None:
        self.submissions.append((image, timestamp_ms))

    def complete(self, index: int = -1) -> None:
        _, timestamp_ms = self.submissions[index]
        assert self.callback is not None
        self.callback(_pose(timestamp_ms), timestamp_ms)

    def close(self) -> None:
        self.closed = True


def test_captured_frame_preserves_identity_dimensions_and_timestamp() -> None:
    frame = _frame(7, 123_456_789)

    assert frame.frame_id == 7
    assert frame.capture_timestamp_ns == 123_456_789
    assert frame.source == "camera:0"
    assert (frame.width, frame.height) == (3, 2)


def test_latest_frame_camera_has_one_slot_and_overwrites_old_frames() -> None:
    capture = FiniteCapture(
        [np.zeros((2, 3, 3), dtype=np.uint8) + value for value in (1, 2, 3)]
    )
    timestamps = iter((10, 20, 30))
    camera = LatestFrameCamera(
        capture,
        source="camera:2",
        clock_ns=lambda: next(timestamps),
        read_failure_limit=1,
    ).start()

    deadline = time.monotonic() + 1.0
    while camera.captured_frame_count < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    latest = camera.get_latest(after_frame_id=0, timeout=0.1)
    camera.stop()

    assert camera.buffer_capacity == 1
    assert latest is not None
    assert latest.frame_id == 3
    assert latest.capture_timestamp_ns == 30
    assert latest.source == "camera:2"
    assert int(latest.image[0, 0, 0]) == 3
    assert camera.overwritten_frame_count == 2
    assert capture.released


def test_latest_frame_camera_stop_releases_capture_and_joins_thread() -> None:
    capture = BlockingCapture()
    camera = LatestFrameCamera(capture).start()

    camera.stop(join_timeout=1.0)

    assert capture.released.is_set()
    assert not camera.is_running


def test_latest_frame_camera_enforces_strictly_increasing_capture_timestamps() -> None:
    capture = FiniteCapture(
        [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)]
    )
    timestamps = iter((10, 10, 9))
    camera = LatestFrameCamera(
        capture,
        clock_ns=lambda: next(timestamps),
        read_failure_limit=1,
    ).start()

    deadline = time.monotonic() + 1.0
    while camera.captured_frame_count < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    latest = camera.get_latest(after_frame_id=0, timeout=0.1)
    camera.stop()

    assert latest is not None
    assert latest.frame_id == 3
    assert latest.capture_timestamp_ns == 12


def test_scheduler_has_one_in_flight_and_retains_only_latest_pending_frame() -> None:
    backend = FakeAsyncBackend()
    scheduler = LatestOnlyMediaPipeScheduler(backend)

    assert scheduler.submit(_frame(1, 1_000_000))
    assert not scheduler.submit(_frame(2, 2_000_000))
    assert not scheduler.submit(_frame(3, 3_000_000))
    assert len(backend.submissions) == 1
    assert scheduler.pending_capacity == 1
    assert scheduler.pending_frame_id == 3
    assert scheduler.busy_drop_count == 1

    backend.complete(0)

    assert len(backend.submissions) == 2
    assert int(backend.submissions[1][0][0, 0, 0]) == 3
    assert scheduler.is_busy
    backend.complete(1)
    assert scheduler.latest_result is not None
    assert scheduler.latest_result.frame_id == 3


def test_scheduler_mediapipe_timestamps_are_strictly_increasing_within_one_ms() -> None:
    backend = FakeAsyncBackend()
    scheduler = LatestOnlyMediaPipeScheduler(backend)

    scheduler.submit(_frame(1, 1_100_000))
    backend.complete()
    scheduler.submit(_frame(2, 1_900_000))

    assert [timestamp for _, timestamp in backend.submissions] == [1, 2]


def test_scheduler_restores_exact_frame_metadata_from_callback_timestamp() -> None:
    clock_values = iter((5_000_000, 9_000_000))
    backend = FakeAsyncBackend()
    scheduler = LatestOnlyMediaPipeScheduler(backend, clock_ns=lambda: next(clock_values))
    frame = _frame(42, 3_000_000)

    scheduler.submit(frame)
    backend.complete()
    result = scheduler.latest_result

    assert result is not None
    assert result.frame_id == 42
    assert result.capture_timestamp_ns == 3_000_000
    assert result.inference_start_ns == 5_000_000
    assert result.inference_end_ns == 9_000_000
    assert result.pose is not None
    assert result.pose.timestamp_ms == 3
    assert result.pose.extra["mediapipe_timestamp_ms"] == 3


def test_scheduler_rejects_unknown_invalidated_and_late_callbacks() -> None:
    backend = FakeAsyncBackend()
    scheduler = LatestOnlyMediaPipeScheduler(backend)
    scheduler.submit(_frame(10, 10_000_000))
    callback = backend.callback
    assert callback is not None

    callback(_pose(999), 999)
    assert scheduler.unknown_callback_count == 1
    assert scheduler.latest_result is None

    scheduler.invalidate()
    backend.complete()
    assert scheduler.stale_drop_count == 1
    assert scheduler.latest_result is None

    scheduler.close()
    callback(_pose(10), 10)
    assert scheduler.latest_result is None
    assert backend.closed


def test_old_duplicate_callback_cannot_overwrite_a_newer_result() -> None:
    backend = FakeAsyncBackend()
    scheduler = LatestOnlyMediaPipeScheduler(backend)
    callback = backend.callback
    assert callback is not None

    scheduler.submit(_frame(10, 10_000_000))
    backend.complete()
    scheduler.submit(_frame(11, 11_000_000))
    backend.complete()
    callback(_pose(10), 10)

    assert scheduler.latest_result is not None
    assert scheduler.latest_result.frame_id == 11
    assert scheduler.unknown_callback_count == 1
    assert scheduler.stale_drop_count == 1


@pytest.mark.parametrize(
    ("current_frame_id", "now_ns"),
    ((16, 120_000_000), (12, 251_000_000)),
)
def test_pose_age_gate_rejects_large_frame_gap_or_old_pose_once(
    current_frame_id: int,
    now_ns: int,
) -> None:
    timed = TimedPoseResult(
        frame_id=10,
        capture_timestamp_ns=100_000_000,
        inference_start_ns=101_000_000,
        inference_end_ns=110_000_000,
        result_ready_ns=111_000_000,
        pose=_pose(100),
        backend_name="mediapipe",
    )
    gate = PoseAgeGate(max_pose_age_ms=150, max_frame_gap=5)

    assert not gate.accept_for_analysis(timed, current_frame_id=current_frame_id, now_ns=now_ns)
    assert not gate.accept_for_analysis(timed, current_frame_id=current_frame_id, now_ns=now_ns)
    assert gate.stale_drop_count == 1


def test_pose_age_gate_requires_monotonic_analysis_frame_ids() -> None:
    def timed(frame_id: int) -> TimedPoseResult:
        return TimedPoseResult(
            frame_id=frame_id,
            capture_timestamp_ns=frame_id * 1_000_000,
            inference_start_ns=frame_id * 1_000_000,
            inference_end_ns=frame_id * 1_000_000,
            result_ready_ns=frame_id * 1_000_000,
            pose=_pose(frame_id),
            backend_name="mediapipe",
        )

    gate = PoseAgeGate(max_pose_age_ms=150, max_frame_gap=5)
    assert gate.accept_for_analysis(timed(11), current_frame_id=11, now_ns=11_000_000)
    assert not gate.accept_for_analysis(timed(10), current_frame_id=11, now_ns=11_000_000)
    assert gate.last_analyzed_frame_id == 11


def test_windows_camera_open_falls_back_when_dshow_fails(monkeypatch: Any) -> None:
    class FakeCapture:
        def __init__(self, opened: bool) -> None:
            self.opened = opened
            self.released = False

        def isOpened(self) -> bool:
            return self.opened

        def release(self) -> None:
            self.released = True

        def set(self, _name: object, _value: object) -> bool:
            return True

        def get(self, _name: object) -> float:
            return 30.0

    dshow_capture = FakeCapture(False)
    fallback_capture = FakeCapture(True)
    calls: list[tuple[object, ...]] = []

    def video_capture(*args: object) -> FakeCapture:
        calls.append(args)
        return dshow_capture if len(calls) == 1 else fallback_capture

    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    monkeypatch.setattr(capture_module.cv2, "VideoCapture", video_capture)
    args = SimpleNamespace(
        input_video="",
        camera=2,
        camera_fourcc="",
        width=640,
        height=480,
        camera_fps=30.0,
    )

    capture, mode, fps = open_capture(args)

    assert calls[0] == (2, capture_module.cv2.CAP_DSHOW)
    assert calls[1] == (2,)
    assert dshow_capture.released
    assert capture is fallback_capture
    assert mode == "camera"
    assert fps == 30.0


def test_live_stream_backend_uses_live_mode_and_async_callback(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    created: dict[str, object] = {}

    class FakeOptions:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    class FakeLandmarker:
        def __init__(self) -> None:
            self.submissions: list[int] = []
            self.closed = False

        def detect_async(self, _image: object, timestamp_ms: int) -> None:
            self.submissions.append(timestamp_ms)

        def close(self) -> None:
            self.closed = True

    landmarker = FakeLandmarker()
    fake_vision = SimpleNamespace(
        RunningMode=SimpleNamespace(LIVE_STREAM="live"),
        PoseLandmarkerOptions=FakeOptions,
        PoseLandmarker=SimpleNamespace(create_from_options=lambda _options: landmarker),
    )
    fake_mp = SimpleNamespace(
        Image=lambda **kwargs: kwargs,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    fake_python = SimpleNamespace(BaseOptions=lambda **kwargs: kwargs)
    monkeypatch.setattr(mediapipe_module, "vision", fake_vision)
    monkeypatch.setattr(mediapipe_module, "mp", fake_mp)
    monkeypatch.setattr(mediapipe_module, "mp_python", fake_python)
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")

    backend = MediaPipeLiveStreamBackend(model)
    received: list[tuple[PoseResult, int]] = []
    backend.set_result_callback(lambda result, timestamp: received.append((result, timestamp)))
    backend.detect_async(np.zeros((2, 2, 3), dtype=np.uint8), 7)
    sdk_callback = created["result_callback"]
    assert callable(sdk_callback)
    sdk_callback(SimpleNamespace(pose_landmarks=[], segmentation_masks=[]), None, 7)

    assert created["running_mode"] == "live"
    assert landmarker.submissions == [7]
    assert received[0][1] == 7
    assert received[0][0].success is False
    with pytest.raises(ValueError, match="strictly increasing"):
        backend.detect_async(np.zeros((2, 2, 3), dtype=np.uint8), 7)
    backend.close()
    assert landmarker.closed


def test_offline_mediapipe_backend_remains_in_video_mode(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    created: dict[str, object] = {}

    class FakeOptions:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    class FakeLandmarker:
        def close(self) -> None:
            pass

    fake_vision = SimpleNamespace(
        RunningMode=SimpleNamespace(VIDEO="video"),
        PoseLandmarkerOptions=FakeOptions,
        PoseLandmarker=SimpleNamespace(create_from_options=lambda _options: FakeLandmarker()),
    )
    monkeypatch.setattr(mediapipe_module, "vision", fake_vision)
    monkeypatch.setattr(mediapipe_module, "mp", SimpleNamespace())
    monkeypatch.setattr(
        mediapipe_module,
        "mp_python",
        SimpleNamespace(BaseOptions=lambda **kwargs: kwargs),
    )
    model = tmp_path / "pose.task"
    model.write_bytes(b"model")

    backend = mediapipe_module.MediaPipeBackend(model)

    assert created["running_mode"] == "video"
    assert "result_callback" not in created
    backend.close()
