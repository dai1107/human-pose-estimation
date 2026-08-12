"""Latest-frame sources for latency-bounded camera and video playback."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from src.realtime.types import CapturedFrame


class CaptureDevice(Protocol):
    def read(self) -> tuple[bool, object | None]: ...

    def release(self) -> None: ...


class LatestFrameBuffer:
    """A bounded latest-only buffer that never lets producers build a backlog."""

    def __init__(self, capacity: int = 1) -> None:
        if int(capacity) not in (1, 2):
            raise ValueError("latest-frame capacity must be 1 or 2")
        self.capacity = int(capacity)
        self._condition = threading.Condition()
        self._frames: list[CapturedFrame] = []
        self._closed = False
        self.overwritten_frame_count = 0

    @property
    def queue_depth(self) -> int:
        with self._condition:
            return len(self._frames)

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    def put(self, frame: CapturedFrame) -> None:
        with self._condition:
            if self._closed:
                return
            while len(self._frames) >= self.capacity:
                self._frames.pop(0)
                self.overwritten_frame_count += 1
            self._frames.append(frame)
            self._condition.notify_all()

    def get_latest(
        self,
        *,
        after_frame_id: int = -1,
        timeout: float | None = None,
    ) -> CapturedFrame | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                candidates = [frame for frame in self._frames if frame.frame_id > after_frame_id]
                if candidates:
                    latest = candidates[-1]
                    self._frames.clear()
                    return latest
                if self._closed:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class LatestFrameCamera:
    """Continuously read a camera on one thread and retain only its newest frame."""

    buffer_capacity = 1

    def __init__(
        self,
        capture: CaptureDevice,
        *,
        source: str = "camera",
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        read_failure_sleep_s: float = 0.01,
        read_failure_limit: int = 30,
    ) -> None:
        self._capture = capture
        self._source = source
        self._clock_ns = clock_ns
        self._read_failure_sleep_s = max(0.001, float(read_failure_sleep_s))
        self._read_failure_limit = max(1, int(read_failure_limit))
        self._condition = threading.Condition()
        self._buffer = LatestFrameBuffer(capacity=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: CapturedFrame | None = None
        self._frame_id = 0
        self._last_capture_timestamp_ns = -1
        self._last_delivered_frame_id = 0
        self._released = False
        self._terminal_read_failure = False
        self.captured_frame_count = 0
        self.overwritten_frame_count = 0
        self.camera_read_failures = 0

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def terminal_read_failure(self) -> bool:
        with self._condition:
            return self._terminal_read_failure

    def start(self) -> "LatestFrameCamera":
        with self._condition:
            if self._thread is not None:
                return self
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="latest-frame-camera",
                daemon=True,
            )
            self._thread.start()
        return self

    def get_latest(
        self,
        *,
        after_frame_id: int = -1,
        timeout: float | None = None,
    ) -> CapturedFrame | None:
        """Return the newest unseen frame, waiting only for camera input when asked."""

        latest = self._buffer.get_latest(after_frame_id=after_frame_id, timeout=timeout)
        if latest is not None:
            self._last_delivered_frame_id = max(self._last_delivered_frame_id, latest.frame_id)
        return latest

    def stop(self, *, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._buffer.close()
        self._release_capture()
        with self._condition:
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(join_timeout)))

    close = stop

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                # Keep the injected clock's historical one-call-per-frame
                # contract; in production both clocks are perf_counter_ns.
                capture_read_start_ns = time.perf_counter_ns()
                ok, image = self._capture.read()
                if not ok or image is None:
                    consecutive_failures += 1
                    with self._condition:
                        self.camera_read_failures += 1
                        if consecutive_failures >= self._read_failure_limit:
                            self._terminal_read_failure = True
                            self._condition.notify_all()
                            return
                    self._stop_event.wait(self._read_failure_sleep_s)
                    continue

                # The capture timestamp is intentionally the first operation after
                # a successful read so mirroring/encoding cannot change its meaning.
                capture_read_end_ns = int(self._clock_ns())
                capture_timestamp_ns = max(
                    capture_read_end_ns,
                    self._last_capture_timestamp_ns + 1,
                )
                self._last_capture_timestamp_ns = capture_timestamp_ns
                consecutive_failures = 0
                height, width = image.shape[:2]
                with self._condition:
                    self._frame_id += 1
                    frame = CapturedFrame(
                        frame_id=self._frame_id,
                        capture_timestamp_ns=capture_timestamp_ns,
                        image=image,
                        source=self._source,
                        width=int(width),
                        height=int(height),
                        capture_read_start_ns=capture_read_start_ns,
                        capture_read_end_ns=capture_read_end_ns,
                        source_timestamp_ms=int(capture_timestamp_ns // 1_000_000),
                    )
                    self._latest = frame
                    before = self._buffer.overwritten_frame_count
                    self._buffer.put(frame)
                    self.overwritten_frame_count += self._buffer.overwritten_frame_count - before
                    self.captured_frame_count += 1
                    self._condition.notify_all()
        finally:
            self._release_capture()
            self._buffer.close()
            with self._condition:
                self._condition.notify_all()

    def _release_capture(self) -> None:
        with self._condition:
            if self._released:
                return
            self._released = True
        self._capture.release()


class LatestFrameVideo:
    """Decode a file on its source clock and retain only the newest display frame."""

    buffer_capacity = 1

    def __init__(
        self,
        capture: CaptureDevice,
        *,
        source_fps: float,
        source: str = "video",
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._capture = capture
        self.source_fps = max(1.0, float(source_fps))
        self._source = source
        self._clock_ns = clock_ns
        self._buffer = LatestFrameBuffer(capacity=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._released = False
        self._release_lock = threading.Lock()
        self.exhausted = False
        self.captured_frame_count = 0
        self.camera_read_failures = 0

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def terminal_read_failure(self) -> bool:
        return False

    @property
    def overwritten_frame_count(self) -> int:
        return self._buffer.overwritten_frame_count

    @property
    def queue_depth(self) -> int:
        return self._buffer.queue_depth

    def start(self) -> "LatestFrameVideo":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._decode_loop,
                name="latest-frame-video",
                daemon=True,
            )
            self._thread.start()
        return self

    def get_latest(
        self,
        *,
        after_frame_id: int = -1,
        timeout: float | None = None,
    ) -> CapturedFrame | None:
        return self._buffer.get_latest(after_frame_id=after_frame_id, timeout=timeout)

    def stop(self, *, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._buffer.close()
        self._release_capture()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(join_timeout)))

    close = stop

    def _decode_loop(self) -> None:
        started_ns = int(self._clock_ns())
        frame_id = 0
        try:
            while not self._stop_event.is_set():
                target_ns = started_ns + int(frame_id * 1_000_000_000.0 / self.source_fps)
                remaining_s = (target_ns - int(self._clock_ns())) / 1_000_000_000.0
                if remaining_s > 0.0 and self._stop_event.wait(remaining_s):
                    return
                read_start_ns = int(self._clock_ns())
                ok, image = self._capture.read()
                read_end_ns = int(self._clock_ns())
                if not ok or image is None:
                    self.exhausted = True
                    return
                frame_id += 1
                height, width = image.shape[:2]
                self._buffer.put(
                    CapturedFrame(
                        frame_id=frame_id,
                        capture_timestamp_ns=read_end_ns,
                        image=image,
                        source=self._source,
                        width=int(width),
                        height=int(height),
                        capture_read_start_ns=read_start_ns,
                        capture_read_end_ns=read_end_ns,
                        source_timestamp_ms=int(round((frame_id - 1) * 1000.0 / self.source_fps)),
                    )
                )
                self.captured_frame_count += 1
        finally:
            self.exhausted = True
            self._release_capture()
            self._buffer.close()

    def _release_capture(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._capture.release()
