"""Single-slot threaded camera capture for latency-bounded realtime input."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from src.realtime.types import CapturedFrame


class CaptureDevice(Protocol):
    def read(self) -> tuple[bool, object | None]: ...

    def release(self) -> None: ...


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

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                latest = self._latest
                if latest is not None and latest.frame_id > after_frame_id:
                    self._last_delivered_frame_id = max(
                        self._last_delivered_frame_id,
                        latest.frame_id,
                    )
                    return latest
                if self._stop_event.is_set() or self._terminal_read_failure:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def stop(self, *, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
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
                    )
                    if (
                        self._latest is not None
                        and self._latest.frame_id > self._last_delivered_frame_id
                    ):
                        self.overwritten_frame_count += 1
                    self._latest = frame
                    self.captured_frame_count += 1
                    self._condition.notify_all()
        finally:
            self._release_capture()
            with self._condition:
                self._condition.notify_all()

    def _release_capture(self) -> None:
        with self._condition:
            if self._released:
                return
            self._released = True
        self._capture.release()
