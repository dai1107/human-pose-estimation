"""Latest-only MediaPipe scheduling and stale-pose admission controls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from src.backends.base import PoseResult
from src.realtime.types import CapturedFrame, TimedPoseResult


class AsyncPoseBackend(Protocol):
    model_name: str

    def set_result_callback(
        self,
        callback: Callable[[PoseResult, int], None] | None,
    ) -> None: ...

    def detect_async(self, frame: object, timestamp_ms: int) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _Submission:
    frame: CapturedFrame
    inference_start_ns: int
    epoch: int
    dropped_before_inference: int


class LatestOnlyMediaPipeScheduler:
    """Allow one request in flight and collapse all waiting input to one frame."""

    pending_capacity = 1

    def __init__(
        self,
        backend: AsyncPoseBackend,
        *,
        result_callback: Callable[[TimedPoseResult], None] | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._backend = backend
        self._result_callback = result_callback
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._closed = False
        self._epoch = 0
        self._pending: CapturedFrame | None = None
        self._in_flight_timestamp_ms: int | None = None
        self._submissions: dict[int, _Submission] = {}
        self._last_timestamp_ms = -1
        self._last_accepted_frame_id = -1
        self._latest_result: TimedPoseResult | None = None
        self.submitted_count = 0
        self.result_count = 0
        self.busy_drop_count = 0
        self.stale_drop_count = 0
        self.unknown_callback_count = 0
        self.late_callback_count = 0
        self._backend.set_result_callback(self._handle_backend_result)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._in_flight_timestamp_ms is not None

    @property
    def latest_result(self) -> TimedPoseResult | None:
        with self._lock:
            return self._latest_result

    @property
    def pending_frame_id(self) -> int | None:
        with self._lock:
            return self._pending.frame_id if self._pending is not None else None

    def submit(self, frame: CapturedFrame) -> bool:
        """Submit now if idle; otherwise retain only this newest pending frame."""

        dispatch: tuple[CapturedFrame, int] | None = None
        with self._lock:
            if self._closed:
                return False
            if self._in_flight_timestamp_ms is not None:
                if self._pending is not None:
                    self.busy_drop_count += 1
                self._pending = frame
                return False
            dispatch = self._reserve_locked(frame)
        self._dispatch(*dispatch)
        return True

    def invalidate(self) -> None:
        """Reject results from the prior action/mirror context without racing callbacks."""

        with self._lock:
            self._epoch += 1
            self._pending = None
            self._latest_result = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._latest_result = None
            self._submissions.clear()
        self._backend.set_result_callback(None)
        self._backend.close()

    def _reserve_locked(self, frame: CapturedFrame) -> tuple[CapturedFrame, int]:
        timestamp_ms = max(
            int(frame.capture_timestamp_ns // 1_000_000),
            self._last_timestamp_ms + 1,
        )
        self._last_timestamp_ms = timestamp_ms
        self._in_flight_timestamp_ms = timestamp_ms
        self._submissions[timestamp_ms] = _Submission(
            frame=frame,
            inference_start_ns=int(self._clock_ns()),
            epoch=self._epoch,
            dropped_before_inference=self.busy_drop_count,
        )
        self.submitted_count += 1
        return frame, timestamp_ms

    def _dispatch(self, frame: CapturedFrame, timestamp_ms: int) -> None:
        try:
            with self._lock:
                if (
                    self._closed
                    or self._in_flight_timestamp_ms != timestamp_ms
                    or timestamp_ms not in self._submissions
                ):
                    return
                self._backend.detect_async(frame.image, timestamp_ms)
        except Exception:
            next_dispatch: tuple[CapturedFrame, int] | None = None
            with self._lock:
                self._submissions.pop(timestamp_ms, None)
                if self._in_flight_timestamp_ms == timestamp_ms:
                    self._in_flight_timestamp_ms = None
                if not self._closed and self._pending is not None:
                    pending = self._pending
                    self._pending = None
                    next_dispatch = self._reserve_locked(pending)
            if next_dispatch is not None:
                self._dispatch(*next_dispatch)
            raise

    def _handle_backend_result(self, pose: PoseResult, timestamp_ms: int) -> None:
        result_ready_ns = int(self._clock_ns())
        next_dispatch: tuple[CapturedFrame, int] | None = None
        accepted: TimedPoseResult | None = None
        with self._lock:
            if self._closed:
                self.late_callback_count += 1
                return
            submission = self._submissions.pop(int(timestamp_ms), None)
            if submission is None:
                self.unknown_callback_count += 1
                if int(timestamp_ms) <= self._last_timestamp_ms:
                    self.stale_drop_count += 1
                return
            if self._in_flight_timestamp_ms == int(timestamp_ms):
                self._in_flight_timestamp_ms = None

            inference_end_ns = result_ready_ns
            inference_ms = max(0, inference_end_ns - submission.inference_start_ns) / 1_000_000.0
            pose_extra = dict(pose.extra)
            pose_extra["mediapipe_timestamp_ms"] = int(timestamp_ms)
            pose = replace(
                pose,
                inference_time_ms=inference_ms,
                timestamp_ms=int(submission.frame.capture_timestamp_ns // 1_000_000),
                extra=pose_extra,
            )
            candidate = TimedPoseResult(
                frame_id=submission.frame.frame_id,
                capture_timestamp_ns=submission.frame.capture_timestamp_ns,
                inference_start_ns=submission.inference_start_ns,
                inference_end_ns=inference_end_ns,
                result_ready_ns=result_ready_ns,
                pose=pose,
                backend_name=self._backend.model_name,
                dropped_before_inference=submission.dropped_before_inference,
            )
            if submission.epoch != self._epoch or candidate.frame_id <= self._last_accepted_frame_id:
                self.stale_drop_count += 1
            else:
                self._last_accepted_frame_id = candidate.frame_id
                self._latest_result = candidate
                self.result_count += 1
                accepted = candidate

            if self._pending is not None:
                pending = self._pending
                self._pending = None
                next_dispatch = self._reserve_locked(pending)

        try:
            if accepted is not None and self._result_callback is not None:
                self._result_callback(accepted)
        finally:
            if next_dispatch is not None:
                self._dispatch(*next_dispatch)


class PoseAgeGate:
    """Admit each observation once, only while it is relevant to the display frame."""

    def __init__(self, *, max_pose_age_ms: float = 150.0, max_frame_gap: int = 5) -> None:
        self.max_pose_age_ms = max(0.0, float(max_pose_age_ms))
        self.max_frame_gap = max(0, int(max_frame_gap))
        self.last_analyzed_frame_id = -1
        self.stale_drop_count = 0
        self._counted_stale_frame_ids: set[int] = set()

    def is_fresh(
        self,
        result: TimedPoseResult | None,
        *,
        current_frame_id: int,
        now_ns: int | None = None,
    ) -> bool:
        if result is None or result.pose is None:
            return False
        now_ns = time.perf_counter_ns() if now_ns is None else int(now_ns)
        frame_gap = int(current_frame_id) - result.frame_id
        return (
            frame_gap >= 0
            and frame_gap <= self.max_frame_gap
            and result.age_ms(now_ns) <= self.max_pose_age_ms
        )

    def accept_for_analysis(
        self,
        result: TimedPoseResult | None,
        *,
        current_frame_id: int,
        now_ns: int | None = None,
    ) -> bool:
        if result is None or result.frame_id <= self.last_analyzed_frame_id:
            return False
        if not self.is_fresh(result, current_frame_id=current_frame_id, now_ns=now_ns):
            if result.frame_id not in self._counted_stale_frame_ids:
                self._counted_stale_frame_ids.add(result.frame_id)
                self.stale_drop_count += 1
            return False
        self.last_analyzed_frame_id = result.frame_id
        return True

    def reset(self) -> None:
        self.last_analyzed_frame_id = -1
        self._counted_stale_frame_ids.clear()
