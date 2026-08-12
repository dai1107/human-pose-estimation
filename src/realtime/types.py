"""Frame identity and timing records for the realtime camera pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backends.base import PoseResult


@dataclass(slots=True)
class CapturedFrame:
    """One camera observation with immutable capture identity metadata."""

    frame_id: int
    capture_timestamp_ns: int
    image: np.ndarray
    source: str
    width: int
    height: int
    capture_read_start_ns: int = 0
    capture_read_end_ns: int = 0
    source_timestamp_ms: int | None = None
    inference_width: int | None = None
    inference_height: int | None = None
    resize_ms: float = 0.0


@dataclass(slots=True)
class TimedPoseResult:
    """A pose result tied back to the exact frame submitted to MediaPipe."""

    frame_id: int
    capture_timestamp_ns: int
    inference_start_ns: int
    inference_end_ns: int
    result_ready_ns: int
    pose: PoseResult | None
    backend_name: str
    dropped_before_inference: int = 0

    @property
    def queue_wait_ms(self) -> float:
        return max(0, self.inference_start_ns - self.capture_timestamp_ns) / 1_000_000.0

    @property
    def inference_ms(self) -> float:
        return max(0, self.inference_end_ns - self.inference_start_ns) / 1_000_000.0

    @property
    def total_latency_ms(self) -> float:
        return max(0, self.result_ready_ns - self.capture_timestamp_ns) / 1_000_000.0

    def age_ms(self, now_ns: int) -> float:
        return max(0, int(now_ns) - self.capture_timestamp_ns) / 1_000_000.0

    @property
    def pose_timestamp_ms(self) -> int:
        if self.pose is not None and self.pose.timestamp_ms is not None:
            return int(self.pose.timestamp_ms)
        return int(self.capture_timestamp_ns // 1_000_000)

    def source_age_ms(self, current_frame_timestamp_ms: int | float) -> float:
        """Age on the camera/video timeline, independent of wall-clock delay."""

        return float(current_frame_timestamp_ms) - float(self.pose_timestamp_ms)
