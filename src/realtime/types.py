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
