"""Inference-only resizing that leaves capture and display resolution untouched."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PreparedInferenceFrame:
    image: np.ndarray
    width: int
    height: int
    resize_ms: float


class InferenceResolutionController:
    """Bound pose input size and step it down when sustained P95 exceeds budget."""

    DEFAULT_STEPS = (640, 512, 416, 320)

    def __init__(
        self,
        inference_width: int = 640,
        *,
        adaptive: bool = True,
        width_steps: Iterable[int] = DEFAULT_STEPS,
        max_inference_p95_ms: float = 55.0,
        sample_window: int = 30,
        min_samples: int = 12,
    ) -> None:
        initial = max(1, int(inference_width))
        candidates = {initial, *(int(value) for value in width_steps if 0 < int(value) <= initial)}
        self.width_steps = tuple(sorted(candidates, reverse=True))
        self._step_index = 0
        self.adaptive = bool(adaptive)
        self.max_inference_p95_ms = max(1.0, float(max_inference_p95_ms))
        self.min_samples = max(2, min(int(min_samples), int(sample_window)))
        self._samples: deque[float] = deque(maxlen=max(self.min_samples, int(sample_window)))
        self.downgrade_count = 0
        self.last_p95_ms = 0.0

    @property
    def current_width(self) -> int:
        return self.width_steps[self._step_index]

    @property
    def can_step_down(self) -> bool:
        return self.adaptive and self._step_index < len(self.width_steps) - 1

    def force_step_down(self) -> bool:
        """Apply one controller-requested inference-only resolution downgrade."""

        if not self.can_step_down:
            return False
        self._step_index += 1
        self.downgrade_count += 1
        self._samples.clear()
        return True

    def prepare(self, frame: np.ndarray) -> PreparedInferenceFrame:
        height, width = frame.shape[:2]
        target_width = min(width, self.current_width)
        if target_width >= width:
            return PreparedInferenceFrame(frame, width, height, 0.0)
        target_height = max(1, int(round(height * target_width / width)))
        started = cv2.getTickCount()
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resize_ms = (cv2.getTickCount() - started) * 1000.0 / cv2.getTickFrequency()
        return PreparedInferenceFrame(resized, target_width, target_height, resize_ms)

    def observe(self, inference_ms: float) -> bool:
        """Record one completed inference; return True only when width steps down."""

        value = float(inference_ms)
        if value < 0.0:
            return False
        self._samples.append(value)
        if len(self._samples) < self.min_samples:
            return False
        ordered = sorted(self._samples)
        self.last_p95_ms = ordered[max(0, ceil(0.95 * len(ordered)) - 1)]
        if (
            not self.adaptive
            or self.last_p95_ms <= self.max_inference_p95_ms
            or self._step_index >= len(self.width_steps) - 1
        ):
            return False
        return self.force_step_down()
