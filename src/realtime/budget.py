"""Adaptive pose-inference budget control without touching the video clock."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Iterable


@dataclass(frozen=True, slots=True)
class RealtimeBudgetDecision:
    """One controller observation, including an optional degradation action."""

    action: str
    target_pose_fps: float
    extra_analysis_stride: int
    p95_inference_ms: float
    p95_pose_result_age_ms: float
    queue_saturation_ratio: float
    overloaded: bool
    reason: str

    @property
    def changed(self) -> bool:
        return self.action != "none"


class RealtimeBudgetController:
    """Keep pose latency bounded using a fixed, reversible degradation order.

    The display/render loop is intentionally absent from this API.  Under load,
    the controller first lowers pose FPS, then requests an inference-resolution
    step, and only then reduces optional analysis frequency.
    """

    def __init__(
        self,
        *,
        target_pose_fps: float = 15.0,
        max_pose_fps: float = 20.0,
        min_pose_fps: float = 12.0,
        fps_steps: Iterable[float] = (20.0, 15.0, 12.0),
        inference_p95_budget_ms: float = 55.0,
        warning_pose_age_ms: float = 80.0,
        sample_window: int = 30,
        min_samples: int = 12,
        recovery_windows: int = 2,
        max_extra_analysis_stride: int = 4,
    ) -> None:
        maximum = max(1.0, float(max_pose_fps))
        minimum = min(maximum, max(1.0, float(min_pose_fps)))
        initial = min(maximum, max(minimum, float(target_pose_fps)))
        candidates = {
            minimum,
            initial,
            maximum,
            *(
                float(value)
                for value in fps_steps
                if minimum <= float(value) <= maximum
            ),
        }
        self.fps_steps = tuple(sorted(candidates, reverse=True))
        self._fps_index = self.fps_steps.index(initial)
        self.inference_p95_budget_ms = max(1.0, float(inference_p95_budget_ms))
        self.warning_pose_age_ms = max(1.0, float(warning_pose_age_ms))
        capacity = max(2, int(sample_window))
        self.min_samples = max(2, min(int(min_samples), capacity))
        self.recovery_windows = max(1, int(recovery_windows))
        self.max_extra_analysis_stride = max(1, int(max_extra_analysis_stride))
        self._inference_samples: deque[float] = deque(maxlen=capacity)
        self._pose_age_samples: deque[float] = deque(maxlen=capacity)
        self._queue_samples: deque[int] = deque(maxlen=capacity)
        self._under_budget_windows = 0
        self.extra_analysis_stride = 1
        self.resolution_downgrade_count = 0
        self.evaluation_count = 0
        self.last_decision = self._decision(reason="warming_up")

    @property
    def target_pose_fps(self) -> float:
        return self.fps_steps[self._fps_index]

    def observe(
        self,
        *,
        inference_ms: float,
        pose_result_age_ms: float,
        queue_depth: int,
        resolution_can_downgrade: bool,
    ) -> RealtimeBudgetDecision:
        self._inference_samples.append(max(0.0, float(inference_ms)))
        self._pose_age_samples.append(max(0.0, float(pose_result_age_ms)))
        self._queue_samples.append(max(0, int(queue_depth)))
        if len(self._inference_samples) < self.min_samples:
            self.last_decision = self._decision(reason="warming_up")
            return self.last_decision

        p95_inference = _percentile95(self._inference_samples)
        p95_pose_age = _percentile95(self._pose_age_samples)
        queue_ratio = sum(value > 0 for value in self._queue_samples) / len(
            self._queue_samples
        )
        queue_overflow = max(self._queue_samples, default=0) > 1
        overloaded = bool(
            p95_inference > self.inference_p95_budget_ms
            or p95_pose_age > self.warning_pose_age_ms
            or queue_overflow
            or (
                queue_ratio >= 0.75
                and (
                    p95_inference > self.inference_p95_budget_ms * 0.80
                    or p95_pose_age > self.warning_pose_age_ms * 0.80
                )
            )
        )
        underloaded = bool(
            p95_inference < self.inference_p95_budget_ms * 0.55
            and p95_pose_age < self.warning_pose_age_ms * 0.60
            and queue_ratio < 0.25
        )
        action = "none"
        reason = "within_budget"
        if overloaded:
            self._under_budget_windows = 0
            if self._fps_index < len(self.fps_steps) - 1:
                self._fps_index += 1
                action = "reduce_pose_fps"
                reason = "pose_budget_exceeded"
            elif resolution_can_downgrade:
                self.resolution_downgrade_count += 1
                action = "reduce_inference_resolution"
                reason = "minimum_pose_fps_still_over_budget"
            elif self.extra_analysis_stride < self.max_extra_analysis_stride:
                self.extra_analysis_stride = min(
                    self.max_extra_analysis_stride,
                    self.extra_analysis_stride * 2,
                )
                action = "reduce_extra_analysis_frequency"
                reason = "minimum_resolution_still_over_budget"
            else:
                reason = "maximum_safe_degradation_reached"
        elif underloaded:
            self._under_budget_windows += 1
            if self._under_budget_windows >= self.recovery_windows:
                self._under_budget_windows = 0
                if self.extra_analysis_stride > 1:
                    self.extra_analysis_stride = max(
                        1, self.extra_analysis_stride // 2
                    )
                    action = "restore_extra_analysis_frequency"
                    reason = "sustained_budget_headroom"
                elif (
                    self.resolution_downgrade_count == 0
                    and self._fps_index > 0
                ):
                    self._fps_index -= 1
                    action = "increase_pose_fps"
                    reason = "sustained_budget_headroom"
        else:
            self._under_budget_windows = 0

        self.evaluation_count += 1
        self.last_decision = RealtimeBudgetDecision(
            action=action,
            target_pose_fps=self.target_pose_fps,
            extra_analysis_stride=self.extra_analysis_stride,
            p95_inference_ms=p95_inference,
            p95_pose_result_age_ms=p95_pose_age,
            queue_saturation_ratio=queue_ratio,
            overloaded=overloaded,
            reason=reason,
        )
        self._inference_samples.clear()
        self._pose_age_samples.clear()
        self._queue_samples.clear()
        return self.last_decision

    def _decision(self, *, reason: str) -> RealtimeBudgetDecision:
        return RealtimeBudgetDecision(
            action="none",
            target_pose_fps=self.target_pose_fps,
            extra_analysis_stride=self.extra_analysis_stride,
            p95_inference_ms=_percentile95(self._inference_samples),
            p95_pose_result_age_ms=_percentile95(self._pose_age_samples),
            queue_saturation_ratio=(
                sum(value > 0 for value in self._queue_samples)
                / len(self._queue_samples)
                if self._queue_samples
                else 0.0
            ),
            overloaded=False,
            reason=reason,
        )


def _percentile95(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    return ordered[max(0, ceil(0.95 * len(ordered)) - 1)]


__all__ = ["RealtimeBudgetController", "RealtimeBudgetDecision"]
