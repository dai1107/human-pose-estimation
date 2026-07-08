from __future__ import annotations

from dataclasses import replace
from math import isfinite, pi, sqrt
from typing import Literal

from src.backends.base import Keypoint, PoseResult


SmoothingMode = Literal["none", "ema", "one-euro"]

HAND_OCCLUSION_POINT_NAMES: frozenset[str] = frozenset(
    {
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
    }
)
OCCLUSION_GUARD_NAMES: frozenset[str] = frozenset(
    {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    }
)


class LowPassFilter:
    def __init__(self) -> None:
        self.value: float | None = None

    def apply(self, value: float, alpha: float) -> float:
        if self.value is None or not isfinite(self.value):
            self.value = value
        else:
            self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class OneEuroValueFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_raw: float | None = None
        self.last_timestamp_ms: int | None = None

    def apply(self, value: float, timestamp_ms: int | None) -> float:
        if self.last_timestamp_ms is None or timestamp_ms is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1e-3, min(1.0, (timestamp_ms - self.last_timestamp_ms) / 1000.0))
        dx = 0.0 if self.last_raw is None else (value - self.last_raw) / dt
        edx = self.dx_filter.apply(dx, smoothing_alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        smoothed = self.x_filter.apply(value, smoothing_alpha(cutoff, dt))
        self.last_raw = value
        self.last_timestamp_ms = timestamp_ms
        return smoothed

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_raw = None
        self.last_timestamp_ms = None


def smoothing_alpha(cutoff: float, dt: float) -> float:
    cutoff = max(1e-6, cutoff)
    tau = 1.0 / (2.0 * pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class KeypointSmoother:
    def __init__(
        self,
        mode: SmoothingMode = "one-euro",
        ema_alpha: float = 0.6,
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.01,
        one_euro_d_cutoff: float = 1.0,
        min_confidence: float = 0.2,
        max_missing_frames: int = 5,
        occlusion_guard: bool = True,
        occlusion_radius: float = 0.06,
        occlusion_jump_threshold: float = 0.045,
        max_occlusion_hold_frames: int = 8,
    ) -> None:
        if mode not in {"none", "ema", "one-euro"}:
            raise ValueError(f"unknown smoothing mode: {mode}")
        self.mode: SmoothingMode = mode
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self.one_euro_min_cutoff = float(one_euro_min_cutoff)
        self.one_euro_beta = float(one_euro_beta)
        self.one_euro_d_cutoff = float(one_euro_d_cutoff)
        self.min_confidence = float(min_confidence)
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.occlusion_guard = bool(occlusion_guard)
        self.occlusion_radius = max(0.0, float(occlusion_radius))
        self.occlusion_jump_threshold = max(0.0, float(occlusion_jump_threshold))
        self.max_occlusion_hold_frames = max(0, int(max_occlusion_hold_frames))
        self._ema_state: dict[str, Keypoint] = {}
        self._one_euro_state: dict[tuple[str, str], OneEuroValueFilter] = {}
        self._last_points: dict[str, Keypoint] = {}
        self._guard_hold_counts: dict[str, int] = {}
        self._last_result: PoseResult | None = None
        self._missing_frames = 0

    def smooth_result(self, result: PoseResult) -> PoseResult:
        if self.mode == "none":
            return result
        if not result.success or not result.keypoints:
            self._missing_frames += 1
            if self._last_result is not None and self._missing_frames <= self.max_missing_frames:
                return self._held_result(result)
            if self._missing_frames > self.max_missing_frames:
                self.reset()
            return result

        self._missing_frames = 0
        occlusion_points = self._occlusion_points(result.keypoints)
        guarded_names: list[str] = []
        smoothed = [
            self._smooth_keypoint(point, result.timestamp_ms, occlusion_points, guarded_names) for point in result.keypoints
        ]
        extra = dict(result.extra)
        if guarded_names:
            extra["occlusion_guarded_keypoints"] = tuple(sorted(set(guarded_names)))
        smoothed_result = replace(result, keypoints=smoothed, extra=extra)
        self._last_result = smoothed_result
        return smoothed_result

    def reset(self) -> None:
        self._ema_state.clear()
        for value_filter in self._one_euro_state.values():
            value_filter.reset()
        self._one_euro_state.clear()
        self._last_points.clear()
        self._guard_hold_counts.clear()
        self._last_result = None
        self._missing_frames = 0

    def _smooth_keypoint(
        self,
        point: Keypoint,
        timestamp_ms: int | None,
        occlusion_points: list[Keypoint],
        guarded_names: list[str],
    ) -> Keypoint:
        key = self._state_key(point)
        if point.confidence < self.min_confidence or not all(isfinite(value) for value in (point.x, point.y, point.z)):
            return point
        previous = self._last_points.get(key)
        if self._should_hold_for_occlusion(point, previous, occlusion_points):
            self._guard_hold_counts[key] = self._guard_hold_counts.get(key, 0) + 1
            guarded_names.append(point.name)
            return self._decay_confidence(previous, self._guard_hold_counts[key], min_confidence=0.15)

        self._guard_hold_counts[key] = 0
        if self.mode == "ema":
            smoothed = self._smooth_ema(point)
        else:
            smoothed = self._smooth_one_euro(point, timestamp_ms)
        self._last_points[key] = smoothed
        return smoothed

    def _smooth_ema(self, point: Keypoint) -> Keypoint:
        key = self._state_key(point)
        previous = self._ema_state.get(key)
        if previous is None or previous.confidence < self.min_confidence:
            self._ema_state[key] = point
            return point
        keep = 1.0 - self.ema_alpha
        smoothed = replace(
            point,
            x=previous.x * keep + point.x * self.ema_alpha,
            y=previous.y * keep + point.y * self.ema_alpha,
            z=previous.z * keep + point.z * self.ema_alpha,
        )
        self._ema_state[key] = smoothed
        return smoothed

    def _smooth_one_euro(self, point: Keypoint, timestamp_ms: int | None) -> Keypoint:
        values: dict[str, float] = {}
        key = self._state_key(point)
        for axis, value in (("x", point.x), ("y", point.y), ("z", point.z)):
            value_filter = self._one_euro_state.setdefault(
                (key, axis),
                OneEuroValueFilter(
                    min_cutoff=self.one_euro_min_cutoff,
                    beta=self.one_euro_beta,
                    d_cutoff=self.one_euro_d_cutoff,
                ),
            )
            values[axis] = value_filter.apply(value, timestamp_ms)
        return replace(point, **values)

    def _state_key(self, point: Keypoint) -> str:
        return f"{point.source_model or 'unknown'}:{point.name}"

    def _held_result(self, missed_result: PoseResult) -> PoseResult:
        if self._last_result is None:
            return missed_result
        keypoints = [self._decay_confidence(point, self._missing_frames, min_confidence=0.05) for point in self._last_result.keypoints]
        extra = dict(self._last_result.extra)
        extra.update(
            {
                "stabilized_hold": True,
                "hold_frames": self._missing_frames,
                "missed_model_name": missed_result.model_name,
                "missed_inference_time_ms": missed_result.inference_time_ms,
            }
        )
        return replace(
            self._last_result,
            keypoints=keypoints,
            success=True,
            inference_time_ms=missed_result.inference_time_ms,
            timestamp_ms=missed_result.timestamp_ms,
            extra=extra,
        )

    def _occlusion_points(self, keypoints: list[Keypoint]) -> list[Keypoint]:
        if not self.occlusion_guard:
            return []
        return [
            point
            for point in keypoints
            if point.name in HAND_OCCLUSION_POINT_NAMES
            and point.confidence >= self.min_confidence
            and isfinite(point.x)
            and isfinite(point.y)
        ]

    def _should_hold_for_occlusion(
        self,
        point: Keypoint,
        previous: Keypoint | None,
        occlusion_points: list[Keypoint],
    ) -> bool:
        if not self.occlusion_guard or previous is None:
            return False
        if point.name not in OCCLUSION_GUARD_NAMES or not occlusion_points:
            return False
        key = self._state_key(point)
        if self._guard_hold_counts.get(key, 0) >= self.max_occlusion_hold_frames:
            return False
        displacement = _xy_distance(point, previous)
        if displacement <= self.occlusion_jump_threshold:
            return False
        return any(_xy_distance(point, hand_point) <= self.occlusion_radius for hand_point in occlusion_points)

    def _decay_confidence(self, point: Keypoint, frame_count: int, min_confidence: float) -> Keypoint:
        decay = max(min_confidence, 1.0 - 0.16 * max(1, frame_count))
        return replace(point, confidence=max(min_confidence, point.confidence * decay))


def _xy_distance(first: Keypoint, second: Keypoint) -> float:
    if not all(isfinite(value) for value in (first.x, first.y, second.x, second.y)):
        return float("inf")
    dx = first.x - second.x
    dy = first.y - second.y
    return sqrt(dx * dx + dy * dy)
