from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, pi, sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping

from src.backends.base import Keypoint, PoseResult

if TYPE_CHECKING:
    from src.product_pose import RealtimeSmoothingConfig


SmoothingMode = Literal["none", "ema", "one-euro"]
SmoothingProfile = Literal["stable", "balanced", "responsive"]
LandmarkSpace = Literal["image", "world"]


@dataclass(frozen=True, slots=True)
class OneEuroParameters:
    min_cutoff: float
    beta: float
    d_cutoff: float


ONE_EURO_PROFILES: Mapping[str, OneEuroParameters] = MappingProxyType(
    {
        "stable": OneEuroParameters(min_cutoff=0.8, beta=0.025, d_cutoff=1.0),
        "balanced": OneEuroParameters(min_cutoff=1.2, beta=0.05, d_cutoff=1.0),
        "responsive": OneEuroParameters(min_cutoff=1.7, beta=0.08, d_cutoff=1.0),
    }
)

FAST_JOINT_NAMES: frozenset[str] = frozenset(
    {
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    }
)
STABLE_JOINT_NAMES: frozenset[str] = frozenset(
    {
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    }
)

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
    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
        *,
        max_gap_ms_before_reset: float = 250.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.max_gap_ns_before_reset = max(0, int(float(max_gap_ms_before_reset) * 1_000_000.0))
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_raw: float | None = None
        self.last_timestamp_ns: int | None = None
        self.last_dt_seconds: float | None = None
        self.gap_reset_count = 0

    def apply(
        self,
        value: float,
        timestamp_ms: int | float | None = None,
        *,
        timestamp_ns: int | None = None,
    ) -> float:
        """Filter one sample using observation time, never callback/display cadence.

        A missing, non-monotonic, or long-gap timestamp starts a fresh track.
        Passing the raw value through is safer than inventing a fixed frame rate.
        """

        if not isfinite(value):
            return value
        resolved_timestamp_ns = _timestamp_ns(timestamp_ms, timestamp_ns)
        if resolved_timestamp_ns is None:
            self.reset()
            return self._seed(value, None)

        if self.last_timestamp_ns is None:
            return self._seed(value, resolved_timestamp_ns)

        gap_ns = resolved_timestamp_ns - self.last_timestamp_ns
        if gap_ns <= 0 or (
            self.max_gap_ns_before_reset > 0
            and gap_ns > self.max_gap_ns_before_reset
        ):
            self.gap_reset_count += 1
            self.reset(preserve_gap_reset_count=True)
            return self._seed(value, resolved_timestamp_ns)

        dt = gap_ns / 1_000_000_000.0
        dx = 0.0 if self.last_raw is None else (value - self.last_raw) / dt
        edx = self.dx_filter.apply(dx, smoothing_alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        smoothed = self.x_filter.apply(value, smoothing_alpha(cutoff, dt))
        self.last_raw = value
        self.last_timestamp_ns = resolved_timestamp_ns
        self.last_dt_seconds = dt
        return smoothed

    def reset(self, *, preserve_gap_reset_count: bool = False) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_raw = None
        self.last_timestamp_ns = None
        self.last_dt_seconds = None
        if not preserve_gap_reset_count:
            self.gap_reset_count = 0

    def _seed(self, value: float, timestamp_ns: int | None) -> float:
        self.last_raw = value
        self.last_timestamp_ns = timestamp_ns
        self.last_dt_seconds = None
        return self.x_filter.apply(value, 1.0)


def _timestamp_ns(timestamp_ms: int | float | None, timestamp_ns: int | None) -> int | None:
    if timestamp_ns is not None:
        return int(timestamp_ns)
    if timestamp_ms is None or isinstance(timestamp_ms, bool):
        return None
    try:
        numeric = float(timestamp_ms)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(numeric):
        return None
    return int(round(numeric * 1_000_000.0))


def smoothing_alpha(cutoff: float, dt: float) -> float:
    cutoff = max(1e-6, cutoff)
    tau = 1.0 / (2.0 * pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class KeypointSmoother:
    def __init__(
        self,
        mode: SmoothingMode = "one-euro",
        ema_alpha: float = 0.6,
        profile: SmoothingProfile = "balanced",
        one_euro_min_cutoff: float | None = None,
        one_euro_beta: float | None = None,
        one_euro_d_cutoff: float | None = None,
        one_euro_profiles: Mapping[str, OneEuroParameters] | None = None,
        max_gap_ms_before_reset: float = 250.0,
        fast_joint_min_cutoff_scale: float = 1.25,
        fast_joint_beta_scale: float = 1.50,
        stable_joint_min_cutoff_scale: float = 0.75,
        stable_joint_beta_scale: float = 0.50,
        world_min_cutoff_scale: float = 0.90,
        world_beta_scale: float = 0.85,
        min_confidence: float = 0.2,
        max_missing_frames: int = 5,
        occlusion_guard: bool = True,
        occlusion_radius: float = 0.06,
        occlusion_jump_threshold: float = 0.045,
        max_occlusion_hold_frames: int = 8,
    ) -> None:
        if mode not in {"none", "ema", "one-euro"}:
            raise ValueError(f"unknown smoothing mode: {mode}")
        profiles = dict(ONE_EURO_PROFILES if one_euro_profiles is None else one_euro_profiles)
        if profile not in profiles:
            raise ValueError(f"unknown One Euro profile: {profile}")
        selected = profiles[profile]
        self.mode: SmoothingMode = mode
        self.profile: SmoothingProfile = profile
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self.one_euro_min_cutoff = float(selected.min_cutoff if one_euro_min_cutoff is None else one_euro_min_cutoff)
        self.one_euro_beta = float(selected.beta if one_euro_beta is None else one_euro_beta)
        self.one_euro_d_cutoff = float(selected.d_cutoff if one_euro_d_cutoff is None else one_euro_d_cutoff)
        if self.one_euro_min_cutoff <= 0 or self.one_euro_beta < 0 or self.one_euro_d_cutoff <= 0:
            raise ValueError("One Euro cutoff values must be positive and beta must be non-negative")
        self.max_gap_ms_before_reset = max(0.0, float(max_gap_ms_before_reset))
        self.fast_joint_min_cutoff_scale = _positive_scale(fast_joint_min_cutoff_scale, "fast_joint_min_cutoff_scale")
        self.fast_joint_beta_scale = _positive_scale(fast_joint_beta_scale, "fast_joint_beta_scale")
        self.stable_joint_min_cutoff_scale = _positive_scale(stable_joint_min_cutoff_scale, "stable_joint_min_cutoff_scale")
        self.stable_joint_beta_scale = _positive_scale(stable_joint_beta_scale, "stable_joint_beta_scale")
        self.world_min_cutoff_scale = _positive_scale(world_min_cutoff_scale, "world_min_cutoff_scale")
        self.world_beta_scale = _positive_scale(world_beta_scale, "world_beta_scale")
        self.min_confidence = float(min_confidence)
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.occlusion_guard = bool(occlusion_guard)
        self.occlusion_radius = max(0.0, float(occlusion_radius))
        self.occlusion_jump_threshold = max(0.0, float(occlusion_jump_threshold))
        self.max_occlusion_hold_frames = max(0, int(max_occlusion_hold_frames))
        self._ema_state: dict[tuple[LandmarkSpace, str], Keypoint] = {}
        self._one_euro_state: dict[tuple[LandmarkSpace, str, str], OneEuroValueFilter] = {}
        self._last_points: dict[str, Keypoint] = {}
        self._guard_hold_counts: dict[str, int] = {}
        self._last_observation_timestamp_ns: dict[LandmarkSpace, int] = {}
        self._last_result: PoseResult | None = None
        self._missing_frames = 0

    @classmethod
    def from_config(
        cls,
        config: RealtimeSmoothingConfig,
        *,
        mode: SmoothingMode | None = None,
        profile: SmoothingProfile | None = None,
        one_euro_min_cutoff: float | None = None,
        one_euro_beta: float | None = None,
        one_euro_d_cutoff: float | None = None,
        **kwargs: object,
    ) -> KeypointSmoother:
        configured_mode: SmoothingMode = "one-euro" if config.mode == "adaptive_one_euro" else "none"
        profiles = {
            name: OneEuroParameters(
                min_cutoff=values.min_cutoff,
                beta=values.beta,
                d_cutoff=values.d_cutoff,
            )
            for name, values in config.profiles.items()
        }
        return cls(
            mode=configured_mode if mode is None else mode,
            profile=config.profile if profile is None else profile,
            one_euro_min_cutoff=one_euro_min_cutoff,
            one_euro_beta=one_euro_beta,
            one_euro_d_cutoff=one_euro_d_cutoff,
            one_euro_profiles=profiles,
            max_gap_ms_before_reset=config.max_gap_ms_before_reset,
            fast_joint_min_cutoff_scale=config.fast_joint_min_cutoff_scale,
            fast_joint_beta_scale=config.fast_joint_beta_scale,
            stable_joint_min_cutoff_scale=config.stable_joint_min_cutoff_scale,
            stable_joint_beta_scale=config.stable_joint_beta_scale,
            world_min_cutoff_scale=config.world_min_cutoff_scale,
            world_beta_scale=config.world_beta_scale,
            **kwargs,
        )

    def smooth_result(
        self,
        result: PoseResult,
        *,
        capture_timestamp_ns: int | None = None,
    ) -> PoseResult:
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
        timestamp_ns = _timestamp_ns(result.timestamp_ms, capture_timestamp_ns)
        gap_reset_spaces: list[str] = []
        if self._reset_space_for_gap("image", timestamp_ns):
            gap_reset_spaces.append("image")
        occlusion_points = self._occlusion_points(result.keypoints)
        guarded_names: list[str] = []
        smoothed = [
            self._smooth_keypoint(
                point,
                timestamp_ns,
                occlusion_points,
                guarded_names,
                space="image",
            )
            for point in result.keypoints
        ]
        extra = dict(result.extra)
        world_keypoints = extra.get("world_keypoints")
        if isinstance(world_keypoints, (list, tuple)) and world_keypoints:
            if self._reset_space_for_gap("world", timestamp_ns):
                gap_reset_spaces.append("world")
            extra["world_keypoints"] = [
                self._smooth_keypoint(
                    point,
                    timestamp_ns,
                    [],
                    [],
                    space="world",
                )
                for point in world_keypoints
                if isinstance(point, Keypoint)
            ]
            extra["world_landmarks_smoothed"] = True
        extra["smoothing_profile"] = self.profile
        extra["smoothing_capture_timestamp_ns"] = timestamp_ns
        if gap_reset_spaces:
            extra["smoothing_gap_reset_spaces"] = tuple(gap_reset_spaces)
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
        self._last_observation_timestamp_ns.clear()
        self._last_result = None
        self._missing_frames = 0

    def _smooth_keypoint(
        self,
        point: Keypoint,
        timestamp_ns: int | None,
        occlusion_points: list[Keypoint],
        guarded_names: list[str],
        *,
        space: LandmarkSpace,
    ) -> Keypoint:
        key = self._state_key(point)
        if point.confidence < self.min_confidence or not all(isfinite(value) for value in (point.x, point.y, point.z)):
            return point
        previous = self._last_points.get(key) if space == "image" else None
        if space == "image" and self._should_hold_for_occlusion(point, previous, occlusion_points):
            self._guard_hold_counts[key] = self._guard_hold_counts.get(key, 0) + 1
            guarded_names.append(point.name)
            return self._decay_confidence(previous, self._guard_hold_counts[key], min_confidence=0.15)

        if space == "image":
            self._guard_hold_counts[key] = 0
        if self.mode == "ema":
            smoothed = self._smooth_ema(point, space=space)
        else:
            smoothed = self._smooth_one_euro(point, timestamp_ns, space=space)
        if space == "image":
            self._last_points[key] = smoothed
        return smoothed

    def _smooth_ema(self, point: Keypoint, *, space: LandmarkSpace) -> Keypoint:
        key = self._state_key(point)
        state_key = (space, key)
        previous = self._ema_state.get(state_key)
        if previous is None or previous.confidence < self.min_confidence:
            self._ema_state[state_key] = point
            return point
        keep = 1.0 - self.ema_alpha
        smoothed = replace(
            point,
            x=previous.x * keep + point.x * self.ema_alpha,
            y=previous.y * keep + point.y * self.ema_alpha,
            z=previous.z * keep + point.z * self.ema_alpha,
        )
        self._ema_state[state_key] = smoothed
        return smoothed

    def _smooth_one_euro(
        self,
        point: Keypoint,
        timestamp_ns: int | None,
        *,
        space: LandmarkSpace,
    ) -> Keypoint:
        values: dict[str, float] = {}
        key = self._state_key(point)
        parameters = self._parameters_for(point.name, space=space)
        for axis, value in (("x", point.x), ("y", point.y), ("z", point.z)):
            value_filter = self._one_euro_state.setdefault(
                (space, key, axis),
                OneEuroValueFilter(
                    min_cutoff=parameters.min_cutoff,
                    beta=parameters.beta,
                    d_cutoff=parameters.d_cutoff,
                    max_gap_ms_before_reset=self.max_gap_ms_before_reset,
                ),
            )
            values[axis] = value_filter.apply(value, timestamp_ns=timestamp_ns)
        return replace(point, **values)

    def _parameters_for(self, point_name: str, *, space: LandmarkSpace) -> OneEuroParameters:
        min_cutoff_scale = 1.0
        beta_scale = 1.0
        if point_name in FAST_JOINT_NAMES:
            min_cutoff_scale *= self.fast_joint_min_cutoff_scale
            beta_scale *= self.fast_joint_beta_scale
        elif point_name in STABLE_JOINT_NAMES:
            min_cutoff_scale *= self.stable_joint_min_cutoff_scale
            beta_scale *= self.stable_joint_beta_scale
        if space == "world":
            min_cutoff_scale *= self.world_min_cutoff_scale
            beta_scale *= self.world_beta_scale
        return OneEuroParameters(
            min_cutoff=self.one_euro_min_cutoff * min_cutoff_scale,
            beta=self.one_euro_beta * beta_scale,
            d_cutoff=self.one_euro_d_cutoff,
        )

    def _reset_space_for_gap(self, space: LandmarkSpace, timestamp_ns: int | None) -> bool:
        previous_ns = self._last_observation_timestamp_ns.get(space)
        self._last_observation_timestamp_ns.pop(space, None)
        if timestamp_ns is None:
            self._clear_space(space)
            return previous_ns is not None
        gap_threshold_ns = int(self.max_gap_ms_before_reset * 1_000_000.0)
        gap_reset = previous_ns is not None and (
            timestamp_ns <= previous_ns
            or (
                gap_threshold_ns > 0
                and timestamp_ns - previous_ns > gap_threshold_ns
            )
        )
        if gap_reset:
            self._clear_space(space)
        self._last_observation_timestamp_ns[space] = timestamp_ns
        return gap_reset

    def _clear_space(self, space: LandmarkSpace) -> None:
        self._ema_state = {
            key: value for key, value in self._ema_state.items() if key[0] != space
        }
        retained_filters: dict[tuple[LandmarkSpace, str, str], OneEuroValueFilter] = {}
        for key, value_filter in self._one_euro_state.items():
            if key[0] == space:
                value_filter.reset()
            else:
                retained_filters[key] = value_filter
        self._one_euro_state = retained_filters
        if space == "image":
            self._last_points.clear()
            self._guard_hold_counts.clear()

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


def _positive_scale(value: float, name: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


__all__ = [
    "FAST_JOINT_NAMES",
    "KeypointSmoother",
    "ONE_EURO_PROFILES",
    "OneEuroParameters",
    "OneEuroValueFilter",
    "STABLE_JOINT_NAMES",
    "smoothing_alpha",
]
