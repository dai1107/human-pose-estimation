"""Bounded, causal motion context for realtime pose analysis."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalPoseBufferConfig:
    window_ms: float = 500.0
    maximum_samples: int = 32
    landmark_jump_body_ratio: float = 0.22
    trend_deadband_deg_s: float = 12.0


@dataclass(frozen=True, slots=True)
class _TemporalSample:
    timestamp_ms: float
    landmarks: Mapping[str, tuple[float, float, float]]
    visibility: Mapping[str, float]
    angles: Mapping[str, float]
    three_d_reliability: float
    contacts: Mapping[str, float]
    phase: str


class TemporalPoseBuffer:
    """Keep only the recent past and expose lightweight causal evidence.

    The buffer never delays a frame and never reads future samples.  Its output
    is auxiliary/debug evidence; established HYROX thresholds do not consume it.
    """

    def __init__(self, config: TemporalPoseBufferConfig | None = None) -> None:
        self.config = config or TemporalPoseBufferConfig()
        self._samples: deque[_TemporalSample] = deque(
            maxlen=max(2, int(self.config.maximum_samples))
        )

    def reset(self) -> None:
        self._samples.clear()

    def update(
        self,
        landmarks: Sequence[object],
        *,
        timestamp_ms: int | float,
        features: Mapping[str, object] | None = None,
        three_d_kinematics: Mapping[str, object] | None = None,
        phase: str = "unknown",
    ) -> dict[str, Any]:
        timestamp = _finite(timestamp_ms, 0.0)
        if self._samples and timestamp <= self._samples[-1].timestamp_ms:
            # A restarted/changed stream must not be compared with old motion.
            self.reset()
        points, visibility = _landmark_maps(landmarks)
        angles = _angle_values(features or {})
        three_d = three_d_kinematics or {}
        reliability = _finite(three_d.get("three_d_reliable_ratio"), 0.0)
        contacts = _contact_confidences(three_d)
        sample = _TemporalSample(
            timestamp_ms=timestamp,
            landmarks=points,
            visibility=visibility,
            angles=angles,
            three_d_reliability=max(0.0, min(1.0, reliability)),
            contacts=contacts,
            phase=str(phase or "unknown"),
        )
        self._samples.append(sample)
        cutoff = timestamp - max(1.0, float(self.config.window_ms))
        while len(self._samples) > 1 and self._samples[0].timestamp_ms < cutoff:
            self._samples.popleft()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        if not self._samples:
            return _empty_snapshot(self.config.window_ms)
        current = self._samples[-1]
        oldest = self._samples[0]
        duration_ms = max(0.0, current.timestamp_ms - oldest.timestamp_ms)
        velocities, outliers = self._landmark_velocities()
        angle_velocity = self._angle_velocities()
        phase_start = current.timestamp_ms
        for sample in reversed(self._samples):
            if sample.phase != current.phase:
                break
            phase_start = sample.timestamp_ms
        contact_dwell: dict[str, float] = {}
        for side in ("left", "right"):
            dwell_start = current.timestamp_ms
            for sample in reversed(self._samples):
                if sample.contacts.get(side, 0.0) < 0.60:
                    break
                dwell_start = sample.timestamp_ms
            contact_dwell[side] = max(0.0, current.timestamp_ms - dwell_start)
        return {
            "schema_version": 1,
            "causal_only": True,
            "future_frames_used": False,
            "formal_rule_replacement_allowed": False,
            "window_limit_ms": float(self.config.window_ms),
            "window_duration_ms": duration_ms,
            "sample_count": len(self._samples),
            "phase": current.phase,
            "phase_duration_ms": max(0.0, current.timestamp_ms - phase_start),
            "mean_visibility": (
                float(np.mean(tuple(current.visibility.values())))
                if current.visibility
                else 0.0
            ),
            "three_d_reliable_ratio": current.three_d_reliability,
            "landmark_velocity_per_s": velocities,
            "landmark_outliers": outliers,
            "angle_velocity_deg_s": angle_velocity,
            "angle_trends": {
                name: (
                    "increasing"
                    if velocity > self.config.trend_deadband_deg_s
                    else "decreasing"
                    if velocity < -self.config.trend_deadband_deg_s
                    else "stable"
                )
                for name, velocity in angle_velocity.items()
            },
            "contact_confidence": dict(current.contacts),
            "contact_dwell_ms": contact_dwell,
        }

    def _landmark_velocities(self) -> tuple[dict[str, float], list[str]]:
        if len(self._samples) < 2:
            return {}, []
        previous, current = self._samples[-2], self._samples[-1]
        dt = (current.timestamp_ms - previous.timestamp_ms) / 1000.0
        if dt <= 0.0:
            return {}, []
        scale = _body_scale(current.landmarks)
        velocities: dict[str, float] = {}
        outliers: list[str] = []
        for name, point in current.landmarks.items():
            prior = previous.landmarks.get(name)
            if prior is None:
                continue
            displacement = float(np.linalg.norm(np.asarray(point) - np.asarray(prior)))
            velocity = displacement / dt
            velocities[name] = velocity
            if scale > 1e-8 and displacement / scale > self.config.landmark_jump_body_ratio:
                outliers.append(name)
        return dict(sorted(velocities.items())), sorted(outliers)

    def _angle_velocities(self) -> dict[str, float]:
        if len(self._samples) < 2:
            return {}
        previous, current = self._samples[-2], self._samples[-1]
        dt = (current.timestamp_ms - previous.timestamp_ms) / 1000.0
        if dt <= 0.0:
            return {}
        return {
            name: (value - previous.angles[name]) / dt
            for name, value in sorted(current.angles.items())
            if name in previous.angles
        }


def _landmark_maps(
    landmarks: Sequence[object],
) -> tuple[dict[str, tuple[float, float, float]], dict[str, float]]:
    points: dict[str, tuple[float, float, float]] = {}
    visibility: dict[str, float] = {}
    for item in landmarks:
        name = _value(item, "name")
        if not name:
            continue
        coordinates = tuple(_finite(_value(item, axis), float("nan")) for axis in ("x", "y", "z"))
        if all(isfinite(value) for value in coordinates):
            points[str(name)] = coordinates
        quality = _value(item, "visibility")
        if quality is None:
            quality = _value(item, "confidence")
        visibility[str(name)] = max(0.0, min(1.0, _finite(quality, 0.0)))
    return points, visibility


def _angle_values(features: Mapping[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in features.items():
        if "angle" not in str(name).lower():
            continue
        number = _finite(value, float("nan"))
        if isfinite(number):
            result[str(name)] = number
    return result


def _contact_confidences(three_d: Mapping[str, object]) -> dict[str, float]:
    raw = three_d.get("foot_contact_evidence")
    if not isinstance(raw, Mapping):
        return {"left": 0.0, "right": 0.0}
    result: dict[str, float] = {}
    for side in ("left", "right"):
        item = raw.get(side)
        value = item.get("foot_contact_confidence") if isinstance(item, Mapping) else 0.0
        result[side] = max(0.0, min(1.0, _finite(value, 0.0)))
    return result


def _body_scale(points: Mapping[str, tuple[float, float, float]]) -> float:
    values: list[float] = []
    for first, second in (
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
    ):
        if first in points and second in points:
            values.append(float(np.linalg.norm(np.asarray(points[first]) - np.asarray(points[second]))))
    return float(np.median(values)) if values else 0.0


def _value(item: object, name: str) -> object:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if isfinite(number) else default


def _empty_snapshot(window_ms: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "causal_only": True,
        "future_frames_used": False,
        "formal_rule_replacement_allowed": False,
        "window_limit_ms": float(window_ms),
        "window_duration_ms": 0.0,
        "sample_count": 0,
    }


__all__ = ["TemporalPoseBuffer", "TemporalPoseBufferConfig"]
