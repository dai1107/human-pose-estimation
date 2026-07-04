from __future__ import annotations

from collections import deque
from math import atan2, degrees, isfinite, nan

import numpy as np

from .types import LandmarkPoint


def _finite_or_nan(value: float | None) -> float:
    if value is None:
        return nan
    return float(value) if isfinite(float(value)) else nan


def compute_motion_energy_proxy(speeds: dict[str, float]) -> float:
    weights = {
        "pelvis_speed": 1.0,
        "left_wrist_speed": 0.75,
        "right_wrist_speed": 0.75,
        "left_ankle_speed": 0.5,
        "right_ankle_speed": 0.5,
    }
    total = 0.0
    used = False
    for name, weight in weights.items():
        speed = _finite_or_nan(speeds.get(name))
        if isfinite(speed):
            total += weight * abs(speed)
            used = True
    return total if used else nan


def compute_symmetry(angles: dict[str, float], speeds: dict[str, float]) -> dict[str, float]:
    def diff(left: str, right: str, source: dict[str, float]) -> float:
        left_value = _finite_or_nan(source.get(left))
        right_value = _finite_or_nan(source.get(right))
        if not isfinite(left_value) or not isfinite(right_value):
            return nan
        return abs(left_value - right_value)

    return {
        "elbow_angle_diff": diff("left_elbow_angle", "right_elbow_angle", angles),
        "knee_angle_diff": diff("left_knee_angle", "right_knee_angle", angles),
        "hip_angle_diff": diff("left_hip_angle", "right_hip_angle", angles),
        "wrist_speed_diff": diff("left_wrist_speed", "right_wrist_speed", speeds),
        "ankle_speed_diff": diff("left_ankle_speed", "right_ankle_speed", speeds),
    }


class StabilityTracker:
    def __init__(self, window_ms: int = 1000) -> None:
        self.window_ms = max(100, int(window_ms))
        self._pelvis_history: deque[tuple[int, np.ndarray]] = deque()
        self._trunk_angle_history: deque[tuple[int, float]] = deque()

    def _trim(self, timestamp_ms: int) -> None:
        minimum_ts = timestamp_ms - self.window_ms
        while self._pelvis_history and self._pelvis_history[0][0] < minimum_ts:
            self._pelvis_history.popleft()
        while self._trunk_angle_history and self._trunk_angle_history[0][0] < minimum_ts:
            self._trunk_angle_history.popleft()

    def update(
        self,
        timestamp_ms: int,
        pelvis_center: LandmarkPoint | None,
        trunk_unit: tuple[float, float, float] | None,
    ) -> dict[str, float]:
        if pelvis_center is not None and pelvis_center.is_finite():
            self._pelvis_history.append((timestamp_ms, np.array(pelvis_center.xyz(), dtype=float)))
        if trunk_unit is not None and all(isfinite(float(value)) for value in trunk_unit[:2]):
            angle = degrees(atan2(float(trunk_unit[0]), -float(trunk_unit[1])))
            self._trunk_angle_history.append((timestamp_ms, angle))

        self._trim(timestamp_ms)

        pelvis_std = nan
        pelvis_mean_speed = nan
        if len(self._pelvis_history) >= 2:
            positions = np.array([position for _, position in self._pelvis_history], dtype=float)
            pelvis_std = float(np.linalg.norm(np.std(positions, axis=0)))
            step_speeds: list[float] = []
            for (prev_ts, prev_pos), (cur_ts, cur_pos) in zip(self._pelvis_history, list(self._pelvis_history)[1:]):
                dt = (cur_ts - prev_ts) / 1000.0
                if dt > 0:
                    step_speeds.append(float(np.linalg.norm(cur_pos - prev_pos) / dt))
            if step_speeds:
                pelvis_mean_speed = float(np.mean(step_speeds))

        trunk_angle_std = nan
        trunk_angle_range = nan
        if len(self._trunk_angle_history) >= 2:
            angles = np.array([angle for _, angle in self._trunk_angle_history], dtype=float)
            trunk_angle_std = float(np.std(angles))
            trunk_angle_range = float(np.max(angles) - np.min(angles))

        return {
            "pelvis_position_std": pelvis_std,
            "pelvis_mean_speed": pelvis_mean_speed,
            "trunk_angle_std": trunk_angle_std,
            "trunk_angle_range": trunk_angle_range,
        }

