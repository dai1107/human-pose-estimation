from __future__ import annotations

from math import isfinite
from typing import Any

from .schema import PhaseResult, ReleaseProxy, ShotFrame


def detect_shot_phases(
    frames: list[ShotFrame],
    release_proxy: ReleaseProxy | None = None,
    config: dict[str, Any] | None = None,
) -> PhaseResult:
    if not frames:
        return PhaseResult([], {phase: None for phase in ("SETUP", "DIP", "RISE", "ARM_EXTENSION", "RELEASE_PROXY", "FOLLOW_THROUGH", "RECOVERY")}, True, ["no frames available"])
    usable = [frame for frame in frames if frame.usable(0.2)]
    if len(usable) < max(3, len(frames) // 3):
        return PhaseResult(
            [{"frame_index": frame.frame_index, "timestamp_ms": frame.timestamp_ms, "phase": "IDLE"} for frame in frames],
            {phase: None for phase in ("SETUP", "DIP", "RISE", "ARM_EXTENSION", "RELEASE_PROXY", "FOLLOW_THROUGH", "RECOVERY")},
            True,
            ["too few usable pose frames"],
        )

    baseline_y = next((frame.pelvis_y for frame in frames if frame.pelvis_y is not None), None)
    scale = next((frame.body_scale for frame in frames if frame.body_scale is not None and frame.body_scale > 1e-9), 1.0)
    displacement = [((frame.pelvis_y - baseline_y) / scale) if frame.pelvis_y is not None and baseline_y is not None and scale else 0.0 for frame in frames]
    knee_values = [frame.shooting_knee_angle if frame.shooting_knee_angle is not None else 180.0 for frame in frames]
    elbow_values = [frame.shooting_elbow_angle if frame.shooting_elbow_angle is not None else 0.0 for frame in frames]
    wrist_height = [frame.wrist_relative_shoulder_height() if frame.wrist_relative_shoulder_height() is not None else -1.0 for frame in frames]

    dip_index = int(max(range(len(frames)), key=lambda index: displacement[index]))
    rise_index = min(len(frames) - 1, dip_index + 1)
    if dip_index == 0 and min(knee_values) < knee_values[0] - 5:
        dip_index = int(min(range(len(frames)), key=lambda index: knee_values[index]))
    arm_candidates = [
        index
        for index in range(max(0, rise_index + 1), len(frames))
        if elbow_values[index] >= elbow_values[max(0, index - 1)] and wrist_height[index] >= -0.05
    ]
    arm_index = arm_candidates[0] if arm_candidates else max(rise_index, int(len(frames) * 0.55))
    arm_index = max(arm_index, min(len(frames) - 1, rise_index + 1))
    release_index = _nearest_index(frames, release_proxy.release_proxy_time) if release_proxy and release_proxy.release_proxy_time is not None else min(len(frames) - 1, arm_index + max(1, len(frames) // 8))
    follow_index = min(len(frames) - 1, release_index + max(1, len(frames) // 8))
    recovery_index = min(len(frames) - 1, follow_index + max(1, len(frames) // 8))

    phase_rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not frame.usable(0.2):
            phase = "IDLE"
        elif index < max(1, dip_index):
            phase = "SETUP"
        elif index <= dip_index:
            phase = "DIP"
        elif index < arm_index:
            phase = "RISE"
        elif index < release_index:
            phase = "ARM_EXTENSION"
        elif index == release_index:
            phase = "RELEASE_PROXY"
        elif index <= follow_index:
            phase = "FOLLOW_THROUGH"
        else:
            phase = "RECOVERY"
        phase_rows.append({"frame_index": frame.frame_index, "timestamp_ms": frame.timestamp_ms, "phase": phase})
    timestamps = {
        "SETUP": _first_phase_ts(phase_rows, "SETUP"),
        "DIP": _first_phase_ts(phase_rows, "DIP"),
        "RISE": _first_phase_ts(phase_rows, "RISE"),
        "ARM_EXTENSION": _first_phase_ts(phase_rows, "ARM_EXTENSION"),
        "RELEASE_PROXY": release_proxy.release_proxy_time if release_proxy and release_proxy.release_proxy_time is not None else _first_phase_ts(phase_rows, "RELEASE_PROXY"),
        "FOLLOW_THROUGH": _first_phase_ts(phase_rows, "FOLLOW_THROUGH"),
        "RECOVERY": _first_phase_ts(phase_rows, "RECOVERY"),
    }
    warnings: list[str] = []
    if release_proxy is None or release_proxy.release_proxy_time is None:
        warnings.append("release proxy is uncertain")
    return PhaseResult(phase_rows, timestamps, bool(warnings), warnings)


def _nearest_index(frames: list[ShotFrame], timestamp_ms: int | None) -> int:
    if timestamp_ms is None:
        return len(frames) // 2
    return int(min(range(len(frames)), key=lambda index: abs(frames[index].timestamp_ms - timestamp_ms)))


def _first_phase_ts(rows: list[dict[str, Any]], phase: str) -> int | None:
    for row in rows:
        if row["phase"] == phase:
            return int(row["timestamp_ms"])
    return None
