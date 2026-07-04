from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np

from .schema import ReleaseProxy, ShotEvent, ShotFrame, finite_or_none


def estimate_release_proxy(
    frames: list[ShotFrame],
    feature_rows: list[dict[str, Any]],
    manual_release_ms: int | None = None,
    config: dict[str, Any] | None = None,
) -> ReleaseProxy:
    auto = _estimate_auto_release(frames, feature_rows, config or {})
    if manual_release_ms is not None:
        return ReleaseProxy(
            release_proxy_time=int(manual_release_ms),
            release_proxy_confidence=1.0,
            release_proxy_reason="manual release proxy override supplied by user",
            release_source="manual",
            automatic_time=auto.release_proxy_time,
            automatic_confidence=auto.release_proxy_confidence,
        )
    return auto


def _estimate_auto_release(frames: list[ShotFrame], feature_rows: list[dict[str, Any]], config: dict[str, Any]) -> ReleaseProxy:
    if not frames:
        return ReleaseProxy(None, 0.0, "no frames available", "auto")
    speeds = np.array([finite_or_none(row.get("shooting_wrist_speed")) or 0.0 for row in feature_rows], dtype=float)
    elbow_velocity = np.array([finite_or_none(row.get("shooting_elbow_angular_velocity")) or 0.0 for row in feature_rows], dtype=float)
    wrist_height = np.array([finite_or_none(row.get("wrist_relative_shoulder_height")) or -10.0 for row in feature_rows], dtype=float)
    visibility = np.array([frame.visibility_mean if frame.visibility_mean is not None else 0.7 for frame in frames], dtype=float)
    if speeds.size == 0 or float(np.max(speeds)) <= 0:
        return ReleaseProxy(None, 0.0, "shooting wrist speed unavailable", "auto")

    speed_score = speeds / max(float(np.max(speeds)), 1e-9)
    elbow_score = np.clip(elbow_velocity / max(float(np.max(np.abs(elbow_velocity))), 1e-9), 0.0, 1.0)
    height_score = np.clip((wrist_height + 0.1) / 0.5, 0.0, 1.0)
    follow_score = np.zeros_like(speed_score)
    for index in range(len(frames)):
        future = wrist_height[index : min(len(frames), index + 4)]
        follow_score[index] = 1.0 if future.size and float(np.nanmax(future)) >= wrist_height[index] - 0.08 else 0.4
    quality_score = np.clip(visibility, 0.0, 1.0)
    score = 0.35 * speed_score + 0.25 * elbow_score + 0.25 * height_score + 0.15 * follow_score
    score *= quality_score
    best_index = int(np.argmax(score))
    confidence = float(np.clip(score[best_index], 0.0, 1.0))
    if confidence < float(config.get("minimum_confidence", 0.25)):
        return ReleaseProxy(None, confidence, "release proxy conditions were weak or low visibility", "auto")
    reasons: list[str] = []
    if speed_score[best_index] >= 0.75:
        reasons.append("shooting wrist speed local high value")
    if elbow_score[best_index] >= 0.4:
        reasons.append("shooting elbow extension signal")
    if height_score[best_index] >= 0.5:
        reasons.append("wrist near or above shooting shoulder")
    if follow_score[best_index] >= 0.8:
        reasons.append("short follow-through trend after proxy frame")
    if visibility[best_index] < 0.55:
        reasons.append("low visibility reduced confidence")
    return ReleaseProxy(frames[best_index].timestamp_ms, confidence, " + ".join(reasons) if reasons else "best combined human-keypoint proxy", "auto")


def release_event(proxy: ReleaseProxy, frames: list[ShotFrame]) -> ShotEvent:
    if proxy.release_proxy_time is None or not frames:
        return ShotEvent("release_proxy_time", None, None, proxy.release_proxy_confidence, "release_proxy", "LOW")
    first = frames[0].timestamp_ms
    duration = max(frames[-1].timestamp_ms - first, 1)
    return ShotEvent(
        "release_proxy_time",
        proxy.release_proxy_time,
        (proxy.release_proxy_time - first) / duration * 100.0,
        proxy.release_proxy_confidence,
        proxy.release_source,
        "GOOD" if proxy.release_proxy_confidence >= 0.7 else "WARNING",
    )

