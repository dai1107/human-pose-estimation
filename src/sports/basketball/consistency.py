from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_shot_summary(path: str | Path) -> dict[str, Any]:
    shot_path = Path(path)
    if shot_path.is_dir():
        shot_path = shot_path / "shot_summary.json"
    return json.loads(shot_path.read_text(encoding="utf-8"))


def analyze_shot_consistency(shots: list[dict[str, Any]]) -> dict[str, Any]:
    release_times = []
    durations = []
    order_ok = 0
    event_offsets: dict[str, list[float]] = {}
    for shot in shots:
        clip = shot.get("clip_range", {})
        duration = clip.get("duration_ms")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
        release = shot.get("release_proxy", {})
        release_time = release.get("release_proxy_time")
        if isinstance(release_time, (int, float)):
            release_times.append(float(release_time))
        sequence = shot.get("event_sequence", {})
        if sequence.get("order_consistent_with_template"):
            order_ok += 1
        for event_name, event in sequence.get("events", {}).items():
            ts = event.get("timestamp_ms")
            if isinstance(ts, (int, float)) and isinstance(release_time, (int, float)):
                event_offsets.setdefault(event_name, []).append(float(ts) - float(release_time))
    return {
        "shot_count": len(shots),
        "duration_mean_ms": mean(durations) if durations else None,
        "duration_std_ms": pstdev(durations) if len(durations) > 1 else (0.0 if durations else None),
        "release_proxy_time_mean_ms": mean(release_times) if release_times else None,
        "release_proxy_time_std_ms": pstdev(release_times) if len(release_times) > 1 else (0.0 if release_times else None),
        "event_time_offsets_from_release_ms": {
            name: {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
            }
            for name, values in event_offsets.items()
        },
        "sequence_order_consistency": {
            "consistent_count": order_ok,
            "total_count": len(shots),
        },
        "interpretation_note": "Consistency describes repeat-to-repeat kinematic similarity, not shooting percentage or skill level.",
    }

