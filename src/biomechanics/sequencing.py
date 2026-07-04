from __future__ import annotations

from math import isfinite
from typing import Iterable, Sequence


def find_local_peaks(
    signal: Sequence[float],
    timestamps: Sequence[int | float],
    min_distance_ms: int = 120,
    min_prominence: float = 0.0,
) -> list[dict[str, float]]:
    values = [float(value) for value in signal]
    times = [float(timestamp) for timestamp in timestamps]
    if len(values) != len(times) or len(values) < 3:
        return []

    candidates: list[dict[str, float]] = []
    for index in range(1, len(values) - 1):
        value = values[index]
        left = values[index - 1]
        right = values[index + 1]
        if not (isfinite(value) and isfinite(left) and isfinite(right)):
            continue
        if value < left or value < right:
            continue
        prominence = value - max(left, right)
        if prominence < min_prominence:
            continue
        candidates.append(
            {
                "index": float(index),
                "timestamp_ms": times[index],
                "value": value,
                "prominence": prominence,
            }
        )

    selected: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=lambda item: item["value"], reverse=True):
        if all(abs(candidate["timestamp_ms"] - peak["timestamp_ms"]) >= min_distance_ms for peak in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: item["timestamp_ms"])


def compare_peak_order(
    events: dict[str, float | int | None],
    expected_order: Iterable[str] | None = None,
) -> dict[str, object]:
    order = list(expected_order) if expected_order is not None else list(events.keys())
    available: list[tuple[str, float]] = []
    missing: list[str] = []
    for name in order:
        value = events.get(name)
        if value is None:
            missing.append(name)
            continue
        timestamp = float(value)
        if not isfinite(timestamp):
            missing.append(name)
            continue
        available.append((name, timestamp))

    deltas: dict[str, float] = {}
    for (prev_name, prev_ts), (cur_name, cur_ts) in zip(available, available[1:]):
        deltas[f"{prev_name}_to_{cur_name}"] = cur_ts - prev_ts

    in_expected_order = all(first[1] <= second[1] for first, second in zip(available, available[1:]))
    chronological = sorted(available, key=lambda item: item[1])
    description = " -> ".join(name for name, _ in chronological) if chronological else "no available peaks"

    return {
        "in_expected_order": in_expected_order and not missing,
        "available_events": {name: timestamp for name, timestamp in available},
        "chronological_order": [name for name, _ in chronological],
        "time_deltas_ms": deltas,
        "missing_events": missing,
        "description": description,
    }

