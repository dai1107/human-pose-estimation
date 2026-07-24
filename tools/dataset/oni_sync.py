"""ONI-internal Color/Depth timestamp synchronization helpers."""

from __future__ import annotations

import bisect
import csv
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def read_timeline(index_path: Path) -> list[dict[str, int]]:
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "output_frame": int(row["output_frame"]),
            "source_frame_index": int(row["source_frame_index"]),
            "timestamp_us": int(row["timestamp_us"]),
        }
        for row in rows
    ]


def can_pair_by_frame_index(
    color: Sequence[Mapping[str, int]],
    depth: Sequence[Mapping[str, int]],
    *,
    capture_sync_confirmed: bool,
) -> bool:
    if not capture_sync_confirmed or not color or not depth:
        return False
    depth_indices = {row["source_frame_index"] for row in depth}
    return all(row["source_frame_index"] in depth_indices for row in color)


def pair_by_frame_index(
    color: Sequence[Mapping[str, int]],
    depth: Sequence[Mapping[str, int]],
) -> list[dict[str, Any]]:
    depth_by_index = {
        row["source_frame_index"]: row for row in depth
    }
    pairs: list[dict[str, Any]] = []
    for rgb in color:
        matched = depth_by_index.get(rgb["source_frame_index"])
        if matched is None:
            continue
        delta = matched["timestamp_us"] - rgb["timestamp_us"]
        pairs.append(
            {
                "rgb_frame": rgb["output_frame"],
                "depth_frame": matched["output_frame"],
                "rgb_source_frame_index": rgb["source_frame_index"],
                "depth_source_frame_index": matched[
                    "source_frame_index"
                ],
                "rgb_timestamp_us": rgb["timestamp_us"],
                "depth_timestamp_us": matched["timestamp_us"],
                "delta_us": delta,
                "sync_method": "frame_index",
            }
        )
    return pairs


def pair_by_nearest_timestamp(
    color: Sequence[Mapping[str, int]],
    depth: Sequence[Mapping[str, int]],
) -> list[dict[str, Any]]:
    if not color or not depth:
        return []
    depth_timestamps = [row["timestamp_us"] for row in depth]
    pairs: list[dict[str, Any]] = []
    for rgb in color:
        timestamp = rgb["timestamp_us"]
        position = bisect.bisect_left(depth_timestamps, timestamp)
        candidates = []
        if position < len(depth):
            candidates.append(depth[position])
        if position > 0:
            candidates.append(depth[position - 1])
        matched = min(
            candidates,
            key=lambda row: (
                abs(row["timestamp_us"] - timestamp),
                row["timestamp_us"],
                row["output_frame"],
            ),
        )
        delta = matched["timestamp_us"] - timestamp
        pairs.append(
            {
                "rgb_frame": rgb["output_frame"],
                "depth_frame": matched["output_frame"],
                "rgb_source_frame_index": rgb["source_frame_index"],
                "depth_source_frame_index": matched[
                    "source_frame_index"
                ],
                "rgb_timestamp_us": timestamp,
                "depth_timestamp_us": matched["timestamp_us"],
                "delta_us": delta,
                "sync_method": "nearest_timestamp",
            }
        )
    return pairs


def grade_sync(
    *,
    pair_count: int,
    rgb_frame_count: int,
    p95_error_ms: float,
) -> str:
    if pair_count == 0 or rgb_frame_count == 0:
        return "invalid"
    coverage = pair_count / rgb_frame_count
    if coverage >= 0.99 and p95_error_ms <= 33.5:
        return "good"
    if coverage >= 0.95 and p95_error_ms <= 50.0:
        return "usable"
    if coverage >= 0.80 and p95_error_ms <= 100.0:
        return "video_level_only"
    return "invalid"


def sync_statistics(
    pairs: Sequence[Mapping[str, Any]],
    *,
    rgb_frame_count: int,
) -> dict[str, Any]:
    if not pairs:
        return {
            "pair_count": 0,
            "rgb_frame_count": rgb_frame_count,
            "coverage_ratio": 0.0,
            "offset_ms": None,
            "drift_scale": 1.0,
            "median_error_ms": None,
            "p95_error_ms": None,
            "max_error_ms": None,
            "sync_quality": "invalid",
        }
    signed_ms = [float(pair["delta_us"]) / 1000.0 for pair in pairs]
    absolute_ms = [abs(value) for value in signed_ms]
    p95 = percentile(absolute_ms, 95.0)
    return {
        "pair_count": len(pairs),
        "rgb_frame_count": rgb_frame_count,
        "coverage_ratio": len(pairs) / rgb_frame_count,
        "offset_ms": statistics.median(signed_ms),
        "drift_scale": 1.0,
        "median_error_ms": statistics.median(absolute_ms),
        "p95_error_ms": p95,
        "max_error_ms": max(absolute_ms),
        "sync_quality": grade_sync(
            pair_count=len(pairs),
            rgb_frame_count=rgb_frame_count,
            p95_error_ms=p95,
        ),
    }


def build_sync_report(
    record: Mapping[str, Any],
    export_metadata: Mapping[str, Any],
    color: Sequence[Mapping[str, int]],
    depth: Sequence[Mapping[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    color_exists = bool(
        export_metadata["streams"]["color"]["exists"]
    )
    depth_exists = bool(
        export_metadata["streams"]["depth"]["exists"]
    )
    base = {
        "schema_version": 1,
        "artifact_type": "oni_internal_color_depth_sync",
        "record_id": record["record_id"],
        "source_filename": record["source_filename"],
        "manifest_sha256": record["sha256"],
        "sync_scope": "oni_internal_color_depth",
        "phone_pairing_used": False,
        "paired_group_id": record.get("paired_group_id"),
        "capture_sync_confirmed": bool(
            record.get("oni_internal_sync", {}).get(
                "capture_sync_confirmed", False
            )
        ),
        "color_present": color_exists,
        "depth_present": depth_exists,
    }
    if not color_exists or not depth_exists:
        missing = []
        if not color_exists:
            missing.append("color")
        if not depth_exists:
            missing.append("depth")
        report = {
            **base,
            "applicable": False,
            "reason": "missing_" + "_and_".join(missing),
            "sync_method": "not_applicable",
            "pair_count": 0,
            "rgb_frame_count": len(color),
            "depth_frame_count": len(depth),
            "coverage_ratio": 0.0,
            "offset_ms": None,
            "drift_scale": 1.0,
            "median_error_ms": None,
            "p95_error_ms": None,
            "max_error_ms": None,
            "sync_quality": "video_level_only",
            "fine_event_training_eligible": False,
            "original_timestamps_preserved": True,
            "validation_errors": [],
        }
        return report, []

    use_frame_index = can_pair_by_frame_index(
        color,
        depth,
        capture_sync_confirmed=base["capture_sync_confirmed"],
    )
    if use_frame_index:
        pairs = pair_by_frame_index(color, depth)
        method = "frame_index"
    else:
        pairs = pair_by_nearest_timestamp(color, depth)
        method = "nearest_timestamp"
    stats = sync_statistics(pairs, rgb_frame_count=len(color))
    quality = stats["sync_quality"]
    report = {
        **base,
        "applicable": True,
        "reason": None,
        "sync_method": method,
        "depth_frame_count": len(depth),
        **stats,
        "fine_event_training_eligible": quality in {"good", "usable"},
        "original_timestamps_preserved": True,
        "validation_errors": [],
    }
    if record.get("paired_group_id") is not None:
        report["validation_errors"].append(
            "current ONI must not have paired_group_id"
        )
    return report, pairs


PAIR_FIELDS = (
    "rgb_frame",
    "depth_frame",
    "rgb_source_frame_index",
    "depth_source_frame_index",
    "rgb_timestamp_us",
    "depth_timestamp_us",
    "delta_us",
    "sync_method",
)


def write_pairs(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pairs)
