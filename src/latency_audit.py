"""Sensor-to-photon timing records and report helpers.

All timestamps within one record must use the same monotonic clock.  Web
records deliberately keep browser and server timestamps separate; cross-host
durations are calculated from the browser clock (RTT), never by subtracting a
browser timestamp from a server timestamp.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


DERIVED_LATENCY_FIELDS = (
    "capture_to_submit_ms",
    "submit_to_result_ms",
    "result_to_render_ms",
    "render_to_expected_display_ms",
    "pose_age_at_render_ms",
    "video_frame_age_at_render_ms",
    "pose_video_age_difference_ms",
    "render_time_ms",
    "render_loop_p95_ms",
    "canvas_draw_p95_ms",
    "dom_update_p95_ms",
    "main_thread_long_task_count",
    "main_thread_long_task_total_ms",
    "long_task_render_count",
    "long_task_dom_update_count",
    "long_task_frame_copy_count",
    "long_task_encode_count",
    "long_task_pose_transfer_count",
    "long_task_other_count",
)


def _number(values: Mapping[str, Any], name: str) -> float | None:
    value = values.get(name)
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _delta(values: Mapping[str, Any], end: str, start: str, *, signed: bool = False) -> float | None:
    end_value = _number(values, end)
    start_value = _number(values, start)
    if end_value is None or start_value is None:
        return None
    value = end_value - start_value
    return round(value if signed else max(0.0, value), 3)


def derive_web_latencies(timing: Mapping[str, Any]) -> dict[str, float | None]:
    """Derive browser-visible latency segments from one rendered pose frame."""

    return {
        "capture_to_submit_ms": _delta(timing, "socket_send_ms", "camera_frame_presented_ms"),
        "submit_to_result_ms": _delta(timing, "client_result_receive_ms", "socket_send_ms"),
        "result_to_render_ms": _delta(timing, "pose_render_start_ms", "client_result_receive_ms"),
        # Negative means drawing completed before the compositor deadline.
        "render_to_expected_display_ms": _delta(
            timing, "expected_display_time_ms", "pose_render_end_ms", signed=True
        ),
        "pose_age_at_render_ms": _delta(
            timing, "pose_render_start_ms", "camera_frame_presented_ms"
        ),
        "video_frame_age_at_render_ms": _delta(
            timing, "pose_render_start_ms", "video_frame_presented_at_render_ms"
        ),
        "pose_video_age_difference_ms": _delta(
            timing, "video_frame_presented_at_render_ms", "camera_frame_presented_ms"
        ),
        "render_time_ms": _delta(timing, "pose_render_end_ms", "pose_render_start_ms"),
        "render_loop_p95_ms": _number(timing, "render_loop_p95_ms"),
        "canvas_draw_p95_ms": _number(timing, "canvas_draw_p95_ms"),
        "dom_update_p95_ms": _number(timing, "dom_update_p95_ms"),
        "main_thread_long_task_count": _number(timing, "main_thread_long_task_count"),
        "main_thread_long_task_total_ms": _number(
            timing, "main_thread_long_task_duration_ms"
        ),
        "long_task_render_count": _number(timing, "long_task_render_count"),
        "long_task_dom_update_count": _number(timing, "long_task_dom_update_count"),
        "long_task_frame_copy_count": _number(timing, "long_task_frame_copy_count"),
        "long_task_encode_count": _number(timing, "long_task_encode_count"),
        "long_task_pose_transfer_count": _number(timing, "long_task_pose_transfer_count"),
        "long_task_other_count": _number(timing, "long_task_other_count"),
    }


def derive_desktop_latencies(timing: Mapping[str, Any]) -> dict[str, float | None]:
    """Derive desktop latency segments from nanosecond OpenCV timestamps."""

    def ns_delta(end: str, start: str, *, signed: bool = False) -> float | None:
        value = _delta(timing, end, start, signed=signed)
        return None if value is None else round(value / 1_000_000.0, 3)

    return {
        "capture_to_submit_ms": ns_delta("inference_start_ns", "capture_read_end_ns"),
        "submit_to_result_ms": ns_delta("inference_end_ns", "inference_start_ns"),
        "result_to_render_ms": ns_delta("draw_start_ns", "inference_end_ns"),
        # OpenCV exposes no compositor deadline; this is the measurable submit proxy.
        "render_to_expected_display_ms": ns_delta("imshow_return_ns", "draw_end_ns"),
        "pose_age_at_render_ms": ns_delta("draw_start_ns", "pose_capture_timestamp_ns"),
        "video_frame_age_at_render_ms": ns_delta("draw_start_ns", "capture_read_end_ns"),
        "pose_video_age_difference_ms": ns_delta(
            "capture_read_end_ns", "pose_capture_timestamp_ns"
        ),
        "render_time_ms": ns_delta("draw_end_ns", "draw_start_ns"),
        "render_loop_p95_ms": None,
        "canvas_draw_p95_ms": None,
        "dom_update_p95_ms": None,
        "main_thread_long_task_count": None,
        "main_thread_long_task_total_ms": None,
        "long_task_render_count": None,
        "long_task_dom_update_count": None,
        "long_task_frame_copy_count": None,
        "long_task_encode_count": None,
        "long_task_pose_transfer_count": None,
        "long_task_other_count": None,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = rank - lower
    return round(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction, 3)


def summarize_latency_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = [dict(sample) for sample in samples]
    summary: dict[str, Any] = {"sample_count": len(records), "metrics": {}}
    for field_name in DERIVED_LATENCY_FIELDS:
        values = [value for row in records if (value := _number(row, field_name)) is not None]
        summary["metrics"][field_name] = {
            "count": len(values),
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "average": round(mean(values), 3) if values else None,
        }
    bottleneck_fields = (
        "capture_to_submit_ms",
        "submit_to_result_ms",
        "result_to_render_ms",
    )
    ranked = [
        (name, summary["metrics"][name]["p50"])
        for name in bottleneck_fields
        if summary["metrics"][name]["p50"] is not None
    ]
    summary["primary_bottleneck"] = max(ranked, key=lambda item: item[1])[0] if ranked else None
    return summary


@dataclass(slots=True)
class LatencyAuditRecorder:
    mode: str
    samples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, timing: Mapping[str, Any], derived: Mapping[str, Any]) -> None:
        self.samples.append({"timing": dict(timing), **dict(derived)})

    def report(self, *, external_sensor_to_photon: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "summary": summarize_latency_samples(self.samples),
            "external_sensor_to_photon": dict(external_sensor_to_photon or {}),
            "samples": self.samples,
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


def external_sensor_to_photon_ms(
    *, recording_fps: float, physical_motion_frame: int, display_motion_frame: int
) -> float:
    if not math.isfinite(recording_fps) or recording_fps <= 0:
        raise ValueError("recording_fps must be positive")
    if physical_motion_frame < 0 or display_motion_frame < physical_motion_frame:
        raise ValueError("frame indices must be ordered and non-negative")
    return round((display_motion_frame - physical_motion_frame) * 1000.0 / recording_fps, 3)


__all__ = [
    "DERIVED_LATENCY_FIELDS",
    "LatencyAuditRecorder",
    "derive_desktop_latencies",
    "derive_web_latencies",
    "external_sensor_to_photon_ms",
    "summarize_latency_samples",
]
