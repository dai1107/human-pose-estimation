from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FRAME_TIMING_FIELDS = (
    "decode_ms",
    "resize_ms",
    "color_convert_ms",
    "pose_inference_ms",
    "smoothing_ms",
    "feature_ms",
    "rule_ms",
    "draw_ms",
    "encode_ms",
    "serialize_ms",
    "web_transfer_ms",
    "total_frame_ms",
)

BOTTLENECK_FIELDS = (
    "pose_inference_ms",
    "draw_ms",
    "encode_ms",
    "web_transfer_ms",
)


def _finite_nonnegative(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0.0 else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def duration_tolerance_seconds(source_fps: float) -> float:
    fps = _finite_nonnegative(source_fps)
    return max(1.0 / fps, 0.05) if fps > 0.0 else 0.05


def video_duration_matches(
    *,
    input_frame_count: int,
    input_fps: float,
    output_frame_count: int,
    output_fps: float,
) -> tuple[bool, float, float]:
    input_rate = _finite_nonnegative(input_fps)
    output_rate = _finite_nonnegative(output_fps)
    if input_rate <= 0.0 or output_rate <= 0.0:
        return False, 0.0, duration_tolerance_seconds(input_rate)
    input_duration = max(0, int(input_frame_count)) / input_rate
    output_duration = max(0, int(output_frame_count)) / output_rate
    difference = abs(output_duration - input_duration)
    tolerance = duration_tolerance_seconds(input_rate)
    return difference <= tolerance, difference, tolerance


class UploadVideoProfiler:
    """Collect per-frame upload-video timings and persist a stable profile."""

    def __init__(
        self,
        *,
        source_fps: float,
        source_frame_count: int,
        output_dir: Path,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.source_fps = _finite_nonnegative(source_fps)
        self.source_frame_count = max(0, int(source_frame_count))
        self.output_dir = Path(output_dir)
        self.rows: list[dict[str, Any]] = []
        self.phase_totals = {field: 0.0 for field in FRAME_TIMING_FIELDS}
        self.pose_inference_count = 0
        self.pose_frame_count = 0
        self.coarse_pose_frames = 0
        self.refinement_pose_frames = 0
        self.output_fps = self.source_fps
        self.output_frame_count = 0
        self.analysis_started = 0.0
        self.summary_metadata: dict[str, Any] = {}

    def start(self, monotonic_time: float) -> None:
        self.analysis_started = float(monotonic_time)

    def set_summary_metadata(self, values: Mapping[str, object]) -> None:
        self.summary_metadata.update(dict(values))

    def record_frame(
        self,
        *,
        frame_index: int,
        timestamp_ms: float,
        timings: Mapping[str, object],
        pose_inference_ran: bool,
        pose_frame_analyzed: bool | None = None,
        pass_name: str = "full",
    ) -> None:
        if not self.enabled:
            return
        row: dict[str, Any] = {
            "frame_index": max(0, int(frame_index)),
            "timestamp_ms": round(_finite_nonnegative(timestamp_ms), 3),
        }
        for field in FRAME_TIMING_FIELDS:
            value = _finite_nonnegative(timings.get(field, 0.0))
            row[field] = round(value, 3)
            self.phase_totals[field] += value
        self.pose_inference_count += int(bool(pose_inference_ran))
        analyzed = bool(pose_inference_ran) if pose_frame_analyzed is None else bool(pose_frame_analyzed)
        self.pose_frame_count += int(analyzed)
        if analyzed and pass_name == "coarse":
            self.coarse_pose_frames += 1
        elif analyzed and pass_name == "refinement":
            self.refinement_pose_frames += 1
        row["pose_inference_count"] = self.pose_inference_count
        row.update(
            {
                "source_fps": round(self.source_fps, 6),
                "display_fps": round(self.output_fps, 6),
                "inference_fps": 0.0,
                "capture_ms": row["decode_ms"],
                "preprocess_ms": round(row["resize_ms"] + row["color_convert_ms"], 3),
                "inference_ms": row["pose_inference_ms"],
                "postprocess_ms": round(row["smoothing_ms"] + row["feature_ms"], 3),
                "render_ms": row["draw_ms"],
                "frame_timestamp_ms": row["timestamp_ms"],
                "pose_timestamp_ms": row["timestamp_ms"],
                "pose_result_age_ms": 0.0,
                "frames_read": len(self.rows) + 1,
                "frames_inferred": self.pose_inference_count,
                "frames_skipped": max(0, len(self.rows) + 1 - self.pose_frame_count),
                "frames_rendered": self.output_frame_count,
                "queue_depth": 0,
                "playback_speed_ratio": 1.0,
            }
        )
        self.rows.append(row)

    def record_output_frame(self, output_fps: float) -> None:
        if not self.enabled:
            return
        self.output_fps = _finite_nonnegative(output_fps) or self.source_fps
        self.output_frame_count += 1

    def record_refinement_frame(
        self,
        *,
        frame_index: int,
        timings: Mapping[str, object],
        pose_inference_ran: bool,
        new_pose_frame: bool = True,
    ) -> None:
        """Add pass-two costs to an already decoded source-frame row."""

        if not self.enabled:
            return
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= len(self.rows):
            raise IndexError(f"refinement frame is not in decode profile: {frame_index}")
        row = self.rows[frame_index]
        for field in FRAME_TIMING_FIELDS:
            value = _finite_nonnegative(timings.get(field, 0.0))
            if value <= 0.0:
                continue
            row[field] = round(_finite_nonnegative(row.get(field, 0.0)) + value, 3)
            self.phase_totals[field] += value
        self.pose_inference_count += int(bool(pose_inference_ran))
        self.pose_frame_count += int(bool(new_pose_frame))
        self.refinement_pose_frames += 1

    def record_replay_cost(
        self,
        *,
        frame_index: int,
        timings: Mapping[str, object],
    ) -> None:
        if not self.enabled:
            return
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= len(self.rows):
            raise IndexError(f"replay frame is not in decode profile: {frame_index}")
        row = self.rows[frame_index]
        for field in (
            "decode_ms",
            "pose_inference_ms",
            "smoothing_ms",
            "feature_ms",
            "rule_ms",
            "serialize_ms",
            "total_frame_ms",
        ):
            value = _finite_nonnegative(timings.get(field, 0.0))
            row[field] = round(_finite_nonnegative(row.get(field, 0.0)) + value, 3)
            self.phase_totals[field] += value

    def record_output_video(self, *, output_fps: float, frame_count: int) -> None:
        if not self.enabled:
            return
        self.output_fps = _finite_nonnegative(output_fps) or self.source_fps
        self.output_frame_count = max(0, int(frame_count))

    def live_metrics(self, elapsed_seconds: float) -> dict[str, Any]:
        processed = len(self.rows)
        elapsed = max(0.0, float(elapsed_seconds))
        processed_fps = processed / elapsed if elapsed > 0.0 else 0.0
        bottleneck = max(
            BOTTLENECK_FIELDS,
            key=lambda field: self.phase_totals[field],
        )
        return {
            "processed_fps": round(processed_fps, 2),
            "pose_inference_count": self.pose_inference_count,
            "pose_frames": self.pose_frame_count,
            "performance_bottleneck": bottleneck,
        }

    def summary(self, total_analysis_time_ms: float) -> dict[str, Any]:
        total_ms = _finite_nonnegative(total_analysis_time_ms)
        processed = len(self.rows)
        source_duration_ms = (
            self.source_frame_count / self.source_fps * 1000.0
            if self.source_fps > 0.0 and self.source_frame_count > 0
            else (
                processed / self.source_fps * 1000.0
                if self.source_fps > 0.0
                else 0.0
            )
        )
        processed_fps = processed / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
        real_time_factor = total_ms / source_duration_ms if source_duration_ms > 0.0 else 0.0
        analysis_speed_ratio = (
            processed_fps / self.source_fps
            if self.source_fps > 0.0
            else 0.0
        )
        timings_by_field = {
            field: [_finite_nonnegative(row[field]) for row in self.rows]
            for field in FRAME_TIMING_FIELDS
        }
        bottleneck = max(
            BOTTLENECK_FIELDS,
            key=lambda field: self.phase_totals[field],
        )
        annotated_generated = self.output_frame_count > 0
        duration_matches: bool | None = None
        duration_difference_ms: float | None = None
        duration_tolerance_ms: float | None = None
        if annotated_generated:
            duration_matches, difference, tolerance = video_duration_matches(
                input_frame_count=processed,
                input_fps=self.source_fps,
                output_frame_count=self.output_frame_count,
                output_fps=self.output_fps,
            )
            duration_difference_ms = round(difference * 1000.0, 3)
            duration_tolerance_ms = round(tolerance * 1000.0, 3)
        playback_speed_ratio = (
            (self.output_frame_count / self.output_fps)
            / (processed / self.source_fps)
            if annotated_generated
            and processed > 0
            and self.source_fps > 0.0
            and self.output_fps > 0.0
            else 1.0
        )

        summary: dict[str, Any] = {
            "source_fps": round(self.source_fps, 6),
            "source_frame_count": self.source_frame_count,
            "source_duration_ms": round(source_duration_ms, 3),
            "processed_frame_count": processed,
            "decode_frames": processed,
            "pose_frames": self.pose_frame_count,
            "coarse_pose_frames": self.coarse_pose_frames,
            "refinement_pose_frames": self.refinement_pose_frames,
            "pose_sampling_ratio": round(
                self.pose_frame_count / processed if processed > 0 else 0.0,
                6,
            ),
            "pose_inference_count": self.pose_inference_count,
            "output_fps": round(self.output_fps, 6),
            "output_frame_count": self.output_frame_count,
            "annotated_video_generated": annotated_generated,
            "output_duration_matches_input": duration_matches,
            "output_duration_difference_ms": duration_difference_ms,
            "output_duration_tolerance_ms": duration_tolerance_ms,
            "total_analysis_time_ms": round(total_ms, 3),
            "total_processing_ms": round(total_ms, 3),
            "processed_fps": round(processed_fps, 3),
            "processing_fps": round(processed_fps, 3),
            "real_time_factor": round(real_time_factor, 6),
            "analysis_speed_ratio": round(analysis_speed_ratio, 6),
            "normal_speed_analysis_passed": (
                analysis_speed_ratio >= 1.0
                if processed > 0 and self.source_fps > 0.0
                else None
            ),
            "display_fps": round(self.output_fps, 6),
            "inference_fps": round(
                self.pose_inference_count / (total_ms / 1000.0)
                if total_ms > 0.0
                else 0.0,
                3,
            ),
            "capture_ms": round(
                statistics.mean(timings_by_field["decode_ms"])
                if timings_by_field["decode_ms"]
                else 0.0,
                3,
            ),
            "preprocess_ms": round(
                statistics.mean(
                    [
                        resize + color
                        for resize, color in zip(
                            timings_by_field["resize_ms"],
                            timings_by_field["color_convert_ms"],
                        )
                    ]
                )
                if processed
                else 0.0,
                3,
            ),
            "inference_ms": round(
                statistics.mean(timings_by_field["pose_inference_ms"])
                if timings_by_field["pose_inference_ms"]
                else 0.0,
                3,
            ),
            "postprocess_ms": round(
                statistics.mean(
                    [
                        smoothing + feature
                        for smoothing, feature in zip(
                            timings_by_field["smoothing_ms"],
                            timings_by_field["feature_ms"],
                        )
                    ]
                )
                if processed
                else 0.0,
                3,
            ),
            "rule_ms": round(
                statistics.mean(timings_by_field["rule_ms"])
                if timings_by_field["rule_ms"]
                else 0.0,
                3,
            ),
            "render_ms": round(
                statistics.mean(timings_by_field["draw_ms"])
                if timings_by_field["draw_ms"]
                else 0.0,
                3,
            ),
            "frame_timestamp_ms": round(
                _finite_nonnegative(self.rows[-1]["timestamp_ms"]) if self.rows else 0.0,
                3,
            ),
            "pose_timestamp_ms": round(
                _finite_nonnegative(self.rows[-1]["timestamp_ms"]) if self.rows else 0.0,
                3,
            ),
            "pose_result_age_ms": 0.0,
            "frames_read": processed,
            "frames_inferred": self.pose_inference_count,
            "frames_skipped": max(0, processed - self.pose_frame_count),
            "frames_rendered": self.output_frame_count,
            "queue_depth": 0,
            "playback_speed_ratio": round(playback_speed_ratio, 6),
            "p50_inference_latency_ms": round(
                statistics.median(timings_by_field["pose_inference_ms"])
                if timings_by_field["pose_inference_ms"]
                else 0.0,
                3,
            ),
            "p95_inference_latency_ms": round(
                _percentile(timings_by_field["pose_inference_ms"], 0.95), 3
            ),
            "p50_pose_result_age_ms": 0.0,
            "p95_pose_result_age_ms": 0.0,
            "primary_bottleneck": bottleneck,
            "decode_ms": round(self.phase_totals["decode_ms"], 3),
            "pose_inference_ms": round(
                self.phase_totals["pose_inference_ms"],
                3,
            ),
            "rule_engine_ms": round(self.phase_totals["rule_ms"], 3),
            "report_ms": 0.0,
            "phase_total_ms": {
                field: round(value, 3)
                for field, value in self.phase_totals.items()
            },
        }
        for field, values in timings_by_field.items():
            summary[f"p50_{field}"] = round(statistics.median(values), 3) if values else 0.0
            summary[f"p95_{field}"] = round(_percentile(values, 0.95), 3)
        summary.update(self.summary_metadata)
        return summary

    def write(self, total_analysis_time_ms: float) -> tuple[Path, Path, dict[str, Any]]:
        report_started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self.output_dir / "upload_video_profile.csv"
        summary_path = self.output_dir / "upload_video_summary.json"
        fieldnames = [
            "frame_index",
            "timestamp_ms",
            *FRAME_TIMING_FIELDS,
            "pose_inference_count",
            "source_fps",
            "display_fps",
            "inference_fps",
            "capture_ms",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "render_ms",
            "frame_timestamp_ms",
            "pose_timestamp_ms",
            "pose_result_age_ms",
            "frames_read",
            "frames_inferred",
            "frames_skipped",
            "frames_rendered",
            "queue_depth",
            "playback_speed_ratio",
        ]
        with profile_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        summary = self.summary(total_analysis_time_ms)
        summary["report_ms"] = round(
            (time.perf_counter() - report_started) * 1000.0,
            3,
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return profile_path, summary_path, summary
