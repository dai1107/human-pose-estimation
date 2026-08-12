from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from webui.upload_performance import (
    FRAME_TIMING_FIELDS,
    UploadVideoProfiler,
)


def _timings(**overrides: float) -> dict[str, float]:
    values = {field: 0.0 for field in FRAME_TIMING_FIELDS}
    values.update(
        {
            "decode_ms": 2.0,
            "pose_inference_ms": 30.0,
            "draw_ms": 5.0,
            "encode_ms": 4.0,
            "total_frame_ms": 45.0,
        }
    )
    values.update(overrides)
    return values


def test_upload_performance_profile_writes_per_frame_csv_and_summary(
    tmp_path: Path,
) -> None:
    profiler = UploadVideoProfiler(
        source_fps=30.0,
        source_frame_count=2,
        output_dir=tmp_path / "performance",
    )
    profiler.start(1.0)
    profiler.record_frame(
        frame_index=0,
        timestamp_ms=0.0,
        timings=_timings(pose_inference_ms=20.0),
        pose_inference_ran=True,
    )
    profiler.record_frame(
        frame_index=1,
        timestamp_ms=1000.0 / 30.0,
        timings=_timings(pose_inference_ms=40.0),
        pose_inference_ran=True,
    )

    profile_path, summary_path, summary = profiler.write(100.0)

    with profile_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert tuple(field for field in FRAME_TIMING_FIELDS if field in rows[0]) == FRAME_TIMING_FIELDS
    assert rows[-1]["pose_inference_count"] == "2"
    assert persisted == summary
    assert summary["source_fps"] == 30.0
    assert summary["processed_frame_count"] == 2
    assert summary["pose_inference_count"] == 2
    assert summary["processed_fps"] == pytest.approx(20.0)
    assert summary["real_time_factor"] == pytest.approx(1.5)
    assert summary["analysis_speed_ratio"] == pytest.approx(2.0 / 3.0)
    assert summary["normal_speed_analysis_passed"] is False
    assert summary["p50_pose_inference_ms"] == pytest.approx(30.0)
    assert summary["p95_pose_inference_ms"] == pytest.approx(39.0)
    assert summary["primary_bottleneck"] == "pose_inference_ms"
    assert summary["playback_speed_ratio"] == pytest.approx(1.0)
    assert summary["frames_read"] == 2
    assert summary["frames_inferred"] == 2
    assert summary["queue_depth"] == 0
    assert summary["p50_inference_latency_ms"] == pytest.approx(30.0)
