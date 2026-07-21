from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.validation.endurance import (
    EnduranceObservation,
    EnduranceThresholds,
    build_endurance_report,
    evaluate_thresholds,
    process_rss_bytes,
)
from src.validation.golden_videos import (
    GoldenObservation,
    build_report,
    compare_observation,
    load_manifest,
)
from tools.run_endurance_test import build_parser as build_endurance_parser
from tools.validate_golden_videos import build_parser as build_golden_parser

ROOT = Path(__file__).resolve().parents[1]


def test_public_main_is_a_thin_stable_entrypoint() -> None:
    lines = (ROOT / "main.py").read_text(encoding="utf-8").splitlines()

    assert len(lines) < 80
    for module in (
        "src.realtime.app",
        "src.realtime.backend_runtime",
        "src.realtime.capture",
        "src.realtime.cli",
        "src.realtime.hyrox_analysis",
        "src.realtime.presentation",
        "src.realtime.recording",
        "src.realtime.session",
    ):
        importlib.import_module(module)


def test_retired_realtime_module_forwards_to_consolidated_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = importlib.import_module("src.realtime_pose")
    captured: list[str] = []

    def fake_main(argv: list[str]) -> int:
        captured.extend(argv)
        return 17

    monkeypatch.setattr(legacy, "_consolidated_main", fake_main)
    with pytest.warns(DeprecationWarning):
        result = legacy.main(["--camera", "2", "--landmark-profile", "upper-body"])

    assert result == 17
    assert captured[:2] == ["--backend", "mediapipe"]
    assert captured[captured.index("--camera") + 1] == "2"


def test_golden_manifest_covers_each_bundled_video_once() -> None:
    _, cases = load_manifest(ROOT / "configs" / "hyrox_golden_videos.json")

    bundled = {str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "HYROX视频").glob("*.mp4")}
    configured = {case.video for case in cases}
    assert len(cases) == 8
    assert configured == bundled
    assert all(case.expectations for case in cases)


def test_golden_comparison_reports_metric_drift() -> None:
    _, cases = load_manifest(ROOT / "configs" / "hyrox_golden_videos.json")
    case = cases[0]
    baseline = GoldenObservation(
        case_id=case.case_id,
        video=case.video,
        action=case.action,
        total_frames=133,
        pose_detected_frames=133,
        pose_detected_rate=1.0,
        candidate_count=1,
        pose_valid_rep_count=0,
        no_rep_count=1,
        unsure_count=0,
        cycle_count=0,
        rep_count=0,
        final_phase="stand",
    )

    assert compare_observation(case, baseline) == []
    drifted = GoldenObservation(**{**baseline.__dict__, "candidate_count": 9})
    failures = compare_observation(case, drifted)
    assert any("candidate_count" in failure for failure in failures)
    assert build_report([case], [drifted])["status"] == "failed"


def _endurance_observation(**overrides: object) -> EnduranceObservation:
    values: dict[str, object] = {
        "target_duration_seconds": 2.0,
        "elapsed_seconds": 2.01,
        "total_frames": 100,
        "pose_detected_frames": 99,
        "pose_detected_rate": 0.99,
        "average_fps": 49.75,
        "average_latency_ms": 18.0,
        "p95_latency_ms": 25.0,
        "memory_start_mb": 200.0,
        "memory_end_mb": 203.0,
        "memory_peak_mb": 205.0,
        "memory_growth_mb": 3.0,
        "source_reopen_count": 1,
        "read_failure_count": 0,
        "read_failure_rate": 0.0,
        "completed": True,
        "output_integrity": True,
        "final_phase": "not_enabled",
    }
    values.update(overrides)
    return EnduranceObservation(**values)  # type: ignore[arg-type]


def test_endurance_thresholds_cover_required_maturity_metrics() -> None:
    assert process_rss_bytes() > 0
    thresholds = EnduranceThresholds(
        min_fps=20.0,
        max_p95_latency_ms=100.0,
        max_memory_growth_mb=32.0,
        max_read_failure_rate=0.01,
    )
    assert evaluate_thresholds(_endurance_observation(), thresholds) == []

    failures = evaluate_thresholds(
        _endurance_observation(
            average_fps=5.0,
            p95_latency_ms=200.0,
            memory_growth_mb=64.0,
            read_failure_rate=0.1,
            output_integrity=False,
        ),
        thresholds,
    )
    assert len(failures) == 5
    report = build_endurance_report(_endurance_observation(), thresholds)
    assert report["artifact_type"] == "pose_endurance_report"
    assert report["status"] == "passed"


def test_validation_clis_support_selective_golden_and_30_60_minute_runs() -> None:
    golden = build_golden_parser().parse_args(["--case", "lunge", "--list"])
    endurance_30 = build_endurance_parser().parse_args(["--minutes", "30"])
    endurance_60 = build_endurance_parser().parse_args(["--minutes", "60"])
    smoke = build_endurance_parser().parse_args(["--duration-seconds", "2"])

    assert golden.case == ["lunge"] and golden.list is True
    assert endurance_30.minutes == 30
    assert endurance_60.minutes == 60
    assert smoke.duration_seconds == 2.0
