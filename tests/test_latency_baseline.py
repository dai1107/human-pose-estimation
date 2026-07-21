from __future__ import annotations

from tools.benchmark_latency_baseline import summarize_samples


def test_latency_baseline_summary_reports_p50_and_p95() -> None:
    summary = summarize_samples([10.0, 20.0, 30.0, 40.0, 50.0])

    assert summary["p50"] == 30.0
    assert summary["p95"] == 48.0
    assert summary["mean"] == 30.0


def test_latency_baseline_summary_handles_empty_samples() -> None:
    assert summarize_samples([]) == {"p50": 0.0, "p95": 0.0, "mean": 0.0}
