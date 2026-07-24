from __future__ import annotations

import json
from pathlib import Path

from src.validation.baseline import (
    GoldenTraceCollector,
    build_schema_snapshot,
    collect_environment,
    sha256_file,
    snapshot_configs,
)
from tools.freeze_baseline import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_trace_collector_preserves_phases_rules_and_three_d_evidence() -> None:
    collector = GoldenTraceCollector()
    collector.observe(
        1,
        33,
        {"phase": "stand", "candidate_count": 0, "cycle_count": 0},
    )
    collector.observe(
        2,
        66,
        {
            "phase": "bottom",
            "candidate_count": 1,
            "cycle_count": 1,
            "last_rep_candidate": {"start_frame": 1, "end_frame": 2},
            "last_rep_decision": {
                "status": "NO_REP",
                "rules": [{"rule_id": "depth", "status": "FAIL"}],
            },
            "last_three_d_assist": {"status": "SHADOW"},
        },
    )

    report = collector.report()

    assert [segment["phase"] for segment in report["phase_segments"]] == [
        "stand",
        "bottom",
    ]
    assert report["candidates"][0]["decision"]["status"] == "NO_REP"
    assert report["candidates"][0]["three_d_assist"]["status"] == "SHADOW"
    assert report["rule_status_totals"] == {"depth:FAIL": 1}
    assert {event["type"] for event in report["count_events"]} == {
        "candidate",
        "cycle",
    }
    assert report["dtw"]["status"] == "not_configured"


def test_config_snapshot_is_a_hash_verified_copy(tmp_path: Path) -> None:
    records = snapshot_configs(ROOT, tmp_path)

    assert records
    product = next(
        item for item in records if item["path"] == "configs/product_pose.yaml"
    )
    copied = tmp_path / product["snapshot_path"]
    assert copied.is_file()
    assert sha256_file(copied) == product["sha256"]


def test_environment_and_schema_snapshots_record_compatibility_contract() -> None:
    environment = collect_environment(ROOT)
    schema = build_schema_snapshot()

    assert environment["dependencies"]["mediapipe"]
    assert environment["isolation"]["oni_runtime_required"] is False
    assert environment["isolation"]["neural_prediction_default_enabled"] is False
    assert environment["git"]["tag_created"] is False
    assert schema["compatibility_contract"]["existing_fields_may_be_removed"] is False
    assert "rep_count" in schema["golden_observation_fields"]


def test_freeze_cli_defaults_to_all_baseline_artifacts() -> None:
    args = build_parser().parse_args([])

    assert args.output_dir == "reports/baseline"
    assert args.skip_golden is False
    assert args.skip_latency is False
