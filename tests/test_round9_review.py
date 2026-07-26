from __future__ import annotations

from pathlib import Path

import pytest

from tools.dataset.round9_review import build_multimethod_review


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "hyrox"
pytestmark = pytest.mark.skipif(
    not (DATASET_ROOT / "annotations" / "action_segments_v1.json").exists(),
    reason="local Round 9 artifacts are not available",
)


def test_round9_multimethod_review_covers_all_records_without_rendering() -> None:
    report = build_multimethod_review(ROOT, DATASET_ROOT, write_sheets=False)
    assert report["record_count"] == 30
    assert report["core_record_count"] == 15
    assert len(report["records"]) == 30
    assert all(row["offline_active_bounds"][0] <= row["offline_active_bounds"][1] for row in report["records"])
    core = [row for row in report["records"] if row["causal_rep_anchor_count"]]
    assert len(core) == 15
    assert all(row["offline_rep_anchor_count"] > 0 for row in core)


def test_round9_ai_review_decisions_are_explicitly_not_human() -> None:
    import json

    decisions = json.loads(
        (
            ROOT / "configs" / "annotation" / "round9_ai_review_decisions_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert decisions["human_reviewer"] is False
    assert len(decisions["records"]) == 30
    assert sum("rep_anchors" in row for row in decisions["records"]) == 15
    assert all(row["action_confirmed"] is True for row in decisions["records"])


def test_applied_round9_ai_review_keeps_release_gate_closed() -> None:
    import json

    summary = json.loads(
        (DATASET_ROOT / "reports" / "round9_implementation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    action = json.loads(
        (DATASET_ROOT / "annotations" / "action_segments_v1.json").read_text(
            encoding="utf-8"
        )
    )
    core = json.loads(
        (
            DATASET_ROOT
            / "annotations"
            / "core_rep_phase_event_error_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["status"] == "round9_ai_assisted_pilot_complete_human_release_gate_pending"
    assert summary["ai_multimethod_reviewed_record_count"] == 30
    assert summary["independent_human_reviewer_count"] == 0
    assert summary["release_gate_passed"] is False
    assert action["human_confirmed_record_count"] == 0
    assert core["rep_proposal_count"] == 70
    assert all(not rep["training_eligible"] for row in core["records"] for rep in row["reps"])
