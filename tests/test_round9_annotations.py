from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.dataset.round9_annotations import (
    PROPOSAL_LAYERS,
    TIMELINE_LABELS,
    build_round9_artifacts,
    validate_round9_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "hyrox"
pytestmark = pytest.mark.skipif(
    not (DATASET_ROOT / "manifests" / "phone_records.json").exists(),
    reason="local HYROX dataset is not available",
)


def test_round9_builder_covers_30_records_and_15_core_records() -> None:
    outputs = build_round9_artifacts(DATASET_ROOT)
    validate_round9_artifacts(outputs)
    by_name = {path.name: payload for path, payload in outputs.items()}
    assert by_name["action_segments_v1.json"]["record_count"] == 30
    assert by_name["core_rep_phase_event_error_v1.json"]["record_count"] == 15
    assert len(by_name["object_scene_evidence_v1.json"]["records"]) == 30
    assert len(by_name["scoring_correction_v1.json"]["records"]) == 15


def test_round9_timelines_are_contiguous_and_not_training_truth() -> None:
    payload = json.loads(
        (DATASET_ROOT / "annotations" / "action_segments_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for record in payload["records"]:
        cursor = 0
        for segment in record["segments"]:
            assert segment["timeline_label"] in TIMELINE_LABELS
            assert segment["start_frame"] == cursor
            assert segment["human_confirmed"] is False
            cursor = segment["end_frame"] + 1
        assert cursor == record["frame_count"]
        assert record["training_eligible"] is False


def test_round9_never_promotes_filename_or_pose_proposals_to_truth() -> None:
    core = json.loads(
        (DATASET_ROOT / "annotations" / "core_rep_phase_event_error_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert core["proposal_is_ground_truth"] is False
    for record in core["records"]:
        for rep in record["reps"]:
            assert rep["validity"] == "UNSURE"
            assert rep["training_eligible"] is False
            assert all(not event["is_ground_truth"] for event in rep["events"])
            assert all(not error["is_ground_truth"] for error in rep["errors"])


def test_round9_reports_human_agreement_as_pending_not_fabricated() -> None:
    agreement = json.loads(
        (DATASET_ROOT / "reports" / "annotation_agreement_v1.json").read_text(
            encoding="utf-8"
        )
    )
    proposal = json.loads(
        (DATASET_ROOT / "reports" / "proposal_acceptance_bias_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert agreement["eligible_reviewer_count"] == 0
    assert agreement["event_anchor_agreement"] is None
    assert agreement["release_gate_passed"] is False
    assert proposal["proposal_layers"] == list(PROPOSAL_LAYERS)
    assert proposal["performance_evaluation_allowed"] is False


def test_round9_unobservable_object_evidence_is_never_a_pass() -> None:
    payload = json.loads(
        (DATASET_ROOT / "annotations" / "object_scene_evidence_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["unobservable_is_not_pass"] is True
    for record in payload["records"]:
        for evidence in record["evidence"]:
            assert evidence["observability"] in {
                "OBSERVABLE_AI_REVIEWED",
                "UNOBSERVABLE",
                "PARTIALLY_OBSERVABLE_NO_READING",
                "UNKNOWN",
            }
            if evidence["observability"] == "UNOBSERVABLE":
                assert evidence["object_visible"] == "unobservable"
            assert evidence["rule_truth_generated"] is False
