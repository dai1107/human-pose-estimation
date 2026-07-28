from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.dataset.human_review_application import (
    ALL_AUTHORIZED_USES,
    apply_quick_review_bundle,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _review_record(*, end_frame: int = 9) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "human_review_record",
        "protocol_version": "human_review_v1.0",
        "reviewer_role": "reviewer_a",
        "reviewer_id": "reviewer",
        "reviewer_type": "human",
        "record_id": "phone_rowing_001",
        "revision": 1,
        "review": {
            "quick_review": {
                "status": "complete",
                "action": "rowing",
                "authorization": "confirmed",
                "target_status": "correct",
                "video_usability": "usable",
                "usable_start_frame": 0,
                "usable_end_frame": end_frame,
                "overall_result": "VALID",
                "observability": "OBSERVABLE",
                "equipment_visibility": "visible",
                "proposal_decision": "unresolved",
                "notes": "完整动作，但没有逐次标注",
                "segments": [],
                "reps": [],
                "phase_error_intervals": [],
                "events": [],
            }
        },
        "audit_log": [{"revision": 1}],
    }


def _dataset_and_bundle(tmp_path: Path, *, end_frame: int = 9) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    bundle = tmp_path / "bundle"
    record = {
        "record_id": "phone_rowing_001",
        "source_filename": "rowing.mp4",
        "action": "rowing",
        "subject_id": "subject_pending",
        "video": {"decoded_frame_count": 10},
        "usage_authorization": {
            "status": "pending_confirmation",
            "authorized_uses": [],
        },
    }
    _write(dataset / "manifests" / "phone_records.json", {"records": [record]})
    _write(
        dataset / "manifests" / "oni_records.json",
        {
            "records": [
                {
                    "record_id": "oni_rowing_001",
                    "usage_authorization": {
                        "status": "pending_confirmation",
                        "authorized_uses": [],
                    },
                }
            ],
            "summary": {"authorization_pending_count": 1},
        },
    )
    _write(
        dataset / "manifests" / "data_roles_v1.json",
        {
            "assignments": [
                {
                    "record_id": "phone_rowing_001",
                    "authorization_status": "pending_confirmation",
                    "training_eligible": False,
                }
            ]
        },
    )
    _write(
        bundle / "human_v1" / "protocol_frozen.json",
        {"protocol_version": "human_review_v1.0"},
    )
    _write(
        bundle / "human_v1" / "reviewer_a" / "codex_handoff.json",
        {"saved_record_count": 1},
    )
    _write(
        bundle
        / "human_v1"
        / "reviewer_a"
        / "records"
        / "phone_rowing_001.json",
        _review_record(end_frame=end_frame),
    )
    return dataset, bundle


def test_apply_quick_review_preserves_truth_gate_and_confirms_all_uses(
    tmp_path: Path,
) -> None:
    dataset, bundle = _dataset_and_bundle(tmp_path)
    outputs = apply_quick_review_bundle(dataset, bundle)
    overlay = json.loads(outputs["review_overlay"].read_text(encoding="utf-8"))
    assert overlay["record_count"] == 1
    assert overlay["human_confirmed_interval_record_count"] == 1
    assert overlay["fine_annotation_complete_record_count"] == 0
    assert overlay["promotion_policy"]["reviewed_ground_truth"] is False
    assert overlay["promotion_policy"]["training_eligible"] is False
    assert overlay["subject_identity_policy"]["manifest_subject_ids_modified"] is False
    applied = overlay["records"][0]
    assert applied["subject_group"] == "subject_group_rowing_01"
    assert applied["dataset_role"] == "development"

    phone = json.loads(outputs["phone_manifest"].read_text(encoding="utf-8"))
    authorization = phone["records"][0]["usage_authorization"]
    assert authorization["status"] == "confirmed"
    assert authorization["authorized_uses"] == list(ALL_AUTHORIZED_USES)
    assert phone["records"][0]["subject_id"] == "subject_pending"
    assert (
        phone["records"][0]["review_status"]["action_expert"]
        == "single_human_review_complete"
    )
    assert (
        phone["records"][0]["review_status"]["data_role"]
        == "temporary_development_assigned"
    )
    roles = json.loads(outputs["data_roles"].read_text(encoding="utf-8"))
    assert roles["checks"]["no_training_evaluation_overlap"] is True
    assert roles["checks"]["no_subject_role_conflicts"] is True


def test_apply_quick_review_rejects_out_of_bounds_closed_interval(
    tmp_path: Path,
) -> None:
    dataset, bundle = _dataset_and_bundle(tmp_path, end_frame=10)
    with pytest.raises(ValueError, match="outside"):
        apply_quick_review_bundle(dataset, bundle)


def test_apply_fine_rgb_annotations_validates_and_audits_observable_override(
    tmp_path: Path,
) -> None:
    dataset, bundle = _dataset_and_bundle(tmp_path)
    path = (
        bundle
        / "human_v1"
        / "reviewer_a"
        / "records"
        / "phone_rowing_001.json"
    )
    review = json.loads(path.read_text(encoding="utf-8"))
    quick = review["review"]["quick_review"]
    quick["reps"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "start_frame": 0,
            "end_frame": 9,
            "validity": "VALID",
        }
    ]
    quick["phase_error_intervals"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "start_frame": 0,
            "end_frame": 9,
            "phase": "stroke",
            "error_code": "NO_ERROR",
            "observability": "UNKNOWN",
        }
    ]
    quick["events"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "event_type": "finish",
            "frame_index": 9,
            "observability": "UNKNOWN",
        }
    ]
    _write(path, review)

    outputs = apply_quick_review_bundle(
        dataset,
        bundle,
        authorize_oni=False,
        accept_fine_annotations_for_internal_rgb_calibration=True,
        force_phone_observable=True,
    )
    fine = json.loads(
        outputs["fine_annotation_overlay"].read_text(encoding="utf-8")
    )
    assert fine["oni_records_included"] is False
    assert fine["record_count"] == 1
    assert fine["internal_rgb_rule_calibration_eligible_record_count"] == 1
    record = fine["records"][0]
    assert record["reviewed_ground_truth"] is True
    assert record["supervised_model_training_eligible"] is False
    assert record["events"][0]["observability"] == "OBSERVABLE"
    assert (
        record["events"][0]["observability_override"]["previous_value"]
        == "UNKNOWN"
    )


def test_apply_fine_rgb_annotations_rejects_phase_gaps(tmp_path: Path) -> None:
    dataset, bundle = _dataset_and_bundle(tmp_path)
    path = (
        bundle
        / "human_v1"
        / "reviewer_a"
        / "records"
        / "phone_rowing_001.json"
    )
    review = json.loads(path.read_text(encoding="utf-8"))
    quick = review["review"]["quick_review"]
    quick["reps"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "start_frame": 0,
            "end_frame": 9,
            "validity": "VALID",
        }
    ]
    quick["phase_error_intervals"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "start_frame": 1,
            "end_frame": 9,
            "phase": "stroke",
            "error_code": "NO_ERROR",
        }
    ]
    quick["events"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "event_type": "finish",
            "frame_index": 9,
        }
    ]
    _write(path, review)
    with pytest.raises(ValueError, match="unexplained gap"):
        apply_quick_review_bundle(dataset, bundle)


def test_confirmed_fine_rgb_review_promotes_internal_training_labels(
    tmp_path: Path,
) -> None:
    dataset, bundle = _dataset_and_bundle(tmp_path)
    path = (
        bundle
        / "human_v1"
        / "reviewer_a"
        / "records"
        / "phone_rowing_001.json"
    )
    review = json.loads(path.read_text(encoding="utf-8"))
    quick = review["review"]["quick_review"]
    quick["reps"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "start_frame": 0,
            "end_frame": 9,
            "validity": "VALID",
            "notes": "AI 候选，待人工核对",
        }
    ]
    quick["phase_error_intervals"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "start_frame": 0,
            "end_frame": 9,
            "phase": "stroke",
            "error_code": "NO_ERROR",
            "notes": "AI 候选，待人工核对",
        }
    ]
    quick["events"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "event_type": "finish",
            "frame_index": 9,
            "notes": "AI 候选，待人工核对",
        }
    ]
    _write(path, review)

    outputs = apply_quick_review_bundle(
        dataset,
        bundle,
        authorize_oni=False,
        accept_fine_annotations_for_internal_rgb_calibration=True,
        confirm_fine_rgb_human_reviewed=True,
    )
    fine = json.loads(
        outputs["fine_annotation_overlay"].read_text(encoding="utf-8")
    )

    assert fine["fine_rgb_human_review"]["status"] == "complete"
    assert (
        fine["supervised_model_training_eligible_record_count"]
        == 1
    )
    record = fine["records"][0]
    assert record["supervised_model_training_eligible"] is True
    assert record["fine_rgb_human_review_status"] == "complete"
    assert record["reps"][0]["notes"] == "AI 候选，已人工核对"
    assert record["reps"][0]["human_review_status"] == "complete"


def test_missing_noncore_events_are_derived_from_reviewed_phases(
    tmp_path: Path,
) -> None:
    dataset, bundle = _dataset_and_bundle(tmp_path)
    path = (
        bundle
        / "human_v1"
        / "reviewer_a"
        / "records"
        / "phone_rowing_001.json"
    )
    review = json.loads(path.read_text(encoding="utf-8"))
    quick = review["review"]["quick_review"]
    quick["reps"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "start_frame": 0,
            "end_frame": 9,
            "validity": "VALID",
        }
    ]
    quick["phase_error_intervals"] = [
        {
            "rep_id": "phone_rowing_001_rep_001",
            "start_frame": 0,
            "end_frame": 9,
            "phase": "catch",
            "error_code": "NO_ERROR",
        }
    ]
    _write(path, review)

    outputs = apply_quick_review_bundle(
        dataset,
        bundle,
        authorize_oni=False,
        accept_fine_annotations_for_internal_rgb_calibration=True,
        derive_missing_noncore_events=True,
    )
    fine = json.loads(
        outputs["fine_annotation_overlay"].read_text(encoding="utf-8")
    )
    event = fine["records"][0]["events"][0]
    assert event["event_type"] == "catch_reached"
    assert event["frame_index"] == 0
    assert (
        event["annotation_origin"]
        == "deterministic_human_reviewed_phase_boundary_v1"
    )
    resolution = fine["records"][0]["fine_annotation_resolution"][0]
    assert resolution["kind"] == (
        "deterministic_events_from_human_reviewed_phases"
    )


def test_blocked_unusable_review_is_a_terminal_review_decision(
    tmp_path: Path,
) -> None:
    dataset, bundle = _dataset_and_bundle(tmp_path)
    path = (
        bundle
        / "human_v1"
        / "reviewer_a"
        / "records"
        / "phone_rowing_001.json"
    )
    review = json.loads(path.read_text(encoding="utf-8"))
    quick = review["review"]["quick_review"]
    quick["status"] = "blocked"
    quick["fine_annotation_status"] = "blocked"
    quick["video_usability"] = "unusable"
    _write(path, review)

    outputs = apply_quick_review_bundle(dataset, bundle, authorize_oni=False)
    overlay = json.loads(
        outputs["review_overlay"].read_text(encoding="utf-8")
    )
    assert overlay["terminal_review_decision_record_count"] == 1
    assert overlay["fine_annotation_complete_record_count"] == 0
