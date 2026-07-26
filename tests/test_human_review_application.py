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

    phone = json.loads(outputs["phone_manifest"].read_text(encoding="utf-8"))
    authorization = phone["records"][0]["usage_authorization"]
    assert authorization["status"] == "confirmed"
    assert authorization["authorized_uses"] == list(ALL_AUTHORIZED_USES)
    assert phone["records"][0]["subject_id"] == "subject_pending"


def test_apply_quick_review_rejects_out_of_bounds_closed_interval(
    tmp_path: Path,
) -> None:
    dataset, bundle = _dataset_and_bundle(tmp_path, end_frame=10)
    with pytest.raises(ValueError, match="outside"):
        apply_quick_review_bundle(dataset, bundle)
