from __future__ import annotations

import json
from pathlib import Path

from tools.evaluate_unreviewed_oni_auxiliary import build_experiment_report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_unreviewed_oni_experiment_expands_only_research_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "hyrox"
    phone_metrics = {
        "human_rep_count": 2,
        "predicted_candidate_count": 1,
        "exact_count_record_count": 0,
        "exact_count_and_status_record_count": 0,
        "matched_rep_status_count": 0,
        "detected_supported_error_record_count": 1,
        "supported_expected_error_record_count": 1,
    }
    _write_json(
        root
        / "reports"
        / "reviewed_phone_rgb_guidance_evaluation_optimized_v1.json",
        {"oni_used": False, **phone_metrics},
    )
    _write_json(
        root / "reviews" / "human_rgb_fine_annotations_v1.json",
        {
            "oni_records_included": False,
            "records": [
                {
                    "action": "lunge",
                    "phase_error_intervals": [
                        {"error_code": "PHONE_ERROR"}
                    ],
                }
            ],
        },
    )
    manifest_record = {
        "record_id": "oni_test_001",
        "action": "wall_ball",
        "recording_intent_code": "weak_error",
        "recording_intent_verified": False,
        "expected_errors_unverified": ["ONI_ONLY_ERROR"],
        "paired_group_id": None,
    }
    _write_json(
        root / "manifests" / "oni_records.json",
        {"records": [manifest_record]},
    )
    audit_record = {
        "record_id": "oni_test_001",
        "modalities": {
            "depth": {"human_review_complete": False},
            "ir": {"human_review_complete": False},
        },
    }
    _write_json(
        root / "reports" / "oni_subject_audit_v1.json",
        {
            "release_or_training_eligible_record_count": 0,
            "accepted_automatic_candidate_count": 3,
            "human_reviewed_modality_count": 0,
            "human_confirmed_target_count": 0,
            "records": [audit_record],
        },
    )
    _write_json(
        root / "reports" / "oni_modality_observability_v1.json",
        {"conclusion": "review required"},
    )
    _write_json(
        root / "extracted" / "oni_test_001" / "metadata.json",
        {
            "streams": {
                "color": {"exists": False},
                "depth": {"exists": True},
                "ir": {"exists": True},
            }
        },
    )
    common = {
        "target_track_id": "candidate_001",
        "human_confirmed": False,
        "confidence": 0.5,
        "bbox_normalized": [0.1, 0.2, 0.3, 0.7],
    }
    _write_jsonl(
        root
        / "oni_tracks"
        / "oni_test_001"
        / "depth_target_proposals.jsonl",
        [
            {
                **common,
                "modality": "depth",
                "source_frame_index": 1,
                "target_lock_status": "automated_candidate",
            },
            {
                "modality": "depth",
                "source_frame_index": 2,
                "target_lock_status": "no_candidate",
                "human_confirmed": False,
            },
        ],
    )
    _write_jsonl(
        root / "oni_tracks" / "oni_test_001" / "ir_target_proposals.jsonl",
        [
            {
                **common,
                "modality": "ir",
                "source_frame_index": 1,
                "target_lock_status": "automated_candidate",
            },
            {
                **common,
                "modality": "ir",
                "source_frame_index": 2,
                "target_lock_status": "automated_candidate",
            },
        ],
    )

    report_path = build_experiment_report(root)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["safety_gates"] == {
        "unreviewed": True,
        "training_eligible": False,
        "release_eligible": False,
        "runtime_eligible": False,
        "runtime_defaults_changed": False,
        "ir_treated_as_rgb": False,
        "phone_oni_pairing_created": False,
        "automatic_subject_proposals_promoted_to_truth": False,
        "recording_intent_promoted_to_truth": False,
    }
    assert report["oni_summary"]["automatic_candidate_count"] == 3
    assert (
        report["oni_summary"]["cross_modality_candidate_presence_agreement"][
            "p50"
        ]
        == 0.5
    )
    assert all(
        values["delta"] == 0
        for values in report["comparison"][
            "measured_phone_guidance_metrics"
        ].values()
    )
    assert report["comparison"]["research_scenario_coverage"] == {
        "action_count": {
            "reviewed_rgb_only": 1,
            "rgb_plus_unreviewed_oni": 2,
            "delta": 1,
        },
        "distinct_error_code_count": {
            "reviewed_rgb_only": 1,
            "rgb_plus_unreviewed_oni": 2,
            "delta": 1,
            "oni_only_codes_unverified": ["ONI_ONLY_ERROR"],
        },
    }
    assert report["verdict"]["measured_guidance_accuracy_improved"] is False
    assert report["verdict"]["research_coverage_improved"] is True
