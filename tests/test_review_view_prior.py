from __future__ import annotations

import json
from pathlib import Path

import pytest

from webui.review import (
    _audit_revision_continuous,
    _build_review_exports,
    _validate_fine_annotations,
    _validate_oni_view_prior,
)


def _view_prior_review() -> dict[str, object]:
    return {
        "review_mode": "view_prior",
        "status": "complete",
        "overall_target_status": "correct",
        "same_subject_throughout": "yes",
        "confirmed_view": "side",
        "action_usability": "usable",
        "usable_start_frame": 1,
        "usable_end_frame": 41,
        "full_body_visibility": "visible",
        "floor_visibility": "visible",
        "equipment_visibility": "partial",
        "identity_switch_intervals": [],
        "observability_items": [
            {
                "item_code": code,
                "status": "OBSERVABLE",
                "reason": "",
                "start_frame": 1,
                "end_frame": 41,
                "evidence_frames": [1, 21],
                "notes": "",
            }
            for code in (
                "squat_depth",
                "heel_rise",
                "trunk_lean",
                "ball_release",
                "target_hit",
                "left_right_symmetry",
            )
        ],
    }


def test_view_prior_requires_reason_and_forbids_positive_error_when_unobservable() -> None:
    review = _view_prior_review()
    item = review["observability_items"][0]  # type: ignore[index]
    item["status"] = "UNOBSERVABLE"
    item["asserted_error"] = True
    with pytest.raises(ValueError, match="不能保存肯定错误"):
        _validate_oni_view_prior(
            review,
            action="wall_ball",
            allowed_frames={1, 21, 41},
        )

    item["asserted_error"] = False
    with pytest.raises(ValueError, match="必须填写原因"):
        _validate_oni_view_prior(
            review,
            action="wall_ball",
            allowed_frames={1, 21, 41},
        )


def test_core_quality_gate_rejects_rep_overlap_and_event_outside_rep() -> None:
    review = {
        "quick_review": {
            "status": "complete",
            "overall_result": "VALID",
            "reps": [
                {"rep_id": "rep_1", "start_frame": 0, "end_frame": 10, "validity": "VALID"},
                {"rep_id": "rep_2", "start_frame": 10, "end_frame": 20, "validity": "VALID"},
            ],
            "phase_error_intervals": [
                {
                    "rep_id": "rep_1",
                    "start_frame": 0,
                    "end_frame": 10,
                    "phase": "descent",
                    "error_code": "NO_ERROR",
                }
            ],
            "events": [{"rep_id": "rep_1", "event_type": "bottom_reached", "frame_index": 11}],
        }
    }
    with pytest.raises(ValueError, match="rep"):
        _validate_fine_annotations(review, "lunge")


def test_independent_exports_keep_modalities_and_eligibility_separate(tmp_path: Path) -> None:
    review_root = tmp_path / "reviews" / "human_v1"
    record_dir = review_root / "reviewer_a" / "view_prior_records"
    record_dir.mkdir(parents=True)
    review = _view_prior_review()
    payload = {
        "revision": 1,
        "review": review,
        "eligibility": {
            "unreviewed": False,
            "training_eligible": False,
            "release_eligible": False,
            "view_policy_calibration_eligible": True,
        },
    }
    (record_dir / "oni_wall_ball_001__depth.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    exports = _build_review_exports(
        review_root,
        {
            "oni_wall_ball_001": {
                "action": "wall_ball",
                "camera_view": "oblique_front",
                "camera_view_raw": "斜前方",
                "recording_intent_code": "standard",
                "expected_errors_unverified": [],
            }
        },
    )
    view_records = exports["view_observability_review_v1.json"]["records"]
    depth = next(item for item in view_records if item["modality"] == "depth")
    ir = next(item for item in view_records if item["modality"] == "ir")
    assert depth["view_policy_calibration_eligible"] is True
    assert depth["training_eligible"] is False
    assert ir["unreviewed"] is True
    assert ir["view_policy_calibration_eligible"] is False
    assert exports["oni_subject_review_v1.json"]["artifact_type"] == "oni_subject_review_v1"
    assert exports["oni_error_truth_review_v1.json"]["artifact_type"] == "oni_error_truth_review_v1"


def test_audit_revision_continuity_is_explicitly_checked() -> None:
    assert _audit_revision_continuous(
        {"revision": 2, "audit_log": [{"revision": 1}, {"revision": 2}]}
    )
    assert not _audit_revision_continuous(
        {"revision": 2, "audit_log": [{"revision": 2}]}
    )


def test_review_page_exposes_view_prior_and_independent_modality_controls() -> None:
    html = Path("webui/templates/review.html").read_text(encoding="utf-8")
    source = Path("webui/static/review.js").read_text(encoding="utf-8")
    assert "视角先验复核" in html
    assert "动作规则可观察性矩阵" in html
    assert "录制意图错误真值" in html
    assert "任务完成度仪表盘" in html
    assert "oniDepthPreviewImage" in html
    assert "oniIrPreviewImage" in html
    assert "data-batch-target" in html
    assert "function collectObservabilityItems" in source
    assert "function collectErrorTruthItems" in source
    assert "single human ONI ${review.review_mode} initial review" in source
