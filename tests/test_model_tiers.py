from __future__ import annotations

from copy import deepcopy

from src.validation.model_tiers import build_model_tier_report, model_tier_gate


def _record() -> dict[str, object]:
    observation = {
        "candidate_count": 2,
        "pose_valid_rep_count": 1,
        "no_rep_count": 1,
        "unsure_count": 0,
        "cycle_count": 2,
        "rep_count": 1,
        "pose_detected_rate": 0.98,
    }
    return {
        "id": "case",
        "full": {
            "observation": deepcopy(observation),
            "golden_failures": [],
            "step_event_count": 4,
            "three_d_available_rate": 0.95,
        },
        "lite": {
            "observation": deepcopy(observation),
            "golden_failures": [],
            "step_event_count": 4,
            "three_d_available_rate": 0.93,
        },
        "differences": {
            "image_landmark_mean": 0.01,
            "joint_angle_mean_deg": 3.0,
        },
    }


def test_model_tier_gate_approves_equivalent_lite_results() -> None:
    record = _record()

    assert model_tier_gate(record) == []
    report = build_model_tier_report([record])
    assert report["status"] == "passed"
    assert report["lite_auto_approved"] is True


def test_model_tier_gate_rejects_rule_and_accuracy_regressions() -> None:
    record = _record()
    record["lite"]["observation"]["no_rep_count"] = 0
    record["differences"]["joint_angle_mean_deg"] = 14.0

    failures = model_tier_gate(record)

    assert any("no_rep_count" in failure for failure in failures)
    assert any("关节角" in failure for failure in failures)
