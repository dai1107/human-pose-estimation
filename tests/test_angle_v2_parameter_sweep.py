from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import tools.sweep_angle_v2_parameters as sweep
from tools.sweep_angle_v2_parameters import SweepCandidate


def test_bounded_sweep_has_expected_small_candidate_count() -> None:
    candidates = sweep.bounded_candidates()

    assert len(candidates) == 26
    assert candidates[0].name == "baseline_shadow"
    assert len({candidate.name for candidate in candidates}) == 26


def test_loss_uses_documented_safety_weights() -> None:
    rows = [
        {
            "expected_angle_status": sweep.PASS,
            "predicted_angle_status": sweep.FAIL,
            "event_frame_error": None,
        },
        {
            "expected_angle_status": sweep.FAIL,
            "predicted_angle_status": sweep.PASS,
            "event_frame_error": None,
        },
        {
            "expected_angle_status": sweep.PASS,
            "predicted_angle_status": sweep.UNSURE,
            "event_frame_error": None,
        },
        {
            "expected_angle_status": sweep.PASS,
            "predicted_angle_status": sweep.PASS,
            "event_frame_error": -4,
        },
    ]

    metrics = sweep._loss_metrics(rows)

    assert metrics["false_no_rep_count"] == 1
    assert metrics["false_valid_count"] == 1
    assert metrics["excess_unsure_count"] == 1
    assert metrics["mean_absolute_event_frame_error"] == pytest.approx(4.0)
    assert metrics["weighted_loss"] == pytest.approx(8.0)


def test_clear_training_improvement_never_replaces_without_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_evaluate(
        candidate: SweepCandidate,
        reviews: Sequence[Mapping[str, Any]],
        lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        loss = 10.0 if candidate.name == "baseline_shadow" else 5.0
        return (
            {
                "candidate": candidate.as_dict(),
                "weighted_loss": loss,
                "false_no_rep_count": 1,
                "false_valid_count": 1,
                "excess_unsure_count": 0,
                "by_record": {"train_1": {"weighted_loss": loss}},
            },
            [],
        )

    monkeypatch.setattr(sweep, "evaluate_candidate", fake_evaluate)
    candidates = [
        SweepCandidate("baseline_shadow"),
        SweepCandidate("better", threshold_offset_deg=2.0),
    ]

    summary, _rows = sweep.run_sweep(
        [{"record_id": "train_1", "dataset_role": "calibration", "reps": []}],
        {},
        candidates=candidates,
    )

    assert summary["clear_shadow_improvement"] is True
    assert summary["independent_holdout_available"] is False
    assert summary["default_replacement_allowed"] is False
    assert summary["default_replaced"] is False

