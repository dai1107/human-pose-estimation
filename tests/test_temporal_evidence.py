from __future__ import annotations

import numpy as np

from src.temporal_evidence import (
    RidgePhaseModel,
    candidate_metrics,
    decode_phase_candidates,
    estimate_standing_baseline,
    phase_feature_matrix,
    phase_metrics,
    rle_roi_foreground_ratio,
)
from tools.run_temporal_evidence_experiment import (
    _fit_rule_confidence_thresholds,
)


def _frame(knee: float, hip: float, hip_y: float) -> dict[str, float]:
    return {
        "left_knee_angle": knee,
        "right_knee_angle": knee,
        "left_hip_angle": hip,
        "right_hip_angle": hip,
        "hip_center_y": hip_y,
        "body_height_norm": 0.5,
        "visible_score": 0.95,
    }


def test_standing_baseline_and_relative_phase_features() -> None:
    frames = [
        _frame(170.0, 168.0, 0.40),
        _frame(171.0, 169.0, 0.40),
        _frame(110.0, 105.0, 0.48),
    ] * 10

    baseline = estimate_standing_baseline(frames)
    matrix = phase_feature_matrix(frames, baseline)

    assert baseline.reliable
    assert baseline.knee_angle >= 170.0
    assert matrix.shape == (30, 57)


def test_standing_baseline_rejects_implausible_front_view_angles() -> None:
    frames = [_frame(86.0, 122.0, 0.40) for _ in range(30)]

    baseline = estimate_standing_baseline(frames)

    assert not baseline.reliable
    assert "IMPLAUSIBLE_STANDING_KNEE_ANGLE" in baseline.rejection_reasons
    assert "IMPLAUSIBLE_STANDING_HIP_ANGLE" in baseline.rejection_reasons


def test_ridge_phase_model_hmm_preserves_ordered_sequence() -> None:
    matrix = np.asarray(
        [[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]],
        dtype=float,
    )
    labels = ["stand", "stand", "bottom", "bottom", "stand", "stand"]
    model = RidgePhaseModel.fit(
        matrix,
        labels,
        (labels,),
        classes=("stand", "bottom"),
    )

    predicted = model.predict_causal_hmm(matrix)

    assert len(predicted) == len(labels)
    assert set(predicted).issubset({"stand", "bottom"})


def test_phase_and_candidate_metrics_report_boundaries_and_misses() -> None:
    expected = ["stand", "stand", "bottom", "bottom", "stand"]
    predicted = ["stand", "stand", "stand", "bottom", "stand"]
    metrics = phase_metrics(expected, predicted, tolerance_frames=2)
    candidates, audit = decode_phase_candidates(
        ["stand", "stand", "descent", "bottom", "ascent", "stand", "stand"],
        ("stand", "descent", "bottom", "ascent", "stand"),
        minimum_run_frames=1,
    )
    cycle_metrics = candidate_metrics(((0, 6),), candidates)

    assert metrics["matched_boundary_count"] == 2
    assert candidates == [(0, 6)]
    assert audit["duplicate_settlement_count"] == 0
    assert cycle_metrics["candidate_recall"] == 1.0


def test_phase_candidate_decoder_can_calibrate_two_missing_phases() -> None:
    candidates, _audit = decode_phase_candidates(
        ["stand", "stand", "contact", "stand", "stand"],
        ("stand", "descent", "bottom", "contact", "ascent", "stand"),
        minimum_run_frames=1,
        maximum_phase_skips=2,
    )

    assert candidates == [(0, 4)]


def test_rule_confidence_threshold_requires_video_diversity_and_zero_fp() -> None:
    def record(record_id: str, labels: list[str]) -> dict[str, object]:
        return {
            "record_id": record_id,
            "matches": [
                {
                    "human_rep": {"validity": label},
                    "candidate": {
                        "rules": [
                            {
                                "rule_id": "extension",
                                "status": "FAIL",
                                "confidence": 0.95,
                            }
                        ]
                    },
                }
                for label in labels
            ],
        }

    supported = [
        record("video_1", ["NO_REP", "NO_REP"]),
        record("video_2", ["NO_REP", "NO_REP"]),
        record("video_3", ["NO_REP"]),
    ]
    contaminated = [*supported, record("video_4", ["VALID"])]

    assert _fit_rule_confidence_thresholds(supported) == {
        "extension": 0.95
    }
    assert _fit_rule_confidence_thresholds(contaminated) == {}


def test_rle_roi_foreground_ratio_uses_row_major_runs() -> None:
    # 3x4 mask:
    # 0000
    # 0110
    # 0000
    mask = {
        "size": [3, 4],
        "counts": [5, 2, 5],
    }

    ratio = rle_roi_foreground_ratio(
        mask,
        x0=1,
        y0=1,
        x1=3,
        y1=2,
    )

    assert ratio == 1.0
