from __future__ import annotations

import numpy as np
import pytest

from src.reference.dtw import constrained_dtw


def test_identical_sequences_have_zero_dtw_distance() -> None:
    sequence = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    result = constrained_dtw(sequence, sequence, feature_names=["a", "b"])
    assert result.total_distance == pytest.approx(0.0)
    assert result.normalized_distance == pytest.approx(0.0)


def test_dtw_handles_tempo_difference_better_than_unaligned_distance() -> None:
    reference = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    candidate = np.array([[0.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    result = constrained_dtw(reference, candidate, window_ratio=1.0, feature_names=["value"])
    unaligned = float(np.mean(np.abs(reference[:, 0] - candidate[: reference.shape[0], 0])))
    assert result.normalized_distance < unaligned


def test_dtw_missing_values_do_not_crash() -> None:
    reference = np.array([[0.0], [np.nan], [2.0]])
    candidate = np.array([[0.0], [1.0], [2.0]])
    result = constrained_dtw(reference, candidate, feature_names=["value"])
    assert result.path
    assert "value" in result.per_feature_error

