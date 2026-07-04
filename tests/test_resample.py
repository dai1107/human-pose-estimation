from __future__ import annotations

import numpy as np
import pytest

from src.reference.resample import resample_sequence


def test_resample_sequence_uses_timestamps_and_interpolates_missing_values() -> None:
    matrix = np.array([[0.0], [np.nan], [10.0]], dtype=float)
    timestamps = np.array([0.0, 100.0, 200.0], dtype=float)
    resampled, axis = resample_sequence(matrix, timestamps, target_length=5)

    assert resampled.shape == (5, 1)
    assert axis.tolist() == pytest.approx([0.0, 25.0, 50.0, 75.0, 100.0])
    assert np.isfinite(resampled).all()
    assert resampled[0, 0] == pytest.approx(0.0)
    assert resampled[-1, 0] == pytest.approx(10.0)


def test_resample_sequence_handles_single_frame() -> None:
    resampled, _ = resample_sequence(np.array([[3.0, 4.0]]), [1000], target_length=4)
    assert resampled.shape == (4, 2)
    assert np.allclose(resampled, np.array([[3.0, 4.0]] * 4))

