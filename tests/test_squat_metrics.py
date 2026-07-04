from __future__ import annotations

import pytest

from src.fitness.squat.calibration import calibrate_standing
from src.fitness.squat.phase_metrics import compute_rep_metrics
from src.fitness.squat.rep_detector import detect_squat_reps
from src.fitness.squat.symmetry import knee_symmetry_proxy

from ._squat_helpers import make_squat_measurements


def test_equal_left_right_knee_angles_have_zero_symmetry_error() -> None:
    frames = make_squat_measurements()
    mean_error, peak_error = knee_symmetry_proxy(frames)
    assert mean_error == pytest.approx(0.0)
    assert peak_error == pytest.approx(0.0)


def test_rep_metrics_include_depth_timing_and_quality() -> None:
    frames = make_squat_measurements()
    detection = detect_squat_reps(frames, camera_view="side")
    metrics = compute_rep_metrics(detection.reps[0], detection.calibration)
    assert metrics.total_duration_ms >= 600
    assert metrics.pelvis_vertical_displacement_normalized == pytest.approx(0.35)
    assert metrics.data_quality_level == "GOOD"

