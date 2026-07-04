from __future__ import annotations

from src.fitness.squat.rep_detector import detect_squat_reps
from src.fitness.squat.schema import load_squat_config

from ._squat_helpers import make_squat_measurements


def test_complete_squat_trajectory_produces_one_rep() -> None:
    result = detect_squat_reps(make_squat_measurements(), camera_view="side", config=load_squat_config())
    assert result.calibration.status == "PASS"
    assert len(result.reps) == 1
    assert result.reps[0].start_timestamp_ms < result.reps[0].bottom_timestamp_ms < result.reps[0].end_timestamp_ms


def test_missing_pose_pauses_or_resets_state_machine() -> None:
    frames = make_squat_measurements(missing_at={5, 6, 7, 8, 9, 10})
    result = detect_squat_reps(frames, camera_view="side", config=load_squat_config())
    assert len(result.reps) == 0
    assert any(row["paused"] == 1 or row["state"] == "PAUSED" for row in result.frame_states)

