from __future__ import annotations

from src.fitness.squat.rep_detector import detect_squat_reps
from src.fitness.squat.schema import load_squat_config

from ._squat_helpers import make_squat_measurements


def test_bottom_jitter_does_not_double_count() -> None:
    frames = make_squat_measurements()
    jitter = frames[:9] + [
        frames[8],
        frames[8],
    ] + frames[9:]
    result = detect_squat_reps(jitter, camera_view="side", config=load_squat_config())
    assert len(result.reps) == 1

