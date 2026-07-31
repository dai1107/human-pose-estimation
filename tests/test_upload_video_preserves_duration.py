from __future__ import annotations

import pytest

from webui.upload_performance import video_duration_matches


def test_output_video_duration_accepts_one_frame_or_50_ms() -> None:
    matches, difference, tolerance = video_duration_matches(
        input_frame_count=300,
        input_fps=30.0,
        output_frame_count=299,
        output_fps=30.0,
    )
    assert matches is True
    assert difference == pytest.approx(1.0 / 30.0)
    assert tolerance == pytest.approx(0.05)

    matches, difference, tolerance = video_duration_matches(
        input_frame_count=300,
        input_fps=30.0,
        output_frame_count=298,
        output_fps=30.0,
    )
    assert matches is False
    assert difference > tolerance
