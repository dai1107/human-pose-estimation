from __future__ import annotations

from src.sports.basketball.consistency import analyze_shot_consistency


def test_multiple_shots_generate_consistency_stats() -> None:
    shots = [
        {
            "clip_range": {"duration_ms": 900},
            "release_proxy": {"release_proxy_time": 600},
            "event_sequence": {"order_consistent_with_template": True, "events": {"right_wrist_speed_peak": {"timestamp_ms": 580}}},
        },
        {
            "clip_range": {"duration_ms": 940},
            "release_proxy": {"release_proxy_time": 640},
            "event_sequence": {"order_consistent_with_template": False, "events": {"right_wrist_speed_peak": {"timestamp_ms": 600}}},
        },
    ]
    result = analyze_shot_consistency(shots)
    assert result["shot_count"] == 2
    assert result["release_proxy_time_std_ms"] > 0
    assert result["sequence_order_consistency"]["consistent_count"] == 1

