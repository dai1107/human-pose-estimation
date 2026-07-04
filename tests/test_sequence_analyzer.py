from __future__ import annotations

from src.sports.basketball.config import load_basketball_config
from src.sports.basketball.schema import ShotEvent
from src.sports.basketball.sequence_analyzer import analyze_event_sequence


def test_event_sequence_detects_order_missing_and_overlap() -> None:
    config = load_basketball_config()
    events = [
        ShotEvent("pelvis_upward_speed_peak", 1000, 10, 0.8, "pelvis"),
        ShotEvent("right_knee_extension_peak", 1040, 15, 0.8, "knee"),
        ShotEvent("right_wrist_speed_peak", 1030, 18, 0.8, "wrist"),
        ShotEvent("right_elbow_extension_peak", 1200, 30, 0.8, "elbow"),
    ]
    result = analyze_event_sequence(events, config, "right")
    assert "shooting_side_hip_extension_peak" in result["missing_events"]
    assert any(item["status"] in {"overlap", "early"} for item in result["pairwise_timing"])

