from __future__ import annotations

from src.reference.session_loader import load_session
from src.sports.basketball.chain_features import build_shot_frames_from_session, detect_chain_events, extract_chain_feature_rows

from ._shot_helpers import make_shot_session


def test_chain_features_extract_events_with_quality_fields(tmp_path) -> None:
    session = load_session(make_shot_session(tmp_path))
    frames = build_shot_frames_from_session(session, "right")
    rows = extract_chain_feature_rows(frames)
    events = detect_chain_events(frames, rows, "right")
    assert rows[0]["shooting_elbow_angle"] is not None
    assert any(event.event == "right_wrist_speed_peak" for event in events)
    assert all(hasattr(event, "confidence") for event in events)

