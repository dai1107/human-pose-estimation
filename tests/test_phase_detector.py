from __future__ import annotations

from src.reference.session_loader import load_session
from src.sports.basketball.chain_features import build_shot_frames_from_session, extract_chain_feature_rows
from src.sports.basketball.phase_detector import detect_shot_phases
from src.sports.basketball.release_proxy import estimate_release_proxy

from ._shot_helpers import make_shot_session


def test_phase_detector_identifies_main_shot_phases(tmp_path) -> None:
    session = load_session(make_shot_session(tmp_path))
    frames = build_shot_frames_from_session(session, "right")
    features = extract_chain_feature_rows(frames)
    release = estimate_release_proxy(frames, features)
    phases = detect_shot_phases(frames, release)
    seen = {row["phase"] for row in phases.phase_by_frame}
    assert {"DIP", "RISE", "ARM_EXTENSION", "FOLLOW_THROUGH"}.issubset(seen)

