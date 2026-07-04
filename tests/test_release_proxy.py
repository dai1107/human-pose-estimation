from __future__ import annotations

from src.reference.session_loader import load_session
from src.sports.basketball.chain_features import build_shot_frames_from_session, extract_chain_feature_rows
from src.sports.basketball.release_proxy import estimate_release_proxy

from ._shot_helpers import make_shot_session


def test_release_proxy_detects_wrist_peak_and_elbow_extension(tmp_path) -> None:
    session = load_session(make_shot_session(tmp_path))
    frames = build_shot_frames_from_session(session, "right")
    features = extract_chain_feature_rows(frames)
    release = estimate_release_proxy(frames, features)
    assert release.release_proxy_time is not None
    assert release.release_proxy_confidence > 0.35
    assert "wrist" in release.release_proxy_reason


def test_manual_release_ms_overrides_auto_proxy(tmp_path) -> None:
    session = load_session(make_shot_session(tmp_path))
    frames = build_shot_frames_from_session(session, "right")
    features = extract_chain_feature_rows(frames)
    release = estimate_release_proxy(frames, features, manual_release_ms=515)
    assert release.release_source == "manual"
    assert release.release_proxy_time == 515
    assert release.automatic_time is not None


def test_low_visibility_reduces_release_confidence(tmp_path) -> None:
    good = load_session(make_shot_session(tmp_path, "good", low_visibility=False))
    bad = load_session(make_shot_session(tmp_path, "bad", low_visibility=True))
    good_frames = build_shot_frames_from_session(good, "right")
    bad_frames = build_shot_frames_from_session(bad, "right")
    good_release = estimate_release_proxy(good_frames, extract_chain_feature_rows(good_frames))
    bad_release = estimate_release_proxy(bad_frames, extract_chain_feature_rows(bad_frames))
    assert bad_release.release_proxy_confidence < good_release.release_proxy_confidence

