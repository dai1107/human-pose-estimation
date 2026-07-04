from __future__ import annotations

from src.sports.basketball.shot_clipper import clip_shot_session, detect_shot_candidates

from ._shot_helpers import make_shot_session


def test_manual_shot_clip_uses_requested_range(tmp_path) -> None:
    session_dir = make_shot_session(tmp_path)
    clip = clip_shot_session(session_dir, "right", start_ms=200, end_ms=700)
    assert clip.start_ms == 200
    assert clip.end_ms == 700
    assert len(clip.frames) == 6


def test_auto_candidates_are_suggestions(tmp_path) -> None:
    session_dir = make_shot_session(tmp_path)
    candidates = detect_shot_candidates(session_dir, "right", min_duration_ms=300)
    assert candidates
    assert "candidate" in candidates[0].reason.lower()

