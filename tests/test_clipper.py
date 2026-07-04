from __future__ import annotations

from src.reference.clipper import clip_session

from ._reference_helpers import make_session


def test_clip_session_by_frame_does_not_modify_source(tmp_path) -> None:
    session_dir = make_session(tmp_path, frame_count=10)
    before = (session_dir / "kinematics.csv").read_text(encoding="utf-8")
    clip = clip_session(session_dir, start_frame=2, end_frame=5)

    assert len(clip.kinematics) == 4
    assert clip.clip_range.start_frame == 2
    assert clip.clip_range.end_frame == 5
    assert clip.kinematics[0]["clip_timestamp_ms"] == "0"
    assert (session_dir / "kinematics.csv").read_text(encoding="utf-8") == before


def test_clip_session_rejects_empty_range(tmp_path) -> None:
    session_dir = make_session(tmp_path, frame_count=5)
    try:
        clip_session(session_dir, start_ms=99999, end_ms=100000)
    except ValueError as exc:
        assert "clip does not contain" in str(exc)
    else:
        raise AssertionError("expected ValueError")

