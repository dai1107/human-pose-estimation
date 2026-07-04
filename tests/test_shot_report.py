from __future__ import annotations

import json

from src.reference.library import create_reference_from_session
from src.reference.session_loader import write_csv_rows
from src.sports.basketball.chain_features import extract_chain_feature_rows
from src.sports.basketball.report import analyze_shot_session
from src.sports.basketball.shot_clipper import clip_shot_session

from ._shot_helpers import make_shot_session


def test_shot_report_exports_required_files_and_limitations(tmp_path) -> None:
    session_dir = make_shot_session(tmp_path)
    report_dir = analyze_shot_session(
        session_dir,
        shot_type="set_shot",
        shooting_side="right",
        camera_view="side",
        output_dir=tmp_path / "reports",
        start_ms=0,
        end_ms=900,
    )
    for name in [
        "metadata.json",
        "shot_summary.json",
        "shot_events.csv",
        "shot_features.csv",
        "chain_sequence.json",
        "phase_timeline.png",
        "angle_curves.png",
        "velocity_curves.png",
        "event_sequence.png",
        "arm_path.png",
        "report.md",
    ]:
        assert (report_dir / name).exists()
    assert (report_dir / "keyframes" / "release_proxy.png").exists()
    summary = json.loads((report_dir / "shot_summary.json").read_text(encoding="utf-8"))
    assert "release_proxy" in summary
    assert summary["arm_alignment"]["computed"] is False
    assert "真实关节力矩" in (report_dir / "report.md").read_text(encoding="utf-8")


def test_shot_reference_comparison_generates_alignment_curve(tmp_path) -> None:
    session_dir = make_shot_session(tmp_path)
    reference_dir = create_reference_from_session(
        session_dir,
        output_root=tmp_path / "references",
        start_ms=0,
        end_ms=900,
        name="shot reference",
        action_type="basketball_set_shot",
        camera_view="side",
        movement_side="right",
    )
    clip = clip_shot_session(session_dir, "right", start_ms=0, end_ms=900)
    write_csv_rows(reference_dir / "shot_features.csv", extract_chain_feature_rows(clip.frames))
    report_dir = analyze_shot_session(
        session_dir,
        shot_type="set_shot",
        shooting_side="right",
        camera_view="side",
        output_dir=tmp_path / "reports",
        start_ms=0,
        end_ms=900,
        reference_dir=reference_dir,
    )
    assert (report_dir / "reference_alignment.png").exists()
    summary = json.loads((report_dir / "shot_summary.json").read_text(encoding="utf-8"))
    assert summary["reference_comparison"]["status"] == "PASS"

