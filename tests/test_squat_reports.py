from __future__ import annotations

import json

from src.fitness.squat.report import analyze_squat_session
from src.reference.library import create_reference_from_session

from ._squat_helpers import make_squat_session


def test_offline_squat_session_exports_report_files(tmp_path) -> None:
    session_dir = make_squat_session(tmp_path)
    report_dir = analyze_squat_session(session_dir, camera_view="side", output_dir=tmp_path / "squat_reports")

    expected = [
        "metadata.json",
        "squat_reps.csv",
        "squat_frames.csv",
        "squat_summary.json",
        "rep_timeline.png",
        "angle_curves_by_rep.png",
        "symmetry_curves.png",
        "report.md",
    ]
    for name in expected:
        assert (report_dir / name).exists()
    assert (report_dir / "annotated_keyframes" / "rep_001_bottom.png").exists()
    summary = json.loads((report_dir / "squat_summary.json").read_text(encoding="utf-8"))
    assert summary["complete_rep_count"] == 1


def test_squat_reference_comparison_generates_alignment_curve(tmp_path) -> None:
    session_dir = make_squat_session(tmp_path, "candidate")
    reference_dir = create_reference_from_session(
        session_dir,
        output_root=tmp_path / "references",
        start_frame=0,
        end_frame=11,
        name="squat reference",
        action_type="squat",
        camera_view="side",
        movement_side="bilateral",
    )
    report_dir = analyze_squat_session(
        session_dir,
        camera_view="side",
        output_dir=tmp_path / "squat_reports",
        reference_dir=reference_dir,
    )
    assert (report_dir / "squat_reference_comparison.csv").exists()
    assert (report_dir / "squat_reference_alignment.png").exists()
    summary = json.loads((report_dir / "squat_summary.json").read_text(encoding="utf-8"))
    assert summary["reference_comparison"]["status"] == "PASS"

