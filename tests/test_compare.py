from __future__ import annotations

import json

from src.reference.canonicalize import canonicalize_feature_rows
from src.reference.compare import compare_reference_to_session
from src.reference.library import create_reference_from_session

from ._reference_helpers import make_session


def test_mirror_canonicalization_swaps_left_and_right_features() -> None:
    rows = [{"left_elbow_angle": "10", "right_elbow_angle": "20", "pelvis_speed": "1"}]
    result = canonicalize_feature_rows(rows, movement_side="left", canonical_side="right")
    assert result.applied
    assert result.rows[0]["left_elbow_angle"] == "20"
    assert result.rows[0]["right_elbow_angle"] == "10"
    assert result.rows[0]["pelvis_speed"] == "1"


def test_compare_session_writes_json_csv_png_and_markdown(tmp_path) -> None:
    reference_session = make_session(tmp_path, session_id="reference_session", offset=0.0)
    candidate_session = make_session(tmp_path, session_id="candidate_session", offset=1.0, time_step_ms=60)
    reference_dir = create_reference_from_session(
        reference_session,
        output_root=tmp_path / "references",
        start_frame=2,
        end_frame=18,
        name="reference",
        action_type="generic_motion",
        camera_view="side",
        movement_side="right",
        mirror_canonicalization_enabled=True,
    )

    comparison_dir = compare_reference_to_session(
        candidate_session,
        reference_dir,
        output_dir=tmp_path / "comparisons",
        start_frame=2,
        end_frame=18,
        canonical_side="right",
        candidate_movement_side="right",
    )

    expected_files = [
        "metadata.json",
        "comparison_summary.json",
        "aligned_features.csv",
        "feature_errors.csv",
        "dtw_path.csv",
        "angle_comparison.png",
        "velocity_comparison.png",
        "phase_difference.png",
        "report.md",
    ]
    for name in expected_files:
        assert (comparison_dir / name).exists()
    summary = json.loads((comparison_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["reference_id"] == reference_dir.name
    assert summary["candidate_session_id"] == "candidate_session"
    assert summary["features_used"] > 0

