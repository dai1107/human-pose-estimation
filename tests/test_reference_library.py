from __future__ import annotations

import json

from src.reference.library import create_reference_from_session, list_references, load_reference

from ._reference_helpers import make_session


def test_create_reference_from_session_writes_library_files(tmp_path) -> None:
    session_dir = make_session(tmp_path)
    original_kinematics = (session_dir / "kinematics.csv").read_text(encoding="utf-8")
    reference_dir = create_reference_from_session(
        session_dir,
        output_root=tmp_path / "references",
        start_ms=1100,
        end_ms=1700,
        name="unit reference",
        action_type="squat",
        camera_view="side",
        movement_side="bilateral",
    )

    assert (reference_dir / "reference.json").exists()
    assert (reference_dir / "clip_kinematics.csv").exists()
    assert (reference_dir / "clip_landmarks.csv").exists()
    assert (reference_dir / "features.csv").exists()
    assert (session_dir / "kinematics.csv").read_text(encoding="utf-8") == original_kinematics

    reference = load_reference(reference_dir)
    assert reference.name == "unit reference"
    assert reference.action_type == "squat"
    assert reference.source_session_ids == ["session_a"]
    assert reference.quality_summary["status"] == "PASS"
    assert len(list_references(tmp_path / "references")) == 1

    payload = json.loads((reference_dir / "reference.json").read_text(encoding="utf-8"))
    assert payload["normalization_method"] == "body_relative"

