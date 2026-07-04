from __future__ import annotations

from src.reference.aggregate import build_reference_template
from src.reference.features import load_feature_config
from src.reference.quality import evaluate_quality
from src.reference.session_loader import load_session

from ._reference_helpers import make_session


def test_low_quality_session_reports_warning(tmp_path) -> None:
    session_dir = make_session(tmp_path, frame_count=10, valid_ratio=0.4)
    session = load_session(session_dir)
    report = evaluate_quality(session.kinematics, session.landmarks, session.metadata)
    assert report.status == "WARNING"
    assert report.warnings


def test_multiple_reference_clips_build_mean_and_std_template(tmp_path) -> None:
    feature_config = load_feature_config()
    session_a = load_session(make_session(tmp_path, session_id="a", offset=0.0))
    session_b = load_session(make_session(tmp_path, session_id="b", offset=2.0))

    template = build_reference_template([session_a.kinematics, session_b.kinematics], feature_config=feature_config, target_length=20)

    assert template["clip_count"] == 2
    assert template["template_stability_status"] == "limited"
    assert len(template["mean_trajectory"]) == 20
    assert len(template["std_trajectory"]) == 20
    assert "right_wrist_speed" in template["peak_events"]

