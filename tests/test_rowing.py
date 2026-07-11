from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.actions import RowingAnalyzer
from hyrox.config import DEFAULT_ROWING_CONFIG, load_rowing_config, resolve_hyrox_config_path
from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.registry import create_action_analyzer
from main import parse_args
from tools.replay_hyrox_video import build_parser, create_analyzer


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "left_knee_angle": 90.0,
        "right_knee_angle": 92.0,
        "left_hip_angle": 95.0,
        "right_hip_angle": 98.0,
        "left_elbow_angle": 165.0,
        "right_elbow_angle": 167.0,
        "torso_angle": 20.0,
    }
    values.update(overrides)
    return values


def _analyzer() -> RowingAnalyzer:
    return RowingAnalyzer.from_config({**DEFAULT_ROWING_CONFIG, "stable_frames": 1})


def test_rowing_config_registry_and_cli(tmp_path: Path) -> None:
    assert load_rowing_config(tmp_path / "missing.yaml") == DEFAULT_ROWING_CONFIG
    assert resolve_hyrox_config_path("rowing") == "configs/hyrox/rowing.yaml"
    assert "rowing" in HYROX_ACTION_NAMES
    assert isinstance(create_action_analyzer("rowing"), RowingAnalyzer)
    assert parse_args(["--hyrox-action", "rowing"]).hyrox_action == "rowing"
    assert build_parser().parse_args(["--video", "row.mp4", "--hyrox-action", "rowing"]).hyrox_action == "rowing"
    assert isinstance(create_analyzer("rowing", "medium"), RowingAnalyzer)


def test_rowing_custom_config_uses_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "side_view.yaml"
    path.write_text("visibility_min: 0.7\nstable_frames: 2\n", encoding="utf-8")
    analyzer = RowingAnalyzer.from_config_path(str(path))
    assert analyzer.config_name == "side_view"
    assert analyzer.min_visible_score == pytest.approx(0.7)
    assert analyzer.confirmation_frames == 2


def test_rowing_counts_complete_stroke_sequence() -> None:
    analyzer = _analyzer()
    assert analyzer.update(_features(), 0)["phase"] == "catch"
    assert analyzer.update(_features(left_knee_angle=125, right_knee_angle=127), 150)["phase"] == "drive"
    assert analyzer.update(_features(left_knee_angle=155, right_knee_angle=157, left_elbow_angle=105, right_elbow_angle=107, torso_angle=-15), 300)["phase"] == "finish"
    assert analyzer.update(_features(left_knee_angle=130, right_knee_angle=132), 450)["phase"] == "recovery"
    completed = analyzer.update(_features(), 600)
    assert completed["phase"] == "catch"
    assert completed["rep_count"] == 1
    assert completed["debug"]["stroke_count"] == 1


def test_rowing_feedback_rules() -> None:
    analyzer = _analyzer()
    low = analyzer.update(_features(visible_score=0.2), 0)
    assert low["phase"] == "unknown"
    assert [message.code for message in low["feedback_messages"]] == ["LOW_VISIBILITY"]

    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    early = analyzer.update(_features(left_knee_angle=120, right_knee_angle=122, left_elbow_angle=100, right_elbow_angle=102), 150)
    assert "EARLY_ARM_PULL" in {message.code for message in early["feedback_messages"]}

    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_features(left_knee_angle=125, right_knee_angle=127), 150)
    lean = analyzer.update(_features(left_knee_angle=155, right_knee_angle=157, left_elbow_angle=105, right_elbow_angle=107, torso_angle=50), 300)
    assert "TOO_MUCH_BACK_LEAN" in {message.code for message in lean["feedback_messages"]}


def test_rowing_reports_incomplete_drive_and_rushed_recovery() -> None:
    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_features(left_knee_angle=125, right_knee_angle=127), 150)
    incomplete = analyzer.update(_features(left_knee_angle=115, right_knee_angle=117), 300)
    assert incomplete["phase"] == "recovery"
    assert "NO_FULL_LEG_DRIVE" in {message.code for message in incomplete["feedback_messages"]}

    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_features(left_knee_angle=125, right_knee_angle=127), 150)
    analyzer.update(_features(left_knee_angle=155, right_knee_angle=157, left_elbow_angle=105, right_elbow_angle=107), 300)
    analyzer.update(_features(left_knee_angle=130, right_knee_angle=132), 450)
    rushed = analyzer.update(_features(), 500)
    assert rushed["rep_count"] == 1
    assert "RUSHED_RECOVERY" in {message.code for message in rushed["feedback_messages"]}


def test_rowing_bad_view_feedback_for_upright_standing_pose() -> None:
    analyzer = _analyzer()
    state = analyzer.update(
        _features(
            left_knee_angle=170,
            right_knee_angle=172,
            left_hip_angle=170,
            right_hip_angle=172,
            left_elbow_angle=170,
            right_elbow_angle=172,
            torso_angle=2,
        ),
        0,
    )
    assert "NOT_SEATED_OR_BAD_VIEW" in {message.code for message in state["feedback_messages"]}

    frontal = analyzer.update(
        _features(hip_width=0.22, body_height_norm=0.6),
        100,
    )
    assert "NOT_SEATED_OR_BAD_VIEW" in {message.code for message in frontal["feedback_messages"]}
