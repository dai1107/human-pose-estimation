from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.actions import SledPullAnalyzer
from hyrox.config import DEFAULT_SLED_PULL_CONFIG, load_sled_pull_config, resolve_hyrox_config_path
from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.registry import create_action_analyzer
from main import parse_args
from tools.replay_hyrox_video import build_parser, create_analyzer


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "left_elbow_angle": 120.0,
        "right_elbow_angle": 122.0,
        "left_knee_angle": 155.0,
        "right_knee_angle": 157.0,
        "left_hip_angle": 150.0,
        "right_hip_angle": 152.0,
        "torso_angle": 10.0,
        "body_center_y": 0.50,
        "wrist_distance_norm": 0.18,
        "left_wrist_y": 0.40,
        "right_wrist_y": 0.41,
    }
    values.update(overrides)
    return values


def _analyzer() -> SledPullAnalyzer:
    return SledPullAnalyzer.from_config({**DEFAULT_SLED_PULL_CONFIG, "stable_frames": 1})


def _reach() -> dict[str, float]:
    return _features(left_elbow_angle=160, right_elbow_angle=162)


def test_sled_pull_config_registry_and_cli(tmp_path: Path) -> None:
    assert load_sled_pull_config(tmp_path / "missing.yaml") == DEFAULT_SLED_PULL_CONFIG
    assert resolve_hyrox_config_path("sled_pull") == "configs/hyrox/sled_pull.yaml"
    assert "sled_pull" in HYROX_ACTION_NAMES
    assert isinstance(create_action_analyzer("sled_pull"), SledPullAnalyzer)
    assert parse_args(["--hyrox-action", "sled_pull"]).hyrox_action == "sled_pull"
    assert build_parser().parse_args(["--video", "pull.mp4", "--hyrox-action", "sled_pull"]).hyrox_action == "sled_pull"
    assert isinstance(create_analyzer("sled_pull", "medium"), SledPullAnalyzer)


def test_sled_pull_custom_config_uses_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "oblique.yaml"
    path.write_text("visibility_min: 0.7\nstable_frames: 2\n", encoding="utf-8")
    analyzer = SledPullAnalyzer.from_config_path(str(path))
    assert analyzer.config_name == "oblique"
    assert analyzer.min_visible_score == pytest.approx(0.7)
    assert analyzer.confirmation_frames == 2


def test_sled_pull_counts_reach_pull_recover_cycle() -> None:
    analyzer = _analyzer()
    assert analyzer.update(_features(), 0)["phase"] == "ready"
    assert analyzer.update(_reach(), 100)["phase"] == "reach"
    assert analyzer.update(_features(left_elbow_angle=125, right_elbow_angle=127), 200)["phase"] == "pull"
    recovered = analyzer.update(_features(left_elbow_angle=145, right_elbow_angle=147), 400)
    assert recovered["phase"] == "recover"
    assert recovered["rep_count"] == 1
    assert recovered["debug"]["rep_completed"] is True
    assert recovered["debug"]["pull_count"] == 1


def test_sled_pull_low_visibility_standing_lean_and_asymmetry_feedback() -> None:
    analyzer = _analyzer()
    low = analyzer.update(_features(visible_score=0.2), 0)
    assert [message.code for message in low["feedback_messages"]] == ["LOW_VISIBILITY"]

    state = _analyzer().update(
        _features(
            left_knee_angle=90,
            right_knee_angle=92,
            body_center_y=0.80,
            torso_angle=42,
            left_wrist_y=0.30,
            right_wrist_y=0.42,
        ),
        0,
    )
    codes = {message.code for message in state["feedback_messages"]}
    assert "NOT_STANDING" in codes
    assert "OVER_LEAN_BACK" in codes

    asymmetry = _analyzer().update(_features(left_wrist_y=0.30, right_wrist_y=0.42), 0)
    assert "ASYMMETRIC_PULL" in {message.code for message in asymmetry["feedback_messages"]}


def test_sled_pull_reports_arms_only_and_no_clear_pull() -> None:
    analyzer = _analyzer()
    analyzer.update(_reach(), 0)
    analyzer.update(_features(left_elbow_angle=125, right_elbow_angle=127), 100)
    arms_only = analyzer.update(_features(left_elbow_angle=145, right_elbow_angle=147), 200)
    assert arms_only["rep_count"] == 1
    assert "ARMS_ONLY_PULL" in {message.code for message in arms_only["feedback_messages"]}

    analyzer = _analyzer()
    analyzer.update(_reach(), 0)
    analyzer.update(_features(left_elbow_angle=150, right_elbow_angle=152), 100)
    unclear = analyzer.update(_features(left_elbow_angle=156, right_elbow_angle=158), 200)
    assert unclear["rep_count"] == 1
    assert "NO_CLEAR_PULL" in {message.code for message in unclear["feedback_messages"]}
