from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.actions import SkiErgAnalyzer
from hyrox.config import DEFAULT_SKIERG_CONFIG, load_skierg_config, resolve_hyrox_config_path
from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.registry import create_action_analyzer
from main import parse_args
from tools.replay_hyrox_video import build_parser, create_analyzer


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "upper_body_visible_score": 0.95,
        "left_wrist_y": 0.15,
        "right_wrist_y": 0.15,
        "left_wrist_above_shoulder": 0.15,
        "right_wrist_above_shoulder": 0.15,
        "shoulder_center_y": 0.30,
        "hip_center_y": 0.55,
        "torso_angle": 5.0,
        "left_knee_angle": 170.0,
        "right_knee_angle": 172.0,
    }
    values.update(overrides)
    return values


def _analyzer() -> SkiErgAnalyzer:
    return SkiErgAnalyzer.from_config({**DEFAULT_SKIERG_CONFIG, "stable_frames": 1})


def _pull_down() -> dict[str, float]:
    return _features(
        left_wrist_y=0.35,
        right_wrist_y=0.35,
        left_wrist_above_shoulder=-0.05,
        right_wrist_above_shoulder=-0.05,
        torso_angle=12,
        left_knee_angle=160,
        right_knee_angle=162,
    )


def _bottom(**overrides: float) -> dict[str, float]:
    values = {
        "left_wrist_y": 0.55,
        "right_wrist_y": 0.55,
        "left_wrist_above_shoulder": -0.25,
        "right_wrist_above_shoulder": -0.25,
        "torso_angle": 25,
        "left_knee_angle": 135,
        "right_knee_angle": 137,
    }
    values.update(overrides)
    return _features(**values)


def _return() -> dict[str, float]:
    return _features(
        left_wrist_y=0.35,
        right_wrist_y=0.35,
        left_wrist_above_shoulder=-0.05,
        right_wrist_above_shoulder=-0.05,
        torso_angle=10,
        left_knee_angle=155,
        right_knee_angle=157,
    )


def test_skierg_config_registry_and_cli(tmp_path: Path) -> None:
    assert load_skierg_config(tmp_path / "missing.yaml") == DEFAULT_SKIERG_CONFIG
    assert resolve_hyrox_config_path("skierg") == "configs/hyrox/skierg.yaml"
    assert "skierg" in HYROX_ACTION_NAMES
    assert isinstance(create_action_analyzer("skierg"), SkiErgAnalyzer)
    assert parse_args(["--hyrox-action", "skierg"]).hyrox_action == "skierg"
    assert build_parser().parse_args(["--video", "ski.mp4", "--hyrox-action", "skierg"]).hyrox_action == "skierg"
    assert isinstance(create_analyzer("skierg", "medium"), SkiErgAnalyzer)


def test_skierg_custom_config_uses_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "front_view.yaml"
    path.write_text("visibility_min: 0.65\nstable_frames: 2\n", encoding="utf-8")
    analyzer = SkiErgAnalyzer.from_config_path(str(path))
    assert analyzer.config_name == "front_view"
    assert analyzer.min_visible_score == pytest.approx(0.65)
    assert analyzer.confirmation_frames == 2


def test_skierg_counts_complete_pull_sequence() -> None:
    analyzer = _analyzer()
    assert analyzer.update(_features(), 0)["phase"] == "top"
    assert analyzer.update(_pull_down(), 150)["phase"] == "pull_down"
    assert analyzer.update(_bottom(), 300)["phase"] == "bottom"
    assert analyzer.update(_return(), 450)["phase"] == "return"
    completed = analyzer.update(_features(), 600)
    assert completed["phase"] == "top"
    assert completed["rep_count"] == 1
    assert completed["debug"]["rep_completed"] is True
    assert completed["debug"]["pull_count"] == 1


def test_skierg_visibility_hinge_squat_and_asymmetry_feedback() -> None:
    analyzer = _analyzer()
    low = analyzer.update(_features(upper_body_visible_score=0.2), 0)
    assert [message.code for message in low["feedback_messages"]] == ["LOW_VISIBILITY"]

    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_pull_down(), 150)
    poor_bottom = analyzer.update(
        _bottom(torso_angle=5, left_knee_angle=100, right_knee_angle=102, right_wrist_y=0.66),
        300,
    )
    codes = {message.code for message in poor_bottom["feedback_messages"]}
    assert "NO_HIP_HINGE" in codes
    assert "TOO_MUCH_SQUAT" in codes

    asymmetry = _analyzer().update(_features(left_wrist_y=0.10, right_wrist_y=0.22), 0)
    assert "ASYMMETRIC_PULL" in {message.code for message in asymmetry["feedback_messages"]}


def test_skierg_reports_rushed_return_and_incomplete_top() -> None:
    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_pull_down(), 150)
    analyzer.update(_bottom(), 300)
    analyzer.update(_return(), 450)
    rushed = analyzer.update(_features(), 500)
    assert rushed["rep_count"] == 1
    assert "RUSHED_RETURN" in {message.code for message in rushed["feedback_messages"]}

    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_pull_down(), 150)
    analyzer.update(_bottom(), 300)
    analyzer.update(_return(), 450)
    analyzer.update(
        _features(
            left_wrist_y=0.25,
            right_wrist_y=0.25,
            left_wrist_above_shoulder=0.02,
            right_wrist_above_shoulder=0.02,
        ),
        600,
    )
    incomplete = analyzer.update(_pull_down(), 750)
    assert "ARMS_NOT_HIGH_ENOUGH" in {message.code for message in incomplete["feedback_messages"]}
