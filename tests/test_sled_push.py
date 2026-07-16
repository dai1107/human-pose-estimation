from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.actions import SledPushAnalyzer
from hyrox.config import DEFAULT_SLED_PUSH_CONFIG, load_sled_push_config, resolve_hyrox_config_path
from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.registry import create_action_analyzer
from main import parse_args
from tools.replay_hyrox_video import build_parser, create_analyzer


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "torso_angle": 35.0,
        "left_knee_angle": 120.0,
        "right_knee_angle": 122.0,
        "body_center_x": 0.40,
        "left_ankle_y": 0.90,
        "right_ankle_y": 0.88,
        "ankle_distance_norm": 0.20,
        "left_wrist_y": 0.32,
        "right_wrist_y": 0.33,
        "shoulder_center_y": 0.30,
        "hip_tilt": 0.01,
    }
    values.update(overrides)
    return values


def _analyzer() -> SledPushAnalyzer:
    return SledPushAnalyzer.from_config({**DEFAULT_SLED_PUSH_CONFIG, "stable_frames": 1})


def test_sled_push_config_registry_and_cli(tmp_path: Path) -> None:
    assert load_sled_push_config(tmp_path / "missing.yaml") == DEFAULT_SLED_PUSH_CONFIG
    assert resolve_hyrox_config_path("sled_push") == "configs/hyrox/sled_push.yaml"
    assert "sled_push" in HYROX_ACTION_NAMES
    assert isinstance(create_action_analyzer("sled_push"), SledPushAnalyzer)
    assert parse_args(["--hyrox-action", "sled_push"]).hyrox_action == "sled_push"
    assert build_parser().parse_args(["--video", "push.mp4", "--hyrox-action", "sled_push"]).hyrox_action == "sled_push"
    assert isinstance(create_analyzer("sled_push", "medium"), SledPushAnalyzer)


def test_sled_push_custom_config_uses_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "side_view.yaml"
    path.write_text("visibility_min: 0.7\nstable_frames: 2\n", encoding="utf-8")
    analyzer = SledPushAnalyzer.from_config_path(str(path))
    assert analyzer.config_name == "side_view"
    assert analyzer.min_visible_score == pytest.approx(0.7)
    assert analyzer.confirmation_frames == 2


def test_sled_push_phases_and_step_count() -> None:
    analyzer = _analyzer()
    assert analyzer.update(_features(), 0)["phase"] == "setup"
    drive = analyzer.update(
        _features(body_center_x=0.41, left_knee_angle=130, right_knee_angle=132),
        100,
    )
    assert drive["phase"] == "drive"
    step = analyzer.update(
        _features(
            body_center_x=0.42,
            left_knee_angle=145,
            right_knee_angle=147,
            left_ankle_y=0.84,
        ),
        400,
    )
    assert step["phase"] == "step"
    assert step["rep_count"] == 1
    assert step["debug"]["rep_completed"] is True
    assert step["debug"]["step_count"] == 1
    assert analyzer.update(_features(torso_angle=10), 500)["phase"] == "reset"


def test_sled_push_counts_each_recurring_drive_step_cycle_at_step_phase() -> None:
    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_features(body_center_x=0.41, left_knee_angle=130, right_knee_angle=132), 100)
    first = analyzer.update(
        _features(body_center_x=0.42, left_knee_angle=145, right_knee_angle=147, left_ankle_y=0.84),
        200,
    )
    analyzer.update(
        _features(body_center_x=0.43, left_knee_angle=130, right_knee_angle=132, left_ankle_y=0.84),
        400,
    )
    second = analyzer.update(
        _features(body_center_x=0.44, left_knee_angle=145, right_knee_angle=147, left_ankle_y=0.76),
        600,
    )

    assert first["rep_count"] == 1
    assert second["phase"] == "step"
    assert second["rep_count"] == 2
    assert second["debug"]["rep_completed"] is True


def test_sled_push_visibility_and_torso_feedback() -> None:
    analyzer = _analyzer()
    low = analyzer.update(_features(visible_score=0.2), 0)
    assert [message.code for message in low["feedback_messages"]] == ["LOW_VISIBILITY"]

    analyzer = SledPushAnalyzer()
    for timestamp in (0, 50, 100):
        analyzer.update(_features(), timestamp)
    upright = analyzer.update(_features(torso_angle=15), 150)
    assert "TORSO_TOO_UPRIGHT" in {message.code for message in upright["feedback_messages"]}
    too_low = analyzer.update(_features(torso_angle=75), 200)
    assert "TORSO_TOO_LOW" in {message.code for message in too_low["feedback_messages"]}


def test_sled_push_short_step_no_leg_drive_and_instability_feedback() -> None:
    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    short = analyzer.update(
        _features(body_center_x=0.41, left_ankle_y=0.88, hip_tilt=0.10),
        100,
    )
    codes = {message.code for message in short["feedback_messages"]}
    assert "SHORT_STEPS" in codes
    assert "HIP_TOO_HIGH_OR_BACK_ROUND" in codes

    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_features(body_center_x=0.41, left_knee_angle=125, right_knee_angle=127), 100)
    weak_step = analyzer.update(
        _features(body_center_x=0.42, left_knee_angle=126, right_knee_angle=128, left_ankle_y=0.84),
        300,
    )
    assert "NO_LEG_DRIVE" in {message.code for message in weak_step["feedback_messages"]}
