from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.actions import BurpeeBroadJumpAnalyzer
from hyrox.config import (
    DEFAULT_BURPEE_BROAD_JUMP_CONFIG,
    load_burpee_broad_jump_config,
    resolve_hyrox_config_path,
)
from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.registry import create_action_analyzer
from main import parse_args
from tools.replay_hyrox_video import build_parser, create_analyzer


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "body_center_x": 0.20,
        "body_center_y": 0.40,
        "body_height_norm": 0.70,
        "torso_angle": 5.0,
        "left_knee_angle": 170.0,
        "right_knee_angle": 172.0,
        "left_hip_angle": 165.0,
        "right_hip_angle": 167.0,
        "left_wrist_y": 0.50,
        "right_wrist_y": 0.50,
        "left_ankle_y": 0.90,
        "right_ankle_y": 0.90,
        "ankle_distance_norm": 0.18,
        "shoulder_center_y": 0.25,
        "hip_center_y": 0.48,
    }
    values.update(overrides)
    return values


def _analyzer() -> BurpeeBroadJumpAnalyzer:
    return BurpeeBroadJumpAnalyzer.from_config(
        {**DEFAULT_BURPEE_BROAD_JUMP_CONFIG, "stable_frames": 1}
    )


def _hands_down() -> dict[str, float]:
    return _features(
        body_center_x=0.22,
        body_center_y=0.60,
        body_height_norm=0.45,
        torso_angle=45,
        left_knee_angle=120,
        right_knee_angle=122,
        left_hip_angle=105,
        right_hip_angle=107,
        left_wrist_y=0.82,
        right_wrist_y=0.82,
    )


def _chest_down(**overrides: float) -> dict[str, float]:
    values = {
        "body_center_x": 0.23,
        "body_center_y": 0.76,
        "body_height_norm": 0.25,
        "torso_angle": 80,
        "left_knee_angle": 150,
        "right_knee_angle": 152,
        "left_hip_angle": 155,
        "right_hip_angle": 157,
        "left_wrist_y": 0.80,
        "right_wrist_y": 0.80,
        "shoulder_center_y": 0.72,
        "hip_center_y": 0.74,
    }
    values.update(overrides)
    return _features(**values)


def _step_in() -> dict[str, float]:
    return _features(
        body_center_x=0.25,
        body_center_y=0.60,
        body_height_norm=0.50,
        torso_angle=40,
        left_knee_angle=110,
        right_knee_angle=112,
        left_hip_angle=105,
        right_hip_angle=107,
        left_wrist_y=0.65,
        right_wrist_y=0.65,
    )


def _takeoff() -> dict[str, float]:
    return _features(
        body_center_x=0.27,
        body_center_y=0.50,
        body_height_norm=0.64,
        torso_angle=15,
        left_knee_angle=155,
        right_knee_angle=157,
        left_hip_angle=155,
        right_hip_angle=157,
    )


def test_burpee_config_registry_and_cli(tmp_path: Path) -> None:
    assert load_burpee_broad_jump_config(tmp_path / "missing.yaml") == DEFAULT_BURPEE_BROAD_JUMP_CONFIG
    assert resolve_hyrox_config_path("burpee_broad_jump") == "configs/hyrox/burpee_broad_jump.yaml"
    assert "burpee_broad_jump" in HYROX_ACTION_NAMES
    assert isinstance(create_action_analyzer("burpee_broad_jump"), BurpeeBroadJumpAnalyzer)
    assert parse_args(["--hyrox-action", "burpee_broad_jump"]).hyrox_action == "burpee_broad_jump"
    assert build_parser().parse_args(["--video", "b.mp4", "--hyrox-action", "burpee_broad_jump"]).hyrox_action == "burpee_broad_jump"
    assert isinstance(create_analyzer("burpee_broad_jump", "medium"), BurpeeBroadJumpAnalyzer)


def test_burpee_custom_config_uses_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "side_45.yaml"
    path.write_text("visibility_min: 0.7\nstable_frames: 2\n", encoding="utf-8")
    analyzer = BurpeeBroadJumpAnalyzer.from_config_path(str(path))
    assert analyzer.config_name == "side_45"
    assert analyzer.min_visible_score == pytest.approx(0.7)
    assert analyzer.confirmation_frames == 2


def test_burpee_counts_chest_takeoff_landing_sequence() -> None:
    analyzer = _analyzer()
    assert analyzer.update(_features(), 0)["phase"] == "stand"
    assert analyzer.update(_hands_down(), 150)["phase"] == "hands_down"
    assert analyzer.update(_chest_down(), 300)["phase"] == "chest_down"
    assert analyzer.update(_step_in(), 450)["phase"] == "step_or_jump_in"
    assert analyzer.update(_takeoff(), 600)["phase"] == "broad_jump_takeoff"
    assert analyzer.update(_features(body_center_x=0.33, body_center_y=0.47, left_knee_angle=165, right_knee_angle=166), 700)["phase"] == "flight_or_move"
    landing = analyzer.update(_features(body_center_x=0.38, body_center_y=0.52, left_knee_angle=135, right_knee_angle=137), 850)
    assert landing["phase"] == "landing"
    assert landing["rep_count"] == 1
    assert landing["debug"]["body_center_delta_x"] == pytest.approx(0.11)


def test_burpee_visibility_bottom_and_feet_feedback() -> None:
    analyzer = _analyzer()
    low = analyzer.update(_features(visible_score=0.2), 0)
    assert [message.code for message in low["feedback_messages"]] == ["LOW_VISIBILITY"]

    analyzer = _analyzer()
    analyzer.update(_hands_down(), 0)
    bottom = analyzer.update(_chest_down(shoulder_center_y=0.75, hip_center_y=0.62), 150)
    assert "HIPS_TOO_HIGH_IN_BOTTOM" in {message.code for message in bottom["feedback_messages"]}

    analyzer.update(_step_in(), 300)
    takeoff = analyzer.update(_takeoff() | {"left_ankle_y": 0.78, "right_ankle_y": 0.90}, 450)
    assert "FEET_STAGGERED" in {message.code for message in takeoff["feedback_messages"]}


def test_burpee_reports_shallow_chest_and_missing_broad_jump() -> None:
    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_hands_down(), 150)
    shallow = analyzer.update(_step_in(), 300)
    assert "CHEST_NOT_LOW" in {message.code for message in shallow["feedback_messages"]}

    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_hands_down(), 100)
    analyzer.update(_chest_down(), 200)
    analyzer.update(_step_in(), 300)
    analyzer.update(_takeoff(), 400)
    analyzer.update(_features(body_center_x=0.30, body_center_y=0.47, left_knee_angle=165, right_knee_angle=166), 500)
    landing = analyzer.update(_features(body_center_x=0.31, body_center_y=0.52, left_knee_angle=135, right_knee_angle=137), 600)
    assert landing["rep_count"] == 1
    assert "NO_BROAD_JUMP" in {message.code for message in landing["feedback_messages"]}


def test_burpee_extra_steps_feedback() -> None:
    analyzer = _analyzer()
    analyzer.update(_features(), 0)
    analyzer.update(_hands_down(), 100)
    analyzer.update(_chest_down(), 200)
    analyzer.update(_step_in(), 300)
    analyzer.update(_takeoff(), 400)
    analyzer.update(_features(body_center_x=0.33, body_center_y=0.47, left_knee_angle=165, right_knee_angle=166), 500)
    analyzer.update(_features(body_center_x=0.38, body_center_y=0.52, left_knee_angle=135, right_knee_angle=137), 600)
    analyzer.update(_features(body_center_x=0.38, body_center_y=0.52, left_knee_angle=135, right_knee_angle=137, left_ankle_y=0.86, right_ankle_y=0.86), 700)
    extra = analyzer.update(_features(body_center_x=0.38, body_center_y=0.52, left_knee_angle=135, right_knee_angle=137, left_ankle_y=0.90, right_ankle_y=0.90), 800)
    assert "EXTRA_STEPS" in {message.code for message in extra["feedback_messages"]}
