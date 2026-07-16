from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.actions import WallBallAnalyzer
from hyrox.config import DEFAULT_WALL_BALL_CONFIG, load_wall_ball_config, resolve_hyrox_config_path


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "left_knee_angle": 175.0,
        "right_knee_angle": 176.0,
        "left_hip_angle": 170.0,
        "right_hip_angle": 171.0,
        "left_elbow_angle": 105.0,
        "right_elbow_angle": 106.0,
        "hip_center_y": 0.45,
        "hip_knee_depth": -0.20,
        "wrist_above_shoulder": -0.08,
        "knee_width": 0.18,
        "ankle_width": 0.20,
    }
    values.update(overrides)
    return values


def _feed(analyzer: WallBallAnalyzer, features: dict[str, float], timestamps: tuple[int, ...]):
    state = None
    for timestamp in timestamps:
        state = analyzer.update(features, timestamp)
    assert state is not None
    return state


def test_wall_ball_config_defaults_and_action_specific_paths(tmp_path: Path) -> None:
    config = load_wall_ball_config(tmp_path / "missing.yaml")

    assert config == DEFAULT_WALL_BALL_CONFIG
    assert resolve_hyrox_config_path("lunge") == "configs/hyrox/lunge.yaml"
    assert resolve_hyrox_config_path("wall_ball") == "configs/hyrox/wall_ball.yaml"
    assert resolve_hyrox_config_path("wall_ball", "custom.yaml") == "custom.yaml"


def test_wall_ball_config_uses_custom_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "front_view.yaml"
    path.write_text("stable_frames: 2\n", encoding="utf-8")

    analyzer = WallBallAnalyzer.from_config_path(str(path))

    assert analyzer.config_name == "front_view"
    assert analyzer.confirmation_frames == 2


def test_wall_ball_counts_deep_stand_bottom_extension_cycle() -> None:
    analyzer = WallBallAnalyzer()
    stand = _features()
    squat_down = _features(
        left_knee_angle=132.0,
        right_knee_angle=134.0,
        left_hip_angle=138.0,
        right_hip_angle=140.0,
        hip_center_y=0.55,
        hip_knee_depth=-0.06,
    )
    bottom = _features(
        left_knee_angle=92.0,
        right_knee_angle=94.0,
        left_hip_angle=105.0,
        right_hip_angle=108.0,
        hip_center_y=0.68,
        hip_knee_depth=0.025,
    )
    extension = _features(
        left_knee_angle=171.0,
        right_knee_angle=172.0,
        left_hip_angle=166.0,
        right_hip_angle=167.0,
        left_elbow_angle=168.0,
        right_elbow_angle=170.0,
        wrist_above_shoulder=0.12,
    )
    reset = _features()

    assert _feed(analyzer, stand, (100, 150, 200))["phase"] == "stand"
    assert _feed(analyzer, squat_down, (250, 300, 350))["phase"] == "squat_down"
    assert _feed(analyzer, bottom, (400, 450, 500))["phase"] == "bottom"
    assert _feed(analyzer, extension, (550, 600, 650))["phase"] == "drive"
    completed = _feed(analyzer, extension, (700,))

    assert completed["phase"] == "throw_extension"
    assert completed["rep_count"] == 1
    assert completed["debug"]["rep_completed"] is True
    assert completed["debug"]["last_rep_time_ms"] == 700
    assert completed["debug"]["bottom_depth_met"] is False

    reset_state = _feed(analyzer, reset, (850, 900, 950))
    assert reset_state["phase"] == "reset"
    assert reset_state["rep_count"] == 1


def test_wall_ball_keeps_rep_progress_across_a_short_landmark_dropout() -> None:
    analyzer = WallBallAnalyzer.from_config({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
    bottom = _features(
        left_knee_angle=94.0,
        right_knee_angle=96.0,
        left_hip_angle=105.0,
        right_hip_angle=108.0,
        hip_knee_depth=0.02,
    )
    extension = _features(
        left_elbow_angle=168.0,
        right_elbow_angle=170.0,
        wrist_above_shoulder=0.12,
    )

    analyzer.update(_features(), 100)
    dropout = analyzer.update({"visible_score": 0.95}, 125)
    analyzer.update(bottom, 150)  # squat-down is transition-only and may be skipped
    analyzer.update(extension, 200)  # explicit drive phase
    completed = analyzer.update(extension, 250)

    assert dropout["phase"] == "stand"
    assert dropout["debug"]["raw_phase"] == "unknown"
    assert completed["rep_count"] == 1
    assert completed["debug"]["rep_completed"] is True


def test_wall_ball_does_not_count_shallow_bottom() -> None:
    analyzer = WallBallAnalyzer.from_config({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
    analyzer.update(_features(), 100)
    analyzer.update(
        _features(left_knee_angle=132, right_knee_angle=134, left_hip_angle=138, right_hip_angle=140),
        125,
    )
    shallow = analyzer.update(
        _features(
            left_knee_angle=98.0,
            right_knee_angle=100.0,
            left_hip_angle=110.0,
            right_hip_angle=112.0,
            hip_knee_depth=-0.08,
        ),
        150,
    )
    returned = analyzer.update(_features(), 200)

    assert shallow["phase"] == "bottom"
    assert [message.code for message in shallow["feedback_messages"]] == ["SQUAT_NOT_DEEP"]
    assert returned["rep_count"] == 0


def test_wall_ball_knee_cave_feedback_is_low_confidence_warning() -> None:
    analyzer = WallBallAnalyzer.from_config({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
    analyzer.update(_features(), 100)
    state = analyzer.update(
        _features(
            left_knee_angle=95.0,
            right_knee_angle=96.0,
            left_hip_angle=105.0,
            right_hip_angle=106.0,
            hip_knee_depth=0.02,
            knee_width=0.07,
            ankle_width=0.20,
        ),
        150,
    )

    message = state["feedback_messages"][0]
    assert message.code == "KNEES_CAVE_IN"
    assert message.level == "warn"
    assert message.confidence <= 0.45


def test_wall_ball_waits_until_throw_before_judging_extension() -> None:
    analyzer = WallBallAnalyzer.from_config({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
    analyzer.update(_features(), 100)
    analyzer.update(
        _features(left_knee_angle=132, right_knee_angle=134, left_hip_angle=138, right_hip_angle=140),
        125,
    )
    analyzer.update(
        _features(
            left_knee_angle=95.0,
            right_knee_angle=96.0,
            left_hip_angle=105.0,
            right_hip_angle=106.0,
            hip_knee_depth=0.02,
        ),
        150,
    )
    rising = analyzer.update(
        _features(
            left_knee_angle=155.0,
            right_knee_angle=156.0,
            left_hip_angle=150.0,
            right_hip_angle=151.0,
        ),
        600,
    )
    state = analyzer.update(
        _features(
            left_knee_angle=155.0,
            right_knee_angle=156.0,
            left_hip_angle=150.0,
            right_hip_angle=151.0,
            left_elbow_angle=160.0,
            right_elbow_angle=162.0,
            wrist_above_shoulder=0.10,
        ),
        650,
    )

    assert rising["phase"] == "drive"
    assert rising["rep_count"] == 0
    assert rising["feedback_messages"] == []
    assert state["phase"] == "throw_extension"
    assert state["rep_count"] == 1
    assert [message.code for message in state["feedback_messages"]] == ["NOT_FULL_EXTENSION"]


def test_wall_ball_low_visibility_is_exclusive() -> None:
    analyzer = WallBallAnalyzer.from_config({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})

    state = analyzer.update(
        _features(visible_score=0.1, knee_width=0.01, ankle_width=0.3),
        100,
    )

    assert [message.code for message in state["feedback_messages"]] == ["LOW_VISIBILITY"]
    assert state["debug"]["raw_phase"] == "low_visibility"


def test_wall_ball_sensitivity_changes_phase_confirmation() -> None:
    low = WallBallAnalyzer.from_config_path("configs/hyrox/wall_ball.yaml", sensitivity="low")
    medium = WallBallAnalyzer.from_config_path("configs/hyrox/wall_ball.yaml", sensitivity="medium")
    high = WallBallAnalyzer.from_config_path("configs/hyrox/wall_ball.yaml", sensitivity="high")

    assert low.confirmation_frames == 3
    assert medium.confirmation_frames == 2
    assert high.confirmation_frames == 1


def test_wall_ball_counts_each_complete_ordered_sequence_without_terminal_lag() -> None:
    analyzer = WallBallAnalyzer.from_config({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
    bottom = _features(
        left_knee_angle=95.0,
        right_knee_angle=96.0,
        left_hip_angle=105.0,
        right_hip_angle=106.0,
        hip_knee_depth=0.02,
    )
    extension = _features(
        left_elbow_angle=168.0,
        right_elbow_angle=170.0,
        wrist_above_shoulder=0.12,
    )

    squat = _features(left_knee_angle=132, right_knee_angle=134, left_hip_angle=138, right_hip_angle=140)

    analyzer.update(_features(), 100)
    analyzer.update(squat, 125)
    analyzer.update(bottom, 150)
    analyzer.update(extension, 550)
    assert analyzer.update(extension, 600)["rep_count"] == 1

    analyzer.update(_features(), 650)  # reset
    analyzer.update(_features(), 700)  # stand
    analyzer.update(squat, 725)
    analyzer.update(bottom, 750)
    analyzer.update(extension, 800)
    assert analyzer.update(extension, 850)["rep_count"] == 2

    analyzer.update(_features(), 900)  # reset
    analyzer.update(_features(), 950)  # stand
    analyzer.update(squat, 975)
    analyzer.update(bottom, 1000)
    analyzer.update(extension, 1050)
    assert analyzer.update(extension, 1100)["rep_count"] == 3
