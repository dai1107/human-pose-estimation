from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.actions import WallBallAnalyzer
from hyrox.actions.wall_ball import WALL_BALL_REQUIRED_RULES
from hyrox.config import DEFAULT_WALL_BALL_CONFIG, load_wall_ball_config, resolve_hyrox_config_path
from hyrox.registry import create_action_analyzer


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "left_knee_angle": 175.0,
        "right_knee_angle": 176.0,
        "left_hip_angle": 170.0,
        "right_hip_angle": 171.0,
        "left_elbow_angle": 105.0,
        "right_elbow_angle": 106.0,
        "torso_angle": 0.0,
        "body_center_x": 0.50,
        "body_box_height_norm": 0.75,
        "body_height_norm": 0.65,
        "lower_body_visible_score": 0.95,
        "shoulder_center_y": 0.30,
        "hip_center_y": 0.45,
        "hip_center_x": 0.50,
        "hip_knee_depth": -0.20,
        "knee_center_x": 0.50,
        "left_wrist_x": 0.44,
        "right_wrist_x": 0.56,
        "left_wrist_y": 0.40,
        "right_wrist_y": 0.40,
        "left_wrist_confidence": 0.95,
        "right_wrist_confidence": 0.95,
        "left_wrist_above_shoulder": -0.10,
        "right_wrist_above_shoulder": -0.10,
        "wrist_above_shoulder": -0.08,
        "knee_width": 0.18,
        "ankle_width": 0.20,
        "left_heel_x": 0.42,
        "left_heel_y": 0.90,
        "left_heel_confidence": 0.95,
        "right_heel_x": 0.58,
        "right_heel_y": 0.90,
        "right_heel_confidence": 0.95,
        "left_foot_index_x": 0.44,
        "left_foot_index_y": 0.90,
        "left_foot_index_confidence": 0.95,
        "right_foot_index_x": 0.60,
        "right_foot_index_y": 0.90,
        "right_foot_index_confidence": 0.95,
    }
    values.update(overrides)
    if "knee_center_y" not in overrides:
        values["knee_center_y"] = (
            values["hip_center_y"] - values["hip_knee_depth"]
        )
    if "wrist_above_shoulder" in overrides:
        above = values["wrist_above_shoulder"]
        if "left_wrist_above_shoulder" not in overrides:
            values["left_wrist_above_shoulder"] = above
        if "right_wrist_above_shoulder" not in overrides:
            values["right_wrist_above_shoulder"] = above
        if "left_wrist_y" not in overrides:
            values["left_wrist_y"] = values["shoulder_center_y"] - above
        if "right_wrist_y" not in overrides:
            values["right_wrist_y"] = values["shoulder_center_y"] - above
    return values


def _analyzer(
    config: dict[str, object] | None = None,
) -> WallBallAnalyzer:
    analyzer = WallBallAnalyzer.from_config(config)
    analyzer.set_manual_floor_line((0.0, 0.90), (1.0, 0.90))
    return analyzer


def _feed(analyzer: WallBallAnalyzer, features: dict[str, float], timestamps: tuple[int, ...]):
    state = None
    for timestamp in timestamps:
        state = analyzer.update(features, timestamp)
    assert state is not None
    return state


def _valid_bottom(**overrides: float) -> dict[str, float]:
    values = _features(
        left_knee_angle=94.0,
        right_knee_angle=96.0,
        left_hip_angle=105.0,
        right_hip_angle=108.0,
        hip_center_y=0.68,
        hip_knee_depth=0.02,
    )
    values.update(overrides)
    if "hip_center_y" in overrides or "hip_knee_depth" in overrides:
        values["knee_center_y"] = (
            values["hip_center_y"] - values["hip_knee_depth"]
        )
    return values


def _valid_throw(**overrides: float) -> dict[str, float]:
    values = _features(
        left_knee_angle=171.0,
        right_knee_angle=172.0,
        left_hip_angle=166.0,
        right_hip_angle=167.0,
        left_elbow_angle=168.0,
        right_elbow_angle=170.0,
        wrist_above_shoulder=0.12,
    )
    values.update(overrides)
    return values


def _rules(state: dict[str, object]) -> dict[str, dict[str, object]]:
    decision = state["last_rep_decision"]
    assert isinstance(decision, dict)
    return {
        str(rule["rule_id"]): rule
        for rule in decision["rules"]
    }


def _run_rule_candidate(
    analyzer: WallBallAnalyzer | None = None,
    *,
    start: dict[str, float] | None = None,
    bottom: dict[str, float] | None = None,
    throw: dict[str, float] | None = None,
) -> dict[str, object]:
    resolved = analyzer or _analyzer(
        {**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1}
    )
    if start is not None:
        resolved.update(start, 100)
    resolved.update(bottom or _valid_bottom(), 200)
    return resolved.update(throw or _valid_throw(), 500)


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
    analyzer = _analyzer()
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
    completed = _feed(analyzer, extension, (550, 600))

    assert completed["phase"] == "throw_extension"
    assert completed["rep_count"] == 1
    assert completed["debug"]["rep_completed"] is True
    assert completed["debug"]["last_rep_time_ms"] == 600
    assert completed["debug"]["bottom_depth_met"] is False

    reset_state = _feed(analyzer, reset, (850, 900, 950))
    assert reset_state["phase"] == "reset"
    assert reset_state["rep_count"] == 1


def test_wall_ball_keeps_rep_progress_across_a_short_landmark_dropout() -> None:
    analyzer = _analyzer({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
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
    completed = analyzer.update(extension, 200)

    assert dropout["phase"] == "stand"
    assert dropout["debug"]["raw_phase"] == "unknown"
    assert completed["rep_count"] == 1
    assert completed["debug"]["rep_completed"] is True


def test_realtime_wall_ball_counts_a_single_sampled_throw_endpoint() -> None:
    analyzer = create_action_analyzer("wall_ball", live_mode=True, camera_view="front")
    analyzer.set_manual_floor_line((0.0, 0.90), (1.0, 0.90))
    bottom = _features(
        left_knee_angle=94.0,
        right_knee_angle=96.0,
        left_hip_angle=105.0,
        right_hip_angle=108.0,
        hip_knee_depth=0.02,
    )
    extension = _features(
        left_elbow_angle=float("nan"),
        right_elbow_angle=float("nan"),
        wrist_above_shoulder=0.12,
    )

    analyzer.update(_features(), 100)
    analyzer.update(bottom, 300)
    completed = analyzer.update(extension, 600)

    assert analyzer.confirmation_frames == 1
    assert completed["phase"] == "throw_extension"
    assert completed["rep_count"] == 1
    assert completed["debug"]["rep_completed"] is True


def test_wall_ball_does_not_count_shallow_bottom() -> None:
    analyzer = _analyzer({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
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
    analyzer = _analyzer({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
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
    analyzer = _analyzer({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
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
    assert state["rep_count"] == 0
    assert state["unsure_count"] == 1
    assert state["last_rep_decision"]["status"] == "UNSURE"
    assert [message.code for message in state["feedback_messages"]] == ["NOT_FULL_EXTENSION"]


def test_wall_ball_low_visibility_is_exclusive() -> None:
    analyzer = _analyzer({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})

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
    analyzer = _analyzer({**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1})
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

    analyzer.update(_features(), 700)  # reset
    analyzer.update(_features(), 750)  # stand
    analyzer.update(squat, 800)
    analyzer.update(bottom, 850)
    analyzer.update(extension, 1000)
    assert analyzer.update(extension, 1050)["rep_count"] == 2

    analyzer.update(_features(), 1100)  # reset
    analyzer.update(_features(), 1150)  # stand
    analyzer.update(squat, 1200)
    analyzer.update(bottom, 1250)
    analyzer.update(extension, 1450)
    assert analyzer.update(extension, 1500)["rep_count"] == 3


def test_wall_ball_validity_uses_all_four_required_rules_and_states() -> None:
    analyzer = _analyzer(
        {**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1}
    )

    start = analyzer.update(_features(), 100)
    bottom = analyzer.update(_valid_bottom(), 200)
    completed = analyzer.update(_valid_throw(), 500)
    rules = _rules(completed)

    assert start["debug"]["wall_ball_validation_state"] == "TALL_START"
    assert bottom["debug"]["wall_ball_validation_state"] == (
        "HIP_BELOW_KNEE_CONFIRMED"
    )
    assert completed["debug"]["wall_ball_validation_state"] == "POSE_VALID_REP"
    assert completed["candidate_count"] == 1
    assert completed["rep_count"] == 1
    assert completed["last_rep_decision"]["status"] == "VALID"
    assert tuple(rules) == WALL_BALL_REQUIRED_RULES
    assert all(rule["status"] == "PASS" for rule in rules.values())
    assert completed["last_rep_candidate"]["events"]["throw_proxy_name"] == (
        "BILATERAL_THROW_PROXY"
    )


def test_throw_extension_is_the_tall_start_of_the_next_continuous_rep() -> None:
    analyzer = _analyzer(
        {**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1}
    )

    analyzer.update(_valid_bottom(), 100)
    first = analyzer.update(_valid_throw(), 300)
    analyzer.update(_valid_bottom(), 500)
    second = analyzer.update(_valid_throw(), 800)

    assert first["last_rep_decision"]["status"] == "UNSURE"
    assert second["candidate_count"] == 2
    assert second["last_rep_decision"]["status"] == "VALID"
    assert _rules(second)["tall_start"]["status"] == "PASS"


def test_wall_ball_requires_a_tall_start_before_descent() -> None:
    missing_start = _run_rule_candidate()
    short_start = _run_rule_candidate(
        start=_features(
            left_knee_angle=160.0,
            right_knee_angle=161.0,
            left_hip_angle=160.0,
            right_hip_angle=161.0,
            torso_angle=30.0,
        )
    )

    assert missing_start["last_rep_decision"]["status"] == "UNSURE"
    assert _rules(missing_start)["tall_start"]["status"] == "UNSURE"
    assert short_start["last_rep_decision"]["status"] == "UNSURE"
    assert _rules(short_start)["tall_start"]["status"] == "FAIL"


def test_wall_ball_hip_depth_uses_floor_relative_height() -> None:
    completed = _run_rule_candidate(
        start=_features(),
        bottom=_valid_bottom(hip_knee_depth=-0.01),
    )
    rule = _rules(completed)["hip_below_knee"]

    assert completed["last_rep_decision"]["status"] == "UNSURE"
    assert rule["status"] == "FAIL"
    assert rule["value"] == pytest.approx(-0.01 / 0.75)


def test_wall_ball_missing_floor_depth_evidence_is_unsure() -> None:
    analyzer = WallBallAnalyzer.from_config(
        {**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1}
    )
    completed = _run_rule_candidate(
        analyzer,
        start=_features(),
    )

    assert completed["last_rep_decision"]["status"] == "UNSURE"
    assert _rules(completed)["hip_below_knee"]["status"] == "UNSURE"


def test_wall_ball_upward_extension_is_required_for_count() -> None:
    completed = _run_rule_candidate(
        start=_features(),
        throw=_valid_throw(
            left_knee_angle=155.0,
            right_knee_angle=156.0,
            left_hip_angle=150.0,
            right_hip_angle=151.0,
        ),
    )

    assert completed["last_rep_decision"]["status"] == "UNSURE"
    assert _rules(completed)["upward_extension"]["status"] == "FAIL"


def test_wall_ball_bilateral_throw_requires_both_wrists() -> None:
    completed = _run_rule_candidate(
        start=_features(),
        throw=_valid_throw(
            right_wrist_y=0.40,
            right_wrist_above_shoulder=-0.10,
        ),
    )

    assert completed["last_rep_decision"]["status"] == "NO_REP"
    assert _rules(completed)["bilateral_throw_proxy"]["status"] == "FAIL"


@pytest.mark.parametrize(
    ("final_timestamp", "decision_status", "rule_status"),
    ((360, "UNSURE", "UNSURE"), (450, "NO_REP", "FAIL")),
)
def test_wall_ball_wrist_peak_timing_has_pass_unsure_fail_windows(
    final_timestamp: int,
    decision_status: str,
    rule_status: str,
) -> None:
    analyzer = _analyzer(
        {**DEFAULT_WALL_BALL_CONFIG, "stable_frames": 1}
    )
    analyzer.update(_features(), 50)
    analyzer.update(_valid_bottom(), 100)
    analyzer.update(
        _features(
            left_knee_angle=155.0,
            right_knee_angle=156.0,
            left_hip_angle=150.0,
            right_hip_angle=151.0,
            left_wrist_y=0.15,
            right_wrist_y=0.40,
            wrist_above_shoulder=-0.05,
        ),
        200,
    )
    completed = analyzer.update(_valid_throw(), final_timestamp)
    rule = _rules(completed)["bilateral_throw_proxy"]

    assert completed["last_rep_decision"]["status"] == decision_status
    assert rule["status"] == rule_status
    assert rule["value"] == final_timestamp - 200


def test_wall_ball_missing_one_wrist_is_unsure_not_valid() -> None:
    completed = _run_rule_candidate(
        start=_features(),
        throw=_valid_throw(
            right_wrist_x=float("nan"),
            right_wrist_y=float("nan"),
            right_wrist_above_shoulder=float("nan"),
        ),
    )

    assert completed["last_rep_decision"]["status"] == "UNSURE"
    assert _rules(completed)["bilateral_throw_proxy"]["status"] == "UNSURE"
