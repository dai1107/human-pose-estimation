from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.actions import FarmersCarryAnalyzer
from hyrox.config import (
    DEFAULT_FARMERS_CARRY_CONFIG,
    load_farmers_carry_config,
    resolve_hyrox_config_path,
)
from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.registry import create_action_analyzer
from main import parse_args
from tools.replay_hyrox_video import build_parser, create_analyzer


def _features(**overrides: float | None) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "visible_score": 0.95,
        "left_knee_angle": 174.0,
        "right_knee_angle": 175.0,
        "left_hip_angle": 168.0,
        "right_hip_angle": 169.0,
        "left_elbow_angle": 170.0,
        "right_elbow_angle": 171.0,
        "left_wrist_to_hip_y": 0.18,
        "right_wrist_to_hip_y": 0.19,
        "shoulder_tilt": 0.01,
        "hip_tilt": 0.01,
        "torso_angle": 3.0,
        "body_center_x": 0.50,
        "body_center_y": 0.42,
        "body_height_norm": 0.72,
        "ankle_distance_norm": 0.20,
    }
    values.update(overrides)
    return values


def _feed(
    analyzer: FarmersCarryAnalyzer,
    features: dict[str, float | None],
    timestamps: tuple[int, ...],
):
    state = None
    for timestamp in timestamps:
        state = analyzer.update(features, timestamp)
    assert state is not None
    return state


def test_farmers_carry_config_and_registry(tmp_path: Path) -> None:
    assert load_farmers_carry_config(tmp_path / "missing.yaml") == DEFAULT_FARMERS_CARRY_CONFIG
    assert resolve_hyrox_config_path("farmers_carry") == "configs/hyrox/farmers_carry.yaml"
    assert "farmers_carry" in HYROX_ACTION_NAMES

    analyzer = create_action_analyzer("farmers_carry")
    assert isinstance(analyzer, FarmersCarryAnalyzer)
    assert analyzer.config_name == "farmers_carry_default"


def test_farmers_carry_custom_config_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "carry_side.yaml"
    path.write_text("visibility_min: 0.7\nstable_frames: 1\nrest_timeout_ms: 900\n", encoding="utf-8")

    analyzer = FarmersCarryAnalyzer.from_config_path(str(path))

    assert analyzer.config_name == "carry_side"
    assert analyzer.min_visible_score == pytest.approx(0.7)
    assert analyzer.confirmation_frames == 1
    assert analyzer.rest_timeout_ms == 900


def test_farmers_carry_transitions_ready_carrying_and_rest() -> None:
    analyzer = FarmersCarryAnalyzer()
    ready = _feed(analyzer, _features(), (0, 100, 200))
    assert ready["phase"] == "ready"

    analyzer.update(_features(body_center_x=0.51), 250)
    analyzer.update(_features(body_center_x=0.52), 300)
    carrying = analyzer.update(_features(body_center_x=0.53), 350)
    assert carrying["phase"] == "carrying"
    assert carrying["rep_count"] == 0
    assert carrying["debug"]["carrying_score"] > 0.8

    analyzer.update(_features(body_center_x=0.53), 1600)
    analyzer.update(_features(body_center_x=0.53), 1650)
    resting = analyzer.update(_features(body_center_x=0.53), 1700)
    assert resting["phase"] == "rest"
    assert resting["debug"]["stationary_ms"] >= 1200


def test_farmers_carry_low_visibility_is_unknown_and_exclusive() -> None:
    analyzer = FarmersCarryAnalyzer.from_config({**DEFAULT_FARMERS_CARRY_CONFIG, "stable_frames": 1})
    state = analyzer.update(
        _features(visible_score=0.2, shoulder_tilt=0.2, torso_angle=40.0),
        100,
    )

    assert state["phase"] == "unknown"
    assert [message.code for message in state["feedback_messages"]] == ["LOW_VISIBILITY"]


def test_farmers_carry_emits_posture_feedback() -> None:
    analyzer = FarmersCarryAnalyzer.from_config({**DEFAULT_FARMERS_CARRY_CONFIG, "stable_frames": 1})
    uneven = analyzer.update(_features(shoulder_tilt=0.12), 100)
    assert {message.code for message in uneven["feedback_messages"]} == {
        "LEAN_LEFT_RIGHT",
        "SHOULDERS_UNEVEN",
    }

    arms_and_torso = analyzer.update(
        _features(
            left_wrist_to_hip_y=-0.08,
            right_wrist_to_hip_y=-0.07,
            torso_angle=31.0,
        ),
        150,
    )
    assert [message.code for message in arms_and_torso["feedback_messages"]] == [
        "ARMS_NOT_DOWN",
        "TORSO_LEAN",
    ]
    assert arms_and_torso["phase"] == "rest"


def test_farmers_carry_detects_unstable_carry() -> None:
    analyzer = FarmersCarryAnalyzer.from_config({**DEFAULT_FARMERS_CARRY_CONFIG, "stable_frames": 1})
    analyzer.update(_features(), 100)
    state = analyzer.update(_features(body_center_x=0.52, body_center_y=0.46), 150)

    assert state["phase"] == "carrying"
    assert "UNSTABLE_CARRY" in {message.code for message in state["feedback_messages"]}


def test_farmers_carry_is_available_in_realtime_and_replay_cli() -> None:
    assert parse_args(["--hyrox-action", "farmers_carry"]).hyrox_action == "farmers_carry"
    replay_args = build_parser().parse_args(
        ["--video", "carry.mp4", "--hyrox-action", "farmers_carry"]
    )
    assert replay_args.hyrox_action == "farmers_carry"
    assert isinstance(create_analyzer("farmers_carry", "medium"), FarmersCarryAnalyzer)
