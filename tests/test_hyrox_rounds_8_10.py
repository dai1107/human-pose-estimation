from __future__ import annotations

from pathlib import Path

import pytest

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.base import BaseActionAnalyzer
from hyrox.config import (
    load_burpee_broad_jump_config,
    load_farmers_carry_config,
    load_lunge_config,
    load_rowing_config,
    load_skierg_config,
    load_sled_pull_config,
    load_sled_push_config,
    load_wall_ball_config,
)
from hyrox.registry import __all__ as registry_exports
from hyrox.registry import create_action_analyzer
from main import parse_args
from tools.replay_hyrox_video import DEBUG_CSV_COLUMNS, build_parser


LOADERS = {
    "lunge": load_lunge_config,
    "wall_ball": load_wall_ball_config,
    "farmers_carry": load_farmers_carry_config,
    "rowing": load_rowing_config,
    "skierg": load_skierg_config,
    "burpee_broad_jump": load_burpee_broad_jump_config,
    "sled_push": load_sled_push_config,
    "sled_pull": load_sled_pull_config,
}


def test_registry_exposes_one_factory_and_creates_all_actions() -> None:
    assert registry_exports == ["create_action_analyzer"]
    assert tuple(LOADERS) == HYROX_ACTION_NAMES
    for action_name in HYROX_ACTION_NAMES:
        analyzer = create_action_analyzer(action_name)
        assert isinstance(analyzer, BaseActionAnalyzer)


def test_registry_reports_unknown_action_clearly() -> None:
    with pytest.raises(ValueError, match=r"^Unknown HYROX action: unknown$"):
        create_action_analyzer("unknown")


def test_default_confirmation_is_camera_safe_without_weakening_low_sensitivity() -> None:
    offline = create_action_analyzer("lunge", sensitivity="medium")
    live = create_action_analyzer("lunge", sensitivity="medium", live_mode=True)
    live_high = create_action_analyzer("lunge", sensitivity="high", live_mode=True)
    live_low = create_action_analyzer("lunge", sensitivity="low", live_mode=True)

    assert offline.confirmation_frames == 2
    assert live.confirmation_frames == 2
    assert live_high.confirmation_frames == 1
    assert live_low.confirmation_frames == 3


def test_registry_accepts_mapping_and_user_config_path(tmp_path: Path) -> None:
    mapped = create_action_analyzer(
        "lunge",
        {
            "config_name": "mapped",
            "stable_frames": 1,
            "feedback_limits": {"max_messages": 1, "low_visibility_exclusive": False},
        },
    )
    assert mapped.config_name == "mapped"
    assert mapped.confirmation_frames == 1
    assert mapped.max_feedback_messages == 1
    assert mapped.low_visibility_exclusive is False

    path = tmp_path / "custom.yaml"
    path.write_text("config_name: custom\nstable_frames: 2\n", encoding="utf-8")
    loaded = create_action_analyzer("lunge", path)
    assert loaded.config_name == "custom"
    assert loaded.confirmation_frames == 2


def test_all_action_yaml_files_have_required_common_fields() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "hyrox"
    for action_name, loader in LOADERS.items():
        path = config_dir / f"{action_name}.yaml"
        text = path.read_text(encoding="utf-8")
        config = loader(path)
        assert config["action_name"] == action_name
        assert "visibility_min" in config
        assert "stable_frames" in config
        assert "cooldown_ms" in text
        assert config["feedback_limits"] == {
            "max_messages": 2,
            "low_visibility_exclusive": True,
        }


def test_realtime_and_replay_parsers_support_every_action() -> None:
    assert parse_args([]).hyrox_action == "none"
    for action_name in HYROX_ACTION_NAMES:
        assert parse_args(["--hyrox-action", action_name]).hyrox_action == action_name
        replay = build_parser().parse_args(["--video", "sample.mp4", "--hyrox-action", action_name])
        assert replay.hyrox_action == action_name


def test_replay_csv_contains_round_10_required_columns() -> None:
    required = {
        "frame_index",
        "timestamp_ms",
        "action",
        "phase",
        "rep_count",
        "feedback_codes",
        "visible_score",
        "torso_angle",
        "left_knee_angle",
        "right_knee_angle",
        "left_elbow_angle",
        "right_elbow_angle",
        "body_center_x",
        "body_center_y",
    }
    assert required <= set(DEBUG_CSV_COLUMNS)
