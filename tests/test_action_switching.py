from __future__ import annotations

import numpy as np

from hyrox.action_names import (
    HYROX_ACTION_OPTIONS,
    action_from_menu_key,
    next_hyrox_action,
)
from main import runtime_hyrox_config_path
from src.utils.draw_utils import draw_hyrox_action_selector, format_hyrox_action_selector_lines


def test_action_menu_keys_cover_off_and_all_eight_actions() -> None:
    assert tuple(action_from_menu_key(ord(str(index))) for index in range(9)) == HYROX_ACTION_OPTIONS
    assert action_from_menu_key(ord("9")) is None
    assert action_from_menu_key(ord("A")) is None


def test_next_action_cycles_without_skipping_off_state() -> None:
    sequence = ["none"]
    for _ in range(8):
        sequence.append(next_hyrox_action(sequence[-1]))

    assert tuple(sequence) == HYROX_ACTION_OPTIONS
    assert next_hyrox_action(sequence[-1]) == "none"


def test_action_selector_marks_current_action_and_renders() -> None:
    lines = format_hyrox_action_selector_lines("rowing")
    frame = np.zeros((420, 720, 3), dtype=np.uint8)

    draw_hyrox_action_selector(frame, "rowing")

    assert any(line.startswith("> 4:") for line, _ in lines)
    assert len(lines) == 11
    assert np.count_nonzero(frame) > 0


def test_explicit_config_is_only_reused_for_startup_action() -> None:
    assert runtime_hyrox_config_path(
        "lunge",
        startup_action="lunge",
        startup_config="custom_lunge.yaml",
    ) == "custom_lunge.yaml"
    assert runtime_hyrox_config_path(
        "rowing",
        startup_action="lunge",
        startup_config="custom_lunge.yaml",
    ) is None
