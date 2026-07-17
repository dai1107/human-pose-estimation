from __future__ import annotations

import numpy as np

from hyrox.base import BaseActionAnalyzer
from hyrox.validity import BodyRuleResult
from src.utils.draw_utils import (
    draw_hyrox_debug_overlay,
    format_hyrox_debug_lines,
)


def _debug_features() -> dict[str, object]:
    features: dict[str, object] = {
        "visible_score": 0.95,
        "left_knee_angle": 92.0,
        "right_knee_angle": 168.0,
        "left_hip_angle": 132.0,
        "right_hip_angle": 170.0,
        "torso_angle": 5.0,
        "body_height_norm": 0.70,
        "body_box_height_norm": 0.75,
        "lower_body_visible_score": 0.95,
    }
    for side, x in (("left", 0.42), ("right", 0.58)):
        features.update(
            {
                f"{side}_knee_x": x,
                f"{side}_knee_y": 0.875,
                f"{side}_knee_confidence": 0.95,
                f"{side}_ankle_x": x,
                f"{side}_ankle_y": 0.65,
                f"{side}_ankle_confidence": 0.95,
                f"{side}_heel_x": x - 0.04,
                f"{side}_heel_y": 0.90,
                f"{side}_heel_confidence": 0.95,
                f"{side}_foot_index_x": x + 0.04,
                f"{side}_foot_index_y": 0.90,
                f"{side}_foot_index_confidence": 0.95,
                f"{side}_hip_x": x,
                f"{side}_hip_y": 0.84,
                f"{side}_hip_confidence": 0.95,
                f"{side}_shoulder_x": x - 0.02,
                f"{side}_shoulder_y": 0.84,
                f"{side}_shoulder_confidence": 0.95,
            }
        )
    return features


def _completed_debug_state() -> tuple[dict[str, object], dict[str, object]]:
    analyzer = BaseActionAnalyzer(action="debug")
    analyzer.set_manual_floor_line((0.0, 0.90), (1.0, 0.90))
    features = _debug_features()
    analyzer.begin_frame(features, 0)
    analyzer.begin_frame(features, 100)
    analyzer.register_rep_candidate(
        (
            BodyRuleResult(
                "rear_knee_contact",
                "PASS",
                0.91,
                True,
                evidence_frames=(1, 2),
            ),
            BodyRuleResult(
                "no_extra_step",
                "FAIL",
                0.88,
                1.0,
                reason_code="EXTRA_STEP",
                evidence_frames=(1, 2),
            ),
        ),
        required_rules=("rear_knee_contact", "no_extra_step"),
    )
    state = analyzer.finalize_state(
        {
            "action": "debug",
            "phase": "bottom",
            "feedback_messages": [],
            "debug": {},
        }
    )
    return features, state


def test_round_10_debug_state_exposes_virtual_surfaces_and_rule_result() -> None:
    features, state = _completed_debug_state()
    contacts = state["debug"]["contacts"]

    assert contacts["knee"]["surface_point"] is not None
    assert contacts["chest_proxy"]["surface_point"] is not None
    assert contacts["knee"]["surface_height_ratio"] is not None
    assert contacts["chest_proxy"]["surface_height_ratio"] is not None

    lines = format_hyrox_debug_lines(
        features,
        has_pose=True,
        action_state=state,
    )

    assert any(line.startswith("floor: READY / manual") for line in lines)
    assert any(line.startswith("height knee/chest:") for line in lines)
    assert any(line.startswith("feet L/R:") for line in lines)
    assert any(line.startswith("sync takeoff/landing:") for line in lines)
    assert any(line.startswith("foot stagger:") for line in lines)
    assert any("REAR_KNEE_CONTACT" in line for line in lines)
    assert any("NO_EXTRA_STEP" in line for line in lines)
    assert "RESULT: NO_REP" in lines


def test_round_10_debug_overlay_draws_floor_and_virtual_surface_points() -> None:
    features, state = _completed_debug_state()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    line_count = draw_hyrox_debug_overlay(
        frame,
        features,
        has_pose=True,
        action_state=state,
    )

    assert line_count >= 10
    assert np.count_nonzero(frame) > 0
    for contact_name in ("knee", "chest_proxy"):
        point = state["debug"]["contacts"][contact_name]["surface_point"]
        x = min(frame.shape[1] - 1, max(0, round(point["x"] * frame.shape[1])))
        y = min(frame.shape[0] - 1, max(0, round(point["y"] * frame.shape[0])))
        roi = frame[
            max(0, y - 8) : min(frame.shape[0], y + 9),
            max(0, x - 8) : min(frame.shape[1], x + 9),
        ]
        assert np.count_nonzero(roi) > 0
