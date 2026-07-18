from __future__ import annotations

from collections.abc import Iterable

import pytest

from hyrox.actions.lunge import LUNGE_REQUIRED_RULES, LungeAnalyzer
from hyrox.foot_events import FootEvent


def _pose(
    phase: str,
    *,
    trailing: str = "left",
    start_x: float = 0.40,
    contact: bool = True,
    extended: bool = True,
    observable: bool = True,
    clear_direction: bool = True,
) -> dict[str, object]:
    body_height = 0.75
    floor_y = 0.90
    movement_x = start_x if phase == "stand" else start_x + 0.04
    if clear_direction:
        centers = (
            {"left": 0.35, "right": 0.60}
            if trailing == "left"
            else {"left": 0.60, "right": 0.35}
        )
    else:
        centers = {"left": 0.45, "right": 0.45}

    if phase == "stand":
        knee_angles = {"left": 175.0, "right": 176.0}
        hip_angles = {
            "left": 171.0 if extended else 158.0,
            "right": 172.0 if extended else 160.0,
        }
        hip_y = 0.45
    elif phase == "descent":
        knee_angles = {"left": 136.0, "right": 168.0}
        hip_angles = {"left": 145.0, "right": 169.0}
        if trailing == "right":
            knee_angles = {"left": 168.0, "right": 136.0}
            hip_angles = {"left": 169.0, "right": 145.0}
        hip_y = 0.48
    elif phase == "bottom":
        knee_angles = {"left": 92.0, "right": 168.0}
        hip_angles = {"left": 132.0, "right": 170.0}
        if trailing == "right":
            knee_angles = {"left": 168.0, "right": 92.0}
            hip_angles = {"left": 170.0, "right": 132.0}
        hip_y = 0.52
    else:
        knee_angles = {"left": 136.0, "right": 168.0}
        hip_angles = {"left": 146.0, "right": 169.0}
        if trailing == "right":
            knee_angles = {"left": 168.0, "right": 136.0}
            hip_angles = {"left": 169.0, "right": 146.0}
        hip_y = 0.48

    features: dict[str, object] = {
        "visible_score": 0.95,
        "left_knee_angle": knee_angles["left"],
        "right_knee_angle": knee_angles["right"],
        "left_hip_angle": hip_angles["left"],
        "right_hip_angle": hip_angles["right"],
        "torso_angle": 5.0,
        "body_center_x": movement_x,
        "hip_center_y": hip_y,
        "body_height_norm": 0.70,
        "body_box_height_norm": body_height,
        "skeleton_height_estimate_norm": 0.72,
        "lower_body_visible_score": 0.95,
    }
    for side in ("left", "right"):
        center = centers[side]
        confidence = 0.95 if observable else 0.20
        features.update(
            {
                f"{side}_heel_x": center - 0.05,
                f"{side}_heel_y": floor_y,
                f"{side}_heel_confidence": confidence,
                f"{side}_foot_index_x": center + 0.05,
                f"{side}_foot_index_y": floor_y,
                f"{side}_foot_index_confidence": confidence,
                f"{side}_hip_x": center,
                f"{side}_hip_y": 0.48,
                f"{side}_hip_confidence": confidence,
            }
        )
        is_contact_knee = phase == "bottom" and side == trailing and contact
        if is_contact_knee:
            knee_y = (0.965 - body_height * 0.005) / 1.10
            ankle_y = 0.65
        else:
            knee_y = 0.70
            ankle_y = 0.84
        features.update(
            {
                f"{side}_knee_x": center,
                f"{side}_knee_y": knee_y,
                f"{side}_knee_confidence": confidence,
                f"{side}_ankle_x": center,
                f"{side}_ankle_y": ankle_y,
                f"{side}_ankle_confidence": confidence,
            }
        )
    return features


def _analyzer() -> LungeAnalyzer:
    analyzer = LungeAnalyzer.from_config(
        {
            "stable_frames": 1,
            "full_extension_hold_frames_medium": 2,
        }
    )
    analyzer.set_manual_floor_line((0.0, 0.90), (1.0, 0.90))
    return analyzer


def _run_cycle(
    analyzer: LungeAnalyzer,
    *,
    trailing: str = "left",
    start_x: float = 0.40,
    contact: bool = True,
    extended: bool = True,
    observable: bool = True,
    clear_direction: bool = True,
    start_ms: int = 0,
) -> dict[str, object]:
    frames: Iterable[tuple[str, int]] = (
        ("stand", start_ms),
        ("descent", start_ms + 100),
        ("bottom", start_ms + 200),
        ("bottom", start_ms + 300),
        ("bottom", start_ms + 400),
        ("ascent", start_ms + 500),
        ("stand", start_ms + 600),
        ("stand", start_ms + 700),
    )
    state: dict[str, object] = {}
    for phase, timestamp in frames:
        state = analyzer.update(
            _pose(
                phase,
                trailing=trailing,
                start_x=start_x,
                contact=contact,
                extended=extended,
                observable=observable,
                clear_direction=clear_direction,
            ),
            timestamp_ms=timestamp,
        )
    return state


def _rules(state: dict[str, object]) -> dict[str, dict[str, object]]:
    decision = state["last_rep_decision"]
    assert isinstance(decision, dict)
    return {
        str(rule["rule_id"]): rule
        for rule in decision["rules"]
    }


def test_valid_lunge_requires_all_five_rules_and_updates_valid_leg() -> None:
    analyzer = _analyzer()

    state = _run_cycle(analyzer, trailing="left")
    rules = _rules(state)

    assert state["candidate_count"] == 1
    assert state["rep_count"] == 1
    assert state["last_rep_decision"]["status"] == "VALID"
    assert tuple(rules) == LUNGE_REQUIRED_RULES
    assert all(rule["status"] == "PASS" for rule in rules.values())
    assert analyzer.previous_valid_contact_leg == "left"
    assert state["last_rep_candidate"]["events"]["trailing_leg_source"] == "movement_direction"
    assert state["last_rep_candidate"]["events"]["contact_confirmed_frame"] is not None
    assert state["last_rep_candidate"]["events"]["full_extension_confirmed_frame"] is not None
    assert state["debug"]["lunge_validation_state"] == "RULE_VALIDATION"


def test_low_confidence_no_knee_contact_is_unsure() -> None:
    state = _run_cycle(_analyzer(), contact=False)
    rules = _rules(state)

    assert state["rep_count"] == 0
    assert state["unsure_count"] == 1
    assert state["last_rep_decision"]["status"] == "UNSURE"
    assert rules["trailing_knee_contact"]["status"] == "FAIL"
    assert state["last_rep_observability"]["reason_codes"] == [
        "DECISIVE_RULE_CONFIDENCE_LOW"
    ]


def test_unobservable_knee_contact_is_unsure() -> None:
    state = _run_cycle(_analyzer(), observable=False)
    rules = _rules(state)

    assert state["rep_count"] == 0
    assert state["unsure_count"] == 1
    assert state["last_rep_decision"]["status"] == "UNSURE"
    assert rules["trailing_knee_contact"]["status"] == "UNSURE"


def test_extension_before_contact_cannot_replace_post_contact_extension() -> None:
    analyzer = _analyzer()

    state = _run_cycle(analyzer, contact=True, extended=False)
    rules = _rules(state)

    assert state["no_rep_count"] == 1
    assert rules["trailing_knee_contact"]["status"] == "PASS"
    assert rules["full_knee_extension"]["status"] == "PASS"
    assert rules["full_hip_extension"]["status"] == "FAIL"


def test_same_contact_leg_fails_and_does_not_replace_previous_valid_leg() -> None:
    analyzer = _analyzer()
    first = _run_cycle(analyzer, trailing="left", start_ms=0)
    second = _run_cycle(
        analyzer,
        trailing="left",
        start_x=0.44,
        start_ms=1000,
    )
    rules = _rules(second)

    assert first["rep_count"] == 1
    assert second["candidate_count"] == 2
    assert second["rep_count"] == 1
    assert second["no_rep_count"] == 1
    assert rules["alternating_contact_leg"]["status"] == "FAIL"
    assert analyzer.previous_valid_contact_leg == "left"


def test_opposite_contact_leg_passes_alternation() -> None:
    analyzer = _analyzer()
    _run_cycle(analyzer, trailing="left", start_ms=0)

    second = _run_cycle(
        analyzer,
        trailing="right",
        start_x=0.44,
        start_ms=1000,
    )

    assert second["rep_count"] == 2
    assert second["last_rep_decision"]["status"] == "VALID"
    assert analyzer.previous_valid_contact_leg == "right"


def test_only_valid_candidate_updates_previous_contact_leg() -> None:
    analyzer = _analyzer()
    failed = _run_cycle(
        analyzer,
        trailing="left",
        extended=False,
        start_ms=0,
    )
    valid = _run_cycle(
        analyzer,
        trailing="left",
        start_x=0.44,
        start_ms=1000,
    )

    assert failed["last_rep_decision"]["status"] == "NO_REP"
    assert valid["last_rep_decision"]["status"] == "VALID"
    assert valid["rep_count"] == 1
    assert analyzer.previous_valid_contact_leg == "left"


def test_clear_lower_knee_resolves_trailing_leg_without_direction() -> None:
    state = _run_cycle(
        _analyzer(),
        trailing="left",
        clear_direction=False,
    )
    events = state["last_rep_candidate"]["events"]

    assert state["last_rep_decision"]["status"] == "VALID"
    assert state["rep_count"] == 1
    assert events["contact_leg"] == "left"
    assert events["trailing_leg_source"] == "knee_height_override"
    assert events["trailing_leg_confidence"] >= 0.72
    assert state["last_rep_observability"]["reason_codes"] == []


def test_clear_knee_height_overrides_a_wrong_direction_side_assignment() -> None:
    analyzer = _analyzer()
    state: dict[str, object] = {}
    frames: Iterable[tuple[str, int]] = (
        ("stand", 0),
        ("descent", 100),
        ("bottom", 200),
        ("bottom", 300),
        ("bottom", 400),
        ("ascent", 500),
        ("stand", 600),
        ("stand", 700),
    )
    for phase, timestamp in frames:
        features = _pose(phase, trailing="left")
        if phase != "stand":
            # Deliberately make the toe ordering suggest the opposite side.
            features["left_foot_index_x"] = 0.70
            features["right_foot_index_x"] = 0.30
        state = analyzer.update(features, timestamp_ms=timestamp)

    events = state["last_rep_candidate"]["events"]
    assert state["last_rep_decision"]["status"] == "VALID"
    assert events["contact_leg"] == "left"
    assert events["trailing_leg_source"] == "knee_height_override"


def test_candidate_foot_observability_ignores_pre_candidate_warmup() -> None:
    analyzer = _analyzer()
    analyzer._foot_interval_observable = False

    state = _run_cycle(analyzer)

    assert state["last_rep_decision"]["status"] == "VALID"
    assert _rules(state)["no_extra_step_or_shuffle"]["status"] == "PASS"


def test_unconfirmed_warmup_stand_cannot_start_or_pollute_a_candidate() -> None:
    analyzer = LungeAnalyzer.from_config(
        {
            "stable_frames": 2,
            "full_extension_hold_frames_medium": 2,
        }
    )
    analyzer.set_manual_floor_line((0.0, 0.90), (1.0, 0.90))

    analyzer.update(_pose("stand"), timestamp_ms=0)
    analyzer.update(_pose("bottom"), timestamp_ms=100)
    analyzer.update(_pose("bottom"), timestamp_ms=200)

    assert analyzer._candidate_rule_active is False
    assert analyzer._contact_confirmed_frame is None


def test_side_view_extension_uses_the_more_observable_leg_chain() -> None:
    analyzer = _analyzer()
    analyzer.set_camera_view("side")
    state: dict[str, object] = {}
    frames: Iterable[tuple[str, int]] = (
        ("stand", 0),
        ("descent", 100),
        ("bottom", 200),
        ("bottom", 300),
        ("bottom", 400),
        ("ascent", 500),
        ("stand", 600),
        ("stand", 700),
    )
    for phase, timestamp in frames:
        features = _pose(phase, trailing="left")
        if phase == "stand" and timestamp >= 600:
            features.update(
                {
                    "left_knee_angle": 178.0,
                    "left_hip_angle": 145.0,
                    "left_hip_confidence": 0.40,
                    "left_knee_confidence": 0.40,
                    "left_ankle_confidence": 0.40,
                    "right_knee_angle": 163.0,
                    "right_hip_angle": 163.0,
                    "right_hip_confidence": 0.95,
                    "right_knee_confidence": 0.95,
                    "right_ankle_confidence": 0.95,
                }
            )
        state = analyzer.update(features, timestamp_ms=timestamp)

    assert state["last_rep_decision"]["status"] == "VALID"
    assert state["last_rep_candidate"]["events"]["extension_side"] == "right"


def test_wrong_side_or_multiple_step_events_fail_no_shuffle_rule() -> None:
    analyzer = _analyzer()
    analyzer.foot_event_detector.event_history.append(
        FootEvent("STEP", "left", 50, 1, 0.40)
    )

    state = _run_cycle(analyzer, trailing="left")
    rules = _rules(state)

    assert state["no_rep_count"] == 1
    assert rules["no_extra_step_or_shuffle"]["status"] == "FAIL"
    assert rules["no_extra_step_or_shuffle"]["reason_code"] == "EXTRA_STEP_OR_SHUFFLE"


def test_one_direct_leading_step_or_parallel_pause_is_allowed() -> None:
    paused = _run_cycle(_analyzer(), trailing="left")

    stepped_analyzer = _analyzer()
    stepped_analyzer.foot_event_detector.event_history.append(
        FootEvent("STEP", "right", 50, 1, 0.60)
    )
    stepped = _run_cycle(stepped_analyzer, trailing="left")

    assert paused["last_rep_decision"]["status"] == "VALID"
    assert stepped["last_rep_decision"]["status"] == "VALID"


def test_extension_hold_is_counted_only_after_contact() -> None:
    analyzer = _analyzer()
    frames = (
        ("stand", 0),
        ("descent", 100),
        ("bottom", 200),
        ("bottom", 300),
        ("bottom", 400),
        ("ascent", 500),
        ("stand", 600),
    )
    state: dict[str, object] = {}
    for phase, timestamp in frames:
        state = analyzer.update(_pose(phase), timestamp_ms=timestamp)

    assert state["candidate_count"] == 0
    assert state["debug"]["knee_extension_hold"] == 1

    state = analyzer.update(_pose("stand"), timestamp_ms=700)

    assert state["candidate_count"] == 1
    assert state["rep_count"] == 1
    assert state["debug"]["knee_extension_hold"] == 2


def test_missing_post_contact_hip_angles_make_extension_unsure() -> None:
    analyzer = _analyzer()
    frames = (
        ("stand", 0),
        ("descent", 100),
        ("bottom", 200),
        ("bottom", 300),
        ("bottom", 400),
        ("ascent", 500),
        ("stand", 600),
        ("stand", 700),
    )
    state: dict[str, object] = {}
    for phase, timestamp in frames:
        features = _pose(phase)
        if phase == "stand" and timestamp >= 600:
            features["left_hip_angle"] = None
            features["right_hip_angle"] = None
        state = analyzer.update(features, timestamp_ms=timestamp)

    rules = _rules(state)
    assert state["last_rep_decision"]["status"] == "UNSURE"
    assert rules["full_hip_extension"]["status"] == "UNSURE"
    assert rules["full_hip_extension"]["reason_code"] == (
        "FULL_HIP_EXTENSION_NOT_OBSERVABLE"
    )


def test_incomplete_post_contact_knee_extension_is_no_rep() -> None:
    analyzer = _analyzer()
    frames = (
        ("stand", 0),
        ("descent", 100),
        ("bottom", 200),
        ("bottom", 300),
        ("bottom", 400),
        ("ascent", 500),
        ("stand", 600),
        ("stand", 700),
    )
    state: dict[str, object] = {}
    for phase, timestamp in frames:
        features = _pose(phase)
        if phase == "stand" and timestamp >= 600:
            features["left_knee_angle"] = 160.0
            features["right_knee_angle"] = 162.0
        state = analyzer.update(features, timestamp_ms=timestamp)

    rules = _rules(state)
    assert state["last_rep_decision"]["status"] == "NO_REP"
    assert rules["full_knee_extension"]["status"] == "FAIL"
    assert rules["full_hip_extension"]["status"] == "PASS"
