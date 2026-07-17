from __future__ import annotations

from collections.abc import Callable

import pytest

from hyrox.actions.burpee_broad_jump import (
    BURPEE_REQUIRED_RULES,
    BurpeeBroadJumpAnalyzer,
)
from hyrox.foot_events import FootEvent, FootStaggerResult


def _pose(
    phase: str,
    *,
    left_foot_height: float = 0.0,
    right_foot_height: float = 0.0,
    left_foot_x: float = 0.40,
    right_foot_x: float = 0.40,
    chest_contact: bool = True,
    wrist_forward_x: float = 0.46,
) -> dict[str, object]:
    floor_y = 0.90
    body_height_reference = 0.75
    phase_values = {
        "hands": (0.25, 0.60, 0.45, 45.0, 120.0, 105.0, 0.82),
        "chest": (0.27, 0.76, 0.25, 80.0, 150.0, 155.0, 0.86),
        "step": (0.30, 0.60, 0.50, 40.0, 110.0, 105.0, 0.65),
        "takeoff": (0.30, 0.50, 0.64, 15.0, 155.0, 155.0, 0.50),
        "flight": (0.48, 0.47, 0.65, 10.0, 165.0, 165.0, 0.50),
        "landing": (0.55, 0.52, 0.62, 15.0, 135.0, 150.0, 0.55),
        "stand": (0.55, 0.40, 0.70, 5.0, 170.0, 168.0, 0.50),
    }
    body_x, body_y, body_height, torso, knee_angle, hip_angle, wrist_y = (
        phase_values[phase]
    )
    features: dict[str, object] = {
        "visible_score": 0.95,
        "body_center_x": body_x,
        "body_center_y": body_y,
        "body_height_norm": body_height,
        "body_box_height_norm": body_height_reference,
        "skeleton_height_estimate_norm": 0.72,
        "torso_angle": torso,
        "left_knee_angle": knee_angle,
        "right_knee_angle": knee_angle + 2.0,
        "left_hip_angle": hip_angle,
        "right_hip_angle": hip_angle + 2.0,
        "left_wrist_x": wrist_forward_x - 0.03,
        "right_wrist_x": wrist_forward_x,
        "left_wrist_y": wrist_y,
        "right_wrist_y": wrist_y,
        "left_wrist_confidence": 0.95,
        "right_wrist_confidence": 0.95,
        "left_ankle_y": floor_y - left_foot_height * body_height_reference,
        "right_ankle_y": floor_y - right_foot_height * body_height_reference,
        "ankle_distance_norm": 0.18,
        "lower_body_visible_score": 0.95,
    }

    if phase == "chest":
        surface_ratio = 0.005 if chest_contact else 0.12
        torso_length = 0.15
        chest_y = floor_y - (
            surface_ratio * body_height_reference + 0.20 * torso_length
        )
        shoulder_centers = {"left": 0.40, "right": 0.45}
        hip_centers = {"left": 0.55, "right": 0.60}
        for side in ("left", "right"):
            features.update(
                {
                    f"{side}_shoulder_x": shoulder_centers[side],
                    f"{side}_shoulder_y": chest_y,
                    f"{side}_shoulder_confidence": 0.95,
                    f"{side}_hip_x": hip_centers[side],
                    f"{side}_hip_y": chest_y,
                    f"{side}_hip_confidence": 0.95,
                }
            )
        features["shoulder_center_y"] = chest_y
        features["hip_center_y"] = chest_y
    else:
        for side, x in (("left", 0.42), ("right", 0.58)):
            features.update(
                {
                    f"{side}_shoulder_x": x,
                    f"{side}_shoulder_y": 0.25,
                    f"{side}_shoulder_confidence": 0.95,
                    f"{side}_hip_x": x,
                    f"{side}_hip_y": 0.30,
                    f"{side}_hip_confidence": 0.95,
                }
            )
        features["shoulder_center_y"] = 0.25
        features["hip_center_y"] = 0.30

    for side, center_x, height in (
        ("left", left_foot_x, left_foot_height),
        ("right", right_foot_x, right_foot_height),
    ):
        foot_y = floor_y - height * body_height_reference
        features.update(
            {
                f"{side}_heel_x": center_x - 0.05,
                f"{side}_heel_y": foot_y,
                f"{side}_heel_confidence": 0.95,
                f"{side}_foot_index_x": center_x + 0.05,
                f"{side}_foot_index_y": foot_y,
                f"{side}_foot_index_confidence": 0.95,
                f"{side}_knee_x": center_x,
                f"{side}_knee_y": 0.55,
                f"{side}_knee_confidence": 0.95,
                f"{side}_ankle_x": center_x,
                f"{side}_ankle_y": 0.80,
                f"{side}_ankle_confidence": 0.95,
            }
        )
    return features


def _analyzer() -> BurpeeBroadJumpAnalyzer:
    analyzer = BurpeeBroadJumpAnalyzer.from_config(
        {
            "stable_frames": 1,
            "visibility_min": 0.55,
            "hand_placement_pass_foot_length_ratio": 1.25,
            "hand_placement_unsure_foot_length_ratio": 1.45,
            "forward_jump_min_com_displacement_leg_ratio": 0.20,
            "forward_jump_min_both_feet_displacement_leg_ratio": 0.15,
        }
    )
    analyzer.set_manual_floor_line((0.0, 0.90), (1.0, 0.90))
    return analyzer


def _run_until_landing(
    analyzer: BurpeeBroadJumpAnalyzer,
    *,
    chest_contact: bool = True,
    wrist_forward_x: float = 0.46,
) -> dict[str, object]:
    analyzer.update(
        _pose("hands", wrist_forward_x=wrist_forward_x),
        0,
    )
    for timestamp in (100, 200, 300):
        analyzer.update(
            _pose(
                "chest",
                chest_contact=chest_contact,
                wrist_forward_x=wrist_forward_x,
            ),
            timestamp,
        )
    analyzer.update(_pose("step"), 400)
    analyzer.update(
        _pose(
            "takeoff",
            left_foot_height=0.08,
            right_foot_height=0.08,
        ),
        500,
    )
    analyzer.update(
        _pose(
            "takeoff",
            left_foot_height=0.08,
            right_foot_height=0.08,
        ),
        533,
    )
    analyzer.update(
        _pose(
            "flight",
            left_foot_height=0.08,
            right_foot_height=0.08,
        ),
        650,
    )
    analyzer.update(
        _pose(
            "landing",
            left_foot_x=0.58,
            right_foot_x=0.58,
        ),
        750,
    )
    landing = analyzer.update(
        _pose(
            "landing",
            left_foot_x=0.58,
            right_foot_x=0.58,
        ),
        783,
    )
    analyzer.update(
        _pose(
            "landing",
            left_foot_x=0.58,
            right_foot_x=0.58,
        ),
        870,
    )
    return landing


def _finish_at_next_hands(
    analyzer: BurpeeBroadJumpAnalyzer,
    *,
    mutate: Callable[[BurpeeBroadJumpAnalyzer], None] | None = None,
) -> dict[str, object]:
    if mutate is not None:
        mutate(analyzer)
    analyzer.update(
        _pose("stand", left_foot_x=0.58, right_foot_x=0.58),
        900,
    )
    analyzer.update(
        _pose("stand", left_foot_x=0.58, right_foot_x=0.58),
        930,
    )
    return analyzer.update(
        _pose("hands", left_foot_x=0.58, right_foot_x=0.58),
        1000,
    )


def _run_candidate(
    *,
    chest_contact: bool = True,
    wrist_forward_x: float = 0.46,
    mutate: Callable[[BurpeeBroadJumpAnalyzer], None] | None = None,
) -> tuple[BurpeeBroadJumpAnalyzer, dict[str, object], dict[str, object]]:
    analyzer = _analyzer()
    landing = _run_until_landing(
        analyzer,
        chest_contact=chest_contact,
        wrist_forward_x=wrist_forward_x,
    )
    final = _finish_at_next_hands(analyzer, mutate=mutate)
    return analyzer, landing, final


def _rules(state: dict[str, object]) -> dict[str, dict[str, object]]:
    decision = state["last_rep_decision"]
    assert isinstance(decision, dict)
    return {
        str(rule["rule_id"]): rule
        for rule in decision["rules"]
    }


def test_valid_burpee_uses_all_eight_rules_and_waits_for_next_hands() -> None:
    analyzer, landing, final = _run_candidate()
    rules = _rules(final)

    assert landing["phase"] == "landing"
    assert landing["candidate_count"] == 0
    assert landing["debug"]["burpee_validation_state"] == "AWAITING_NEXT_HANDS"
    assert final["candidate_count"] == 1
    assert final["rep_count"] == 1
    assert final["last_rep_decision"]["status"] == "VALID"
    assert tuple(rules) == BURPEE_REQUIRED_RULES
    assert all(rule["status"] == "PASS" for rule in rules.values())
    assert final["last_rep_candidate"]["events"]["validation_boundary"] == (
        "next_hands_down"
    )
    assert analyzer._post_landing_steps == []


def test_chest_must_be_confirmed_not_merely_low_phase() -> None:
    _, _, final = _run_candidate(chest_contact=False)
    rules = _rules(final)

    assert final["rep_count"] == 0
    assert final["unsure_count"] == 1
    assert final["last_rep_decision"]["status"] == "UNSURE"
    assert rules["chest_ground_contact"]["status"] == "FAIL"


@pytest.mark.parametrize(
    ("delta_ms", "expected_decision", "expected_rule"),
    ((140, "UNSURE", "UNSURE"), (220, "NO_REP", "FAIL")),
)
def test_takeoff_sync_window_gates_candidate(
    delta_ms: int,
    expected_decision: str,
    expected_rule: str,
) -> None:
    def mutate(analyzer: BurpeeBroadJumpAnalyzer) -> None:
        left = analyzer._cycle_takeoffs["left"]
        assert left is not None
        analyzer._cycle_takeoffs["right"] = left + delta_ms

    _, _, final = _run_candidate(mutate=mutate)

    assert final["last_rep_decision"]["status"] == expected_decision
    assert _rules(final)["simultaneous_takeoff"]["status"] == expected_rule


def test_takeoff_and_landing_stagger_proxies_are_required() -> None:
    def mutate(analyzer: BurpeeBroadJumpAnalyzer) -> None:
        analyzer._takeoff_stagger = FootStaggerResult(
            "FOOT_STAGGER_PROXY",
            "FAIL",
            0.95,
            0.40,
            0.10,
        )

    _, _, final = _run_candidate(mutate=mutate)

    assert final["last_rep_decision"]["status"] == "NO_REP"
    assert _rules(final)["takeoff_stagger_proxy"]["status"] == "FAIL"


@pytest.mark.parametrize(
    ("wrist_x", "expected_decision", "expected_rule"),
    ((0.58, "UNSURE", "UNSURE"), (0.62, "UNSURE", "FAIL")),
)
def test_legal_hand_placement_proxy_uses_foot_length_thresholds(
    wrist_x: float,
    expected_decision: str,
    expected_rule: str,
) -> None:
    _, _, final = _run_candidate(wrist_forward_x=wrist_x)
    rule = _rules(final)["legal_hand_placement_proxy"]

    assert final["last_rep_decision"]["status"] == expected_decision
    assert rule["status"] == expected_rule
    assert final["last_rep_candidate"]["events"]["hand_placement_proxy_name"] == (
        "LEGAL_HAND_PLACEMENT_PROXY"
    )


def test_extra_step_after_landing_fails_but_jump_landing_steps_are_ignored() -> None:
    valid_analyzer, _, valid = _run_candidate()

    def mutate(analyzer: BurpeeBroadJumpAnalyzer) -> None:
        analyzer._post_landing_steps.append(
            FootEvent("STEP", "left", 920, 12, 0.62)
        )

    _, _, invalid = _run_candidate(mutate=mutate)

    assert valid["last_rep_decision"]["status"] == "VALID"
    assert valid_analyzer._post_landing_steps == []
    assert invalid["last_rep_decision"]["status"] == "NO_REP"
    assert _rules(invalid)["no_extra_step_or_shuffle"]["status"] == "FAIL"


def test_forward_jump_requires_com_and_both_feet_displacement() -> None:
    def mutate(analyzer: BurpeeBroadJumpAnalyzer) -> None:
        assert analyzer._takeoff_snapshot is not None
        analyzer._landing_snapshot = dict(analyzer._takeoff_snapshot)

    _, _, final = _run_candidate(mutate=mutate)

    assert final["last_rep_decision"]["status"] == "NO_REP"
    assert _rules(final)["forward_jump_detected"]["status"] == "FAIL"


def test_following_jump_must_continue_in_established_forward_direction() -> None:
    analyzer = _analyzer()
    analyzer._last_forward_direction = 1
    _run_until_landing(analyzer)

    assert analyzer._takeoff_snapshot is not None
    assert analyzer._landing_snapshot is not None
    takeoff = analyzer._takeoff_snapshot
    landing = dict(analyzer._landing_snapshot)
    for key in ("body_center_x", "left_foot_x", "right_foot_x"):
        start_x = takeoff[key]
        assert isinstance(start_x, float)
        landing[key] = start_x - 0.25
    analyzer._landing_snapshot = landing
    analyzer._observed_jump_direction = -1

    final = _finish_at_next_hands(analyzer)

    assert analyzer._cycle_expected_direction == 1
    assert final["last_rep_decision"]["status"] == "NO_REP"
    assert _rules(final)["forward_jump_detected"]["status"] == "FAIL"
    assert analyzer._last_forward_direction == 1


def test_missing_landing_evidence_is_unsure_not_silently_valid() -> None:
    def mutate(analyzer: BurpeeBroadJumpAnalyzer) -> None:
        analyzer._takeoff_snapshot = None
        analyzer._cycle_foot_observable = False

    _, _, final = _run_candidate(mutate=mutate)

    assert final["last_rep_decision"]["status"] == "UNSURE"
    assert _rules(final)["forward_jump_detected"]["status"] == "UNSURE"
