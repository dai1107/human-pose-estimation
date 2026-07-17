from __future__ import annotations

import pytest

from hyrox.base import BaseActionAnalyzer
from hyrox.foot_events import FootEventDetectorSuite


def _features(
    left_height: float = 0.0,
    right_height: float = 0.0,
    *,
    left_x: float = 0.40,
    right_x: float = 0.40,
    confidence: float = 0.95,
    left_heel_height: float | None = None,
    left_toe_height: float | None = None,
) -> dict[str, object]:
    body_height = 0.70
    floor_y = 0.90
    features: dict[str, object] = {
        "floor_reference_status": "READY",
        "floor_line_x1": 0.0,
        "floor_line_y1": floor_y,
        "floor_line_x2": 1.0,
        "floor_line_y2": floor_y,
        "body_height_reference": body_height,
    }
    heights = {
        "left": (
            left_height if left_heel_height is None else left_heel_height,
            left_height if left_toe_height is None else left_toe_height,
        ),
        "right": (right_height, right_height),
    }
    for side, center_x in (("left", left_x), ("right", right_x)):
        heel_height, toe_height = heights[side]
        features.update(
            {
                f"{side}_heel_x": center_x - 0.05,
                f"{side}_heel_y": floor_y - heel_height * body_height,
                f"{side}_heel_confidence": confidence,
                f"{side}_foot_index_x": center_x + 0.05,
                f"{side}_foot_index_y": floor_y - toe_height * body_height,
                f"{side}_foot_index_confidence": confidence,
                f"{side}_hip_x": center_x,
                f"{side}_hip_y": 0.30,
                f"{side}_hip_confidence": confidence,
                f"{side}_knee_x": center_x,
                f"{side}_knee_y": 0.55,
                f"{side}_knee_confidence": confidence,
                f"{side}_ankle_x": center_x,
                f"{side}_ankle_y": 0.80,
                f"{side}_ankle_confidence": confidence,
            }
        )
    return features


def _update(
    suite: FootEventDetectorSuite,
    features: dict[str, object],
    frame: int,
    timestamp_ms: int,
) -> dict[str, object]:
    return suite.update(
        features,
        frame_index=frame,
        timestamp_ms=timestamp_ms,
    )


def test_each_foot_moves_through_takeoff_and_landing_states() -> None:
    suite = FootEventDetectorSuite(sensitivity="medium")

    grounded = _update(suite, _features(), 1, 0)
    takeoff_candidate = _update(suite, _features(left_height=0.08), 2, 100)
    airborne = _update(suite, _features(left_height=0.08), 3, 133)
    landing_candidate = _update(suite, _features(), 4, 250)
    landed = _update(suite, _features(), 5, 283)

    assert grounded["left"].state == "GROUNDED"
    assert takeoff_candidate["left"].state == "TAKEOFF_CANDIDATE"
    assert airborne["left"].state == "AIRBORNE"
    assert [event.event_type for event in airborne["new_events"]] == ["TAKEOFF"]
    assert airborne["left"].takeoff_ms == 100
    assert landing_candidate["left"].state == "LANDING_CANDIDATE"
    assert landed["left"].state == "GROUNDED"
    assert [event.event_type for event in landed["new_events"]] == ["LANDING"]
    assert landed["left"].landing_ms == 250


def test_toe_contact_keeps_foot_grounded_when_heel_is_raised() -> None:
    suite = FootEventDetectorSuite()
    _update(suite, _features(), 1, 0)

    result = _update(
        suite,
        _features(left_heel_height=0.10, left_toe_height=0.0),
        2,
        100,
    )

    assert result["left"].state == "GROUNDED"
    assert result["left"].support_height_ratio == pytest.approx(0.0)
    assert result["new_events"] == ()


def test_transition_frames_are_consecutive_and_high_sensitivity_can_confirm_once() -> None:
    medium = FootEventDetectorSuite(sensitivity="medium")
    _update(medium, _features(), 1, 0)
    first_candidate = _update(
        medium,
        _features(left_height=0.08),
        2,
        100,
    )
    interrupted = _update(
        medium,
        _features(left_height=0.04),
        3,
        133,
    )
    second_candidate = _update(
        medium,
        _features(left_height=0.08),
        4,
        166,
    )

    high = FootEventDetectorSuite(sensitivity="high")
    _update(high, _features(), 1, 0)
    immediate = _update(high, _features(left_height=0.08), 2, 100)

    assert first_candidate["left"].state == "TAKEOFF_CANDIDATE"
    assert interrupted["left"].state == "GROUNDED"
    assert second_candidate["left"].state == "TAKEOFF_CANDIDATE"
    assert immediate["left"].state == "AIRBORNE"
    assert [event.event_type for event in immediate["new_events"]] == ["TAKEOFF"]


def _takeoff_sync(delta_ms: int) -> dict[str, object]:
    suite = FootEventDetectorSuite()
    frame = 1
    _update(suite, _features(), frame, 0)
    frame += 1
    _update(suite, _features(left_height=0.08), frame, 100)
    frame += 1
    _update(suite, _features(left_height=0.08), frame, 133)
    frame += 1
    right_start = 100 + delta_ms
    _update(
        suite,
        _features(left_height=0.08, right_height=0.08),
        frame,
        right_start,
    )
    frame += 1
    return _update(
        suite,
        _features(left_height=0.08, right_height=0.08),
        frame,
        right_start + 33,
    )


@pytest.mark.parametrize(
    ("delta_ms", "expected"),
    ((80, "PASS"), (140, "UNSURE"), (220, "FAIL")),
)
def test_takeoff_sync_uses_pass_unsure_fail_windows(
    delta_ms: int,
    expected: str,
) -> None:
    result = _takeoff_sync(delta_ms)

    assert result["sync"].takeoff_status == expected
    assert result["sync"].takeoff_delta_ms == delta_ms
    assert result["sync"].left_takeoff_ms == 100
    assert result["sync"].right_takeoff_ms == 100 + delta_ms


def _landing_sync(delta_ms: int) -> dict[str, object]:
    suite = FootEventDetectorSuite()
    _update(suite, _features(left_height=0.08, right_height=0.08), 1, 0)
    _update(suite, _features(left_height=0.0, right_height=0.08), 2, 200)
    _update(suite, _features(left_height=0.0, right_height=0.08), 3, 233)
    _update(suite, _features(), 4, 200 + delta_ms)
    return _update(suite, _features(), 5, 233 + delta_ms)


@pytest.mark.parametrize(
    ("delta_ms", "expected"),
    ((80, "PASS"), (140, "UNSURE"), (220, "FAIL")),
)
def test_landing_sync_uses_pass_unsure_fail_windows(
    delta_ms: int,
    expected: str,
) -> None:
    result = _landing_sync(delta_ms)

    assert result["sync"].landing_status == expected
    assert result["sync"].landing_delta_ms == delta_ms
    assert result["sync"].left_landing_ms == 200
    assert result["sync"].right_landing_ms == 200 + delta_ms


@pytest.mark.parametrize(
    ("offset", "expected"),
    ((0.015, "PASS"), (0.025, "UNSURE"), (0.040, "FAIL")),
)
def test_foot_stagger_proxy_is_normalized_by_mean_foot_length(
    offset: float,
    expected: str,
) -> None:
    suite = FootEventDetectorSuite()

    result = _update(
        suite,
        _features(left_x=0.40, right_x=0.40 + offset),
        1,
        0,
    )

    stagger = result["stagger"]
    assert stagger.rule_id == "FOOT_STAGGER_PROXY"
    assert stagger.status == expected
    assert stagger.mean_foot_length == pytest.approx(0.10)
    assert stagger.stagger_ratio == pytest.approx(offset / 0.10)


def _single_step(landing_x: float) -> dict[str, object]:
    suite = FootEventDetectorSuite()
    _update(suite, _features(left_x=0.30), 1, 0)
    _update(suite, _features(left_height=0.08, left_x=0.30), 2, 100)
    _update(suite, _features(left_height=0.08, left_x=0.30), 3, 133)
    _update(suite, _features(left_x=landing_x), 4, 250)
    _update(suite, _features(left_x=landing_x), 5, 283)
    return _update(suite, _features(left_x=landing_x), 6, 330)


def test_step_event_requires_airtime_ground_hold_and_leg_normalized_displacement() -> None:
    result = _single_step(0.40)
    step_events = [
        event for event in result["new_events"] if event.event_type == "STEP"
    ]

    assert result["step_event_count"] == 1
    assert len(step_events) == 1
    assert step_events[0].side == "left"
    assert step_events[0].airborne_ms == 150
    assert step_events[0].horizontal_displacement_leg_ratio == pytest.approx(0.20)
    assert step_events[0].signed_horizontal_displacement == pytest.approx(0.10)


def test_small_keypoint_jitter_does_not_create_step_event() -> None:
    result = _single_step(0.32)

    assert result["step_event_count"] == 0
    assert not any(event.event_type == "STEP" for event in result["new_events"])


def test_short_airtime_or_short_ground_hold_does_not_create_step_event() -> None:
    suite = FootEventDetectorSuite()
    _update(suite, _features(left_x=0.30), 1, 0)
    _update(suite, _features(left_height=0.08, left_x=0.30), 2, 100)
    _update(suite, _features(left_height=0.08, left_x=0.30), 3, 133)
    _update(suite, _features(left_x=0.40), 4, 150)
    landed = _update(suite, _features(left_x=0.40), 5, 183)
    settled = _update(suite, _features(left_x=0.40), 6, 270)

    assert landed["step_event_count"] == 0
    assert settled["step_event_count"] == 0

    normal = FootEventDetectorSuite()
    _update(normal, _features(left_x=0.30), 1, 0)
    _update(normal, _features(left_height=0.08, left_x=0.30), 2, 100)
    _update(normal, _features(left_height=0.08, left_x=0.30), 3, 133)
    _update(normal, _features(left_x=0.40), 4, 250)
    just_landed = _update(normal, _features(left_x=0.40), 5, 283)

    assert just_landed["step_event_count"] == 0


def test_missing_heel_or_toe_is_not_observable_and_emits_no_event() -> None:
    suite = FootEventDetectorSuite()
    missing = _features()
    missing["left_foot_index_confidence"] = 0.20

    result = _update(suite, missing, 1, 0)

    assert result["left"].observable is False
    assert result["right"].observable is True
    assert result["new_events"] == ()


def test_base_analyzer_exposes_lightweight_foot_event_debug() -> None:
    analyzer = BaseActionAnalyzer(action="test")
    analyzer.set_manual_floor_line((0.0, 0.90), (1.0, 0.90))
    features = _features()
    features.update(
        {
            "visible_score": 0.95,
            "min_knee_angle": 175.0,
            "min_hip_angle": 175.0,
            "body_height_norm": 0.70,
            "body_box_height_norm": 0.75,
            "lower_body_visible_score": 0.95,
        }
    )

    state = analyzer.update(features, timestamp_ms=0)

    assert state["debug"]["foot_events"]["left"]["state"] == "GROUNDED"
    assert features["left_foot_support_state"] == "GROUNDED"
    assert features["foot_stagger_proxy_status"] == "PASS"
