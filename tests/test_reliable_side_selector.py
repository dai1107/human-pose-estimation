from __future__ import annotations

import pytest

from hyrox.reliable_side import ReliableSideSelector


def _chain(
    *,
    left_confidence: float = 0.95,
    right_confidence: float = 0.70,
    left_knee: float | None = 160.0,
    right_knee: float | None = 162.0,
    camera_view: str = "unknown",
) -> dict[str, object]:
    return {
        "camera_view": camera_view,
        "visible_score": 0.90,
        "left_hip_confidence": left_confidence,
        "left_knee_confidence": left_confidence,
        "left_ankle_confidence": left_confidence,
        "right_hip_confidence": right_confidence,
        "right_knee_confidence": right_confidence,
        "right_ankle_confidence": right_confidence,
        "left_knee_angle": left_knee,
        "right_knee_angle": right_knee,
        "left_hip_angle": 165.0,
        "right_hip_angle": 166.0,
    }


def _select(
    selector: ReliableSideSelector,
    features: dict[str, object],
    *,
    preferred_side: str | None = None,
):
    return selector.select(
        features,
        required_landmarks=("hip", "knee", "ankle"),
        required_metrics=("knee_angle", "hip_angle"),
        preferred_side=preferred_side,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "camera_view",
    (
        "unknown",
        "front",
        "rear",
        "side",
        "front_left",
        "front_right",
        "oblique_front",
        "oblique_rear",
    ),
)
def test_selection_uses_evidence_not_camera_label(camera_view: str) -> None:
    selection = _select(
        ReliableSideSelector(),
        _chain(camera_view=camera_view),
    )

    assert selection.selected_side == "left"
    assert selection.left.score > selection.right.score
    assert selection.left.observable is True


def test_missing_required_metric_makes_only_that_side_unobservable() -> None:
    selection = _select(
        ReliableSideSelector(),
        _chain(left_knee=None, right_confidence=0.90),
    )

    assert selection.selected_side == "right"
    assert selection.left.observable is False
    assert selection.left.reason_codes == ("REQUIRED_METRIC_UNAVAILABLE",)
    assert selection.reason == "ONLY_OBSERVABLE_SIDE"


def test_low_confidence_on_both_sides_returns_no_selection() -> None:
    selection = _select(
        ReliableSideSelector(min_confidence=0.52),
        _chain(left_confidence=0.30, right_confidence=0.40),
    )

    assert selection.selected_side is None
    assert selection.reason == "NO_OBSERVABLE_SIDE"
    assert selection.left.reason_codes == (
        "REQUIRED_LANDMARK_CONFIDENCE_LOW",
    )


def test_preferred_role_breaks_near_tie_without_redefining_leg_identity() -> None:
    selection = _select(
        ReliableSideSelector(switch_margin=0.08),
        _chain(left_confidence=0.90, right_confidence=0.90),
        preferred_side="right",
    )

    assert selection.selected_side == "right"
    assert selection.reason == "PREFERRED_SIDE_TIE_BREAK"


def test_switch_requires_consecutive_advantage_while_current_side_is_visible() -> None:
    selector = ReliableSideSelector(
        switch_margin=0.08,
        switch_confirmation_frames=2,
    )
    first = _select(
        selector,
        _chain(left_confidence=0.95, right_confidence=0.60),
    )
    pending = _select(
        selector,
        _chain(left_confidence=0.70, right_confidence=0.98),
    )
    switched = _select(
        selector,
        _chain(left_confidence=0.70, right_confidence=0.98),
    )

    assert first.selected_side == "left"
    assert pending.selected_side == "left"
    assert pending.reason == "SWITCH_PENDING"
    assert pending.pending_side == "right"
    assert switched.selected_side == "right"
    assert switched.reason == "CONFIRMED_RELIABILITY_SWITCH"
    assert switched.switch_count == 1


def test_unobservable_current_side_fails_over_immediately() -> None:
    selector = ReliableSideSelector(switch_confirmation_frames=3)
    _select(selector, _chain(left_confidence=0.95, right_confidence=0.60))

    failed_over = _select(
        selector,
        _chain(
            left_confidence=0.95,
            right_confidence=0.90,
            left_knee=None,
        ),
    )

    assert failed_over.selected_side == "right"
    assert failed_over.reason == "CURRENT_SIDE_UNOBSERVABLE_FAILOVER"


def test_reset_clears_hysteresis_state() -> None:
    selector = ReliableSideSelector()
    _select(selector, _chain())
    selector.reset()

    assert selector.current_side is None
    assert selector.last_selection is None
    assert selector.switch_count == 0
