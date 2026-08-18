from __future__ import annotations

import pytest

from hyrox.base import BaseActionAnalyzer
from hyrox.config import (
    DEFAULT_OBSERVABILITY_CONFIG,
    load_observability_config,
)
from hyrox.validity import (
    BodyRuleResult,
    ObservabilityPolicy,
    RepCandidate,
    aggregate_rep_decision,
    apply_observability_policy,
)
from hyrox.view_policy import CAMERA_VIEWS, action_view_suitability


def _rule(
    status: str,
    *,
    confidence: float = 0.95,
    evidence_frames: tuple[int, ...] = (1, 2),
) -> BodyRuleResult:
    return BodyRuleResult(
        "required_rule",
        status,  # type: ignore[arg-type]
        confidence,
        reason_code=None if status == "PASS" else "RULE_FAILED",
        evidence_frames=evidence_frames,
    )


def _candidate(
    *frames: dict[str, object],
) -> RepCandidate:
    return RepCandidate(
        action="Test",
        start_frame=1,
        end_frame=len(frames),
        frames=tuple(frames),
    )


def _policy() -> ObservabilityPolicy:
    return ObservabilityPolicy.from_mapping(DEFAULT_OBSERVABILITY_CONFIG)


def test_observability_config_uses_round_nine_thresholds() -> None:
    config = load_observability_config()

    assert config["required_landmark_confidence"] == 0.60
    assert config["rep_mean_confidence"] == 0.65
    assert config["decisive_rule_confidence"] == 0.72
    assert (
        config["rep_mean_confidence_overrides"][
            "lunge__front__default"
        ]
        == 0.56
    )


def test_observability_thresholds_resolve_by_action_view_and_rule() -> None:
    policy = ObservabilityPolicy.from_mapping(
        {
            **DEFAULT_OBSERVABILITY_CONFIG,
            "rep_mean_confidence_overrides": {
                "lunge__front__default": 0.56,
                "lunge__front__full_hip_extension": 0.54,
            },
        }
    )

    hip = policy.thresholds_for(
        "lunge",
        "front",
        "full_hip_extension",
    )
    knee = policy.thresholds_for(
        "lunge",
        "front",
        "full_knee_extension",
    )
    side = policy.thresholds_for(
        "lunge",
        "side",
        "full_hip_extension",
    )

    assert hip["rep_mean_confidence"] == 0.54
    assert knee["rep_mean_confidence"] == 0.56
    assert side["rep_mean_confidence"] == 0.65


def test_low_rep_mean_confidence_downgrades_valid_to_unsure() -> None:
    raw = aggregate_rep_decision((_rule("PASS"),))
    candidate = _candidate(
        {"visible_score": 0.50},
        {"visible_score": 0.60},
    )

    decision, assessment = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
    )

    assert decision.status == "UNSURE"
    assert decision.reason_codes == ("REP_MEAN_CONFIDENCE_LOW",)
    assert assessment.rep_mean_confidence == 0.55


def test_low_required_landmark_confidence_is_unsure() -> None:
    raw = aggregate_rep_decision((_rule("FAIL"),))
    candidate = _candidate(
        {
            "visible_score": 0.95,
            "left_wrist_confidence": 0.59,
        },
        {
            "visible_score": 0.95,
            "left_wrist_confidence": 0.59,
        },
    )

    decision, assessment = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
        required_landmarks=("left_wrist",),
    )

    assert decision.status == "UNSURE"
    assert "REQUIRED_LANDMARK_CONFIDENCE_LOW" in decision.reason_codes
    assert assessment.required_landmark_confidence == 0.59


def test_one_transient_low_landmark_frame_does_not_hide_clear_evidence() -> None:
    raw = aggregate_rep_decision(
        (_rule("PASS", evidence_frames=(1, 2, 3)),)
    )
    candidate = _candidate(
        {"visible_score": 0.95, "left_wrist_confidence": 0.30},
        {"visible_score": 0.95, "left_wrist_confidence": 0.90},
        {"visible_score": 0.95, "left_wrist_confidence": 0.92},
    )

    decision, assessment = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
        required_landmarks=("left_wrist",),
    )

    assert decision.status == "VALID"
    assert assessment.required_landmark_confidence == 0.90


def test_low_decisive_rule_confidence_is_unsure() -> None:
    raw = aggregate_rep_decision(
        (_rule("FAIL", confidence=0.70),)
    )
    candidate = _candidate(
        {"visible_score": 0.95},
        {"visible_score": 0.95},
    )

    decision, _ = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
    )

    assert decision.status == "UNSURE"
    assert decision.reason_codes == ("DECISIVE_RULE_CONFIDENCE_LOW",)


def test_invalid_floor_is_unsure_but_view_remains_metadata() -> None:
    raw = aggregate_rep_decision((_rule("FAIL"),))
    candidate = _candidate(
        {
            "visible_score": 0.95,
            "floor_reference_status": "UNSURE",
        },
        {
            "visible_score": 0.95,
            "floor_reference_status": "UNSURE",
        },
    )

    decision, assessment = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
        floor_required=True,
        camera_view_suitable=False,
    )

    assert decision.status == "UNSURE"
    assert decision.reason_codes == ("FLOOR_REFERENCE_UNSURE",)
    assert assessment.floor_reference_ready is False
    assert assessment.camera_view_suitable is False
    assert assessment.as_dict()["camera_view_recommended"] is False
    assert assessment.as_dict()["camera_view_advisory_only"] is True


def test_action_view_suitability_is_recommendation_metadata() -> None:
    assert action_view_suitability("Lunge", "side") is True
    assert action_view_suitability("Lunge", "front") is True
    assert action_view_suitability("Lunge", "unknown") is None
    assert action_view_suitability("Wall Ball", "front") is True


@pytest.mark.parametrize(
    "action",
    (
        "Lunge",
        "Wall Ball",
        "Farmers Carry",
        "Rowing",
        "SkiErg",
        "Burpee Broad Jump",
        "Sled Push",
        "Sled Pull",
    ),
)
@pytest.mark.parametrize("camera_view", CAMERA_VIEWS)
def test_all_actions_keep_valid_decision_for_every_view_label(
    action: str,
    camera_view: str,
) -> None:
    raw = aggregate_rep_decision((_rule("PASS"),))
    candidate = RepCandidate(
        action=action,
        start_frame=1,
        end_frame=2,
        frames=(
            {"visible_score": 0.95},
            {"visible_score": 0.95},
        ),
    )

    decision, assessment = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
        camera_view_suitable=action_view_suitability(action, camera_view),
        action=action,
        camera_view=camera_view,
    )

    assert decision.status == "VALID"
    assert decision.reason_codes == ("required_rule",)
    assert assessment.status == "OBSERVABLE"
    assert assessment.reason_codes == ()


def test_nonrecommended_view_keeps_clear_repeated_failure_as_no_rep() -> None:
    raw = aggregate_rep_decision((_rule("FAIL", evidence_frames=(1, 2)),))
    candidate = _candidate(
        {"visible_score": 0.95},
        {"visible_score": 0.95},
    )

    decision, assessment = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
        camera_view_suitable=False,
    )

    assert decision.status == "NO_REP"
    assert decision.reason_codes == ("RULE_FAILED",)
    assert assessment.status == "OBSERVABLE"


def test_unavailable_required_metric_stays_unsure_without_view_reason() -> None:
    raw = aggregate_rep_decision(
        (
            BodyRuleResult(
                "required_rule",
                "UNSURE",
                0.0,
                reason_code="REQUIRED_LANDMARK_UNAVAILABLE",
                evidence_frames=(1, 2),
            ),
        )
    )
    candidate = _candidate(
        {"visible_score": 0.95},
        {"visible_score": 0.95},
    )

    decision, _ = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
        camera_view_suitable=False,
    )

    assert decision.status == "UNSURE"
    assert decision.reason_codes == ("REQUIRED_LANDMARK_UNAVAILABLE",)


def test_single_abnormal_frame_cannot_be_no_rep() -> None:
    raw = aggregate_rep_decision(
        (_rule("FAIL", evidence_frames=(1,)),)
    )
    candidate = _candidate({"visible_score": 0.95})

    decision, assessment = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
    )

    assert decision.status == "UNSURE"
    assert decision.reason_codes == ("SINGLE_FRAME_RULE_FAILURE",)
    assert assessment.single_frame_failure is True


def test_repeated_clear_failure_remains_no_rep() -> None:
    raw = aggregate_rep_decision(
        (_rule("FAIL", evidence_frames=(1, 2)),)
    )
    candidate = _candidate(
        {"visible_score": 0.95},
        {"visible_score": 0.95},
    )

    decision, assessment = apply_observability_policy(
        raw,
        candidate,
        policy=_policy(),
    )

    assert decision.status == "NO_REP"
    assert assessment.status == "OBSERVABLE"
    assert assessment.single_frame_failure is False


def test_base_candidate_counters_and_debug_use_observability_gate() -> None:
    analyzer = BaseActionAnalyzer(action="Test")
    analyzer.begin_frame({"visible_score": 0.95}, 100)
    decision = analyzer.register_rep_candidate(
        (_rule("FAIL", evidence_frames=(1,)),)
    )
    state = analyzer.finalize_state(
        {"action": "Test", "phase": "stand", "feedback_messages": []}
    )

    assert decision.status == "UNSURE"
    assert state["candidate_count"] == 1
    assert state["no_rep_count"] == 0
    assert state["unsure_count"] == 1
    assert state["last_rep_observability"]["status"] == "UNSURE"
    assert state["debug"]["last_rep_observability"]["reason_codes"] == [
        "SINGLE_FRAME_RULE_FAILURE"
    ]
