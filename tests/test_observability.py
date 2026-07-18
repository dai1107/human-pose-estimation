from __future__ import annotations

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
from hyrox.view_policy import action_view_suitability


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


def test_invalid_floor_or_known_unsuitable_view_is_unsure() -> None:
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
    assert decision.reason_codes == (
        "CAMERA_VIEW_UNSUITABLE",
        "FLOOR_REFERENCE_UNSURE",
    )
    assert assessment.floor_reference_ready is False
    assert assessment.camera_view_suitable is False


def test_only_known_incompatible_camera_view_is_rejected() -> None:
    assert action_view_suitability("Lunge", "side") is True
    assert action_view_suitability("Lunge", "front") is False
    assert action_view_suitability("Lunge", "unknown") is None
    assert action_view_suitability("Wall Ball", "front") is True


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
