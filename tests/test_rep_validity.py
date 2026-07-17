from __future__ import annotations

from hyrox.base import BaseActionAnalyzer
from hyrox.validity import BodyRuleResult, aggregate_rep_decision


def _rule(
    rule_id: str,
    status: str,
    *,
    required: bool = True,
    reason_code: str | None = None,
) -> BodyRuleResult:
    return BodyRuleResult(
        rule_id=rule_id,
        status=status,  # type: ignore[arg-type]
        confidence=0.8,
        value=status == "PASS",
        reason_code=reason_code,
        evidence_frames=(3,),
        required_for_count=required,
    )


def test_required_rule_aggregation_maps_fail_unsure_and_pass() -> None:
    failed = aggregate_rep_decision((_rule("depth", "PASS"), _rule("extension", "FAIL")))
    uncertain = aggregate_rep_decision(
        (_rule("depth", "PASS"), _rule("contact", "UNSURE", reason_code="OCCLUDED"))
    )
    valid = aggregate_rep_decision((_rule("depth", "PASS"), _rule("extension", "PASS")))

    assert failed.status == "NO_REP"
    assert failed.reason_codes == ("extension",)
    assert uncertain.status == "UNSURE"
    assert uncertain.reason_codes == ("OCCLUDED",)
    assert valid.status == "VALID"


def test_technique_only_failure_does_not_block_counting() -> None:
    decision = aggregate_rep_decision(
        (
            _rule("body_sequence_valid", "PASS"),
            _rule("minor_knee_cave", "FAIL", required=False),
            _rule("small_asymmetry", "UNSURE", required=False),
        )
    )

    assert decision.status == "VALID"
    assert len(decision.rules) == 3


def test_candidate_counters_are_mutually_exclusive_and_rep_count_is_valid_alias() -> None:
    analyzer = BaseActionAnalyzer(action="Test")
    decisions = (
        (_rule("sequence", "PASS"),),
        (_rule("sequence", "FAIL"),),
        (_rule("sequence", "UNSURE"),),
    )
    for frame, rules in enumerate(decisions, start=1):
        analyzer.begin_frame({"visible_score": 0.9, "sample": frame})
        analyzer.observe_candidate_phase("bottom")
        analyzer.register_rep_candidate(rules, events={"sample": frame})

    state = analyzer.finalize_state({"action": "Test", "phase": "stand"})

    assert state["candidate_count"] == 3
    assert state["pose_valid_rep_count"] == 1
    assert state["no_rep_count"] == 0
    assert state["unsure_count"] == 2
    assert state["rep_count"] == 1
    assert (
        state["candidate_count"]
        == state["pose_valid_rep_count"] + state["no_rep_count"] + state["unsure_count"]
    )
    assert state["last_rep_candidate"]["events"] == {"sample": 3}  # type: ignore[index]


def test_no_required_rule_is_unsure_instead_of_silently_valid() -> None:
    decision = aggregate_rep_decision((_rule("torso_hint", "PASS", required=False),))

    assert decision.status == "UNSURE"
    assert decision.reason_codes == ("NO_REQUIRED_RULES",)


def test_declared_required_rule_missing_from_results_is_unsure() -> None:
    decision = aggregate_rep_decision(
        (_rule("body_sequence_valid", "PASS"),),
        required_rules=("body_sequence_valid", "ground_contact"),
    )

    assert decision.status == "UNSURE"
    assert decision.reason_codes == ("RULE_NOT_EVALUATED",)
    assert decision.rules[-1].rule_id == "ground_contact"
