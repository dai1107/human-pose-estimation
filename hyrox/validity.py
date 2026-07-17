from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence


RuleStatus = Literal["PASS", "FAIL", "UNSURE", "NOT_APPLICABLE"]
DecisionStatus = Literal["VALID", "NO_REP", "UNSURE"]


def _clamp_confidence(value: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if resolved != resolved:
        return 0.0
    return max(0.0, min(1.0, resolved))


@dataclass(frozen=True)
class RepCandidate:
    action: str
    start_frame: int
    end_frame: int
    phases_seen: frozenset[str] = field(default_factory=frozenset)
    events: Mapping[str, object] = field(default_factory=dict)
    frames: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def as_dict(self, *, include_frames: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "action": self.action,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "phases_seen": sorted(self.phases_seen),
            "events": dict(self.events),
            "frame_count": len(self.frames),
        }
        if include_frames:
            result["frames"] = [dict(frame) for frame in self.frames]
        return result


@dataclass(frozen=True)
class BodyRuleResult:
    rule_id: str
    status: RuleStatus
    confidence: float
    value: float | bool | None = None
    reason_code: str | None = None
    evidence_frames: tuple[int, ...] = field(default_factory=tuple)
    required_for_count: bool = True

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL", "UNSURE", "NOT_APPLICABLE"}:
            raise ValueError(f"unsupported body rule status: {self.status}")
        object.__setattr__(self, "confidence", _clamp_confidence(self.confidence))
        object.__setattr__(
            self,
            "evidence_frames",
            tuple(int(frame) for frame in self.evidence_frames),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "confidence": self.confidence,
            "value": self.value,
            "reason_code": self.reason_code,
            "evidence_frames": list(self.evidence_frames),
            "required_for_count": self.required_for_count,
        }


@dataclass(frozen=True)
class RepDecision:
    status: DecisionStatus
    rules: tuple[BodyRuleResult, ...]
    reason_codes: tuple[str, ...]
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "rules": [rule.as_dict() for rule in self.rules],
            "reason_codes": list(self.reason_codes),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ObservabilityPolicy:
    required_landmark_confidence: float = 0.60
    rep_mean_confidence: float = 0.65
    decisive_rule_confidence: float = 0.72

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | None,
    ) -> ObservabilityPolicy:
        resolved = values or {}
        return cls(
            required_landmark_confidence=_clamp_confidence(
                resolved.get("required_landmark_confidence", 0.60)  # type: ignore[arg-type]
            ),
            rep_mean_confidence=_clamp_confidence(
                resolved.get("rep_mean_confidence", 0.65)  # type: ignore[arg-type]
            ),
            decisive_rule_confidence=_clamp_confidence(
                resolved.get("decisive_rule_confidence", 0.72)  # type: ignore[arg-type]
            ),
        )


@dataclass(frozen=True)
class ObservabilityAssessment:
    status: Literal["OBSERVABLE", "UNSURE"]
    reason_codes: tuple[str, ...]
    rep_mean_confidence: float | None
    required_landmark_confidence: float | None
    decisive_rule_confidence: float
    floor_reference_ready: bool | None
    camera_view_suitable: bool | None
    single_frame_failure: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "rep_mean_confidence": self.rep_mean_confidence,
            "required_landmark_confidence": (
                self.required_landmark_confidence
            ),
            "decisive_rule_confidence": self.decisive_rule_confidence,
            "floor_reference_ready": self.floor_reference_ready,
            "camera_view_suitable": self.camera_view_suitable,
            "single_frame_failure": self.single_frame_failure,
        }


def _safe_confidence(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if resolved != resolved:
        return None
    return max(0.0, min(1.0, resolved))


def _required_rule_results(
    decision: RepDecision,
    required_rules: Sequence[str] | None,
) -> tuple[BodyRuleResult, ...]:
    if required_rules is None:
        return tuple(
            rule for rule in decision.rules if rule.required_for_count
        )
    required_ids = tuple(dict.fromkeys(str(item) for item in required_rules))
    by_id = {rule.rule_id: rule for rule in decision.rules}
    return tuple(by_id[item] for item in required_ids if item in by_id)


def apply_observability_policy(
    decision: RepDecision,
    candidate: RepCandidate,
    *,
    policy: ObservabilityPolicy,
    required_rules: Sequence[str] | None = None,
    required_landmarks: Sequence[str] = (),
    floor_required: bool = False,
    camera_view_suitable: bool | None = None,
) -> tuple[RepDecision, ObservabilityAssessment]:
    """Downgrade otherwise decisive results when their evidence is not observable."""
    required = _required_rule_results(decision, required_rules)
    if decision.status == "NO_REP":
        decisive = tuple(rule for rule in required if rule.status == "FAIL")
    elif decision.status == "UNSURE":
        decisive = tuple(
            rule
            for rule in required
            if rule.status in {"UNSURE", "NOT_APPLICABLE"}
        )
    else:
        decisive = required

    visible_scores = tuple(
        score
        for score in (
            _safe_confidence(frame.get("visible_score"))
            for frame in candidate.frames
        )
        if score is not None
    )
    rep_mean = (
        None
        if not visible_scores
        else sum(visible_scores) / len(visible_scores)
    )

    evidence_frames = {
        frame
        for rule in decisive
        for frame in rule.evidence_frames
        if candidate.start_frame <= frame <= candidate.end_frame
    }
    if evidence_frames:
        evidence_snapshots = tuple(
            candidate.frames[frame - candidate.start_frame]
            for frame in sorted(evidence_frames)
            if 0 <= frame - candidate.start_frame < len(candidate.frames)
        )
    else:
        evidence_snapshots = candidate.frames[-1:] if candidate.frames else ()

    landmark_scores = tuple(
        confidence
        for frame in evidence_snapshots
        for name in required_landmarks
        for confidence in (
            _safe_confidence(frame.get(f"{name}_confidence")),
        )
        if confidence is not None
    )
    landmark_confidence = (
        min(landmark_scores) if landmark_scores else None
    )

    floor_statuses = tuple(
        str(frame.get("floor_reference_status"))
        for frame in evidence_snapshots
        if frame.get("floor_reference_status") is not None
    )
    floor_ready: bool | None
    if not floor_required:
        floor_ready = None
    elif not floor_statuses:
        floor_ready = False
    else:
        floor_ready = all(status == "READY" for status in floor_statuses)

    single_frame_failure = (
        decision.status == "NO_REP"
        and bool(decisive)
        and all(
            bool(rule.evidence_frames)
            and len(set(rule.evidence_frames)) == 1
            for rule in decisive
        )
    )
    reasons: list[str] = []
    if rep_mean is not None and rep_mean < policy.rep_mean_confidence:
        reasons.append("REP_MEAN_CONFIDENCE_LOW")
    if (
        landmark_confidence is not None
        and landmark_confidence < policy.required_landmark_confidence
    ):
        reasons.append("REQUIRED_LANDMARK_CONFIDENCE_LOW")
    if (
        decision.status != "UNSURE"
        and decision.confidence < policy.decisive_rule_confidence
    ):
        reasons.append("DECISIVE_RULE_CONFIDENCE_LOW")
    if camera_view_suitable is False:
        reasons.append("CAMERA_VIEW_UNSUITABLE")
    if floor_ready is False:
        reasons.append("FLOOR_REFERENCE_UNSURE")
    if single_frame_failure:
        reasons.append("SINGLE_FRAME_RULE_FAILURE")
    reason_codes = tuple(dict.fromkeys(reasons))
    assessment = ObservabilityAssessment(
        status="UNSURE" if reason_codes else "OBSERVABLE",
        reason_codes=reason_codes,
        rep_mean_confidence=rep_mean,
        required_landmark_confidence=landmark_confidence,
        decisive_rule_confidence=decision.confidence,
        floor_reference_ready=floor_ready,
        camera_view_suitable=camera_view_suitable,
        single_frame_failure=single_frame_failure,
    )
    if not reason_codes or decision.status == "UNSURE":
        return decision, assessment

    confidence_values = tuple(
        value
        for value in (
            rep_mean,
            landmark_confidence,
            decision.confidence,
        )
        if value is not None
    )
    unsure_confidence = min(confidence_values, default=0.0)
    if camera_view_suitable is False or single_frame_failure:
        unsure_confidence = min(unsure_confidence, 0.49)
    return (
        RepDecision(
            status="UNSURE",
            rules=decision.rules,
            reason_codes=reason_codes,
            confidence=unsure_confidence,
        ),
        assessment,
    )


def aggregate_rep_decision(
    rules: Sequence[BodyRuleResult],
    required_rules: Sequence[str] | None = None,
) -> RepDecision:
    """Aggregate only count-required rules; technique-only rules never gate a rep."""
    resolved_list = list(rules)
    if required_rules is None:
        required = tuple(rule for rule in resolved_list if rule.required_for_count)
    else:
        required_ids = tuple(dict.fromkeys(str(rule_id) for rule_id in required_rules))
        by_id = {rule.rule_id: rule for rule in resolved_list}
        missing = tuple(rule_id for rule_id in required_ids if rule_id not in by_id)
        for rule_id in missing:
            result = BodyRuleResult(
                rule_id=rule_id,
                status="UNSURE",
                confidence=0.0,
                reason_code="RULE_NOT_EVALUATED",
                required_for_count=True,
            )
            resolved_list.append(result)
            by_id[rule_id] = result
        required = tuple(by_id[rule_id] for rule_id in required_ids)
    resolved = tuple(resolved_list)
    if not required:
        return RepDecision(
            status="UNSURE",
            rules=resolved,
            reason_codes=("NO_REQUIRED_RULES",),
            confidence=0.0,
        )

    failed = tuple(rule for rule in required if rule.status == "FAIL")
    uncertain = tuple(
        rule for rule in required if rule.status in {"UNSURE", "NOT_APPLICABLE"}
    )
    if failed:
        status: DecisionStatus = "NO_REP"
        decisive = failed
    elif uncertain:
        status = "UNSURE"
        decisive = uncertain
    else:
        status = "VALID"
        decisive = required

    reason_codes = tuple(
        dict.fromkeys(rule.reason_code or rule.rule_id for rule in decisive)
    )
    confidence = min((rule.confidence for rule in decisive), default=0.0)
    return RepDecision(
        status=status,
        rules=resolved,
        reason_codes=reason_codes,
        confidence=confidence,
    )


__all__ = [
    "ObservabilityAssessment",
    "ObservabilityPolicy",
    "BodyRuleResult",
    "DecisionStatus",
    "RepCandidate",
    "RepDecision",
    "RuleStatus",
    "apply_observability_policy",
    "aggregate_rep_decision",
]
