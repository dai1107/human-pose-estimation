from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import median
from typing import Literal, Mapping, Sequence


RuleStatus = Literal["PASS", "FAIL", "UNSURE", "NOT_APPLICABLE"]
DecisionStatus = Literal["VALID", "NO_REP", "UNSURE"]


THREE_D_ASSIST_RULE_ANGLES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "lunge": {
        "full_knee_extension": ("left_knee_angle", "right_knee_angle"),
        "full_hip_extension": ("left_hip_angle", "right_hip_angle"),
    },
    "wall_ball": {
        "tall_start": (
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
        ),
        "upward_extension": (
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
        ),
    },
    "rowing": {
        "body_sequence_valid": (
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
            "left_elbow_angle",
            "right_elbow_angle",
        ),
    },
    "skierg": {
        "body_sequence_valid": (
            "left_hip_angle",
            "right_hip_angle",
            "left_elbow_angle",
            "right_elbow_angle",
            "left_shoulder_angle",
            "right_shoulder_angle",
        ),
    },
    "sled_push": {
        "body_sequence_valid": (
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
        ),
    },
    "sled_pull": {
        "body_sequence_valid": (
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
            "left_elbow_angle",
            "right_elbow_angle",
            "left_shoulder_angle",
            "right_shoulder_angle",
        ),
    },
}

_THREE_D_FOOT_TIMING_RULES = frozenset(
    {
        "simultaneous_takeoff",
        "simultaneous_landing",
        "takeoff_stagger_proxy",
        "landing_stagger_proxy",
    }
)


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
class ThreeDAssistAssessment:
    status: Literal[
        "DISABLED",
        "SHADOW",
        "FALLBACK_2D",
        "NOT_APPLICABLE",
        "SUPPORTING",
        "CONFLICT",
    ]
    decision_mode: str
    original_status: DecisionStatus
    final_status: DecisionStatus
    confidence_before: float
    confidence_after: float
    supported_rules: tuple[str, ...] = field(default_factory=tuple)
    boosted_rules: tuple[str, ...] = field(default_factory=tuple)
    conflicted_rules: tuple[str, ...] = field(default_factory=tuple)
    relevant_angles: tuple[str, ...] = field(default_factory=tuple)
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "decision_mode": self.decision_mode,
            "original_status": self.original_status,
            "final_status": self.final_status,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "supported_rules": list(self.supported_rules),
            "boosted_rules": list(self.boosted_rules),
            "conflicted_rules": list(self.conflicted_rules),
            "relevant_angles": list(self.relevant_angles),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ObservabilityPolicy:
    required_landmark_confidence: float = 0.60
    rep_mean_confidence: float = 0.65
    decisive_rule_confidence: float = 0.72
    required_landmark_confidence_overrides: Mapping[str, float] = field(
        default_factory=dict
    )
    rep_mean_confidence_overrides: Mapping[str, float] = field(
        default_factory=dict
    )
    decisive_rule_confidence_overrides: Mapping[str, float] = field(
        default_factory=dict
    )

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
            required_landmark_confidence_overrides=cls._overrides(
                resolved.get("required_landmark_confidence_overrides")
            ),
            rep_mean_confidence_overrides=cls._overrides(
                resolved.get("rep_mean_confidence_overrides")
            ),
            decisive_rule_confidence_overrides=cls._overrides(
                resolved.get("decisive_rule_confidence_overrides")
            ),
        )

    @staticmethod
    def _overrides(value: object) -> dict[str, float]:
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key).strip().lower().replace("-", "_"): _clamp_confidence(
                threshold  # type: ignore[arg-type]
            )
            for key, threshold in value.items()
        }

    @staticmethod
    def _selector_candidates(
        action: str,
        camera_view: str,
        rule_id: str,
    ) -> tuple[str, ...]:
        action_key = action.strip().lower().replace(" ", "_").replace("-", "_")
        view_key = camera_view.strip().lower().replace("-", "_")
        rule_key = rule_id.strip().lower().replace(" ", "_").replace("-", "_")
        return (
            f"{action_key}__{view_key}__{rule_key}",
            f"{action_key}__{view_key}__default",
            f"{action_key}__default__{rule_key}",
            f"{action_key}__default__default",
            f"default__{view_key}__{rule_key}",
            f"default__{view_key}__default",
            f"default__default__{rule_key}",
        )

    def thresholds_for(
        self,
        action: str,
        camera_view: str,
        rule_id: str,
    ) -> dict[str, float]:
        def resolve(
            overrides: Mapping[str, float],
            fallback: float,
        ) -> float:
            for selector in self._selector_candidates(
                action,
                camera_view,
                rule_id,
            ):
                if selector in overrides:
                    return overrides[selector]
            return fallback

        return {
            "required_landmark_confidence": resolve(
                self.required_landmark_confidence_overrides,
                self.required_landmark_confidence,
            ),
            "rep_mean_confidence": resolve(
                self.rep_mean_confidence_overrides,
                self.rep_mean_confidence,
            ),
            "decisive_rule_confidence": resolve(
                self.decisive_rule_confidence_overrides,
                self.decisive_rule_confidence,
            ),
        }


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
    thresholds_by_rule: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )

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
            "thresholds_by_rule": {
                rule_id: dict(thresholds)
                for rule_id, thresholds in self.thresholds_by_rule.items()
            },
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
    required_landmarks_by_rule: Mapping[str, Sequence[str]] | None = None,
    floor_required: bool = False,
    camera_view_suitable: bool | None = None,
    action: str | None = None,
    camera_view: str = "unknown",
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

    action_key = str(action or candidate.action)
    thresholds_by_rule = {
        rule.rule_id: policy.thresholds_for(
            action_key,
            camera_view,
            rule.rule_id,
        )
        for rule in decisive
    }

    def snapshots_for(rule: BodyRuleResult) -> tuple[Mapping[str, object], ...]:
        evidence_frames = {
            frame
            for frame in rule.evidence_frames
            if candidate.start_frame <= frame <= candidate.end_frame
        }
        if not evidence_frames:
            return candidate.frames[-1:] if candidate.frames else ()
        return tuple(
            candidate.frames[frame - candidate.start_frame]
            for frame in sorted(evidence_frames)
            if 0 <= frame - candidate.start_frame < len(candidate.frames)
        )

    evidence_by_rule = {
        rule.rule_id: snapshots_for(rule)
        for rule in decisive
    }
    landmark_confidence_by_rule: dict[str, float | None] = {}
    for rule in decisive:
        landmarks = list(
            tuple(required_landmarks_by_rule.get(rule.rule_id, ()))
            if required_landmarks_by_rule is not None
            else tuple(required_landmarks)
        )
        selected_side: str | None = None
        if rule.rule_id in {"full_knee_extension", "full_hip_extension"}:
            selected_side = str(candidate.events.get("extension_side") or "")
        elif rule.rule_id in {
            "trailing_knee_contact",
            "alternating_contact_leg",
        }:
            selected_side = str(candidate.events.get("contact_leg") or "")
        pose_strategy = str(candidate.events.get("pose_side_strategy") or "")
        if pose_strategy.startswith("selected_"):
            selected_side = pose_strategy.removeprefix("selected_")
        if selected_side in {"left", "right"}:
            side_landmarks = [
                name
                for name in landmarks
                if name.startswith(f"{selected_side}_")
            ]
            if side_landmarks:
                landmarks = side_landmarks
        snapshots = evidence_by_rule[rule.rule_id]
        landmark_scores = tuple(
            float(median(scores))
            for name in landmarks
            for scores in (
                tuple(
                    confidence
                    for frame in snapshots
                    for confidence in (
                        _safe_confidence(frame.get(f"{name}_confidence")),
                    )
                    if confidence is not None
                ),
            )
            if scores
        )
        landmark_confidence_by_rule[rule.rule_id] = (
            min(landmark_scores) if landmark_scores else None
        )
    observed_landmark_confidences = tuple(
        value
        for value in landmark_confidence_by_rule.values()
        if value is not None
    )
    landmark_confidence = (
        min(observed_landmark_confidences)
        if observed_landmark_confidences
        else None
    )

    floor_statuses = tuple(
        str(frame.get("floor_reference_status"))
        for snapshots in evidence_by_rule.values()
        for frame in snapshots
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
    if rep_mean is not None and any(
        rep_mean < thresholds["rep_mean_confidence"]
        for thresholds in thresholds_by_rule.values()
    ):
        reasons.append("REP_MEAN_CONFIDENCE_LOW")
    if any(
        confidence is not None
        and confidence
        < thresholds_by_rule[rule_id]["required_landmark_confidence"]
        for rule_id, confidence in landmark_confidence_by_rule.items()
    ):
        reasons.append("REQUIRED_LANDMARK_CONFIDENCE_LOW")
    if decision.status != "UNSURE" and any(
        rule.confidence
        < thresholds_by_rule[rule.rule_id]["decisive_rule_confidence"]
        for rule in decisive
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
        thresholds_by_rule=thresholds_by_rule,
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


def _candidate_rule_snapshots(
    candidate: RepCandidate,
    rule: BodyRuleResult,
) -> tuple[Mapping[str, object], ...]:
    if rule.evidence_frames:
        snapshots = tuple(
            candidate.frames[frame - candidate.start_frame]
            for frame in sorted(set(rule.evidence_frames))
            if 0 <= frame - candidate.start_frame < len(candidate.frames)
        )
        if snapshots:
            return snapshots
    return candidate.frames[-1:] if candidate.frames else ()


def _unit_interval(value: object, default: float) -> float:
    resolved = _safe_confidence(value)
    return default if resolved is None else resolved


def _assist_measurement_state(
    payload: Mapping[str, object],
    angle_name: str,
) -> str:
    raw_measurements = payload.get("measurements")
    if not isinstance(raw_measurements, Mapping):
        return "unavailable"
    measurement = raw_measurements.get(angle_name)
    if not isinstance(measurement, Mapping):
        return "unavailable"
    raw_reasons = measurement.get("quality_reasons")
    reasons = (
        {str(reason) for reason in raw_reasons}
        if isinstance(raw_reasons, (list, tuple))
        else set()
    )
    # A 2D/3D disagreement is usable only when no independent quality gate
    # failed.  Low visibility, temporal jumps or invalid geometry make the
    # apparent disagreement unobservable rather than trustworthy conflict.
    if reasons == {"two_d_three_d_conflict"}:
        return "conflict"
    if bool(measurement.get("three_d_reliable")):
        return "support"
    return "unavailable"


def _experimental_body_payload(
    snapshot: Mapping[str, object],
) -> Mapping[str, object] | None:
    payload = snapshot.get("three_d_kinematics")
    if (
        not isinstance(payload, Mapping)
        or not bool(payload.get("experimental_fusion_enabled"))
        or not bool(payload.get("experimental_body_fusion_enabled", True))
    ):
        return None
    body = payload.get("body_relative")
    if not isinstance(body, Mapping) or not bool(body.get("reliable")):
        return None
    return body


def _experimental_angle_rule_state(
    candidate: RepCandidate,
    angles: Sequence[str],
) -> str | None:
    """Resolve experimental angle evidence using temporal consensus.

    ``None`` means the candidate is on the legacy/non-experimental path.
    ``disabled`` is an explicit ablation that must not map the angle rule.
    """

    saw_experiment = False
    saw_enabled = False
    support_frames = 0
    conflict_frames = 0
    thresholds: Mapping[str, object] = {}
    for snapshot in candidate.frames:
        payload = snapshot.get("three_d_kinematics")
        if (
            not isinstance(payload, Mapping)
            or not bool(payload.get("experimental_fusion_enabled"))
        ):
            continue
        saw_experiment = True
        if not bool(payload.get("experimental_angle_fusion_enabled", True)):
            continue
        saw_enabled = True
        raw_thresholds = payload.get("experimental_temporal_thresholds")
        if isinstance(raw_thresholds, Mapping):
            thresholds = raw_thresholds
        states = tuple(
            _assist_measurement_state(payload, angle_name)
            for angle_name in angles
        )
        if "conflict" in states:
            conflict_frames += 1
        elif "support" in states:
            support_frames += 1
    if not saw_experiment:
        return None
    if not saw_enabled:
        return "disabled"
    observed = support_frames + conflict_frames
    if not observed:
        return "unavailable"
    conflict_min_frames = int(
        _safe_number(thresholds.get("angle_conflict_min_frames")) or 3
    )
    conflict_min_ratio = (
        _safe_number(thresholds.get("angle_conflict_min_ratio")) or 0.35
    )
    support_min_frames = int(
        _safe_number(thresholds.get("angle_support_min_frames")) or 3
    )
    support_min_ratio = (
        _safe_number(thresholds.get("angle_support_min_ratio")) or 0.50
    )
    if (
        conflict_frames >= conflict_min_frames
        and conflict_frames / observed >= conflict_min_ratio
    ):
        return "conflict"
    if (
        support_frames >= support_min_frames
        and support_frames / observed >= support_min_ratio
    ):
        return "support"
    return "unavailable"


def _safe_number(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if resolved == resolved else None


def _body_rule_state(
    candidate: RepCandidate,
    rule: BodyRuleResult,
) -> str:
    """Compare 3D temporal evidence with a 2D rule, never create its status."""

    bodies = [
        body
        for snapshot in candidate.frames
        if (body := _experimental_body_payload(snapshot)) is not None
    ]
    if not bodies:
        return "unavailable"
    if rule.rule_id in _THREE_D_FOOT_TIMING_RULES:
        event = "takeoff" if "takeoff" in rule.rule_id else "landing"
        deltas = []
        for body in bodies:
            motion = body.get("foot_motion")
            if not isinstance(motion, Mapping):
                continue
            value = _safe_number(
                motion.get(f"{event}_time_difference_ms")
            )
            if value is not None:
                deltas.append(value)
        if not deltas:
            return "unavailable"
        thresholds = bodies[-1].get("thresholds")
        thresholds = thresholds if isinstance(thresholds, Mapping) else {}
        support_limit = _safe_number(
            thresholds.get("synchronous_event_ms")
        )
        conflict_limit = _safe_number(
            thresholds.get("conflict_event_ms")
        )
        support_limit = 120.0 if support_limit is None else support_limit
        conflict_limit = 220.0 if conflict_limit is None else conflict_limit
        delta = deltas[-1]
        three_d_pass = (
            True
            if delta <= support_limit
            else False if delta >= conflict_limit else None
        )
        if three_d_pass is None or rule.status not in {"PASS", "FAIL"}:
            return "unavailable"
        return (
            "support"
            if (rule.status == "PASS") == three_d_pass
            else "conflict"
        )
    if rule.rule_id == "no_extra_step_or_shuffle":
        counts: list[int] = []
        for body in bodies:
            motion = body.get("foot_motion")
            if not isinstance(motion, Mapping):
                continue
            resolved = [
                _safe_number(motion.get(f"{side}_{event}_count"))
                for side in ("left", "right")
                for event in ("takeoff", "landing")
            ]
            if all(value is not None for value in resolved):
                counts.append(sum(int(value) for value in resolved if value is not None))
        if len(counts) < 2 or rule.status not in {"PASS", "FAIL"}:
            return "unavailable"
        transitions = max(counts) - min(counts)
        expected_max = 4 if candidate.action == "burpee_broad_jump" else 2
        three_d_pass = (
            True
            if transitions <= expected_max
            else False if transitions >= expected_max + 2 else None
        )
        if three_d_pass is None:
            return "unavailable"
        return (
            "support"
            if (rule.status == "PASS") == three_d_pass
            else "conflict"
        )
    if rule.rule_id == "alternating_contact_leg":
        selected = str(candidate.events.get("contact_leg", "")).lower()
        hints = []
        for body in bodies:
            depth = body.get("leg_depth_order")
            if isinstance(depth, Mapping):
                hint = str(depth.get("trailing_side_hint", "")).lower()
                if hint in {"left", "right"}:
                    hints.append(hint)
        if not hints or selected not in {"left", "right"}:
            return "unavailable"
        consensus = max(set(hints), key=hints.count)
        return "support" if selected == consensus else "conflict"
    return "unavailable"


def apply_three_d_assist(
    decision: RepDecision,
    candidate: RepCandidate,
    *,
    required_rules: Sequence[str] | None = None,
) -> tuple[RepDecision, ThreeDAssistAssessment]:
    """Use reliable 3D only to qualify confidence; never replace a 2D rule status."""
    action_key = str(candidate.action).strip().lower().replace(" ", "_").replace("-", "_")
    rule_angles = THREE_D_ASSIST_RULE_ANGLES.get(action_key, {})
    original_status = decision.status
    confidence_before = decision.confidence
    rules: list[BodyRuleResult] = []
    supported_rules: list[str] = []
    boosted_rules: list[str] = []
    conflicted_rules: list[str] = []
    relevant_angles: set[str] = set()
    mapped_rule_seen = False
    saw_assist = False
    saw_shadow = False
    saw_disabled = False

    for rule in decision.rules:
        angles = rule_angles.get(rule.rule_id)
        body_state = _body_rule_state(candidate, rule)
        angle_state = (
            _experimental_angle_rule_state(candidate, angles)
            if angles
            else "disabled"
        )
        experimental_angle = angle_state is not None
        angle_mapped = bool(angles) and angle_state != "disabled"
        if not angle_mapped and body_state == "unavailable":
            rules.append(rule)
            continue
        mapped_rule_seen = True
        if angle_mapped:
            relevant_angles.update(angles or ())
        support = body_state == "support" or angle_state == "support"
        conflict = body_state == "conflict" or angle_state == "conflict"
        confidence_boost = 0.05
        conflict_cap = 0.49
        for snapshot in _candidate_rule_snapshots(candidate, rule):
            payload = snapshot.get("three_d_kinematics")
            if not isinstance(payload, Mapping):
                continue
            mode = str(payload.get("decision_mode", "shadow")).strip().lower()
            if mode != "assist":
                saw_shadow = True
                continue
            if not bool(payload.get("enabled", True)):
                saw_disabled = True
                continue
            saw_assist = True
            confidence_boost = _unit_interval(
                payload.get("assist_confidence_boost"),
                confidence_boost,
            )
            conflict_cap = _unit_interval(
                payload.get("assist_conflict_confidence_cap"),
                conflict_cap,
            )
            if not experimental_angle:
                states = tuple(
                    _assist_measurement_state(payload, angle_name)
                    for angle_name in (angles or ())
                )
                support = support or "support" in states
                conflict = conflict or "conflict" in states

        adjusted_rule = rule
        if conflict:
            conflicted_rules.append(rule.rule_id)
            adjusted_rule = replace(rule, confidence=min(rule.confidence, conflict_cap))
        elif support:
            supported_rules.append(rule.rule_id)
            adjusted_confidence = min(1.0, rule.confidence + confidence_boost)
            if adjusted_confidence > rule.confidence:
                boosted_rules.append(rule.rule_id)
            adjusted_rule = replace(rule, confidence=adjusted_confidence)
        rules.append(adjusted_rule)

    if not mapped_rule_seen:
        status = "NOT_APPLICABLE"
        decision_mode = "assist" if any(
            isinstance(frame.get("three_d_kinematics"), Mapping)
            and str(frame["three_d_kinematics"].get("decision_mode", "")).lower()
            == "assist"
            for frame in candidate.frames
        ) else "none"
        resolved = decision
        reasons: tuple[str, ...] = ()
    elif conflicted_rules:
        assisted = aggregate_rep_decision(rules, required_rules=required_rules)
        caps = tuple(
            rule.confidence
            for rule in rules
            if rule.rule_id in set(conflicted_rules)
        )
        resolved = RepDecision(
            status="UNSURE",
            rules=assisted.rules,
            reason_codes=tuple(
                dict.fromkeys(("THREE_D_ASSIST_CONFLICT", *assisted.reason_codes))
            ),
            confidence=min((assisted.confidence, *caps), default=0.0),
        )
        status = "CONFLICT"
        decision_mode = "assist"
        reasons = ("THREE_D_ASSIST_CONFLICT",)
    elif supported_rules:
        resolved = aggregate_rep_decision(rules, required_rules=required_rules)
        status = "SUPPORTING"
        decision_mode = "assist"
        reasons = ()
    else:
        resolved = decision
        reasons = ("THREE_D_ASSIST_UNAVAILABLE",)
        if saw_assist:
            status = "FALLBACK_2D"
            decision_mode = "assist"
        elif saw_disabled:
            status = "DISABLED"
            decision_mode = "assist"
        elif saw_shadow:
            status = "SHADOW"
            decision_mode = "shadow"
        else:
            status = "FALLBACK_2D"
            decision_mode = "none"

    assessment = ThreeDAssistAssessment(
        status=status,  # type: ignore[arg-type]
        decision_mode=decision_mode,
        original_status=original_status,
        final_status=resolved.status,
        confidence_before=confidence_before,
        confidence_after=resolved.confidence,
        supported_rules=tuple(dict.fromkeys(supported_rules)),
        boosted_rules=tuple(dict.fromkeys(boosted_rules)),
        conflicted_rules=tuple(dict.fromkeys(conflicted_rules)),
        relevant_angles=tuple(sorted(relevant_angles)),
        reason_codes=reasons,
    )
    return resolved, assessment


__all__ = [
    "ObservabilityAssessment",
    "ObservabilityPolicy",
    "BodyRuleResult",
    "DecisionStatus",
    "RepCandidate",
    "RepDecision",
    "RuleStatus",
    "THREE_D_ASSIST_RULE_ANGLES",
    "ThreeDAssistAssessment",
    "apply_observability_policy",
    "apply_three_d_assist",
    "aggregate_rep_decision",
]
