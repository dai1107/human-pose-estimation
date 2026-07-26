"""Versioned Round 10 product contracts and additive output adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from src.configuration import ConfigValidationError, load_simple_yaml, reject_unknown_fields
from src.paths import installation_root


DEFAULT_CONTRACT_DIR = installation_root() / "configs" / "contracts"
EVIDENCE_STATUSES = frozenset(
    {"PASS", "FAIL", "UNSURE", "UNOBSERVABLE", "NOT_APPLICABLE"}
)


def _body_part_for_code(code: str) -> str:
    resolved = code.upper()
    for token, body_part in (
        ("CHEST", "chest"),
        ("HAND", "hands"),
        ("WRIST", "wrists"),
        ("HEEL", "heels"),
        ("FOOT", "feet"),
        ("FEET", "feet"),
        ("KNEE", "knees"),
        ("HIP", "hips"),
        ("ARM", "arms"),
        ("ELBOW", "elbows"),
        ("TRUNK", "trunk"),
        ("STEP", "feet"),
    ):
        if token in resolved:
            return body_part
    return "whole_body"


@dataclass(frozen=True, slots=True)
class ActionGatingContract:
    version: str
    enabled_by_default: bool
    runtime_mode: str
    model_family: str
    feature_schema_version: str
    classes: tuple[str, ...]
    enter_confidence: float
    exit_confidence: float
    unknown_confidence: float
    minimum_margin: float
    enter_duration_ms: int
    exit_duration_ms: int
    switch_cooldown_ms: int
    window_ms: int
    minimum_window_frames: int
    maximum_window_frames: int
    minimum_records_per_class: int
    group_split_key: str
    fallback_group_split_key: str
    manual_override_allowed: bool


@dataclass(frozen=True, slots=True)
class ScoringCorrectionContract:
    version: str
    rubric_version: str
    validity_priority: tuple[str, ...]
    score_min: float
    score_max: float
    max_corrections: int
    suppress_corrections_if: tuple[str, ...]
    competition_judgement_claim: str


@dataclass(frozen=True, slots=True)
class CoordinateSpacesContract:
    version: str
    normalization_version: str
    coordinate_spaces: tuple[str, ...]
    default_analysis_space: str
    fallback_space: str
    quality_gates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RealtimeLatencyContract:
    version: str
    latest_frame_only: bool
    queue_capacity: int
    max_pose_age_ms: float
    maximum_frame_gap: int
    body_prediction_horizon_ms: float
    endpoint_prediction_horizon_ms: float
    support_foot_prediction_horizon_ms: float
    analysis_uses_prediction: bool
    display_prediction_only: bool


@dataclass(frozen=True, slots=True)
class ContractBundle:
    action_gating: ActionGatingContract
    scoring_correction: ScoringCorrectionContract
    coordinate_spaces: CoordinateSpacesContract
    realtime_latency: RealtimeLatencyContract

    @property
    def versions(self) -> dict[str, str]:
        return {
            "action_gating": self.action_gating.version,
            "scoring_correction": self.scoring_correction.version,
            "coordinate_spaces": self.coordinate_spaces.version,
            "realtime_latency": self.realtime_latency.version,
        }


@dataclass(frozen=True, slots=True)
class Evidence:
    """Traceable product evidence; sources are never silently merged."""

    evidence_id: str
    source: Literal["rule", "pose", "three_d", "equipment", "temporal_model"]
    status: str
    confidence: float
    action: str
    phase: str
    body_part: str | None = None
    rule_id: str | None = None
    criterion_id: str | None = None
    value: float | bool | str | None = None
    unit: str | None = None
    frame_ids: tuple[int, ...] = ()
    observability: str = "OBSERVABLE"
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in {"rule", "pose", "three_d", "equipment", "temporal_model"}:
            raise ValueError(f"unsupported evidence source: {self.source}")
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(f"unsupported evidence status: {self.status}")
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "frame_ids", tuple(int(value) for value in self.frame_ids))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _mapping(path: Path) -> dict[str, Any]:
    values = load_simple_yaml(path)
    if values.get("schema_version") != 1:
        raise ConfigValidationError("schema_version must be 1", path=path, key="schema_version")
    return values


def _unit(value: object, *, path: Path, key: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigValidationError("must be a number in [0, 1]", path=path, key=key) from exc
    if not 0.0 <= resolved <= 1.0:
        raise ConfigValidationError("must be in [0, 1]", path=path, key=key)
    return resolved


def _positive_int(value: object, *, path: Path, key: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ConfigValidationError("must be an integer", path=path, key=key)
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigValidationError("must be an integer", path=path, key=key) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ConfigValidationError("must be an integer", path=path, key=key)
    if resolved < minimum:
        raise ConfigValidationError(f"must be >= {minimum}", path=path, key=key)
    return resolved


def _boolean(value: object, *, path: Path, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError("must be true or false", path=path, key=key)
    return value


def _nonnegative_number(value: object, *, path: Path, key: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ConfigValidationError("must be a number", path=path, key=key)
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigValidationError("must be a number", path=path, key=key) from exc
    if resolved != resolved or resolved == float("inf") or resolved == float("-inf"):
        raise ConfigValidationError("must be finite", path=path, key=key)
    if resolved < 0 or (positive and resolved <= 0):
        qualifier = "> 0" if positive else ">= 0"
        raise ConfigValidationError(f"must be {qualifier}", path=path, key=key)
    return resolved


def _string_list(value: object, *, path: Path, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ConfigValidationError("must be a non-empty list of strings", path=path, key=key)
    return tuple(dict.fromkeys(value))


def _load_action_gating(path: Path) -> ActionGatingContract:
    values = _mapping(path)
    allowed = {
        "schema_version", "contract_version", "enabled_by_default", "runtime_mode",
        "model_family", "feature_schema_version", "classes", "enter_confidence",
        "exit_confidence", "unknown_confidence", "minimum_margin", "enter_duration_ms",
        "exit_duration_ms", "switch_cooldown_ms", "window_ms", "minimum_window_frames",
        "maximum_window_frames", "minimum_records_per_class", "group_split_key",
        "fallback_group_split_key", "low_confidence_output", "manual_override_allowed",
    }
    reject_unknown_fields(values, allowed, path=path)
    classes = _string_list(values.get("classes"), path=path, key="classes")
    required = {"idle", "transition", "unknown"}
    if not required <= set(classes):
        raise ConfigValidationError("classes must include idle, transition, and unknown", path=path, key="classes")
    enter = _unit(values.get("enter_confidence"), path=path, key="enter_confidence")
    exit_threshold = _unit(values.get("exit_confidence"), path=path, key="exit_confidence")
    unknown = _unit(values.get("unknown_confidence"), path=path, key="unknown_confidence")
    if exit_threshold > enter:
        raise ConfigValidationError("must be <= enter_confidence", path=path, key="exit_confidence")
    minimum_frames = _positive_int(values.get("minimum_window_frames"), path=path, key="minimum_window_frames")
    maximum_frames = _positive_int(values.get("maximum_window_frames"), path=path, key="maximum_window_frames")
    if minimum_frames > maximum_frames:
        raise ConfigValidationError("must be <= maximum_window_frames", path=path, key="minimum_window_frames")
    return ActionGatingContract(
        version=str(values.get("contract_version")),
        enabled_by_default=_boolean(values.get("enabled_by_default"), path=path, key="enabled_by_default"),
        runtime_mode=str(values.get("runtime_mode")),
        model_family=str(values.get("model_family")),
        feature_schema_version=str(values.get("feature_schema_version")),
        classes=classes,
        enter_confidence=enter,
        exit_confidence=exit_threshold,
        unknown_confidence=unknown,
        minimum_margin=_unit(values.get("minimum_margin"), path=path, key="minimum_margin"),
        enter_duration_ms=_positive_int(values.get("enter_duration_ms"), path=path, key="enter_duration_ms", minimum=0),
        exit_duration_ms=_positive_int(values.get("exit_duration_ms"), path=path, key="exit_duration_ms", minimum=0),
        switch_cooldown_ms=_positive_int(values.get("switch_cooldown_ms"), path=path, key="switch_cooldown_ms", minimum=0),
        window_ms=_positive_int(values.get("window_ms"), path=path, key="window_ms"),
        minimum_window_frames=minimum_frames,
        maximum_window_frames=maximum_frames,
        minimum_records_per_class=_positive_int(values.get("minimum_records_per_class"), path=path, key="minimum_records_per_class"),
        group_split_key=str(values.get("group_split_key")),
        fallback_group_split_key=str(values.get("fallback_group_split_key")),
        manual_override_allowed=_boolean(values.get("manual_override_allowed"), path=path, key="manual_override_allowed"),
    )


def _load_scoring(path: Path) -> ScoringCorrectionContract:
    values = _mapping(path)
    allowed = {
        "schema_version", "contract_version", "rubric_version", "validity_priority",
        "score_min", "score_max", "max_corrections", "unobservable_score_policy",
        "uncalibrated_score_policy", "required_evidence_sources", "suppress_corrections_if",
        "competition_judgement_claim",
    }
    reject_unknown_fields(values, allowed, path=path)
    priority = _string_list(values.get("validity_priority"), path=path, key="validity_priority")
    if set(priority) != {"VALID", "NO_REP", "UNSURE"}:
        raise ConfigValidationError("must contain VALID, NO_REP, and UNSURE exactly once", path=path, key="validity_priority")
    score_min = float(values.get("score_min"))
    score_max = float(values.get("score_max"))
    if score_min >= score_max:
        raise ConfigValidationError("score_min must be < score_max", path=path)
    return ScoringCorrectionContract(
        version=str(values.get("contract_version")),
        rubric_version=str(values.get("rubric_version")),
        validity_priority=priority,
        score_min=score_min,
        score_max=score_max,
        max_corrections=_positive_int(values.get("max_corrections"), path=path, key="max_corrections"),
        suppress_corrections_if=_string_list(values.get("suppress_corrections_if"), path=path, key="suppress_corrections_if"),
        competition_judgement_claim=str(values.get("competition_judgement_claim")),
    )


def _load_coordinates(path: Path) -> CoordinateSpacesContract:
    values = _mapping(path)
    allowed = {
        "schema_version", "contract_version", "normalization_version", "coordinate_spaces",
        "default_analysis_space", "fallback_space", "metric_depth_requires_calibration",
        "metric_depth_requires_subject_identity", "world_is_metric_ground_truth",
        "require_visibility_gate", "require_bone_length_gate", "require_left_right_gate",
        "require_z_stability_gate", "require_2d_3d_consistency_gate", "require_target_identity_gate",
    }
    reject_unknown_fields(values, allowed, path=path)
    spaces = _string_list(values.get("coordinate_spaces"), path=path, key="coordinate_spaces")
    default = str(values.get("default_analysis_space"))
    fallback = str(values.get("fallback_space"))
    if default not in spaces or fallback not in spaces:
        raise ConfigValidationError("default and fallback spaces must be declared", path=path)
    gate_values = {
        key: _boolean(value, path=path, key=key)
        for key, value in values.items()
        if key.startswith("require_")
    }
    for boolean_key in (
        "metric_depth_requires_calibration",
        "metric_depth_requires_subject_identity",
        "world_is_metric_ground_truth",
    ):
        _boolean(values.get(boolean_key), path=path, key=boolean_key)
    gates = tuple(key.removeprefix("require_") for key, enabled in gate_values.items() if enabled)
    return CoordinateSpacesContract(
        version=str(values.get("contract_version")),
        normalization_version=str(values.get("normalization_version")),
        coordinate_spaces=spaces,
        default_analysis_space=default,
        fallback_space=fallback,
        quality_gates=gates,
    )


def _load_latency(path: Path) -> RealtimeLatencyContract:
    values = _mapping(path)
    allowed = {
        "schema_version", "contract_version", "latest_frame_only", "queue_capacity",
        "max_pose_age_ms", "maximum_frame_gap", "body_prediction_horizon_ms",
        "endpoint_prediction_horizon_ms", "support_foot_prediction_horizon_ms",
        "analysis_uses_prediction", "display_prediction_only", "drop_stale_pose",
        "drop_stale_action", "drop_stale_correction", "required_report_percentiles",
        "required_quality_metrics",
    }
    reject_unknown_fields(values, allowed, path=path)
    queue_capacity = _positive_int(values.get("queue_capacity"), path=path, key="queue_capacity")
    if queue_capacity != 1:
        raise ConfigValidationError("must be exactly 1 for latest-frame semantics", path=path, key="queue_capacity")
    if _boolean(values.get("analysis_uses_prediction"), path=path, key="analysis_uses_prediction"):
        raise ConfigValidationError("analysis prediction is prohibited", path=path, key="analysis_uses_prediction")
    latest_frame_only = _boolean(values.get("latest_frame_only"), path=path, key="latest_frame_only")
    display_prediction_only = _boolean(values.get("display_prediction_only"), path=path, key="display_prediction_only")
    for boolean_key in ("drop_stale_pose", "drop_stale_action", "drop_stale_correction"):
        _boolean(values.get(boolean_key), path=path, key=boolean_key)
    return RealtimeLatencyContract(
        version=str(values.get("contract_version")),
        latest_frame_only=latest_frame_only,
        queue_capacity=queue_capacity,
        max_pose_age_ms=_nonnegative_number(values.get("max_pose_age_ms"), path=path, key="max_pose_age_ms", positive=True),
        maximum_frame_gap=_positive_int(values.get("maximum_frame_gap"), path=path, key="maximum_frame_gap", minimum=0),
        body_prediction_horizon_ms=_nonnegative_number(values.get("body_prediction_horizon_ms"), path=path, key="body_prediction_horizon_ms"),
        endpoint_prediction_horizon_ms=_nonnegative_number(values.get("endpoint_prediction_horizon_ms"), path=path, key="endpoint_prediction_horizon_ms"),
        support_foot_prediction_horizon_ms=_nonnegative_number(values.get("support_foot_prediction_horizon_ms"), path=path, key="support_foot_prediction_horizon_ms"),
        analysis_uses_prediction=False,
        display_prediction_only=display_prediction_only,
    )


def load_contract_bundle(directory: str | Path | None = None) -> ContractBundle:
    root = DEFAULT_CONTRACT_DIR if directory is None else Path(directory)
    return ContractBundle(
        action_gating=_load_action_gating(root / "action_gating_v1.yaml"),
        scoring_correction=_load_scoring(root / "scoring_correction_v1.yaml"),
        coordinate_spaces=_load_coordinates(root / "coordinate_spaces_v1.yaml"),
        realtime_latency=_load_latency(root / "realtime_latency_v1.yaml"),
    )


def evidence_from_action_state(
    state: Mapping[str, object] | None,
    *,
    action: str,
    phase: str,
) -> list[dict[str, object]]:
    if not isinstance(state, Mapping):
        return []
    decision = state.get("last_rep_decision")
    if not isinstance(decision, Mapping):
        return []
    raw_rules = decision.get("rules")
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
        return []
    result: list[dict[str, object]] = []
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, Mapping):
            continue
        raw_status = str(raw.get("status", "UNSURE"))
        status = raw_status if raw_status in EVIDENCE_STATUSES else "UNSURE"
        frames = raw.get("evidence_frames")
        frame_ids = tuple(frames) if isinstance(frames, list) else ()
        reason = raw.get("reason_code")
        evidence = Evidence(
            evidence_id=f"{action}:{raw.get('rule_id', index)}:{index}",
            source="rule",
            status=status,
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            action=action,
            phase=phase,
            rule_id=str(raw.get("rule_id", "unknown")),
            value=raw.get("value"),  # type: ignore[arg-type]
            frame_ids=frame_ids,
            observability="UNSURE" if status in {"UNSURE", "NOT_APPLICABLE"} else "OBSERVABLE",
            reason_codes=() if reason in {None, ""} else (str(reason),),
        )
        result.append(evidence.as_dict())
    return result


def build_scoring_output(
    state: Mapping[str, object] | None,
    contract: ScoringCorrectionContract,
    *,
    action: str,
    phase: str,
) -> dict[str, object]:
    decision = state.get("last_rep_decision") if isinstance(state, Mapping) else None
    validity = str(decision.get("status", "UNSURE")) if isinstance(decision, Mapping) else "UNSURE"
    feedback = state.get("feedback_messages") if isinstance(state, Mapping) else None
    corrections: list[dict[str, object]] = []
    evidence = evidence_from_action_state(state, action=action, phase=phase)
    if isinstance(feedback, Sequence) and not isinstance(feedback, (str, bytes)):
        for priority, raw in enumerate(feedback[: contract.max_corrections], start=1):
            if isinstance(raw, Mapping):
                code = str(raw.get("code", "UNSPECIFIED"))
                text = str(raw.get("text", ""))
            else:
                code = str(getattr(raw, "code", "UNSPECIFIED"))
                text = str(getattr(raw, "text", ""))
            matching = [
                str(item["evidence_id"])
                for item in evidence
                if code in set(item.get("reason_codes") or ())
                or code == str(item.get("rule_id", ""))
            ]
            corrections.append(
                {
                    "correction_id": code,
                    "cue": text,
                    "priority": priority,
                    "target_phase": phase,
                    "body_part": _body_part_for_code(code),
                    "evidence_ids": matching,
                    "suppress_if": list(contract.suppress_corrections_if),
                }
            )
    return {
        "validity": validity if validity in {"VALID", "NO_REP", "UNSURE"} else "UNSURE",
        "score_breakdown": {
            name: {"score": None, "status": "UNSURE", "reason": "rubric_not_human_calibrated"}
            for name in ("completion", "control", "symmetry", "technique", "overall")
        },
        "rubric_version": contract.rubric_version,
        "rubric_calibrated": False,
        "competition_judgement_claim": contract.competition_judgement_claim,
        "corrections": corrections,
    }


def build_coordinate_output(
    contract: CoordinateSpacesContract,
    *,
    three_d_kinematics: Mapping[str, object] | None = None,
    body_canonical_landmarks: object | None = None,
    world_landmarks: object | None = None,
    camera_ray_landmarks: object | None = None,
    metric_depth_landmarks: object | None = None,
) -> dict[str, object]:
    payload = dict(three_d_kinematics or {})
    reliable_ratio = float(payload.get("three_d_reliable_ratio", 0.0) or 0.0)
    failure_reasons = payload.get("failure_reasons") or payload.get("quality_reasons") or []
    if not isinstance(failure_reasons, list):
        failure_reasons = [str(failure_reasons)]
    return {
        "coordinate_spaces": list(contract.coordinate_spaces),
        "coordinate_transform_version": contract.version,
        "normalization_version": contract.normalization_version,
        "analysis_space": contract.default_analysis_space,
        "fallback_space": contract.fallback_space,
        "body_canonical_landmarks": body_canonical_landmarks,
        "world_landmarks": world_landmarks,
        "camera_ray_landmarks": camera_ray_landmarks,
        "metric_depth_landmarks": metric_depth_landmarks,
        "three_d_confidence": max(0.0, min(1.0, reliable_ratio)),
        "three_d_failure_reasons": list(dict.fromkeys(str(item) for item in failure_reasons)),
        "quality_gates": list(contract.quality_gates),
        "fallback_to_2d": reliable_ratio <= 0.0,
    }


def build_latency_output(
    contract: RealtimeLatencyContract,
    *,
    capture_timestamp: float,
    pose_input_timestamp: float,
    pose_finished_timestamp: float,
    analysis_timestamp: float,
    render_timestamp: float,
    source_frame_id: int,
    prediction_horizon_ms: float = 0.0,
) -> dict[str, object]:
    pose_age_ms = max(0.0, (render_timestamp - capture_timestamp) * 1000.0)
    stale = pose_age_ms > contract.max_pose_age_ms
    return {
        "capture_timestamp": capture_timestamp,
        "pose_input_timestamp": pose_input_timestamp,
        "pose_finished_timestamp": pose_finished_timestamp,
        "analysis_timestamp": analysis_timestamp,
        "render_timestamp": render_timestamp,
        "pose_age_ms": pose_age_ms,
        "prediction_horizon_ms": prediction_horizon_ms,
        "source_frame_id": int(source_frame_id),
        "stale": stale,
        "analysis_uses_prediction": False,
        "display_prediction_only": contract.display_prediction_only,
        "suppress_pose": stale,
        "suppress_action": stale,
        "suppress_correction": stale,
    }


def extend_action_state(
    state: Mapping[str, object] | None,
    *,
    bundle: ContractBundle,
    action: str,
    action_source: str,
    action_gate: Mapping[str, object] | None = None,
    coordinate_output: Mapping[str, object] | None = None,
    latency_output: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Add Round 10 fields without removing or renaming any legacy field."""
    result = dict(state or {})
    phase = str(result.get("phase", "unknown"))
    result["contract_versions"] = bundle.versions
    result["action_source"] = action_source
    result["action_gating"] = dict(action_gate or {})
    evidence = evidence_from_action_state(result, action=action, phase=phase)
    debug = result.get("debug")
    if isinstance(debug, Mapping):
        visible_score = debug.get("visible_score")
        if visible_score is not None:
            confidence = max(0.0, min(1.0, float(visible_score)))
            evidence.append(
                Evidence(
                    evidence_id=f"{action}:pose_visibility",
                    source="pose",
                    status="PASS" if confidence >= 0.35 else "UNSURE",
                    confidence=confidence,
                    action=action,
                    phase=phase,
                    body_part="whole_body",
                    observability="OBSERVABLE" if confidence >= 0.35 else "UNSURE",
                    reason_codes=() if confidence >= 0.35 else ("LOW_VISIBILITY",),
                ).as_dict()
            )
    three_d = result.get("last_three_d_assist")
    if isinstance(three_d, Mapping):
        assist_status = str(three_d.get("status", "FALLBACK_2D"))
        evidence.append(
            Evidence(
                evidence_id=f"{action}:three_d_assist",
                source="three_d",
                status=(
                    "PASS" if assist_status == "SUPPORTING"
                    else "UNSURE" if assist_status == "CONFLICT"
                    else "NOT_APPLICABLE"
                ),
                confidence=float(three_d.get("confidence_after", 0.0) or 0.0),
                action=action,
                phase=phase,
                body_part="whole_body",
                observability="OBSERVABLE" if assist_status == "SUPPORTING" else "UNSURE",
                reason_codes=tuple(str(value) for value in (three_d.get("reason_codes") or [])),
            ).as_dict()
        )
    gate = dict(action_gate or {})
    if gate.get("action_source") == "auto_shadow":
        predicted = str(gate.get("predicted_action", "unknown"))
        evidence.append(
            Evidence(
                evidence_id=f"{action}:temporal_model:{gate.get('action_model_version', 'unknown')}",
                source="temporal_model",
                status="PASS" if predicted != "unknown" else "UNSURE",
                confidence=float(gate.get("action_confidence", 0.0) or 0.0),
                action=action,
                phase=phase,
                value=predicted,
                observability="OBSERVABLE" if predicted != "unknown" else "UNSURE",
                reason_codes=() if predicted != "unknown" else ("UNKNOWN_ACTION",),
            ).as_dict()
        )
    equipment_context = str(gate.get("equipment_context", "unknown"))
    if equipment_context != "unknown":
        evidence.append(
            Evidence(
                evidence_id=f"{action}:equipment:{equipment_context}",
                source="equipment",
                status="PASS",
                confidence=1.0,
                action=action,
                phase=phase,
                value=equipment_context,
            ).as_dict()
        )
    result["evidence"] = evidence
    result["scoring_correction"] = build_scoring_output(
        result, bundle.scoring_correction, action=action, phase=phase
    )
    result["coordinate_contract"] = dict(
        coordinate_output or build_coordinate_output(bundle.coordinate_spaces)
    )
    result["latency_contract"] = dict(latency_output or {})
    return result


__all__ = [
    "ActionGatingContract",
    "ContractBundle",
    "CoordinateSpacesContract",
    "Evidence",
    "RealtimeLatencyContract",
    "ScoringCorrectionContract",
    "build_coordinate_output",
    "build_latency_output",
    "build_scoring_output",
    "evidence_from_action_state",
    "extend_action_state",
    "load_contract_bundle",
]
