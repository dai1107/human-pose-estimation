from __future__ import annotations

from collections.abc import Mapping, Sequence

from .config import load_observability_config
from .contact import ContactDetectorSuite
from .feedback import FeedbackMessage
from .floor_reference import FloorReferenceResult, LocalFloorReference
from .foot_events import FootEventDetectorSuite
from .validity import (
    BodyRuleResult,
    ObservabilityAssessment,
    ObservabilityPolicy,
    RepCandidate,
    RepDecision,
    ThreeDAssistAssessment,
    aggregate_rep_decision,
    apply_observability_policy,
    apply_three_d_assist,
)
from .view_policy import (
    action_view_suitability,
    filter_feedback_for_view,
    normalize_camera_view,
    view_profile,
)


TRANSIENT_PHASES = frozenset({"unknown", "no_pose", "low_visibility"})
_FLOOR_REQUIRED_ACTIONS = frozenset(
    {"lunge", "burpee_broad_jump", "wall_ball"}
)
_REQUIRED_LANDMARKS: dict[str, tuple[str, ...]] = {
    "lunge": (
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ),
    "burpee_broad_jump": (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_wrist",
        "right_wrist",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ),
    "wall_ball": (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_wrist",
        "right_wrist",
    ),
}


class PhaseSequenceTracker:
    """Track an ordered movement cycle without throwing away valid progress.

    Pose estimators regularly emit one-frame gaps or skip a short transition at
    the fastest part of a movement.  Those events should not invalidate an
    already observed endpoint.  Callers can therefore mark transition-only
    phases as optional while the biomechanically meaningful endpoints remain
    mandatory.
    """

    def __init__(
        self,
        sequence: Sequence[str],
        *,
        optional_phases: Sequence[str] = (),
        ignored_phases: Sequence[str] = tuple(TRANSIENT_PHASES),
    ) -> None:
        self.sequence = tuple(str(phase) for phase in sequence)
        if len(self.sequence) < 2:
            raise ValueError("phase sequence must contain at least two phases")
        self.optional_phases = frozenset(str(phase) for phase in optional_phases)
        unknown_optional = self.optional_phases.difference(self.sequence)
        if unknown_optional:
            raise ValueError(f"optional phases are not in sequence: {sorted(unknown_optional)}")
        if self.sequence[0] in self.optional_phases or self.sequence[-1] in self.optional_phases:
            raise ValueError("the first and terminal phases cannot be optional")
        self.ignored_phases = frozenset(str(phase) for phase in ignored_phases)
        self.reset()

    def reset(self) -> None:
        self.progress = 0
        self.observed: list[str] = []
        self.skipped_optional: list[str] = []
        self.last_ignored_phase: str | None = None
        self.just_completed = False

    def _consume(self, phase: str, index: int) -> bool:
        if index > self.progress:
            self.skipped_optional.extend(self.sequence[self.progress:index])
        self.observed.append(phase)
        self.progress = index + 1
        if self.progress != len(self.sequence):
            return False

        self.just_completed = True
        if self.sequence[-1] == self.sequence[0]:
            self.progress = 1
            self.observed = [phase]
        else:
            self.progress = 0
            self.observed = []
        self.skipped_optional = []
        return True

    def update(self, phase: str) -> bool:
        """Consume a new stable phase and return True only at full-cycle completion."""
        phase = str(phase)
        self.just_completed = False
        if phase in self.ignored_phases:
            self.last_ignored_phase = phase
            return False
        self.last_ignored_phase = None

        if self.progress > 0 and phase == self.sequence[self.progress - 1]:
            return False

        expected = self.sequence[self.progress]
        if phase == expected:
            return self._consume(phase, self.progress)

        try:
            later_index = self.sequence.index(phase, self.progress + 1)
        except ValueError:
            later_index = -1
        if later_index >= 0 and all(
            skipped in self.optional_phases for skipped in self.sequence[self.progress:later_index]
        ):
            return self._consume(phase, later_index)

        if phase == self.sequence[0]:
            self.progress = 1
            self.observed = [phase]
            self.skipped_optional = []
        return False

    def debug(self) -> dict[str, object]:
        return {
            "required_phase_sequence": list(self.sequence),
            "observed_phase_sequence": list(self.observed),
            "phase_sequence_progress": self.progress,
            "optional_phase_sequence": sorted(self.optional_phases),
            "skipped_optional_phases": list(self.skipped_optional),
            "last_ignored_phase": self.last_ignored_phase,
            "rep_completed": self.just_completed,
        }


def _feature_score(features: dict[str, object] | None) -> float:
    if not isinstance(features, dict):
        return 0.0
    try:
        value = float(features.get("visible_score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


class BaseActionAnalyzer:
    def __init__(self, action: str = "unknown", min_visible_score: float = 0.35) -> None:
        self.action = action
        self.min_visible_score = max(0.0, min(1.0, float(min_visible_score)))
        self.observability_policy = ObservabilityPolicy.from_mapping(
            load_observability_config()
        )
        self.max_feedback_messages = 2
        self.low_visibility_exclusive = True
        self.tracking_loss_grace_frames = 3
        self.camera_view = "unknown"
        self.reset()

    def _advance_confirmed_phase(
        self,
        raw_phase: str,
        confirmation_frames: int,
    ) -> tuple[str, str]:
        """Debounce a phase while preserving state across brief pose dropouts."""
        previous = str(getattr(self, "stable_phase", "unknown"))
        current_raw = str(getattr(self, "raw_phase", "unknown"))
        frames = int(getattr(self, "frames_in_phase", 0))
        raw_phase = str(raw_phase)
        if raw_phase == current_raw:
            frames += 1
        else:
            current_raw = raw_phase
            frames = 1

        threshold = max(1, int(confirmation_frames))
        if raw_phase in TRANSIENT_PHASES and previous not in TRANSIENT_PHASES:
            # A person can be briefly occluded at the bottom or terminal pose.
            # Hold the last confirmed phase long enough for the detector to
            # recover instead of breaking the in-progress repetition.
            threshold = max(threshold, self.tracking_loss_grace_frames + 1)
        if frames >= threshold:
            self.stable_phase = raw_phase

        self.raw_phase = current_raw
        self.frames_in_phase = frames
        self.phase = self.stable_phase
        return previous, str(self.stable_phase)

    def set_camera_view(self, camera_view: str) -> None:
        self.camera_view = normalize_camera_view(camera_view)

    @property
    def camera_view_profile(self) -> str:
        return view_profile(self.camera_view)

    def configure_feedback_limits(self, config: Mapping[str, object] | None) -> None:
        limits = config.get("feedback_limits") if isinstance(config, Mapping) else None
        if not isinstance(limits, Mapping):
            return
        try:
            self.max_feedback_messages = max(0, int(limits.get("max_messages", 2)))
        except (TypeError, ValueError, OverflowError):
            self.max_feedback_messages = 2
        value = limits.get("low_visibility_exclusive", True)
        self.low_visibility_exclusive = value if isinstance(value, bool) else True

    def limit_feedback(self, messages: Sequence[FeedbackMessage]) -> list[FeedbackMessage]:
        resolved = list(messages)
        if self.low_visibility_exclusive:
            low_visibility = [message for message in resolved if message.code.upper() == "LOW_VISIBILITY"]
            if low_visibility:
                return low_visibility[: self.max_feedback_messages]
        resolved, _ = filter_feedback_for_view(self.action, self.camera_view, resolved)
        return resolved[: self.max_feedback_messages]

    def attach_view_context(self, state: dict[str, object]) -> dict[str, object]:
        self.finalize_state(state)
        debug = state.get("debug")
        if not isinstance(debug, dict):
            debug = {}
            state["debug"] = debug
        debug["camera_view"] = self.camera_view
        debug["view_profile"] = self.camera_view_profile
        messages = state.get("feedback_messages")
        if self.camera_view_profile == "unknown" and isinstance(messages, list) and not messages:
            messages.append(
                FeedbackMessage(
                    level="info",
                    code="CAMERA_VIEW_REQUIRED",
                    text="请选择正面或侧面视角，以启用对应评价标准",
                    confidence=1.0,
                )
            )
        return state

    def reset(self) -> None:
        self.phase = "idle"
        self.rep_count = 0
        self.candidate_count = 0
        self.pose_valid_rep_count = 0
        self.no_rep_count = 0
        self.unsure_count = 0
        self.frame_index = 0
        self.last_rep_candidate: RepCandidate | None = None
        self.last_rep_decision: RepDecision | None = None
        self.last_observability_assessment: (
            ObservabilityAssessment | None
        ) = None
        self.last_three_d_assist_assessment: ThreeDAssistAssessment | None = None
        self._candidate_completed_frame: int | None = None
        self._candidate_start_frame = 1
        self._candidate_frames: list[Mapping[str, object]] = []
        self._candidate_phases: set[str] = set()
        floor_action = str(self.action).strip().lower().replace(" ", "_")
        self.floor_reference = LocalFloorReference(
            allow_supported_pose_calibration=floor_action in {"lunge", "wall_ball"}
        )
        self.last_floor_reference: FloorReferenceResult = self.floor_reference.last_result
        self.last_timestamp_ms: int | None = None
        self.contact_detectors = ContactDetectorSuite(
            sensitivity=str(getattr(self, "sensitivity", "medium"))
        )
        self.foot_event_detector = FootEventDetectorSuite(
            sensitivity=str(getattr(self, "sensitivity", "medium"))
        )
        self._current_features: dict[str, object] = {}
        self._contacts_updated_frame: int | None = None
        self._foot_events_updated_frame: int | None = None
        self._last_foot_events_update: dict[str, object] = {}

    def begin_frame(
        self,
        features: Mapping[str, object] | None,
        timestamp_ms: int | None = None,
    ) -> int:
        """Start one analyzer frame and retain a bounded candidate evidence buffer."""
        self.frame_index += 1
        self.last_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms)
        floor_features = features if isinstance(features, dict) else dict(features or {})
        self.last_floor_reference = self.floor_reference.enrich_features(
            floor_features,
            timestamp_ms=timestamp_ms,
            frame_index=self.frame_index,
        )
        desired_sensitivity = str(getattr(self, "sensitivity", "medium"))
        if desired_sensitivity != self.contact_detectors.sensitivity:
            self.contact_detectors = ContactDetectorSuite(sensitivity=desired_sensitivity)
        if desired_sensitivity != self.foot_event_detector.sensitivity:
            self.foot_event_detector = FootEventDetectorSuite(
                sensitivity=desired_sensitivity
            )
        self._current_features = floor_features
        self._contacts_updated_frame = None
        self._foot_events_updated_frame = None
        snapshot = {
            key: value
            for key, value in floor_features.items()
            if not str(key).startswith("_")
        }
        self._candidate_frames.append(snapshot)
        if len(self._candidate_frames) > 900:
            self._candidate_frames.pop(0)
            self._candidate_start_frame = max(
                self._candidate_start_frame + 1,
                self.frame_index - len(self._candidate_frames) + 1,
            )
        return self.frame_index

    def set_manual_floor_line(
        self,
        point1: object | None,
        point2: object | None,
    ) -> None:
        self.floor_reference.set_manual_line(point1, point2)

    def observe_candidate_phase(self, phase: str) -> None:
        resolved = str(phase)
        if resolved not in TRANSIENT_PHASES and resolved not in {"idle", "reset"}:
            self._candidate_phases.add(resolved)

    def register_rep_candidate(
        self,
        rules: Sequence[BodyRuleResult],
        *,
        required_rules: Sequence[str] | None = None,
        phases_seen: Sequence[str] = (),
        events: Mapping[str, object] | None = None,
    ) -> RepDecision:
        """Finalize one completed movement cycle and update mutually exclusive counts."""
        observed_phases = frozenset((*self._candidate_phases, *(str(item) for item in phases_seen)))
        candidate = RepCandidate(
            action=self.action,
            start_frame=min(self._candidate_start_frame, self.frame_index),
            end_frame=self.frame_index,
            phases_seen=observed_phases,
            events=dict(events or {}),
            frames=tuple(self._candidate_frames),
        )
        decision = aggregate_rep_decision(rules, required_rules=required_rules)
        decision, three_d_assist = apply_three_d_assist(
            decision,
            candidate,
            required_rules=required_rules,
        )
        action_key = self.action.strip().lower().replace(" ", "_")
        decision, observability = apply_observability_policy(
            decision,
            candidate,
            policy=self.observability_policy,
            required_rules=required_rules,
            required_landmarks=_REQUIRED_LANDMARKS.get(action_key, ()),
            floor_required=action_key in _FLOOR_REQUIRED_ACTIONS,
            camera_view_suitable=action_view_suitability(
                self.action,
                self.camera_view,
            ),
        )
        self.candidate_count += 1
        if decision.status == "VALID":
            self.pose_valid_rep_count += 1
        elif decision.status == "NO_REP":
            self.no_rep_count += 1
        else:
            self.unsure_count += 1
        self.rep_count = self.pose_valid_rep_count
        self.last_rep_candidate = candidate
        self.last_rep_decision = decision
        self.last_observability_assessment = observability
        self.last_three_d_assist_assessment = three_d_assist
        self._candidate_completed_frame = self.frame_index
        self._candidate_start_frame = self.frame_index + 1
        self._candidate_frames = []
        self._candidate_phases = set()
        return decision

    def register_completed_sequence(
        self,
        *,
        confidence: float,
        phases_seen: Sequence[str] = (),
        events: Mapping[str, object] | None = None,
    ) -> RepDecision:
        """First-round count gate for an already confirmed ordered body sequence."""
        terminal_phase = None if events is None else events.get("terminal_phase")
        if terminal_phase is not None:
            self.observe_candidate_phase(str(terminal_phase))
        return self.register_rep_candidate(
            (
                BodyRuleResult(
                    rule_id="body_sequence_valid",
                    status="PASS",
                    confidence=confidence,
                    value=True,
                    evidence_frames=(self.frame_index,),
                    required_for_count=True,
                ),
            ),
            required_rules=("body_sequence_valid",),
            phases_seen=phases_seen,
            events=events,
        )

    def rep_summary(self) -> dict[str, object]:
        return {
            "rep_count": self.pose_valid_rep_count,
            "candidate_count": self.candidate_count,
            "pose_valid_rep_count": self.pose_valid_rep_count,
            "no_rep_count": self.no_rep_count,
            "unsure_count": self.unsure_count,
            "last_rep_candidate": (
                None if self.last_rep_candidate is None else self.last_rep_candidate.as_dict()
            ),
            "last_rep_decision": (
                None if self.last_rep_decision is None else self.last_rep_decision.as_dict()
            ),
            "last_rep_observability": (
                None
                if self.last_observability_assessment is None
                else self.last_observability_assessment.as_dict()
            ),
            "last_three_d_assist": (
                None
                if self.last_three_d_assist_assessment is None
                else self.last_three_d_assist_assessment.as_dict()
            ),
        }

    def update_foot_events_for_current_frame(self) -> dict[str, object]:
        if self._foot_events_updated_frame == self.frame_index:
            return self._last_foot_events_update
        foot_events = self.foot_event_detector.update(
            self._current_features,
            frame_index=self.frame_index,
            timestamp_ms=self.last_timestamp_ms,
        )
        self._last_foot_events_update = foot_events
        left_foot = foot_events["left"]
        right_foot = foot_events["right"]
        sync = foot_events["sync"]
        stagger = foot_events["stagger"]
        self._current_features.update(
            {
                "left_foot_support_state": left_foot.state,
                "right_foot_support_state": right_foot.state,
                "left_takeoff_ms": sync.left_takeoff_ms,
                "right_takeoff_ms": sync.right_takeoff_ms,
                "left_landing_ms": sync.left_landing_ms,
                "right_landing_ms": sync.right_landing_ms,
                "takeoff_sync_status": sync.takeoff_status,
                "landing_sync_status": sync.landing_status,
                "foot_stagger_proxy_status": stagger.status,
                "foot_stagger_proxy_ratio": stagger.stagger_ratio,
                "step_event_count": foot_events["step_event_count"],
                "new_foot_event_types": tuple(
                    event.event_type for event in foot_events["new_events"]
                ),
            }
        )
        self._foot_events_updated_frame = self.frame_index
        return foot_events

    def finalize_state(self, state: dict[str, object]) -> dict[str, object]:
        if self._candidate_completed_frame != self.frame_index:
            self.observe_candidate_phase(str(state.get("phase", "unknown")))
        if self._contacts_updated_frame != self.frame_index:
            contacts = self.contact_detectors.update(
                self._current_features,
                phase=str(state.get("phase", "unknown")),
                frame_index=self.frame_index,
                timestamp_ms=self.last_timestamp_ms,
            )
            for name, result in contacts.items():
                prefix = "chest_contact_proxy" if name == "chest_proxy" else f"{name}_contact"
                self._current_features[f"{prefix}_status"] = result.status
                self._current_features[f"{prefix}_confidence"] = result.confidence
                self._current_features[f"{prefix}_surface_height_ratio"] = (
                    result.surface_height_ratio
                )
            self._contacts_updated_frame = self.frame_index
        self.update_foot_events_for_current_frame()
        state.update(self.rep_summary())
        debug = state.get("debug")
        if not isinstance(debug, dict):
            debug = {}
            state["debug"] = debug
        debug.update(
            {
                "candidate_count": self.candidate_count,
                "pose_valid_rep_count": self.pose_valid_rep_count,
                "no_rep_count": self.no_rep_count,
                "unsure_count": self.unsure_count,
                "last_rep_decision": (
                    None
                    if self.last_rep_decision is None
                    else self.last_rep_decision.as_dict()
                ),
                "last_rep_observability": (
                    None
                    if self.last_observability_assessment is None
                    else self.last_observability_assessment.as_dict()
                ),
                "last_three_d_assist": (
                    None
                    if self.last_three_d_assist_assessment is None
                    else self.last_three_d_assist_assessment.as_dict()
                ),
                "floor_reference": self.last_floor_reference.as_dict(),
                "contacts": self.contact_detectors.as_dict(),
                "foot_events": self.foot_event_detector.as_dict(),
            }
        )
        return state

    def determine_phase(self, features: dict[str, object] | None) -> str:
        visible_score = _feature_score(features)
        if visible_score <= 0.0:
            return "no_pose"
        if visible_score < self.min_visible_score:
            return "low_visibility"
        return "ready"

    def build_feedback_messages(self, features: dict[str, object] | None) -> list[FeedbackMessage]:
        visible_score = _feature_score(features)
        if visible_score <= 0.0:
            return [
                FeedbackMessage(
                    level="error",
                    code="pose_missing",
                    text="未检测到稳定姿态",
                    confidence=1.0,
                )
            ]
        if visible_score < self.min_visible_score:
            return [
                FeedbackMessage(
                    level="warn",
                    code="low_visibility",
                    text="关键点可见度偏低",
                    confidence=1.0 - visible_score,
                )
            ]
        return []

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, object]:
        self.begin_frame(features, timestamp_ms)
        self.phase = self.determine_phase(features)
        self.last_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms)
        feedback_messages = self.build_feedback_messages(features)
        visible_score = _feature_score(features)
        return self.finalize_state({
            "action": self.action,
            "phase": self.phase,
            "rep_count": self.rep_count,
            "feedback_messages": feedback_messages,
            "debug": {
                "timestamp_ms": self.last_timestamp_ms,
                "visible_score": visible_score,
                "min_visible_score": self.min_visible_score,
                "camera_view": self.camera_view,
                "view_profile": self.camera_view_profile,
            },
        })


__all__ = [
    "BaseActionAnalyzer",
    "BodyRuleResult",
    "PhaseSequenceTracker",
    "RepCandidate",
    "RepDecision",
    "TRANSIENT_PHASES",
    "aggregate_rep_decision",
]
