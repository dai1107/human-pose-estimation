from __future__ import annotations

from collections.abc import Mapping, Sequence

from .feedback import FeedbackMessage
from .view_policy import filter_feedback_for_view, normalize_camera_view, view_profile


TRANSIENT_PHASES = frozenset({"unknown", "no_pose", "low_visibility"})


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
        self.last_timestamp_ms: int | None = None

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
        self.phase = self.determine_phase(features)
        self.last_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms)
        feedback_messages = self.build_feedback_messages(features)
        visible_score = _feature_score(features)
        return {
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
        }


__all__ = ["BaseActionAnalyzer", "PhaseSequenceTracker", "TRANSIENT_PHASES"]
