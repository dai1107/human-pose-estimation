"""View-aware phase evidence scoring and deterministic temporal decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


TRANSIENT_PHASES = frozenset({"unknown", "no_pose", "low_visibility"})


def _clamp(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number:
        return default
    return max(0.0, min(1.0, number))


@dataclass(frozen=True, slots=True)
class PhaseScoreResult:
    scores: Mapping[str, float]
    selected_phase: str
    confidence: float
    evidence_quality: float
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "scores": dict(self.scores),
            "selected_phase": self.selected_phase,
            "confidence": self.confidence,
            "evidence_quality": self.evidence_quality,
            "reason_codes": list(self.reason_codes),
        }


class PhaseEvidenceScorer:
    """Expose calibrated-looking *scores* without claiming learned probabilities.

    The legacy rule phase remains the hypothesis.  Its score is attenuated by
    actual formal observability and by the action-view capability metadata.
    """

    def score(
        self,
        raw_phase: str,
        features: Mapping[str, object] | None,
        *,
        view_multiplier: float = 1.0,
        alternatives: Sequence[str] = (),
    ) -> PhaseScoreResult:
        phase = str(raw_phase)
        values = features or {}
        quality = _clamp(
            values.get("formal_evidence_quality", values.get("visible_score", 0.0))
        )
        allowed = values.get("endpoint_evidence_allowed", True) is not False
        reasons = [str(value) for value in (values.get("formal_quality_reason_codes") or ())]
        if phase not in TRANSIENT_PHASES and not allowed:
            phase = "low_visibility"
            reasons.append("ENDPOINT_EVIDENCE_UNOBSERVABLE")
        multiplier = max(0.0, min(1.0, float(view_multiplier)))
        primary = max(0.0, min(1.0, quality * multiplier))
        if phase in TRANSIENT_PHASES:
            primary = max(primary, 1.0 - quality)
        scores = {str(item): 0.0 for item in alternatives}
        scores[phase] = primary
        # Scores are deterministic evidence strengths, not class probabilities.
        return PhaseScoreResult(
            scores=scores,
            selected_phase=phase,
            confidence=primary,
            evidence_quality=quality,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )


@dataclass(frozen=True, slots=True)
class DecodedPhase:
    previous_phase: str
    stable_phase: str
    proposed_phase: str
    frames_in_proposal: int
    required_frames: int
    transition_legal: bool
    held_for_hysteresis: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "previous_phase": self.previous_phase,
            "stable_phase": self.stable_phase,
            "proposed_phase": self.proposed_phase,
            "frames_in_proposal": self.frames_in_proposal,
            "required_frames": self.required_frames,
            "transition_legal": self.transition_legal,
            "held_for_hysteresis": self.held_for_hysteresis,
        }


class TemporalPhaseDecoder:
    """Decode phase evidence with hysteresis, duration and legal transitions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, stable_phase: str = "unknown") -> None:
        self.stable_phase = str(stable_phase)
        self.proposed_phase = "unknown"
        self.frames_in_proposal = 0

    @staticmethod
    def _legal(previous: str, proposed: str, sequence: Sequence[str] | None) -> bool:
        if sequence is None or previous in TRANSIENT_PHASES or proposed in TRANSIENT_PHASES:
            return True
        phases = tuple(str(value) for value in sequence)
        if previous == proposed or previous not in phases or proposed not in phases:
            return True
        indices = [index for index, value in enumerate(phases) if value == previous]
        return any(index + 1 < len(phases) and phases[index + 1] == proposed for index in indices)

    def update(
        self,
        proposed_phase: str,
        *,
        current_stable_phase: str,
        minimum_duration_frames: int,
        transient_hold_frames: int = 0,
        legal_sequence: Sequence[str] | None = None,
        enforce_legal_transitions: bool = True,
    ) -> DecodedPhase:
        previous = str(current_stable_phase)
        proposed = str(proposed_phase)
        if proposed == self.proposed_phase:
            self.frames_in_proposal += 1
        else:
            self.proposed_phase = proposed
            self.frames_in_proposal = 1
        required = max(1, int(minimum_duration_frames))
        if proposed in TRANSIENT_PHASES and previous not in TRANSIENT_PHASES:
            required = max(required, int(transient_hold_frames) + 1)
        legal = self._legal(previous, proposed, legal_sequence)
        can_commit = self.frames_in_proposal >= required and (
            legal or not enforce_legal_transitions
        )
        self.stable_phase = proposed if can_commit else previous
        return DecodedPhase(
            previous_phase=previous,
            stable_phase=self.stable_phase,
            proposed_phase=proposed,
            frames_in_proposal=self.frames_in_proposal,
            required_frames=required,
            transition_legal=legal,
            held_for_hysteresis=not can_commit,
        )


__all__ = [
    "DecodedPhase",
    "PhaseEvidenceScorer",
    "PhaseScoreResult",
    "TemporalPhaseDecoder",
]
