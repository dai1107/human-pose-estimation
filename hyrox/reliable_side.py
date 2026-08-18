from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


BodySide = Literal["left", "right"]


def _safe_unit(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if resolved != resolved:
        return None
    return max(0.0, min(1.0, resolved))


def _available(value: object) -> bool:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return resolved == resolved


@dataclass(frozen=True)
class SideReliability:
    side: BodySide
    score: float
    confidence: float | None
    landmark_coverage: float
    metric_coverage: float
    observable: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "score": self.score,
            "confidence": self.confidence,
            "landmark_coverage": self.landmark_coverage,
            "metric_coverage": self.metric_coverage,
            "observable": self.observable,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ReliableSideSelection:
    selected_side: BodySide | None
    reason: str
    left: SideReliability
    right: SideReliability
    pending_side: BodySide | None
    pending_frames: int
    stable_frames: int
    switch_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_side": self.selected_side,
            "reason": self.reason,
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
            "pending_side": self.pending_side,
            "pending_frames": self.pending_frames,
            "stable_frames": self.stable_frames,
            "switch_count": self.switch_count,
        }


class ReliableSideSelector:
    """Select one observable joint chain from actual evidence, never camera labels.

    Scores combine landmark confidence, required-metric availability and
    confidence-field coverage.  A switch needs a material score advantage for
    consecutive frames while the current side remains observable; loss of the
    current side permits immediate failover.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.45,
        min_metric_coverage: float = 1.0,
        switch_margin: float = 0.08,
        switch_confirmation_frames: int = 2,
    ) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.min_metric_coverage = max(
            0.0,
            min(1.0, float(min_metric_coverage)),
        )
        self.switch_margin = max(0.0, min(1.0, float(switch_margin)))
        self.switch_confirmation_frames = max(
            1,
            int(switch_confirmation_frames),
        )
        self.reset()

    def reset(self) -> None:
        self.current_side: BodySide | None = None
        self.pending_side: BodySide | None = None
        self.pending_frames = 0
        self.stable_frames = 0
        self.switch_count = 0
        self.last_selection: ReliableSideSelection | None = None

    def _assess(
        self,
        features: Mapping[str, object],
        side: BodySide,
        *,
        required_landmarks: Sequence[str],
        required_metrics: Sequence[str],
    ) -> SideReliability:
        confidence_values = tuple(
            score
            for landmark in required_landmarks
            for score in (
                _safe_unit(features.get(f"{side}_{landmark}_confidence")),
            )
            if score is not None
        )
        landmark_coverage = (
            1.0
            if not required_landmarks
            else len(confidence_values) / len(required_landmarks)
        )
        confidence = min(confidence_values) if confidence_values else None
        if confidence is None:
            confidence = _safe_unit(features.get(f"{side}_side_visible_score"))
        if confidence is None:
            confidence = _safe_unit(features.get("visible_score"))

        available_metrics = sum(
            _available(features.get(f"{side}_{metric}"))
            for metric in required_metrics
        )
        metric_coverage = (
            1.0
            if not required_metrics
            else available_metrics / len(required_metrics)
        )
        reasons: list[str] = []
        if metric_coverage < self.min_metric_coverage:
            reasons.append("REQUIRED_METRIC_UNAVAILABLE")
        if confidence is None:
            reasons.append("RELIABILITY_CONFIDENCE_UNAVAILABLE")
        elif confidence < self.min_confidence:
            reasons.append("REQUIRED_LANDMARK_CONFIDENCE_LOW")
        observable = not reasons
        score = (
            0.65 * (confidence or 0.0)
            + 0.25 * metric_coverage
            + 0.10 * landmark_coverage
        )
        return SideReliability(
            side=side,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            landmark_coverage=landmark_coverage,
            metric_coverage=metric_coverage,
            observable=observable,
            reason_codes=tuple(reasons),
        )

    def select(
        self,
        features: Mapping[str, object],
        *,
        required_landmarks: Sequence[str],
        required_metrics: Sequence[str],
        preferred_side: BodySide | None = None,
    ) -> ReliableSideSelection:
        left = self._assess(
            features,
            "left",
            required_landmarks=required_landmarks,
            required_metrics=required_metrics,
        )
        right = self._assess(
            features,
            "right",
            required_landmarks=required_landmarks,
            required_metrics=required_metrics,
        )
        by_side = {"left": left, "right": right}
        observable = [item for item in (left, right) if item.observable]

        target: BodySide | None
        reason: str
        if not observable:
            target = None
            reason = "NO_OBSERVABLE_SIDE"
        elif len(observable) == 1:
            target = observable[0].side
            reason = "ONLY_OBSERVABLE_SIDE"
        else:
            ranked = sorted(
                observable,
                key=lambda item: (
                    item.score,
                    item.side == preferred_side,
                    item.side == self.current_side,
                    item.side == "left",
                ),
                reverse=True,
            )
            best, other = ranked
            if (
                preferred_side in by_side
                and by_side[preferred_side].observable
                and abs(left.score - right.score) < self.switch_margin
            ):
                target = preferred_side
                reason = "PREFERRED_SIDE_TIE_BREAK"
            elif (
                self.current_side in by_side
                and by_side[self.current_side].observable
                and best.side != self.current_side
                and best.score
                < by_side[self.current_side].score + self.switch_margin
            ):
                target = self.current_side
                reason = "CURRENT_SIDE_HYSTERESIS"
            else:
                target = best.side
                reason = (
                    "HIGHER_RELIABILITY_SCORE"
                    if best.score > other.score
                    else "DETERMINISTIC_TIE_BREAK"
                )

        current_observable = (
            self.current_side in by_side
            and by_side[self.current_side].observable
        )
        if target is None:
            self.current_side = None
            self.pending_side = None
            self.pending_frames = 0
            self.stable_frames = 0
        elif self.current_side is None:
            self.current_side = target
            self.pending_side = None
            self.pending_frames = 0
            self.stable_frames = 1
        elif target == self.current_side:
            self.pending_side = None
            self.pending_frames = 0
            self.stable_frames += 1
        elif not current_observable:
            self.current_side = target
            self.pending_side = None
            self.pending_frames = 0
            self.stable_frames = 1
            self.switch_count += 1
            reason = "CURRENT_SIDE_UNOBSERVABLE_FAILOVER"
        else:
            if self.pending_side == target:
                self.pending_frames += 1
            else:
                self.pending_side = target
                self.pending_frames = 1
            if self.pending_frames >= self.switch_confirmation_frames:
                self.current_side = target
                self.pending_side = None
                self.pending_frames = 0
                self.stable_frames = 1
                self.switch_count += 1
                reason = "CONFIRMED_RELIABILITY_SWITCH"
            else:
                reason = "SWITCH_PENDING"
                self.stable_frames += 1

        selection = ReliableSideSelection(
            selected_side=self.current_side,
            reason=reason,
            left=left,
            right=right,
            pending_side=self.pending_side,
            pending_frames=self.pending_frames,
            stable_frames=self.stable_frames,
            switch_count=self.switch_count,
        )
        self.last_selection = selection
        return selection


__all__ = [
    "BodySide",
    "ReliableSideSelection",
    "ReliableSideSelector",
    "SideReliability",
]
