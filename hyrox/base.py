from __future__ import annotations

from .feedback import FeedbackMessage


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
        self.reset()

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
            },
        }


__all__ = ["BaseActionAnalyzer"]
