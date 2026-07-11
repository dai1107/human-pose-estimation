from __future__ import annotations

from collections.abc import Mapping, Sequence

from .feedback import FeedbackMessage
from .view_policy import filter_feedback_for_view, normalize_camera_view, view_profile


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
        self.camera_view = "unknown"
        self.reset()

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


__all__ = ["BaseActionAnalyzer"]
