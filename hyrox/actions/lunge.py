from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyrox.base import BaseActionAnalyzer
from hyrox.config import load_lunge_config
from hyrox.feedback import FeedbackMessage


PHASE_CONFIRMATION_FRAMES = 3
REP_COOLDOWN_MS = 400
FEEDBACK_PRIORITY = {"error": 0, "warn": 1, "info": 2}
SENSITIVITY_PROFILES: dict[str, dict[str, float | int]] = {
    "low": {
        "min_visible_score": 0.55,
        "stand_knee_angle": 158.0,
        "stand_hip_angle": 150.0,
        "full_extension_knee_angle": 170.0,
        "full_extension_hip_angle": 165.0,
        "bottom_knee_angle": 105.0,
        "deep_knee_angle": 95.0,
        "torso_lean_warn_angle": 18.0,
        "motion_tolerance": 5.0,
        "confirmation_frames": 4,
    },
    "medium": {
        "min_visible_score": 0.45,
        "stand_knee_angle": 150.0,
        "stand_hip_angle": 145.0,
        "full_extension_knee_angle": 165.0,
        "full_extension_hip_angle": 160.0,
        "bottom_knee_angle": 115.0,
        "deep_knee_angle": 100.0,
        "torso_lean_warn_angle": 20.0,
        "motion_tolerance": 3.0,
        "confirmation_frames": PHASE_CONFIRMATION_FRAMES,
    },
    "high": {
        "min_visible_score": 0.35,
        "stand_knee_angle": 145.0,
        "stand_hip_angle": 140.0,
        "full_extension_knee_angle": 158.0,
        "full_extension_hip_angle": 152.0,
        "bottom_knee_angle": 125.0,
        "deep_knee_angle": 108.0,
        "torso_lean_warn_angle": 24.0,
        "motion_tolerance": 2.0,
        "confirmation_frames": 2,
    },
}


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _min_metric(*values: object) -> float | None:
    valid = [_safe_float(value) for value in values]
    filtered = [value for value in valid if value is not None]
    if not filtered:
        return None
    return min(filtered)


def _profile_value(profile: dict[str, float | int], key: str, override: float | int | None) -> float | int:
    return profile[key] if override is None else override


def _feedback_sort_key(message: FeedbackMessage) -> tuple[int, float, str]:
    return (
        FEEDBACK_PRIORITY.get(message.level, 99),
        -message.confidence,
        message.code,
    )


class LungeAnalyzer(BaseActionAnalyzer):
    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        config: Mapping[str, object] | None = None,
        config_name: str | None = None,
        min_visible_score: float | None = None,
        stand_knee_angle: float | None = None,
        stand_hip_angle: float | None = None,
        full_extension_knee_angle: float | None = None,
        full_extension_hip_angle: float | None = None,
        bottom_knee_angle: float | None = None,
        deep_knee_angle: float | None = None,
        torso_lean_warn_angle: float | None = None,
        motion_tolerance: float | None = None,
        confirmation_frames: int | None = None,
        rep_cooldown_ms: int = REP_COOLDOWN_MS,
    ) -> None:
        if sensitivity not in SENSITIVITY_PROFILES:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        profile = SENSITIVITY_PROFILES[sensitivity]
        config_data = dict(config) if config is not None else {}

        def config_value(key: str, fallback: float | int | None) -> float | int | None:
            return config_data.get(key, fallback)

        resolved_min_visible_score = float(
            config_value("visibility_min", _profile_value(profile, "min_visible_score", min_visible_score))
        )
        super().__init__(action="Lunge", min_visible_score=resolved_min_visible_score)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or config_data.get("config_name") or "lunge_default")
        self.stand_knee_angle = float(
            config_value("stand_knee_angle_min", _profile_value(profile, "stand_knee_angle", stand_knee_angle))
        )
        self.stand_hip_angle = float(
            config_value("stand_hip_angle_min", _profile_value(profile, "stand_hip_angle", stand_hip_angle))
        )
        self.full_extension_knee_angle = float(
            config_value("full_extension_knee_angle_min", _profile_value(profile, "full_extension_knee_angle", full_extension_knee_angle))
        )
        self.full_extension_hip_angle = float(
            config_value("full_extension_hip_angle_min", _profile_value(profile, "full_extension_hip_angle", full_extension_hip_angle))
        )
        self.bottom_knee_angle = float(
            config_value("bottom_knee_angle_max", _profile_value(profile, "bottom_knee_angle", bottom_knee_angle))
        )
        self.deep_knee_angle = float(
            config_value("deep_knee_angle_max", _profile_value(profile, "deep_knee_angle", deep_knee_angle))
        )
        self.torso_lean_warn_angle = float(
            config_value("torso_lean_warn", _profile_value(profile, "torso_lean_warn_angle", torso_lean_warn_angle))
        )
        self.motion_tolerance = float(
            config_value("motion_tolerance", _profile_value(profile, "motion_tolerance", motion_tolerance))
        )
        self.confirmation_frames = max(
            1,
            int(config_value("stable_frames", _profile_value(profile, "confirmation_frames", confirmation_frames))),
        )
        self.rep_cooldown_ms = max(0, int(config_value("rep_cooldown_ms", rep_cooldown_ms)))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        sensitivity: str = "medium",
        config_name: str | None = None,
    ) -> LungeAnalyzer:
        return cls(
            sensitivity=sensitivity,
            config=config,
            config_name=config_name,
        )

    @classmethod
    def from_config_path(
        cls,
        path: str | None,
        *,
        sensitivity: str = "medium",
    ) -> LungeAnalyzer:
        config = load_lunge_config(path)
        resolved_name = str(config.get("config_name") or (path or "lunge_default"))
        return cls.from_config(config, sensitivity=sensitivity, config_name=resolved_name)

    def reset(self) -> None:
        super().reset()
        self.phase = "unknown"
        self.raw_phase = "unknown"
        self.stable_phase = "unknown"
        self.frames_in_phase = 0
        self.previous_min_knee_angle: float | None = None
        self.bottom_seen = False
        self.current_rep_min_knee_angle: float | None = None
        self.just_completed_rep = False
        self.last_rep_time_ms: int | None = None

    def _visible_score(self, features: dict[str, object] | None) -> float:
        if not isinstance(features, dict):
            return 0.0
        try:
            visible_score = float(features.get("visible_score", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, visible_score))

    def _phase_from_features(
        self,
        min_knee_angle: float | None,
        min_hip_angle: float | None,
    ) -> str:
        if min_knee_angle is None or min_hip_angle is None:
            return "unknown"
        if min_knee_angle >= self.stand_knee_angle and min_hip_angle >= self.stand_hip_angle:
            return "stand"
        if min_knee_angle <= self.bottom_knee_angle:
            return "bottom"
        if self.previous_min_knee_angle is not None:
            delta = min_knee_angle - self.previous_min_knee_angle
            if delta <= -self.motion_tolerance:
                return "descent"
            if delta >= self.motion_tolerance:
                return "ascent"
        if self.raw_phase in {"descent", "bottom"} and min_knee_angle < self.stand_knee_angle:
            return "descent"
        if self.raw_phase == "ascent" and min_knee_angle < self.stand_knee_angle:
            return "ascent"
        if self.stable_phase in {"descent", "bottom"} and min_knee_angle < self.stand_knee_angle:
            return "descent"
        if self.stable_phase == "ascent" and min_knee_angle < self.stand_knee_angle:
            return "ascent"
        return "unknown"

    def _visibility_feedback(self, visible_score: float) -> list[FeedbackMessage]:
        confidence = 1.0 if visible_score <= 0.0 else max(0.0, min(1.0, 1.0 - visible_score))
        return [
            FeedbackMessage(
                level="warn",
                code="LOW_VISIBILITY",
                text="请站到画面中间，保证全身入镜",
                confidence=confidence,
            )
        ]

    def _advance_phase(self, raw_phase: str) -> tuple[str, str]:
        previous_stable_phase = self.stable_phase
        if raw_phase == self.raw_phase:
            self.frames_in_phase += 1
        else:
            self.raw_phase = raw_phase
            self.frames_in_phase = 1
        if self.frames_in_phase >= self.confirmation_frames:
            self.stable_phase = self.raw_phase
        self.phase = self.stable_phase
        return previous_stable_phase, self.stable_phase

    def _clear_rep_tracking(self) -> None:
        self.bottom_seen = False
        self.current_rep_min_knee_angle = None

    def _cooldown_elapsed(self, timestamp_ms: int | None) -> bool:
        if timestamp_ms is None or self.last_rep_time_ms is None:
            return True
        return timestamp_ms - self.last_rep_time_ms >= self.rep_cooldown_ms

    def _finalize_feedback(self, messages: list[FeedbackMessage]) -> list[FeedbackMessage]:
        ordered = sorted(messages, key=_feedback_sort_key)
        return ordered[:2]

    def _build_lunge_feedback(
        self,
        *,
        stable_phase: str,
        min_knee_angle: float | None,
        min_hip_angle: float | None,
        torso_angle: float | None,
        just_completed_rep: bool,
    ) -> list[FeedbackMessage]:
        messages: list[FeedbackMessage] = []
        if stable_phase == "bottom" and min_knee_angle is not None and min_knee_angle > self.deep_knee_angle:
            confidence = min(1.0, max(0.0, (min_knee_angle - self.deep_knee_angle) / 20.0))
            messages.append(
                FeedbackMessage(
                    level="warn",
                    code="NOT_DEEP_ENOUGH",
                    text="下蹲幅度不够，后侧膝盖应接近地面",
                    confidence=confidence,
                )
            )
        if torso_angle is not None and abs(torso_angle) > self.torso_lean_warn_angle:
            confidence = min(1.0, max(0.0, (abs(torso_angle) - self.torso_lean_warn_angle) / 15.0))
            messages.append(
                FeedbackMessage(
                    level="warn",
                    code="LEAN_TOO_MUCH",
                    text="躯干前倾过多，保持核心稳定",
                    confidence=confidence,
                )
            )
        if just_completed_rep and (
            min_knee_angle is None
            or min_hip_angle is None
            or min_knee_angle < self.full_extension_knee_angle
            or min_hip_angle < self.full_extension_hip_angle
        ):
            knee_gap = 0.0 if min_knee_angle is None else max(0.0, self.full_extension_knee_angle - min_knee_angle)
            hip_gap = 0.0 if min_hip_angle is None else max(0.0, self.full_extension_hip_angle - min_hip_angle)
            confidence = min(1.0, max(0.2, max(knee_gap, hip_gap) / 20.0))
            messages.append(
                FeedbackMessage(
                    level="info",
                    code="STAND_EXTENSION",
                    text="每次站起时膝盖和髋部要伸直",
                    confidence=confidence,
                )
            )
        return self._finalize_feedback(messages)

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, Any]:
        self.last_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms)
        self.just_completed_rep = False
        visible_score = self._visible_score(features)
        min_knee_angle = _min_metric(
            None if features is None else features.get("left_knee_angle"),
            None if features is None else features.get("right_knee_angle"),
        )
        min_hip_angle = _min_metric(
            None if features is None else features.get("left_hip_angle"),
            None if features is None else features.get("right_hip_angle"),
        )
        torso_angle = _safe_float(None if features is None else features.get("torso_angle"))

        if visible_score < self.min_visible_score:
            raw_phase = "no_pose" if visible_score <= 0.0 else "low_visibility"
            previous_stable_phase, stable_phase = self._advance_phase(raw_phase)
            if stable_phase != previous_stable_phase and stable_phase in {"low_visibility", "no_pose"}:
                self._clear_rep_tracking()
            self.previous_min_knee_angle = None
            feedback_messages = self._visibility_feedback(visible_score)
            return {
                "action": self.action,
                "phase": self.stable_phase,
                "rep_count": self.rep_count,
                "feedback_messages": feedback_messages,
                "debug": {
                    "timestamp_ms": self.last_timestamp_ms,
                    "visible_score": visible_score,
                    "min_visible_score": self.min_visible_score,
                    "min_knee_angle": min_knee_angle,
                    "min_hip_angle": min_hip_angle,
                    "torso_angle": torso_angle,
                    "raw_phase": self.raw_phase,
                    "stable_phase": self.stable_phase,
                    "frames_in_phase": self.frames_in_phase,
                    "last_rep_time_ms": self.last_rep_time_ms,
                    "confirmation_frames": self.confirmation_frames,
                    "rep_cooldown_ms": self.rep_cooldown_ms,
                    "sensitivity": self.sensitivity,
                    "config_name": self.config_name,
                },
            }

        raw_phase = self._phase_from_features(min_knee_angle, min_hip_angle)
        previous_stable_phase, stable_phase = self._advance_phase(raw_phase)

        if raw_phase in {"descent", "bottom", "ascent"} and min_knee_angle is not None:
            if self.current_rep_min_knee_angle is None:
                self.current_rep_min_knee_angle = min_knee_angle
            else:
                self.current_rep_min_knee_angle = min(self.current_rep_min_knee_angle, min_knee_angle)

        if stable_phase != previous_stable_phase:
            if stable_phase == "bottom":
                self.bottom_seen = True
            if stable_phase == "stand":
                if self.bottom_seen and self._cooldown_elapsed(self.last_timestamp_ms):
                    self.rep_count += 1
                    self.just_completed_rep = True
                    self.last_rep_time_ms = self.last_timestamp_ms
                self._clear_rep_tracking()
            elif stable_phase in {"low_visibility", "no_pose"}:
                self._clear_rep_tracking()

        self.previous_min_knee_angle = min_knee_angle
        feedback_messages = self._build_lunge_feedback(
            stable_phase=stable_phase,
            min_knee_angle=min_knee_angle,
            min_hip_angle=min_hip_angle,
            torso_angle=torso_angle,
            just_completed_rep=self.just_completed_rep,
        )
        return {
            "action": self.action,
            "phase": self.stable_phase,
            "rep_count": self.rep_count,
            "feedback_messages": feedback_messages,
            "debug": {
                "timestamp_ms": self.last_timestamp_ms,
                "visible_score": visible_score,
                "min_visible_score": self.min_visible_score,
                "min_knee_angle": min_knee_angle,
                "min_hip_angle": min_hip_angle,
                "torso_angle": torso_angle,
                "raw_phase": self.raw_phase,
                "stable_phase": self.stable_phase,
                "frames_in_phase": self.frames_in_phase,
                "last_rep_time_ms": self.last_rep_time_ms,
                "bottom_seen": self.bottom_seen,
                "just_completed_rep": self.just_completed_rep,
                "confirmation_frames": self.confirmation_frames,
                "rep_cooldown_ms": self.rep_cooldown_ms,
                "sensitivity": self.sensitivity,
                "config_name": self.config_name,
            },
        }


__all__ = ["LungeAnalyzer", "PHASE_CONFIRMATION_FRAMES", "REP_COOLDOWN_MS", "SENSITIVITY_PROFILES"]
