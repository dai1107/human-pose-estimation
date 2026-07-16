from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from hyrox.base import BaseActionAnalyzer, PhaseSequenceTracker
from hyrox.config import load_wall_ball_config
from hyrox.feedback import FeedbackMessage


REP_COOLDOWN_MS = 400
FEEDBACK_PRIORITY = {"error": 0, "warn": 1, "info": 2}
WALL_BALL_SENSITIVITY_DELTAS: dict[str, dict[str, float | int]] = {
    "low": {
        "visibility_min": 0.10,
        "stand_knee_angle_min": 5.0,
        "stand_hip_angle_min": 5.0,
        "bottom_knee_angle_max": -5.0,
        "hip_below_knee_margin": 0.015,
        "throw_knee_angle_min": 5.0,
        "throw_hip_angle_min": 5.0,
        "throw_elbow_angle_min": 5.0,
        "wrist_above_shoulder_min": 0.02,
        "stable_frames": 1,
    },
    "medium": {},
    "high": {
        "visibility_min": -0.10,
        "stand_knee_angle_min": -5.0,
        "stand_hip_angle_min": -5.0,
        "bottom_knee_angle_max": 10.0,
        "hip_below_knee_margin": -0.015,
        "throw_knee_angle_min": -5.0,
        "throw_hip_angle_min": -5.0,
        "throw_elbow_angle_min": -10.0,
        "wrist_above_shoulder_min": -0.015,
        "stable_frames": -1,
    },
}


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _resolved_float(value: object, fallback: float, *, minimum: float | None = None) -> float:
    resolved = _safe_float(value)
    if resolved is None:
        resolved = fallback
    return max(minimum, resolved) if minimum is not None else resolved


def _resolved_int(value: object, fallback: int, *, minimum: int = 0) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        resolved = fallback
    return max(minimum, resolved)


def _apply_sensitivity(config: dict[str, object], sensitivity: str) -> dict[str, object]:
    adjusted = dict(config)
    for key, delta in WALL_BALL_SENSITIVITY_DELTAS[sensitivity].items():
        current = _safe_float(adjusted.get(key))
        if current is None:
            continue
        value = current + float(delta)
        adjusted[key] = int(round(value)) if key == "stable_frames" else value
    return adjusted


def _minimum_feature(features: Mapping[str, object] | None, *names: str) -> float | None:
    if features is None:
        return None
    values = [_safe_float(features.get(name)) for name in names]
    valid = [value for value in values if value is not None]
    return min(valid) if valid else None


def _maximum_feature(features: Mapping[str, object] | None, *names: str) -> float | None:
    if features is None:
        return None
    values = [_safe_float(features.get(name)) for name in names]
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def _feedback_sort_key(message: FeedbackMessage) -> tuple[int, float, str]:
    return FEEDBACK_PRIORITY.get(message.level, 99), -message.confidence, message.code


class WallBallAnalyzer(BaseActionAnalyzer):
    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        config: Mapping[str, object] | None = None,
        config_name: str | None = None,
    ) -> None:
        if sensitivity not in WALL_BALL_SENSITIVITY_DELTAS:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        config_data = _apply_sensitivity(dict(config or load_wall_ball_config()), sensitivity)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or config_data.get("config_name") or "wall_ball_default")
        visibility_min = _resolved_float(config_data.get("visibility_min"), 0.45, minimum=0.0)
        super().__init__(action="Wall Ball", min_visible_score=min(1.0, visibility_min))
        self.configure_feedback_limits(config_data)
        self.stand_knee_angle = _resolved_float(config_data.get("stand_knee_angle_min"), 150.0)
        self.stand_hip_angle = _resolved_float(config_data.get("stand_hip_angle_min"), 145.0)
        self.bottom_knee_angle = _resolved_float(config_data.get("bottom_knee_angle_max"), 110.0)
        self.hip_below_knee_margin = _resolved_float(config_data.get("hip_below_knee_margin"), -0.05)
        self.throw_knee_angle = _resolved_float(config_data.get("throw_knee_angle_min"), 150.0)
        self.throw_hip_angle = _resolved_float(config_data.get("throw_hip_angle_min"), 145.0)
        self.throw_elbow_angle = _resolved_float(config_data.get("throw_elbow_angle_min"), 125.0)
        self.wrist_above_shoulder_min = _resolved_float(config_data.get("wrist_above_shoulder_min"), 0.03)
        self.full_extension_knee_angle = _resolved_float(
            config_data.get("full_extension_knee_angle_min"), 165.0
        )
        self.full_extension_hip_angle = _resolved_float(
            config_data.get("full_extension_hip_angle_min"), 160.0
        )
        self.knee_cave_ratio_max = _resolved_float(
            config_data.get("knee_cave_ratio_max"), 0.72, minimum=0.0
        )
        self.minimum_frontal_ankle_width = _resolved_float(
            config_data.get("minimum_frontal_ankle_width"), 0.08, minimum=0.0
        )
        self.motion_tolerance = _resolved_float(config_data.get("motion_tolerance"), 3.0, minimum=0.0)
        self.hip_motion_tolerance = _resolved_float(
            config_data.get("hip_motion_tolerance"), 0.004, minimum=0.0
        )
        self.confirmation_frames = _resolved_int(config_data.get("stable_frames"), 3, minimum=1)
        self.rep_cooldown_ms = _resolved_int(
            config_data.get("rep_cooldown_ms"), REP_COOLDOWN_MS, minimum=0
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        sensitivity: str = "medium",
        config_name: str | None = None,
    ) -> WallBallAnalyzer:
        return cls(config=config, sensitivity=sensitivity, config_name=config_name)

    @classmethod
    def from_config_path(
        cls,
        path: str | None,
        *,
        sensitivity: str = "medium",
    ) -> WallBallAnalyzer:
        config = load_wall_ball_config(path)
        resolved_name = str(config.get("config_name") or (path or "wall_ball_default"))
        return cls.from_config(config, sensitivity=sensitivity, config_name=resolved_name)

    def reset(self) -> None:
        super().reset()
        self.phase = "unknown"
        self.raw_phase = "unknown"
        self.stable_phase = "unknown"
        self.frames_in_phase = 0
        self.previous_min_knee_angle: float | None = None
        self.previous_hip_center_y: float | None = None
        self.stand_seen = False
        self.bottom_seen = False
        self.bottom_depth_met = False
        self.extension_seen = False
        self.extension_pending = False
        self.just_completed_rep = False
        self.just_finished_attempt = False
        self.last_rep_time_ms: int | None = None
        self.rep_sequence = PhaseSequenceTracker(
            ("stand", "squat_down", "bottom", "drive", "throw_extension"),
            optional_phases=("squat_down", "drive"),
        )

    def _visible_score(self, features: Mapping[str, object] | None) -> float:
        if features is None:
            return 0.0
        return max(0.0, min(1.0, _safe_float(features.get("visible_score")) or 0.0))

    def _advance_phase(self, raw_phase: str) -> tuple[str, str]:
        return self._advance_confirmed_phase(raw_phase, self.confirmation_frames)

    def _cooldown_elapsed(self, timestamp_ms: int | None) -> bool:
        if timestamp_ms is None or self.last_rep_time_ms is None:
            return True
        return timestamp_ms - self.last_rep_time_ms >= self.rep_cooldown_ms

    def _lower_body_standing(self, min_knee: float | None, hip_angle: float | None) -> bool:
        return (
            min_knee is not None
            and hip_angle is not None
            and min_knee >= self.stand_knee_angle
            and hip_angle >= self.stand_hip_angle
        )

    def _throw_posture(
        self,
        min_knee: float | None,
        hip_angle: float | None,
        elbow_angle: float | None,
        wrist_above_shoulder: float | None,
    ) -> bool:
        return (
            min_knee is not None
            and hip_angle is not None
            and elbow_angle is not None
            and wrist_above_shoulder is not None
            and min_knee >= self.throw_knee_angle
            and hip_angle >= self.throw_hip_angle
            and elbow_angle >= self.throw_elbow_angle
            and wrist_above_shoulder >= self.wrist_above_shoulder_min
        )

    def _phase_from_features(
        self,
        *,
        min_knee: float | None,
        hip_angle: float | None,
        elbow_angle: float | None,
        hip_center_y: float | None,
        wrist_above_shoulder: float | None,
    ) -> str:
        if min_knee is None or hip_angle is None:
            return "unknown"
        lower_standing = self._lower_body_standing(min_knee, hip_angle)
        throw_posture = self._throw_posture(
            min_knee, hip_angle, elbow_angle, wrist_above_shoulder
        )
        if (self.bottom_seen or self.extension_pending or self.extension_seen) and throw_posture:
            # Preserve an explicit drive phase before the terminal extension so
            # completion can only be emitted after the full ordered sequence.
            return "drive" if self.stable_phase == "bottom" else "throw_extension"
        if self.bottom_seen or self.extension_pending:
            if min_knee > self.bottom_knee_angle or lower_standing:
                return "drive"
        if self.extension_seen and lower_standing and not throw_posture:
            return "reset"
        if min_knee <= self.bottom_knee_angle:
            return "bottom"
        if self.previous_min_knee_angle is not None:
            knee_delta = min_knee - self.previous_min_knee_angle
            if knee_delta <= -self.motion_tolerance:
                return "squat_down"
        if hip_center_y is not None and self.previous_hip_center_y is not None:
            if hip_center_y - self.previous_hip_center_y >= self.hip_motion_tolerance:
                return "squat_down"
        if lower_standing:
            return "stand"
        if self.raw_phase in {"squat_down", "bottom"} or self.stable_phase in {"squat_down", "bottom"}:
            return "squat_down"
        return "unknown"

    def _low_visibility_feedback(self) -> list[FeedbackMessage]:
        return [
            FeedbackMessage(
                level="warn",
                code="LOW_VISIBILITY",
                text="请保证全身入镜",
                confidence=1.0,
            )
        ]

    def _feedback(
        self,
        *,
        stable_phase: str,
        depth_met: bool,
        hip_knee_depth: float | None,
        knee_width: float | None,
        ankle_width: float | None,
        min_knee: float | None,
        hip_angle: float | None,
    ) -> list[FeedbackMessage]:
        messages: list[FeedbackMessage] = []
        if stable_phase == "bottom" and not depth_met:
            gap = self.hip_below_knee_margin - (hip_knee_depth or 0.0)
            messages.append(
                FeedbackMessage(
                    level="warn",
                    code="SQUAT_NOT_DEEP",
                    text="下蹲深度不够，髋部需要低于膝盖",
                    confidence=min(1.0, max(0.25, gap / 0.08)),
                )
            )
        if (
            stable_phase in {"squat_down", "bottom"}
            and knee_width is not None
            and ankle_width is not None
            and ankle_width >= self.minimum_frontal_ankle_width
        ):
            ratio = knee_width / max(ankle_width, 1e-6)
            if ratio < self.knee_cave_ratio_max:
                messages.append(
                    FeedbackMessage(
                        level="warn",
                        code="KNEES_CAVE_IN",
                        text="膝盖内扣，保持膝盖与脚尖方向一致",
                        confidence=min(0.45, max(0.15, self.knee_cave_ratio_max - ratio)),
                    )
                )
        if self.just_finished_attempt and (
            min_knee is None
            or hip_angle is None
            or min_knee < self.full_extension_knee_angle
            or hip_angle < self.full_extension_hip_angle
        ):
            messages.append(
                FeedbackMessage(
                    level="warn",
                    code="NOT_FULL_EXTENSION",
                    text="投球前站起不充分，髋膝需要伸展",
                    confidence=0.7,
                )
            )
        return self.limit_feedback(sorted(messages, key=_feedback_sort_key))

    def _debug(
        self,
        *,
        visible_score: float,
        min_knee: float | None,
        hip_angle: float | None,
        elbow_angle: float | None,
        hip_knee_depth: float | None,
        wrist_above_shoulder: float | None,
        knee_width: float | None,
        ankle_width: float | None,
    ) -> dict[str, object]:
        return {
            "timestamp_ms": self.last_timestamp_ms,
            "visible_score": visible_score,
            "min_visible_score": self.min_visible_score,
            "min_knee_angle": min_knee,
            "selected_hip_angle": hip_angle,
            "selected_elbow_angle": elbow_angle,
            "hip_knee_depth": hip_knee_depth,
            "wrist_above_shoulder": wrist_above_shoulder,
            "knee_width": knee_width,
            "ankle_width": ankle_width,
            "raw_phase": self.raw_phase,
            "stable_phase": self.stable_phase,
            "frames_in_phase": self.frames_in_phase,
            "last_rep_time_ms": self.last_rep_time_ms,
            "stand_seen": self.stand_seen,
            "bottom_seen": self.bottom_seen,
            "bottom_depth_met": self.bottom_depth_met,
            "extension_seen": self.extension_seen,
            "extension_pending": self.extension_pending,
            "just_completed_rep": self.just_completed_rep,
            "confirmation_frames": self.confirmation_frames,
            "rep_cooldown_ms": self.rep_cooldown_ms,
            "sensitivity": self.sensitivity,
            "config_name": self.config_name,
            **self.rep_sequence.debug(),
        }

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, Any]:
        self.last_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms)
        self.just_completed_rep = False
        self.just_finished_attempt = False
        self.rep_sequence.just_completed = False
        visible_score = self._visible_score(features)
        min_knee = _minimum_feature(features, "left_knee_angle", "right_knee_angle")
        hip_angle = _maximum_feature(features, "left_hip_angle", "right_hip_angle")
        elbow_angle = _maximum_feature(features, "left_elbow_angle", "right_elbow_angle")
        hip_center_y = _safe_float(None if features is None else features.get("hip_center_y"))
        hip_knee_depth = _safe_float(None if features is None else features.get("hip_knee_depth"))
        wrist_above_shoulder = _maximum_feature(
            features,
            "wrist_above_shoulder",
            "left_wrist_above_shoulder",
            "right_wrist_above_shoulder",
        )
        knee_width = _safe_float(None if features is None else features.get("knee_width"))
        ankle_width = _safe_float(None if features is None else features.get("ankle_width"))

        if visible_score < self.min_visible_score:
            raw_phase = "no_pose" if visible_score <= 0.0 else "low_visibility"
            previous, stable = self._advance_phase(raw_phase)
            if stable != previous and stable in {"no_pose", "low_visibility"}:
                self.stand_seen = False
                self.bottom_seen = False
                self.bottom_depth_met = False
                self.extension_seen = False
                self.extension_pending = False
                self.rep_sequence.reset()
            self.previous_min_knee_angle = None
            self.previous_hip_center_y = None
            return {
                "action": self.action,
                "phase": self.stable_phase,
                "rep_count": self.rep_count,
                "feedback_messages": self._low_visibility_feedback(),
                "debug": self._debug(
                    visible_score=visible_score,
                    min_knee=min_knee,
                    hip_angle=hip_angle,
                    elbow_angle=elbow_angle,
                    hip_knee_depth=hip_knee_depth,
                    wrist_above_shoulder=wrist_above_shoulder,
                    knee_width=knee_width,
                    ankle_width=ankle_width,
                ),
            }

        raw_phase = self._phase_from_features(
            min_knee=min_knee,
            hip_angle=hip_angle,
            elbow_angle=elbow_angle,
            hip_center_y=hip_center_y,
            wrist_above_shoulder=wrist_above_shoulder,
        )
        previous, stable = self._advance_phase(raw_phase)
        depth_met = hip_knee_depth is not None and hip_knee_depth >= self.hip_below_knee_margin

        if stable == "bottom" and self.stand_seen:
            self.bottom_seen = True
            self.bottom_depth_met = self.bottom_depth_met or depth_met

        sequence_completed = False
        if stable != previous:
            sequence_completed = self.rep_sequence.update(stable)
            if stable == "stand":
                self.stand_seen = True
            if stable == "drive" and self.bottom_seen:
                self.extension_pending = True
            if stable == "throw_extension" and (self.bottom_seen or self.extension_pending):
                self.just_finished_attempt = True
                if sequence_completed:
                    self.rep_count += 1
                    self.just_completed_rep = True
                    self.last_rep_time_ms = self.last_timestamp_ms
                self.bottom_seen = False
                self.bottom_depth_met = False
                self.extension_pending = False
            if stable == "throw_extension":
                self.extension_seen = True
                self.extension_pending = False
            elif stable == "reset":
                self.extension_seen = False
                self.extension_pending = False
            elif stable == "squat_down":
                self.extension_pending = False

        self.previous_min_knee_angle = min_knee
        self.previous_hip_center_y = hip_center_y
        feedback_messages = self._feedback(
            stable_phase=raw_phase,
            depth_met=depth_met,
            hip_knee_depth=hip_knee_depth,
            knee_width=knee_width,
            ankle_width=ankle_width,
            min_knee=min_knee,
            hip_angle=hip_angle,
        )
        return {
            "action": self.action,
            "phase": self.stable_phase,
            "rep_count": self.rep_count,
            "feedback_messages": feedback_messages,
            "debug": self._debug(
                visible_score=visible_score,
                min_knee=min_knee,
                hip_angle=hip_angle,
                elbow_angle=elbow_angle,
                hip_knee_depth=hip_knee_depth,
                wrist_above_shoulder=wrist_above_shoulder,
                knee_width=knee_width,
                ankle_width=ankle_width,
            ),
        }


__all__ = ["REP_COOLDOWN_MS", "WALL_BALL_SENSITIVITY_DELTAS", "WallBallAnalyzer"]
