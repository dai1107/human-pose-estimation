from __future__ import annotations

from collections.abc import Mapping

from hyrox.base import BaseActionAnalyzer, PhaseSequenceTracker
from hyrox.config import load_burpee_broad_jump_config
from hyrox.feedback import FeedbackMessage


SENSITIVITY_FRAME_DELTAS = {"low": 1, "medium": 0, "high": -1}


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _resolved_float(value: object, fallback: float, *, minimum: float = 0.0) -> float:
    number = _safe_float(value)
    return max(minimum, fallback if number is None else number)


def _resolved_int(value: object, fallback: int, *, minimum: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = fallback
    return max(minimum, number)


def _mean_metric(*values: object) -> float | None:
    valid = [number for number in (_safe_float(value) for value in values) if number is not None]
    return sum(valid) / len(valid) if valid else None


def _min_metric(*values: object) -> float | None:
    valid = [number for number in (_safe_float(value) for value in values) if number is not None]
    return min(valid) if valid else None


class BurpeeBroadJumpAnalyzer(BaseActionAnalyzer):
    """Approximate full-body burpee broad jump analyzer for a 45-degree side view."""

    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        config: Mapping[str, object] | None = None,
        config_name: str | None = None,
    ) -> None:
        if sensitivity not in SENSITIVITY_FRAME_DELTAS:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        values = dict(config or load_burpee_broad_jump_config())
        visibility_min = _resolved_float(values.get("visibility_min"), 0.55)
        super().__init__(action="burpee_broad_jump", min_visible_score=min(1.0, visibility_min))
        self.configure_feedback_limits(values)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or values.get("config_name") or "burpee_broad_jump_default")
        self.chest_down_body_height_max = _resolved_float(values.get("chest_down_body_height_max"), 0.35)
        self.bottom_torso_horizontal_min = _resolved_float(values.get("bottom_torso_horizontal_min"), 60.0)
        self.jump_forward_min_delta_x = _resolved_float(values.get("jump_forward_min_delta_x"), 0.08)
        self.feet_stagger_warn = _resolved_float(values.get("feet_stagger_warn"), 0.08)
        self.extra_step_window_ms = _resolved_int(values.get("extra_step_window_ms"), 500)
        base_frames = _resolved_int(values.get("stable_frames"), 3, minimum=1)
        self.confirmation_frames = max(1, base_frames + SENSITIVITY_FRAME_DELTAS[sensitivity])
        self.rep_cooldown_ms = _resolved_int(values.get("rep_cooldown_ms"), 800)
        self.min_phase_duration_ms = _resolved_int(values.get("min_phase_duration_ms"), 100)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        sensitivity: str = "medium",
        config_name: str | None = None,
    ) -> BurpeeBroadJumpAnalyzer:
        return cls(config=config, sensitivity=sensitivity, config_name=config_name)

    @classmethod
    def from_config_path(
        cls,
        path: str | None,
        *,
        sensitivity: str = "medium",
    ) -> BurpeeBroadJumpAnalyzer:
        config = load_burpee_broad_jump_config(path)
        return cls.from_config(
            config,
            sensitivity=sensitivity,
            config_name=str(config.get("config_name") or (path or "burpee_broad_jump_default")),
        )

    def reset(self) -> None:
        super().reset()
        self.phase = "unknown"
        self.raw_phase = "unknown"
        self.stable_phase = "unknown"
        self.frames_in_phase = 0
        self.phase_started_ms: int | None = None
        self.previous_body_center_x: float | None = None
        self.previous_body_center_y: float | None = None
        self.previous_body_height: float | None = None
        self.previous_knee_angle: float | None = None
        self.previous_ankle_center_y: float | None = None
        self.previous_ankle_distance: float | None = None
        self.takeoff_center_x: float | None = None
        self.chest_down_seen = False
        self.takeoff_seen = False
        self.last_rep_time_ms: int | None = None
        self.chest_not_low = False
        self.no_broad_jump = False
        self.extra_steps = False
        self.landing_step_events = 0
        self.landing_started_ms: int | None = None
        self.rep_sequence = PhaseSequenceTracker(
            ("chest_down", "step_or_jump_in", "broad_jump_takeoff", "flight_or_move", "landing"),
            optional_phases=("step_or_jump_in", "flight_or_move"),
        )

    def _visible_score(self, features: dict[str, object]) -> float:
        score = _safe_float(features.get("visible_score"))
        return max(0.0, min(1.0, score or 0.0))

    def _raw_phase(
        self,
        *,
        body_center_x: float | None,
        body_center_y: float | None,
        body_height: float | None,
        torso_angle: float | None,
        knee_angle: float | None,
        hip_angle: float | None,
        wrist_height: float | None,
        ankle_height: float | None,
    ) -> str:
        if body_center_x is None or body_center_y is None or knee_angle is None:
            return "unknown"
        torso_abs = None if torso_angle is None else abs(torso_angle)
        body_height_rising = body_height is not None and self.previous_body_height is not None and body_height - self.previous_body_height > 0.006
        knee_delta = None if self.previous_knee_angle is None else knee_angle - self.previous_knee_angle
        center_y_delta = None if self.previous_body_center_y is None else body_center_y - self.previous_body_center_y
        standing = knee_angle >= 158.0 and (hip_angle is None or hip_angle >= 150.0) and (torso_abs is None or torso_abs < 25.0)
        chest_down = body_height is not None and body_height <= self.chest_down_body_height_max and torso_abs is not None and torso_abs >= self.bottom_torso_horizontal_min
        hands_low = wrist_height is not None and ankle_height is not None and wrist_height >= ankle_height - 0.18

        if self.stable_phase == "landing" and standing:
            return "reset"
        if self.stable_phase == "reset":
            return "stand" if standing else "reset"
        if chest_down:
            return "chest_down"
        if self.stable_phase == "chest_down" and (body_height_rising or (torso_abs is not None and torso_abs < self.bottom_torso_horizontal_min - 10.0)):
            return "step_or_jump_in"
        if self.stable_phase == "hands_down" and (body_height_rising or not hands_low):
            return "step_or_jump_in"
        if self.stable_phase == "step_or_jump_in" and (
            (knee_angle >= 142.0 and (knee_delta is None or knee_delta >= 3.0))
            or (body_height_rising and body_height is not None and body_height >= 0.18 and torso_abs is not None and torso_abs < 25.0)
        ):
            return "broad_jump_takeoff"
        body_center_delta_x = None if self.takeoff_center_x is None else abs(body_center_x - self.takeoff_center_x)
        if self.stable_phase == "broad_jump_takeoff":
            if (body_center_delta_x is not None and body_center_delta_x >= 0.02) or (center_y_delta is not None and center_y_delta < -0.008):
                return "flight_or_move"
            return "broad_jump_takeoff"
        if self.stable_phase == "flight_or_move":
            if (center_y_delta is not None and center_y_delta > 0.004) or knee_angle < 155.0:
                return "landing"
            return "flight_or_move"
        if self.stable_phase == "landing":
            return "landing"
        if hands_low and torso_abs is not None and torso_abs >= 30.0:
            return "hands_down"
        if standing:
            return "stand"
        if self.stable_phase in {"hands_down", "chest_down", "step_or_jump_in"}:
            return "step_or_jump_in"
        return "unknown"

    def _advance_phase(self, raw_phase: str, timestamp_ms: int | None) -> tuple[str, int | None]:
        previous, _ = self._advance_confirmed_phase(raw_phase, self.confirmation_frames)
        previous_duration = None
        if self.stable_phase != previous:
            if timestamp_ms is not None and self.phase_started_ms is not None:
                previous_duration = max(0, timestamp_ms - self.phase_started_ms)
            self.phase_started_ms = timestamp_ms
        return previous, previous_duration

    def _cooldown_elapsed(self, timestamp_ms: int | None) -> bool:
        return timestamp_ms is None or self.last_rep_time_ms is None or timestamp_ms - self.last_rep_time_ms >= self.rep_cooldown_ms

    def _update_sequence(
        self,
        previous: str,
        body_center_x: float | None,
        timestamp_ms: int | None,
    ) -> None:
        self.chest_not_low = False
        self.no_broad_jump = False
        self.rep_sequence.just_completed = False
        if self.stable_phase == previous:
            return
        sequence_completed = self.rep_sequence.update(self.stable_phase)
        if self.stable_phase != "landing":
            self.extra_steps = False
            self.landing_step_events = 0
        if self.stable_phase == "chest_down":
            self.chest_down_seen = True
        elif self.stable_phase == "step_or_jump_in" and previous == "hands_down" and not self.chest_down_seen:
            self.chest_not_low = True
        elif self.stable_phase == "broad_jump_takeoff":
            self.takeoff_seen = self.chest_down_seen
            self.takeoff_center_x = body_center_x
        elif self.stable_phase == "landing":
            self.landing_started_ms = timestamp_ms
            self.landing_step_events = 0
            delta_x = 0.0 if body_center_x is None or self.takeoff_center_x is None else abs(body_center_x - self.takeoff_center_x)
            self.no_broad_jump = delta_x < self.jump_forward_min_delta_x
            if sequence_completed:
                self.rep_count += 1
                self.last_rep_time_ms = timestamp_ms
        elif self.stable_phase in {"reset", "stand"}:
            self.chest_down_seen = False
            self.takeoff_seen = False
            self.takeoff_center_x = None

    def _track_extra_steps(
        self,
        *,
        ankle_center_y: float | None,
        ankle_distance: float | None,
        timestamp_ms: int | None,
    ) -> None:
        if self.stable_phase != "landing" or timestamp_ms is None or self.landing_started_ms is None:
            return
        if timestamp_ms - self.landing_started_ms > self.extra_step_window_ms:
            return
        ankle_y_delta = None if ankle_center_y is None or self.previous_ankle_center_y is None else abs(ankle_center_y - self.previous_ankle_center_y)
        distance_delta = None if ankle_distance is None or self.previous_ankle_distance is None else abs(ankle_distance - self.previous_ankle_distance)
        if (ankle_y_delta is not None and ankle_y_delta > 0.012) or (distance_delta is not None and distance_delta > 0.018):
            self.landing_step_events += 1
        if self.landing_step_events >= 2:
            self.extra_steps = True

    def _feedback(
        self,
        *,
        visible_score: float,
        feet_stagger_score: float | None,
        hips_too_high: bool,
    ) -> list[FeedbackMessage]:
        if visible_score < self.min_visible_score:
            return [FeedbackMessage("warn", "LOW_VISIBILITY", "请保证全身入镜，尤其是手腕、肩、髋、膝、脚踝", max(0.2, 1.0 - visible_score))]
        messages: list[FeedbackMessage] = []
        if self.chest_not_low:
            messages.append(FeedbackMessage("warn", "CHEST_NOT_LOW", "俯卧阶段不够低，胸部需要明显接近地面", 0.8))
        if self.stable_phase in {"broad_jump_takeoff", "landing"} and feet_stagger_score is not None and feet_stagger_score > self.feet_stagger_warn:
            messages.append(FeedbackMessage("warn", "FEET_STAGGERED", "起跳或落地时双脚前后差较大，尽量双脚同时起跳落地", 0.4))
        if self.extra_steps:
            messages.append(FeedbackMessage("warn", "EXTRA_STEPS", "落地后出现小碎步，尽量稳定落地后再进入下一次", 0.65))
        if self.no_broad_jump:
            messages.append(FeedbackMessage("warn", "NO_BROAD_JUMP", "没有明显向前跳跃，burpee 后需要完成 broad jump", 0.75))
        if self.stable_phase == "chest_down" and hips_too_high:
            messages.append(FeedbackMessage("warn", "HIPS_TOO_HIGH_IN_BOTTOM", "底部时髋部过高，身体应更接近俯卧姿态", 0.75))
        return self.limit_feedback(messages)

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, object]:
        values = features if isinstance(features, dict) else {}
        current_timestamp = None if timestamp_ms is None else int(timestamp_ms)
        visible_score = self._visible_score(values)
        body_center_x = _safe_float(values.get("body_center_x"))
        body_center_y = _safe_float(values.get("body_center_y"))
        body_height = _safe_float(values.get("body_height_norm"))
        torso_angle = _safe_float(values.get("torso_angle"))
        knee_angle = _min_metric(values.get("left_knee_angle"), values.get("right_knee_angle"))
        hip_angle = _min_metric(values.get("left_hip_angle"), values.get("right_hip_angle"))
        wrist_height = _mean_metric(values.get("left_wrist_y"), values.get("right_wrist_y"))
        left_ankle_y = _safe_float(values.get("left_ankle_y"))
        right_ankle_y = _safe_float(values.get("right_ankle_y"))
        ankle_height = _mean_metric(left_ankle_y, right_ankle_y)
        ankle_distance = _safe_float(values.get("ankle_distance_norm"))
        feet_stagger_score = None if left_ankle_y is None or right_ankle_y is None else abs(left_ankle_y - right_ankle_y)
        shoulder_y = _safe_float(values.get("shoulder_center_y"))
        hip_y = _safe_float(values.get("hip_center_y"))
        hips_too_high = bool(shoulder_y is not None and hip_y is not None and shoulder_y - hip_y > 0.08)

        raw_phase = "unknown" if visible_score < self.min_visible_score else self._raw_phase(
            body_center_x=body_center_x,
            body_center_y=body_center_y,
            body_height=body_height,
            torso_angle=torso_angle,
            knee_angle=knee_angle,
            hip_angle=hip_angle,
            wrist_height=wrist_height,
            ankle_height=ankle_height,
        )
        previous, _ = self._advance_phase(raw_phase, current_timestamp)
        self._update_sequence(previous, body_center_x, current_timestamp)
        self._track_extra_steps(
            ankle_center_y=ankle_height,
            ankle_distance=ankle_distance,
            timestamp_ms=current_timestamp,
        )
        body_center_delta_x = None if body_center_x is None or self.takeoff_center_x is None else abs(body_center_x - self.takeoff_center_x)
        phase_duration = None if current_timestamp is None or self.phase_started_ms is None else max(0, current_timestamp - self.phase_started_ms)
        feedback_messages = self._feedback(
            visible_score=visible_score,
            feet_stagger_score=feet_stagger_score,
            hips_too_high=hips_too_high,
        )

        self.previous_body_center_x = body_center_x
        self.previous_body_center_y = body_center_y
        self.previous_body_height = body_height
        self.previous_knee_angle = knee_angle
        self.previous_ankle_center_y = ankle_height
        self.previous_ankle_distance = ankle_distance
        self.last_timestamp_ms = current_timestamp
        return {
            "action": self.action,
            "phase": self.phase,
            "rep_count": self.rep_count,
            "feedback_messages": feedback_messages,
            "debug": {
                "raw_phase": self.raw_phase,
                "stable_phase": self.stable_phase,
                "rep_count": self.rep_count,
                "body_center_x": body_center_x,
                "body_center_y": body_center_y,
                "body_center_delta_x": body_center_delta_x,
                "ankle_distance_norm": ankle_distance,
                "feet_stagger_score": feet_stagger_score,
                "torso_angle": torso_angle,
                "phase_duration_ms": phase_duration,
                "frames_in_phase": self.frames_in_phase,
                "config_name": self.config_name,
                "sensitivity": self.sensitivity,
                **self.rep_sequence.debug(),
            },
        }


__all__ = ["BurpeeBroadJumpAnalyzer"]
