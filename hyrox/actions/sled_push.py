from __future__ import annotations

from collections.abc import Mapping

from hyrox.base import BaseActionAnalyzer
from hyrox.config import load_sled_push_config
from hyrox.feedback import FeedbackMessage


SENSITIVITY_FRAME_DELTAS = {"low": 1, "medium": 0, "high": -1}
STEP_PHASE_HOLD_MS = 160


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


class SledPushAnalyzer(BaseActionAnalyzer):
    """Pose-only Sled Push posture monitor and approximate step counter."""

    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        config: Mapping[str, object] | None = None,
        config_name: str | None = None,
    ) -> None:
        if sensitivity not in SENSITIVITY_FRAME_DELTAS:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        values = dict(config or load_sled_push_config())
        visibility_min = _resolved_float(values.get("visibility_min"), 0.55)
        super().__init__(action="sled_push", min_visible_score=min(1.0, visibility_min))
        self.configure_feedback_limits(values)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or values.get("config_name") or "sled_push_default")
        self.drive_torso_angle_min = _resolved_float(values.get("drive_torso_angle_min"), 25.0)
        self.drive_torso_angle_max = _resolved_float(values.get("drive_torso_angle_max"), 65.0)
        self.too_upright_angle = _resolved_float(values.get("too_upright_angle"), 20.0)
        self.too_low_angle = _resolved_float(values.get("too_low_angle"), 70.0)
        self.leg_drive_knee_extension_min = _resolved_float(values.get("leg_drive_knee_extension_min"), 20.0)
        self.short_step_ankle_delta_min = _resolved_float(values.get("short_step_ankle_delta_min"), 0.04)
        base_frames = _resolved_int(values.get("stable_frames"), 3, minimum=1)
        self.confirmation_frames = max(1, base_frames + SENSITIVITY_FRAME_DELTAS[sensitivity])
        self.step_cooldown_ms = _resolved_int(values.get("step_cooldown_ms"), 250)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        sensitivity: str = "medium",
        config_name: str | None = None,
    ) -> SledPushAnalyzer:
        return cls(config=config, sensitivity=sensitivity, config_name=config_name)

    @classmethod
    def from_config_path(cls, path: str | None, *, sensitivity: str = "medium") -> SledPushAnalyzer:
        config = load_sled_push_config(path)
        return cls.from_config(
            config,
            sensitivity=sensitivity,
            config_name=str(config.get("config_name") or (path or "sled_push_default")),
        )

    def reset(self) -> None:
        super().reset()
        self.phase = "unknown"
        self.raw_phase = "unknown"
        self.stable_phase = "unknown"
        self.frames_in_phase = 0
        self.previous_body_center_x: float | None = None
        self.previous_left_ankle_y: float | None = None
        self.previous_right_ankle_y: float | None = None
        self.previous_ankle_distance: float | None = None
        self.previous_knee_angle: float | None = None
        self.previous_torso_angle: float | None = None
        self.last_step_time_ms: int | None = None
        self.step_hold_until_ms: int | None = None
        self.drive_start_knee_angle: float | None = None
        self.max_drive_knee_angle: float | None = None
        self.no_leg_drive = False
        self.short_steps = False
        self.hip_or_back_unstable = False

    def _visible_score(self, features: dict[str, object]) -> float:
        score = _safe_float(features.get("visible_score"))
        return max(0.0, min(1.0, score or 0.0))

    def _advance_phase(self, raw_phase: str) -> None:
        if raw_phase == "unknown":
            self.raw_phase = "unknown"
            self.stable_phase = "unknown"
            self.frames_in_phase = 1
        else:
            if raw_phase == self.raw_phase:
                self.frames_in_phase += 1
            else:
                self.raw_phase = raw_phase
                self.frames_in_phase = 1
            if self.frames_in_phase >= self.confirmation_frames:
                self.stable_phase = raw_phase
        self.phase = self.stable_phase

    def _step_cooldown_elapsed(self, timestamp_ms: int | None) -> bool:
        return timestamp_ms is None or self.last_step_time_ms is None or timestamp_ms - self.last_step_time_ms >= self.step_cooldown_ms

    def _feedback(
        self,
        *,
        visible_score: float,
        torso_angle: float | None,
    ) -> list[FeedbackMessage]:
        if visible_score < self.min_visible_score:
            return [FeedbackMessage("warn", "LOW_VISIBILITY", "请保证侧面全身入镜", max(0.2, 1.0 - visible_score))]
        messages: list[FeedbackMessage] = []
        torso_abs = None if torso_angle is None else abs(torso_angle)
        if self.stable_phase in {"setup", "drive", "step"} and torso_abs is not None and torso_abs < self.too_upright_angle:
            messages.append(FeedbackMessage("warn", "TORSO_TOO_UPRIGHT", "推雪橇时身体过直，适当前倾并用腿驱动", 0.8))
        if torso_abs is not None and torso_abs > self.too_low_angle:
            messages.append(FeedbackMessage("warn", "TORSO_TOO_LOW", "身体压得过低，保持稳定前倾，不要塌腰", 0.8))
        if self.short_steps:
            messages.append(FeedbackMessage("warn", "SHORT_STEPS", "步幅过小，尝试稳定连续地蹬地", 0.7))
        if self.no_leg_drive:
            messages.append(FeedbackMessage("warn", "NO_LEG_DRIVE", "腿部伸展不明显，推的力量应来自腿和髋", 0.75))
        if self.hip_or_back_unstable:
            messages.append(FeedbackMessage("warn", "HIP_TOO_HIGH_OR_BACK_ROUND", "髋部和躯干姿态不稳定，保持核心收紧", 0.65))
        return self.limit_feedback(messages)

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, object]:
        values = features if isinstance(features, dict) else {}
        current_timestamp = None if timestamp_ms is None else int(timestamp_ms)
        visible_score = self._visible_score(values)
        torso_angle = _safe_float(values.get("torso_angle"))
        torso_abs = None if torso_angle is None else abs(torso_angle)
        knee_angle = _mean_metric(values.get("left_knee_angle"), values.get("right_knee_angle"))
        body_center_x = _safe_float(values.get("body_center_x"))
        left_ankle_y = _safe_float(values.get("left_ankle_y"))
        right_ankle_y = _safe_float(values.get("right_ankle_y"))
        ankle_distance = _safe_float(values.get("ankle_distance_norm"))
        wrist_height = _mean_metric(values.get("left_wrist_y"), values.get("right_wrist_y"))
        shoulder_height = _safe_float(values.get("shoulder_center_y"))
        hip_tilt = _safe_float(values.get("hip_tilt"))

        body_center_delta_x = None if body_center_x is None or self.previous_body_center_x is None else abs(body_center_x - self.previous_body_center_x)
        left_delta = None if left_ankle_y is None or self.previous_left_ankle_y is None else abs(left_ankle_y - self.previous_left_ankle_y)
        right_delta = None if right_ankle_y is None or self.previous_right_ankle_y is None else abs(right_ankle_y - self.previous_right_ankle_y)
        distance_delta = None if ankle_distance is None or self.previous_ankle_distance is None else abs(ankle_distance - self.previous_ankle_distance)
        ankle_delta = max((value for value in (left_delta, right_delta, distance_delta) if value is not None), default=None)
        knee_extension = None if knee_angle is None or self.previous_knee_angle is None else knee_angle - self.previous_knee_angle
        wrists_ready = wrist_height is not None and shoulder_height is not None and abs(wrist_height - shoulder_height) <= 0.25
        drive_posture = torso_abs is not None and self.drive_torso_angle_min <= torso_abs <= 90.0

        step_event = ankle_delta is not None and ankle_delta >= self.short_step_ankle_delta_min
        if step_event:
            self.step_hold_until_ms = None if current_timestamp is None else current_timestamp + STEP_PHASE_HOLD_MS
            if self._step_cooldown_elapsed(current_timestamp):
                self.rep_count += 1
                self.last_step_time_ms = current_timestamp
                extension = 0.0 if knee_angle is None or self.drive_start_knee_angle is None else max(0.0, knee_angle - self.drive_start_knee_angle)
                self.no_leg_drive = extension < self.leg_drive_knee_extension_min
        step_held = step_event or (
            current_timestamp is not None
            and self.step_hold_until_ms is not None
            and current_timestamp <= self.step_hold_until_ms
        )
        moving_forward = body_center_delta_x is not None and body_center_delta_x >= 0.003
        extending = knee_extension is not None and knee_extension >= 3.0

        if visible_score < self.min_visible_score:
            raw_phase = "unknown"
        elif torso_abs is None:
            raw_phase = "unknown"
        elif step_held and torso_abs >= self.too_upright_angle:
            raw_phase = "step"
        elif drive_posture and (moving_forward or extending):
            raw_phase = "drive"
        elif torso_abs < self.drive_torso_angle_min:
            raw_phase = "reset"
        elif wrists_ready:
            raw_phase = "setup" if self.stable_phase not in {"drive", "step"} else "drive"
        elif drive_posture and self.stable_phase in {"drive", "step"}:
            raw_phase = "drive"
        else:
            raw_phase = "reset"
        self._advance_phase(raw_phase)

        if self.stable_phase == "drive":
            if self.drive_start_knee_angle is None:
                self.drive_start_knee_angle = knee_angle
            if knee_angle is not None:
                self.max_drive_knee_angle = knee_angle if self.max_drive_knee_angle is None else max(self.max_drive_knee_angle, knee_angle)
        elif self.stable_phase in {"setup", "reset", "unknown"}:
            self.drive_start_knee_angle = knee_angle
            self.max_drive_knee_angle = knee_angle
            if self.stable_phase in {"reset", "unknown"}:
                self.no_leg_drive = False

        self.short_steps = bool(
            ankle_delta is not None
            and 0.008 <= ankle_delta < self.short_step_ankle_delta_min
            and moving_forward
            and self.stable_phase in {"drive", "step"}
        )
        torso_delta = None if torso_angle is None or self.previous_torso_angle is None else abs(torso_angle - self.previous_torso_angle)
        self.hip_or_back_unstable = bool(
            self.stable_phase in {"drive", "step"}
            and ((hip_tilt is not None and abs(hip_tilt) > 0.08) or (torso_delta is not None and torso_delta > 15.0))
        )
        lean_score = 0.0 if torso_abs is None else max(0.0, 1.0 - abs(torso_abs - 45.0) / 45.0)
        drive_score = min(
            1.0,
            0.45 * lean_score
            + 0.25 * float(moving_forward)
            + 0.20 * float(extending)
            + 0.10 * float(wrists_ready),
        )
        feedback_messages = self._feedback(visible_score=visible_score, torso_angle=torso_angle)

        self.previous_body_center_x = body_center_x
        self.previous_left_ankle_y = left_ankle_y
        self.previous_right_ankle_y = right_ankle_y
        self.previous_ankle_distance = ankle_distance
        self.previous_knee_angle = knee_angle
        self.previous_torso_angle = torso_angle
        self.last_timestamp_ms = current_timestamp
        return {
            "action": self.action,
            "phase": self.phase,
            "rep_count": self.rep_count,
            "feedback_messages": feedback_messages,
            "debug": {
                "raw_phase": self.raw_phase,
                "stable_phase": self.stable_phase,
                "step_count": self.rep_count,
                "torso_angle": torso_angle,
                "knee_angle_mean": knee_angle,
                "ankle_delta": ankle_delta,
                "body_center_delta_x": body_center_delta_x,
                "drive_score": drive_score,
                "frames_in_phase": self.frames_in_phase,
                "config_name": self.config_name,
                "sensitivity": self.sensitivity,
            },
        }


__all__ = ["SledPushAnalyzer"]
