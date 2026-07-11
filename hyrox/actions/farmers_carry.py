from __future__ import annotations

from collections.abc import Mapping

from hyrox.base import BaseActionAnalyzer
from hyrox.config import load_farmers_carry_config
from hyrox.feedback import FeedbackMessage


SENSITIVITY_FRAME_DELTAS = {"low": 1, "medium": 0, "high": -1}
HORIZONTAL_MOTION_MIN = 0.004
GAIT_CHANGE_MIN = 0.012
KNEE_MOTION_MIN_DEGREES = 4.0
STANDING_KNEE_MIN = 145.0
STANDING_HIP_MIN = 140.0


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


def _min_metric(*values: object) -> float | None:
    numbers = [_safe_float(value) for value in values]
    valid = [number for number in numbers if number is not None]
    return min(valid) if valid else None


class FarmersCarryAnalyzer(BaseActionAnalyzer):
    """Single-camera posture monitor for the Farmers Carry station.

    This analyzer deliberately does not infer carried weight, travelled distance,
    or event completion. Movement is only a short-term pose cue used to separate
    ready/carrying/rest states.
    """

    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        config: Mapping[str, object] | None = None,
        config_name: str | None = None,
    ) -> None:
        if sensitivity not in SENSITIVITY_FRAME_DELTAS:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        config_data = dict(config or load_farmers_carry_config())
        visibility_min = _resolved_float(config_data.get("visibility_min"), 0.55)
        super().__init__(action="farmers_carry", min_visible_score=min(1.0, visibility_min))
        self.configure_feedback_limits(config_data)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or config_data.get("config_name") or "farmers_carry_default")
        self.shoulder_tilt_warn = _resolved_float(config_data.get("shoulder_tilt_warn"), 0.08)
        self.hip_tilt_warn = _resolved_float(config_data.get("hip_tilt_warn"), 0.08)
        self.torso_lean_warn = _resolved_float(config_data.get("torso_lean_warn"), 25.0)
        self.arms_down_margin = _resolved_float(config_data.get("arms_down_margin"), 0.05)
        base_frames = _resolved_int(config_data.get("stable_frames"), 3, minimum=1)
        self.confirmation_frames = max(1, base_frames + SENSITIVITY_FRAME_DELTAS[sensitivity])
        self.rest_timeout_ms = _resolved_int(config_data.get("rest_timeout_ms"), 1200)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        sensitivity: str = "medium",
        config_name: str | None = None,
    ) -> FarmersCarryAnalyzer:
        return cls(config=config, sensitivity=sensitivity, config_name=config_name)

    @classmethod
    def from_config_path(
        cls,
        path: str | None,
        *,
        sensitivity: str = "medium",
    ) -> FarmersCarryAnalyzer:
        config = load_farmers_carry_config(path)
        return cls.from_config(
            config,
            sensitivity=sensitivity,
            config_name=str(config.get("config_name") or (path or "farmers_carry_default")),
        )

    def reset(self) -> None:
        super().reset()
        self.phase = "unknown"
        self.raw_phase = "unknown"
        self.stable_phase = "unknown"
        self.frames_in_phase = 0
        self.previous_body_center_x: float | None = None
        self.previous_body_center_y: float | None = None
        self.previous_ankle_distance: float | None = None
        self.previous_min_knee_angle: float | None = None
        self.previous_torso_angle: float | None = None
        self.previous_shoulder_tilt: float | None = None
        self.previous_hip_tilt: float | None = None
        self.last_motion_ms: int | None = None
        self.motion_detected = False
        self.unstable_carry = False

    def _visible_score(self, features: dict[str, object] | None) -> float:
        if not isinstance(features, dict):
            return 0.0
        score = _safe_float(features.get("visible_score"))
        return max(0.0, min(1.0, score or 0.0))

    def _advance_phase(self, raw_phase: str) -> None:
        if raw_phase == "unknown":
            self.raw_phase = raw_phase
            self.stable_phase = raw_phase
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

    def _feedback(
        self,
        *,
        visible_score: float,
        shoulder_tilt: float | None,
        hip_tilt: float | None,
        torso_angle: float | None,
        arms_down: bool,
    ) -> list[FeedbackMessage]:
        if visible_score < self.min_visible_score:
            return [
                FeedbackMessage(
                    level="warn",
                    code="LOW_VISIBILITY",
                    text="请保证全身入镜，尤其是肩、髋、膝、脚踝",
                    confidence=max(0.2, 1.0 - visible_score),
                )
            ]

        messages: list[FeedbackMessage] = []
        shoulder_excess = shoulder_tilt is not None and abs(shoulder_tilt) > self.shoulder_tilt_warn
        hip_excess = hip_tilt is not None and abs(hip_tilt) > self.hip_tilt_warn
        if shoulder_excess or hip_excess:
            messages.append(
                FeedbackMessage("warn", "LEAN_LEFT_RIGHT", "身体左右倾斜过多，保持躯干居中", 0.75)
            )
        if shoulder_excess:
            messages.append(
                FeedbackMessage("warn", "SHOULDERS_UNEVEN", "左右肩高度差过大，注意两侧负重平衡", 0.8)
            )
        if not arms_down:
            messages.append(
                FeedbackMessage("warn", "ARMS_NOT_DOWN", "手臂没有自然下垂，保持双臂伸展在身体两侧", 0.8)
            )
        if torso_angle is not None and abs(torso_angle) > self.torso_lean_warn:
            messages.append(
                FeedbackMessage("warn", "TORSO_LEAN", "身体前后倾斜过多，保持核心稳定", 0.8)
            )
        if self.stable_phase == "carrying" and self.unstable_carry:
            messages.append(
                FeedbackMessage("warn", "UNSTABLE_CARRY", "行走晃动较大，降低速度并保持步伐稳定", 0.7)
            )
        return self.limit_feedback(messages)

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, object]:
        values = features if isinstance(features, dict) else {}
        current_timestamp = None if timestamp_ms is None else int(timestamp_ms)
        visible_score = self._visible_score(values)
        shoulder_tilt = _safe_float(values.get("shoulder_tilt"))
        hip_tilt = _safe_float(values.get("hip_tilt"))
        torso_angle = _safe_float(values.get("torso_angle"))
        body_center_x = _safe_float(values.get("body_center_x"))
        body_center_y = _safe_float(values.get("body_center_y"))
        body_height = _safe_float(values.get("body_height_norm"))
        ankle_distance = _safe_float(values.get("ankle_distance_norm"))
        min_knee = _min_metric(values.get("left_knee_angle"), values.get("right_knee_angle"))
        min_hip = _min_metric(values.get("left_hip_angle"), values.get("right_hip_angle"))
        left_wrist_to_hip = _safe_float(values.get("left_wrist_to_hip_y"))
        right_wrist_to_hip = _safe_float(values.get("right_wrist_to_hip_y"))

        wrists_available = left_wrist_to_hip is not None and right_wrist_to_hip is not None
        # ``arms_down_margin`` is a tolerance around hip height. Pose landmarks
        # often place a loaded wrist slightly above the hip when the handle or
        # forearm is occluded, so values down to ``-margin`` still count as down.
        wrists_low = wrists_available and min(left_wrist_to_hip, right_wrist_to_hip) >= -self.arms_down_margin
        arms_down = bool(wrists_low)
        angle_standing = (
            min_knee is not None
            and min_knee >= STANDING_KNEE_MIN
            and (min_hip is None or min_hip >= STANDING_HIP_MIN)
        )
        upright_body = (
            body_height is not None
            and body_height >= 0.25
            and (torso_angle is None or abs(torso_angle) <= 45.0)
        )
        standing = bool(angle_standing or upright_body)

        horizontal_delta = None if body_center_x is None or self.previous_body_center_x is None else abs(body_center_x - self.previous_body_center_x)
        gait_delta = None if ankle_distance is None or self.previous_ankle_distance is None else abs(ankle_distance - self.previous_ankle_distance)
        knee_delta = None if min_knee is None or self.previous_min_knee_angle is None else abs(min_knee - self.previous_min_knee_angle)
        self.motion_detected = bool(
            (horizontal_delta is not None and horizontal_delta >= HORIZONTAL_MOTION_MIN)
            or (gait_delta is not None and gait_delta >= GAIT_CHANGE_MIN)
            or (knee_delta is not None and knee_delta >= KNEE_MOTION_MIN_DEGREES)
        )

        vertical_delta = None if body_center_y is None or self.previous_body_center_y is None else abs(body_center_y - self.previous_body_center_y)
        torso_delta = None if torso_angle is None or self.previous_torso_angle is None else abs(torso_angle - self.previous_torso_angle)
        shoulder_delta = None if shoulder_tilt is None or self.previous_shoulder_tilt is None else abs(shoulder_tilt - self.previous_shoulder_tilt)
        hip_delta = None if hip_tilt is None or self.previous_hip_tilt is None else abs(hip_tilt - self.previous_hip_tilt)
        self.unstable_carry = bool(
            (vertical_delta is not None and vertical_delta > 0.025)
            or (torso_delta is not None and torso_delta > 10.0)
            or (shoulder_delta is not None and shoulder_delta > 0.04)
            or (hip_delta is not None and hip_delta > 0.04)
        )

        if self.motion_detected and current_timestamp is not None:
            self.last_motion_ms = current_timestamp
        elif self.last_motion_ms is None and current_timestamp is not None:
            self.last_motion_ms = current_timestamp

        stationary_ms = (
            None
            if current_timestamp is None or self.last_motion_ms is None
            else max(0, current_timestamp - self.last_motion_ms)
        )
        if visible_score < self.min_visible_score:
            raw_phase = "unknown"
        elif not wrists_available or not arms_down:
            raw_phase = "rest"
        elif standing and self.motion_detected:
            raw_phase = "carrying"
        elif standing and stationary_ms is not None and stationary_ms >= self.rest_timeout_ms:
            raw_phase = "rest"
        elif standing:
            raw_phase = "ready"
        else:
            raw_phase = "unknown"
        self._advance_phase(raw_phase)

        carrying_score = (
            0.45 * float(self.motion_detected)
            + 0.25 * float(arms_down)
            + 0.15 * float(standing)
            + 0.15 * visible_score
        )
        wrist_to_hip = (
            None
            if left_wrist_to_hip is None or right_wrist_to_hip is None
            else min(left_wrist_to_hip, right_wrist_to_hip)
        )
        feedback_messages = self._feedback(
            visible_score=visible_score,
            shoulder_tilt=shoulder_tilt,
            hip_tilt=hip_tilt,
            torso_angle=torso_angle,
            arms_down=arms_down,
        )

        self.previous_body_center_x = body_center_x
        self.previous_body_center_y = body_center_y
        self.previous_ankle_distance = ankle_distance
        self.previous_min_knee_angle = min_knee
        self.previous_torso_angle = torso_angle
        self.previous_shoulder_tilt = shoulder_tilt
        self.previous_hip_tilt = hip_tilt
        self.last_timestamp_ms = current_timestamp
        return {
            "action": self.action,
            "phase": self.phase,
            "rep_count": 0,
            "feedback_messages": feedback_messages,
            "debug": {
                "shoulder_tilt": shoulder_tilt,
                "hip_tilt": hip_tilt,
                "torso_angle": torso_angle,
                "wrist_to_hip": wrist_to_hip,
                "carrying_score": carrying_score,
                "raw_phase": self.raw_phase,
                "stable_phase": self.stable_phase,
                "frames_in_phase": self.frames_in_phase,
                "motion_detected": self.motion_detected,
                "stationary_ms": stationary_ms,
                "config_name": self.config_name,
                "sensitivity": self.sensitivity,
            },
        }


__all__ = ["FarmersCarryAnalyzer"]
