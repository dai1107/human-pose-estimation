from __future__ import annotations

from collections.abc import Mapping

from hyrox.base import BaseActionAnalyzer
from hyrox.config import load_skierg_config
from hyrox.feedback import FeedbackMessage


SENSITIVITY_FRAME_DELTAS = {"low": 1, "medium": 0, "high": -1}
WRIST_MOTION_MIN = 0.004


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


class SkiErgAnalyzer(BaseActionAnalyzer):
    """Front/oblique-view SkiErg pull analyzer based on pose landmarks only."""

    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        config: Mapping[str, object] | None = None,
        config_name: str | None = None,
    ) -> None:
        if sensitivity not in SENSITIVITY_FRAME_DELTAS:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        values = dict(config or load_skierg_config())
        visibility_min = _resolved_float(values.get("visibility_min"), 0.50)
        super().__init__(action="skierg", min_visible_score=min(1.0, visibility_min))
        self.configure_feedback_limits(values)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or values.get("config_name") or "skierg_default")
        self.top_wrist_above_shoulder_margin = _resolved_float(values.get("top_wrist_above_shoulder_margin"), 0.03)
        self.bottom_wrist_below_chest_margin = _resolved_float(values.get("bottom_wrist_below_chest_margin"), 0.05)
        self.hip_hinge_torso_angle_min = _resolved_float(values.get("hip_hinge_torso_angle_min"), 15.0)
        self.too_much_squat_knee_angle_max = _resolved_float(values.get("too_much_squat_knee_angle_max"), 110.0)
        self.wrist_asymmetry_warn = _resolved_float(values.get("wrist_asymmetry_warn"), 0.08)
        base_frames = _resolved_int(values.get("stable_frames"), 3, minimum=1)
        self.confirmation_frames = max(1, base_frames + SENSITIVITY_FRAME_DELTAS[sensitivity])
        self.pull_cooldown_ms = _resolved_int(values.get("pull_cooldown_ms"), 350)
        self.min_phase_duration_ms = _resolved_int(values.get("min_phase_duration_ms"), 100)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        sensitivity: str = "medium",
        config_name: str | None = None,
    ) -> SkiErgAnalyzer:
        return cls(config=config, sensitivity=sensitivity, config_name=config_name)

    @classmethod
    def from_config_path(cls, path: str | None, *, sensitivity: str = "medium") -> SkiErgAnalyzer:
        config = load_skierg_config(path)
        return cls.from_config(
            config,
            sensitivity=sensitivity,
            config_name=str(config.get("config_name") or (path or "skierg_default")),
        )

    def reset(self) -> None:
        super().reset()
        self.phase = "unknown"
        self.raw_phase = "unknown"
        self.stable_phase = "unknown"
        self.frames_in_phase = 0
        self.phase_started_ms: int | None = None
        self.previous_wrist_height: float | None = None
        self.pull_down_seen = False
        self.bottom_seen = False
        self.return_seen = False
        self.last_rep_time_ms: int | None = None
        self.rushed_return = False
        self.arms_not_high_enough = False

    def _visible_score(self, features: dict[str, object]) -> float:
        score = _safe_float(features.get("upper_body_visible_score"))
        if score is None:
            score = _safe_float(features.get("visible_score"))
        return max(0.0, min(1.0, score or 0.0))

    def _raw_phase(
        self,
        *,
        wrist_height: float | None,
        wrist_above_shoulder: float | None,
        wrist_below_chest: float | None,
        torso_angle: float | None,
        knee_angle: float | None,
    ) -> str:
        if wrist_height is None:
            return "unknown"
        wrist_delta = None if self.previous_wrist_height is None else wrist_height - self.previous_wrist_height
        upright = torso_angle is None or abs(torso_angle) < self.hip_hinge_torso_angle_min
        if wrist_above_shoulder is not None and wrist_above_shoulder >= self.top_wrist_above_shoulder_margin and upright:
            return "top"
        if wrist_delta is not None and wrist_delta <= -WRIST_MOTION_MIN and self.stable_phase in {"bottom", "return"}:
            return "return"
        hinged = torso_angle is not None and abs(torso_angle) >= self.hip_hinge_torso_angle_min
        knees_flexed = knee_angle is not None and knee_angle < 155.0
        if wrist_below_chest is not None and wrist_below_chest >= self.bottom_wrist_below_chest_margin and (hinged or knees_flexed):
            return "bottom"
        if wrist_delta is not None and wrist_delta >= WRIST_MOTION_MIN:
            return "pull_down"
        if wrist_delta is not None and wrist_delta <= -WRIST_MOTION_MIN:
            return "return"
        if self.raw_phase in {"pull_down", "return"}:
            return self.raw_phase
        if self.stable_phase in {"top", "pull_down"}:
            return "pull_down"
        if self.stable_phase in {"bottom", "return"}:
            return "return"
        return "unknown"

    def _advance_phase(self, raw_phase: str, timestamp_ms: int | None) -> tuple[str, int | None]:
        previous = self.stable_phase
        previous_duration = None
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
        if self.stable_phase != previous:
            if timestamp_ms is not None and self.phase_started_ms is not None:
                previous_duration = max(0, timestamp_ms - self.phase_started_ms)
            self.phase_started_ms = timestamp_ms
        self.phase = self.stable_phase
        return previous, previous_duration

    def _cooldown_elapsed(self, timestamp_ms: int | None) -> bool:
        return timestamp_ms is None or self.last_rep_time_ms is None or timestamp_ms - self.last_rep_time_ms >= self.pull_cooldown_ms

    def _update_sequence(self, previous: str, previous_duration: int | None, timestamp_ms: int | None) -> None:
        self.rushed_return = False
        self.arms_not_high_enough = False
        if self.stable_phase == previous:
            return
        if self.stable_phase == "pull_down":
            if previous == "return":
                self.arms_not_high_enough = True
            self.pull_down_seen = previous == "top" or self.pull_down_seen
        elif self.stable_phase == "bottom" and self.pull_down_seen:
            self.bottom_seen = True
        elif self.stable_phase == "return" and self.bottom_seen:
            self.return_seen = True
        elif self.stable_phase == "top":
            self.rushed_return = previous == "return" and previous_duration is not None and previous_duration < self.min_phase_duration_ms
            if self.pull_down_seen and self.bottom_seen and self.return_seen and self._cooldown_elapsed(timestamp_ms):
                self.rep_count += 1
                self.last_rep_time_ms = timestamp_ms
            self.pull_down_seen = False
            self.bottom_seen = False
            self.return_seen = False

    def _feedback(
        self,
        *,
        visible_score: float,
        wrist_below_chest: float | None,
        wrist_asymmetry: float | None,
        torso_angle: float | None,
        knee_angle: float | None,
    ) -> list[FeedbackMessage]:
        if visible_score < self.min_visible_score:
            return [FeedbackMessage("warn", "LOW_VISIBILITY", "请保证上半身和手腕入镜", max(0.2, 1.0 - visible_score))]
        messages: list[FeedbackMessage] = []
        if self.arms_not_high_enough:
            messages.append(FeedbackMessage("warn", "ARMS_NOT_HIGH_ENOUGH", "回到顶部时手没有充分上举", 0.8))
        if wrist_below_chest is not None and wrist_below_chest >= self.bottom_wrist_below_chest_margin and (torso_angle is None or abs(torso_angle) < self.hip_hinge_torso_angle_min):
            messages.append(FeedbackMessage("warn", "NO_HIP_HINGE", "下拉时髋部折叠不明显，尝试用核心和髋部发力", 0.8))
        if self.stable_phase == "bottom" and knee_angle is not None and knee_angle <= self.too_much_squat_knee_angle_max:
            messages.append(FeedbackMessage("warn", "TOO_MUCH_SQUAT", "下拉时下蹲过多，动作应以髋部折叠和下拉为主", 0.8))
        if wrist_asymmetry is not None and wrist_asymmetry > self.wrist_asymmetry_warn:
            messages.append(FeedbackMessage("warn", "ASYMMETRIC_PULL", "左右手高度差过大，保持双手同步下拉", 0.75))
        if self.rushed_return:
            messages.append(FeedbackMessage("warn", "RUSHED_RETURN", "回程过快，保持稳定节奏", 0.75))
        return self.limit_feedback(messages)

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, object]:
        values = features if isinstance(features, dict) else {}
        current_timestamp = None if timestamp_ms is None else int(timestamp_ms)
        visible_score = self._visible_score(values)
        left_wrist_y = _safe_float(values.get("left_wrist_y"))
        right_wrist_y = _safe_float(values.get("right_wrist_y"))
        wrist_height = _mean_metric(left_wrist_y, right_wrist_y)
        wrist_asymmetry = None if left_wrist_y is None or right_wrist_y is None else abs(left_wrist_y - right_wrist_y)
        left_above = _safe_float(values.get("left_wrist_above_shoulder"))
        right_above = _safe_float(values.get("right_wrist_above_shoulder"))
        wrist_above_shoulder = None if left_above is None or right_above is None else min(left_above, right_above)
        shoulder_y = _safe_float(values.get("shoulder_center_y"))
        hip_y = _safe_float(values.get("hip_center_y"))
        chest_y = None if shoulder_y is None or hip_y is None else (shoulder_y + hip_y) / 2.0
        wrist_below_chest = None if wrist_height is None or chest_y is None else wrist_height - chest_y
        torso_angle = _safe_float(values.get("torso_angle"))
        knee_angle = _mean_metric(values.get("left_knee_angle"), values.get("right_knee_angle"))
        raw_phase = "unknown" if visible_score < self.min_visible_score else self._raw_phase(
            wrist_height=wrist_height,
            wrist_above_shoulder=wrist_above_shoulder,
            wrist_below_chest=wrist_below_chest,
            torso_angle=torso_angle,
            knee_angle=knee_angle,
        )
        previous, previous_duration = self._advance_phase(raw_phase, current_timestamp)
        self._update_sequence(previous, previous_duration, current_timestamp)
        phase_duration = None if current_timestamp is None or self.phase_started_ms is None else max(0, current_timestamp - self.phase_started_ms)
        feedback_messages = self._feedback(
            visible_score=visible_score,
            wrist_below_chest=wrist_below_chest,
            wrist_asymmetry=wrist_asymmetry,
            torso_angle=torso_angle,
            knee_angle=knee_angle,
        )
        self.previous_wrist_height = wrist_height
        self.last_timestamp_ms = current_timestamp
        return {
            "action": self.action,
            "phase": self.phase,
            "rep_count": self.rep_count,
            "feedback_messages": feedback_messages,
            "debug": {
                "raw_phase": self.raw_phase,
                "stable_phase": self.stable_phase,
                "pull_count": self.rep_count,
                "wrist_height_mean": wrist_height,
                "wrist_asymmetry": wrist_asymmetry,
                "torso_angle": torso_angle,
                "knee_angle_mean": knee_angle,
                "phase_duration_ms": phase_duration,
                "frames_in_phase": self.frames_in_phase,
                "config_name": self.config_name,
                "sensitivity": self.sensitivity,
            },
        }


__all__ = ["SkiErgAnalyzer"]
