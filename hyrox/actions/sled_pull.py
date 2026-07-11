from __future__ import annotations

from collections.abc import Mapping

from hyrox.base import BaseActionAnalyzer
from hyrox.config import load_sled_pull_config
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


class SledPullAnalyzer(BaseActionAnalyzer):
    """Pose-only Sled Pull cycle analyzer; no rope or sled detection."""

    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        config: Mapping[str, object] | None = None,
        config_name: str | None = None,
    ) -> None:
        if sensitivity not in SENSITIVITY_FRAME_DELTAS:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        values = dict(config or load_sled_pull_config())
        visibility_min = _resolved_float(values.get("visibility_min"), 0.55)
        super().__init__(action="sled_pull", min_visible_score=min(1.0, visibility_min))
        self.configure_feedback_limits(values)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or values.get("config_name") or "sled_pull_default")
        self.not_standing_knee_angle_max = _resolved_float(values.get("not_standing_knee_angle_max"), 95.0)
        self.not_standing_body_center_y_max = _resolved_float(values.get("not_standing_body_center_y_max"), 0.75)
        self.over_lean_back_angle = _resolved_float(values.get("over_lean_back_angle"), 35.0)
        self.pull_elbow_delta_min = _resolved_float(values.get("pull_elbow_delta_min"), 25.0)
        self.hip_knee_drive_delta_min = _resolved_float(values.get("hip_knee_drive_delta_min"), 8.0)
        self.wrist_asymmetry_warn = _resolved_float(values.get("wrist_asymmetry_warn"), 0.08)
        base_frames = _resolved_int(values.get("stable_frames"), 3, minimum=1)
        self.confirmation_frames = max(1, base_frames + SENSITIVITY_FRAME_DELTAS[sensitivity])
        self.pull_cooldown_ms = _resolved_int(values.get("pull_cooldown_ms"), 350)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        sensitivity: str = "medium",
        config_name: str | None = None,
    ) -> SledPullAnalyzer:
        return cls(config=config, sensitivity=sensitivity, config_name=config_name)

    @classmethod
    def from_config_path(cls, path: str | None, *, sensitivity: str = "medium") -> SledPullAnalyzer:
        config = load_sled_pull_config(path)
        return cls.from_config(
            config,
            sensitivity=sensitivity,
            config_name=str(config.get("config_name") or (path or "sled_pull_default")),
        )

    def reset(self) -> None:
        super().reset()
        self.phase = "unknown"
        self.raw_phase = "unknown"
        self.stable_phase = "unknown"
        self.frames_in_phase = 0
        self.previous_elbow_angle: float | None = None
        self.reach_elbow_angle: float | None = None
        self.pull_min_elbow_angle: float | None = None
        self.pull_start_knee_angle: float | None = None
        self.pull_start_hip_angle: float | None = None
        self.max_lower_body_delta = 0.0
        self.last_rep_time_ms: int | None = None
        self.no_clear_pull = False
        self.arms_only_pull = False

    def _visible_score(self, features: dict[str, object]) -> float:
        score = _safe_float(features.get("visible_score"))
        return max(0.0, min(1.0, score or 0.0))

    def _raw_phase(self, elbow_angle: float | None, elbow_delta: float | None) -> str:
        if elbow_angle is None:
            return "unknown"
        reach_amplitude = 0.0 if self.reach_elbow_angle is None else max(0.0, self.reach_elbow_angle - elbow_angle)
        if self.stable_phase == "pull":
            if elbow_delta is not None and elbow_delta >= 3.0:
                return "recover"
            return "pull"
        if self.stable_phase == "recover":
            if elbow_angle >= 140.0 and (elbow_delta is None or abs(elbow_delta) < 4.0):
                return "reach"
            return "recover"
        if self.stable_phase == "reach":
            if reach_amplitude >= 8.0 or (elbow_delta is not None and elbow_delta <= -3.0):
                return "pull"
            return "reach"
        if elbow_angle >= 145.0:
            return "reach"
        return "ready"

    def _advance_phase(self, raw_phase: str) -> str:
        previous = self.stable_phase
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
        return previous

    def _cooldown_elapsed(self, timestamp_ms: int | None) -> bool:
        return timestamp_ms is None or self.last_rep_time_ms is None or timestamp_ms - self.last_rep_time_ms >= self.pull_cooldown_ms

    def _update_cycle(
        self,
        *,
        previous_phase: str,
        elbow_angle: float | None,
        knee_angle: float | None,
        hip_angle: float | None,
        timestamp_ms: int | None,
    ) -> None:
        self.no_clear_pull = False
        self.arms_only_pull = False
        if self.stable_phase == "reach":
            if previous_phase != "reach":
                self.reach_elbow_angle = elbow_angle
                self.pull_min_elbow_angle = elbow_angle
            elif elbow_angle is not None:
                self.reach_elbow_angle = elbow_angle if self.reach_elbow_angle is None else max(self.reach_elbow_angle, elbow_angle)
            return
        if self.stable_phase == "pull":
            if previous_phase != "pull":
                self.pull_start_knee_angle = knee_angle
                self.pull_start_hip_angle = hip_angle
                self.max_lower_body_delta = 0.0
            if elbow_angle is not None:
                self.pull_min_elbow_angle = elbow_angle if self.pull_min_elbow_angle is None else min(self.pull_min_elbow_angle, elbow_angle)
            deltas: list[float] = []
            if knee_angle is not None and self.pull_start_knee_angle is not None:
                deltas.append(abs(knee_angle - self.pull_start_knee_angle))
            if hip_angle is not None and self.pull_start_hip_angle is not None:
                deltas.append(abs(hip_angle - self.pull_start_hip_angle))
            if deltas:
                self.max_lower_body_delta = max(self.max_lower_body_delta, max(deltas))
            return
        if self.stable_phase == "recover" and previous_phase == "pull":
            pull_amplitude = (
                0.0
                if self.reach_elbow_angle is None or self.pull_min_elbow_angle is None
                else max(0.0, self.reach_elbow_angle - self.pull_min_elbow_angle)
            )
            clear_pull = pull_amplitude >= self.pull_elbow_delta_min
            self.no_clear_pull = not clear_pull
            self.arms_only_pull = clear_pull and self.max_lower_body_delta < self.hip_knee_drive_delta_min
            if clear_pull and self._cooldown_elapsed(timestamp_ms):
                self.rep_count += 1
                self.last_rep_time_ms = timestamp_ms

    def _feedback(
        self,
        *,
        visible_score: float,
        not_standing: bool,
        torso_angle: float | None,
        wrist_asymmetry: float | None,
    ) -> list[FeedbackMessage]:
        if visible_score < self.min_visible_score:
            return [FeedbackMessage("warn", "LOW_VISIBILITY", "请保证上半身、髋、膝、踝都在画面内", max(0.2, 1.0 - visible_score))]
        messages: list[FeedbackMessage] = []
        if not_standing:
            messages.append(FeedbackMessage("warn", "NOT_STANDING", "拉雪橇时需要保持站立，不要坐下或跪拉", 0.8))
        if torso_angle is not None and abs(torso_angle) > self.over_lean_back_angle:
            messages.append(FeedbackMessage("warn", "OVER_LEAN_BACK", "后仰过多，保持身体稳定，避免失去平衡", 0.75))
        if self.arms_only_pull:
            messages.append(FeedbackMessage("warn", "ARMS_ONLY_PULL", "不要只用手臂拉，配合髋腿发力", 0.75))
        if self.no_clear_pull:
            messages.append(FeedbackMessage("warn", "NO_CLEAR_PULL", "没有检测到明显拉绳动作，尝试让手臂前伸后再拉回", 0.75))
        if wrist_asymmetry is not None and wrist_asymmetry > self.wrist_asymmetry_warn:
            messages.append(FeedbackMessage("warn", "ASYMMETRIC_PULL", "左右手动作不一致，保持双手同步", 0.7))
        return self.limit_feedback(messages)

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, object]:
        values = features if isinstance(features, dict) else {}
        current_timestamp = None if timestamp_ms is None else int(timestamp_ms)
        visible_score = self._visible_score(values)
        left_elbow = _safe_float(values.get("left_elbow_angle"))
        right_elbow = _safe_float(values.get("right_elbow_angle"))
        elbow_angle = _mean_metric(left_elbow, right_elbow)
        elbow_delta = None if elbow_angle is None or self.previous_elbow_angle is None else elbow_angle - self.previous_elbow_angle
        knee_angle = _mean_metric(values.get("left_knee_angle"), values.get("right_knee_angle"))
        hip_angle = _mean_metric(values.get("left_hip_angle"), values.get("right_hip_angle"))
        torso_angle = _safe_float(values.get("torso_angle"))
        body_center_y = _safe_float(values.get("body_center_y"))
        wrist_distance = _safe_float(values.get("wrist_distance_norm"))
        left_wrist_y = _safe_float(values.get("left_wrist_y"))
        right_wrist_y = _safe_float(values.get("right_wrist_y"))
        wrist_asymmetry = None if left_wrist_y is None or right_wrist_y is None else abs(left_wrist_y - right_wrist_y)
        not_standing = bool(
            (knee_angle is not None and knee_angle <= self.not_standing_knee_angle_max)
            or (body_center_y is not None and body_center_y >= self.not_standing_body_center_y_max)
        )

        raw_phase = "unknown" if visible_score < self.min_visible_score else self._raw_phase(elbow_angle, elbow_delta)
        previous_phase = self._advance_phase(raw_phase)
        self._update_cycle(
            previous_phase=previous_phase,
            elbow_angle=elbow_angle,
            knee_angle=knee_angle,
            hip_angle=hip_angle,
            timestamp_ms=current_timestamp,
        )
        feedback_messages = self._feedback(
            visible_score=visible_score,
            not_standing=not_standing,
            torso_angle=torso_angle,
            wrist_asymmetry=wrist_asymmetry,
        )
        self.previous_elbow_angle = elbow_angle
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
                "elbow_angle_mean": elbow_angle,
                "elbow_delta": elbow_delta,
                "torso_angle": torso_angle,
                "knee_angle_mean": knee_angle,
                "wrist_distance_norm": wrist_distance,
                "wrist_asymmetry": wrist_asymmetry,
                "body_center_y": body_center_y,
                "frames_in_phase": self.frames_in_phase,
                "config_name": self.config_name,
                "sensitivity": self.sensitivity,
            },
        }


__all__ = ["SledPullAnalyzer"]
