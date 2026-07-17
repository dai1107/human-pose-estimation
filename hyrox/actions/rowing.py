from __future__ import annotations

from collections.abc import Mapping

from hyrox.base import BaseActionAnalyzer, PhaseSequenceTracker
from hyrox.config import load_rowing_config
from hyrox.feedback import FeedbackMessage
from hyrox.violations import TemporalViolationTracker, ViolationResult


SENSITIVITY_FRAME_DELTAS = {"low": 1, "medium": 0, "high": -1}
KNEE_MOTION_MIN = 2.0


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


class RowingAnalyzer(BaseActionAnalyzer):
    """Approximate side-view rowing stroke analyzer based on body pose only."""

    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        config: Mapping[str, object] | None = None,
        config_name: str | None = None,
    ) -> None:
        if sensitivity not in SENSITIVITY_FRAME_DELTAS:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        values = dict(config or load_rowing_config())
        self.early_stand_tracker = TemporalViolationTracker(
            "ROWING_EARLY_STAND_PROXY",
            _resolved_int(
                values.get("standing_violation_min_hold_ms"),
                300,
            ),
        )
        visibility_min = _resolved_float(values.get("visibility_min"), 0.55)
        super().__init__(action="rowing", min_visible_score=min(1.0, visibility_min))
        self.configure_feedback_limits(values)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or values.get("config_name") or "rowing_default")
        self.catch_knee_angle_max = _resolved_float(values.get("catch_knee_angle_max"), 105.0)
        self.finish_knee_angle_min = _resolved_float(values.get("finish_knee_angle_min"), 145.0)
        self.finish_torso_lean_max = _resolved_float(values.get("finish_torso_lean_max"), 35.0)
        self.too_much_back_lean = _resolved_float(values.get("too_much_back_lean"), 45.0)
        self.early_arm_pull_elbow_angle = _resolved_float(values.get("early_arm_pull_elbow_angle"), 120.0)
        self.standing_violation_knee_angle = _resolved_float(
            values.get("standing_violation_knee_angle_min_deg"),
            160.0,
        )
        self.standing_violation_hip_angle = _resolved_float(
            values.get("standing_violation_hip_angle_min_deg"),
            155.0,
        )
        self.standing_violation_trunk_max = _resolved_float(
            values.get("standing_violation_trunk_from_vertical_max_deg"),
            30.0,
        )
        self.standing_violation_hip_rise_ratio = _resolved_float(
            values.get(
                "standing_violation_hip_vertical_rise_body_ratio_min"
            ),
            0.18,
        )
        base_frames = _resolved_int(values.get("stable_frames"), 3, minimum=1)
        self.confirmation_frames = max(1, base_frames + SENSITIVITY_FRAME_DELTAS[sensitivity])
        self.stroke_cooldown_ms = _resolved_int(values.get("stroke_cooldown_ms"), 500)
        self.min_phase_duration_ms = _resolved_int(values.get("min_phase_duration_ms"), 120)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None = None,
        *,
        sensitivity: str = "medium",
        config_name: str | None = None,
    ) -> RowingAnalyzer:
        return cls(config=config, sensitivity=sensitivity, config_name=config_name)

    @classmethod
    def from_config_path(cls, path: str | None, *, sensitivity: str = "medium") -> RowingAnalyzer:
        config = load_rowing_config(path)
        return cls.from_config(
            config,
            sensitivity=sensitivity,
            config_name=str(config.get("config_name") or (path or "rowing_default")),
        )

    def reset(self) -> None:
        super().reset()
        self.phase = "unknown"
        self.raw_phase = "unknown"
        self.stable_phase = "unknown"
        self.frames_in_phase = 0
        self.phase_started_ms: int | None = None
        self.previous_knee_angle: float | None = None
        self.drive_seen = False
        self.finish_seen = False
        self.recovery_seen = False
        self.max_drive_knee_angle: float | None = None
        self.last_rep_time_ms: int | None = None
        self.incomplete_leg_drive = False
        self.rushed_recovery = False
        self.seated_hip_center_y: float | None = None
        self.early_stand_tracker.reset()
        self.early_stand_result = self.early_stand_tracker.last_result
        self.rep_sequence = PhaseSequenceTracker(
            ("catch", "drive", "finish"), optional_phases=("drive",)
        )

    def _visible_score(self, features: dict[str, object]) -> float:
        score = _safe_float(features.get("visible_score"))
        return max(0.0, min(1.0, score or 0.0))

    def _raw_phase(
        self,
        knee_angle: float | None,
        elbow_angle: float | None,
        torso_angle: float | None,
    ) -> str:
        if knee_angle is None:
            return "unknown"
        knee_delta = None if self.previous_knee_angle is None else knee_angle - self.previous_knee_angle
        if knee_angle <= self.catch_knee_angle_max:
            return "catch"
        if (
            knee_angle >= self.finish_knee_angle_min
            and elbow_angle is not None
            and elbow_angle <= self.early_arm_pull_elbow_angle + 25.0
        ):
            return "finish"
        if knee_delta is not None and knee_delta >= KNEE_MOTION_MIN:
            return "drive"
        if knee_delta is not None and knee_delta <= -KNEE_MOTION_MIN:
            return "recovery"
        if self.raw_phase in {"drive", "recovery"}:
            return self.raw_phase
        if self.stable_phase == "finish" and elbow_angle is not None and elbow_angle > self.early_arm_pull_elbow_angle:
            return "recovery"
        if self.stable_phase in {"catch", "drive"}:
            return "drive"
        if self.stable_phase == "recovery":
            return "recovery"
        return "unknown"

    def _cooldown_elapsed(self, timestamp_ms: int | None) -> bool:
        return timestamp_ms is None or self.last_rep_time_ms is None or timestamp_ms - self.last_rep_time_ms >= self.stroke_cooldown_ms

    def _advance_phase(self, raw_phase: str, timestamp_ms: int | None) -> tuple[str, int | None]:
        previous, _ = self._advance_confirmed_phase(raw_phase, self.confirmation_frames)
        previous_duration = None
        if self.stable_phase != previous:
            if timestamp_ms is not None and self.phase_started_ms is not None:
                previous_duration = max(0, timestamp_ms - self.phase_started_ms)
            self.phase_started_ms = timestamp_ms
        return previous, previous_duration

    def _update_stroke_sequence(
        self,
        previous_phase: str,
        previous_duration: int | None,
        knee_angle: float | None,
        timestamp_ms: int | None,
        visible_score: float,
    ) -> None:
        self.rushed_recovery = False
        self.rep_sequence.just_completed = False
        if self.stable_phase == "drive" and knee_angle is not None:
            self.drive_seen = True
            self.max_drive_knee_angle = knee_angle if self.max_drive_knee_angle is None else max(self.max_drive_knee_angle, knee_angle)
        if self.stable_phase == previous_phase:
            return
        sequence_completed = self.rep_sequence.update(self.stable_phase)
        if self.stable_phase == "drive":
            self.drive_seen = previous_phase == "catch" or self.drive_seen
            self.incomplete_leg_drive = False
        elif self.stable_phase == "finish" and self.drive_seen:
            self.finish_seen = True
            if sequence_completed:
                self.register_completed_sequence(
                    confidence=visible_score,
                    events={"terminal_phase": "finish"},
                )
                self.last_rep_time_ms = timestamp_ms
        elif self.stable_phase == "recovery":
            if self.drive_seen and not self.finish_seen:
                self.incomplete_leg_drive = True
            if self.finish_seen:
                self.recovery_seen = True
        elif self.stable_phase == "catch":
            self.rushed_recovery = previous_phase == "recovery" and previous_duration is not None and previous_duration < self.min_phase_duration_ms
            self.drive_seen = False
            self.finish_seen = False
            self.recovery_seen = False
            self.max_drive_knee_angle = None

    def _feedback(
        self,
        *,
        visible_score: float,
        knee_angle: float | None,
        hip_angle: float | None,
        elbow_angle: float | None,
        torso_angle: float | None,
        bad_view: bool,
        early_stand: ViolationResult,
    ) -> list[FeedbackMessage]:
        if visible_score < self.min_visible_score:
            return [FeedbackMessage("warn", "LOW_VISIBILITY", "请调整摄像头，保证肩、髋、膝、踝、手腕可见", max(0.2, 1.0 - visible_score))]
        messages: list[FeedbackMessage] = []
        if early_stand.active:
            messages.append(
                FeedbackMessage(
                    "warn",
                    "ROWING_EARLY_STAND_PROXY",
                    "检测到训练区间内明显站起，请保持划船坐姿",
                    early_stand.confidence,
                )
            )
        if self.stable_phase == "finish" and torso_angle is not None and abs(torso_angle) > self.too_much_back_lean:
            messages.append(FeedbackMessage("warn", "TOO_MUCH_BACK_LEAN", "结束阶段后仰过多，控制躯干角度", 0.8))
        if self.incomplete_leg_drive:
            messages.append(FeedbackMessage("warn", "NO_FULL_LEG_DRIVE", "蹬腿不充分，drive 阶段膝盖需要明显伸展", 0.8))
        if self.stable_phase == "drive" and knee_angle is not None and knee_angle < self.finish_knee_angle_min and elbow_angle is not None and elbow_angle < self.early_arm_pull_elbow_angle:
            messages.append(FeedbackMessage("warn", "EARLY_ARM_PULL", "手臂拉得过早，先蹬腿再带动手臂", 0.75))
        if self.rushed_recovery:
            messages.append(FeedbackMessage("warn", "RUSHED_RECOVERY", "恢复阶段过快，保持节奏稳定", 0.75))
        if bad_view or knee_angle is None or elbow_angle is None or (knee_angle > 160.0 and hip_angle is not None and hip_angle > 160.0 and (torso_angle is None or abs(torso_angle) < 15.0)):
            messages.append(FeedbackMessage("info", "NOT_SEATED_OR_BAD_VIEW", "当前姿态不像划船坐姿，建议侧面拍摄并保持全身入镜", 0.65))
        return self.limit_feedback(messages)

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, object]:
        self.begin_frame(features, timestamp_ms)
        values = features if isinstance(features, dict) else {}
        current_timestamp = None if timestamp_ms is None else int(timestamp_ms)
        visible_score = self._visible_score(values)
        left_knee = _safe_float(values.get("left_knee_angle"))
        right_knee = _safe_float(values.get("right_knee_angle"))
        knee_angle = _min_metric(left_knee, right_knee)
        hip_angle = _min_metric(values.get("left_hip_angle"), values.get("right_hip_angle"))
        elbow_angle = _mean_metric(values.get("left_elbow_angle"), values.get("right_elbow_angle"))
        torso_angle = _safe_float(values.get("torso_angle"))
        hip_center_y = _safe_float(values.get("hip_center_y"))
        hip_width = _safe_float(values.get("hip_width"))
        body_height = _safe_float(
            values.get("body_height_reference")
        ) or _safe_float(values.get("body_height_norm"))
        bad_view = bool(hip_width is not None and body_height is not None and body_height > 0.0 and hip_width / body_height > 0.20)
        raw_phase = "unknown" if visible_score < self.min_visible_score else self._raw_phase(knee_angle, elbow_angle, torso_angle)
        previous_phase, previous_duration = self._advance_phase(raw_phase, current_timestamp)
        self._update_stroke_sequence(
            previous_phase,
            previous_duration,
            knee_angle,
            current_timestamp,
            visible_score,
        )
        seated_reference_frame = (
            hip_center_y is not None
            and knee_angle is not None
            and (
                knee_angle < self.standing_violation_knee_angle
                or self.stable_phase
                in {"catch", "drive", "finish", "recovery"}
            )
        )
        if seated_reference_frame:
            self.seated_hip_center_y = (
                hip_center_y
                if self.seated_hip_center_y is None
                else max(self.seated_hip_center_y, hip_center_y)
            )
        early_stand_condition: bool | None
        if (
            visible_score < self.min_visible_score
            or knee_angle is None
            or hip_angle is None
            or torso_angle is None
            or hip_center_y is None
            or body_height is None
            or body_height <= 1e-6
            or self.seated_hip_center_y is None
            or self.camera_view_profile == "front"
        ):
            early_stand_condition = None
        else:
            hip_rise_ratio = (
                self.seated_hip_center_y - hip_center_y
            ) / body_height
            early_stand_condition = (
                knee_angle >= self.standing_violation_knee_angle
                and hip_angle >= self.standing_violation_hip_angle
                and abs(torso_angle)
                <= self.standing_violation_trunk_max
                and hip_rise_ratio
                >= self.standing_violation_hip_rise_ratio
            )
        self.early_stand_result = self.early_stand_tracker.update(
            early_stand_condition,
            current_timestamp,
            confidence=visible_score,
        )
        phase_duration = None if current_timestamp is None or self.phase_started_ms is None else max(0, current_timestamp - self.phase_started_ms)
        feedback_messages = self._feedback(
            visible_score=visible_score,
            knee_angle=knee_angle,
            hip_angle=hip_angle,
            elbow_angle=elbow_angle,
            torso_angle=torso_angle,
            bad_view=bad_view,
            early_stand=self.early_stand_result,
        )
        self.previous_knee_angle = knee_angle
        self.last_timestamp_ms = current_timestamp
        violation_results = [self.early_stand_result.as_dict()]
        active_violation_codes = (
            [self.early_stand_result.code]
            if self.early_stand_result.active
            else []
        )
        return self.finalize_state({
            "action": self.action,
            "phase": self.phase,
            "rep_count": self.rep_count,
            "cycle_count": self.rep_count,
            "count_semantics": "analysis_cycle",
            "official_rep_count_supported": False,
            "violation_scope": "active_analysis_interval",
            "violation_results": violation_results,
            "active_violation_codes": active_violation_codes,
            "feedback_messages": feedback_messages,
            "debug": {
                "raw_phase": self.raw_phase,
                "stable_phase": self.stable_phase,
                "stroke_count": self.rep_count,
                "left_knee_angle": left_knee,
                "right_knee_angle": right_knee,
                "torso_angle": torso_angle,
                "elbow_angle_mean": elbow_angle,
                "phase_duration_ms": phase_duration,
                "side_view_likely": not bad_view,
                "seated_hip_center_y": self.seated_hip_center_y,
                "rowing_early_stand_proxy": (
                    self.early_stand_result.as_dict()
                ),
                "frames_in_phase": self.frames_in_phase,
                "config_name": self.config_name,
                "sensitivity": self.sensitivity,
                **self.rep_sequence.debug(),
            },
        })


__all__ = ["RowingAnalyzer"]
