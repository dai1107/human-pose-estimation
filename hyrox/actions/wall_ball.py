from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from hyrox.base import BaseActionAnalyzer, PhaseSequenceTracker
from hyrox.config import load_wall_ball_config
from hyrox.feedback import FeedbackMessage
from hyrox.validity import BodyRuleResult, RepDecision


REP_COOLDOWN_MS = 400
WALL_BALL_REQUIRED_RULES = (
    "tall_start",
    "hip_below_knee",
    "upward_extension",
    "bilateral_throw_proxy",
)
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
        self.tall_start_knee_angle = _resolved_float(
            config_data.get("tall_start_knee_angle_min"), 165.0
        )
        self.tall_start_hip_angle = _resolved_float(
            config_data.get("tall_start_hip_angle_min"), 165.0
        )
        self.tall_start_trunk_max = _resolved_float(
            config_data.get("tall_start_trunk_from_vertical_max_deg"),
            25.0,
            minimum=0.0,
        )
        self.bottom_knee_angle = _resolved_float(config_data.get("bottom_knee_angle_max"), 110.0)
        self.hip_below_knee_margin = _resolved_float(
            config_data.get("hip_below_knee_margin"), 0.01
        )
        self.throw_knee_angle = _resolved_float(config_data.get("throw_knee_angle_min"), 150.0)
        self.throw_hip_angle = _resolved_float(config_data.get("throw_hip_angle_min"), 145.0)
        self.throw_elbow_angle = _resolved_float(config_data.get("throw_elbow_angle_min"), 125.0)
        self.wrist_above_shoulder_min = _resolved_float(config_data.get("wrist_above_shoulder_min"), 0.03)
        self.full_extension_knee_angle = _resolved_float(
            config_data.get("full_extension_knee_angle_min"), 165.0
        )
        self.full_extension_hip_angle = _resolved_float(
            config_data.get("full_extension_hip_angle_min"), 165.0
        )
        self.wrist_peak_time_diff_pass_ms = _resolved_int(
            config_data.get("wrist_peak_time_diff_ms_pass"), 120, minimum=0
        )
        self.wrist_peak_time_diff_unsure_ms = max(
            self.wrist_peak_time_diff_pass_ms,
            _resolved_int(
                config_data.get("wrist_peak_time_diff_ms_unsure"),
                220,
                minimum=0,
            ),
        )
        both_wrists_required = config_data.get(
            "both_wrists_above_shoulders_required", True
        )
        self.both_wrists_above_shoulders_required = (
            both_wrists_required
            if isinstance(both_wrists_required, bool)
            else True
        )
        self.throw_wrist_rise_body_ratio_min = _resolved_float(
            config_data.get("throw_wrist_rise_body_ratio_min"),
            0.12,
            minimum=0.0,
        )
        self.throw_wrist_chest_band_body_ratio = _resolved_float(
            config_data.get("throw_wrist_chest_band_body_ratio"),
            0.25,
            minimum=0.0,
        )
        self.throw_wrist_midline_body_ratio_max = _resolved_float(
            config_data.get("throw_wrist_midline_body_ratio_max"),
            0.60,
            minimum=0.0,
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
        self._pending_tall_rule: BodyRuleResult | None = None
        self._pending_tall_evidence: list[int] = []
        self._cycle_tall_rule: BodyRuleResult | None = None
        self._candidate_rule_active = False
        self._hip_depth_best_ratio: float | None = None
        self._hip_depth_evidence: list[int] = []
        self._upward_extension_rule: BodyRuleResult | None = None
        self._pending_wrist_start_y: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._pending_wrist_start_observable = False
        self._pending_wrists_near_chest = False
        self._pending_wrist_start_evidence: list[int] = []
        self._cycle_wrist_start_y: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._cycle_wrist_start_observable = False
        self._cycle_wrists_near_chest = False
        self._cycle_wrist_start_evidence: list[int] = []
        self._wrist_peak_y: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._wrist_peak_x: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._wrist_peak_ms: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._wrist_peak_frame: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._wrist_above_shoulder_peak: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._throw_body_height: float | None = None
        self._throw_body_center_x: float | None = None
        self._upward_extension_evidence: list[int] = []
        self._last_wall_ball_rules: tuple[BodyRuleResult, ...] = ()
        self._last_wall_ball_validation_state = "UNKNOWN"
        self.rep_sequence = PhaseSequenceTracker(
            ("stand", "squat_down", "bottom", "drive", "throw_extension"),
            optional_phases=("squat_down", "drive"),
        )

    @staticmethod
    def _body_height(features: Mapping[str, object]) -> float | None:
        for name in (
            "body_height_reference",
            "body_box_height_norm",
            "skeleton_height_estimate_norm",
            "body_height_norm",
        ):
            value = _safe_float(features.get(name))
            if value is not None and value > 1e-6:
                return value
        return None

    @staticmethod
    def _all_side_values(
        features: Mapping[str, object],
        metric: str,
    ) -> tuple[float, float] | None:
        left = _safe_float(features.get(f"left_{metric}"))
        right = _safe_float(features.get(f"right_{metric}"))
        if left is None or right is None:
            return None
        return left, right

    def _capture_pending_tall_rule(
        self,
        features: Mapping[str, object],
    ) -> None:
        knees = self._all_side_values(features, "knee_angle")
        hips = self._all_side_values(features, "hip_angle")
        trunk = _safe_float(features.get("torso_angle"))
        if knees is None or hips is None or trunk is None:
            self._pending_tall_rule = BodyRuleResult(
                "tall_start",
                "UNSURE",
                0.0,
                reason_code="TALL_START_NOT_OBSERVABLE",
                evidence_frames=tuple(self._pending_tall_evidence),
            )
            return
        minimum_knee = min(knees)
        minimum_hip = min(hips)
        passed = (
            minimum_knee >= self.tall_start_knee_angle
            and minimum_hip >= self.tall_start_hip_angle
            and abs(trunk) <= self.tall_start_trunk_max
        )
        self._pending_tall_rule = BodyRuleResult(
            "tall_start",
            "PASS" if passed else "FAIL",
            0.95,
            value=passed,
            reason_code=None if passed else "TALL_START_REQUIREMENTS_NOT_MET",
            evidence_frames=tuple(self._pending_tall_evidence),
        )

    def _observe_pending_wrist_start(
        self,
        features: Mapping[str, object],
    ) -> None:
        left_y = _safe_float(features.get("left_wrist_y"))
        right_y = _safe_float(features.get("right_wrist_y"))
        shoulder_y = _safe_float(features.get("shoulder_center_y"))
        hip_y = _safe_float(features.get("hip_center_y"))
        body_height = self._body_height(features)
        if (
            left_y is None
            or right_y is None
            or shoulder_y is None
            or hip_y is None
            or body_height is None
        ):
            return
        chest_y = (shoulder_y + hip_y) / 2.0
        tolerance = self.throw_wrist_chest_band_body_ratio * body_height
        near_chest = (
            abs(left_y - chest_y) <= tolerance
            and abs(right_y - chest_y) <= tolerance
        )
        self._pending_wrist_start_observable = True
        if self.frame_index not in self._pending_wrist_start_evidence:
            self._pending_wrist_start_evidence.append(self.frame_index)
        if near_chest or not self._pending_wrists_near_chest:
            self._pending_wrist_start_y = {
                "left": left_y,
                "right": right_y,
            }
        self._pending_wrists_near_chest = (
            self._pending_wrists_near_chest or near_chest
        )

    def _start_rule_candidate(self) -> None:
        self._candidate_rule_active = True
        self._cycle_tall_rule = self._pending_tall_rule or BodyRuleResult(
            "tall_start",
            "UNSURE",
            0.0,
            reason_code="TALL_START_NOT_OBSERVABLE",
        )
        self._hip_depth_best_ratio = None
        self._hip_depth_evidence = []
        self._upward_extension_rule = None
        self._cycle_wrist_start_y = dict(self._pending_wrist_start_y)
        self._cycle_wrist_start_observable = (
            self._pending_wrist_start_observable
        )
        self._cycle_wrists_near_chest = self._pending_wrists_near_chest
        self._cycle_wrist_start_evidence = list(
            self._pending_wrist_start_evidence
        )
        self._wrist_peak_y = {"left": None, "right": None}
        self._wrist_peak_x = {"left": None, "right": None}
        self._wrist_peak_ms = {"left": None, "right": None}
        self._wrist_peak_frame = {"left": None, "right": None}
        self._wrist_above_shoulder_peak = {"left": None, "right": None}
        self._throw_body_height = None
        self._throw_body_center_x = None
        self._upward_extension_evidence = []

    def _clear_rule_candidate(self, *, clear_pending: bool = False) -> None:
        self._candidate_rule_active = False
        self._cycle_tall_rule = None
        self._hip_depth_best_ratio = None
        self._hip_depth_evidence = []
        self._upward_extension_rule = None
        self._cycle_wrist_start_y = {"left": None, "right": None}
        self._cycle_wrist_start_observable = False
        self._cycle_wrists_near_chest = False
        self._cycle_wrist_start_evidence = []
        self._wrist_peak_y = {"left": None, "right": None}
        self._wrist_peak_x = {"left": None, "right": None}
        self._wrist_peak_ms = {"left": None, "right": None}
        self._wrist_peak_frame = {"left": None, "right": None}
        self._wrist_above_shoulder_peak = {"left": None, "right": None}
        self._throw_body_height = None
        self._throw_body_center_x = None
        self._upward_extension_evidence = []
        if clear_pending:
            self._pending_tall_rule = None
            self._pending_tall_evidence = []
            self._pending_wrist_start_y = {"left": None, "right": None}
            self._pending_wrist_start_observable = False
            self._pending_wrists_near_chest = False
            self._pending_wrist_start_evidence = []

    def _observe_hip_depth(self, features: Mapping[str, object]) -> None:
        if str(features.get("floor_reference_status")) != "READY":
            return
        hip_height = _safe_float(features.get("hip_center_height_to_floor"))
        knee_height = _safe_float(features.get("knee_center_height_to_floor"))
        body_height = _safe_float(features.get("body_height_reference"))
        if (
            hip_height is None
            or knee_height is None
            or body_height is None
            or body_height <= 1e-6
        ):
            return
        depth_ratio = (knee_height - hip_height) / body_height
        if (
            self._hip_depth_best_ratio is None
            or depth_ratio > self._hip_depth_best_ratio
        ):
            self._hip_depth_best_ratio = depth_ratio
        self._hip_depth_evidence.append(self.frame_index)

    def _observe_wrist_peaks(self, features: Mapping[str, object]) -> None:
        body_height = self._body_height(features)
        body_center_x = _safe_float(features.get("body_center_x"))
        if body_height is not None:
            self._throw_body_height = body_height
        if body_center_x is not None:
            self._throw_body_center_x = body_center_x
        for side in ("left", "right"):
            wrist_y = _safe_float(features.get(f"{side}_wrist_y"))
            wrist_x = _safe_float(features.get(f"{side}_wrist_x"))
            above = _safe_float(
                features.get(f"{side}_wrist_above_shoulder")
            )
            current_peak = self._wrist_peak_y[side]
            if wrist_y is not None and (
                current_peak is None or wrist_y < current_peak - 1e-6
            ):
                self._wrist_peak_y[side] = wrist_y
                self._wrist_peak_x[side] = wrist_x
                self._wrist_peak_ms[side] = self.last_timestamp_ms
                self._wrist_peak_frame[side] = self.frame_index
            current_above = self._wrist_above_shoulder_peak[side]
            if above is not None and (
                current_above is None or above > current_above
            ):
                self._wrist_above_shoulder_peak[side] = above

    def _capture_upward_extension_rule(
        self,
        features: Mapping[str, object],
    ) -> None:
        if self.frame_index not in self._upward_extension_evidence:
            self._upward_extension_evidence.append(self.frame_index)
        knees = self._all_side_values(features, "knee_angle")
        hips = self._all_side_values(features, "hip_angle")
        if knees is None or hips is None:
            self._upward_extension_rule = BodyRuleResult(
                "upward_extension",
                "UNSURE",
                0.0,
                reason_code="UPWARD_EXTENSION_NOT_OBSERVABLE",
                evidence_frames=tuple(self._upward_extension_evidence),
            )
            return
        minimum_knee = min(knees)
        minimum_hip = min(hips)
        passed = (
            minimum_knee >= self.full_extension_knee_angle
            and minimum_hip >= self.full_extension_hip_angle
        )
        self._upward_extension_rule = BodyRuleResult(
            "upward_extension",
            "PASS" if passed else "FAIL",
            0.95,
            value=min(minimum_knee, minimum_hip),
            reason_code=None if passed else "UPWARD_EXTENSION_INCOMPLETE",
            evidence_frames=tuple(self._upward_extension_evidence),
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
        arms_extended = (
            elbow_angle is None
            or elbow_angle >= self.throw_elbow_angle
        )
        return (
            min_knee is not None
            and hip_angle is not None
            and wrist_above_shoulder is not None
            and min_knee >= self.throw_knee_angle
            and hip_angle >= self.throw_hip_angle
            and arms_extended
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
            # A browser realtime stream may sample the short overhead endpoint
            # only once. Drive is transition-only, so complete directly from a
            # previously confirmed bottom when the full throw posture appears.
            return "throw_extension"
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

    def _hip_below_knee_rule(self) -> BodyRuleResult:
        if self._hip_depth_best_ratio is None:
            return BodyRuleResult(
                "hip_below_knee",
                "UNSURE",
                0.0,
                reason_code="HIP_KNEE_FLOOR_HEIGHT_NOT_OBSERVABLE",
                evidence_frames=tuple(self._hip_depth_evidence),
            )
        passed = self._hip_depth_best_ratio >= self.hip_below_knee_margin
        return BodyRuleResult(
            "hip_below_knee",
            "PASS" if passed else "FAIL",
            0.95 if passed else 0.90,
            value=self._hip_depth_best_ratio,
            reason_code=None if passed else "HIP_NOT_BELOW_KNEE",
            evidence_frames=tuple(self._hip_depth_evidence),
        )

    def _bilateral_throw_rule(self) -> BodyRuleResult:
        throw_evidence = tuple(
            sorted(
                {
                    *self._cycle_wrist_start_evidence,
                    *(
                        frame
                        for frame in self._wrist_peak_frame.values()
                        if frame is not None
                    ),
                    self.frame_index,
                }
            )
        )
        if not self._cycle_wrist_start_observable:
            return BodyRuleResult(
                "bilateral_throw_proxy",
                "UNSURE",
                0.0,
                reason_code="WRIST_START_NOT_OBSERVABLE",
                evidence_frames=throw_evidence,
            )
        if not self._cycle_wrists_near_chest:
            return BodyRuleResult(
                "bilateral_throw_proxy",
                "FAIL",
                0.90,
                reason_code="WRISTS_DID_NOT_START_NEAR_CHEST",
                evidence_frames=throw_evidence,
            )
        endpoint_values = tuple(
            _safe_float(self._current_features.get(f"{side}_{metric}"))
            for side in ("left", "right")
            for metric in (
                "wrist_x",
                "wrist_y",
                "wrist_above_shoulder",
            )
        )
        if any(value is None for value in endpoint_values):
            return BodyRuleResult(
                "bilateral_throw_proxy",
                "UNSURE",
                0.0,
                reason_code="BILATERAL_THROW_ENDPOINT_NOT_OBSERVABLE",
                evidence_frames=throw_evidence,
            )

        start_left = self._cycle_wrist_start_y["left"]
        start_right = self._cycle_wrist_start_y["right"]
        peak_left = self._wrist_peak_y["left"]
        peak_right = self._wrist_peak_y["right"]
        peak_left_x = self._wrist_peak_x["left"]
        peak_right_x = self._wrist_peak_x["right"]
        left_ms = self._wrist_peak_ms["left"]
        right_ms = self._wrist_peak_ms["right"]
        left_above = self._wrist_above_shoulder_peak["left"]
        right_above = self._wrist_above_shoulder_peak["right"]
        body_height = self._throw_body_height
        body_center_x = self._throw_body_center_x
        required_values = (
            start_left,
            start_right,
            peak_left,
            peak_right,
            peak_left_x,
            peak_right_x,
            left_ms,
            right_ms,
            left_above,
            right_above,
            body_height,
            body_center_x,
        )
        if any(value is None for value in required_values):
            return BodyRuleResult(
                "bilateral_throw_proxy",
                "UNSURE",
                0.0,
                reason_code="BILATERAL_THROW_NOT_OBSERVABLE",
                evidence_frames=throw_evidence,
            )
        assert (
            start_left is not None
            and start_right is not None
            and peak_left is not None
            and peak_right is not None
            and peak_left_x is not None
            and peak_right_x is not None
            and left_ms is not None
            and right_ms is not None
            and left_above is not None
            and right_above is not None
            and body_height is not None
            and body_center_x is not None
        )
        if body_height <= 1e-6:
            return BodyRuleResult(
                "bilateral_throw_proxy",
                "UNSURE",
                0.0,
                reason_code="BILATERAL_THROW_BODY_SCALE_NOT_OBSERVABLE",
                evidence_frames=throw_evidence,
            )

        left_rise = (start_left - peak_left) / body_height
        right_rise = (start_right - peak_right) / body_height
        both_above = (
            left_above >= self.wrist_above_shoulder_min
            and right_above >= self.wrist_above_shoulder_min
        )
        if not self.both_wrists_above_shoulders_required:
            both_above = max(left_above, right_above) >= self.wrist_above_shoulder_min
        wrists_centered = (
            abs(peak_left_x - body_center_x) / body_height
            <= self.throw_wrist_midline_body_ratio_max
            and abs(peak_right_x - body_center_x) / body_height
            <= self.throw_wrist_midline_body_ratio_max
        )
        rise_ok = (
            left_rise >= self.throw_wrist_rise_body_ratio_min
            and right_rise >= self.throw_wrist_rise_body_ratio_min
        )
        peak_diff_ms = abs(left_ms - right_ms)
        if not both_above:
            status = "FAIL"
            reason = "BOTH_WRISTS_NOT_ABOVE_SHOULDERS"
        elif not rise_ok:
            status = "FAIL"
            reason = "BILATERAL_WRIST_RISE_TOO_SMALL"
        elif not wrists_centered:
            status = "FAIL"
            reason = "WRISTS_TOO_FAR_FROM_BODY_MIDLINE"
        elif peak_diff_ms <= self.wrist_peak_time_diff_pass_ms:
            status = "PASS"
            reason = None
        elif peak_diff_ms <= self.wrist_peak_time_diff_unsure_ms:
            status = "UNSURE"
            reason = "WRIST_PEAK_TIMING_UNSURE"
        else:
            status = "FAIL"
            reason = "WRIST_PEAKS_NOT_SYNCHRONIZED"
        return BodyRuleResult(
            "bilateral_throw_proxy",
            status,  # type: ignore[arg-type]
            0.95 if status == "PASS" else 0.75 if status == "UNSURE" else 0.90,
            value=peak_diff_ms,
            reason_code=reason,
            evidence_frames=throw_evidence,
        )

    def _validate_wall_ball_candidate(
        self,
        visible_score: float,
    ) -> RepDecision:
        rules = (
            self._cycle_tall_rule
            or BodyRuleResult(
                "tall_start",
                "UNSURE",
                0.0,
                reason_code="TALL_START_NOT_OBSERVABLE",
            ),
            self._hip_below_knee_rule(),
            self._upward_extension_rule
            or BodyRuleResult(
                "upward_extension",
                "UNSURE",
                0.0,
                reason_code="UPWARD_EXTENSION_NOT_OBSERVABLE",
            ),
            self._bilateral_throw_rule(),
        )
        self._last_wall_ball_rules = rules
        decision = self.register_rep_candidate(
            rules,
            required_rules=WALL_BALL_REQUIRED_RULES,
            events={
                "terminal_phase": "throw_extension",
                "hip_below_knee_ratio": self._hip_depth_best_ratio,
                "wrist_peak_time_diff_ms": (
                    None
                    if self._wrist_peak_ms["left"] is None
                    or self._wrist_peak_ms["right"] is None
                    else abs(
                        self._wrist_peak_ms["left"]
                        - self._wrist_peak_ms["right"]
                    )
                ),
                "throw_proxy_name": "BILATERAL_THROW_PROXY",
                "visible_score": visible_score,
            },
        )
        self.just_completed_rep = decision.status == "VALID"
        self.last_rep_time_ms = self.last_timestamp_ms
        self._last_wall_ball_validation_state = (
            "POSE_VALID_REP"
            if decision.status == "VALID"
            else "RULE_VALIDATION"
        )
        self._clear_rule_candidate(clear_pending=True)
        return decision

    def _validation_state(self, raw_phase: str) -> str:
        if self.just_finished_attempt:
            return self._last_wall_ball_validation_state
        if raw_phase == "throw_extension" and all(
            (self._wrist_above_shoulder_peak[side] or 0.0)
            >= self.wrist_above_shoulder_min
            for side in ("left", "right")
        ):
            return "BILATERAL_THROW_CONFIRMED"
        if raw_phase in {"drive", "stand"} and self._candidate_rule_active:
            return "ASCENDING"
        if (
            self._hip_depth_best_ratio is not None
            and self._hip_depth_best_ratio >= self.hip_below_knee_margin
        ):
            return "HIP_BELOW_KNEE_CONFIRMED"
        if raw_phase in {"squat_down", "bottom"}:
            return "DESCENDING"
        if (
            raw_phase == "stand"
            and self._pending_tall_rule is not None
            and self._pending_tall_rule.status == "PASS"
        ):
            return "TALL_START"
        return "UNKNOWN"

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
            "wall_ball_validation_state": self._validation_state(self.raw_phase),
            "wall_ball_required_rules": list(WALL_BALL_REQUIRED_RULES),
            "wall_ball_rules": [
                rule.as_dict() for rule in self._last_wall_ball_rules
            ],
            "candidate_rule_active": self._candidate_rule_active,
            "hip_below_knee_best_ratio": self._hip_depth_best_ratio,
            "wrist_peak_ms": dict(self._wrist_peak_ms),
            "confirmation_frames": self.confirmation_frames,
            "rep_cooldown_ms": self.rep_cooldown_ms,
            "sensitivity": self.sensitivity,
            "config_name": self.config_name,
            **self.rep_sequence.debug(),
        }

    def update(self, features: dict[str, object] | None, timestamp_ms: int | None) -> dict[str, Any]:
        self.begin_frame(features, timestamp_ms)
        self.last_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms)
        self.just_completed_rep = False
        self.just_finished_attempt = False
        self.rep_sequence.just_completed = False
        resolved_features = self._current_features
        visible_score = self._visible_score(resolved_features)
        min_knee = _minimum_feature(
            resolved_features, "left_knee_angle", "right_knee_angle"
        )
        hip_angle = _maximum_feature(
            resolved_features, "left_hip_angle", "right_hip_angle"
        )
        elbow_angle = _maximum_feature(
            resolved_features, "left_elbow_angle", "right_elbow_angle"
        )
        hip_center_y = _safe_float(resolved_features.get("hip_center_y"))
        hip_knee_depth = _safe_float(resolved_features.get("hip_knee_depth"))
        wrist_above_shoulder = _maximum_feature(
            resolved_features,
            "wrist_above_shoulder",
            "left_wrist_above_shoulder",
            "right_wrist_above_shoulder",
        )
        knee_width = _safe_float(resolved_features.get("knee_width"))
        ankle_width = _safe_float(resolved_features.get("ankle_width"))

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
                self._clear_rule_candidate(clear_pending=True)
            self.previous_min_knee_angle = None
            self.previous_hip_center_y = None
            return self.finalize_state({
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
            })

        raw_phase = self._phase_from_features(
            min_knee=min_knee,
            hip_angle=hip_angle,
            elbow_angle=elbow_angle,
            hip_center_y=hip_center_y,
            wrist_above_shoulder=wrist_above_shoulder,
        )
        previous, stable = self._advance_phase(raw_phase)
        depth_met = hip_knee_depth is not None and hip_knee_depth >= self.hip_below_knee_margin

        if not self._candidate_rule_active:
            self._observe_pending_wrist_start(resolved_features)
            if (
                raw_phase == "stand"
                and self.frame_index not in self._pending_tall_evidence
            ):
                self._pending_tall_evidence.append(self.frame_index)
            if raw_phase == "stand" and stable == "stand":
                self._capture_pending_tall_rule(resolved_features)

        if (
            not self._candidate_rule_active
            and stable in {"squat_down", "bottom"}
        ):
            self._start_rule_candidate()

        if self._candidate_rule_active:
            self._observe_wrist_peaks(resolved_features)
            if raw_phase == "bottom" or stable == "bottom":
                self._observe_hip_depth(resolved_features)
            if raw_phase == "throw_extension":
                self._capture_upward_extension_rule(resolved_features)

        if stable == "bottom" and self._candidate_rule_active:
            self.bottom_seen = True
            self.bottom_depth_met = self.bottom_depth_met or (
                self._hip_depth_best_ratio is not None
                and self._hip_depth_best_ratio >= self.hip_below_knee_margin
            )

        if stable != previous:
            self.rep_sequence.update(stable)
            self.observe_candidate_phase(stable)
            if stable == "stand":
                self.stand_seen = True
            if stable == "drive" and self.bottom_seen:
                self.extension_pending = True
            if stable == "throw_extension" and (self.bottom_seen or self.extension_pending):
                self.just_finished_attempt = True
                if (
                    self._candidate_rule_active
                    and self._cooldown_elapsed(self.last_timestamp_ms)
                ):
                    self._capture_upward_extension_rule(resolved_features)
                    self._validate_wall_ball_candidate(visible_score)
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
        return self.finalize_state({
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
        })


__all__ = [
    "REP_COOLDOWN_MS",
    "WALL_BALL_REQUIRED_RULES",
    "WALL_BALL_SENSITIVITY_DELTAS",
    "WallBallAnalyzer",
]
