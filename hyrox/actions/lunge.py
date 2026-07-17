from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from hyrox.base import BaseActionAnalyzer, PhaseSequenceTracker
from hyrox.config import load_lunge_config
from hyrox.contact import ContactResult, KneeContactDetector
from hyrox.feedback import FeedbackMessage
from hyrox.validity import BodyRuleResult, RepDecision


PHASE_CONFIRMATION_FRAMES = 2
REP_COOLDOWN_MS = 400
LUNGE_REQUIRED_RULES = (
    "trailing_knee_contact",
    "full_knee_extension",
    "full_hip_extension",
    "alternating_contact_leg",
    "no_extra_step_or_shuffle",
)
FULL_EXTENSION_HOLD_FRAMES = {"high": 1, "medium": 2, "low": 3}
FEEDBACK_PRIORITY = {"error": 0, "warn": 1, "info": 2}
SENSITIVITY_PROFILES: dict[str, dict[str, float | int]] = {
    "low": {
        "min_visible_score": 0.55,
        "stand_knee_angle": 158.0,
        "stand_hip_angle": 150.0,
        "full_extension_knee_angle": 170.0,
        "full_extension_hip_angle": 170.0,
        "bottom_knee_angle": 105.0,
        "deep_knee_angle": 95.0,
        "torso_lean_warn_angle": 18.0,
        "motion_tolerance": 5.0,
        "hip_motion_tolerance": 0.006,
        "hip_drop_min": 0.045,
        "confirmation_frames": 3,
    },
    "medium": {
        "min_visible_score": 0.45,
        "stand_knee_angle": 150.0,
        "stand_hip_angle": 145.0,
        "full_extension_knee_angle": 165.0,
        "full_extension_hip_angle": 165.0,
        "bottom_knee_angle": 115.0,
        "deep_knee_angle": 100.0,
        "torso_lean_warn_angle": 20.0,
        "motion_tolerance": 3.0,
        "hip_motion_tolerance": 0.004,
        "hip_drop_min": 0.035,
        "confirmation_frames": PHASE_CONFIRMATION_FRAMES,
    },
    "high": {
        "min_visible_score": 0.35,
        "stand_knee_angle": 145.0,
        "stand_hip_angle": 140.0,
        "full_extension_knee_angle": 158.0,
        "full_extension_hip_angle": 157.0,
        "bottom_knee_angle": 125.0,
        "deep_knee_angle": 108.0,
        "torso_lean_warn_angle": 24.0,
        "motion_tolerance": 2.0,
        "hip_motion_tolerance": 0.002,
        "hip_drop_min": 0.025,
        "confirmation_frames": 1,
    },
}

CONFIG_PROFILE_KEYS: dict[str, str] = {
    "visibility_min": "min_visible_score",
    "stand_knee_angle_min": "stand_knee_angle",
    "stand_hip_angle_min": "stand_hip_angle",
    "full_extension_knee_angle_min": "full_extension_knee_angle",
    "full_extension_hip_angle_min": "full_extension_hip_angle",
    "bottom_knee_angle_max": "bottom_knee_angle",
    "deep_knee_angle_max": "deep_knee_angle",
    "torso_lean_warn": "torso_lean_warn_angle",
    "motion_tolerance": "motion_tolerance",
    "hip_motion_tolerance": "hip_motion_tolerance",
    "hip_drop_min": "hip_drop_min",
    "stable_frames": "confirmation_frames",
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


def _resolved_float(value: object, fallback: float, *, minimum: float | None = None) -> float:
    resolved = _safe_float(value)
    if resolved is None:
        resolved = float(fallback)
    if minimum is not None:
        resolved = max(minimum, resolved)
    return resolved


def _resolved_int(value: object, fallback: int, *, minimum: int = 0) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        resolved = int(fallback)
    return max(minimum, resolved)


def _apply_sensitivity_to_config(config: dict[str, object], sensitivity: str) -> dict[str, object]:
    if not config or sensitivity == "medium":
        return config
    adjusted = dict(config)
    profile = SENSITIVITY_PROFILES[sensitivity]
    medium_profile = SENSITIVITY_PROFILES["medium"]
    for config_key, profile_key in CONFIG_PROFILE_KEYS.items():
        value = _safe_float(config.get(config_key))
        if value is None:
            continue
        delta = float(profile[profile_key]) - float(medium_profile[profile_key])
        adjusted_value = value + delta
        adjusted[config_key] = int(round(adjusted_value)) if config_key == "stable_frames" else adjusted_value
    return adjusted


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
        hip_motion_tolerance: float | None = None,
        hip_drop_min: float | None = None,
        confirmation_frames: int | None = None,
        rep_cooldown_ms: int = REP_COOLDOWN_MS,
    ) -> None:
        if sensitivity not in SENSITIVITY_PROFILES:
            raise ValueError(f"unsupported HYROX sensitivity: {sensitivity}")
        profile = SENSITIVITY_PROFILES[sensitivity]
        config_data = _apply_sensitivity_to_config(dict(config) if config is not None else {}, sensitivity)

        def config_value(key: str, fallback: float | int | None) -> float | int | None:
            return config_data.get(key, fallback)

        resolved_min_visible_score = _resolved_float(
            config_value("visibility_min", _profile_value(profile, "min_visible_score", min_visible_score)),
            float(profile["min_visible_score"]),
            minimum=0.0,
        )
        super().__init__(action="Lunge", min_visible_score=resolved_min_visible_score)
        self.configure_feedback_limits(config_data)
        self.sensitivity = sensitivity
        self.config_name = str(config_name or config_data.get("config_name") or "lunge_default")
        self.stand_knee_angle = _resolved_float(
            config_value("stand_knee_angle_min", _profile_value(profile, "stand_knee_angle", stand_knee_angle)),
            float(profile["stand_knee_angle"]),
        )
        self.stand_hip_angle = _resolved_float(
            config_value("stand_hip_angle_min", _profile_value(profile, "stand_hip_angle", stand_hip_angle)),
            float(profile["stand_hip_angle"]),
        )
        self.full_extension_knee_angle = _resolved_float(
            config_value("full_extension_knee_angle_min", _profile_value(profile, "full_extension_knee_angle", full_extension_knee_angle)),
            float(profile["full_extension_knee_angle"]),
        )
        self.full_extension_hip_angle = _resolved_float(
            config_value("full_extension_hip_angle_min", _profile_value(profile, "full_extension_hip_angle", full_extension_hip_angle)),
            float(profile["full_extension_hip_angle"]),
        )
        self.bottom_knee_angle = _resolved_float(
            config_value("bottom_knee_angle_max", _profile_value(profile, "bottom_knee_angle", bottom_knee_angle)),
            float(profile["bottom_knee_angle"]),
        )
        self.deep_knee_angle = _resolved_float(
            config_value("deep_knee_angle_max", _profile_value(profile, "deep_knee_angle", deep_knee_angle)),
            float(profile["deep_knee_angle"]),
        )
        self.torso_lean_warn_angle = _resolved_float(
            config_value("torso_lean_warn", _profile_value(profile, "torso_lean_warn_angle", torso_lean_warn_angle)),
            float(profile["torso_lean_warn_angle"]),
            minimum=0.0,
        )
        self.motion_tolerance = _resolved_float(
            config_value("motion_tolerance", _profile_value(profile, "motion_tolerance", motion_tolerance)),
            float(profile["motion_tolerance"]),
            minimum=0.0,
        )
        self.hip_motion_tolerance = _resolved_float(
            config_value("hip_motion_tolerance", _profile_value(profile, "hip_motion_tolerance", hip_motion_tolerance)),
            float(profile["hip_motion_tolerance"]),
            minimum=0.0,
        )
        self.hip_drop_min = _resolved_float(
            config_value("hip_drop_min", _profile_value(profile, "hip_drop_min", hip_drop_min)),
            float(profile["hip_drop_min"]),
            minimum=0.0,
        )
        self.confirmation_frames = _resolved_int(
            config_value("stable_frames", _profile_value(profile, "confirmation_frames", confirmation_frames)),
            int(profile["confirmation_frames"]),
            minimum=1,
        )
        self.rep_cooldown_ms = _resolved_int(config_value("rep_cooldown_ms", rep_cooldown_ms), REP_COOLDOWN_MS)
        self.full_extension_hold_frames = _resolved_int(
            config_value(
                f"full_extension_hold_frames_{sensitivity}",
                FULL_EXTENSION_HOLD_FRAMES[sensitivity],
            ),
            FULL_EXTENSION_HOLD_FRAMES[sensitivity],
            minimum=1,
        )

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
        self.previous_hip_center_y: float | None = None
        self.stand_hip_center_y: float | None = None
        self.hip_drop: float | None = None
        self.bottom_seen = False
        self.current_rep_min_knee_angle: float | None = None
        self.just_completed_rep = False
        self.last_rep_time_ms: int | None = None
        self.previous_valid_contact_leg: Literal["left", "right"] | None = None
        self._seen_step_event_keys: set[tuple[object, ...]] = set()
        self._steps_since_last_candidate: list[object] = []
        self._foot_interval_observable = True
        self._last_stand_body_center_x: float | None = None
        self._previous_raw_phase_for_rules = "unknown"
        self._candidate_rule_active = False
        self._side_knee_detectors = {
            side: KneeContactDetector(
                sensitivity=str(getattr(self, "sensitivity", "medium"))
            )
            for side in ("left", "right")
        }
        self._reset_candidate_rule_tracking()
        self.rep_sequence = PhaseSequenceTracker(
            ("stand", "descent", "bottom", "ascent", "stand"),
            optional_phases=("descent", "ascent"),
        )

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
        hip_center_y: float | None,
    ) -> str:
        if min_knee_angle is None:
            return "unknown"
        # The documented stand phase is driven by both knees being extended.
        # Hip extension remains a form-feedback signal, but should not prevent a
        # loaded lunge from reaching stand when the bag obscures/distorts one hip.
        if min_knee_angle >= self.stand_knee_angle:
            return "stand"
        has_vertical_reference = hip_center_y is not None and self.stand_hip_center_y is not None
        is_low_enough = not has_vertical_reference or self.hip_drop is None or self.hip_drop >= self.hip_drop_min
        if min_knee_angle <= self.bottom_knee_angle and is_low_enough:
            return "bottom"
        if self.previous_min_knee_angle is not None:
            delta = min_knee_angle - self.previous_min_knee_angle
            if delta <= -self.motion_tolerance:
                return "descent"
            if delta >= self.motion_tolerance:
                return "ascent"
        if hip_center_y is not None and self.previous_hip_center_y is not None:
            hip_delta = hip_center_y - self.previous_hip_center_y
            if hip_delta >= self.hip_motion_tolerance:
                return "descent"
            if hip_delta <= -self.hip_motion_tolerance:
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
        return self._advance_confirmed_phase(raw_phase, self.confirmation_frames)

    def _clear_rep_tracking(self) -> None:
        self.bottom_seen = False
        self.current_rep_min_knee_angle = None

    def _reset_candidate_rule_tracking(self) -> None:
        self.current_contact_leg: Literal["left", "right"] | None = None
        self.current_leading_leg: Literal["left", "right"] | None = None
        self.trailing_leg_source = "unresolved"
        self.trailing_leg_confidence = 0.0
        self.movement_direction = 0
        self.rep_start_body_center_x = self._last_stand_body_center_x
        self._best_side_contacts = {
            "left": ContactResult("UNSURE", 0.0, None, 0, []),
            "right": ContactResult("UNSURE", 0.0, None, 0, []),
        }
        self._contact_confirmed_frame: int | None = None
        self._full_extension_confirmed_frame: int | None = None
        self._sequence_ready_for_validation = False
        self._knee_extension_hold = 0
        self._hip_extension_hold = 0
        self._extension_observation_frames = 0
        self._knee_extension_observable_frames = 0
        self._hip_extension_observable_frames = 0
        self._knee_extension_evidence: list[int] = []
        self._hip_extension_evidence: list[int] = []
        self._last_lunge_rules: tuple[BodyRuleResult, ...] = ()

    def _start_candidate_rules(self) -> None:
        self._candidate_rule_active = True
        self._reset_candidate_rule_tracking()
        for detector in self._side_knee_detectors.values():
            detector.reset()

    def _consume_step_events(self) -> None:
        for event in self.foot_event_detector.event_history:
            key = (
                event.event_type,
                event.side,
                event.timestamp_ms,
                event.frame_index,
            )
            if key in self._seen_step_event_keys:
                continue
            self._seen_step_event_keys.add(key)
            if event.event_type == "STEP":
                self._steps_since_last_candidate.append(event)

    def _observe_foot_interval(self) -> None:
        if self.frame_index <= 1:
            return
        left = self.foot_event_detector.trackers["left"].last_result
        right = self.foot_event_detector.trackers["right"].last_result
        if not left.observable or not right.observable:
            self._foot_interval_observable = False

    def _update_side_contacts(
        self,
        features: Mapping[str, object],
        *,
        phase: str,
    ) -> None:
        desired_sensitivity = str(getattr(self, "sensitivity", "medium"))
        if any(
            detector.sensitivity != desired_sensitivity
            for detector in self._side_knee_detectors.values()
        ):
            self._side_knee_detectors = {
                side: KneeContactDetector(sensitivity=desired_sensitivity)
                for side in ("left", "right")
            }
        rank = {"NOT_OBSERVABLE": 0, "UNSURE": 1, "NO_CONTACT": 2, "CONTACT": 3}
        for side, detector in self._side_knee_detectors.items():
            result = detector.update(
                features,
                phase=phase,
                frame_index=self.frame_index,
                timestamp_ms=self.last_timestamp_ms,
                side=side,  # type: ignore[arg-type]
            )
            previous = self._best_side_contacts[side]
            if (
                result.status == "CONTACT"
                or (
                    phase == "bottom"
                    and rank[result.status] >= rank[previous.status]
                )
            ):
                self._best_side_contacts[side] = result

    def _toe_x(
        self,
        features: Mapping[str, object],
        side: Literal["left", "right"],
    ) -> tuple[float | None, float]:
        x = _safe_float(features.get(f"{side}_foot_index_x"))
        confidence = _safe_float(features.get(f"{side}_foot_index_confidence"))
        return x, max(0.0, min(1.0, confidence or 0.0))

    def _select_trailing_leg(
        self,
        features: Mapping[str, object],
        body_center_x: float | None,
    ) -> None:
        if self.current_contact_leg is not None and self.trailing_leg_source == "movement_direction":
            return
        start_x = self.rep_start_body_center_x
        displacement = (
            None
            if start_x is None or body_center_x is None
            else body_center_x - start_x
        )
        left_toe, left_confidence = self._toe_x(features, "left")
        right_toe, right_confidence = self._toe_x(features, "right")
        if (
            displacement is not None
            and abs(displacement) >= 0.015
            and left_toe is not None
            and right_toe is not None
            and abs(left_toe - right_toe) >= 0.015
        ):
            self.movement_direction = 1 if displacement > 0.0 else -1
            if self.movement_direction > 0:
                leading = "left" if left_toe > right_toe else "right"
            else:
                leading = "left" if left_toe < right_toe else "right"
            self.current_leading_leg = leading
            self.current_contact_leg = "right" if leading == "left" else "left"
            self.trailing_leg_source = "movement_direction"
            self.trailing_leg_confidence = min(left_confidence, right_confidence)
            return

        left_height = self._best_side_contacts["left"].surface_height_ratio
        right_height = self._best_side_contacts["right"].surface_height_ratio
        if left_height is None and right_height is None:
            return
        if right_height is None or (
            left_height is not None and left_height <= right_height
        ):
            trailing: Literal["left", "right"] = "left"
        else:
            trailing = "right"
        self.current_contact_leg = trailing
        self.current_leading_leg = "right" if trailing == "left" else "left"
        self.trailing_leg_source = "knee_height_fallback"
        height_gap = (
            0.0
            if left_height is None or right_height is None
            else abs(left_height - right_height)
        )
        self.trailing_leg_confidence = min(0.65, 0.45 + height_gap * 2.0)

    def _observe_post_contact_extension(
        self,
        *,
        raw_phase: str,
        left_knee_angle: float | None,
        right_knee_angle: float | None,
        left_hip_angle: float | None,
        right_hip_angle: float | None,
    ) -> None:
        if self._contact_confirmed_frame is None or raw_phase != "stand":
            if raw_phase != "stand":
                self._knee_extension_hold = 0
                self._hip_extension_hold = 0
            return
        self._extension_observation_frames += 1
        if left_knee_angle is not None and right_knee_angle is not None:
            self._knee_extension_observable_frames += 1
        if left_hip_angle is not None and right_hip_angle is not None:
            self._hip_extension_observable_frames += 1
        knee_pass = (
            left_knee_angle is not None
            and right_knee_angle is not None
            and min(left_knee_angle, right_knee_angle)
            >= self.full_extension_knee_angle
        )
        hip_pass = (
            left_hip_angle is not None
            and right_hip_angle is not None
            and min(left_hip_angle, right_hip_angle)
            >= self.full_extension_hip_angle
        )
        if knee_pass:
            self._knee_extension_hold += 1
            self._knee_extension_evidence.append(self.frame_index)
        else:
            self._knee_extension_hold = 0
            self._knee_extension_evidence.clear()
        if hip_pass:
            self._hip_extension_hold += 1
            self._hip_extension_evidence.append(self.frame_index)
        else:
            self._hip_extension_hold = 0
            self._hip_extension_evidence.clear()
        if (
            self._knee_extension_hold >= self.full_extension_hold_frames
            and self._hip_extension_hold >= self.full_extension_hold_frames
            and self._full_extension_confirmed_frame is None
        ):
            self._full_extension_confirmed_frame = self.frame_index

    def _validation_state(self, raw_phase: str) -> str:
        if self.just_completed_rep:
            return "RULE_VALIDATION"
        if self._full_extension_confirmed_frame is not None:
            return "FULL_EXTENSION_CONFIRMED"
        if self._contact_confirmed_frame is not None and raw_phase == "bottom":
            return "KNEE_CONTACT_CONFIRMED"
        if raw_phase in {"descent", "bottom"}:
            return "DESCENDING"
        if raw_phase == "ascent" or (
            raw_phase == "stand" and self._contact_confirmed_frame is not None
        ):
            return "ASCENDING"
        if raw_phase == "stand":
            return "STANDING"
        return "UNKNOWN"

    def _contact_rule(self) -> BodyRuleResult:
        if self.current_contact_leg is None:
            return BodyRuleResult(
                "trailing_knee_contact",
                "UNSURE",
                0.0,
                reason_code="TRAILING_LEG_UNRESOLVED",
            )
        result = self._best_side_contacts[self.current_contact_leg]
        if result.status == "CONTACT":
            status = "PASS"
            reason = None
        elif result.status == "NO_CONTACT":
            status = "FAIL"
            reason = "TRAILING_KNEE_NO_CONTACT"
        else:
            status = "UNSURE"
            reason = (
                "TRAILING_KNEE_NOT_OBSERVABLE"
                if result.status == "NOT_OBSERVABLE"
                else "TRAILING_KNEE_CONTACT_UNSURE"
            )
        return BodyRuleResult(
            "trailing_knee_contact",
            status,  # type: ignore[arg-type]
            min(result.confidence, self.trailing_leg_confidence or result.confidence),
            value=result.surface_height_ratio,
            reason_code=reason,
            evidence_frames=tuple(result.evidence_frames),
        )

    def _extension_rule(
        self,
        *,
        rule_id: str,
        hold_frames: int,
        observable_frames: int,
        evidence: list[int],
    ) -> BodyRuleResult:
        if self._contact_confirmed_frame is None:
            return BodyRuleResult(
                rule_id,
                "UNSURE",
                0.0,
                reason_code="EXTENSION_NOT_AFTER_CONFIRMED_CONTACT",
            )
        if observable_frames < self.full_extension_hold_frames:
            return BodyRuleResult(
                rule_id,
                "UNSURE",
                0.0,
                value=observable_frames,
                reason_code=f"{rule_id.upper()}_NOT_OBSERVABLE",
                evidence_frames=tuple(evidence),
            )
        passed = hold_frames >= self.full_extension_hold_frames
        return BodyRuleResult(
            rule_id,
            "PASS" if passed else "FAIL",
            0.95 if passed else 0.85,
            value=hold_frames,
            reason_code=None if passed else f"{rule_id.upper()}_NOT_HELD",
            evidence_frames=tuple(evidence[-self.full_extension_hold_frames :]),
        )

    def _alternating_rule(self) -> BodyRuleResult:
        if self.current_contact_leg is None:
            return BodyRuleResult(
                "alternating_contact_leg",
                "UNSURE",
                0.0,
                reason_code="CONTACT_LEG_UNRESOLVED",
            )
        alternating = (
            self.previous_valid_contact_leg is None
            or self.current_contact_leg != self.previous_valid_contact_leg
        )
        return BodyRuleResult(
            "alternating_contact_leg",
            "PASS" if alternating else "FAIL",
            max(0.5, self.trailing_leg_confidence),
            value=alternating,
            reason_code=None if alternating else "SAME_CONTACT_LEG_REPEATED",
        )

    def _step_rule(self) -> BodyRuleResult:
        if not self._foot_interval_observable:
            return BodyRuleResult(
                "no_extra_step_or_shuffle",
                "UNSURE",
                0.0,
                reason_code="FOOT_EVENTS_NOT_OBSERVABLE",
            )
        if self.current_leading_leg is None:
            return BodyRuleResult(
                "no_extra_step_or_shuffle",
                "UNSURE",
                0.0,
                reason_code="LEADING_LEG_UNRESOLVED",
            )
        step_sides = [getattr(event, "side", None) for event in self._steps_since_last_candidate]
        allowed = (
            not step_sides
            or (
                len(step_sides) == 1
                and step_sides[0] == self.current_leading_leg
            )
        )
        return BodyRuleResult(
            "no_extra_step_or_shuffle",
            "PASS" if allowed else "FAIL",
            0.90,
            value=len(step_sides),
            reason_code=None if allowed else "EXTRA_STEP_OR_SHUFFLE",
            evidence_frames=tuple(
                frame
                for event in self._steps_since_last_candidate
                for frame in (
                    int(
                        getattr(
                            event,
                            "frame_index",
                            self.frame_index,
                        )
                    ),
                    min(
                        self.frame_index,
                        int(
                            getattr(
                                event,
                                "frame_index",
                                self.frame_index,
                            )
                        )
                        + 1,
                    ),
                )
            ),
        )

    def _validate_lunge_candidate(self, visible_score: float) -> RepDecision:
        rules = (
            self._contact_rule(),
            self._extension_rule(
                rule_id="full_knee_extension",
                hold_frames=self._knee_extension_hold,
                observable_frames=self._knee_extension_observable_frames,
                evidence=self._knee_extension_evidence,
            ),
            self._extension_rule(
                rule_id="full_hip_extension",
                hold_frames=self._hip_extension_hold,
                observable_frames=self._hip_extension_observable_frames,
                evidence=self._hip_extension_evidence,
            ),
            self._alternating_rule(),
            self._step_rule(),
        )
        self._last_lunge_rules = rules
        decision = self.register_rep_candidate(
            rules,
            required_rules=LUNGE_REQUIRED_RULES,
            events={
                "terminal_phase": "stand",
                "contact_leg": self.current_contact_leg,
                "leading_leg": self.current_leading_leg,
                "trailing_leg_source": self.trailing_leg_source,
                "trailing_leg_confidence": self.trailing_leg_confidence,
                "movement_direction": self.movement_direction,
                "step_event_count": len(self._steps_since_last_candidate),
                "contact_confirmed_frame": self._contact_confirmed_frame,
                "full_extension_confirmed_frame": (
                    self._full_extension_confirmed_frame
                ),
                "visible_score": visible_score,
            },
        )
        if decision.status == "VALID":
            self.previous_valid_contact_leg = self.current_contact_leg
        self._steps_since_last_candidate = []
        self._foot_interval_observable = True
        self._candidate_rule_active = False
        return decision

    def _cooldown_elapsed(self, timestamp_ms: int | None) -> bool:
        if timestamp_ms is None or self.last_rep_time_ms is None:
            return True
        return timestamp_ms - self.last_rep_time_ms >= self.rep_cooldown_ms

    def _finalize_feedback(self, messages: list[FeedbackMessage]) -> list[FeedbackMessage]:
        ordered = sorted(messages, key=_feedback_sort_key)
        return self.limit_feedback(ordered)

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
        self.begin_frame(features, timestamp_ms)
        self.last_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms)
        self.update_foot_events_for_current_frame()
        self._consume_step_events()
        self._observe_foot_interval()
        self.just_completed_rep = False
        self.rep_sequence.just_completed = False
        visible_score = self._visible_score(features)
        left_knee_angle = _safe_float(
            None if features is None else features.get("left_knee_angle")
        )
        right_knee_angle = _safe_float(
            None if features is None else features.get("right_knee_angle")
        )
        left_hip_angle = _safe_float(
            None if features is None else features.get("left_hip_angle")
        )
        right_hip_angle = _safe_float(
            None if features is None else features.get("right_hip_angle")
        )
        min_knee_angle = _min_metric(left_knee_angle, right_knee_angle)
        min_hip_angle = _min_metric(left_hip_angle, right_hip_angle)
        torso_angle = _safe_float(None if features is None else features.get("torso_angle"))
        hip_center_y = _safe_float(None if features is None else features.get("hip_center_y"))
        body_center_x = _safe_float(
            None if features is None else features.get("body_center_x")
        )
        self.hip_drop = (
            None
            if hip_center_y is None or self.stand_hip_center_y is None
            else hip_center_y - self.stand_hip_center_y
        )

        if visible_score < self.min_visible_score:
            raw_phase = "no_pose" if visible_score <= 0.0 else "low_visibility"
            previous_stable_phase, stable_phase = self._advance_phase(raw_phase)
            if stable_phase != previous_stable_phase and stable_phase in {"low_visibility", "no_pose"}:
                self._clear_rep_tracking()
                self.rep_sequence.reset()
                self._candidate_rule_active = False
                self._reset_candidate_rule_tracking()
            self.previous_min_knee_angle = None
            self.previous_hip_center_y = None
            feedback_messages = self._visibility_feedback(visible_score)
            return self.finalize_state({
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
                    "hip_center_y": hip_center_y,
                    "stand_hip_center_y": self.stand_hip_center_y,
                    "hip_drop": self.hip_drop,
                    "body_center_x": body_center_x,
                    "raw_phase": self.raw_phase,
                    "stable_phase": self.stable_phase,
                    "frames_in_phase": self.frames_in_phase,
                    "last_rep_time_ms": self.last_rep_time_ms,
                    "confirmation_frames": self.confirmation_frames,
                    "rep_cooldown_ms": self.rep_cooldown_ms,
                    "sensitivity": self.sensitivity,
                    "config_name": self.config_name,
                    "previous_valid_contact_leg": self.previous_valid_contact_leg,
                    "current_contact_leg": self.current_contact_leg,
                    "lunge_required_rules": list(LUNGE_REQUIRED_RULES),
                    **self.rep_sequence.debug(),
                },
            })

        raw_phase = self._phase_from_features(min_knee_angle, hip_center_y)
        if raw_phase == "stand":
            self._last_stand_body_center_x = body_center_x
        if (
            raw_phase in {"descent", "bottom"}
            and not self._candidate_rule_active
            and (
                self._previous_raw_phase_for_rules == "stand"
                or self.rep_sequence.progress > 0
            )
        ):
            self._start_candidate_rules()
        if self._candidate_rule_active:
            self._update_side_contacts(
                features if isinstance(features, Mapping) else {},
                phase=raw_phase,
            )
            self._select_trailing_leg(
                features if isinstance(features, Mapping) else {},
                body_center_x,
            )
            if self.current_contact_leg is not None:
                contact = self._best_side_contacts[self.current_contact_leg]
                if contact.status == "CONTACT" and self._contact_confirmed_frame is None:
                    self._contact_confirmed_frame = self.frame_index
            self._observe_post_contact_extension(
                raw_phase=raw_phase,
                left_knee_angle=left_knee_angle,
                right_knee_angle=right_knee_angle,
                left_hip_angle=left_hip_angle,
                right_hip_angle=right_hip_angle,
            )
        previous_stable_phase, stable_phase = self._advance_phase(raw_phase)

        if raw_phase == "stand" and hip_center_y is not None:
            if self.stand_hip_center_y is None:
                self.stand_hip_center_y = hip_center_y
            else:
                self.stand_hip_center_y = 0.9 * self.stand_hip_center_y + 0.1 * hip_center_y
            self.hip_drop = hip_center_y - self.stand_hip_center_y

        if raw_phase in {"descent", "bottom", "ascent"} and min_knee_angle is not None:
            if self.current_rep_min_knee_angle is None:
                self.current_rep_min_knee_angle = min_knee_angle
            else:
                self.current_rep_min_knee_angle = min(self.current_rep_min_knee_angle, min_knee_angle)

        sequence_completed = False
        if stable_phase != previous_stable_phase:
            sequence_completed = self.rep_sequence.update(stable_phase)
            if stable_phase == "bottom":
                self.bottom_seen = True
            if stable_phase == "stand":
                if sequence_completed:
                    self._sequence_ready_for_validation = True
                self._clear_rep_tracking()
            elif stable_phase in {"low_visibility", "no_pose"}:
                self._clear_rep_tracking()

        if self._sequence_ready_for_validation:
            ready_without_contact = self._contact_confirmed_frame is None
            extension_window_complete = (
                self._extension_observation_frames
                >= self.full_extension_hold_frames
            )
            if ready_without_contact or extension_window_complete:
                self._validate_lunge_candidate(visible_score)
                self._sequence_ready_for_validation = False
                self.just_completed_rep = True
                self.last_rep_time_ms = self.last_timestamp_ms

        self.previous_min_knee_angle = min_knee_angle
        self.previous_hip_center_y = hip_center_y
        self._previous_raw_phase_for_rules = raw_phase
        feedback_messages = self._build_lunge_feedback(
            stable_phase=stable_phase,
            min_knee_angle=min_knee_angle,
            min_hip_angle=min_hip_angle,
            torso_angle=torso_angle,
            just_completed_rep=self.just_completed_rep,
        )
        return self.finalize_state({
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
                "hip_center_y": hip_center_y,
                "stand_hip_center_y": self.stand_hip_center_y,
                "hip_drop": self.hip_drop,
                "body_center_x": body_center_x,
                "raw_phase": self.raw_phase,
                "stable_phase": self.stable_phase,
                "frames_in_phase": self.frames_in_phase,
                "last_rep_time_ms": self.last_rep_time_ms,
                "bottom_seen": self.bottom_seen,
                "just_completed_rep": self.just_completed_rep,
                "confirmation_frames": self.confirmation_frames,
                "rep_cooldown_ms": self.rep_cooldown_ms,
                "hip_motion_tolerance": self.hip_motion_tolerance,
                "hip_drop_min": self.hip_drop_min,
                "sensitivity": self.sensitivity,
                "config_name": self.config_name,
                "full_extension_hold_frames": self.full_extension_hold_frames,
                "knee_extension_hold": self._knee_extension_hold,
                "hip_extension_hold": self._hip_extension_hold,
                "extension_observation_frames": self._extension_observation_frames,
                "contact_confirmed_frame": self._contact_confirmed_frame,
                "full_extension_confirmed_frame": self._full_extension_confirmed_frame,
                "lunge_validation_state": self._validation_state(raw_phase),
                "current_contact_leg": self.current_contact_leg,
                "current_leading_leg": self.current_leading_leg,
                "previous_valid_contact_leg": self.previous_valid_contact_leg,
                "trailing_leg_source": self.trailing_leg_source,
                "trailing_leg_confidence": self.trailing_leg_confidence,
                "movement_direction": self.movement_direction,
                "steps_since_last_candidate": len(self._steps_since_last_candidate),
                "foot_interval_observable": self._foot_interval_observable,
                "lunge_required_rules": list(LUNGE_REQUIRED_RULES),
                "last_lunge_rules": [
                    rule.as_dict() for rule in self._last_lunge_rules
                ],
                "side_knee_contacts": {
                    side: result.as_dict()
                    for side, result in self._best_side_contacts.items()
                },
                **self.rep_sequence.debug(),
            },
        })


__all__ = [
    "FULL_EXTENSION_HOLD_FRAMES",
    "LUNGE_REQUIRED_RULES",
    "LungeAnalyzer",
    "PHASE_CONFIRMATION_FRAMES",
    "REP_COOLDOWN_MS",
    "SENSITIVITY_PROFILES",
]
