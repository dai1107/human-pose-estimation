from __future__ import annotations

from collections.abc import Mapping
from math import hypot
from typing import Literal

from hyrox.base import BaseActionAnalyzer, PhaseSequenceTracker
from hyrox.config import load_burpee_broad_jump_config
from hyrox.contact import ChestContactDetector, ContactResult
from hyrox.feedback import FeedbackMessage
from hyrox.foot_events import FootStaggerResult
from hyrox.validity import BodyRuleResult, RepDecision


SENSITIVITY_FRAME_DELTAS = {"low": 1, "medium": 0, "high": -1}
BURPEE_REQUIRED_RULES = (
    "chest_ground_contact",
    "simultaneous_takeoff",
    "simultaneous_landing",
    "takeoff_stagger_proxy",
    "landing_stagger_proxy",
    "no_extra_step_or_shuffle",
    "legal_hand_placement_proxy",
    "forward_jump_detected",
)


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
        self.hand_placement_pass_ratio = _resolved_float(
            values.get("hand_placement_pass_foot_length_ratio"),
            1.25,
        )
        self.hand_placement_unsure_ratio = _resolved_float(
            values.get("hand_placement_unsure_foot_length_ratio"),
            1.45,
        )
        self.forward_jump_com_leg_ratio = _resolved_float(
            values.get("forward_jump_min_com_displacement_leg_ratio"),
            0.20,
        )
        self.forward_jump_feet_leg_ratio = _resolved_float(
            values.get("forward_jump_min_both_feet_displacement_leg_ratio"),
            0.15,
        )

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
        self._burpee_chest_detector = ChestContactDetector(
            sensitivity=str(getattr(self, "sensitivity", "medium"))
        )
        self._pending_hand_rule: BodyRuleResult | None = None
        self._hand_proxy_debug: dict[str, object] = {
            "proxy_name": "LEGAL_HAND_PLACEMENT_PROXY",
            "status": "NOT_OBSERVABLE",
            "ratio": None,
        }
        self._last_forward_direction = 0
        self._cycle_active = False
        self._awaiting_next_hands = False
        self._sequence_landing_ready = False
        self._post_landing_steps: list[object] = []
        self._post_landing_foot_observable = True
        self._last_burpee_rules: tuple[BodyRuleResult, ...] = ()
        self._validated_this_frame = False
        self._reset_cycle_evidence()
        self.rep_sequence = PhaseSequenceTracker(
            ("chest_down", "step_or_jump_in", "broad_jump_takeoff", "flight_or_move", "landing"),
            optional_phases=("step_or_jump_in", "flight_or_move"),
        )

    def _reset_cycle_evidence(self) -> None:
        self._best_chest_contact = ContactResult("UNSURE", 0.0, None, 0, [])
        self._cycle_takeoffs: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._cycle_landings: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self._takeoff_stagger: FootStaggerResult | None = None
        self._landing_stagger: FootStaggerResult | None = None
        self._last_grounded_snapshot: dict[str, object] | None = None
        self._last_grounded_stagger: FootStaggerResult | None = None
        self._takeoff_snapshot: dict[str, object] | None = None
        self._landing_snapshot: dict[str, object] | None = None
        self._cycle_expected_direction = self._last_forward_direction
        self._observed_jump_direction = 0
        self._cycle_foot_observable = True
        self._cycle_hand_rule = BodyRuleResult(
            "legal_hand_placement_proxy",
            "UNSURE",
            0.0,
            reason_code="LEGAL_HAND_PLACEMENT_PROXY_NOT_OBSERVABLE",
        )

    def _start_cycle(self) -> None:
        desired_sensitivity = str(getattr(self, "sensitivity", "medium"))
        self._burpee_chest_detector = ChestContactDetector(
            sensitivity=desired_sensitivity
        )
        pending_hand = self._pending_hand_rule
        pending_debug = dict(self._hand_proxy_debug)
        self._reset_cycle_evidence()
        if pending_hand is not None:
            self._cycle_hand_rule = pending_hand
            self._hand_proxy_debug = pending_debug
        self._pending_hand_rule = None
        self._cycle_active = True
        self._sequence_landing_ready = False
        self._post_landing_steps = []
        self._post_landing_foot_observable = True

    def _feature_point(
        self,
        features: Mapping[str, object],
        name: str,
        *,
        min_confidence: float = 0.60,
    ) -> tuple[float, float, float] | None:
        x = _safe_float(features.get(f"{name}_x"))
        y = _safe_float(features.get(f"{name}_y"))
        confidence = _safe_float(features.get(f"{name}_confidence"))
        if (
            x is None
            or y is None
            or confidence is None
            or confidence < min_confidence
        ):
            return None
        return x, y, confidence

    def _foot_center(
        self,
        features: Mapping[str, object],
        side: Literal["left", "right"],
    ) -> tuple[float, float] | None:
        heel = self._feature_point(features, f"{side}_heel")
        toe = self._feature_point(features, f"{side}_foot_index")
        if heel is None or toe is None:
            return None
        return (heel[0] + toe[0]) / 2.0, (heel[1] + toe[1]) / 2.0

    def _mean_leg_length(self, features: Mapping[str, object]) -> float | None:
        lengths: list[float] = []
        for side in ("left", "right"):
            hip = self._feature_point(features, f"{side}_hip")
            knee = self._feature_point(features, f"{side}_knee")
            ankle = self._feature_point(features, f"{side}_ankle")
            if hip is None or knee is None or ankle is None:
                continue
            lengths.append(
                hypot(hip[0] - knee[0], hip[1] - knee[1])
                + hypot(knee[0] - ankle[0], knee[1] - ankle[1])
            )
        if lengths:
            return sum(lengths) / len(lengths)
        body_height = _safe_float(features.get("body_height_reference"))
        return None if body_height is None else 0.53 * body_height

    def _foot_snapshot(
        self,
        features: Mapping[str, object],
        body_center_x: float | None,
    ) -> dict[str, object] | None:
        left = self._foot_center(features, "left")
        right = self._foot_center(features, "right")
        leg_length = self._mean_leg_length(features)
        if left is None or right is None or leg_length is None or leg_length <= 1e-6:
            return None
        return {
            "body_center_x": body_center_x,
            "left_foot_x": left[0],
            "right_foot_x": right[0],
            "mean_leg_length": leg_length,
        }

    def _capture_foot_evidence(
        self,
        features: Mapping[str, object],
        body_center_x: float | None,
    ) -> None:
        left_state = self.foot_event_detector.trackers["left"].last_result
        right_state = self.foot_event_detector.trackers["right"].last_result
        both_observable = left_state.observable and right_state.observable
        if self._cycle_active and not both_observable:
            self._cycle_foot_observable = False
        if self._awaiting_next_hands and not both_observable:
            self._post_landing_foot_observable = False

        both_grounded = (
            both_observable
            and left_state.state == "GROUNDED"
            and right_state.state == "GROUNDED"
        )
        if self._cycle_active and both_grounded:
            snapshot = self._foot_snapshot(features, body_center_x)
            if not any(self._cycle_takeoffs.values()):
                self._last_grounded_snapshot = snapshot
                self._last_grounded_stagger = self.foot_event_detector.last_stagger
            if (
                all(value is not None for value in self._cycle_landings.values())
                and self._landing_snapshot is None
            ):
                self._landing_snapshot = snapshot
                self._landing_stagger = self.foot_event_detector.last_stagger
                if (
                    snapshot is not None
                    and self._takeoff_snapshot is not None
                ):
                    start_x = _safe_float(
                        self._takeoff_snapshot.get("body_center_x")
                    )
                    end_x = _safe_float(snapshot.get("body_center_x"))
                    if start_x is not None and end_x is not None and abs(end_x - start_x) >= 0.005:
                        self._observed_jump_direction = 1 if end_x > start_x else -1

        for event in self.foot_event_detector.new_events:
            if event.event_type == "TAKEOFF" and self._cycle_active:
                if self._cycle_takeoffs[event.side] is None:
                    self._cycle_takeoffs[event.side] = event.timestamp_ms
                    if self._takeoff_snapshot is None:
                        self._takeoff_snapshot = self._last_grounded_snapshot
                        self._takeoff_stagger = self._last_grounded_stagger
            elif event.event_type == "LANDING" and self._cycle_active:
                if self._cycle_landings[event.side] is None:
                    self._cycle_landings[event.side] = event.timestamp_ms
            elif event.event_type == "STEP" and self._awaiting_next_hands:
                jump_landing = self._cycle_landings.get(event.side)
                if jump_landing is not None and event.timestamp_ms == jump_landing:
                    continue
                self._post_landing_steps.append(event)

    def _hand_placement_rule(
        self,
        features: Mapping[str, object],
    ) -> BodyRuleResult:
        names = (
            "left_wrist",
            "right_wrist",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        )
        points = {
            name: self._feature_point(features, name)
            for name in names
        }
        if any(point is None for point in points.values()):
            self._hand_proxy_debug = {
                "proxy_name": "LEGAL_HAND_PLACEMENT_PROXY",
                "status": "NOT_OBSERVABLE",
                "ratio": None,
            }
            return BodyRuleResult(
                "legal_hand_placement_proxy",
                "UNSURE",
                0.0,
                reason_code="LEGAL_HAND_PLACEMENT_PROXY_NOT_OBSERVABLE",
            )
        left_wrist = points["left_wrist"]
        right_wrist = points["right_wrist"]
        left_heel = points["left_heel"]
        right_heel = points["right_heel"]
        left_toe = points["left_foot_index"]
        right_toe = points["right_foot_index"]
        assert (
            left_wrist
            and right_wrist
            and left_heel
            and right_heel
            and left_toe
            and right_toe
        )
        floor_x1 = _safe_float(features.get("floor_line_x1"))
        floor_y1 = _safe_float(features.get("floor_line_y1"))
        floor_x2 = _safe_float(features.get("floor_line_x2"))
        floor_y2 = _safe_float(features.get("floor_line_y2"))
        if None in {floor_x1, floor_y1, floor_x2, floor_y2}:
            return BodyRuleResult(
                "legal_hand_placement_proxy",
                "UNSURE",
                0.0,
                reason_code="LEGAL_HAND_PLACEMENT_PROXY_FLOOR_UNSURE",
            )
        dx = float(floor_x2) - float(floor_x1)
        dy = float(floor_y2) - float(floor_y1)
        line_length = hypot(dx, dy)
        if line_length <= 1e-6:
            return BodyRuleResult(
                "legal_hand_placement_proxy",
                "UNSURE",
                0.0,
                reason_code="LEGAL_HAND_PLACEMENT_PROXY_FLOOR_UNSURE",
            )
        unit = (dx / line_length, dy / line_length)

        def project(point: tuple[float, float, float]) -> float:
            return point[0] * unit[0] + point[1] * unit[1]

        wrist_positions = [project(left_wrist), project(right_wrist)]
        toe_positions = [project(left_toe), project(right_toe)]
        foot_lengths = [
            hypot(left_toe[0] - left_heel[0], left_toe[1] - left_heel[1]),
            hypot(right_toe[0] - right_heel[0], right_toe[1] - right_heel[1]),
        ]
        mean_foot_length = sum(foot_lengths) / 2.0
        if mean_foot_length <= 1e-6:
            return BodyRuleResult(
                "legal_hand_placement_proxy",
                "UNSURE",
                0.0,
                reason_code="LEGAL_HAND_PLACEMENT_PROXY_FOOT_LENGTH_UNSURE",
            )
        if self._last_forward_direction > 0:
            forward_distance = max(0.0, max(wrist_positions) - max(toe_positions))
        elif self._last_forward_direction < 0:
            forward_distance = max(0.0, min(toe_positions) - min(wrist_positions))
        else:
            forward_distance = max(
                0.0,
                max(wrist_positions) - max(toe_positions),
                min(toe_positions) - min(wrist_positions),
            )
        ratio = forward_distance / mean_foot_length
        if ratio <= self.hand_placement_pass_ratio:
            status = "PASS"
            reason = None
        elif ratio <= self.hand_placement_unsure_ratio:
            status = "UNSURE"
            reason = "LEGAL_HAND_PLACEMENT_PROXY_BORDERLINE"
        else:
            status = "FAIL"
            reason = "LEGAL_HAND_PLACEMENT_PROXY_TOO_FAR"
        confidence = min(point[2] for point in points.values() if point is not None)
        self._hand_proxy_debug = {
            "proxy_name": "LEGAL_HAND_PLACEMENT_PROXY",
            "status": status,
            "ratio": ratio,
            "mean_foot_length": mean_foot_length,
            "forward_direction": self._last_forward_direction,
        }
        return BodyRuleResult(
            "legal_hand_placement_proxy",
            status,  # type: ignore[arg-type]
            confidence,
            value=ratio,
            reason_code=reason,
            evidence_frames=(self.frame_index,),
        )

    def _chest_rule(self) -> BodyRuleResult:
        result = self._best_chest_contact
        if result.status == "CONTACT":
            status = "PASS"
            reason = None
        elif result.status == "NO_CONTACT":
            status = "FAIL"
            reason = "CHEST_GROUND_CONTACT_NOT_CONFIRMED"
        else:
            status = "UNSURE"
            reason = (
                "CHEST_GROUND_CONTACT_NOT_OBSERVABLE"
                if result.status == "NOT_OBSERVABLE"
                else "CHEST_GROUND_CONTACT_UNSURE"
            )
        return BodyRuleResult(
            "chest_ground_contact",
            status,  # type: ignore[arg-type]
            result.confidence,
            value=result.surface_height_ratio,
            reason_code=reason,
            evidence_frames=tuple(result.evidence_frames),
        )

    def _sync_rule(
        self,
        rule_id: str,
        events: Mapping[str, int | None],
    ) -> BodyRuleResult:
        left = events.get("left")
        right = events.get("right")
        if left is None or right is None:
            status = "FAIL" if self._cycle_foot_observable else "UNSURE"
            return BodyRuleResult(
                rule_id,
                status,  # type: ignore[arg-type]
                0.85 if status == "FAIL" else 0.0,
                reason_code=(
                    f"{rule_id.upper()}_EVENT_MISSING"
                    if status == "FAIL"
                    else f"{rule_id.upper()}_NOT_OBSERVABLE"
                ),
            )
        delta = abs(left - right)
        if delta <= self.foot_event_detector.sync_config.pass_ms:
            status = "PASS"
            reason = None
        elif delta <= self.foot_event_detector.sync_config.unsure_ms:
            status = "UNSURE"
            reason = f"{rule_id.upper()}_BORDERLINE"
        else:
            status = "FAIL"
            reason = f"{rule_id.upper()}_ASYNCHRONOUS"
        return BodyRuleResult(
            rule_id,
            status,  # type: ignore[arg-type]
            0.95,
            value=delta,
            reason_code=reason,
        )

    def _stagger_rule(
        self,
        rule_id: str,
        result: FootStaggerResult | None,
    ) -> BodyRuleResult:
        if result is None or result.status == "NOT_OBSERVABLE":
            return BodyRuleResult(
                rule_id,
                "UNSURE",
                0.0,
                reason_code=f"{rule_id.upper()}_NOT_OBSERVABLE",
            )
        return BodyRuleResult(
            rule_id,
            result.status,  # type: ignore[arg-type]
            result.confidence,
            value=result.stagger_ratio,
            reason_code=(
                None
                if result.status == "PASS"
                else f"{rule_id.upper()}_{result.status}"
            ),
        )

    def _forward_jump_rule(self) -> BodyRuleResult:
        start = self._takeoff_snapshot
        end = self._landing_snapshot
        if start is None or end is None:
            return BodyRuleResult(
                "forward_jump_detected",
                "UNSURE",
                0.0,
                reason_code="FORWARD_JUMP_NOT_OBSERVABLE",
            )
        leg_length = _safe_float(start.get("mean_leg_length"))
        start_body = _safe_float(start.get("body_center_x"))
        end_body = _safe_float(end.get("body_center_x"))
        values = (
            leg_length,
            start_body,
            end_body,
            _safe_float(start.get("left_foot_x")),
            _safe_float(start.get("right_foot_x")),
            _safe_float(end.get("left_foot_x")),
            _safe_float(end.get("right_foot_x")),
        )
        if any(value is None for value in values) or leg_length is None or leg_length <= 1e-6:
            return BodyRuleResult(
                "forward_jump_detected",
                "UNSURE",
                0.0,
                reason_code="FORWARD_JUMP_NOT_OBSERVABLE",
            )
        (
            _,
            resolved_start_body,
            resolved_end_body,
            start_left,
            start_right,
            end_left,
            end_right,
        ) = values
        assert (
            resolved_start_body is not None
            and resolved_end_body is not None
            and start_left is not None
            and start_right is not None
            and end_left is not None
            and end_right is not None
        )
        com_ratio = abs(resolved_end_body - resolved_start_body) / leg_length
        left_displacement = end_left - start_left
        right_displacement = end_right - start_right
        left_ratio = abs(left_displacement) / leg_length
        right_ratio = abs(right_displacement) / leg_length
        observed_direction = (
            self._observed_jump_direction
            if self._observed_jump_direction != 0
            else (1 if resolved_end_body > resolved_start_body else -1)
        )
        direction_ok = (
            self._cycle_expected_direction == 0
            or observed_direction == self._cycle_expected_direction
        )
        feet_direction_ok = (
            left_displacement * observed_direction > 0.0
            and right_displacement * observed_direction > 0.0
        )
        passed = (
            com_ratio >= self.forward_jump_com_leg_ratio
            and min(left_ratio, right_ratio) >= self.forward_jump_feet_leg_ratio
            and direction_ok
            and feet_direction_ok
        )
        return BodyRuleResult(
            "forward_jump_detected",
            "PASS" if passed else "FAIL",
            0.95,
            value=min(com_ratio, left_ratio, right_ratio),
            reason_code=None if passed else "FORWARD_JUMP_DISPLACEMENT_TOO_SMALL",
        )

    def _no_extra_step_rule(self) -> BodyRuleResult:
        if not self._post_landing_foot_observable:
            return BodyRuleResult(
                "no_extra_step_or_shuffle",
                "UNSURE",
                0.0,
                reason_code="POST_LANDING_FEET_NOT_OBSERVABLE",
            )
        passed = not self._post_landing_steps
        return BodyRuleResult(
            "no_extra_step_or_shuffle",
            "PASS" if passed else "FAIL",
            0.90,
            value=len(self._post_landing_steps),
            reason_code=None if passed else "EXTRA_STEP_OR_SHUFFLE",
            evidence_frames=tuple(
                frame
                for event in self._post_landing_steps
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

    def _validate_pending_candidate(
        self,
        visible_score: float,
    ) -> RepDecision:
        rules = (
            self._chest_rule(),
            self._sync_rule("simultaneous_takeoff", self._cycle_takeoffs),
            self._sync_rule("simultaneous_landing", self._cycle_landings),
            self._stagger_rule("takeoff_stagger_proxy", self._takeoff_stagger),
            self._stagger_rule("landing_stagger_proxy", self._landing_stagger),
            self._no_extra_step_rule(),
            self._cycle_hand_rule,
            self._forward_jump_rule(),
        )
        self._last_burpee_rules = rules
        decision = self.register_rep_candidate(
            rules,
            required_rules=BURPEE_REQUIRED_RULES,
            events={
                "terminal_phase": "landing",
                "validation_boundary": "next_hands_down",
                "takeoff_ms": dict(self._cycle_takeoffs),
                "landing_ms": dict(self._cycle_landings),
                "takeoff_stagger_ratio": (
                    None
                    if self._takeoff_stagger is None
                    else self._takeoff_stagger.stagger_ratio
                ),
                "landing_stagger_ratio": (
                    None
                    if self._landing_stagger is None
                    else self._landing_stagger.stagger_ratio
                ),
                "hand_placement_proxy_name": "LEGAL_HAND_PLACEMENT_PROXY",
                "post_landing_step_count": len(self._post_landing_steps),
                "visible_score": visible_score,
            },
        )
        if decision.status == "VALID" and self._observed_jump_direction != 0:
            self._last_forward_direction = self._observed_jump_direction
        self.last_rep_time_ms = self.last_timestamp_ms
        self._awaiting_next_hands = False
        self._sequence_landing_ready = False
        self._cycle_active = False
        self._post_landing_steps = []
        self._post_landing_foot_observable = True
        return decision

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
        visible_score: float,
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
                self._sequence_landing_ready = True
                self._awaiting_next_hands = True
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
        self.begin_frame(features, timestamp_ms)
        self.last_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms)
        self.update_foot_events_for_current_frame()
        values = features if isinstance(features, dict) else {}
        current_timestamp = None if timestamp_ms is None else int(timestamp_ms)
        self._validated_this_frame = False
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
        self._capture_foot_evidence(values, body_center_x)
        if (
            self._awaiting_next_hands
            and raw_phase in {"hands_down", "chest_down"}
        ):
            self._validate_pending_candidate(visible_score)
            self._validated_this_frame = True

        if raw_phase == "hands_down":
            self._pending_hand_rule = self._hand_placement_rule(values)
        elif raw_phase == "chest_down":
            if self._pending_hand_rule is None:
                self._pending_hand_rule = self._hand_placement_rule(values)
            if not self._cycle_active:
                self._start_cycle()
                self._capture_foot_evidence(values, body_center_x)
            desired_sensitivity = str(getattr(self, "sensitivity", "medium"))
            if self._burpee_chest_detector.sensitivity != desired_sensitivity:
                self._burpee_chest_detector = ChestContactDetector(
                    sensitivity=desired_sensitivity
                )
            chest_result = self._burpee_chest_detector.update(
                values,
                phase="chest_down",
                frame_index=self.frame_index,
                timestamp_ms=current_timestamp,
            )
            rank = {
                "NOT_OBSERVABLE": 0,
                "UNSURE": 1,
                "NO_CONTACT": 2,
                "CONTACT": 3,
            }
            if rank[chest_result.status] >= rank[self._best_chest_contact.status]:
                self._best_chest_contact = chest_result
        previous, _ = self._advance_phase(raw_phase, current_timestamp)
        self._update_sequence(previous, body_center_x, current_timestamp, visible_score)
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
        return self.finalize_state({
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
                "burpee_required_rules": list(BURPEE_REQUIRED_RULES),
                "burpee_validation_state": (
                    "RULE_VALIDATION"
                    if self._validated_this_frame
                    else "AWAITING_NEXT_HANDS"
                    if self._awaiting_next_hands
                    else "CHEST_CONTACT_CONFIRMED"
                    if self._best_chest_contact.status == "CONTACT"
                    else "CHEST_CONTACT_CANDIDATE"
                    if self._cycle_active and raw_phase == "chest_down"
                    else "MOVEMENT_SEQUENCE"
                ),
                "chest_contact": self._best_chest_contact.as_dict(),
                "cycle_takeoff_ms": dict(self._cycle_takeoffs),
                "cycle_landing_ms": dict(self._cycle_landings),
                "takeoff_stagger_proxy": (
                    None
                    if self._takeoff_stagger is None
                    else self._takeoff_stagger.as_dict()
                ),
                "landing_stagger_proxy": (
                    None
                    if self._landing_stagger is None
                    else self._landing_stagger.as_dict()
                ),
                "legal_hand_placement_proxy": dict(self._hand_proxy_debug),
                "post_landing_step_count": len(self._post_landing_steps),
                "cycle_foot_observable": self._cycle_foot_observable,
                "last_burpee_rules": [
                    rule.as_dict() for rule in self._last_burpee_rules
                ],
                **self.rep_sequence.debug(),
            },
        })


__all__ = ["BURPEE_REQUIRED_RULES", "BurpeeBroadJumpAnalyzer"]
