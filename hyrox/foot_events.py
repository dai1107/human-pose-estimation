from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Literal

from .floor_reference import FloorLine, signed_distance_to_floor
from .geometry import PosePoint


FootSupportState = Literal[
    "GROUNDED",
    "TAKEOFF_CANDIDATE",
    "AIRBORNE",
    "LANDING_CANDIDATE",
]
FootEventType = Literal["TAKEOFF", "LANDING", "STEP"]
RuleStatus = Literal["PASS", "UNSURE", "FAIL", "NOT_OBSERVABLE"]


def _safe_float(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if isfinite(resolved) else None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _timestamp(timestamp_ms: int | None, frame_index: int) -> int:
    return int(timestamp_ms) if timestamp_ms is not None else max(0, int(frame_index) * 33)


def _floor_line(features: Mapping[str, object]) -> FloorLine | None:
    values = [
        _safe_float(features.get(name))
        for name in ("floor_line_x1", "floor_line_y1", "floor_line_x2", "floor_line_y2")
    ]
    if any(value is None for value in values):
        return None
    try:
        return FloorLine(
            PosePoint(float(values[0]), float(values[1])),
            PosePoint(float(values[2]), float(values[3])),
        )
    except (TypeError, ValueError):
        return None


def _point(
    features: Mapping[str, object],
    name: str,
    min_confidence: float,
) -> PosePoint | None:
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
    return PosePoint(x, y, visibility=confidence, presence=confidence)


@dataclass(frozen=True)
class FootSupportConfig:
    grounded_height_body_ratio: float = 0.025
    airborne_height_body_ratio: float = 0.055
    min_vertical_speed_body_per_second: float = 0.02
    min_landmark_confidence: float = 0.60
    transition_frames: Mapping[str, int] = field(
        default_factory=lambda: {"high": 1, "medium": 2, "low": 3}
    )


@dataclass(frozen=True)
class FootSyncConfig:
    pass_ms: int = 100
    unsure_ms: int = 180


@dataclass(frozen=True)
class FootStaggerConfig:
    pass_foot_length_ratio: float = 0.20
    unsure_foot_length_ratio: float = 0.30


@dataclass(frozen=True)
class StepEventConfig:
    min_horizontal_displacement_leg_ratio: float = 0.07
    min_airborne_ms: int = 80
    min_grounded_ms: int = 80


@dataclass(frozen=True)
class FootEvent:
    event_type: FootEventType
    side: Literal["left", "right"]
    timestamp_ms: int
    frame_index: int
    foot_x: float | None
    airborne_ms: int | None = None
    horizontal_displacement_leg_ratio: float | None = None
    signed_horizontal_displacement: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "side": self.side,
            "timestamp_ms": self.timestamp_ms,
            "frame_index": self.frame_index,
            "foot_x": self.foot_x,
            "airborne_ms": self.airborne_ms,
            "horizontal_displacement_leg_ratio": (
                self.horizontal_displacement_leg_ratio
            ),
            "signed_horizontal_displacement": self.signed_horizontal_displacement,
        }


@dataclass(frozen=True)
class FootStateResult:
    side: Literal["left", "right"]
    state: FootSupportState
    observable: bool
    confidence: float
    heel_height_ratio: float | None
    toe_height_ratio: float | None
    support_height_ratio: float | None
    vertical_speed_body_per_second: float | None
    takeoff_ms: int | None
    landing_ms: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "state": self.state,
            "observable": self.observable,
            "confidence": round(_clamp(self.confidence), 4),
            "heel_height_ratio": self.heel_height_ratio,
            "toe_height_ratio": self.toe_height_ratio,
            "support_height_ratio": self.support_height_ratio,
            "vertical_speed_body_per_second": (
                self.vertical_speed_body_per_second
            ),
            "takeoff_ms": self.takeoff_ms,
            "landing_ms": self.landing_ms,
        }


@dataclass(frozen=True)
class FootSyncResult:
    takeoff_status: RuleStatus
    takeoff_delta_ms: int | None
    landing_status: RuleStatus
    landing_delta_ms: int | None
    left_takeoff_ms: int | None
    right_takeoff_ms: int | None
    left_landing_ms: int | None
    right_landing_ms: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "takeoff_status": self.takeoff_status,
            "takeoff_delta_ms": self.takeoff_delta_ms,
            "landing_status": self.landing_status,
            "landing_delta_ms": self.landing_delta_ms,
            "left_takeoff_ms": self.left_takeoff_ms,
            "right_takeoff_ms": self.right_takeoff_ms,
            "left_landing_ms": self.left_landing_ms,
            "right_landing_ms": self.right_landing_ms,
        }


@dataclass(frozen=True)
class FootStaggerResult:
    rule_id: Literal["FOOT_STAGGER_PROXY"]
    status: RuleStatus
    confidence: float
    stagger_ratio: float | None
    mean_foot_length: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "confidence": round(_clamp(self.confidence), 4),
            "stagger_ratio": self.stagger_ratio,
            "mean_foot_length": self.mean_foot_length,
        }


@dataclass
class _PendingStep:
    takeoff_ms: int
    takeoff_x: float | None
    landing_ms: int
    landing_x: float | None
    landing_frame: int
    airborne_ms: int


class _FootTracker:
    def __init__(
        self,
        side: Literal["left", "right"],
        config: FootSupportConfig,
        *,
        sensitivity: str,
    ) -> None:
        self.side = side
        self.config = config
        self.sensitivity = sensitivity
        self.reset()

    def reset(self) -> None:
        self.state: FootSupportState = "GROUNDED"
        self.initialized = False
        self.candidate_frames = 0
        self.candidate_ms: int | None = None
        self.candidate_frame: int | None = None
        self.candidate_x: float | None = None
        self.previous_height: float | None = None
        self.previous_ms: int | None = None
        self.takeoff_ms: int | None = None
        self.landing_ms: int | None = None
        self.takeoff_x: float | None = None
        self.pending_step: _PendingStep | None = None
        self.last_result = FootStateResult(
            self.side,
            self.state,
            False,
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def _transition_frames(self) -> int:
        return max(1, int(self.config.transition_frames[self.sensitivity]))

    def _start_candidate(
        self,
        state: FootSupportState,
        *,
        timestamp_ms: int,
        frame_index: int,
        foot_x: float | None,
    ) -> None:
        self.state = state
        self.candidate_frames = 1
        self.candidate_ms = timestamp_ms
        self.candidate_frame = frame_index
        self.candidate_x = foot_x

    def _clear_candidate(self) -> None:
        self.candidate_frames = 0
        self.candidate_ms = None
        self.candidate_frame = None
        self.candidate_x = None

    def _confirm_takeoff(
        self,
        *,
        timestamp_ms: int,
        frame_index: int,
    ) -> FootEvent:
        event_ms = self.candidate_ms if self.candidate_ms is not None else timestamp_ms
        event_frame = (
            self.candidate_frame if self.candidate_frame is not None else frame_index
        )
        event_x = self.candidate_x
        self.state = "AIRBORNE"
        self.takeoff_ms = event_ms
        self.takeoff_x = event_x
        self._clear_candidate()
        return FootEvent(
            "TAKEOFF",
            self.side,
            event_ms,
            event_frame,
            event_x,
        )

    def _confirm_landing(
        self,
        *,
        timestamp_ms: int,
        frame_index: int,
    ) -> FootEvent:
        event_ms = self.candidate_ms if self.candidate_ms is not None else timestamp_ms
        event_frame = (
            self.candidate_frame if self.candidate_frame is not None else frame_index
        )
        event_x = self.candidate_x
        airborne_ms = (
            None
            if self.takeoff_ms is None
            else max(0, event_ms - self.takeoff_ms)
        )
        self.state = "GROUNDED"
        self.landing_ms = event_ms
        if self.takeoff_ms is not None:
            self.pending_step = _PendingStep(
                takeoff_ms=self.takeoff_ms,
                takeoff_x=self.takeoff_x,
                landing_ms=event_ms,
                landing_x=event_x,
                landing_frame=event_frame,
                airborne_ms=max(0, event_ms - self.takeoff_ms),
            )
        self._clear_candidate()
        return FootEvent(
            "LANDING",
            self.side,
            event_ms,
            event_frame,
            event_x,
            airborne_ms=airborne_ms,
        )

    def update(
        self,
        *,
        heel_height: float,
        toe_height: float,
        confidence: float,
        foot_x: float | None,
        timestamp_ms: int,
        frame_index: int,
    ) -> tuple[FootStateResult, list[FootEvent]]:
        support_height = min(heel_height, toe_height)
        speed = None
        if self.previous_height is not None and self.previous_ms is not None:
            elapsed = timestamp_ms - self.previous_ms
            if elapsed > 0:
                speed = (support_height - self.previous_height) * 1000.0 / elapsed
        self.previous_height = support_height
        self.previous_ms = timestamp_ms

        grounded = support_height <= self.config.grounded_height_body_ratio
        airborne = support_height >= self.config.airborne_height_body_ratio
        moving_up = speed is None or speed >= self.config.min_vertical_speed_body_per_second
        moving_down = speed is None or speed <= -self.config.min_vertical_speed_body_per_second
        events: list[FootEvent] = []

        if not self.initialized:
            self.state = "AIRBORNE" if airborne else "GROUNDED"
            self.initialized = True
        elif self.state == "GROUNDED":
            if airborne and moving_up:
                self._start_candidate(
                    "TAKEOFF_CANDIDATE",
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    foot_x=foot_x,
                )
                if self._transition_frames() <= 1:
                    events.append(
                        self._confirm_takeoff(
                            timestamp_ms=timestamp_ms,
                            frame_index=frame_index,
                        )
                    )
        elif self.state == "TAKEOFF_CANDIDATE":
            if airborne:
                self.candidate_frames += 1
                if self.candidate_frames >= self._transition_frames():
                    events.append(
                        self._confirm_takeoff(
                            timestamp_ms=timestamp_ms,
                            frame_index=frame_index,
                        )
                    )
            else:
                self.state = "GROUNDED"
                self._clear_candidate()
        elif self.state == "AIRBORNE":
            if grounded and moving_down:
                self._start_candidate(
                    "LANDING_CANDIDATE",
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    foot_x=foot_x,
                )
                if self._transition_frames() <= 1:
                    events.append(
                        self._confirm_landing(
                            timestamp_ms=timestamp_ms,
                            frame_index=frame_index,
                        )
                    )
        elif self.state == "LANDING_CANDIDATE":
            if grounded:
                self.candidate_frames += 1
                if self.candidate_frames >= self._transition_frames():
                    events.append(
                        self._confirm_landing(
                            timestamp_ms=timestamp_ms,
                            frame_index=frame_index,
                        )
                    )
            else:
                self.state = "AIRBORNE"
                self._clear_candidate()

        result = FootStateResult(
            side=self.side,
            state=self.state,
            observable=True,
            confidence=confidence,
            heel_height_ratio=heel_height,
            toe_height_ratio=toe_height,
            support_height_ratio=support_height,
            vertical_speed_body_per_second=speed,
            takeoff_ms=self.takeoff_ms,
            landing_ms=self.landing_ms,
        )
        self.last_result = result
        return result, events

    def not_observable(self) -> FootStateResult:
        result = FootStateResult(
            side=self.side,
            state=self.state,
            observable=False,
            confidence=0.0,
            heel_height_ratio=None,
            toe_height_ratio=None,
            support_height_ratio=None,
            vertical_speed_body_per_second=None,
            takeoff_ms=self.takeoff_ms,
            landing_ms=self.landing_ms,
        )
        self.last_result = result
        return result


class FootEventDetectorSuite:
    """Track independent left/right foot support, synchronization and steps."""

    def __init__(
        self,
        *,
        sensitivity: str = "medium",
        support_config: FootSupportConfig | None = None,
        sync_config: FootSyncConfig | None = None,
        stagger_config: FootStaggerConfig | None = None,
        step_config: StepEventConfig | None = None,
    ) -> None:
        self.sensitivity = sensitivity if sensitivity in {"high", "medium", "low"} else "medium"
        self.support_config = support_config or FootSupportConfig()
        self.sync_config = sync_config or FootSyncConfig()
        self.stagger_config = stagger_config or FootStaggerConfig()
        self.step_config = step_config or StepEventConfig()
        self.trackers = {
            "left": _FootTracker(
                "left",
                self.support_config,
                sensitivity=self.sensitivity,
            ),
            "right": _FootTracker(
                "right",
                self.support_config,
                sensitivity=self.sensitivity,
            ),
        }
        self.reset()

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset()
        self._sync_pairs: dict[str, dict[str, int | None]] = {
            "takeoff": {"left": None, "right": None},
            "landing": {"left": None, "right": None},
        }
        self.event_history: deque[FootEvent] = deque(maxlen=60)
        self.new_events: list[FootEvent] = []
        self.step_event_count = 0
        self.last_stagger = FootStaggerResult(
            "FOOT_STAGGER_PROXY",
            "NOT_OBSERVABLE",
            0.0,
            None,
            None,
        )
        self.last_sync = FootSyncResult(
            "NOT_OBSERVABLE",
            None,
            "NOT_OBSERVABLE",
            None,
            None,
            None,
            None,
            None,
        )

    def _foot_geometry(
        self,
        features: Mapping[str, object],
        side: str,
        floor: FloorLine,
        body_height: float,
    ) -> tuple[float, float, float, float | None] | None:
        heel = _point(
            features,
            f"{side}_heel",
            self.support_config.min_landmark_confidence,
        )
        toe = _point(
            features,
            f"{side}_foot_index",
            self.support_config.min_landmark_confidence,
        )
        if heel is None or toe is None:
            return None
        heel_height = signed_distance_to_floor(heel, floor)
        toe_height = signed_distance_to_floor(toe, floor)
        if heel_height is None or toe_height is None:
            return None
        confidence = min(heel.visibility, toe.visibility)
        foot_x = (heel.x + toe.x) / 2.0
        return (
            heel_height / body_height,
            toe_height / body_height,
            confidence,
            foot_x,
        )

    def _register_sync_event(self, event: FootEvent) -> None:
        if event.event_type not in {"TAKEOFF", "LANDING"}:
            return
        kind = event.event_type.lower()
        pair = self._sync_pairs[kind]
        if pair["left"] is not None and pair["right"] is not None:
            pair["left"] = None
            pair["right"] = None
        pair[event.side] = event.timestamp_ms

    def _sync_status(
        self,
        kind: Literal["takeoff", "landing"],
        current_ms: int,
    ) -> tuple[RuleStatus, int | None]:
        pair = self._sync_pairs[kind]
        left = pair["left"]
        right = pair["right"]
        if left is None and right is None:
            return "NOT_OBSERVABLE", None
        if left is None or right is None:
            observed = left if left is not None else right
            assert observed is not None
            return (
                "FAIL"
                if current_ms - observed > self.sync_config.unsure_ms
                else "UNSURE",
                None,
            )
        delta = abs(left - right)
        if delta <= self.sync_config.pass_ms:
            return "PASS", delta
        if delta <= self.sync_config.unsure_ms:
            return "UNSURE", delta
        return "FAIL", delta

    def _sync_result(self, current_ms: int) -> FootSyncResult:
        takeoff_status, takeoff_delta = self._sync_status("takeoff", current_ms)
        landing_status, landing_delta = self._sync_status("landing", current_ms)
        takeoff = self._sync_pairs["takeoff"]
        landing = self._sync_pairs["landing"]
        return FootSyncResult(
            takeoff_status=takeoff_status,
            takeoff_delta_ms=takeoff_delta,
            landing_status=landing_status,
            landing_delta_ms=landing_delta,
            left_takeoff_ms=takeoff["left"],
            right_takeoff_ms=takeoff["right"],
            left_landing_ms=landing["left"],
            right_landing_ms=landing["right"],
        )

    def _stagger_result(
        self,
        features: Mapping[str, object],
        floor: FloorLine | None,
    ) -> FootStaggerResult:
        if floor is None:
            return FootStaggerResult(
                "FOOT_STAGGER_PROXY",
                "NOT_OBSERVABLE",
                0.0,
                None,
                None,
            )
        points = {
            name: _point(
                features,
                name,
                self.support_config.min_landmark_confidence,
            )
            for name in (
                "left_heel",
                "left_foot_index",
                "right_heel",
                "right_foot_index",
            )
        }
        if any(point is None for point in points.values()):
            return FootStaggerResult(
                "FOOT_STAGGER_PROXY",
                "NOT_OBSERVABLE",
                0.0,
                None,
                None,
            )
        left_heel = points["left_heel"]
        left_toe = points["left_foot_index"]
        right_heel = points["right_heel"]
        right_toe = points["right_foot_index"]
        assert left_heel and left_toe and right_heel and right_toe
        left_length = hypot(left_toe.x - left_heel.x, left_toe.y - left_heel.y)
        right_length = hypot(right_toe.x - right_heel.x, right_toe.y - right_heel.y)
        mean_length = (left_length + right_length) / 2.0
        if mean_length <= 1e-6:
            return FootStaggerResult(
                "FOOT_STAGGER_PROXY",
                "NOT_OBSERVABLE",
                0.0,
                None,
                None,
            )
        floor_dx = floor.point2.x - floor.point1.x
        floor_dy = floor.point2.y - floor.point1.y
        floor_length = hypot(floor_dx, floor_dy)
        unit_x = floor_dx / floor_length
        unit_y = floor_dy / floor_length
        left_forward = left_toe.x * unit_x + left_toe.y * unit_y
        right_forward = right_toe.x * unit_x + right_toe.y * unit_y
        ratio = abs(left_forward - right_forward) / mean_length
        if ratio <= self.stagger_config.pass_foot_length_ratio:
            status: RuleStatus = "PASS"
        elif ratio <= self.stagger_config.unsure_foot_length_ratio:
            status = "UNSURE"
        else:
            status = "FAIL"
        confidence = min(
            left_heel.visibility,
            left_toe.visibility,
            right_heel.visibility,
            right_toe.visibility,
        )
        return FootStaggerResult(
            "FOOT_STAGGER_PROXY",
            status,
            confidence,
            ratio,
            mean_length,
        )

    def _leg_length(self, features: Mapping[str, object], side: str) -> float | None:
        hip = _point(features, f"{side}_hip", 0.0)
        knee = _point(features, f"{side}_knee", 0.0)
        ankle = _point(features, f"{side}_ankle", 0.0)
        if hip is not None and knee is not None and ankle is not None:
            return hypot(hip.x - knee.x, hip.y - knee.y) + hypot(
                knee.x - ankle.x,
                knee.y - ankle.y,
            )
        body_height = _safe_float(features.get("body_height_reference"))
        return None if body_height is None else 0.53 * body_height

    def _confirm_pending_steps(
        self,
        features: Mapping[str, object],
        *,
        current_ms: int,
    ) -> list[FootEvent]:
        events: list[FootEvent] = []
        for side, tracker in self.trackers.items():
            pending = tracker.pending_step
            if pending is None:
                continue
            if tracker.state != "GROUNDED":
                tracker.pending_step = None
                continue
            if current_ms - pending.landing_ms < self.step_config.min_grounded_ms:
                continue
            tracker.pending_step = None
            if (
                pending.airborne_ms < self.step_config.min_airborne_ms
                or pending.takeoff_x is None
                or pending.landing_x is None
            ):
                continue
            leg_length = self._leg_length(features, side)
            if leg_length is None or leg_length <= 1e-6:
                continue
            displacement = pending.landing_x - pending.takeoff_x
            ratio = abs(displacement) / leg_length
            if ratio < self.step_config.min_horizontal_displacement_leg_ratio:
                continue
            events.append(
                FootEvent(
                    "STEP",
                    side,  # type: ignore[arg-type]
                    pending.landing_ms,
                    pending.landing_frame,
                    pending.landing_x,
                    airborne_ms=pending.airborne_ms,
                    horizontal_displacement_leg_ratio=ratio,
                    signed_horizontal_displacement=displacement,
                )
            )
        return events

    def update(
        self,
        features: Mapping[str, object],
        *,
        frame_index: int,
        timestamp_ms: int | None,
    ) -> dict[str, object]:
        current_ms = _timestamp(timestamp_ms, frame_index)
        floor = _floor_line(features)
        body_height = _safe_float(features.get("body_height_reference"))
        floor_ready = (
            features.get("floor_reference_status") == "READY"
            and floor is not None
            and body_height is not None
            and body_height > 1e-6
        )
        self.new_events = []
        results: dict[str, FootStateResult] = {}
        for side, tracker in self.trackers.items():
            geometry = (
                self._foot_geometry(features, side, floor, body_height)
                if floor_ready and floor is not None and body_height is not None
                else None
            )
            if geometry is None:
                results[side] = tracker.not_observable()
                continue
            heel_height, toe_height, confidence, foot_x = geometry
            result, events = tracker.update(
                heel_height=heel_height,
                toe_height=toe_height,
                confidence=confidence,
                foot_x=foot_x,
                timestamp_ms=current_ms,
                frame_index=frame_index,
            )
            results[side] = result
            for event in events:
                self._register_sync_event(event)
            self.new_events.extend(events)

        step_events = self._confirm_pending_steps(features, current_ms=current_ms)
        self.new_events.extend(step_events)
        self.step_event_count += len(step_events)
        self.event_history.extend(self.new_events)
        self.last_sync = self._sync_result(current_ms)
        self.last_stagger = self._stagger_result(features, floor if floor_ready else None)
        return {
            "left": results["left"],
            "right": results["right"],
            "sync": self.last_sync,
            "stagger": self.last_stagger,
            "new_events": tuple(self.new_events),
            "step_event_count": self.step_event_count,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "left": self.trackers["left"].last_result.as_dict(),
            "right": self.trackers["right"].last_result.as_dict(),
            "sync": self.last_sync.as_dict(),
            "stagger": self.last_stagger.as_dict(),
            "new_events": [event.as_dict() for event in self.new_events],
            "step_event_count": self.step_event_count,
            "recent_events": [
                event.as_dict()
                for event in list(self.event_history)[-12:]
            ],
        }


__all__ = [
    "FootEvent",
    "FootEventDetectorSuite",
    "FootEventType",
    "FootStaggerConfig",
    "FootStaggerResult",
    "FootStateResult",
    "FootSupportConfig",
    "FootSupportState",
    "FootSyncConfig",
    "FootSyncResult",
    "RuleStatus",
    "StepEventConfig",
]
