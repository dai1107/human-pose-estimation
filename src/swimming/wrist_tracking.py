"""Persistent anatomical wrist identity and short-gap LK tracking for swimming.

MediaPipe handedness is treated as a candidate label rather than identity
truth.  The tracker owns immutable anatomical left/right tracks, associates
the two wrist candidates with motion and arm-chain evidence, and uses a
confirmation margin before changing the semantic-label mapping.  Sparse LK
flow may bridge short pose gaps, but it never changes track identity.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from math import hypot, isfinite
from pathlib import Path
from statistics import median
from typing import Any, Literal

import cv2
import numpy as np

from src.configuration import ConfigValidationError, load_simple_yaml
from src.paths import installation_root


Side = Literal["left", "right"]
TrackState = Literal[
    "uninitialized",
    "tracked",
    "occluded",
    "lost",
    "reacquiring",
    "reacquired",
]

SIDES: tuple[Side, Side] = ("left", "right")
DEFAULT_SWIM_WRIST_TRACKING_CONFIG = (
    installation_root() / "configs" / "swim_wrist_tracking.yaml"
)


@dataclass(frozen=True, slots=True)
class SwimWristTrackerConfig:
    motion_weight: float = 0.40
    chain_weight: float = 0.35
    semantic_weight: float = 0.15
    smoothness_weight: float = 0.10
    minimum_pose_confidence: float = 0.15
    identity_change_margin: float = 0.15
    identity_confirmation_frames: int = 3
    reacquire_confirmation_frames: int = 2
    maximum_occlusion_frames: int = 5
    maximum_prediction_frames: int = 10
    kalman_process_noise: float = 0.015
    kalman_measurement_noise: float = 0.025
    outlier_history_size: int = 31
    outlier_minimum_history: int = 5
    outlier_mad_scale: float = 6.0
    outlier_minimum_speed_body_s: float = 2.0
    outlier_maximum_speed_body_s: float = 30.0
    lk_maximum_gap_frames: int = 5
    lk_maximum_forward_backward_error_px: float = 1.5
    lk_maximum_forward_backward_body_ratio: float = 0.04
    lk_minimum_displacement_body_ratio: float = 0.35
    lk_maximum_displacement_body_ratio: float = 1.50
    lk_minimum_history: int = 5
    lk_mad_scale: float = 6.0
    appearance_weight: float = 0.15
    appearance_roi_forearm_ratio: float = 0.25
    appearance_ema_alpha: float = 0.05
    appearance_minimum_identity_confidence: float = 0.75
    appearance_minimum_visibility: float = 0.65
    cotracker_maximum_width: int = 384
    cotracker_window_frames: int = 120
    cotracker_overlap_frames: int = 20
    cotracker_visibility_threshold: float = 0.50


def load_swim_wrist_tracker_config(
    path: str | Path = DEFAULT_SWIM_WRIST_TRACKING_CONFIG,
) -> SwimWristTrackerConfig:
    source = Path(path)
    payload = load_simple_yaml(source)
    section = payload.get("swim_wrist_tracking")
    if not isinstance(section, Mapping):
        raise ConfigValidationError(
            "swim_wrist_tracking must be a mapping",
            path=source,
            key="swim_wrist_tracking",
        )
    if str(section.get("mode", "experimental")) != "experimental":
        raise ConfigValidationError(
            "swimming wrist rounds 4-5 may only run in experimental mode",
            path=source,
            key="swim_wrist_tracking.mode",
        )
    defaults = SwimWristTrackerConfig()
    allowed = {descriptor.name for descriptor in fields(defaults)} | {"mode"}
    unknown = sorted(str(name) for name in section if str(name) not in allowed)
    if unknown:
        raise ConfigValidationError(
            f"unknown keys: {', '.join(unknown)}",
            path=source,
            key="swim_wrist_tracking",
        )
    values: dict[str, Any] = {}
    for descriptor in fields(defaults):
        name = descriptor.name
        default = getattr(defaults, name)
        raw = section.get(name, default)
        if isinstance(default, int):
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ConfigValidationError(
                    "must be a positive integer", path=source, key=name
                )
            values[name] = int(raw)
        else:
            try:
                resolved = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ConfigValidationError(
                    "must be numeric", path=source, key=name
                ) from exc
            if not isfinite(resolved) or resolved < 0.0:
                raise ConfigValidationError(
                    "must be finite and >= 0", path=source, key=name
                )
            values[name] = resolved
    config = SwimWristTrackerConfig(**values)
    weight_sum = (
        config.motion_weight
        + config.chain_weight
        + config.semantic_weight
        + config.smoothness_weight
    )
    if abs(weight_sum - 1.0) > 1e-6:
        raise ConfigValidationError(
            "association weights must sum to 1.0",
            path=source,
            key="swim_wrist_tracking",
        )
    if not 0.0 <= config.minimum_pose_confidence <= 1.0:
        raise ConfigValidationError(
            "must be between 0 and 1",
            path=source,
            key="minimum_pose_confidence",
        )
    bounded_fields = (
        "appearance_weight",
        "appearance_ema_alpha",
        "appearance_minimum_identity_confidence",
        "appearance_minimum_visibility",
        "cotracker_visibility_threshold",
    )
    for name in bounded_fields:
        if not 0.0 <= float(getattr(config, name)) <= 1.0:
            raise ConfigValidationError(
                "must be between 0 and 1",
                path=source,
                key=name,
            )
    if config.cotracker_overlap_frames >= config.cotracker_window_frames:
        raise ConfigValidationError(
            "must be smaller than cotracker_window_frames",
            path=source,
            key="cotracker_overlap_frames",
        )
    return config


@dataclass(frozen=True, slots=True)
class WristCandidate:
    semantic_side: Side
    position: tuple[float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class ArmChainObservation:
    shoulder: tuple[float, float] | None
    elbow: tuple[float, float] | None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class OpticalFlowObservation:
    side: Side
    position: tuple[float, float] | None
    confidence: float
    reliable: bool
    forward_backward_error_px: float | None
    displacement_body_ratio: float | None
    reason: str
    age_frames: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "position": list(self.position) if self.position is not None else None,
            "confidence": self.confidence,
            "reliable": self.reliable,
            "forward_backward_error_px": self.forward_backward_error_px,
            "displacement_body_ratio": self.displacement_body_ratio,
            "reason": self.reason,
            "age_frames": self.age_frames,
        }


@dataclass(frozen=True, slots=True)
class WristTrackSnapshot:
    side: Side
    track_id: str
    state: TrackState
    position: tuple[float, float] | None
    predicted_position: tuple[float, float] | None
    velocity: tuple[float, float]
    confidence: float
    source: str
    observed_semantic_side: Side | None
    missing_pose_frames: int
    reacquisition_count: int
    outlier_rejection_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "track_id": self.track_id,
            "state": self.state,
            "position": list(self.position) if self.position is not None else None,
            "predicted_position": (
                list(self.predicted_position)
                if self.predicted_position is not None
                else None
            ),
            "velocity": list(self.velocity),
            "confidence": self.confidence,
            "source": self.source,
            "observed_semantic_side": self.observed_semantic_side,
            "missing_pose_frames": self.missing_pose_frames,
            "reacquisition_count": self.reacquisition_count,
            "outlier_rejection_count": self.outlier_rejection_count,
        }


@dataclass(frozen=True, slots=True)
class SwimWristFrame:
    frame_index: int
    timestamp_ms: float
    tracks: Mapping[Side, WristTrackSnapshot]
    proposed_semantic_mapping_swapped: bool | None
    committed_semantic_mapping_swapped: bool
    mapping_change_pending_frames: int
    mapping_changed: bool
    direct_assignment_cost: float | None
    swapped_assignment_cost: float | None
    assignment_cost_matrix: tuple[tuple[float, float], ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "tracks": {
                side: self.tracks[side].as_dict() for side in SIDES
            },
            "proposed_semantic_mapping_swapped": (
                self.proposed_semantic_mapping_swapped
            ),
            "committed_semantic_mapping_swapped": (
                self.committed_semantic_mapping_swapped
            ),
            "mapping_change_pending_frames": self.mapping_change_pending_frames,
            "mapping_changed": self.mapping_changed,
            "direct_assignment_cost": self.direct_assignment_cost,
            "swapped_assignment_cost": self.swapped_assignment_cost,
            "assignment_cost_matrix": [list(row) for row in self.assignment_cost_matrix],
            "reason_codes": list(self.reason_codes),
            "persistent_track_ids": {
                "left": "swim_wrist_left",
                "right": "swim_wrist_right",
            },
            "persistent_track_identity_switch_count": 0,
        }


def hungarian_2x2(cost_matrix: Sequence[Sequence[float]]) -> tuple[int, int]:
    """Return the minimum-cost column for rows 0 and 1.

    This is the complete Hungarian solution for the fixed 2×2 wrist problem,
    avoiding a heavyweight SciPy runtime dependency.
    """

    matrix = np.asarray(cost_matrix, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError("wrist assignment cost matrix must be 2x2")
    direct = float(matrix[0, 0] + matrix[1, 1])
    swapped = float(matrix[0, 1] + matrix[1, 0])
    return (0, 1) if direct <= swapped else (1, 0)


class _ConstantVelocityKalman2D:
    def __init__(self, config: SwimWristTrackerConfig) -> None:
        self.config = config
        self.state: np.ndarray | None = None
        self.covariance = np.eye(4, dtype=np.float64)
        self.timestamp_ms: float | None = None

    @property
    def initialized(self) -> bool:
        return self.state is not None

    def predict(self, timestamp_ms: float) -> tuple[float, float] | None:
        if self.state is None:
            self.timestamp_ms = float(timestamp_ms)
            return None
        if self.timestamp_ms is None:
            self.timestamp_ms = float(timestamp_ms)
            return self.position
        dt = max(0.0, min(0.5, (float(timestamp_ms) - self.timestamp_ms) / 1000.0))
        transition = np.asarray(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        process = self.config.kalman_process_noise * max(dt, 1e-3)
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T
            + np.diag([process, process, process * 4.0, process * 4.0])
        )
        self.timestamp_ms = float(timestamp_ms)
        return self.position

    def correct(
        self,
        position: tuple[float, float],
        *,
        confidence: float,
        timestamp_ms: float,
    ) -> tuple[float, float]:
        measurement = np.asarray(position, dtype=np.float64)
        if self.state is None:
            self.state = np.asarray(
                [measurement[0], measurement[1], 0.0, 0.0], dtype=np.float64
            )
            self.covariance = np.eye(4, dtype=np.float64) * 0.1
            self.timestamp_ms = float(timestamp_ms)
            return self.position
        if self.timestamp_ms != float(timestamp_ms):
            self.predict(timestamp_ms)
        observation = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        noise = self.config.kalman_measurement_noise / max(0.05, confidence)
        innovation = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T
            + np.eye(2, dtype=np.float64) * noise
        )
        gain = (
            self.covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + gain @ innovation
        self.covariance = (
            np.eye(4, dtype=np.float64) - gain @ observation
        ) @ self.covariance
        return self.position

    @property
    def position(self) -> tuple[float, float]:
        assert self.state is not None
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        if self.state is None:
            return 0.0, 0.0
        return float(self.state[2]), float(self.state[3])


class _RobustTrajectoryGate:
    def __init__(self, config: SwimWristTrackerConfig) -> None:
        self.config = config
        self._last: tuple[np.ndarray, float] | None = None
        self._speeds: deque[float] = deque(maxlen=config.outlier_history_size)

    def accept(
        self,
        position: tuple[float, float],
        *,
        timestamp_ms: float,
        body_scale: float,
    ) -> tuple[bool, float | None, float | None]:
        current = np.asarray(position, dtype=np.float64)
        if self._last is None:
            self._last = current, float(timestamp_ms)
            return True, None, None
        previous, previous_timestamp = self._last
        dt = (float(timestamp_ms) - previous_timestamp) / 1000.0
        if dt <= 0.0 or dt > 1.0:
            self._last = current, float(timestamp_ms)
            self._speeds.clear()
            return True, None, None
        speed = float(np.linalg.norm(current - previous)) / max(body_scale, 1e-4) / dt
        limit = self.config.outlier_maximum_speed_body_s
        if len(self._speeds) >= self.config.outlier_minimum_history:
            center = median(self._speeds)
            mad = median(abs(value - center) for value in self._speeds)
            robust = center + self.config.outlier_mad_scale * 1.4826 * mad
            limit = min(
                self.config.outlier_maximum_speed_body_s,
                max(self.config.outlier_minimum_speed_body_s, robust),
            )
        accepted = speed <= limit
        if accepted:
            self._last = current, float(timestamp_ms)
            self._speeds.append(speed)
        return accepted, speed, limit


class _WristTrack:
    def __init__(self, side: Side, config: SwimWristTrackerConfig) -> None:
        self.side = side
        self.track_id = f"swim_wrist_{side}"
        self.config = config
        self.kalman = _ConstantVelocityKalman2D(config)
        self.gate = _RobustTrajectoryGate(config)
        self.state: TrackState = "uninitialized"
        self.predicted_position: tuple[float, float] | None = None
        self.position: tuple[float, float] | None = None
        self.confidence = 0.0
        self.source = "none"
        self.observed_semantic_side: Side | None = None
        self.missing_pose_frames = 0
        self.reacquire_frames = 0
        self.reacquisition_count = 0
        self.outlier_rejection_count = 0
        self.forearm_lengths: deque[float] = deque(maxlen=60)
        self.forearm_directions: deque[np.ndarray] = deque(maxlen=15)

    def begin_frame(self, timestamp_ms: float) -> None:
        self.predicted_position = self.kalman.predict(timestamp_ms)
        if self.state == "reacquired":
            self.state = "tracked"

    def chain_cost(
        self,
        candidate: WristCandidate,
        chain: ArmChainObservation | None,
        *,
        body_scale: float,
    ) -> float:
        if chain is None or chain.shoulder is None or chain.elbow is None:
            return 1.0
        shoulder = np.asarray(chain.shoulder, dtype=np.float64)
        elbow = np.asarray(chain.elbow, dtype=np.float64)
        wrist = np.asarray(candidate.position, dtype=np.float64)
        upper = float(np.linalg.norm(elbow - shoulder)) / max(body_scale, 1e-4)
        forearm = float(np.linalg.norm(wrist - elbow)) / max(body_scale, 1e-4)
        if upper <= 1e-5 or forearm <= 1e-5:
            return 2.0
        ratio = forearm / upper
        plausibility = min(2.0, abs(np.log(max(ratio, 1e-4))))
        expected_cost = 0.0
        if self.forearm_lengths:
            expected = median(self.forearm_lengths)
            expected_cost = min(2.0, abs(forearm - expected) / max(expected, 0.05))
        direction_cost = 0.0
        direction = wrist - elbow
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        if self.forearm_directions:
            expected_direction = np.mean(tuple(self.forearm_directions), axis=0)
            expected_direction /= max(float(np.linalg.norm(expected_direction)), 1e-6)
            direction_cost = 0.5 * (
                1.0 - float(np.clip(np.dot(direction, expected_direction), -1.0, 1.0))
            )
        confidence_penalty = 1.0 - max(0.0, min(1.0, chain.confidence))
        return float(
            np.clip(
                0.35 * plausibility
                + 0.35 * expected_cost
                + 0.20 * direction_cost
                + 0.10 * confidence_penalty,
                0.0,
                2.0,
            )
        )

    def association_cost(
        self,
        candidate: WristCandidate,
        chain: ArmChainObservation | None,
        *,
        body_scale: float,
        dt_seconds: float,
    ) -> float:
        if self.predicted_position is None:
            motion = 0.5
        else:
            motion = min(
                2.5,
                _distance(candidate.position, self.predicted_position)
                / max(body_scale, 1e-4),
            )
        chain_value = self.chain_cost(candidate, chain, body_scale=body_scale)
        semantic = 0.0 if candidate.semantic_side == self.side else 1.0
        smoothness = 0.5
        if self.position is not None and dt_seconds > 1e-5:
            implied_velocity = (
                (candidate.position[0] - self.position[0]) / dt_seconds,
                (candidate.position[1] - self.position[1]) / dt_seconds,
            )
            expected_velocity = self.kalman.velocity
            smoothness = min(
                2.5,
                _distance(implied_velocity, expected_velocity)
                * dt_seconds
                / max(body_scale, 1e-4),
            )
        return (
            self.config.motion_weight * motion
            + self.config.chain_weight * chain_value
            + self.config.semantic_weight * semantic
            + self.config.smoothness_weight * smoothness
        )

    def apply_pose(
        self,
        candidate: WristCandidate,
        chain: ArmChainObservation | None,
        *,
        timestamp_ms: float,
        body_scale: float,
    ) -> tuple[bool, str]:
        accepted, _speed, _limit = self.gate.accept(
            candidate.position,
            timestamp_ms=timestamp_ms,
            body_scale=body_scale,
        )
        if not accepted:
            self.outlier_rejection_count += 1
            return False, f"{self.side}_trajectory_outlier"
        corrected = self.kalman.correct(
            candidate.position,
            confidence=candidate.confidence,
            timestamp_ms=timestamp_ms,
        )
        was_lost = self.state in {"lost", "reacquiring"}
        self.position = corrected
        self.predicted_position = corrected
        self.confidence = candidate.confidence
        self.source = "pose"
        self.observed_semantic_side = candidate.semantic_side
        self.missing_pose_frames = 0
        if was_lost:
            self.reacquire_frames += 1
            if self.reacquire_frames >= self.config.reacquire_confirmation_frames:
                self.state = "reacquired"
                self.reacquisition_count += 1
                self.reacquire_frames = 0
            else:
                self.state = "reacquiring"
        else:
            self.state = "tracked"
            self.reacquire_frames = 0
        self._update_chain(candidate, chain, body_scale=body_scale)
        return True, "accepted"

    def apply_flow(
        self,
        flow: OpticalFlowObservation,
        *,
        timestamp_ms: float,
        body_scale: float,
    ) -> tuple[bool, str]:
        if not flow.reliable or flow.position is None:
            return False, flow.reason
        if flow.age_frames > self.config.lk_maximum_gap_frames:
            return False, "optical_flow_gap_exceeded"
        accepted, _speed, _limit = self.gate.accept(
            flow.position,
            timestamp_ms=timestamp_ms,
            body_scale=body_scale,
        )
        if not accepted:
            self.outlier_rejection_count += 1
            return False, f"{self.side}_optical_flow_trajectory_outlier"
        corrected = self.kalman.correct(
            flow.position,
            confidence=max(0.05, min(0.45, flow.confidence)),
            timestamp_ms=timestamp_ms,
        )
        self.missing_pose_frames += 1
        self.position = corrected
        self.predicted_position = corrected
        self.confidence = flow.confidence
        self.source = "optical_flow"
        self.observed_semantic_side = None
        self.state = "occluded"
        return True, "accepted"

    def apply_missing(self) -> None:
        self.missing_pose_frames += 1
        self.confidence = 0.0
        self.observed_semantic_side = None
        if (
            self.kalman.initialized
            and self.missing_pose_frames <= self.config.maximum_prediction_frames
        ):
            self.position = self.predicted_position
            self.source = "prediction"
        else:
            self.position = None
            self.source = "none"
        if self.missing_pose_frames <= self.config.maximum_occlusion_frames:
            self.state = "occluded" if self.kalman.initialized else "uninitialized"
        else:
            self.state = "lost"
            self.position = None
            self.source = "none"
            self.reacquire_frames = 0

    def snapshot(self) -> WristTrackSnapshot:
        return WristTrackSnapshot(
            side=self.side,
            track_id=self.track_id,
            state=self.state,
            position=self.position,
            predicted_position=self.predicted_position,
            velocity=self.kalman.velocity,
            confidence=self.confidence,
            source=self.source,
            observed_semantic_side=self.observed_semantic_side,
            missing_pose_frames=self.missing_pose_frames,
            reacquisition_count=self.reacquisition_count,
            outlier_rejection_count=self.outlier_rejection_count,
        )

    def _update_chain(
        self,
        candidate: WristCandidate,
        chain: ArmChainObservation | None,
        *,
        body_scale: float,
    ) -> None:
        if chain is None or chain.elbow is None:
            return
        elbow = np.asarray(chain.elbow, dtype=np.float64)
        wrist = np.asarray(candidate.position, dtype=np.float64)
        vector = wrist - elbow
        length = float(np.linalg.norm(vector)) / max(body_scale, 1e-4)
        if length <= 1e-5:
            return
        self.forearm_lengths.append(length)
        self.forearm_directions.append(vector / float(np.linalg.norm(vector)))


class SwimWristIdentityTracker:
    """Own immutable anatomical L/R tracks and associate pose candidates."""

    def __init__(self, config: SwimWristTrackerConfig | None = None) -> None:
        self.config = config or SwimWristTrackerConfig()
        self._tracks = {
            side: _WristTrack(side, self.config) for side in SIDES
        }
        self._committed_swapped = False
        self._pending_swapped: bool | None = None
        self._pending_frames = 0
        self.confirmed_mapping_change_count = 0
        self.instantaneous_mapping_flip_count = 0
        self._last_proposed_swapped: bool | None = None
        self._last_timestamp_ms: float | None = None

    def update(
        self,
        candidates: Sequence[WristCandidate],
        chains: Mapping[Side, ArmChainObservation],
        *,
        frame_index: int,
        timestamp_ms: float,
        body_scale: float,
        optical_flow: Mapping[Side, OpticalFlowObservation] | None = None,
        appearance_costs: Mapping[tuple[Side, Side], float] | None = None,
    ) -> SwimWristFrame:
        timestamp = float(timestamp_ms)
        dt_seconds = (
            max(1e-3, min(0.5, (timestamp - self._last_timestamp_ms) / 1000.0))
            if self._last_timestamp_ms is not None
            else 1.0 / 30.0
        )
        self._last_timestamp_ms = timestamp
        resolved_scale = max(0.01, float(body_scale))
        for track in self._tracks.values():
            track.begin_frame(timestamp)
        usable = [
            candidate
            for candidate in candidates
            if _finite_position(candidate.position)
            and candidate.confidence >= self.config.minimum_pose_confidence
        ]
        costs = np.empty((len(usable), 2), dtype=np.float64)
        for row, candidate in enumerate(usable):
            for column, side in enumerate(SIDES):
                costs[row, column] = self._tracks[side].association_cost(
                    candidate,
                    chains.get(side),
                    body_scale=resolved_scale,
                    dt_seconds=dt_seconds,
                )
                if appearance_costs is not None:
                    appearance = appearance_costs.get(
                        (candidate.semantic_side, side)
                    )
                    if appearance is not None and isfinite(float(appearance)):
                        weight = max(0.0, min(0.5, self.config.appearance_weight))
                        costs[row, column] = (
                            (1.0 - weight) * costs[row, column]
                            + weight * max(0.0, min(2.0, float(appearance)))
                        )
        assignments: dict[Side, WristCandidate] = {}
        reasons: set[str] = set()
        proposed_swapped: bool | None = None
        direct_cost: float | None = None
        swapped_cost: float | None = None
        mapping_changed = False
        semantic_indices = {
            candidate.semantic_side: index
            for index, candidate in enumerate(usable)
        }
        if len(usable) == 2 and set(semantic_indices) == set(SIDES):
            left_index = semantic_indices["left"]
            right_index = semantic_indices["right"]
            direct_cost = float(costs[left_index, 0] + costs[right_index, 1])
            swapped_cost = float(costs[left_index, 1] + costs[right_index, 0])
            row_assignment = hungarian_2x2(costs)
            proposed_swapped = row_assignment[left_index] == 1
            if (
                self._last_proposed_swapped is not None
                and proposed_swapped != self._last_proposed_swapped
            ):
                self.instantaneous_mapping_flip_count += 1
            self._last_proposed_swapped = proposed_swapped
            current_cost = swapped_cost if self._committed_swapped else direct_cost
            proposed_cost = swapped_cost if proposed_swapped else direct_cost
            improvement = current_cost - proposed_cost
            if proposed_swapped != self._committed_swapped:
                reliable = min(candidate.confidence for candidate in usable) >= max(
                    0.35, self.config.minimum_pose_confidence
                )
                if reliable and improvement >= self.config.identity_change_margin:
                    if self._pending_swapped == proposed_swapped:
                        self._pending_frames += 1
                    else:
                        self._pending_swapped = proposed_swapped
                        self._pending_frames = 1
                    reasons.add("IDENTITY_MAPPING_CHANGE_PENDING")
                    if self._pending_frames >= self.config.identity_confirmation_frames:
                        self._committed_swapped = proposed_swapped
                        self._pending_swapped = None
                        self._pending_frames = 0
                        self.confirmed_mapping_change_count += 1
                        mapping_changed = True
                        reasons.add("IDENTITY_MAPPING_CHANGE_CONFIRMED")
                        assignments = self._mapping_assignments(usable, semantic_indices)
                    else:
                        reasons.add("IDENTITY_HYSTERESIS_HOLD")
                else:
                    self._clear_pending()
                    assignments = self._mapping_assignments(usable, semantic_indices)
                    reasons.add("IDENTITY_CHANGE_MARGIN_NOT_MET")
            else:
                self._clear_pending()
                assignments = self._mapping_assignments(usable, semantic_indices)
        elif len(usable) == 1:
            self._clear_pending()
            candidate = usable[0]
            left_cost = float(costs[0, 0])
            right_cost = float(costs[0, 1])
            if abs(left_cost - right_cost) >= self.config.identity_change_margin:
                side = "left" if left_cost < right_cost else "right"
                assignments[side] = candidate
            else:
                reasons.add("SINGLE_WRIST_ASSIGNMENT_UNSURE")
        else:
            self._clear_pending()
            if not usable:
                reasons.add("WRISTS_MISSING")

        flow_map = optical_flow or {}
        for side in SIDES:
            track = self._tracks[side]
            candidate = assignments.get(side)
            accepted = False
            if candidate is not None:
                accepted, reason = track.apply_pose(
                    candidate,
                    chains.get(side),
                    timestamp_ms=timestamp,
                    body_scale=resolved_scale,
                )
                if not accepted:
                    reasons.add(reason)
            if not accepted:
                flow = flow_map.get(side)
                if flow is not None:
                    accepted, reason = track.apply_flow(
                        flow,
                        timestamp_ms=timestamp,
                        body_scale=resolved_scale,
                    )
                    if not accepted and reason not in {"not_initialized", "warming_up"}:
                        reasons.add(f"{side}_{reason}")
            if not accepted:
                track.apply_missing()

        matrix = tuple(tuple(float(value) for value in row) for row in costs)
        return SwimWristFrame(
            frame_index=int(frame_index),
            timestamp_ms=timestamp,
            tracks={side: self._tracks[side].snapshot() for side in SIDES},
            proposed_semantic_mapping_swapped=proposed_swapped,
            committed_semantic_mapping_swapped=self._committed_swapped,
            mapping_change_pending_frames=self._pending_frames,
            mapping_changed=mapping_changed,
            direct_assignment_cost=direct_cost,
            swapped_assignment_cost=swapped_cost,
            assignment_cost_matrix=matrix,
            reason_codes=tuple(sorted(reasons)),
        )

    def _mapping_assignments(
        self,
        candidates: Sequence[WristCandidate],
        semantic_indices: Mapping[Side, int],
    ) -> dict[Side, WristCandidate]:
        if self._committed_swapped:
            return {
                "left": candidates[semantic_indices["right"]],
                "right": candidates[semantic_indices["left"]],
            }
        return {
            "left": candidates[semantic_indices["left"]],
            "right": candidates[semantic_indices["right"]],
        }

    def _clear_pending(self) -> None:
        self._pending_swapped = None
        self._pending_frames = 0


class LKOpticalFlowWristTracker:
    """Sparse LK short-gap compensation with forward/backward validation."""

    def __init__(self, config: SwimWristTrackerConfig | None = None) -> None:
        self.config = config or SwimWristTrackerConfig()
        self._previous_gray: np.ndarray | None = None
        self._points: dict[Side, np.ndarray] = {}
        self._ages: dict[Side, int] = {side: 0 for side in SIDES}
        self._displacements: dict[Side, deque[float]] = {
            side: deque(maxlen=31) for side in SIDES
        }

    def reset(self) -> None:
        self._previous_gray = None
        self._points.clear()
        self._ages = {side: 0 for side in SIDES}
        for history in self._displacements.values():
            history.clear()

    def advance(
        self,
        frame: np.ndarray,
        *,
        body_scale_px: float,
    ) -> dict[Side, OpticalFlowObservation]:
        gray = _gray(frame)
        if gray is None:
            return {
                side: self._unavailable(side, "invalid_frame") for side in SIDES
            }
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            self._previous_gray = gray
            return {
                side: self._unavailable(side, "warming_up") for side in SIDES
            }
        results: dict[Side, OpticalFlowObservation] = {}
        for side in SIDES:
            previous = self._points.get(side)
            if previous is None:
                results[side] = self._unavailable(side, "not_initialized")
                continue
            if self._ages[side] >= self.config.lk_maximum_gap_frames:
                self._points.pop(side, None)
                results[side] = self._unavailable(
                    side, "optical_flow_gap_exceeded"
                )
                continue
            start = previous.reshape(1, 1, 2).astype(np.float32)
            current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                start,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            if (
                current is None
                or forward_status is None
                or not bool(forward_status.reshape(-1)[0])
            ):
                self._points.pop(side, None)
                results[side] = self._unavailable(side, "forward_flow_failed")
                continue
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self._previous_gray,
                current,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            if (
                backward is None
                or backward_status is None
                or not bool(backward_status.reshape(-1)[0])
            ):
                self._points.pop(side, None)
                results[side] = self._unavailable(side, "backward_flow_failed")
                continue
            current_point = current.reshape(2).astype(np.float64)
            backward_point = backward.reshape(2).astype(np.float64)
            fb_error = float(np.linalg.norm(backward_point - previous))
            scale = max(1.0, float(body_scale_px))
            fb_limit = max(
                self.config.lk_maximum_forward_backward_error_px,
                self.config.lk_maximum_forward_backward_body_ratio * scale,
            )
            displacement_ratio = float(np.linalg.norm(current_point - previous)) / scale
            displacement_limit = self._displacement_limit(side)
            if fb_error > fb_limit:
                self._points.pop(side, None)
                results[side] = OpticalFlowObservation(
                    side,
                    None,
                    0.0,
                    False,
                    fb_error,
                    displacement_ratio,
                    "forward_backward_failure",
                    self._ages[side] + 1,
                )
                continue
            if displacement_ratio > displacement_limit:
                self._points.pop(side, None)
                results[side] = OpticalFlowObservation(
                    side,
                    None,
                    0.0,
                    False,
                    fb_error,
                    displacement_ratio,
                    "optical_flow_displacement_outlier",
                    self._ages[side] + 1,
                )
                continue
            self._points[side] = current_point
            self._ages[side] += 1
            self._displacements[side].append(displacement_ratio)
            confidence = max(0.05, min(0.45, 0.45 * (1.0 - fb_error / max(fb_limit, 1e-6))))
            results[side] = OpticalFlowObservation(
                side,
                (float(current_point[0] / gray.shape[1]), float(current_point[1] / gray.shape[0])),
                confidence,
                True,
                fb_error,
                displacement_ratio,
                "accepted",
                self._ages[side],
            )
        self._previous_gray = gray
        return results

    def reanchor(
        self,
        side: Side,
        position_normalized: tuple[float, float],
        *,
        frame_shape: Sequence[int],
    ) -> None:
        if not _finite_position(position_normalized) or len(frame_shape) < 2:
            return
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if width <= 0 or height <= 0:
            return
        self._points[side] = np.asarray(
            [position_normalized[0] * width, position_normalized[1] * height],
            dtype=np.float64,
        )
        self._ages[side] = 0

    def _displacement_limit(self, side: Side) -> float:
        history = self._displacements[side]
        if len(history) < self.config.lk_minimum_history:
            return self.config.lk_maximum_displacement_body_ratio
        center = median(history)
        mad = median(abs(value - center) for value in history)
        robust = center + self.config.lk_mad_scale * 1.4826 * mad
        return min(
            self.config.lk_maximum_displacement_body_ratio,
            max(self.config.lk_minimum_displacement_body_ratio, robust),
        )

    def _unavailable(self, side: Side, reason: str) -> OpticalFlowObservation:
        return OpticalFlowObservation(
            side=side,
            position=None,
            confidence=0.0,
            reliable=False,
            forward_backward_error_px=None,
            displacement_body_ratio=None,
            reason=reason,
            age_frames=self._ages[side],
        )


def _gray(frame: np.ndarray) -> np.ndarray | None:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    return None


def _finite_position(position: Sequence[float]) -> bool:
    return len(position) >= 2 and isfinite(float(position[0])) and isfinite(float(position[1]))


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


__all__ = [
    "ArmChainObservation",
    "LKOpticalFlowWristTracker",
    "OpticalFlowObservation",
    "SIDES",
    "SwimWristFrame",
    "SwimWristIdentityTracker",
    "SwimWristTrackerConfig",
    "WristCandidate",
    "WristTrackSnapshot",
    "hungarian_2x2",
    "load_swim_wrist_tracker_config",
]
