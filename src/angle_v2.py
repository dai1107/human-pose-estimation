"""Shadow-only HYROX Angle V2 primitives.

The module deliberately does not connect itself to formal action analyzers.
It consumes observed angle/quality evidence, exposes causal diagnostics and
supports offline parameter calibration without replacing the current 2D
rules.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from statistics import median
from types import MappingProxyType
from typing import Any, Literal

from src.configuration import ConfigValidationError, load_simple_yaml
from src.paths import installation_root
from src.utils.smoothing import OneEuroValueFilter


DEFAULT_ANGLE_V2_CONFIG = installation_root() / "configs" / "angle_v2_shadow.yaml"
EvidenceState = Literal["PASS", "FAIL", "UNSURE"]
ThresholdDirection = Literal["above", "below"]

PASS: EvidenceState = "PASS"
FAIL: EvidenceState = "FAIL"
UNSURE: EvidenceState = "UNSURE"

BONE_LENGTH_REASONS = frozenset(
    {
        "invalid_bone_length",
        "bone_length_jump",
        "left_right_bone_mismatch",
    }
)


@dataclass(frozen=True, slots=True)
class JointAngleSmoothingConfig:
    min_cutoff: float
    beta: float
    d_cutoff: float = 1.0


def _default_joint_smoothing() -> Mapping[str, JointAngleSmoothingConfig]:
    return MappingProxyType(
        {
            "knee": JointAngleSmoothingConfig(1.6, 0.10),
            "hip": JointAngleSmoothingConfig(1.3, 0.06),
            "elbow": JointAngleSmoothingConfig(2.0, 0.15),
            "shoulder": JointAngleSmoothingConfig(1.3, 0.06),
            "ankle": JointAngleSmoothingConfig(1.0, 0.03),
            "wrist": JointAngleSmoothingConfig(2.0, 0.15),
            "torso": JointAngleSmoothingConfig(1.3, 0.06),
            "default": JointAngleSmoothingConfig(1.6, 0.08),
        }
    )


def _default_conflict_thresholds() -> Mapping[str, float]:
    return MappingProxyType(
        {
            "knee": 25.0,
            "hip": 30.0,
            "elbow": 35.0,
            "shoulder": 35.0,
            "ankle": 30.0,
            "wrist": 40.0,
            "torso": 35.0,
            "default": 30.0,
        }
    )


@dataclass(frozen=True, slots=True)
class AngleV2Config:
    enabled: bool = True
    mode: str = "shadow"
    joint_smoothing: Mapping[str, JointAngleSmoothingConfig] = field(
        default_factory=_default_joint_smoothing
    )
    minimum_confidence: float = 0.50
    bone_length_deviation_ratio: float = 0.20
    temporal_history_size: int = 31
    temporal_minimum_history: int = 5
    temporal_mad_scale: float = 6.0
    temporal_minimum_velocity_limit_deg_s: float = 90.0
    temporal_maximum_velocity_deg_s: float = 720.0
    maximum_gap_ms_before_reset: float = 250.0
    endpoint_radius_frames: int = 2
    endpoint_minimum_confidence: float = 0.60
    endpoint_minimum_prominence_deg: float = 1.0
    evidence_window_size: int = 5
    evidence_pass_count: int = 3
    evidence_fail_count: int = 3
    evidence_minimum_hold_ms: float = 100.0
    hysteresis_width_deg: float = 3.0
    conflict_thresholds_deg: Mapping[str, float] = field(
        default_factory=_default_conflict_thresholds
    )

    def smoothing_for(self, joint: str) -> JointAngleSmoothingConfig:
        group = joint_group(joint)
        return self.joint_smoothing.get(
            group,
            self.joint_smoothing.get(
                "default", JointAngleSmoothingConfig(1.6, 0.08)
            ),
        )

    def conflict_threshold_for(self, joint: str) -> float:
        group = joint_group(joint)
        return float(
            self.conflict_thresholds_deg.get(
                group,
                self.conflict_thresholds_deg.get("default", 30.0),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "formal_rule_integration": False,
            "joint_smoothing": {
                name: {
                    "min_cutoff": value.min_cutoff,
                    "beta": value.beta,
                    "d_cutoff": value.d_cutoff,
                }
                for name, value in self.joint_smoothing.items()
            },
            "minimum_confidence": self.minimum_confidence,
            "bone_length_deviation_ratio": self.bone_length_deviation_ratio,
            "temporal_outlier": {
                "history_size": self.temporal_history_size,
                "minimum_history": self.temporal_minimum_history,
                "mad_scale": self.temporal_mad_scale,
                "minimum_velocity_limit_deg_s": (
                    self.temporal_minimum_velocity_limit_deg_s
                ),
                "maximum_velocity_deg_s": self.temporal_maximum_velocity_deg_s,
                "maximum_gap_ms_before_reset": self.maximum_gap_ms_before_reset,
            },
            "endpoint_preservation": {
                "radius_frames": self.endpoint_radius_frames,
                "minimum_confidence": self.endpoint_minimum_confidence,
                "minimum_prominence_deg": self.endpoint_minimum_prominence_deg,
            },
            "temporal_evidence": {
                "window_size": self.evidence_window_size,
                "pass_count": self.evidence_pass_count,
                "fail_count": self.evidence_fail_count,
                "minimum_hold_ms": self.evidence_minimum_hold_ms,
            },
            "hysteresis_width_deg": self.hysteresis_width_deg,
            "2d_3d_conflict_thresholds_deg": dict(
                self.conflict_thresholds_deg
            ),
        }


@dataclass(frozen=True, slots=True)
class EndpointEvent:
    joint: str
    kind: Literal["minimum", "maximum"]
    frame_index: int
    timestamp_ms: float
    raw_extremum_angle_deg: float
    filtered_center_angle_deg: float
    confidence: float
    confirmed_at_frame: int
    window_start_frame: int
    window_end_frame: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "kind": self.kind,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "raw_extremum_angle_deg": self.raw_extremum_angle_deg,
            "filtered_center_angle_deg": self.filtered_center_angle_deg,
            "endpoint_preservation_delta_deg": (
                self.raw_extremum_angle_deg - self.filtered_center_angle_deg
            ),
            "confidence": self.confidence,
            "confirmed_at_frame": self.confirmed_at_frame,
            "confirmation_delay_frames": self.confirmed_at_frame - self.frame_index,
            "window_start_frame": self.window_start_frame,
            "window_end_frame": self.window_end_frame,
            "formal_rule_applied": False,
        }


@dataclass(frozen=True, slots=True)
class AngleV2Frame:
    joint: str
    frame_index: int
    timestamp_ms: float
    raw_2d_angle_deg: float | None
    filtered_2d_angle_deg: float | None
    raw_3d_angle_deg: float | None
    confidence: float
    angle_valid: bool
    evidence_valid: bool
    bone_length_valid: bool
    temporal_outlier: bool
    two_d_three_d_conflict: bool
    disagreement_deg: float | None
    reason_codes: tuple[str, ...]
    endpoints: tuple[EndpointEvent, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint,
            "joint_group": joint_group(self.joint),
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "raw_2d_angle_deg": self.raw_2d_angle_deg,
            "filtered_2d_angle_deg": self.filtered_2d_angle_deg,
            "raw_3d_angle_deg": self.raw_3d_angle_deg,
            "confidence": self.confidence,
            "angle_valid": self.angle_valid,
            "evidence_valid": self.evidence_valid,
            "bone_length_valid": self.bone_length_valid,
            "temporal_outlier": self.temporal_outlier,
            "two_d_three_d_conflict": self.two_d_three_d_conflict,
            "disagreement_deg": self.disagreement_deg,
            "reason_codes": list(self.reason_codes),
            "endpoints": [item.as_dict() for item in self.endpoints],
            "shadow_only": True,
        }


class RobustTemporalAngleGate:
    """Reject implausible causal angular velocity using median/MAD history."""

    def __init__(self, config: AngleV2Config) -> None:
        self.config = config
        self._samples: deque[tuple[float, float]] = deque(
            maxlen=config.temporal_history_size
        )
        self._velocities: deque[float] = deque(
            maxlen=config.temporal_history_size
        )

    def reset(self) -> None:
        self._samples.clear()
        self._velocities.clear()

    def observe(self, value: float, timestamp_ms: float) -> tuple[bool, float | None]:
        if not isfinite(value) or not isfinite(timestamp_ms):
            return False, None
        if not self._samples:
            self._samples.append((timestamp_ms, value))
            return True, None
        previous_timestamp, previous_value = self._samples[-1]
        delta_ms = timestamp_ms - previous_timestamp
        if delta_ms <= 0.0 or delta_ms > self.config.maximum_gap_ms_before_reset:
            self.reset()
            self._samples.append((timestamp_ms, value))
            return True, None
        velocity = abs(value - previous_value) * 1000.0 / delta_ms
        limit = self.config.temporal_maximum_velocity_deg_s
        if len(self._velocities) >= self.config.temporal_minimum_history:
            center = median(self._velocities)
            mad = median(abs(item - center) for item in self._velocities)
            robust_limit = center + self.config.temporal_mad_scale * 1.4826 * mad
            limit = min(
                self.config.temporal_maximum_velocity_deg_s,
                max(
                    self.config.temporal_minimum_velocity_limit_deg_s,
                    robust_limit,
                ),
            )
        accepted = velocity <= limit
        if accepted:
            self._samples.append((timestamp_ms, value))
            self._velocities.append(velocity)
        return accepted, limit


class EndpointPreservingAngleFilter:
    """Confirm local extrema after a small causal look-ahead window.

    The middle sample may sit on a short (at most three-frame) plateau.  An
    endpoint is emitted only when that central plateau is prominent relative
    to both sides of the complete window.  This avoids missing held extension
    or bottom positions while still rejecting a monotonic slope.
    """

    def __init__(self, joint: str, config: AngleV2Config) -> None:
        self.joint = joint
        self.config = config
        self._window: deque[tuple[int, float, float, float, float]] = deque(
            maxlen=2 * config.endpoint_radius_frames + 1
        )
        self._last_frame: int | None = None
        self._last_event: dict[str, int] = {}

    def reset(self) -> None:
        self._window.clear()
        self._last_frame = None
        self._last_event.clear()

    def observe(
        self,
        *,
        frame_index: int,
        timestamp_ms: float,
        raw_angle_deg: float,
        filtered_angle_deg: float,
        confidence: float,
        valid: bool,
    ) -> tuple[EndpointEvent, ...]:
        if not valid or confidence < self.config.endpoint_minimum_confidence:
            return ()
        if self._last_frame is not None and frame_index != self._last_frame + 1:
            self._window.clear()
        self._last_frame = frame_index
        self._window.append(
            (
                int(frame_index),
                float(timestamp_ms),
                float(raw_angle_deg),
                float(filtered_angle_deg),
                float(confidence),
            )
        )
        if len(self._window) < self._window.maxlen:
            return ()
        values = list(self._window)
        radius = self.config.endpoint_radius_frames
        center = values[radius]
        plateau_radius = min(1, max(0, radius - 1))
        plateau_start = radius - plateau_radius
        plateau_end = radius + plateau_radius + 1
        plateau = values[plateau_start:plateau_end]
        left_flank = values[:plateau_start]
        right_flank = values[plateau_end:]
        candidates: list[Literal["minimum", "maximum"]] = []
        prominence = self.config.endpoint_minimum_prominence_deg
        plateau_min = min(item[3] for item in plateau)
        plateau_max = max(item[3] for item in plateau)
        if (
            left_flank
            and right_flank
            and min(item[3] for item in left_flank) - plateau_min >= prominence
            and min(item[3] for item in right_flank) - plateau_min >= prominence
        ):
            candidates.append("minimum")
        if (
            left_flank
            and right_flank
            and plateau_max - max(item[3] for item in left_flank) >= prominence
            and plateau_max - max(item[3] for item in right_flank) >= prominence
        ):
            candidates.append("maximum")
        output: list[EndpointEvent] = []
        for kind in candidates:
            if center[0] - self._last_event.get(kind, -10_000) <= radius:
                continue
            selected = (
                min(plateau, key=lambda item: (item[2], -item[4]))
                if kind == "minimum"
                else max(plateau, key=lambda item: (item[2], item[4]))
            )
            event = EndpointEvent(
                joint=self.joint,
                kind=kind,
                frame_index=selected[0],
                timestamp_ms=selected[1],
                raw_extremum_angle_deg=selected[2],
                filtered_center_angle_deg=selected[3],
                confidence=selected[4],
                confirmed_at_frame=frame_index,
                window_start_frame=values[0][0],
                window_end_frame=values[-1][0],
            )
            self._last_event[kind] = event.frame_index
            output.append(event)
        return tuple(output)


class AngleHysteresis:
    def __init__(
        self,
        *,
        threshold: float,
        direction: ThresholdDirection,
        width_deg: float,
    ) -> None:
        if direction not in {"above", "below"}:
            raise ValueError(f"unsupported threshold direction: {direction}")
        self.threshold = float(threshold)
        self.direction = direction
        self.width_deg = max(0.0, float(width_deg))
        self.active = False

    def reset(self) -> None:
        self.active = False

    @property
    def enter_threshold(self) -> float:
        return self.threshold

    @property
    def exit_threshold(self) -> float:
        return (
            self.threshold - self.width_deg
            if self.direction == "above"
            else self.threshold + self.width_deg
        )

    def observe(self, value: float | None, *, valid: bool = True) -> bool | None:
        if not valid or value is None or not isfinite(value):
            return None
        if self.direction == "above":
            if not self.active and value >= self.enter_threshold:
                self.active = True
            elif self.active and value < self.exit_threshold:
                self.active = False
        else:
            if not self.active and value <= self.enter_threshold:
                self.active = True
            elif self.active and value > self.exit_threshold:
                self.active = False
        return self.active


class TemporalRuleEvidence:
    """Settle PASS/FAIL only when a timestamped window has stable support."""

    def __init__(
        self,
        *,
        window_size: int,
        pass_count: int,
        fail_count: int,
        minimum_hold_ms: float,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.pass_count = max(1, min(self.window_size, int(pass_count)))
        self.fail_count = max(1, min(self.window_size, int(fail_count)))
        self.minimum_hold_ms = max(0.0, float(minimum_hold_ms))
        self._window: deque[tuple[float, EvidenceState]] = deque(
            maxlen=self.window_size
        )

    @classmethod
    def from_config(cls, config: AngleV2Config) -> TemporalRuleEvidence:
        return cls(
            window_size=config.evidence_window_size,
            pass_count=config.evidence_pass_count,
            fail_count=config.evidence_fail_count,
            minimum_hold_ms=config.evidence_minimum_hold_ms,
        )

    def reset(self) -> None:
        self._window.clear()

    def observe(
        self, state: EvidenceState | bool | None, *, timestamp_ms: float
    ) -> EvidenceState:
        resolved: EvidenceState
        if state is True or state == PASS:
            resolved = PASS
        elif state is False or state == FAIL:
            resolved = FAIL
        else:
            resolved = UNSURE
        self._window.append((float(timestamp_ms), resolved))
        pass_ready = self._ready(PASS, self.pass_count)
        fail_ready = self._ready(FAIL, self.fail_count)
        if pass_ready == fail_ready:
            return UNSURE
        return PASS if pass_ready else FAIL

    def _ready(self, state: EvidenceState, required: int) -> bool:
        opposite = FAIL if state == PASS else PASS
        if any(value == opposite for _, value in self._window):
            return False
        timestamps = [timestamp for timestamp, value in self._window if value == state]
        if len(timestamps) < required:
            return False
        return timestamps[-1] - timestamps[0] >= self.minimum_hold_ms


class AngleV2ShadowProcessor:
    """Causal per-joint Angle V2 shadow stream."""

    def __init__(self, config: AngleV2Config | None = None) -> None:
        self.config = config or AngleV2Config()
        self._filters: dict[str, OneEuroValueFilter] = {}
        self._outlier_gates: dict[str, RobustTemporalAngleGate] = {}
        self._endpoint_filters: dict[str, EndpointPreservingAngleFilter] = {}

    def reset(self) -> None:
        self._filters.clear()
        self._outlier_gates.clear()
        self._endpoint_filters.clear()

    def observe(
        self,
        *,
        joint: str,
        frame_index: int,
        timestamp_ms: float,
        raw_2d_angle_deg: float | None,
        raw_3d_angle_deg: float | None,
        confidence: float,
        quality_reasons: Sequence[str] = (),
    ) -> AngleV2Frame:
        reasons = {str(item) for item in quality_reasons}
        raw_2d = _finite(raw_2d_angle_deg)
        raw_3d = _finite(raw_3d_angle_deg)
        resolved_confidence = max(0.0, min(1.0, float(confidence)))
        bone_valid = not BONE_LENGTH_REASONS.intersection(reasons)
        confidence_valid = resolved_confidence >= self.config.minimum_confidence
        outlier = False
        if raw_2d is not None and confidence_valid and bone_valid:
            accepted, _limit = self._outlier_gate(joint).observe(
                raw_2d, float(timestamp_ms)
            )
            outlier = not accepted
        angle_valid = (
            self.config.enabled
            and raw_2d is not None
            and confidence_valid
            and bone_valid
            and not outlier
        )
        filtered = None
        if angle_valid and raw_2d is not None:
            filtered = self._filter(joint).apply(
                raw_2d,
                timestamp_ms=float(timestamp_ms),
            )
        disagreement = (
            abs(raw_2d - raw_3d)
            if raw_2d is not None and raw_3d is not None
            else None
        )
        conflict = (
            angle_valid
            and disagreement is not None
            and resolved_confidence >= self.config.minimum_confidence
            and disagreement > self.config.conflict_threshold_for(joint)
        )
        evidence_valid = angle_valid and not conflict
        if raw_2d is None:
            reasons.add("ANGLE_MISSING")
        if not confidence_valid:
            reasons.add("ANGLE_CONFIDENCE_LOW")
        if not bone_valid:
            reasons.add("BONE_LENGTH_INCONSISTENT")
        if outlier:
            reasons.add("TEMPORAL_ANGLE_OUTLIER")
        if conflict:
            reasons.add("TWO_D_THREE_D_CONFLICT")
        endpoints = ()
        if raw_2d is not None and filtered is not None:
            endpoints = self._endpoint_filter(joint).observe(
                frame_index=frame_index,
                timestamp_ms=float(timestamp_ms),
                raw_angle_deg=raw_2d,
                filtered_angle_deg=filtered,
                confidence=resolved_confidence,
                valid=evidence_valid,
            )
        return AngleV2Frame(
            joint=joint,
            frame_index=int(frame_index),
            timestamp_ms=float(timestamp_ms),
            raw_2d_angle_deg=raw_2d,
            filtered_2d_angle_deg=filtered,
            raw_3d_angle_deg=raw_3d,
            confidence=resolved_confidence,
            angle_valid=angle_valid,
            evidence_valid=evidence_valid,
            bone_length_valid=bone_valid,
            temporal_outlier=outlier,
            two_d_three_d_conflict=conflict,
            disagreement_deg=disagreement,
            reason_codes=tuple(sorted(reasons)),
            endpoints=endpoints,
        )

    def _filter(self, joint: str) -> OneEuroValueFilter:
        if joint not in self._filters:
            parameters = self.config.smoothing_for(joint)
            self._filters[joint] = OneEuroValueFilter(
                min_cutoff=parameters.min_cutoff,
                beta=parameters.beta,
                d_cutoff=parameters.d_cutoff,
                max_gap_ms_before_reset=self.config.maximum_gap_ms_before_reset,
            )
        return self._filters[joint]

    def _outlier_gate(self, joint: str) -> RobustTemporalAngleGate:
        if joint not in self._outlier_gates:
            self._outlier_gates[joint] = RobustTemporalAngleGate(self.config)
        return self._outlier_gates[joint]

    def _endpoint_filter(self, joint: str) -> EndpointPreservingAngleFilter:
        if joint not in self._endpoint_filters:
            self._endpoint_filters[joint] = EndpointPreservingAngleFilter(
                joint, self.config
            )
        return self._endpoint_filters[joint]


def joint_group(joint: str) -> str:
    normalized = str(joint).strip().lower().removesuffix("_angle")
    for side in ("left_", "right_"):
        normalized = normalized.removeprefix(side)
    return normalized or "default"


def load_angle_v2_config(
    path: str | Path = DEFAULT_ANGLE_V2_CONFIG,
) -> AngleV2Config:
    source = Path(path)
    payload = load_simple_yaml(source)
    section = payload.get("angle_v2_shadow")
    if not isinstance(section, Mapping):
        raise ConfigValidationError(
            "angle_v2_shadow must be a mapping",
            path=source,
            key="angle_v2_shadow",
        )
    smoothing = {
        name.removeprefix("angle_v2_smoothing_"): values
        for name, values in payload.items()
        if name.startswith("angle_v2_smoothing_")
        and isinstance(values, Mapping)
    }
    joint_smoothing = {
        str(name): JointAngleSmoothingConfig(
            min_cutoff=_positive(
                values.get("min_cutoff"),
                source,
                f"joint_smoothing.{name}.min_cutoff",
            ),
            beta=_nonnegative(values.get("beta"), source, f"joint_smoothing.{name}.beta"),
            d_cutoff=_positive(
                values.get("d_cutoff", 1.0),
                source,
                f"joint_smoothing.{name}.d_cutoff",
            ),
        )
        for name, raw_values in smoothing.items()
        if (values := _mapping(raw_values)) is not None
    }
    defaults = AngleV2Config()
    conflict = _mapping(payload.get("angle_v2_conflict_thresholds_deg"))
    resolved = AngleV2Config(
        enabled=_boolean(section.get("enabled", True), source, "enabled"),
        mode=str(section.get("mode", "shadow")),
        joint_smoothing=MappingProxyType(joint_smoothing or dict(defaults.joint_smoothing)),
        minimum_confidence=_unit_interval(
            section.get("minimum_confidence", defaults.minimum_confidence),
            source,
            "minimum_confidence",
        ),
        bone_length_deviation_ratio=_positive(
            section.get("bone_length_deviation_ratio", defaults.bone_length_deviation_ratio),
            source,
            "bone_length_deviation_ratio",
        ),
        temporal_history_size=_positive_int(
            section.get("temporal_history_size", defaults.temporal_history_size),
            source,
            "temporal_history_size",
        ),
        temporal_minimum_history=_positive_int(
            section.get(
                "temporal_minimum_history", defaults.temporal_minimum_history
            ),
            source,
            "temporal_minimum_history",
        ),
        temporal_mad_scale=_positive(
            section.get("temporal_mad_scale", defaults.temporal_mad_scale),
            source,
            "temporal_mad_scale",
        ),
        temporal_minimum_velocity_limit_deg_s=_positive(
            section.get(
                "temporal_minimum_velocity_limit_deg_s",
                defaults.temporal_minimum_velocity_limit_deg_s,
            ),
            source,
            "temporal_minimum_velocity_limit_deg_s",
        ),
        temporal_maximum_velocity_deg_s=_positive(
            section.get(
                "temporal_maximum_velocity_deg_s",
                defaults.temporal_maximum_velocity_deg_s,
            ),
            source,
            "temporal_maximum_velocity_deg_s",
        ),
        maximum_gap_ms_before_reset=_positive(
            section.get("maximum_gap_ms_before_reset", defaults.maximum_gap_ms_before_reset),
            source,
            "maximum_gap_ms_before_reset",
        ),
        endpoint_radius_frames=_positive_int(
            section.get(
                "endpoint_radius_frames", defaults.endpoint_radius_frames
            ),
            source,
            "endpoint_radius_frames",
        ),
        endpoint_minimum_confidence=_unit_interval(
            section.get("endpoint_minimum_confidence", defaults.endpoint_minimum_confidence),
            source,
            "endpoint_minimum_confidence",
        ),
        endpoint_minimum_prominence_deg=_nonnegative(
            section.get(
                "endpoint_minimum_prominence_deg",
                defaults.endpoint_minimum_prominence_deg,
            ),
            source,
            "endpoint_minimum_prominence_deg",
        ),
        evidence_window_size=_positive_int(
            section.get("evidence_window_size", defaults.evidence_window_size),
            source,
            "evidence_window_size",
        ),
        evidence_pass_count=_positive_int(
            section.get("evidence_pass_count", defaults.evidence_pass_count),
            source,
            "evidence_pass_count",
        ),
        evidence_fail_count=_positive_int(
            section.get("evidence_fail_count", defaults.evidence_fail_count),
            source,
            "evidence_fail_count",
        ),
        evidence_minimum_hold_ms=_nonnegative(
            section.get(
                "evidence_minimum_hold_ms", defaults.evidence_minimum_hold_ms
            ),
            source,
            "evidence_minimum_hold_ms",
        ),
        hysteresis_width_deg=_nonnegative(
            section.get("hysteresis_width_deg", defaults.hysteresis_width_deg),
            source,
            "hysteresis_width_deg",
        ),
        conflict_thresholds_deg=MappingProxyType(
            {
                str(name): _positive(value, source, f"2d_3d_conflict_thresholds_deg.{name}")
                for name, value in (conflict or defaults.conflict_thresholds_deg).items()
            }
        ),
    )
    if resolved.mode != "shadow":
        raise ConfigValidationError(
            "Angle V2 may only run in shadow mode in rounds 2-3",
            path=source,
            key="angle_v2_shadow.mode",
        )
    if (
        max(resolved.evidence_pass_count, resolved.evidence_fail_count)
        > resolved.evidence_window_size
    ):
        raise ConfigValidationError(
            "pass_count and fail_count must be <= window_size",
            path=source,
            key="angle_v2_shadow.evidence_window_size",
        )
    return resolved


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object, path: Path, key: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigValidationError("must be numeric", path=path, key=key) from exc
    if not isfinite(resolved):
        raise ConfigValidationError("must be finite", path=path, key=key)
    return resolved


def _positive(value: object, path: Path, key: str) -> float:
    resolved = _number(value, path, key)
    if resolved <= 0.0:
        raise ConfigValidationError("must be > 0", path=path, key=key)
    return resolved


def _nonnegative(value: object, path: Path, key: str) -> float:
    resolved = _number(value, path, key)
    if resolved < 0.0:
        raise ConfigValidationError("must be >= 0", path=path, key=key)
    return resolved


def _unit_interval(value: object, path: Path, key: str) -> float:
    resolved = _number(value, path, key)
    if not 0.0 <= resolved <= 1.0:
        raise ConfigValidationError("must be between 0 and 1", path=path, key=key)
    return resolved


def _positive_int(value: object, path: Path, key: str) -> int:
    resolved = int(_positive(value, path, key))
    if resolved != float(value):
        raise ConfigValidationError("must be an integer", path=path, key=key)
    return resolved


def _boolean(value: object, path: Path, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError("must be a boolean", path=path, key=key)
    return value


def _finite(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if isfinite(resolved) else None


__all__ = [
    "AngleHysteresis",
    "AngleV2Config",
    "AngleV2Frame",
    "AngleV2ShadowProcessor",
    "EndpointEvent",
    "EndpointPreservingAngleFilter",
    "EvidenceState",
    "FAIL",
    "JointAngleSmoothingConfig",
    "PASS",
    "RobustTemporalAngleGate",
    "TemporalRuleEvidence",
    "UNSURE",
    "joint_group",
    "load_angle_v2_config",
]
