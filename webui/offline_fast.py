"""Timestamp-driven scheduling for the MediaPipe-only offline Fast path.

The scheduler is deliberately independent from OpenCV and the HYROX rule
implementations.  A caller feeds coarse rule observations back into it; those
observations may open a short high-density window for subsequent source
timestamps.  Every selected frame therefore remains in chronological order and
the same MediaPipe VIDEO instance can be reused safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


_INACTIVE_PHASES = frozenset(
    {
        "",
        "idle",
        "unknown",
        "no_pose",
        "low_visibility",
        "ready",
        "rest",
        "reset",
        "stand",
        "standing",
    }
)


@dataclass(frozen=True, slots=True)
class FastFrameSelection:
    analyze: bool
    coarse: bool = False
    refinement: bool = False


@dataclass(frozen=True, slots=True)
class CandidateWindow:
    start_ms: float
    end_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
        }


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    timestamp_ms: float
    phase: str
    candidate_count: int = 0


def build_candidate_windows(
    observations: list[CandidateObservation],
    *,
    margin_ms: float,
    duration_ms: float,
) -> tuple[CandidateWindow, ...]:
    """Build merged candidate_start-margin/candidate_end+margin windows."""

    if not observations:
        return ()
    margin_ms = max(0.0, float(margin_ms))
    duration_ms = max(0.0, float(duration_ms))
    raw: list[CandidateWindow] = []
    previous: CandidateObservation | None = None
    for observation in observations:
        phase = str(observation.phase or "unknown").strip().lower()
        active = phase not in _INACTIVE_PHASES
        phase_transition = bool(
            previous is not None
            and phase != str(previous.phase).strip().lower()
            and (
                active
                or str(previous.phase).strip().lower() not in _INACTIVE_PHASES
            )
        )
        candidate_incremented = bool(
            previous is not None
            and observation.candidate_count > previous.candidate_count
        )
        if active or phase_transition or candidate_incremented:
            raw.append(
                CandidateWindow(
                    start_ms=max(0.0, observation.timestamp_ms - margin_ms),
                    end_ms=min(duration_ms, observation.timestamp_ms + margin_ms),
                )
            )
        previous = observation
    merged: list[CandidateWindow] = []
    for window in raw:
        if merged and window.start_ms <= merged[-1].end_ms + 1e-6:
            merged[-1] = CandidateWindow(
                merged[-1].start_ms,
                max(merged[-1].end_ms, window.end_ms),
            )
        else:
            merged.append(window)
    return tuple(merged)


def timestamp_in_windows(
    timestamp_ms: float,
    windows: tuple[CandidateWindow, ...],
) -> bool:
    return any(
        window.start_ms - 1e-6 <= timestamp_ms <= window.end_ms + 1e-6
        for window in windows
    )


class TimestampSampler:
    """Select frames by media timestamps instead of frame-number modulus."""

    def __init__(self, target_fps: float) -> None:
        target_fps = float(target_fps)
        if not isfinite(target_fps) or target_fps <= 0.0:
            raise ValueError("target_fps must be a positive finite number")
        self.target_fps = target_fps
        self.interval_ms = 1000.0 / target_fps
        self._next_due_ms: float | None = None
        self._last_timestamp_ms: float | None = None

    def sample(self, timestamp_ms: float) -> bool:
        timestamp_ms = float(timestamp_ms)
        if not isfinite(timestamp_ms) or timestamp_ms < 0.0:
            raise ValueError("timestamp_ms must be a finite non-negative number")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            raise ValueError("timestamps must be monotonic")
        self._last_timestamp_ms = timestamp_ms
        if self._next_due_ms is None:
            self._next_due_ms = timestamp_ms + self.interval_ms
            return True
        # Container timestamps are commonly rounded to three decimals (for
        # example 33.333 ms at 30 FPS).  A sub-millisecond tolerance prevents
        # that harmless quantization from accidentally halving the target FPS.
        tolerance_ms = 0.5
        if timestamp_ms + tolerance_ms < self._next_due_ms:
            return False
        while self._next_due_ms <= timestamp_ms + tolerance_ms:
            self._next_due_ms += self.interval_ms
        return True


class AdaptiveOfflineFastScheduler:
    """Coarse scan plus forward dense refinement on one media timeline.

    Only coarse observations are allowed to open/extend candidate windows.  A
    separate formal analyzer can consume all selected frames, so discovery
    state never becomes a VALID/NO_REP/UNSURE decision in the final report.
    """

    def __init__(
        self,
        *,
        target_pose_fps: float = 15.0,
        refinement_pose_fps: float = 30.0,
        candidate_margin_ms: float = 300.0,
        refinement_enabled: bool = True,
    ) -> None:
        if refinement_pose_fps < target_pose_fps:
            raise ValueError("refinement_pose_fps must be >= target_pose_fps")
        if candidate_margin_ms < 0.0 or not isfinite(float(candidate_margin_ms)):
            raise ValueError("candidate_margin_ms must be finite and non-negative")
        self.target_pose_fps = float(target_pose_fps)
        self.refinement_pose_fps = float(refinement_pose_fps)
        self.candidate_margin_ms = float(candidate_margin_ms)
        self.refinement_enabled = bool(refinement_enabled)
        self._coarse = TimestampSampler(self.target_pose_fps)
        self._dense = TimestampSampler(self.refinement_pose_fps)
        self._refinement_until_ms = -1.0
        self._last_phase = ""
        self._last_candidate_count = 0
        self._windows: list[CandidateWindow] = []
        self.coarse_pose_frames = 0
        self.refinement_pose_frames = 0

    @property
    def candidate_windows(self) -> tuple[CandidateWindow, ...]:
        return tuple(self._windows)

    def select(self, timestamp_ms: float) -> FastFrameSelection:
        coarse = self._coarse.sample(timestamp_ms)
        dense_due = self._dense.sample(timestamp_ms)
        refinement = bool(
            self.refinement_enabled
            and timestamp_ms <= self._refinement_until_ms + 1e-6
            and dense_due
            and not coarse
        )
        if coarse:
            self.coarse_pose_frames += 1
        elif refinement:
            self.refinement_pose_frames += 1
        return FastFrameSelection(
            analyze=coarse or refinement,
            coarse=coarse,
            refinement=refinement,
        )

    def observe_coarse(
        self,
        *,
        timestamp_ms: float,
        phase: str,
        candidate_count: int,
    ) -> bool:
        """Feed discovery-only rule state and possibly open a dense window."""

        normalized = str(phase or "unknown").strip().lower()
        candidate_count = max(0, int(candidate_count))
        candidate_incremented = candidate_count > self._last_candidate_count
        phase_changed = bool(
            self._last_phase
            and normalized != self._last_phase
            and (
                normalized not in _INACTIVE_PHASES
                or self._last_phase not in _INACTIVE_PHASES
            )
        )
        active = normalized not in _INACTIVE_PHASES
        triggered = bool(
            self.refinement_enabled
            and (active or phase_changed or candidate_incremented)
        )
        if triggered:
            self._open_or_extend_window(float(timestamp_ms))
        self._last_phase = normalized
        self._last_candidate_count = max(
            self._last_candidate_count,
            candidate_count,
        )
        return triggered

    def _open_or_extend_window(self, timestamp_ms: float) -> None:
        end_ms = timestamp_ms + self.candidate_margin_ms
        self._refinement_until_ms = max(self._refinement_until_ms, end_ms)
        if self._windows and timestamp_ms <= self._windows[-1].end_ms + 1e-6:
            previous = self._windows[-1]
            self._windows[-1] = CandidateWindow(
                start_ms=previous.start_ms,
                end_ms=max(previous.end_ms, end_ms),
            )
        else:
            self._windows.append(CandidateWindow(timestamp_ms, end_ms))

    def summary(self, *, source_frame_count: int) -> dict[str, Any]:
        pose_frames = self.coarse_pose_frames + self.refinement_pose_frames
        source_frames = max(0, int(source_frame_count))
        return {
            "offline_fast_enabled": True,
            "target_pose_fps": self.target_pose_fps,
            "refinement_enabled": self.refinement_enabled,
            "refinement_pose_fps": self.refinement_pose_fps,
            "candidate_margin_ms": self.candidate_margin_ms,
            "coarse_pose_frames": self.coarse_pose_frames,
            "refinement_pose_frames": self.refinement_pose_frames,
            "pose_frames": pose_frames,
            "source_frames": source_frames,
            "pose_sampling_ratio": (
                round(pose_frames / source_frames, 6)
                if source_frames > 0
                else 0.0
            ),
            "refinement_candidate_count": len(self._windows),
            "candidate_windows": [window.as_dict() for window in self._windows],
            "discovery_decisions_are_formal": False,
        }


def media_timestamp_ms(
    capture: Any,
    *,
    frame_index: int,
    source_fps: float,
    previous_timestamp_ms: float | None,
    pos_msec_property: int,
) -> float:
    """Return a monotonic media timestamp, preferring valid container PTS."""

    fallback = max(0, int(frame_index)) * 1000.0 / max(float(source_fps), 1e-6)
    try:
        candidate = float(capture.get(pos_msec_property))
    except (AttributeError, TypeError, ValueError):
        candidate = fallback
    if not isfinite(candidate) or candidate < 0.0:
        candidate = fallback
    if previous_timestamp_ms is not None and candidate <= previous_timestamp_ms:
        candidate = max(fallback, previous_timestamp_ms + 1000.0 / max(source_fps, 1e-6))
    return round(candidate, 3)


def discovery_state_values(state: Mapping[str, Any] | None) -> tuple[str, int]:
    if not isinstance(state, Mapping):
        return "idle", 0
    phase = str(state.get("phase", "unknown"))
    count = int(state.get("candidate_count", state.get("rep_count", 0)) or 0)
    return phase, count


__all__ = [
    "AdaptiveOfflineFastScheduler",
    "CandidateWindow",
    "CandidateObservation",
    "FastFrameSelection",
    "TimestampSampler",
    "build_candidate_windows",
    "discovery_state_values",
    "media_timestamp_ms",
    "timestamp_in_windows",
]
