from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .schema import SquatCalibration, SquatFrameMeasurement, SquatRep


@dataclass(frozen=True)
class SquatStateUpdate:
    state: str
    completed_rep: SquatRep | None
    paused: bool
    displacement: float | None
    message: str = ""


def _delta_angle(baseline: float | None, current: float | None) -> float:
    if baseline is None or current is None:
        return 0.0
    return float(baseline) - float(current)


class SquatStateMachine:
    def __init__(self, config: dict, calibration: SquatCalibration | None = None) -> None:
        self.config = config
        self.rep_config = config.get("rep_detection", {})
        self.calibration = calibration
        self.state = "IDLE"
        self.rep_count = 0
        self._ready_since_ms: int | None = None
        self._rep_frames: list[SquatFrameMeasurement] = []
        self._last_usable: SquatFrameMeasurement | None = None
        self._bottom_frame: SquatFrameMeasurement | None = None
        self._bottom_entered_ms: int | None = None
        self._ascent_started_ms: int | None = None
        self._descent_started_ms: int | None = None
        self._lost_since_ms: int | None = None
        if calibration is not None and calibration.passed:
            self.state = "READY"

    def set_calibration(self, calibration: SquatCalibration) -> None:
        self.calibration = calibration
        self.reset(clear_calibration=False)
        self.state = "READY" if calibration.passed else "IDLE"

    def reset(self, clear_calibration: bool = False) -> None:
        if clear_calibration:
            self.calibration = None
        self.state = "READY" if self.calibration is not None and self.calibration.passed else "IDLE"
        self._ready_since_ms = None
        self._rep_frames = []
        self._last_usable = None
        self._bottom_frame = None
        self._bottom_entered_ms = None
        self._ascent_started_ms = None
        self._descent_started_ms = None
        self._lost_since_ms = None

    def _displacement(self, frame: SquatFrameMeasurement) -> float | None:
        calibration = self.calibration
        if calibration is None or calibration.baseline_pelvis_y is None or calibration.baseline_body_scale is None:
            return None
        if frame.pelvis_y is None or calibration.baseline_body_scale <= 1e-9:
            return None
        return (frame.pelvis_y - calibration.baseline_pelvis_y) / calibration.baseline_body_scale

    def _is_flexing(self, frame: SquatFrameMeasurement) -> bool:
        calibration = self.calibration
        if calibration is None:
            return False
        knee_delta = max(
            _delta_angle(calibration.baseline_left_knee_angle, frame.left_knee_angle),
            _delta_angle(calibration.baseline_right_knee_angle, frame.right_knee_angle),
        )
        hip_delta = max(
            _delta_angle(calibration.baseline_left_hip_angle, frame.left_hip_angle),
            _delta_angle(calibration.baseline_right_hip_angle, frame.right_hip_angle),
        )
        return (
            knee_delta >= float(self.rep_config.get("min_knee_flexion_delta", 8.0))
            or hip_delta >= float(self.rep_config.get("min_hip_flexion_delta", 6.0))
        )

    def _velocity(self, frame: SquatFrameMeasurement, displacement: float | None) -> float | None:
        previous = self._last_usable
        if previous is None or displacement is None:
            return None
        previous_disp = self._displacement(previous)
        dt = (frame.timestamp_ms - previous.timestamp_ms) / 1000.0
        if previous_disp is None or dt <= 0 or dt > 1.0:
            return None
        return (displacement - previous_disp) / dt

    def update(self, frame: SquatFrameMeasurement) -> SquatStateUpdate:
        if self.calibration is None or not self.calibration.passed:
            self.state = "IDLE"
            return SquatStateUpdate(self.state, None, False, None, "calibration unavailable")

        minimum_visibility = float(self.config.get("data_quality", {}).get("minimum_landmark_visibility", 0.65))
        if not frame.usable(min_visibility=minimum_visibility * 0.5):
            if self._lost_since_ms is None:
                self._lost_since_ms = frame.timestamp_ms
            if frame.timestamp_ms - self._lost_since_ms >= int(self.rep_config.get("lost_reset_ms", 350)):
                self.reset(clear_calibration=False)
                self.state = "PAUSED"
            return SquatStateUpdate(self.state, None, True, None, "pose unavailable")

        self._lost_since_ms = None
        displacement = self._displacement(frame)
        velocity = self._velocity(frame, displacement)
        completed: SquatRep | None = None
        baseline_tolerance = float(self.rep_config.get("baseline_return_tolerance", 0.15))
        descent_start = float(self.rep_config.get("descent_start_displacement", baseline_tolerance * 0.55))
        min_rep = int(self.rep_config.get("min_rep_duration_ms", 600))
        min_descent = int(self.rep_config.get("min_descent_duration_ms", 180))
        min_ascent = int(self.rep_config.get("min_ascent_duration_ms", 180))
        bottom_velocity = float(self.rep_config.get("bottom_velocity_threshold", 0.15))
        min_bottom = float(self.rep_config.get("min_bottom_displacement", baseline_tolerance))

        if self.state in {"IDLE", "PAUSED", "COMPLETE"}:
            self.state = "READY"

        if self.state == "READY":
            stable_ready_ms = int(self.rep_config.get("stable_ready_ms", 180))
            ready_stable = self._ready_since_ms is not None and frame.timestamp_ms - self._ready_since_ms >= stable_ready_ms
            if ready_stable and displacement is not None and displacement >= descent_start and self._is_flexing(frame):
                self.state = "DESCENT"
                self._descent_started_ms = frame.timestamp_ms
                self._rep_frames = [frame]
                self._bottom_frame = frame
                self._bottom_entered_ms = None
            elif displacement is not None and abs(displacement) <= baseline_tolerance:
                if self._ready_since_ms is None:
                    self._ready_since_ms = frame.timestamp_ms
            else:
                self._ready_since_ms = None

        elif self.state == "DESCENT":
            self._rep_frames.append(frame)
            current_bottom_disp = self._displacement(self._bottom_frame) if self._bottom_frame else None
            if displacement is not None and (current_bottom_disp is None or displacement >= current_bottom_disp):
                self._bottom_frame = frame
            descent_elapsed = frame.timestamp_ms - (self._descent_started_ms or frame.timestamp_ms)
            if (
                descent_elapsed >= min_descent
                and displacement is not None
                and displacement >= min_bottom
                and velocity is not None
                and velocity <= bottom_velocity
            ):
                self.state = "BOTTOM"
                if velocity < -bottom_velocity and self._bottom_frame is not None:
                    self._bottom_entered_ms = self._bottom_frame.timestamp_ms
                else:
                    self._bottom_entered_ms = frame.timestamp_ms

        elif self.state == "BOTTOM":
            self._rep_frames.append(frame)
            current_bottom_disp = self._displacement(self._bottom_frame) if self._bottom_frame else None
            if displacement is not None and (current_bottom_disp is None or displacement >= current_bottom_disp):
                self._bottom_frame = frame
            hold_elapsed = frame.timestamp_ms - (self._bottom_entered_ms or frame.timestamp_ms)
            if velocity is not None and velocity < -bottom_velocity:
                self.state = "ASCENT"
                self._ascent_started_ms = self._bottom_entered_ms or frame.timestamp_ms
            elif hold_elapsed > int(self.rep_config.get("bottom_hold_max_ms", 1200)):
                self.state = "ASCENT"
                self._ascent_started_ms = frame.timestamp_ms

        elif self.state == "ASCENT":
            self._rep_frames.append(frame)
            ascent_elapsed = frame.timestamp_ms - (self._ascent_started_ms or frame.timestamp_ms)
            total_elapsed = frame.timestamp_ms - self._rep_frames[0].timestamp_ms if self._rep_frames else 0
            returned = displacement is not None and displacement <= baseline_tolerance
            extending = not self._is_flexing(frame) or (velocity is not None and velocity <= 0)
            if ascent_elapsed >= min_ascent and total_elapsed >= min_rep and returned and extending:
                self.rep_count += 1
                bottom = self._bottom_frame or min(self._rep_frames, key=lambda item: self._displacement(item) or 0.0)
                completed = SquatRep(
                    rep_index=self.rep_count,
                    start_timestamp_ms=self._rep_frames[0].timestamp_ms,
                    bottom_timestamp_ms=bottom.timestamp_ms,
                    end_timestamp_ms=frame.timestamp_ms,
                    start_frame_index=self._rep_frames[0].frame_index,
                    bottom_frame_index=bottom.frame_index,
                    end_frame_index=frame.frame_index,
                    frames=list(self._rep_frames),
                )
                self.state = "READY"
                self._rep_frames = []
                self._bottom_frame = None
                self._bottom_entered_ms = None
                self._ascent_started_ms = None
                self._descent_started_ms = None
                self._ready_since_ms = frame.timestamp_ms

        self._last_usable = frame
        return SquatStateUpdate(self.state, completed, False, displacement)
