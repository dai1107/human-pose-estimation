from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import calibrate_standing
from .schema import SquatCalibration, SquatFrameMeasurement, SquatRep, load_squat_config
from .state_machine import SquatStateMachine


@dataclass(frozen=True)
class SquatDetectionResult:
    calibration: SquatCalibration
    reps: list[SquatRep]
    frame_states: list[dict[str, Any]]


class SquatRepDetector:
    def __init__(
        self,
        config: dict | None = None,
        calibration: SquatCalibration | None = None,
        camera_view: str = "unknown",
    ) -> None:
        self.config = config or load_squat_config()
        self.camera_view = camera_view
        self.calibration = calibration
        self.machine = SquatStateMachine(self.config, calibration)
        self.reps: list[SquatRep] = []
        self.frame_states: list[dict[str, Any]] = []

    def set_calibration(self, calibration: SquatCalibration) -> None:
        self.calibration = calibration
        self.machine.set_calibration(calibration)

    def update(self, frame: SquatFrameMeasurement) -> SquatRep | None:
        update = self.machine.update(frame)
        self.frame_states.append(
            {
                "frame_index": frame.frame_index,
                "timestamp_ms": frame.timestamp_ms,
                "state": update.state,
                "rep_count": self.machine.rep_count,
                "pose_detected": int(frame.pose_detected),
                "pelvis_displacement_normalized": "" if update.displacement is None else f"{update.displacement:.8g}",
                "paused": int(update.paused),
                "message": update.message,
            }
        )
        if update.completed_rep is not None:
            self.reps.append(update.completed_rep)
            return update.completed_rep
        return None


def detect_squat_reps(
    frames: list[SquatFrameMeasurement],
    camera_view: str = "unknown",
    config: dict | None = None,
    calibration_frames: int = 5,
) -> SquatDetectionResult:
    active_config = config or load_squat_config()
    minimum_visibility = float(active_config.get("data_quality", {}).get("minimum_landmark_visibility", 0.65))
    calibration_source = frames[: max(1, calibration_frames)]
    calibration = calibrate_standing(calibration_source, camera_view=camera_view, minimum_visibility=minimum_visibility)
    detector = SquatRepDetector(active_config, calibration, camera_view=camera_view)
    for frame in frames:
        detector.update(frame)
    return SquatDetectionResult(calibration=calibration, reps=detector.reps, frame_states=detector.frame_states)

