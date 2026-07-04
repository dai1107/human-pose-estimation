from __future__ import annotations

from math import isfinite
from statistics import mean, pstdev

import numpy as np

from .schema import CAMERA_VIEWS, SquatCalibration, SquatFrameMeasurement


def _finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and isfinite(float(value))]


def _mean(values: list[float | None]) -> float | None:
    finite = _finite(values)
    return mean(finite) if finite else None


def calibrate_standing(
    frames: list[SquatFrameMeasurement],
    camera_view: str = "unknown",
    minimum_visibility: float = 0.65,
) -> SquatCalibration:
    view = camera_view if camera_view in CAMERA_VIEWS else "unknown"
    usable = [frame for frame in frames if frame.usable(min_visibility=0.2)]
    warnings: list[str] = []
    if not usable:
        return SquatCalibration(
            status="FAIL",
            baseline_pelvis_y=None,
            baseline_pelvis_x=None,
            baseline_body_scale=None,
            baseline_left_knee_angle=None,
            baseline_right_knee_angle=None,
            baseline_left_hip_angle=None,
            baseline_right_hip_angle=None,
            baseline_trunk_tilt_proxy=None,
            visibility_mean=0.0,
            stability_proxy=None,
            camera_view=view,
            warnings=["no usable pose frames for standing calibration"],
        )

    visibility_values = _finite([frame.visibility_mean for frame in usable])
    visibility_mean = mean(visibility_values) if visibility_values else 0.0
    if visibility_mean < minimum_visibility:
        warnings.append("key landmark visibility is below the configured training-rule threshold")

    pelvis_y_values = _finite([frame.pelvis_y for frame in usable])
    pelvis_x_values = _finite([frame.pelvis_x for frame in usable])
    scale_values = _finite([frame.body_scale for frame in usable])
    baseline_scale = mean(scale_values) if scale_values else None
    stability_proxy = None
    if len(pelvis_y_values) >= 2 and baseline_scale and baseline_scale > 1e-9:
        stability_proxy = pstdev(pelvis_y_values) / baseline_scale
        if stability_proxy > 0.05:
            warnings.append("initial standing posture is not stable enough for high-confidence baseline")

    required_names = {
        "left_knee_angle": [frame.left_knee_angle for frame in usable],
        "right_knee_angle": [frame.right_knee_angle for frame in usable],
        "left_hip_angle": [frame.left_hip_angle for frame in usable],
        "right_hip_angle": [frame.right_hip_angle for frame in usable],
    }
    for name, values in required_names.items():
        if not _finite(values):
            warnings.append(f"{name} unavailable during calibration")

    if baseline_scale is None or not pelvis_y_values:
        status = "FAIL"
    elif warnings:
        status = "WARNING"
    else:
        status = "PASS"

    return SquatCalibration(
        status=status,
        baseline_pelvis_y=mean(pelvis_y_values) if pelvis_y_values else None,
        baseline_pelvis_x=mean(pelvis_x_values) if pelvis_x_values else None,
        baseline_body_scale=baseline_scale,
        baseline_left_knee_angle=_mean([frame.left_knee_angle for frame in usable]),
        baseline_right_knee_angle=_mean([frame.right_knee_angle for frame in usable]),
        baseline_left_hip_angle=_mean([frame.left_hip_angle for frame in usable]),
        baseline_right_hip_angle=_mean([frame.right_hip_angle for frame in usable]),
        baseline_trunk_tilt_proxy=_mean([frame.trunk_tilt_proxy for frame in usable]),
        visibility_mean=visibility_mean,
        stability_proxy=stability_proxy,
        camera_view=view,
        warnings=warnings,
    )


class StandingCalibrationBuilder:
    def __init__(self, duration_ms: int = 2500, camera_view: str = "unknown", minimum_visibility: float = 0.65) -> None:
        self.duration_ms = max(500, int(duration_ms))
        self.camera_view = camera_view
        self.minimum_visibility = minimum_visibility
        self.frames: list[SquatFrameMeasurement] = []
        self.started_at_ms: int | None = None
        self._requested = False

    @property
    def active(self) -> bool:
        return self._requested

    def start(self) -> None:
        self.frames = []
        self.started_at_ms = None
        self._requested = True

    def add(self, frame: SquatFrameMeasurement) -> SquatCalibration | None:
        if self.started_at_ms is None:
            self.started_at_ms = frame.timestamp_ms
        self.frames.append(frame)
        if frame.timestamp_ms - self.started_at_ms >= self.duration_ms:
            result = calibrate_standing(self.frames, self.camera_view, self.minimum_visibility)
            self.started_at_ms = None
            self._requested = False
            return result
        return None

    def progress(self, timestamp_ms: int) -> float:
        if self.started_at_ms is None:
            return 0.0
        return float(np.clip((timestamp_ms - self.started_at_ms) / self.duration_ms, 0.0, 1.0))
