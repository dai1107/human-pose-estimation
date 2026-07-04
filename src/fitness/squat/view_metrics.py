from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schema import SquatRepMetrics


SIDE_AVAILABLE = (
    "pelvis_vertical_displacement_normalized",
    "left_knee_angle_range",
    "right_knee_angle_range",
    "left_hip_angle_range",
    "right_hip_angle_range",
    "trunk_tilt_range",
    "descent_duration_ms",
    "ascent_duration_ms",
    "bottom_duration_ms",
)

FRONT_AVAILABLE = (
    "left_right_knee_difference_mean",
    "left_right_knee_difference_peak",
    "left_right_hip_difference_mean",
    "left_right_hip_difference_peak",
    "pelvis_lateral_drift_proxy",
    "trunk_lateral_drift_proxy",
)


@dataclass(frozen=True)
class ViewMetricSummary:
    camera_view: str
    available_metrics: list[str]
    unavailable_or_low_reliability: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_view_metrics(camera_view: str, metrics: list[SquatRepMetrics] | None = None) -> ViewMetricSummary:
    view = camera_view if camera_view in {"side", "front", "front_left", "front_right"} else "unknown"
    if view == "side":
        available = list(SIDE_AVAILABLE)
        unavailable = [
            "knee_lateral_trajectory_proxy",
            "pelvis_lateral_drift_proxy",
            "trunk_lateral_drift_proxy",
        ]
        notes = ["SIDE view: lateral knee trajectory is not reported or is low reliability."]
    elif view == "front":
        available = list(FRONT_AVAILABLE)
        unavailable = [
            "precise_squat_depth",
            "precise_hip_knee_flexion_depth",
        ]
        notes = ["FRONT view: hip and knee flexion depth is only a 2D visual proxy, not precise depth measurement."]
    elif view in {"front_left", "front_right"}:
        available = list(dict.fromkeys(list(SIDE_AVAILABLE) + list(FRONT_AVAILABLE)))
        unavailable = ["precise_depth_or_lateral_tracking"]
        notes = ["Angled front view provides mixed visual proxy metrics with lower reliability."]
    else:
        available = [
            "rep_count",
            "left_knee_min_angle",
            "right_knee_min_angle",
            "left_hip_min_angle",
            "right_hip_min_angle",
        ]
        unavailable = ["view_sensitive_metrics"] + list(dict.fromkeys(list(SIDE_AVAILABLE) + list(FRONT_AVAILABLE)))
        notes = ["UNKNOWN view: only basic joint angles and repetition count are reported."]
    return ViewMetricSummary(view.upper(), available, unavailable, notes)
