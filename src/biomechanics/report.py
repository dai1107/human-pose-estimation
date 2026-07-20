from __future__ import annotations

import json
import os
from math import isfinite, nan
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from src.output_schema import versioned_payload

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"))

from .sequencing import compare_peak_order, find_local_peaks
from .types import KinematicFrame, PoseFrame


ANGLE_FIELDS = (
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
)

VELOCITY_FIELDS = (
    "pelvis_speed",
    "left_wrist_speed",
    "right_wrist_speed",
    "left_ankle_speed",
    "right_ankle_speed",
)


def _finite_values(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if isfinite(float(value))]


def _field_values(frames: list[KinematicFrame], field_name: str) -> list[float]:
    return [float(getattr(frame, field_name, nan)) for frame in frames]


def _stats(values: list[float], include_std: bool = True) -> dict[str, float | None]:
    finite = _finite_values(values)
    if not finite:
        result: dict[str, float | None] = {"mean": None, "min": None, "max": None}
        if include_std:
            result["std"] = None
        return result
    result = {"mean": mean(finite), "min": min(finite), "max": max(finite)}
    if include_std:
        result["std"] = pstdev(finite) if len(finite) > 1 else 0.0
    return result


def build_summary(pose_frames: list[PoseFrame], kinematic_frames: list[KinematicFrame]) -> dict[str, object]:
    angle_stats = {field: _stats(_field_values(kinematic_frames, field)) for field in ANGLE_FIELDS}
    velocity_stats = {
        field: {"mean": _stats(_field_values(kinematic_frames, field), include_std=False)["mean"],
                "max": _stats(_field_values(kinematic_frames, field), include_std=False)["max"]}
        for field in VELOCITY_FIELDS
    }
    detected_count = sum(1 for frame in pose_frames if frame.pose_detected)
    valid_ratio = detected_count / len(pose_frames) if pose_frames else 0.0
    energy_values = _finite_values(_field_values(kinematic_frames, "motion_energy_proxy"))

    peak_events = detect_peak_events(kinematic_frames)
    return {
        "angle_stats": angle_stats,
        "velocity_stats": velocity_stats,
        "pose_valid_frame_ratio": valid_ratio,
        "motion_energy_proxy_peak": max(energy_values) if energy_values else None,
        "peak_events": peak_events,
    }


def detect_peak_events(kinematic_frames: list[KinematicFrame]) -> dict[str, float]:
    timestamps = [frame.timestamp_ms for frame in kinematic_frames]
    sources = {
        "pelvis_speed_peak": "pelvis_speed",
        "right_elbow_angular_velocity_peak": "right_elbow_angular_velocity",
        "right_wrist_speed_peak": "right_wrist_speed",
    }
    events: dict[str, float] = {}
    for event_name, field_name in sources.items():
        values = _field_values(kinematic_frames, field_name)
        peaks = find_local_peaks(values, timestamps, min_distance_ms=120, min_prominence=0.0)
        if peaks:
            strongest = max(peaks, key=lambda peak: peak["value"])
            events[event_name] = strongest["timestamp_ms"]
    return events


def build_sequence_summary(kinematic_frames: list[KinematicFrame]) -> dict[str, object]:
    events = detect_peak_events(kinematic_frames)
    expected = ("pelvis_speed_peak", "right_elbow_angular_velocity_peak", "right_wrist_speed_peak")
    comparison = compare_peak_order(events, expected_order=expected)
    return {"events": events, "comparison": comparison}


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_fields(path: Path, frames: list[KinematicFrame], fields: tuple[str, ...], title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frames:
        return
    first_ts = frames[0].timestamp_ms
    times = [(frame.timestamp_ms - first_ts) / 1000.0 for frame in frames]

    plt.figure(figsize=(10, 5))
    for field in fields:
        values = _field_values(frames, field)
        plt.plot(times, values, label=field)
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def write_report_outputs(
    session_dir: Path,
    pose_frames: list[PoseFrame],
    kinematic_frames: list[KinematicFrame],
    plot_on_save: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    session_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(pose_frames, kinematic_frames)
    sequence_summary = build_sequence_summary(kinematic_frames)
    save_json(
        session_dir / "summary.json",
        versioned_payload("pose_session_summary", summary),
    )
    save_json(
        session_dir / "sequence_summary.json",
        versioned_payload("pose_session_sequence_summary", sequence_summary),
    )

    if plot_on_save and kinematic_frames:
        _plot_fields(session_dir / "angle_curves.png", kinematic_frames, ANGLE_FIELDS, "Joint angle curves", "angle (deg)")
        _plot_fields(session_dir / "velocity_curves.png", kinematic_frames, VELOCITY_FIELDS, "Velocity proxy curves", "normalized units / s")

    return summary, sequence_summary
