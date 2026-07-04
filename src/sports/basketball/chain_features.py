from __future__ import annotations

from collections import defaultdict
from math import hypot, isfinite
from pathlib import Path
from typing import Any

import numpy as np

from src.reference.session_loader import SessionData, parse_number

from .schema import ShotEvent, ShotFrame, finite_or_none
from .side_selector import opposite_side, side_field, validate_shooting_side


def _finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and isfinite(float(value))]


def _landmark_point(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    x = finite_or_none(row.get("smoothed_x"))
    y = finite_or_none(row.get("smoothed_y"))
    z = finite_or_none(row.get("smoothed_z"))
    if x is None:
        x = finite_or_none(row.get("image_x"))
    if y is None:
        y = finite_or_none(row.get("image_y"))
    if z is None:
        z = finite_or_none(row.get("image_z"))
    return x, y, z


def _distance(a: tuple[float | None, float | None, float | None] | None, b: tuple[float | None, float | None, float | None] | None) -> float | None:
    if a is None or b is None or a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return None
    return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _midpoint(a: tuple[float | None, float | None, float | None] | None, b: tuple[float | None, float | None, float | None] | None) -> tuple[float | None, float | None, float | None] | None:
    if a is None or b is None or a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return None
    return ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0, ((a[2] or 0.0) + (b[2] or 0.0)) / 2.0)


def build_shot_frames_from_session(session: SessionData, shooting_side: str) -> list[ShotFrame]:
    side = validate_shooting_side(shooting_side)
    other = opposite_side(side)
    landmarks_by_frame: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in session.landmarks:
        frame_index = int(parse_number(row.get("frame_index")))
        landmarks_by_frame[frame_index][str(row.get("landmark_name", ""))] = row

    frames: list[ShotFrame] = []
    for row in session.kinematics:
        frame_index = int(parse_number(row.get("frame_index")))
        landmark_rows = landmarks_by_frame.get(frame_index, {})
        points = {name: _landmark_point(value) for name, value in landmark_rows.items()}
        pelvis = _midpoint(points.get("left_hip"), points.get("right_hip"))
        shoulder = _midpoint(points.get("left_shoulder"), points.get("right_shoulder"))
        shoulder_width = _distance(points.get("left_shoulder"), points.get("right_shoulder"))
        hip_width = _distance(points.get("left_hip"), points.get("right_hip"))
        torso = _distance(pelvis, shoulder)
        scale = next((value for value in (shoulder_width, hip_width, torso) if value is not None and value > 1e-9), None)
        pose_raw = row.get("pose_detected")
        pose_detected = str(pose_raw).strip().lower() in {"1", "true", "yes"} if pose_raw is not None else True
        visibility_mean = finite_or_none(row.get("visibility_mean"))
        if visibility_mean is None:
            vis = _finite([finite_or_none(value.get("visibility")) for value in landmark_rows.values()])
            visibility_mean = sum(vis) / len(vis) if vis else None

        wrist = points.get(f"{side}_wrist")
        elbow = points.get(f"{side}_elbow")
        shoulder_point = points.get(f"{side}_shoulder")
        frames.append(
            ShotFrame(
                frame_index=frame_index,
                timestamp_ms=int(parse_number(row.get("timestamp_ms"))),
                pose_detected=pose_detected,
                pelvis_x=pelvis[0] if pelvis else None,
                pelvis_y=pelvis[1] if pelvis else None,
                shoulder_x=shoulder[0] if shoulder else None,
                shoulder_y=shoulder[1] if shoulder else None,
                body_scale=scale,
                shooting_side=side,
                shooting_knee_angle=finite_or_none(row.get(side_field(side, "knee_angle"))),
                non_shooting_knee_angle=finite_or_none(row.get(side_field(other, "knee_angle"))),
                shooting_hip_angle=finite_or_none(row.get(side_field(side, "hip_angle"))),
                non_shooting_hip_angle=finite_or_none(row.get(side_field(other, "hip_angle"))),
                shooting_shoulder_angle=finite_or_none(row.get(side_field(side, "shoulder_angle"))),
                shooting_elbow_angle=finite_or_none(row.get(side_field(side, "elbow_angle"))),
                shooting_elbow_angular_velocity=finite_or_none(row.get(side_field(side, "elbow_angular_velocity"))),
                shooting_wrist_speed=finite_or_none(row.get(side_field(side, "wrist_speed"))),
                shooting_ankle_speed=finite_or_none(row.get(side_field(side, "ankle_speed"))),
                pelvis_speed=finite_or_none(row.get("pelvis_speed")),
                shoulder_center_speed=finite_or_none(row.get("shoulder_center_speed")),
                trunk_tilt_proxy=finite_or_none(row.get("trunk_tilt_proxy")),
                motion_energy_proxy=finite_or_none(row.get("motion_energy_proxy")),
                shooting_wrist_x=wrist[0] if wrist else None,
                shooting_wrist_y=wrist[1] if wrist else None,
                shooting_elbow_x=elbow[0] if elbow else None,
                shooting_elbow_y=elbow[1] if elbow else None,
                shooting_shoulder_x=shoulder_point[0] if shoulder_point else None,
                shooting_shoulder_y=shoulder_point[1] if shoulder_point else None,
                visibility_mean=visibility_mean,
                missing_ratio=finite_or_none(row.get("missing_ratio")),
                source=dict(row),
            )
        )
    return frames


def derivative(values: list[float | None], timestamps: list[int]) -> list[float | None]:
    output: list[float | None] = [None]
    for prev_value, cur_value, prev_ts, cur_ts in zip(values, values[1:], timestamps, timestamps[1:]):
        if prev_value is None or cur_value is None:
            output.append(None)
            continue
        dt = (cur_ts - prev_ts) / 1000.0
        output.append((float(cur_value) - float(prev_value)) / dt if 0 < dt <= 1.0 else None)
    return output


def extract_chain_feature_rows(frames: list[ShotFrame]) -> list[dict[str, Any]]:
    timestamps = [frame.timestamp_ms for frame in frames]
    knee_velocity = derivative([frame.shooting_knee_angle for frame in frames], timestamps)
    hip_velocity = derivative([frame.shooting_hip_angle for frame in frames], timestamps)
    trunk_velocity = derivative([frame.trunk_tilt_proxy for frame in frames], timestamps)
    pelvis_y_velocity = derivative([frame.pelvis_y for frame in frames], timestamps)
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        scale = frame.body_scale or 1.0
        shoulder_pelvis_offset = None
        if frame.shoulder_y is not None and frame.pelvis_y is not None and scale > 1e-9:
            shoulder_pelvis_offset = (frame.shoulder_y - frame.pelvis_y) / scale
        rows.append(
            {
                "frame_index": frame.frame_index,
                "timestamp_ms": frame.timestamp_ms,
                "shooting_knee_angle": frame.shooting_knee_angle,
                "non_shooting_knee_angle": frame.non_shooting_knee_angle,
                "shooting_hip_angle": frame.shooting_hip_angle,
                "non_shooting_hip_angle": frame.non_shooting_hip_angle,
                "shooting_knee_angular_velocity": knee_velocity[index],
                "shooting_hip_angular_velocity": hip_velocity[index],
                "shooting_ankle_speed": frame.shooting_ankle_speed,
                "pelvis_vertical_velocity_proxy": pelvis_y_velocity[index],
                "trunk_tilt_proxy": frame.trunk_tilt_proxy,
                "trunk_tilt_velocity_proxy": trunk_velocity[index],
                "shoulder_center_speed": frame.shoulder_center_speed,
                "shoulder_pelvis_relative_offset": shoulder_pelvis_offset,
                "shooting_shoulder_angle": frame.shooting_shoulder_angle,
                "shooting_elbow_angle": frame.shooting_elbow_angle,
                "shooting_elbow_angular_velocity": frame.shooting_elbow_angular_velocity,
                "shooting_wrist_speed": frame.shooting_wrist_speed,
                "wrist_relative_shoulder_height": frame.wrist_relative_shoulder_height(),
                "wrist_relative_elbow_height": frame.wrist_relative_elbow_height(),
                "motion_energy_proxy": frame.motion_energy_proxy,
                "pose_detected": int(frame.pose_detected),
                "visibility_mean": frame.visibility_mean,
                "missing_ratio": frame.missing_ratio,
            }
        )
    return rows


def _peak_event(
    event_name: str,
    frames: list[ShotFrame],
    rows: list[dict[str, Any]],
    field: str,
    prefer_abs: bool = True,
    quality_field: str = "visibility_mean",
) -> ShotEvent:
    values = [finite_or_none(row.get(field)) for row in rows]
    finite_pairs = [(index, abs(value) if prefer_abs else value) for index, value in enumerate(values) if value is not None and isfinite(float(value))]
    if not finite_pairs:
        return ShotEvent(event_name, None, None, 0.0, field, "LOW")
    index, _ = max(finite_pairs, key=lambda item: item[1])
    first = frames[0].timestamp_ms if frames else 0
    last = frames[-1].timestamp_ms if frames else first
    duration = max(last - first, 1)
    visibility = finite_or_none(rows[index].get(quality_field))
    confidence = 0.8 if visibility is None or visibility >= 0.65 else 0.45
    return ShotEvent(
        event=event_name,
        timestamp_ms=frames[index].timestamp_ms,
        normalized_time_percent=(frames[index].timestamp_ms - first) / duration * 100.0,
        confidence=confidence,
        source_signal=field,
        data_quality="GOOD" if confidence >= 0.7 else "WARNING",
    )


def detect_chain_events(frames: list[ShotFrame], feature_rows: list[dict[str, Any]], shooting_side: str) -> list[ShotEvent]:
    side = validate_shooting_side(shooting_side)
    return [
        _peak_event("pelvis_upward_speed_peak", frames, feature_rows, "pelvis_vertical_velocity_proxy", prefer_abs=True),
        _peak_event(f"{side}_knee_extension_peak", frames, feature_rows, "shooting_knee_angular_velocity", prefer_abs=True),
        _peak_event(f"{side}_hip_extension_peak", frames, feature_rows, "shooting_hip_angular_velocity", prefer_abs=True),
        _peak_event("shoulder_elevation_peak", frames, feature_rows, "shooting_shoulder_angle", prefer_abs=True),
        _peak_event(f"{side}_elbow_extension_peak", frames, feature_rows, "shooting_elbow_angular_velocity", prefer_abs=True),
        _peak_event(f"{side}_wrist_speed_peak", frames, feature_rows, "shooting_wrist_speed", prefer_abs=False),
    ]

