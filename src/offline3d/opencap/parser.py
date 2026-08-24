from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..base import Offline3DFrame, Offline3DResult, parse_joints


def _sequence(value: Any) -> list[Any]:
    return (
        list(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else []
    )


def parse_opencap_payload(payload: Mapping[str, Any]) -> Offline3DResult:
    raw_frames = payload.get("frames") or payload.get("kinematics") or []
    if not isinstance(raw_frames, list):
        raise ValueError("OpenCap output must contain a frames/kinematics array")
    fps = float(payload.get("source_fps") or payload.get("fps") or 0.0)
    frames: list[Offline3DFrame] = []
    for position, raw in enumerate(raw_frames):
        if not isinstance(raw, Mapping):
            continue
        frame_index_value = raw.get("frame_index", raw.get("frame", position))
        frame_index = int(frame_index_value) if frame_index_value is not None else None
        timestamp = raw.get("timestamp_ms", raw.get("source_timestamp_ms"))
        if timestamp is None:
            timestamp = frame_index * 1000.0 / fps if fps > 0 and frame_index is not None else position
        kinematics = raw.get("joint_kinematics", raw.get("coordinates", {}))
        pelvis_motion = raw.get("pelvis_motion", raw.get("pelvis", {}))
        extra = {
            "joint_kinematics": dict(kinematics) if isinstance(kinematics, Mapping) else {},
            "pelvis_motion": dict(pelvis_motion) if isinstance(pelvis_motion, Mapping) else {},
            "opensim_ik": dict(raw.get("opensim_ik") or {}),
        }
        frames.append(
            Offline3DFrame(
                timestamp_ms=float(timestamp),
                frame_index=frame_index,
                joints_3d=parse_joints(raw.get("joints_3d")),
                body_orientation=_sequence(raw.get("body_orientation")),
                body_translation=_sequence(raw.get("body_translation")),
                global_trajectory=_sequence(raw.get("global_trajectory")),
                confidence=(None if raw.get("confidence") is None else float(raw["confidence"])),
                extra=extra,
            )
        )
    frames.sort(key=lambda frame: frame.timestamp_ms)
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "native_schema": str(payload.get("schema_version", "unknown")),
            "biomechanical_model": str(payload.get("biomechanical_model", "OpenSim")),
            "optimization_level": str(payload.get("optimization_level", "OpenCap-style external")),
            "formal_rule_replacement_allowed": False,
            "absolute_angle_mae_allowed": False,
        }
    )
    return Offline3DResult(
        backend="opencap",
        status="COMPLETED",
        reference_source=str(payload.get("reference_source", "OpenCap Monocular / OpenSim IK reference")),
        frames=frames,
        coordinate_system=str(payload.get("coordinate_system", "opensim_model_coordinates")),
        angle_definition=str(payload.get("angle_definition", "OpenSim joint coordinates; not MediaPipe three-point angles")),
        trajectory_confidence=(None if payload.get("trajectory_confidence") is None else float(payload["trajectory_confidence"])),
        metadata=metadata,
        warnings=[str(value) for value in payload.get("warnings", [])],
    )


def parse_opencap_file(path: str | Path) -> Offline3DResult:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("OpenCap output root must be a JSON object")
    return parse_opencap_payload(payload)
