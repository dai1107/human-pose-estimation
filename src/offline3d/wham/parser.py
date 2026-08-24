from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..base import Offline3DFrame, Offline3DResult, parse_joints


def _sequence(value: Any) -> Sequence[Any] | None:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else None


def parse_wham_payload(payload: Mapping[str, Any]) -> Offline3DResult:
    native_frames = payload.get("frames") or payload.get("results") or []
    if not isinstance(native_frames, list):
        raise ValueError("WHAM output must contain a frames/results array")
    joint_names = payload.get("joint_names")
    if not isinstance(joint_names, list):
        joint_names = None
    fps = float(payload.get("source_fps") or payload.get("fps") or 0.0)
    frames: list[Offline3DFrame] = []
    for position, raw in enumerate(native_frames):
        if not isinstance(raw, Mapping):
            continue
        frame_index_value = raw.get("frame_index", raw.get("frame", position))
        frame_index = int(frame_index_value) if frame_index_value is not None else None
        timestamp = raw.get("timestamp_ms", raw.get("frame_timestamp_ms"))
        if timestamp is None:
            timestamp = frame_index * 1000.0 / fps if fps > 0 and frame_index is not None else position
        known = {
            "timestamp_ms", "frame_timestamp_ms", "frame_index", "frame",
            "joints_3d", "joints", "smpl_pose", "pose", "body_orientation",
            "root_orientation", "body_translation", "translation", "camera_motion",
            "global_trajectory", "trajectory", "confidence",
        }
        frames.append(
            Offline3DFrame(
                timestamp_ms=float(timestamp),
                frame_index=frame_index,
                joints_3d=parse_joints(
                    raw.get("joints_3d", raw.get("joints")), joint_names
                ),
                smpl_pose=list(_sequence(raw.get("smpl_pose", raw.get("pose"))) or ()),
                body_orientation=list(_sequence(raw.get("body_orientation", raw.get("root_orientation"))) or ()),
                body_translation=list(_sequence(raw.get("body_translation", raw.get("translation"))) or ()),
                camera_motion=list(_sequence(raw.get("camera_motion")) or ()),
                global_trajectory=list(_sequence(raw.get("global_trajectory", raw.get("trajectory"))) or ()),
                confidence=(None if raw.get("confidence") is None else float(raw["confidence"])),
                extra={key: value for key, value in raw.items() if key not in known},
            )
        )
    frames.sort(key=lambda frame: frame.timestamp_ms)
    metadata = dict(payload.get("metadata") or {})
    metadata["native_schema"] = str(payload.get("schema_version", "unknown"))
    return Offline3DResult(
        backend="wham",
        status="COMPLETED",
        reference_source=str(payload.get("reference_source", "WHAM reconstructed 3D")),
        frames=frames,
        coordinate_system=str(payload.get("coordinate_system", "wham_native")),
        angle_definition=str(payload.get("angle_definition", "wham_native_joints_not_legacy_3point")),
        trajectory_confidence=(None if payload.get("trajectory_confidence") is None else float(payload["trajectory_confidence"])),
        metadata=metadata,
        warnings=[str(value) for value in payload.get("warnings", [])],
    )


def parse_wham_file(path: str | Path) -> Offline3DResult:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("WHAM output root must be a JSON object")
    return parse_wham_payload(payload)
