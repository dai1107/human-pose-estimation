from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from .schema import ClipRange
from .session_loader import SessionData, filter_rows, load_session, parse_number, write_csv_rows


@dataclass(frozen=True)
class SessionClip:
    session: SessionData
    kinematics: list[dict[str, Any]]
    landmarks: list[dict[str, Any]]
    clip_range: ClipRange


def _session_id(session: SessionData) -> str:
    return str(session.metadata.get("session_id") or session.path.name)


def _range_from_rows(session: SessionData, rows: list[dict[str, Any]]) -> ClipRange:
    timestamps = [parse_number(row.get("timestamp_ms")) for row in rows]
    timestamps = [value for value in timestamps if isfinite(value)]
    frames = [parse_number(row.get("frame_index")) for row in rows]
    frames = [value for value in frames if isfinite(value)]
    start_ms = int(min(timestamps)) if timestamps else None
    end_ms = int(max(timestamps)) if timestamps else None
    return ClipRange(
        session_id=_session_id(session),
        start_ms=start_ms,
        end_ms=end_ms,
        start_frame=int(min(frames)) if frames else None,
        end_frame=int(max(frames)) if frames else None,
        duration_ms=int(end_ms - start_ms) if start_ms is not None and end_ms is not None else None,
    )


def add_clip_timestamps(rows: list[dict[str, Any]], start_ms: int | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    origin = float(start_ms) if start_ms is not None else None
    for row in rows:
        new_row = dict(row)
        timestamp = parse_number(row.get("timestamp_ms"))
        new_row["source_timestamp_ms"] = row.get("timestamp_ms", "")
        if origin is not None and isfinite(timestamp):
            new_row["clip_timestamp_ms"] = f"{timestamp - origin:.8g}"
        else:
            new_row["clip_timestamp_ms"] = ""
        result.append(new_row)
    return result


def clip_session(
    session_dir: str | Path,
    start_ms: int | None = None,
    end_ms: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> SessionClip:
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise ValueError("start_ms must be <= end_ms")
    if start_frame is not None and end_frame is not None and start_frame > end_frame:
        raise ValueError("start_frame must be <= end_frame")
    session = load_session(session_dir)
    kinematics = filter_rows(session.kinematics, start_ms, end_ms, start_frame, end_frame)
    landmarks = filter_rows(session.landmarks, start_ms, end_ms, start_frame, end_frame)
    if not kinematics:
        raise ValueError("clip does not contain kinematic frames")
    clip_range = _range_from_rows(session, kinematics)
    return SessionClip(
        session=session,
        kinematics=add_clip_timestamps(kinematics, clip_range.start_ms),
        landmarks=add_clip_timestamps(landmarks, clip_range.start_ms),
        clip_range=clip_range,
    )


def write_clip(output_dir: str | Path, clip: SessionClip) -> None:
    path = Path(output_dir)
    write_csv_rows(path / "clip_kinematics.csv", clip.kinematics)
    if clip.landmarks:
        write_csv_rows(path / "clip_landmarks.csv", clip.landmarks)

