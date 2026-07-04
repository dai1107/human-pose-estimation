from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from src.reference.session_loader import filter_rows, load_session, parse_number

from .chain_features import build_shot_frames_from_session
from .schema import ShotClip


@dataclass(frozen=True)
class ShotCandidate:
    start_ms: int
    end_ms: int
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms, "confidence": self.confidence, "reason": self.reason}


def clip_shot_session(
    session_dir: str | Path,
    shooting_side: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> ShotClip:
    session = load_session(session_dir)
    kinematics = filter_rows(session.kinematics, start_ms=start_ms, end_ms=end_ms, start_frame=start_frame, end_frame=end_frame)
    landmarks = filter_rows(session.landmarks, start_ms=start_ms, end_ms=end_ms, start_frame=start_frame, end_frame=end_frame)
    if not kinematics:
        raise ValueError("shot clip does not contain kinematic frames")
    temp_session = type(session)(session.path, session.session_id, session.metadata, kinematics, landmarks)
    frames = build_shot_frames_from_session(temp_session, shooting_side)
    timestamps = [parse_number(row.get("timestamp_ms")) for row in kinematics]
    finite_ts = [int(value) for value in timestamps if isfinite(value)]
    frame_values = [parse_number(row.get("frame_index")) for row in kinematics]
    finite_frames = [int(value) for value in frame_values if isfinite(value)]
    return ShotClip(
        session_id=session.session_id,
        start_ms=min(finite_ts),
        end_ms=max(finite_ts),
        start_frame=min(finite_frames) if finite_frames else None,
        end_frame=max(finite_frames) if finite_frames else None,
        frames=frames,
        kinematics_rows=kinematics,
        landmark_rows=landmarks,
    )


def detect_shot_candidates(session_dir: str | Path, shooting_side: str, min_duration_ms: int = 700) -> list[ShotCandidate]:
    session = load_session(session_dir)
    frames = build_shot_frames_from_session(session, shooting_side)
    if len(frames) < 3:
        return []
    wrist_speed = np.array([frame.shooting_wrist_speed if frame.shooting_wrist_speed is not None else 0.0 for frame in frames], dtype=float)
    energy = np.array([frame.motion_energy_proxy if frame.motion_energy_proxy is not None else 0.0 for frame in frames], dtype=float)
    signal = wrist_speed + 0.5 * energy
    if not np.isfinite(signal).any() or float(np.max(signal)) <= 0:
        return []
    threshold = float(np.percentile(signal, 75))
    active = signal >= threshold
    candidates: list[ShotCandidate] = []
    start_index: int | None = None
    for index, is_active in enumerate(active.tolist() + [False]):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            end_index = max(start_index, index - 1)
            start_ms = max(frames[0].timestamp_ms, frames[start_index].timestamp_ms - 400)
            end_ms = min(frames[-1].timestamp_ms, frames[end_index].timestamp_ms + 500)
            if end_ms - start_ms >= min_duration_ms:
                candidates.append(ShotCandidate(start_ms, end_ms, 0.55, "wrist speed and motion energy candidate"))
            start_index = None
    return candidates

