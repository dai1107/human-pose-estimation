from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np

from .schema import ShotFrame


def compute_arm_alignment_proxy(frames: list[ShotFrame], camera_view: str, allow_side_view: bool = False) -> dict[str, Any]:
    if camera_view == "side" and not allow_side_view:
        return {
            "computed": False,
            "data_quality": "LOW",
            "reason": "side view does not support high-confidence lateral arm alignment proxy",
            "metrics": {},
        }
    if camera_view == "unknown":
        return {"computed": False, "data_quality": "LOW", "reason": "camera view unknown", "metrics": {}}
    rows = [
        frame
        for frame in frames
        if frame.shooting_shoulder_x is not None and frame.shooting_elbow_x is not None and frame.shooting_wrist_x is not None and frame.body_scale
    ]
    if len(rows) < 2:
        return {"computed": False, "data_quality": "LOW", "reason": "insufficient arm keypoints", "metrics": {}}
    deviations: list[float] = []
    elbow_offsets: list[float] = []
    wrist_offsets: list[float] = []
    wrist_path: list[float] = []
    for frame in rows:
        scale = frame.body_scale or 1.0
        shoulder_x = float(frame.shooting_shoulder_x)
        elbow_x = float(frame.shooting_elbow_x)
        wrist_x = float(frame.shooting_wrist_x)
        line_mid = (shoulder_x + wrist_x) / 2.0
        deviations.append(abs(elbow_x - line_mid) / scale)
        elbow_offsets.append(abs(elbow_x - shoulder_x) / scale)
        wrist_offsets.append(abs(wrist_x - shoulder_x) / scale)
        wrist_path.append(wrist_x / scale)
    smoothness = float(np.std(np.diff(wrist_path))) if len(wrist_path) >= 3 else 0.0
    return {
        "computed": True,
        "data_quality": "MEDIUM",
        "reason": "camera-plane geometry proxy only",
        "metrics": {
            "shoulder_elbow_wrist_path_deviation": float(np.mean(deviations)),
            "elbow_lateral_offset_proxy": float(max(elbow_offsets)),
            "wrist_lateral_offset_proxy": float(max(wrist_offsets)),
            "arm_path_smoothness": smoothness,
        },
    }

