from __future__ import annotations

from math import isfinite
from typing import Any

import cv2
import numpy as np


def _fmt(value: Any, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not isfinite(number):
        return "N/A"
    return f"{number:.1f}{suffix}"


def draw_squat_overlay(frame: np.ndarray, metrics: dict[str, Any], origin: tuple[int, int] = (14, 460)) -> None:
    lines = [
        "MODE: SQUAT",
        f"VIEW: {str(metrics.get('camera_view', 'UNKNOWN')).upper()}",
        f"CALIBRATION: {metrics.get('calibration_status', 'N/A')}",
        f"STATE: {metrics.get('state', 'IDLE')}",
        f"REPS: {metrics.get('rep_count', 0)}",
        f"LEFT KNEE: {_fmt(metrics.get('left_knee_angle'), ' deg')}",
        f"RIGHT KNEE: {_fmt(metrics.get('right_knee_angle'), ' deg')}",
        f"TRUNK TILT: {_fmt(metrics.get('trunk_tilt_proxy'), ' deg')}",
        f"PELVIS DISPLACEMENT: {_fmt(metrics.get('pelvis_displacement'), ' body-scale')}",
        f"DATA QUALITY: {metrics.get('data_quality', 'N/A')}",
    ]
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 1
    line_height = 24
    width = 360
    height = line_height * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 20), (min(frame.shape[1] - 1, x + width), min(frame.shape[0] - 1, y + height)), (16, 18, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, text in enumerate(lines):
        color = (235, 235, 235)
        if text.startswith("CALIBRATION") and "FAIL" in text:
            color = (60, 80, 255)
        elif text.startswith("CALIBRATION") and "WARNING" in text:
            color = (0, 190, 255)
        elif text.startswith("STATE"):
            color = (80, 220, 150)
        cv2.putText(frame, text, (x, y + index * line_height), font, scale, color, thickness, cv2.LINE_AA)

