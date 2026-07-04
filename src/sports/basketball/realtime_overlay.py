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
    return f"{number:.2f}{suffix}" if isfinite(number) else "N/A"


def draw_basketball_overlay(frame: np.ndarray, metrics: dict[str, Any], origin: tuple[int, int] = (14, 500)) -> None:
    side = str(metrics.get("shooting_side", "right")).upper()
    lines = [
        "MODE: BASKETBALL",
        f"SHOT TYPE: {str(metrics.get('shot_type', 'set_shot')).upper()}",
        f"SHOOTING SIDE: {side}",
        f"VIEW: {str(metrics.get('camera_view', 'UNKNOWN')).upper()}",
        f"PHASE: {metrics.get('phase', 'IDLE')}",
        f"{side} KNEE: {_fmt(metrics.get('knee_angle'), ' deg')}",
        f"{side} ELBOW: {_fmt(metrics.get('elbow_angle'), ' deg')}",
        f"PELVIS SPEED: {_fmt(metrics.get('pelvis_speed'), ' proxy-units/s')}",
        f"{side} WRIST SPEED: {_fmt(metrics.get('wrist_speed'), ' proxy-units/s')}",
        f"RELEASE PROXY: {metrics.get('release_proxy', 'PENDING')}",
        f"DATA QUALITY: {metrics.get('data_quality', 'N/A')}",
    ]
    x, y = origin
    width = 390
    line_height = 23
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 20), (min(frame.shape[1] - 1, x + width), min(frame.shape[0] - 1, y + line_height * len(lines) + 10)), (16, 18, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, text in enumerate(lines):
        color = (235, 235, 235)
        if text.startswith("PHASE"):
            color = (80, 220, 150)
        if text.startswith("RELEASE") and "PENDING" not in text:
            color = (80, 230, 120)
        cv2.putText(frame, text, (x, y + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 1, cv2.LINE_AA)

