from __future__ import annotations

from math import isfinite
from typing import Mapping

import cv2
import numpy as np


def _format_metric(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if not isfinite(value):
        return "N/A"
    return f"{value:.1f}{suffix}"


def draw_metrics_overlay(
    frame: np.ndarray,
    metrics: Mapping[str, object],
    origin: tuple[int, int] = (14, 214),
) -> None:
    lines = [
        f"POSE: {'YES' if metrics.get('pose_detected') else 'NO'}",
        f"FPS: {_format_metric(metrics.get('fps'))}",
        f"SESSION: {metrics.get('session_state', 'IDLE')}",
        f"MIRROR: {'ON' if metrics.get('mirror') else 'OFF'}",
        f"R ELBOW: {_format_metric(metrics.get('right_elbow_angle'), ' deg')}",
        f"R KNEE: {_format_metric(metrics.get('right_knee_angle'), ' deg')}",
        f"R WRIST SPEED: {_format_metric(metrics.get('right_wrist_speed'))}",
        f"PELVIS SPEED: {_format_metric(metrics.get('pelvis_speed'))}",
        f"ENERGY: {_format_metric(metrics.get('motion_energy_proxy'))}",
    ]

    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 24
    width = 330
    height = line_height * len(lines) + 14

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 20), (min(frame.shape[1] - 1, x + width), min(frame.shape[0] - 1, y + height)), (18, 22, 24), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    for index, text in enumerate(lines):
        color = (245, 245, 245)
        if text.startswith("POSE:"):
            color = (80, 230, 120) if metrics.get("pose_detected") else (70, 90, 255)
        if text.startswith("SESSION:") and metrics.get("session_state") == "RECORDING":
            color = (70, 90, 255)
        cv2.putText(frame, text, (x, y + index * line_height), font, scale, color, thickness, cv2.LINE_AA)

