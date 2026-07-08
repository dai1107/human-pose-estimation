from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np

from src.backends.base import PoseResult
from src.realtime.feedback_engine import FeedbackState
from src.utils.metrics import RealtimeMetricsSnapshot


def to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return (
        min(width - 1, max(0, int(round(x * width)))),
        min(height - 1, max(0, int(round(y * height)))),
    )


def draw_pose_result(frame: np.ndarray, result: PoseResult, min_confidence: float = 0.2) -> None:
    if not result.success:
        return
    height, width = frame.shape[:2]
    for start, end in result.connections:
        if start >= len(result.keypoints) or end >= len(result.keypoints):
            continue
        a = result.keypoints[start]
        b = result.keypoints[end]
        if a.confidence < min_confidence or b.confidence < min_confidence:
            continue
        cv2.line(frame, to_pixel(a.x, a.y, width, height), to_pixel(b.x, b.y, width, height), (80, 220, 120), 2, cv2.LINE_AA)
    for point in result.keypoints:
        if point.confidence < min_confidence:
            continue
        cv2.circle(frame, to_pixel(point.x, point.y, width, height), 4, (255, 210, 80), -1, cv2.LINE_AA)


def draw_bbox(frame: np.ndarray, bbox: tuple[float, float, float, float] | None) -> None:
    if bbox is None:
        return
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, to_pixel(x1, y1, width, height), to_pixel(x2, y2, width, height), (0, 190, 255), 2, cv2.LINE_AA)


def put_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int] = (245, 245, 245)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    top_left = (max(0, x - 5), max(0, y - text_height - 6))
    bottom_right = (min(frame.shape[1] - 1, x + text_width + 5), min(frame.shape[0] - 1, y + baseline + 6))
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (20, 22, 24), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_realtime_overlay(
    frame: np.ndarray,
    *,
    backend: str,
    fusion: str,
    person_detector: str,
    detector_every_n: int,
    smoothing: str,
    input_mode: str,
    result: PoseResult,
    metrics: RealtimeMetricsSnapshot,
    feedback: FeedbackState,
    recording: bool,
    raw_recording: bool,
    angles: Mapping[str, float | None] | None = None,
    status_message: str = "",
) -> None:
    del angles
    lines = [
        f"backend: {backend}",
        f"fusion: {fusion}",
        f"person_detector: {person_detector}",
        f"detector_every_n: {detector_every_n}",
        f"smoothing: {smoothing}",
        f"input: {input_mode}",
        f"FPS: {metrics.realtime_fps:4.1f}",
        f"inference: {metrics.inference_time_ms:4.1f} ms",
        f"latency: {metrics.end_to_end_latency_ms:4.1f} ms",
        f"success: {'YES' if result.success else 'NO'}",
        f"keypoints: {result.num_keypoints}",
        f"avg_conf: {metrics.avg_keypoint_confidence:.2f}",
        f"feedback: {feedback.message}",
        f"record: {'ON' if recording else 'OFF'} raw: {'ON' if raw_recording else 'OFF'}",
    ]
    if result.extra.get("stabilized_hold"):
        lines.append(f"tracking: HOLD {result.extra.get('hold_frames', 0)}")
    guarded = result.extra.get("occlusion_guarded_keypoints") or ()
    if guarded:
        lines.append(f"occlusion_guard: {len(guarded)}")
    if status_message:
        lines.append(f"status: {status_message}")
    for row, line in enumerate(lines):
        color = (245, 245, 245)
        if line.startswith("success:"):
            color = (80, 230, 120) if result.success else (60, 80, 255)
        if line.startswith("feedback:") and feedback.message != "Tracking stable":
            color = (0, 190, 255)
        if line.startswith("record:") and (recording or raw_recording):
            color = (70, 90, 255)
        if line.startswith("tracking:") or line.startswith("occlusion_guard:"):
            color = (0, 190, 255)
        if line.startswith("status:"):
            color = (0, 190, 255)
        put_text(frame, line, (14, 26 + row * 27), color)
