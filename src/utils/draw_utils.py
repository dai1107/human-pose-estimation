from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - OpenCV ASCII fallback remains available.
    Image = None
    ImageDraw = None
    ImageFont = None

from src.backends.base import PoseResult
from src.biomechanics.hand_landmarks import SUPPLEMENTAL_FINGER_CONNECTIONS, SUPPLEMENTAL_FINGER_DISPLAY_INDICES
from src.biomechanics.types import LandmarkPoint
from src.realtime.feedback_engine import FeedbackState
from src.utils.metrics import RealtimeMetricsSnapshot
from hyrox.action_names import HYROX_ACTION_LABELS, HYROX_ACTION_OPTIONS


HAND_TIP_INDICES = frozenset({4, 8, 12, 16, 20})


@lru_cache(maxsize=1)
def _load_unicode_font() -> object | None:
    if ImageFont is None:
        return None
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), 18)
        except (OSError, ValueError):
            continue
    return None


def _put_unicode_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> bool:
    font = _load_unicode_font()
    if font is None or Image is None or ImageDraw is None:
        return False
    x, y = origin
    rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb_image)
    try:
        bbox = draw.textbbox((x, y), text, font=font, anchor="ls")
    except (TypeError, ValueError):
        bbox = draw.textbbox((x, y - 18), text, font=font)
    top_left = (max(0, bbox[0] - 5), max(0, bbox[1] - 5))
    bottom_right = (min(frame.shape[1] - 1, bbox[2] + 5), min(frame.shape[0] - 1, bbox[3] + 5))
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (20, 22, 24), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb_image)
    rgb_color = (color[2], color[1], color[0])
    try:
        draw.text((x, y), text, font=font, fill=rgb_color, anchor="ls")
    except (TypeError, ValueError):
        draw.text((x, y - 18), text, font=font, fill=rgb_color)
    frame[:] = cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)
    return True


def to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return (
        min(width - 1, max(0, int(round(x * width)))),
        min(height - 1, max(0, int(round(y * height)))),
    )


def draw_pose_result(frame: np.ndarray, result: PoseResult, min_confidence: float = 0.2) -> None:
    draw_pose_result_filtered(frame, result, min_confidence=min_confidence)


def draw_pose_result_filtered(
    frame: np.ndarray,
    result: PoseResult,
    *,
    min_confidence: float = 0.2,
    visible_names: set[str] | None = None,
    highlight_names: set[str] | None = None,
) -> None:
    if not result.success:
        return
    visible_names = visible_names or set()
    highlight_names = highlight_names or set()
    height, width = frame.shape[:2]
    for start, end in result.connections:
        if start >= len(result.keypoints) or end >= len(result.keypoints):
            continue
        a = result.keypoints[start]
        b = result.keypoints[end]
        if visible_names and (a.name not in visible_names or b.name not in visible_names):
            continue
        if a.confidence < min_confidence or b.confidence < min_confidence:
            continue
        cv2.line(frame, to_pixel(a.x, a.y, width, height), to_pixel(b.x, b.y, width, height), (80, 220, 120), 2, cv2.LINE_AA)
    for point in result.keypoints:
        if visible_names and point.name not in visible_names:
            continue
        if point.confidence < min_confidence:
            continue
        radius = 7 if point.name in highlight_names else 4
        color = (0, 170, 255) if point.name in highlight_names else (255, 210, 80)
        cv2.circle(frame, to_pixel(point.x, point.y, width, height), radius, color, -1, cv2.LINE_AA)


def draw_hand_landmarks(
    frame: np.ndarray,
    hands: Mapping[str, Sequence[LandmarkPoint]],
    *,
    min_confidence: float = 0.05,
) -> None:
    height, width = frame.shape[:2]
    side_colors = {
        "left": ((255, 190, 90), (255, 235, 160)),
        "right": ((90, 190, 255), (170, 235, 255)),
    }
    for side, landmarks in sorted(hands.items()):
        line_color, point_color = side_colors.get(side, ((170, 220, 170), (220, 255, 220)))
        for start, end in SUPPLEMENTAL_FINGER_CONNECTIONS:
            if start >= len(landmarks) or end >= len(landmarks):
                continue
            first = landmarks[start]
            second = landmarks[end]
            if not first.is_usable(min_confidence, min_confidence) or not second.is_usable(min_confidence, min_confidence):
                continue
            cv2.line(frame, to_pixel(first.x, first.y, width, height), to_pixel(second.x, second.y, width, height), line_color, 2, cv2.LINE_AA)
        for index in sorted(SUPPLEMENTAL_FINGER_DISPLAY_INDICES):
            if index >= len(landmarks):
                continue
            point = landmarks[index]
            if not point.is_usable(min_confidence, min_confidence):
                continue
            radius = 5 if index in HAND_TIP_INDICES else 3
            cv2.circle(frame, to_pixel(point.x, point.y, width, height), radius, point_color, -1, cv2.LINE_AA)


def draw_bbox(frame: np.ndarray, bbox: tuple[float, float, float, float] | None) -> None:
    if bbox is None:
        return
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, to_pixel(x1, y1, width, height), to_pixel(x2, y2, width, height), (0, 190, 255), 2, cv2.LINE_AA)


def put_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int] = (245, 245, 245)) -> None:
    if any(ord(character) > 127 for character in text) and _put_unicode_text(frame, text, origin, color):
        return
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


def format_hyrox_debug_lines(
    features: Mapping[str, float | None] | None,
    *,
    has_pose: bool,
) -> list[str]:
    if not has_pose or not features:
        return ["No pose"]

    def fmt(name: str, value: float | None, decimals: int = 1) -> str:
        if value is None or not np.isfinite(value):
            return f"{name}: N/A"
        return f"{name}: {value:.{decimals}f}"

    return [
        fmt("visible", features.get("visible_score"), decimals=2),
        fmt("lknee", features.get("left_knee_angle")),
        fmt("rknee", features.get("right_knee_angle")),
        fmt("lhip", features.get("left_hip_angle")),
        fmt("rhip", features.get("right_hip_angle")),
        fmt("torso", features.get("torso_angle")),
    ]


def draw_hyrox_debug_overlay(
    frame: np.ndarray,
    features: Mapping[str, float | None] | None,
    *,
    has_pose: bool,
    origin: tuple[int, int] = (250, 26),
) -> None:
    lines = format_hyrox_debug_lines(features, has_pose=has_pose)
    for row, line in enumerate(lines):
        color = (245, 245, 245)
        if line == "No pose":
            color = (0, 190, 255)
        put_text(frame, line, (origin[0], origin[1] + row * 27), color)


def _feedback_message_parts(message: object) -> tuple[str, str]:
    if isinstance(message, Mapping):
        level = str(message.get("level", "info"))
        text = str(message.get("text", ""))
        return level, text
    return str(getattr(message, "level", "info")), str(getattr(message, "text", ""))


def format_hyrox_action_lines(state: Mapping[str, object] | None) -> list[tuple[str, tuple[int, int, int]]]:
    if not state:
        return [("action: OFF", (0, 190, 255))]

    debug = state.get("debug", {}) if isinstance(state, Mapping) else {}
    config_name = debug.get("config_name") if isinstance(debug, Mapping) else None
    camera_view = debug.get("camera_view") if isinstance(debug, Mapping) else None
    view_profile = debug.get("view_profile") if isinstance(debug, Mapping) else None
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"action: {state.get('action', 'unknown')}", (245, 245, 245)),
        (f"cfg: {config_name or 'default'}", (180, 220, 255)),
        (f"view: {camera_view or 'unknown'} / {view_profile or 'unknown'}", (180, 220, 255)),
        (f"phase: {state.get('phase', 'unknown')}", (245, 245, 245)),
        (f"reps: {state.get('rep_count', 0)}", (245, 245, 245)),
    ]
    feedback_messages = state.get("feedback_messages")
    if isinstance(feedback_messages, Sequence):
        for message in feedback_messages[:2]:
            level, text = _feedback_message_parts(message)
            color = (80, 230, 120)
            if level == "warn":
                color = (0, 190, 255)
            elif level == "error":
                color = (60, 80, 255)
            lines.append((f"tip: {text}", color))
    if len(lines) == 5:
        lines.append(("tip: 动作稳定", (80, 230, 120)))
    return lines


def draw_hyrox_action_overlay(
    frame: np.ndarray,
    state: Mapping[str, object] | None,
    *,
    origin: tuple[int, int] = (250, 26),
) -> None:
    for row, (line, color) in enumerate(format_hyrox_action_lines(state)):
        put_text(frame, line, (origin[0], origin[1] + row * 27), color)


def format_hyrox_action_selector_lines(current_action: str) -> list[tuple[str, tuple[int, int, int]]]:
    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("选择动作 / Select action (0-8)", (80, 230, 255)),
    ]
    for index, action_name in enumerate(HYROX_ACTION_OPTIONS):
        marker = ">" if action_name == current_action else " "
        color = (80, 230, 120) if action_name == current_action else (245, 245, 245)
        lines.append((f"{marker} {index}: {HYROX_ACTION_LABELS[action_name]}", color))
    lines.append(("A/ESC: 取消   N: 快速切换下一个", (180, 220, 255)))
    return lines


def draw_hyrox_action_selector(
    frame: np.ndarray,
    current_action: str,
    *,
    origin: tuple[int, int] = (24, 48),
) -> None:
    lines = format_hyrox_action_selector_lines(current_action)
    panel_width = min(max(430, frame.shape[1] // 2), max(1, frame.shape[1] - origin[0] - 8))
    panel_height = min(len(lines) * 29 + 18, max(1, frame.shape[0] - origin[1] + 18))
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (origin[0] - 12, origin[1] - 28),
        (origin[0] - 12 + panel_width, origin[1] - 28 + panel_height),
        (12, 16, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    for row, (line, color) in enumerate(lines):
        put_text(frame, line, (origin[0], origin[1] + row * 27), color)
