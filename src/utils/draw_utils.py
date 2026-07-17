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
    line_color: tuple[int, int, int] = (80, 220, 120),
    point_color: tuple[int, int, int] = (255, 210, 80),
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
        cv2.line(frame, to_pixel(a.x, a.y, width, height), to_pixel(b.x, b.y, width, height), line_color, 2, cv2.LINE_AA)
    for point in result.keypoints:
        if visible_names and point.name not in visible_names:
            continue
        if point.confidence < min_confidence:
            continue
        radius = 7 if point.name in highlight_names else 4
        color = (0, 170, 255) if point.name in highlight_names else point_color
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
    features: Mapping[str, object] | None,
    *,
    has_pose: bool,
    action_state: Mapping[str, object] | None = None,
) -> list[str]:
    if not has_pose or not features:
        return ["No pose"]

    def fmt(name: str, value: float | None, decimals: int = 1) -> str:
        if value is None or not np.isfinite(value):
            return f"{name}: N/A"
        return f"{name}: {value:.{decimals}f}"

    lines = [
        fmt("visible", features.get("visible_score"), decimals=2),
        fmt("lknee", features.get("left_knee_angle")),
        fmt("rknee", features.get("right_knee_angle")),
        fmt("lhip", features.get("left_hip_angle")),
        fmt("rhip", features.get("right_hip_angle")),
    ]
    if "floor_reference_status" in features:
        lines.extend(
            [
                f"floor: {features.get('floor_reference_status', 'UNSURE')} / {features.get('floor_reference_source', 'none')}",
                fmt("floor_y", features.get("floor_y"), decimals=3),
                fmt("floor_conf", features.get("floor_reference_confidence"), decimals=2),
                fmt("body_h", features.get("body_height_reference"), decimals=3),
            ]
        )
    lines.append(fmt("torso", features.get("torso_angle")))

    if not isinstance(action_state, Mapping):
        return lines
    debug = action_state.get("debug")
    debug = debug if isinstance(debug, Mapping) else {}
    action = str(action_state.get("action", "unknown"))
    phase = str(action_state.get("phase", "unknown"))
    candidate_count = int(action_state.get("candidate_count", 0) or 0)
    lines.append(f"{action.upper()} CANDIDATE #{candidate_count} / {phase}")

    contacts = debug.get("contacts")
    contacts = contacts if isinstance(contacts, Mapping) else {}
    knee_contact = contacts.get("knee")
    chest_contact = contacts.get("chest_proxy")
    knee_contact = knee_contact if isinstance(knee_contact, Mapping) else {}
    chest_contact = chest_contact if isinstance(chest_contact, Mapping) else {}
    lines.append(
        "contact knee/chest: "
        f"{knee_contact.get('status', 'UNSURE')} / "
        f"{chest_contact.get('status', 'UNSURE')}"
    )
    lines.append(
        "height knee/chest: "
        f"{_format_optional_ratio(knee_contact.get('surface_height_ratio'))} / "
        f"{_format_optional_ratio(chest_contact.get('surface_height_ratio'))}"
    )

    foot_events = debug.get("foot_events")
    foot_events = foot_events if isinstance(foot_events, Mapping) else {}
    left_foot = foot_events.get("left")
    right_foot = foot_events.get("right")
    sync = foot_events.get("sync")
    stagger = foot_events.get("stagger")
    left_foot = left_foot if isinstance(left_foot, Mapping) else {}
    right_foot = right_foot if isinstance(right_foot, Mapping) else {}
    sync = sync if isinstance(sync, Mapping) else {}
    stagger = stagger if isinstance(stagger, Mapping) else {}
    lines.append(
        "feet L/R: "
        f"{left_foot.get('state', 'NOT_OBSERVABLE')} / "
        f"{right_foot.get('state', 'NOT_OBSERVABLE')}"
    )
    lines.append(
        "sync takeoff/landing: "
        f"{_format_optional_ms(sync.get('takeoff_delta_ms'))} / "
        f"{_format_optional_ms(sync.get('landing_delta_ms'))}"
    )
    lines.append(
        "foot stagger: "
        f"{stagger.get('status', 'UNSURE')} "
        f"{_format_optional_ratio(stagger.get('stagger_ratio'))}"
    )

    decision = action_state.get("last_rep_decision")
    decision = decision if isinstance(decision, Mapping) else None
    if decision is None:
        lines.append("RULES: awaiting completed candidate")
        lines.append("RESULT: PENDING")
    else:
        rules = decision.get("rules")
        if isinstance(rules, Sequence) and not isinstance(rules, (str, bytes)):
            for rule in rules:
                if not isinstance(rule, Mapping):
                    continue
                status = str(rule.get("status", "UNSURE"))
                rule_id = str(rule.get("rule_id", "unknown")).upper()
                confidence = _format_optional_confidence(rule.get("confidence"))
                value = _format_rule_value(rule.get("value"))
                suffix = f" {value}" if value else ""
                lines.append(f"{status:<6} {rule_id} {confidence}{suffix}")
        lines.append(f"RESULT: {decision.get('status', 'UNSURE')}")

    feedback_messages = action_state.get("feedback_messages")
    if isinstance(feedback_messages, Sequence) and not isinstance(
        feedback_messages,
        (str, bytes),
    ):
        for message in feedback_messages[:2]:
            _, text = _feedback_message_parts(message)
            if text:
                lines.append(f"tip: {text}")
    return lines


def _finite_float(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if np.isfinite(resolved) else None


def _format_optional_ratio(value: object) -> str:
    resolved = _finite_float(value)
    return "N/A" if resolved is None else f"{resolved:.3f}"


def _format_optional_ms(value: object) -> str:
    resolved = _finite_float(value)
    return "N/A" if resolved is None else f"{resolved:.0f}ms"


def _format_optional_confidence(value: object) -> str:
    resolved = _finite_float(value)
    return "N/A" if resolved is None else f"{resolved:.2f}"


def _format_rule_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).upper()
    resolved = _finite_float(value)
    if resolved is not None:
        return f"{resolved:.2f}"
    return str(value)


def draw_hyrox_debug_overlay(
    frame: np.ndarray,
    features: Mapping[str, object] | None,
    *,
    has_pose: bool,
    action_state: Mapping[str, object] | None = None,
    origin: tuple[int, int] = (250, 26),
) -> int:
    lines = format_hyrox_debug_lines(
        features,
        has_pose=has_pose,
        action_state=action_state,
    )
    for row, line in enumerate(lines):
        color = (245, 245, 245)
        if line == "No pose":
            color = (0, 190, 255)
        elif line.startswith("PASS") or line == "RESULT: VALID":
            color = (80, 230, 120)
        elif line.startswith("FAIL") or line == "RESULT: NO_REP":
            color = (60, 80, 255)
        elif line.startswith("UNSURE") or line == "RESULT: UNSURE":
            color = (0, 190, 255)
        put_text(frame, line, (origin[0], origin[1] + row * 21), color)
    if features:
        try:
            x1 = float(features.get("floor_line_x1"))
            y1 = float(features.get("floor_line_y1"))
            x2 = float(features.get("floor_line_x2"))
            y2 = float(features.get("floor_line_y2"))
        except (TypeError, ValueError, OverflowError):
            x1 = y1 = x2 = y2 = float("nan")
        if all(np.isfinite(value) for value in (x1, y1, x2, y2)):
            height, width = frame.shape[:2]
            status = str(features.get("floor_reference_status", "UNSURE"))
            color = (80, 230, 120) if status == "READY" else (0, 190, 255)
            cv2.line(
                frame,
                (int(round(x1 * width)), int(round(y1 * height))),
                (int(round(x2 * width)), int(round(y2 * height))),
                color,
                2,
                cv2.LINE_AA,
            )
    _draw_contact_surface_points(frame, action_state)
    return len(lines)


def _draw_contact_surface_points(
    frame: np.ndarray,
    action_state: Mapping[str, object] | None,
) -> None:
    if not isinstance(action_state, Mapping):
        return
    debug = action_state.get("debug")
    if not isinstance(debug, Mapping):
        return
    contacts = debug.get("contacts")
    if not isinstance(contacts, Mapping):
        return
    height, width = frame.shape[:2]
    styles = {
        "knee": ((255, 80, 220), "K"),
        "chest_proxy": ((255, 210, 80), "C"),
    }
    for name, (color, label) in styles.items():
        result = contacts.get(name)
        if not isinstance(result, Mapping):
            continue
        point = result.get("surface_point")
        if not isinstance(point, Mapping):
            continue
        x = _finite_float(point.get("x"))
        y = _finite_float(point.get("y"))
        if x is None or y is None:
            continue
        pixel = to_pixel(x, y, width, height)
        cv2.circle(frame, pixel, 7, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (pixel[0] + 8, pixel[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


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
