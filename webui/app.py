from __future__ import annotations

import hmac
import hashlib
import csv
import io
import json
import logging
import math
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
from flask import Flask, Response, g, jsonify, redirect, render_template, request
from flask_sock import Sock
from simple_websocket.errors import ConnectionClosed
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from hyrox.view_policy import CAMERA_VIEWS
from src.backends.mediapipe_backend import MediaPipeBackend
from src.backends.yolo_guided_mediapipe_backend import YoloGuidedMediaPipeBackend
from src.backends.yolo_pose_backend import YoloPoseBackend
from src.backends.yolo_rtmw_backend import YoloRtmwWholeBodyBackend
from src.utils.backend_policy import resolve_backend_choice
from src.utils.device import resolve_torch_device
from src.utils.draw_utils import draw_hyrox_action_overlay, draw_pose_result_filtered, to_pixel
from src.utils.smoothing import KeypointSmoother
from src.paths import installation_root, runtime_output_root
from src.output_schema import artifact_metadata, versioned_csv_columns, versioned_csv_row
from webui.analysis import RepVoiceFeedbackTracker, assess_action, enrich_report, official_rules_for, render_text_report, standards_for, visible_feedback
from webui.hands import (
    WebHandOverlay,
    draw_hand_overlay,
    hand_overlay_visible,
    rtmw_hand_detections,
    serialize_hand_overlay,
)
from webui.realtime import (
    BrowserSession,
    RealtimePoseSession,
    RealtimeProtocolError,
    SessionCapacityError,
    SessionManager,
    validate_manual_floor_points,
)


PROJECT_ROOT = installation_root()
OUTPUT_ROOT = runtime_output_root()
UPLOAD_DIR = OUTPUT_ROOT / "web_uploads"
WEB_OUTPUT_DIR = OUTPUT_ROOT / "web"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
SESSION_COOKIE = "pose_session"

ACTION_LABELS = {
    "none": "关闭动作指导",
    "lunge": "负重箭步蹲",
    "wall_ball": "投掷药球",
    "farmers_carry": "农夫行走",
    "rowing": "划船机",
    "skierg": "滑雪机",
    "burpee_broad_jump": "波比跳远",
    "sled_push": "推雪橇",
    "sled_pull": "拉雪橇",
}
ACTION_BY_SAMPLE_STEM = {
    "负重箭步蹲": "lunge",
    "投掷药球": "wall_ball",
    "农夫行走": "farmers_carry",
    "划船机": "rowing",
    "滑雪机": "skierg",
    "波比跳远": "burpee_broad_jump",
    "推雪橇": "sled_push",
    "拉雪橇": "sled_pull",
}
VIEW_LABELS = {
    "unknown": "自动 / 未指定",
    "front": "正面",
    "side": "侧面",
    "front_left": "左前方",
    "front_right": "右前方",
}

FACE_KEYPOINTS = {
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
}
FINGER_KEYPOINTS = {
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
}
UPPER_BODY_KEYPOINTS = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
}
LOWER_BODY_KEYPOINTS = {
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
}


def _visible_names(profile: str) -> set[str] | None:
    if profile == "no-face":
        return None  # resolved from the backend result below
    if profile == "upper-body":
        return UPPER_BODY_KEYPOINTS
    if profile == "lower-body":
        return LOWER_BODY_KEYPOINTS
    return None


def _result_visible_names(profile: str, result_names: set[str], *, show_fingers: bool) -> set[str]:
    visible = _visible_names(profile)
    if profile == "no-face":
        visible = result_names - FACE_KEYPOINTS
    if visible is None:
        visible = set(result_names)
    else:
        visible &= result_names
    if not show_fingers:
        visible -= FINGER_KEYPOINTS
    return visible


def _feedback_items(state: Mapping[str, object] | None, *, phase_aware: bool = True) -> list[dict[str, Any]]:
    if not state:
        return []
    messages = state.get("feedback_messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return []
    items: list[dict[str, Any]] = []
    for message in messages[:3]:
        if isinstance(message, Mapping):
            level = str(message.get("level", "info"))
            code = str(message.get("code", ""))
            text = str(message.get("text", ""))
            confidence = message.get("confidence")
        else:
            level = str(getattr(message, "level", "info"))
            code = str(getattr(message, "code", ""))
            text = str(getattr(message, "text", ""))
            confidence = getattr(message, "confidence", None)
        if text:
            item: dict[str, Any] = {"level": level, "code": code, "text": text}
            if isinstance(confidence, (int, float)):
                item["confidence"] = round(max(0.0, min(1.0, float(confidence))), 3)
            items.append(item)
    if not phase_aware:
        return items
    return visible_feedback(items, str(state.get("phase", "unknown")))


def _draw_angle_overlay(frame: Any, result: Any, assessment: Mapping[str, Any]) -> None:
    if not result.success:
        return
    height, width = frame.shape[:2]
    points = {point.name: point for point in result.keypoints}
    for item in list(assessment.get("angles") or [])[:5]:
        point = points.get(str(item.get("anchor", "")))
        if point is None or point.confidence < 0.2:
            continue
        x, y = to_pixel(point.x, point.y, width, height)
        color = (70, 220, 110) if item.get("status") != "bad" else (50, 70, 245)
        label = f"{float(item.get('value', 0)):.0f} deg"
        origin = (x + 8, max(20, y - 8))
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(
            frame,
            (origin[0] - 3, origin[1] - text_height - 3),
            (origin[0] + text_width + 3, origin[1] + baseline + 3),
            (25, 25, 23),
            -1,
        )
        cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _backend_plan(config: Mapping[str, Any]) -> tuple[str, str]:
    """Lock the foreground athlete throughout the crowded lunge sample."""
    requested = str(config.get("backend", "auto"))
    if config.get("source_mode") == "sample" and config.get("action") == "lunge":
        if requested == "auto":
            return "yolo-mediapipe", "tracking"
    return requested, "tracking"


class PoseStreamEngine:
    """Owns one local capture and publishes its latest annotated frame."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_jpeg: bytes | None = None
        self._latest_frame: Any | None = None
        self._frame_version = 0
        self._record_requested = False
        self._output_dir = output_dir or WEB_OUTPUT_DIR
        self._history: deque[dict[str, Any]] = deque(maxlen=9000)
        self._voice_feedback = RepVoiceFeedbackTracker()
        self._settings: dict[str, Any] = {
            "action": "lunge",
            "camera_view": "side",
            "sensitivity": "medium",
            "mirror": True,
            "landmark_profile": "full",
            "show_fingers": False,
            "paused": False,
            "manual_floor_points": [],
        }
        self._state: dict[str, Any] = {
            "running": False,
            "status": "idle",
            "status_text": "等待开始",
            "source_mode": "camera",
            "source_name": "摄像头 0",
            "backend": "-",
            "action": "lunge",
            "action_label": ACTION_LABELS["lunge"],
            "camera_view": "side",
            "view_label": VIEW_LABELS["side"],
            "sensitivity": "medium",
            "pose_detected": False,
            "fps": 0.0,
            "inference_ms": 0.0,
            "phase": "idle",
            "reps": 0,
            "candidate_count": 0,
            "pose_valid_rep_count": 0,
            "no_rep_count": 0,
            "unsure_count": 0,
            "floor_reference": {},
            "contacts": {},
            "foot_events": {},
            "feedback": [],
            "voice_feedback": None,
            "frame_index": 0,
            "progress": 0.0,
            "recording": False,
            "record_path": "",
            "paused": False,
            "error": "",
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self, config: Mapping[str, Any]) -> None:
        self.stop()
        with self._lock:
            self._settings.update(
                {
                    "action": config["action"],
                    "camera_view": config["camera_view"],
                    "sensitivity": config["sensitivity"],
                    "mirror": bool(config.get("mirror", True)),
                    "landmark_profile": config.get("landmark_profile", "full"),
                    "show_fingers": bool(config.get("show_fingers", False)),
                    "paused": False,
                    "manual_floor_points": validate_manual_floor_points(
                        config.get("manual_floor_points", ())
                    ),
                }
            )
            self._record_requested = False
            self._history.clear()
            self._voice_feedback.reset()
            self._latest_jpeg = None
            self._latest_frame = None
            self._state.update(
                {
                    "running": True,
                    "status": "starting",
                    "status_text": "正在加载模型…",
                    "source_mode": config["source_mode"],
                    "source_name": config["source_name"],
                    "backend": config["backend"],
                    "action": config["action"],
                    "action_label": ACTION_LABELS[config["action"]],
                    "camera_view": config["camera_view"],
                    "view_label": VIEW_LABELS[config["camera_view"]],
                    "sensitivity": config["sensitivity"],
                    "pose_detected": False,
                    "fps": 0.0,
                    "inference_ms": 0.0,
                    "phase": "idle",
                    "reps": 0,
                    "candidate_count": 0,
                    "pose_valid_rep_count": 0,
                    "no_rep_count": 0,
                    "unsure_count": 0,
                    "floor_reference": {},
                    "contacts": {},
                    "foot_events": {},
                    "feedback": [],
                    "voice_feedback": None,
                    "frame_index": 0,
                    "progress": 0.0,
                    "recording": False,
                    "record_path": "",
                    "paused": False,
                    "error": "",
                }
            )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(dict(config),), daemon=True, name="pose-web-stream")
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            self._stop_event.set()
            thread.join(timeout=5.0)
        with self._lock:
            self._record_requested = False
            if self._state["status"] not in {"completed", "error"}:
                self._state.update({"running": False, "status": "idle", "status_text": "已停止", "recording": False})
            self._frame_ready.notify_all()
        self._thread = None

    def update_settings(self, values: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "action": set(("none", *HYROX_ACTION_NAMES)),
            "camera_view": set(CAMERA_VIEWS),
            "sensitivity": {"low", "medium", "high"},
            "landmark_profile": {"full", "no-face", "upper-body", "lower-body"},
        }
        with self._lock:
            for key, choices in allowed.items():
                if key in values:
                    value = str(values[key])
                    if value not in choices:
                        raise ValueError(f"invalid {key}: {value}")
                    self._settings[key] = value
            for key in ("mirror", "paused", "show_fingers"):
                if key in values:
                    self._settings[key] = bool(values[key])
            if "manual_floor_points" in values:
                self._settings["manual_floor_points"] = validate_manual_floor_points(
                    values["manual_floor_points"]
                )
            self._state.update(
                {
                    "action": self._settings["action"],
                    "action_label": ACTION_LABELS[self._settings["action"]],
                    "camera_view": self._settings["camera_view"],
                    "view_label": VIEW_LABELS[self._settings["camera_view"]],
                    "sensitivity": self._settings["sensitivity"],
                    "paused": self._settings["paused"],
                    "status_text": "已暂停" if self._settings["paused"] else ("分析中" if self._state["running"] else self._state["status_text"]),
                }
            )
            return dict(self._state)

    def request_recording(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            if enabled and not self._state["running"]:
                raise RuntimeError("请先开始分析")
            self._record_requested = bool(enabled)
            return dict(self._state)

    def save_screenshot(self) -> Path:
        with self._lock:
            if self._latest_frame is None:
                raise RuntimeError("当前还没有可保存的画面")
            frame = self._latest_frame.copy()
        target = self._output_dir / "screenshots" / f"{time.strftime('%Y-%m-%d_%H%M%S')}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), frame):
            raise RuntimeError("截图保存失败")
        return target

    def wait_for_frame(self, version: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self._frame_ready:
            if self._frame_version <= version and not self._stop_event.is_set():
                self._frame_ready.wait(timeout=timeout)
            return self._frame_version, self._latest_jpeg

    def report(self) -> dict[str, Any]:
        with self._lock:
            frames = list(self._history)
            state = dict(self._state)
        return enrich_report({
            **artifact_metadata("web_pose_report"),
            "generated_at_unix_ms": int(time.time() * 1000),
            "retention_minutes": 10,
            "privacy": "报告不包含原始或标注视频帧",
            "summary": {
                "source_name": state["source_name"],
                "action": state["action"],
                "action_label": state["action_label"],
                "reps": state["reps"],
                "candidate_count": state["candidate_count"],
                "pose_valid_rep_count": state["pose_valid_rep_count"],
                "no_rep_count": state["no_rep_count"],
                "unsure_count": state["unsure_count"],
                "last_phase": state["phase"],
                "processed_frames": len(frames),
                "dropped_frames": 0,
                "backend": state["backend"],
            },
            "frames": frames,
        })

    def report_csv(self) -> str:
        output = io.StringIO(newline="")
        fieldnames = versioned_csv_columns([
            "sequence", "timestamp_unix_ms", "action", "phase", "reps", "pose_detected",
            "inference_ms", "server_ms", "width", "height", "feedback", "keypoints_json",
        ])
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for frame in self.report()["frames"]:
            metrics = frame.get("metrics", {})
            writer.writerow(
                versioned_csv_row({
                    "sequence": frame.get("sequence"),
                    "timestamp_unix_ms": frame.get("timestamp_unix_ms"),
                    "action": frame.get("action"),
                    "phase": frame.get("phase"),
                    "reps": frame.get("reps"),
                    "pose_detected": frame.get("pose_detected"),
                    "inference_ms": metrics.get("inference_ms"),
                    "server_ms": metrics.get("server_ms"),
                    "width": metrics.get("width"),
                    "height": metrics.get("height"),
                    "feedback": json.dumps(frame.get("feedback", []), ensure_ascii=False),
                    "keypoints_json": json.dumps(frame.get("keypoints", []), ensure_ascii=False),
                })
            )
        return output.getvalue()

    def _open_capture(self, config: Mapping[str, Any]) -> tuple[Any, float, int]:
        if config["source_mode"] == "camera":
            camera_index = int(config.get("camera_index", 0))
            capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(camera_index)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            capture.set(cv2.CAP_PROP_FPS, 30)
        else:
            capture = cv2.VideoCapture(str(config["video_path"]))
        if not capture.isOpened():
            raise RuntimeError("无法打开所选摄像头或视频")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if config["source_mode"] != "camera" else 0
        return capture, fps if fps > 0 else 30.0, max(0, total_frames)

    def _create_backend(
        self,
        requested: str,
        action: str,
        source_path: str,
        *,
        target_select: str = "tracking",
    ) -> tuple[Any, str]:
        resolved = resolve_backend_choice(requested, action_type=action, input_video=source_path)
        if requested == "auto" and action == "lunge":
            resolved = "yolo-mediapipe"
        if resolved == "mediapipe":
            # MediaPipe 0.10.35 on Windows can abort the entire process while
            # producing segmentation masks for some non-standard video sizes.
            # Keypoint detection remains fully available without the mask.
            return (
                MediaPipeBackend(
                    str(PROJECT_ROOT / "models" / "pose_landmarker_full.task"),
                    output_segmentation_masks=False,
                ),
                resolved,
            )
        if resolved == "yolo-mediapipe":
            return (
                YoloGuidedMediaPipeBackend(
                    PROJECT_ROOT / "yolo11n-pose.pt",
                    PROJECT_ROOT / "models" / "pose_landmarker_full.task",
                    target_select=target_select,
                    device=resolve_torch_device("auto"),
                ),
                "yolo-guided-mediapipe",
            )
        if resolved == "yolo-pose":
            return YoloPoseBackend(str(PROJECT_ROOT / "yolo11n-pose.pt"), target_select=target_select, device=resolve_torch_device("auto")), resolved
        if resolved == "rtmw-wholebody":
            try:
                return (
                    YoloRtmwWholeBodyBackend(
                        PROJECT_ROOT
                        / "models"
                        / "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
                        PROJECT_ROOT / "yolo11n-pose.pt",
                        target_select=target_select,
                        yolo_device=resolve_torch_device("auto"),
                        rtmw_device="auto",
                    ),
                    "yolo-rtmw-wholebody",
                )
            except Exception as exc:
                logging.getLogger("pose.web").warning(
                    "RTMW WholeBody unavailable; falling back to YOLO + MediaPipe: %s",
                    exc,
                )
                return (
                    YoloGuidedMediaPipeBackend(
                        PROJECT_ROOT / "yolo11n-pose.pt",
                        PROJECT_ROOT / "models" / "pose_landmarker_full.task",
                        target_select=target_select,
                        device=resolve_torch_device("auto"),
                    ),
                    "yolo-guided-mediapipe-fallback",
                )
        raise ValueError(f"unknown backend: {resolved}")

    def _run(self, config: dict[str, Any]) -> None:
        capture = None
        backend = None
        writer = None
        record_path: Path | None = None
        finished_normally = False
        hand_overlay = WebHandOverlay(PROJECT_ROOT / "models" / "hand_landmarker.task")
        try:
            source_path = str(config.get("video_path", ""))
            capture, source_fps, total_frames = self._open_capture(config)
            backend_request, target_select = _backend_plan(config)
            backend, resolved_backend = self._create_backend(
                backend_request,
                config["action"],
                source_path,
                target_select=target_select,
            )
            smoother = KeypointSmoother(
                mode="none" if config["source_mode"] == "sample" else "one-euro",
                max_missing_frames=5,
                occlusion_guard=True,
            )
            sample_frame_step = max(1, int(math.ceil(source_fps / 10.0))) if config["source_mode"] == "sample" else 1
            analyzer = None
            analyzer_key: tuple[str, str, str] | None = None
            started = time.perf_counter()
            last_frame_time = started
            smooth_fps = 0.0
            frame_index = 0
            with self._lock:
                self._state.update({"backend": resolved_backend, "status": "running", "status_text": "分析中"})

            while not self._stop_event.is_set():
                with self._lock:
                    settings = dict(self._settings)
                    record_requested = self._record_requested
                if settings["paused"]:
                    time.sleep(0.05)
                    continue

                frame_started = time.perf_counter()
                ok, raw_frame = capture.read()
                if not ok or raw_frame is None:
                    finished_normally = config["source_mode"] != "camera"
                    break
                frame_index += 1
                frame = cv2.flip(raw_frame, 1) if config["source_mode"] == "camera" and settings["mirror"] else raw_frame.copy()
                timestamp_ms = int(round((frame_index * 1000.0) / source_fps)) if config["source_mode"] != "camera" else int((time.perf_counter() - started) * 1000)
                result = smoother.smooth_result(backend.detect(frame, timestamp_ms=timestamp_ms))
                has_pose = bool(result.success and result.keypoints)
                show_hand_overlay = hand_overlay_visible(
                    str(settings["landmark_profile"]),
                    bool(settings["show_fingers"]),
                )
                hand_detections = (
                    rtmw_hand_detections(result.extra)
                    if show_hand_overlay
                    else {}
                )
                hand_detections = hand_detections or hand_overlay.update(
                    frame,
                    timestamp_ms=timestamp_ms,
                    enabled=show_hand_overlay,
                )
                if hand_detections and result.extra.get("rtmw_hand_keypoints"):
                    hand_overlay.update(
                        frame,
                        timestamp_ms=timestamp_ms,
                        enabled=False,
                    )

                key = (settings["action"], settings["sensitivity"], settings["camera_view"])
                if key != analyzer_key:
                    analyzer = (
                        create_action_analyzer(
                            key[0],
                            sensitivity=key[1],
                            camera_view=key[2],
                            live_mode=config["source_mode"] == "camera",
                        )
                        if key[0] != "none"
                        else None
                    )
                    analyzer_key = key
                action_state = None
                features = None
                if analyzer is not None:
                    manual_points = settings.get("manual_floor_points") or []
                    analyzer.set_manual_floor_line(
                        manual_points[0] if len(manual_points) == 2 else None,
                        manual_points[1] if len(manual_points) == 2 else None,
                    )
                    if has_pose:
                        height, width = frame.shape[:2]
                        features = extract_basic_pose_features(
                            result.keypoints,
                            image_width=width,
                            image_height=height,
                            segmentation_mask=result.extra.get("segmentation_mask"),
                        )
                    action_state = analyzer.attach_view_context(analyzer.update(features, timestamp_ms=timestamp_ms))

                phase = "idle" if action_state is None else str(action_state.get("phase", "unknown"))
                action_debug = action_state.get("debug") if isinstance(action_state, Mapping) else None
                floor_reference = (
                    dict(action_debug.get("floor_reference") or {})
                    if isinstance(action_debug, Mapping)
                    else {}
                )
                contacts = (
                    dict(action_debug.get("contacts") or {})
                    if isinstance(action_debug, Mapping)
                    else {}
                )
                foot_events = (
                    dict(action_debug.get("foot_events") or {})
                    if isinstance(action_debug, Mapping)
                    else {}
                )
                last_rep_decision = (
                    dict(action_state.get("last_rep_decision") or {})
                    if isinstance(action_state, Mapping)
                    else {}
                )
                last_rep_observability = (
                    dict(action_state.get("last_rep_observability") or {})
                    if isinstance(action_state, Mapping)
                    else {}
                )
                evaluation_phase = str(action_debug.get("raw_phase", phase)) if isinstance(action_debug, Mapping) else phase
                all_feedback = _feedback_items(action_state, phase_aware=False)
                feedback = visible_feedback(all_feedback, evaluation_phase)
                assessment = assess_action(settings["action"], evaluation_phase, features, feedback)
                annotated = frame.copy()
                result_names = {point.name for point in result.keypoints}
                visible_names = _result_visible_names(
                    settings["landmark_profile"],
                    result_names,
                    show_fingers=bool(settings["show_fingers"]),
                )
                skeleton_color = {
                    "bad": (50, 70, 245),
                    "good": (70, 220, 110),
                }.get(str(assessment["status"]), (70, 190, 240))
                draw_pose_result_filtered(
                    annotated,
                    result,
                    visible_names=visible_names,
                    line_color=skeleton_color,
                    point_color=skeleton_color,
                )
                if hand_detections:
                    draw_hand_overlay(annotated, hand_detections)
                _draw_angle_overlay(annotated, result, assessment)
                if action_state is not None:
                    draw_hyrox_action_overlay(annotated, action_state, origin=(16, 32))

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
                smooth_fps = instant_fps if smooth_fps <= 0 else smooth_fps * 0.85 + instant_fps * 0.15
                last_frame_time = now
                reps = 0 if action_state is None else int(action_state.get("rep_count", 0))
                candidate_count = (
                    0
                    if action_state is None
                    else int(action_state.get("candidate_count", reps))
                )
                pose_valid_rep_count = (
                    reps
                    if action_state is None
                    else int(action_state.get("pose_valid_rep_count", reps))
                )
                no_rep_count = (
                    0 if action_state is None else int(action_state.get("no_rep_count", 0))
                )
                unsure_count = (
                    0 if action_state is None else int(action_state.get("unsure_count", 0))
                )
                voice_feedback = self._voice_feedback.update(
                    action=str(settings["action"]),
                    reps=reps,
                    assessment=assessment,
                    detected_issues=all_feedback,
                    timestamp_ms=int(time.time() * 1000),
                )
                visible_names = _result_visible_names(
                    settings["landmark_profile"],
                    result_names,
                    show_fingers=bool(settings["show_fingers"]),
                )
                keypoints = [
                    {
                        "name": point.name,
                        "x": round(float(point.x), 6),
                        "y": round(float(point.y), 6),
                        "z": round(float(point.z), 6) if math.isfinite(float(point.z)) else 0.0,
                        "visibility": round(float(point.confidence), 4),
                    }
                    for point in result.keypoints
                    if point.name in visible_names and math.isfinite(float(point.x)) and math.isfinite(float(point.y))
                ]
                hand_keypoints, _ = serialize_hand_overlay(hand_detections)
                keypoints.extend(hand_keypoints)

                if record_requested and writer is None:
                    record_path = self._output_dir / "recordings" / f"{time.strftime('%Y-%m-%d_%H%M%S')}.mp4"
                    record_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), min(60.0, max(1.0, source_fps)), (width, height))
                    if not writer.isOpened():
                        writer = None
                        raise RuntimeError("无法创建录制文件")
                elif not record_requested and writer is not None:
                    writer.release()
                    writer = None
                if writer is not None:
                    writer.write(annotated)

                encoded, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
                if encoded:
                    progress = (frame_index / total_frames * 100.0) if total_frames > 0 else 0.0
                    with self._frame_ready:
                        self._history.append(
                            {
                                "sequence": frame_index,
                                "timestamp_unix_ms": int(time.time() * 1000),
                                "action": settings["action"],
                                "phase": phase,
                                "reps": reps,
                                "candidate_count": candidate_count,
                                "pose_valid_rep_count": pose_valid_rep_count,
                                "no_rep_count": no_rep_count,
                                "unsure_count": unsure_count,
                                "floor_reference": floor_reference,
                                "contacts": contacts,
                                "foot_events": foot_events,
                                "last_rep_decision": last_rep_decision,
                                "last_rep_observability": last_rep_observability,
                                "pose_detected": has_pose,
                                "hands_detected": bool(hand_keypoints),
                                "feedback": feedback,
                                "voice_feedback": voice_feedback,
                                "detected_issues": all_feedback,
                                "assessment": assessment,
                                "keypoints": keypoints,
                                "metrics": {
                                    "backend": resolved_backend,
                                    "inference_ms": round(float(result.inference_time_ms), 1),
                                    "server_ms": round((time.perf_counter() - frame_started) * 1000.0, 1),
                                    "width": int(annotated.shape[1]),
                                    "height": int(annotated.shape[0]),
                                    "fps": round(smooth_fps, 1),
                                },
                            }
                        )
                        self._latest_frame = annotated
                        self._latest_jpeg = jpeg.tobytes()
                        self._frame_version += 1
                        self._state.update(
                            {
                                "running": True,
                                "status": "running",
                                "status_text": "录制中" if writer is not None else "分析中",
                                "pose_detected": has_pose,
                                "fps": round(smooth_fps, 1),
                                "inference_ms": round(float(result.inference_time_ms), 1),
                                "phase": phase,
                                "reps": reps,
                                "candidate_count": candidate_count,
                                "pose_valid_rep_count": pose_valid_rep_count,
                                "no_rep_count": no_rep_count,
                                "unsure_count": unsure_count,
                                "floor_reference": floor_reference,
                                "contacts": contacts,
                                "foot_events": foot_events,
                                "feedback": feedback,
                                "voice_feedback": voice_feedback,
                                "frame_index": frame_index,
                                "progress": round(min(progress, 100.0), 1),
                                "recording": writer is not None,
                                "record_path": str(record_path.relative_to(PROJECT_ROOT)) if record_path else "",
                                "paused": False,
                            }
                        )
                        self._frame_ready.notify_all()

                if config["source_mode"] == "sample" and sample_frame_step > 1:
                    for _ in range(sample_frame_step - 1):
                        if not capture.grab():
                            finished_normally = True
                            break
                        frame_index += 1
                if config["source_mode"] != "camera":
                    elapsed = time.perf_counter() - frame_started
                    time.sleep(max(0.0, (sample_frame_step / source_fps) - elapsed))
        except Exception as exc:
            with self._frame_ready:
                self._state.update({"running": False, "status": "error", "status_text": "运行出错", "error": str(exc), "recording": False})
                self._frame_ready.notify_all()
        finally:
            hand_overlay.close()
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            if backend is not None:
                backend.close()
            if config.get("delete_source_after") and config.get("video_path"):
                Path(str(config["video_path"])).unlink(missing_ok=True)
            with self._frame_ready:
                if self._state["status"] != "error":
                    self._state.update(
                        {
                            "running": False,
                            "status": "completed" if finished_normally else "idle",
                            "status_text": "视频分析完成" if finished_normally else "已停止",
                            "progress": 100.0 if finished_normally else self._state["progress"],
                            "recording": False,
                        }
                    )
                self._frame_ready.notify_all()


def _discover_sample_videos() -> list[dict[str, str]]:
    video_dir = PROJECT_ROOT / "HYROX视频"
    if not video_dir.exists():
        return []
    samples: list[dict[str, str]] = []
    for index, path in enumerate(sorted(video_dir.iterdir(), key=lambda item: item.name)):
        if path.is_file() and path.suffix.lower() in ALLOWED_VIDEO_SUFFIXES:
            action = ACTION_BY_SAMPLE_STEM.get(path.stem)
            if action:
                samples.append({"id": f"sample-{index}", "name": path.stem, "path": str(path), "action": action})
    return samples


def create_app(
    engine: PoseStreamEngine | None = None,
    access_token: str | None = None,
    *,
    engine_factory: Any | None = None,
    realtime_factory: Any | None = None,
) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    if os.environ.get("POSE_TRUST_PROXY", "").lower() in {"1", "true", "yes"}:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.secret_key = os.environ.get("POSE_WEB_SECRET") or secrets.token_hex(32)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        ACCESS_TOKEN=access_token or "",
        SESSION_COOKIE_SECURE=os.environ.get("POSE_SECURE_COOKIES", "").lower() in {"1", "true", "yes"},
    )
    samples = _discover_sample_videos()
    sample_lookup = {item["id"]: item for item in samples}
    storage_root = OUTPUT_ROOT / "web_sessions"

    def make_engine(session_id: str) -> Any:
        if engine_factory is not None:
            return engine_factory(session_id)
        return engine if engine is not None else PoseStreamEngine(storage_root / session_id / "outputs")

    manager = SessionManager(
        engine_factory=make_engine,
        realtime_factory=realtime_factory,
        storage_root=storage_root,
    )
    sock = Sock(app)
    app.extensions["pose_sessions"] = manager
    app.extensions["pose_sock"] = sock
    rate_windows: dict[str, deque[float]] = defaultdict(deque)
    logger = logging.getLogger("pose.web")
    app_started_at = time.monotonic()

    def access_cookie(expected: str) -> str:
        return hmac.new(app.secret_key.encode("utf-8"), expected.encode("utf-8"), hashlib.sha256).hexdigest()

    def current_session() -> BrowserSession:
        return g.pose_session

    def json_error(message: str, status: int, code: str) -> tuple[Response, int]:
        return jsonify({"error": message, "code": code, "request_id": getattr(g, "request_id", "")}), status

    def available_report(item: BrowserSession) -> tuple[dict[str, Any], str]:
        realtime_report = item.realtime.report()
        if realtime_report["frames"]:
            return realtime_report, item.realtime.report_csv()
        if hasattr(item.engine, "report") and hasattr(item.engine, "report_csv"):
            engine_report = item.engine.report()
            if engine_report["frames"]:
                return engine_report, item.engine.report_csv()
        return realtime_report, ""

    def runtime_stats() -> dict[str, Any]:
        stats: dict[str, Any] = {**manager.stats(), "uptime_seconds": int(time.monotonic() - app_started_at)}
        try:
            import psutil

            process = psutil.Process()
            stats.update(
                {
                    "process_memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
                    "system_memory_percent": round(float(psutil.virtual_memory().percent), 1),
                    "system_cpu_percent": round(float(psutil.cpu_percent(interval=None)), 1),
                }
            )
        except (ImportError, OSError):
            pass
        try:
            import torch

            if torch.cuda.is_available():
                stats.update(
                    {
                        "gpu": torch.cuda.get_device_name(0),
                        "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1),
                        "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1),
                    }
                )
        except (ImportError, RuntimeError):
            pass
        return stats

    @app.before_request
    def prepare_request() -> Response | tuple[Response, int] | None:
        g.request_id = uuid.uuid4().hex
        if request.path in {"/healthz", "/status"} or request.path.startswith("/static/"):
            return None

        expected = str(app.config["ACCESS_TOKEN"])
        supplied = request.args.get("access_token", "")
        if expected and supplied and hmac.compare_digest(supplied, expected):
            response = redirect(request.path)
            response.set_cookie(
                "pose_access",
                access_cookie(expected),
                max_age=7 * 24 * 60 * 60,
                httponly=True,
                secure=bool(app.config["SESSION_COOKIE_SECURE"] or request.is_secure),
                samesite="Lax",
            )
            return response
        if expected:
            cookie = request.cookies.get("pose_access", "")
            if not cookie or not hmac.compare_digest(cookie, access_cookie(expected)):
                if request.path.startswith("/api/") or request.path.startswith("/ws/"):
                    return json_error("访问链接无效或已过期", 401, "unauthorized")
                return Response(
                    "<!doctype html><meta charset='utf-8'><title>需要访问链接</title>"
                    "<h1>需要有效的分享链接</h1><p>请向服务提供者索取完整链接。</p>",
                    status=401,
                    mimetype="text/html",
                )

        now = time.monotonic()
        remote_ip = request.remote_addr or "unknown"
        window = rate_windows[remote_ip]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= 240:
            return json_error("请求过于频繁，请稍后重试", 429, "rate_limited")
        window.append(now)
        try:
            item, created = manager.get_or_create(request.cookies.get(SESSION_COOKIE), remote_ip)
        except SessionCapacityError as exc:
            return json_error(str(exc), 503, "server_busy")
        g.pose_session = item
        g.pose_session_created = created

        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.path.startswith("/api/"):
            supplied_csrf = request.headers.get("X-CSRF-Token", "")
            if not supplied_csrf or not hmac.compare_digest(supplied_csrf, item.csrf_token):
                return json_error("页面安全令牌已失效，请刷新后重试", 403, "csrf_failed")
        return None

    @app.after_request
    def secure_response(response: Response) -> Response:
        response.headers["X-Request-ID"] = getattr(g, "request_id", uuid.uuid4().hex)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' ws: wss:"
        )
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if getattr(g, "pose_session_created", False):
            response.set_cookie(
                SESSION_COOKIE,
                current_session().session_id,
                max_age=24 * 60 * 60,
                httponly=True,
                secure=bool(app.config["SESSION_COOKIE_SECURE"] or request.is_secure),
                samesite="Lax",
            )
        session_prefix = getattr(getattr(g, "pose_session", None), "session_id", "-")[:8]
        logger.info(
            "request_id=%s session_id=%s method=%s path=%s status=%s",
            getattr(g, "request_id", "-"),
            session_prefix,
            request.method,
            request.path,
            response.status_code,
        )
        return response

    @app.errorhandler(413)
    def payload_too_large(_: Any) -> tuple[Response, int]:
        return json_error("视频文件不能超过 250 MB", 413, "upload_too_large")

    @app.get("/healthz")
    def healthz() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/status")
    def status_page() -> str:
        return render_template("status.html", stats=runtime_stats())

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/options")
    def options() -> Response:
        item = current_session()
        return jsonify(
            {
                "actions": [{"value": value, "label": ACTION_LABELS[value]} for value in ("none", *HYROX_ACTION_NAMES)],
                "views": [{"value": value, "label": VIEW_LABELS[value]} for value in CAMERA_VIEWS],
                "samples": [{"id": sample["id"], "name": sample["name"], "action": sample["action"]} for sample in samples],
                "standards": {action: standards_for(action) for action in HYROX_ACTION_NAMES},
                "official_rules": {action: official_rules_for(action) for action in HYROX_ACTION_NAMES},
                "csrf_token": item.csrf_token,
                "realtime": {
                    "frame_width": 640,
                    "frame_height": 480,
                    "target_fps": 30,
                    "camera_fps": 60,
                    "max_frame_bytes": 512 * 1024,
                    "report_retention_seconds": 600,
                },
                "privacy": {
                    "audio": False,
                    "stores_camera_frames": False,
                    "stores_uploaded_video": False,
                    "report_retention_minutes": 10,
                },
            }
        )

    @app.post("/api/upload")
    def upload() -> tuple[Response, int] | Response:
        item = current_session()
        file = request.files.get("video")
        if file is None or not file.filename:
            return json_error("请选择视频文件", 400, "missing_upload")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            return json_error("不支持该视频格式", 400, "unsupported_media")
        used_bytes = sum(int(value.get("size", 0)) for value in item.uploads.values())
        if used_bytes >= 500 * 1024 * 1024:
            return json_error("当前会话的临时上传空间已满", 413, "quota_exceeded")
        upload_dir = storage_root / item.session_id / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(24)
        safe_stem = secure_filename(Path(file.filename).stem)[:50] or "video"
        target = upload_dir / f"{token}_{safe_stem}{suffix}"
        file.save(target)
        size = target.stat().st_size
        if size <= 0 or used_bytes + size > 500 * 1024 * 1024:
            target.unlink(missing_ok=True)
            return json_error("视频为空或超出当前会话空间配额", 413, "quota_exceeded")
        capture = cv2.VideoCapture(str(target))
        valid_media = capture.isOpened()
        ok, frame = capture.read() if valid_media else (False, None)
        capture.release()
        if not ok or frame is None:
            target.unlink(missing_ok=True)
            return json_error("文件内容不是可读取的视频", 400, "invalid_media")
        item.uploads[token] = {"path": str(target), "name": file.filename, "size": size}
        return jsonify({"id": token, "name": file.filename, "expires_in_seconds": 600})

    @app.post("/api/start")
    def start() -> tuple[Response, int] | Response:
        item = current_session()
        data = request.get_json(silent=True) or {}
        try:
            source_mode = str(data.get("source_mode", "sample"))
            if source_mode not in {"camera", "sample", "upload"}:
                raise ValueError("无效的输入方式")
            action = str(data.get("action", "lunge"))
            view = str(data.get("camera_view", "side"))
            sensitivity = str(data.get("sensitivity", "medium"))
            backend = str(data.get("backend", "auto"))
            profile = str(data.get("landmark_profile", "full"))
            if action not in {"none", *HYROX_ACTION_NAMES}:
                raise ValueError("无效的动作")
            if view not in CAMERA_VIEWS:
                raise ValueError("无效的拍摄视角")
            if sensitivity not in {"low", "medium", "high"}:
                raise ValueError("无效的灵敏度")
            if backend not in {
                "auto",
                "mediapipe",
                "yolo-mediapipe",
                "yolo-pose",
                "rtmw-wholebody",
            }:
                raise ValueError("无效的识别后端")
            if profile not in {"full", "no-face", "upper-body", "lower-body"}:
                raise ValueError("无效的骨架显示模式")
            config: dict[str, Any] = {
                "source_mode": source_mode,
                "action": action,
                "camera_view": view,
                "sensitivity": sensitivity,
                "backend": backend,
                "landmark_profile": profile,
                "show_fingers": bool(data.get("show_fingers", False)),
                "mirror": bool(data.get("mirror", source_mode == "camera")),
                "manual_floor_points": validate_manual_floor_points(
                    data.get("manual_floor_points", ())
                ),
            }
            if source_mode == "camera":
                camera_index = int(data.get("camera_index", 0))
                if not 0 <= camera_index <= 16:
                    raise ValueError("摄像头编号应在 0 到 16 之间")
                config.update({"camera_index": camera_index, "source_name": f"服务器摄像头 {camera_index}"})
            elif source_mode == "sample":
                sample_id = str(data.get("video_id", ""))
                if sample_id not in sample_lookup:
                    raise ValueError("请选择示例视频")
                sample = sample_lookup[sample_id]
                if action != sample["action"]:
                    raise ValueError("所选动作与示例视频不一致")
                config.update({"video_path": sample["path"], "source_name": Path(sample["path"]).stem})
            else:
                upload_id = str(data.get("video_id", ""))
                upload = item.uploads.pop(upload_id, None)
                if upload is None:
                    raise ValueError("请先上传视频")
                config.update(
                    {
                        "video_path": upload["path"],
                        "source_name": upload["name"],
                        "delete_source_after": True,
                    }
                )
            item.engine.start(config)
            return jsonify(item.engine.snapshot())
        except (TypeError, ValueError) as exc:
            return json_error(str(exc), 400, "invalid_request")

    @app.post("/api/stop")
    def stop() -> Response:
        item = current_session()
        item.engine.stop()
        item.realtime.stop()
        return jsonify(item.engine.snapshot())

    @app.post("/api/settings")
    def settings() -> tuple[Response, int] | Response:
        item = current_session()
        values = request.get_json(silent=True) or {}
        try:
            if item.realtime.running:
                return jsonify(item.realtime.update_settings(values))
            return jsonify(item.engine.update_settings(values))
        except (ValueError, RealtimeProtocolError) as exc:
            return json_error(str(exc), 400, "invalid_settings")

    @app.post("/api/record")
    def record() -> tuple[Response, int] | Response:
        return json_error("为避免保存用户画面，服务器录制功能已关闭", 410, "recording_disabled")

    @app.post("/api/screenshot")
    def screenshot() -> tuple[Response, int] | Response:
        return json_error("截图只下载到当前设备，服务器不保存画面", 410, "server_screenshot_disabled")

    @app.get("/api/state")
    def state() -> Response:
        item = current_session()
        return jsonify(item.realtime.snapshot() if item.realtime.running else item.engine.snapshot())

    @app.get("/api/stream")
    def stream() -> Response:
        item = current_session()

        def generate() -> Iterator[bytes]:
            version = 0
            while True:
                version, jpeg = item.engine.wait_for_frame(version)
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"
                snapshot = item.engine.snapshot()
                if not snapshot["running"] and snapshot["status"] in {"idle", "error", "completed"}:
                    break

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/report.json")
    def download_json_report() -> tuple[Response, int] | Response:
        report, _ = available_report(current_session())
        if not report["frames"]:
            return json_error("当前会话还没有可下载的分析结果", 409, "report_empty")
        body = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
        response = Response(body, mimetype="application/json")
        response.headers["Content-Disposition"] = "attachment; filename=hyrox-analysis.json"
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get("/api/report")
    def generated_report() -> tuple[Response, int] | Response:
        report, _ = available_report(current_session())
        if not report["frames"]:
            return json_error("当前会话还没有可生成的分析结果", 409, "report_empty")
        response = jsonify(enrich_report(report))
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get("/api/report.csv")
    def download_csv_report() -> tuple[Response, int] | Response:
        report, csv_body = available_report(current_session())
        if not report["frames"]:
            return json_error("当前会话还没有可下载的分析结果", 409, "report_empty")
        response = Response("\ufeff" + csv_body, mimetype="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=hyrox-analysis.csv"
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get("/api/report.txt")
    def download_text_report() -> tuple[Response, int] | Response:
        report, _ = available_report(current_session())
        if not report["frames"]:
            return json_error("当前会话还没有可下载的分析结果", 409, "report_empty")
        response = Response("\ufeff" + render_text_report(report), mimetype="text/plain")
        response.headers["Content-Disposition"] = "attachment; filename=hyrox-analysis.txt"
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.delete("/api/session")
    def delete_session_data() -> Response:
        session_id = current_session().session_id
        manager.delete(session_id)
        response = jsonify({"status": "deleted"})
        response.delete_cookie(SESSION_COOKIE)
        return response

    @sock.route("/ws/pose")
    def pose_socket(ws: Any) -> None:
        item: BrowserSession = current_session()
        csrf = request.args.get("csrf", "")
        if not csrf or not hmac.compare_digest(csrf, item.csrf_token):
            ws.send(json.dumps({"type": "error", "code": "csrf_failed", "message": "连接安全令牌无效"}, ensure_ascii=False))
            ws.close(reason=1008, message="csrf_failed")
            return
        origin = request.headers.get("Origin", "")
        if origin:
            origin_parts = urlsplit(origin)
            if origin_parts.netloc != request.host:
                ws.send(json.dumps({"type": "error", "code": "origin_rejected", "message": "连接来源不受信任"}, ensure_ascii=False))
                ws.close(reason=1008, message="origin_rejected")
                return

        realtime: RealtimePoseSession = item.realtime
        realtime.mark_connected()
        ws.send(
            json.dumps(
                {
                    "type": "connected",
                    "protocol_version": 1,
                    "max_frame_bytes": 512 * 1024,
                    "max_receive_fps": 30,
                    "state": realtime.snapshot(),
                },
                ensure_ascii=False,
            )
        )
        try:
            while True:
                message: str | bytes | None | object
                try:
                    message = ws.receive(timeout=0.05)
                except TimeoutError:
                    message = Ellipsis
                if message is None:
                    if getattr(ws, "connected", True):
                        message = Ellipsis
                    else:
                        break
                if message is not Ellipsis:
                    item.last_seen = time.monotonic()
                if isinstance(message, str):
                    try:
                        payload = json.loads(message)
                        message_type = str(payload.get("type", ""))
                        if message_type == "start":
                            if not manager.can_activate(item.session_id):
                                raise RealtimeProtocolError("server_busy", "实时分析人数已满，请稍后重试")
                            state_value = realtime.start(payload.get("settings") or {})
                            ws.send(json.dumps({"type": "started", "state": state_value}, ensure_ascii=False))
                        elif message_type == "settings":
                            state_value = realtime.update_settings(payload.get("settings") or {})
                            ws.send(json.dumps({"type": "state", "state": state_value}, ensure_ascii=False))
                        elif message_type == "stop":
                            state_value = realtime.stop()
                            ws.send(json.dumps({"type": "stopped", "state": state_value}, ensure_ascii=False))
                        elif message_type == "ping":
                            ws.send(json.dumps({"type": "pong", "client_time": payload.get("client_time")}, ensure_ascii=False))
                        else:
                            raise RealtimeProtocolError("unknown_message", "无法识别的 WebSocket 消息")
                    except json.JSONDecodeError:
                        raise RealtimeProtocolError("invalid_json", "WebSocket JSON 消息格式错误")
                elif isinstance(message, bytes):
                    accepted = realtime.submit(message)
                    if not accepted:
                        ws.send(json.dumps({"type": "frame_dropped", "reason": "rate_limited"}, ensure_ascii=False))

                result = realtime.next_result()
                if result is not None:
                    ws.send(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        except RealtimeProtocolError as exc:
            ws.send(json.dumps(exc.as_message(), ensure_ascii=False))
        except (ConnectionClosed, ConnectionError, OSError):
            pass
        finally:
            realtime.mark_disconnected()

    return app


__all__ = ["PoseStreamEngine", "create_app"]
