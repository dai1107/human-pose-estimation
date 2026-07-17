from __future__ import annotations

import csv
import io
import json
import math
import os
import queue
import struct
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from hyrox.view_policy import CAMERA_VIEWS
from src.backends.mediapipe_backend import MediaPipeBackend
from src.backends.yolo_pose_backend import YoloPoseBackend
from src.utils.backend_policy import resolve_backend_choice
from src.utils.device import resolve_torch_device
from src.utils.smoothing import KeypointSmoother
from webui.analysis import RepVoiceFeedbackTracker, assess_action, enrich_report, visible_feedback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_HEADER = struct.Struct(">I")
MAX_FRAME_BYTES = 512 * 1024
MAX_FRAME_WIDTH = 1280
MAX_FRAME_HEIGHT = 720
MAX_FRAME_PIXELS = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT
DEFAULT_RECEIVE_FPS = 30.0
REPORT_RETENTION_SECONDS = 10 * 60
DISCONNECT_GRACE_SECONDS = 30

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
FACE_NAMES = {
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
}
FINGER_NAMES = {"left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb"}
UPPER_BODY_NAMES = {
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip",
}
LOWER_BODY_NAMES = {
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
}


class RealtimeProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_message(self) -> dict[str, str]:
        return {"type": "error", "code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    jpeg: bytes
    received_at: float


class LatestFrameQueue:
    """A one-slot queue that always keeps the newest submitted frame."""

    def __init__(self) -> None:
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self.dropped = 0

    def put_latest(self, packet: FramePacket) -> None:
        try:
            self._queue.put_nowait(packet)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self.dropped += 1
        self._queue.put_nowait(packet)

    def get(self, timeout: float | None = None) -> FramePacket:
        return self._queue.get(timeout=timeout)

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


def unpack_frame(message: bytes) -> tuple[int, bytes]:
    if len(message) <= FRAME_HEADER.size:
        raise RealtimeProtocolError("frame_too_small", "摄像头帧数据不完整")
    sequence = FRAME_HEADER.unpack_from(message, 0)[0]
    jpeg = message[FRAME_HEADER.size :]
    if len(jpeg) > MAX_FRAME_BYTES:
        raise RealtimeProtocolError("frame_too_large", "单帧不能超过 512 KB")
    is_jpeg = jpeg.startswith(b"\xff\xd8\xff")
    is_webp = len(jpeg) >= 12 and jpeg[:4] == b"RIFF" and jpeg[8:12] == b"WEBP"
    if not (is_jpeg or is_webp):
        raise RealtimeProtocolError("unsupported_frame", "仅接受 JPEG 或 WebP 摄像头帧")
    return sequence, jpeg


def validate_settings(values: Mapping[str, Any]) -> dict[str, Any]:
    action = str(values.get("action", "lunge"))
    view = str(values.get("camera_view", "side"))
    sensitivity = str(values.get("sensitivity", "medium"))
    backend = str(values.get("backend", "auto"))
    profile = str(values.get("landmark_profile", "full"))
    if action not in {"none", *HYROX_ACTION_NAMES}:
        raise RealtimeProtocolError("invalid_action", "无效的 HYROX 动作")
    if view not in CAMERA_VIEWS:
        raise RealtimeProtocolError("invalid_view", "无效的拍摄视角")
    if sensitivity not in {"low", "medium", "high"}:
        raise RealtimeProtocolError("invalid_sensitivity", "无效的灵敏度")
    if backend not in {"auto", "mediapipe", "yolo-pose"}:
        raise RealtimeProtocolError("invalid_backend", "无效的识别模型")
    if profile not in {"full", "no-face", "upper-body", "lower-body"}:
        raise RealtimeProtocolError("invalid_profile", "无效的骨架显示模式")
    manual_floor_points = validate_manual_floor_points(values.get("manual_floor_points", ()))
    return {
        "action": action,
        "camera_view": view,
        "sensitivity": sensitivity,
        "backend": backend,
        "landmark_profile": profile,
        "show_fingers": bool(values.get("show_fingers", True)),
        "mirror": bool(values.get("mirror", True)),
        "paused": bool(values.get("paused", False)),
        "manual_floor_points": manual_floor_points,
    }


def validate_manual_floor_points(value: object) -> list[list[float]]:
    if value in (None, (), []):
        return []
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RealtimeProtocolError("invalid_floor_points", "手动地板线需要两个点")
    resolved: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise RealtimeProtocolError("invalid_floor_points", "地板点必须包含 x、y 坐标")
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RealtimeProtocolError("invalid_floor_points", "地板点坐标无效") from exc
        if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise RealtimeProtocolError("invalid_floor_points", "地板点必须位于画面内")
        resolved.append([x, y])
    if abs(resolved[1][0] - resolved[0][0]) <= 0.05:
        raise RealtimeProtocolError("invalid_floor_points", "两个地板点需要有足够的水平间距")
    slope = abs(
        (resolved[1][1] - resolved[0][1])
        / (resolved[1][0] - resolved[0][0])
    )
    if slope > 1.0:
        raise RealtimeProtocolError("invalid_floor_points", "地板线倾斜过大，请重新点击")
    return resolved


def default_backend_factory(requested: str, action: str) -> tuple[Any, str]:
    resolved = resolve_backend_choice(requested, action_type=action, input_video="")
    if resolved == "mediapipe":
        # Avoid a native MediaPipe crash seen on Windows when mask generation
        # receives certain camera/video dimensions. Pose landmarks are enough
        # for the web analyzer and remain enabled.
        return (
            MediaPipeBackend(
                PROJECT_ROOT / "models" / "pose_landmarker_full.task",
                output_segmentation_masks=False,
            ),
            resolved,
        )
    if resolved == "yolo-pose":
        return (
            YoloPoseBackend(
                str(PROJECT_ROOT / "yolo11n-pose.pt"),
                target_select="tracking",
                device=resolve_torch_device("auto"),
            ),
            resolved,
        )
    raise RealtimeProtocolError("invalid_backend", "无法选择识别模型")


def _feedback_items(state: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not state:
        return []
    messages = state.get("feedback_messages") or []
    items: list[dict[str, str]] = []
    for message in list(messages)[:3]:
        if isinstance(message, Mapping):
            text = str(message.get("text", ""))
            level = str(message.get("level", "info"))
            code = str(message.get("code", ""))
            confidence = message.get("confidence")
        else:
            text = str(getattr(message, "text", ""))
            level = str(getattr(message, "level", "info"))
            code = str(getattr(message, "code", ""))
            confidence = getattr(message, "confidence", None)
        if text:
            item: dict[str, Any] = {"level": level, "code": code, "text": text}
            if isinstance(confidence, (int, float)):
                item["confidence"] = round(max(0.0, min(1.0, float(confidence))), 3)
            items.append(item)
    return items


def _profile_names(profile: str, all_names: set[str], *, show_fingers: bool = True) -> set[str]:
    if profile == "no-face":
        visible = all_names - FACE_NAMES
    elif profile == "upper-body":
        visible = all_names & UPPER_BODY_NAMES
    elif profile == "lower-body":
        visible = all_names & LOWER_BODY_NAMES
    else:
        visible = set(all_names)
    return visible if show_fingers else visible - FINGER_NAMES


class RealtimePoseSession:
    """Per-browser pose state, newest-frame queue and downloadable result history."""

    def __init__(
        self,
        session_id: str,
        *,
        backend_factory: Callable[[str, str], tuple[Any, str]] = default_backend_factory,
        inference_gate: threading.BoundedSemaphore | None = None,
        max_receive_fps: float = DEFAULT_RECEIVE_FPS,
        report_retention_seconds: int = REPORT_RETENTION_SECONDS,
    ) -> None:
        self.session_id = session_id
        self._backend_factory = backend_factory
        self._inference_gate = inference_gate or threading.BoundedSemaphore(1)
        self._max_receive_fps = max(1.0, float(max_receive_fps))
        self._retention_seconds = max(60, int(report_retention_seconds))
        self._frames = LatestFrameQueue()
        self._results: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._settings = validate_settings({})
        self._state: dict[str, Any] = {
            "running": False,
            "status": "idle",
            "status_text": "等待开启本机摄像头",
            "source_mode": "browser-camera",
            "source_name": "本机摄像头",
            "backend": "-",
            "action": "lunge",
            "action_label": ACTION_LABELS["lunge"],
            "camera_view": "side",
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
            "paused": False,
            "error": "",
            "queue_dropped": 0,
        }
        self._history: deque[dict[str, Any]] = deque(maxlen=9000)
        self._last_submit_at = 0.0
        self._connected = False
        self._disconnected_at: float | None = None
        self._report_expires_at: float | None = None
        self._voice_feedback = RepVoiceFeedbackTracker()

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._state["running"])

    @property
    def disconnected_at(self) -> float | None:
        with self._lock:
            return self._disconnected_at

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def mark_connected(self) -> None:
        with self._lock:
            self._connected = True
            self._disconnected_at = None

    def mark_disconnected(self) -> None:
        with self._lock:
            self._connected = False
            self._disconnected_at = time.monotonic()

    def start(self, values: Mapping[str, Any]) -> dict[str, Any]:
        settings = validate_settings(values)
        with self._lock:
            new_run = self._thread is None or not self._thread.is_alive()
            if new_run:
                self._history.clear()
                self._frames = LatestFrameQueue()
                self._voice_feedback.reset()
            self._settings.update(settings)
            self._state.update(
                {
                    "running": True,
                    "status": "starting",
                    "status_text": "正在加载识别模型…",
                    "backend": settings["backend"],
                    "action": settings["action"],
                    "action_label": ACTION_LABELS[settings["action"]],
                    "camera_view": settings["camera_view"],
                    "sensitivity": settings["sensitivity"],
                    "paused": settings["paused"],
                    "error": "",
                    **(
                        {
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
                            "queue_dropped": 0,
                        }
                        if new_run
                        else {}
                    ),
                }
            )
            self._report_expires_at = None
            if new_run:
                self._stop_event = threading.Event()
                self._thread = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name=f"browser-pose-{self.session_id[:8]}",
                )
                self._thread.start()
            return dict(self._state)

    def update_settings(self, values: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(self._settings)
        merged.update(values)
        settings = validate_settings(merged)
        with self._lock:
            self._settings.update(settings)
            self._state.update(
                {
                    "action": settings["action"],
                    "action_label": ACTION_LABELS[settings["action"]],
                    "camera_view": settings["camera_view"],
                    "sensitivity": settings["sensitivity"],
                    "paused": settings["paused"],
                    "status_text": "已暂停" if settings["paused"] else "分析中",
                }
            )
            return dict(self._state)

    def submit(self, message: bytes) -> bool:
        if not self.running:
            raise RealtimeProtocolError("not_started", "请先发送 start 消息")
        sequence, jpeg = unpack_frame(message)
        now = time.monotonic()
        if now - self._last_submit_at < 1.0 / self._max_receive_fps:
            return False
        self._last_submit_at = now
        self._frames.put_latest(FramePacket(sequence=sequence, jpeg=jpeg, received_at=now))
        return True

    def next_result(self, timeout: float = 0.0) -> dict[str, Any] | None:
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self, reason: str = "stopped") -> dict[str, Any]:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            self._thread = None
            self._frames.clear()
            self._state.update(
                {
                    "running": False,
                    "status": "idle",
                    "status_text": "已停止" if reason == "stopped" else "连接已释放",
                    "paused": False,
                }
            )
            self._report_expires_at = time.monotonic() + self._retention_seconds
            return dict(self._state)

    def should_release_after_disconnect(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            return bool(
                self._state["running"]
                and not self._connected
                and self._disconnected_at is not None
                and now - self._disconnected_at >= DISCONNECT_GRACE_SECONDS
            )

    def report_expired(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            return self._report_expires_at is not None and now >= self._report_expires_at

    def clear_results(self) -> None:
        with self._lock:
            self._history.clear()
            self._report_expires_at = None

    def report(self) -> dict[str, Any]:
        with self._lock:
            frames = list(self._history)
            state = dict(self._state)
        return enrich_report({
            "schema_version": 1,
            "generated_at_unix_ms": int(time.time() * 1000),
            "retention_minutes": self._retention_seconds // 60,
            "privacy": "报告不包含原始或标注视频帧",
            "summary": {
                "action": state["action"],
                "action_label": state["action_label"],
                "reps": state["reps"],
                "last_phase": state["phase"],
                "processed_frames": len(frames),
                "dropped_frames": state["queue_dropped"],
                "backend": state["backend"],
            },
            "frames": frames,
        })

    def report_csv(self) -> str:
        report = self.report()
        output = io.StringIO(newline="")
        fieldnames = [
            "sequence", "timestamp_unix_ms", "action", "phase", "reps", "pose_detected",
            "inference_ms", "server_ms", "width", "height", "feedback", "keypoints_json",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for frame in report["frames"]:
            metrics = frame.get("metrics", {})
            writer.writerow(
                {
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
                }
            )
        return output.getvalue()

    def _publish(self, message: dict[str, Any]) -> None:
        try:
            self._results.put_nowait(message)
            return
        except queue.Full:
            pass
        try:
            self._results.get_nowait()
        except queue.Empty:
            pass
        self._results.put_nowait(message)

    def _run(self) -> None:
        backend: Any | None = None
        backend_name = "-"
        backend_request: tuple[str, str] | None = None
        smoother = KeypointSmoother(mode="one-euro", max_missing_frames=5, occlusion_guard=True)
        analyzer: Any | None = None
        analyzer_key: tuple[str, str, str] | None = None
        last_processed = time.perf_counter()
        smooth_fps = 0.0
        try:
            while not self._stop_event.is_set():
                try:
                    packet = self._frames.get(timeout=0.25)
                except queue.Empty:
                    if self.should_release_after_disconnect():
                        break
                    continue
                with self._lock:
                    settings = dict(self._settings)
                if settings["paused"]:
                    continue
                try:
                    requested_backend = (str(settings["backend"]), str(settings["action"]))
                    if backend is None or backend_request != requested_backend:
                        if backend is not None:
                            backend.close()
                        backend, backend_name = self._backend_factory(settings["backend"], settings["action"])
                        backend_request = requested_backend
                        smoother = KeypointSmoother(mode="one-euro", max_missing_frames=5, occlusion_guard=True)
                    message = self._process_packet(packet, backend, backend_name, smoother, settings, analyzer, analyzer_key)
                    analyzer = message.pop("_analyzer")
                    analyzer_key = message.pop("_analyzer_key")
                    now = time.perf_counter()
                    instant_fps = 1.0 / max(now - last_processed, 1e-6)
                    smooth_fps = instant_fps if smooth_fps <= 0 else smooth_fps * 0.8 + instant_fps * 0.2
                    last_processed = now
                    message["metrics"]["fps"] = round(smooth_fps, 1)
                    history_item = {key: value for key, value in message.items() if key != "type"}
                    with self._lock:
                        self._history.append(history_item)
                        self._state.update(
                            {
                                "running": True,
                                "status": "running",
                                "status_text": "分析中",
                                "backend": backend_name,
                                "pose_detected": message["pose_detected"],
                                "fps": message["metrics"]["fps"],
                                "inference_ms": message["metrics"]["inference_ms"],
                                "phase": message["phase"],
                                "reps": message["reps"],
                                "candidate_count": message["candidate_count"],
                                "pose_valid_rep_count": message["pose_valid_rep_count"],
                                "no_rep_count": message["no_rep_count"],
                                "unsure_count": message["unsure_count"],
                                "floor_reference": message["floor_reference"],
                                "contacts": message["contacts"],
                                "foot_events": message["foot_events"],
                                "feedback": message["feedback"],
                                "voice_feedback": message["voice_feedback"],
                                "frame_index": len(self._history),
                                "queue_dropped": self._frames.dropped,
                                "error": "",
                            }
                        )
                    self._publish(message)
                except RealtimeProtocolError as exc:
                    self._publish(exc.as_message())
                except Exception as exc:
                    error = RealtimeProtocolError("inference_failed", f"姿态识别失败：{exc}")
                    with self._lock:
                        self._state.update({"status": "error", "status_text": "识别出错", "error": str(exc)})
                    self._publish(error.as_message())
        finally:
            if backend is not None:
                backend.close()
            with self._lock:
                self._state["running"] = False
                if self._state["status"] != "error":
                    self._state.update({"status": "idle", "status_text": "连接已释放"})
                self._report_expires_at = time.monotonic() + self._retention_seconds

    def _process_packet(
        self,
        packet: FramePacket,
        backend: Any,
        backend_name: str,
        smoother: KeypointSmoother,
        settings: Mapping[str, Any],
        analyzer: Any | None,
        analyzer_key: tuple[str, str, str] | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        array = np.frombuffer(packet.jpeg, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3:
            raise RealtimeProtocolError("decode_failed", "无法解码摄像头帧")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0 or width > MAX_FRAME_WIDTH or height > MAX_FRAME_HEIGHT or width * height > MAX_FRAME_PIXELS:
            raise RealtimeProtocolError("invalid_dimensions", "摄像头帧最大支持 1280×720")

        with self._inference_gate:
            result = smoother.smooth_result(backend.detect(frame, timestamp_ms=int(time.monotonic() * 1000)))
        has_pose = bool(result.success and result.keypoints)
        next_analyzer_key = (str(settings["action"]), str(settings["sensitivity"]), str(settings["camera_view"]))
        if next_analyzer_key != analyzer_key:
            analyzer = (
                create_action_analyzer(
                    next_analyzer_key[0],
                    sensitivity=next_analyzer_key[1],
                    camera_view=next_analyzer_key[2],
                    live_mode=True,
                )
                if next_analyzer_key[0] != "none"
                else None
            )
            analyzer_key = next_analyzer_key
        action_state: Mapping[str, Any] | None = None
        features = None
        if analyzer is not None:
            manual_points = settings.get("manual_floor_points") or []
            analyzer.set_manual_floor_line(
                manual_points[0] if len(manual_points) == 2 else None,
                manual_points[1] if len(manual_points) == 2 else None,
            )
            features = (
                extract_basic_pose_features(
                    result.keypoints,
                    width,
                    height,
                    segmentation_mask=result.extra.get("segmentation_mask"),
                )
                if has_pose
                else None
            )
            action_state = analyzer.attach_view_context(analyzer.update(features, timestamp_ms=int(time.monotonic() * 1000)))

        all_names = {point.name for point in result.keypoints}
        visible_names = _profile_names(
            str(settings["landmark_profile"]),
            all_names,
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
        names_by_index = [point.name for point in result.keypoints]
        connections = [
            [names_by_index[start], names_by_index[end]]
            for start, end in result.connections
            if start < len(names_by_index)
            and end < len(names_by_index)
            and names_by_index[start] in visible_names
            and names_by_index[end] in visible_names
        ]
        phase = "idle" if action_state is None else str(action_state.get("phase", "unknown"))
        reps = 0 if action_state is None else int(action_state.get("rep_count", 0))
        candidate_count = 0 if action_state is None else int(action_state.get("candidate_count", reps))
        pose_valid_rep_count = (
            reps
            if action_state is None
            else int(action_state.get("pose_valid_rep_count", reps))
        )
        no_rep_count = 0 if action_state is None else int(action_state.get("no_rep_count", 0))
        unsure_count = 0 if action_state is None else int(action_state.get("unsure_count", 0))
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
        evaluation_phase = str(action_debug.get("raw_phase", phase)) if isinstance(action_debug, Mapping) else phase
        detected_issues = _feedback_items(action_state)
        feedback = visible_feedback(detected_issues, evaluation_phase)
        assessment = assess_action(str(settings["action"]), evaluation_phase, features, feedback)
        voice_feedback = self._voice_feedback.update(
            action=str(settings["action"]),
            reps=reps,
            assessment=assessment,
            detected_issues=detected_issues,
            timestamp_ms=int(time.time() * 1000),
        )
        server_ms = (time.perf_counter() - started) * 1000.0
        return {
            "type": "result",
            "sequence": packet.sequence,
            "timestamp_unix_ms": int(time.time() * 1000),
            "action": settings["action"],
            "action_label": ACTION_LABELS[str(settings["action"])],
            "phase": phase,
            "reps": reps,
            "candidate_count": candidate_count,
            "pose_valid_rep_count": pose_valid_rep_count,
            "no_rep_count": no_rep_count,
            "unsure_count": unsure_count,
            "floor_reference": floor_reference,
            "contacts": contacts,
            "foot_events": foot_events,
            "pose_detected": has_pose,
            "feedback": feedback,
            "detected_issues": detected_issues,
            "assessment": assessment,
            "voice_feedback": voice_feedback,
            "keypoints": keypoints,
            "connections": connections,
            "metrics": {
                "backend": backend_name,
                "inference_ms": round(float(result.inference_time_ms), 1),
                "server_ms": round(server_ms, 1),
                "queue_dropped": self._frames.dropped,
                "width": width,
                "height": height,
                "fps": 0.0,
            },
            "_analyzer": analyzer,
            "_analyzer_key": analyzer_key,
        }


@dataclass(slots=True)
class BrowserSession:
    session_id: str
    csrf_token: str
    remote_ip: str
    engine: Any
    realtime: RealtimePoseSession
    uploads: dict[str, dict[str, Any]]
    created_at: float
    last_seen: float


class SessionCapacityError(RuntimeError):
    pass


class SessionManager:
    def __init__(
        self,
        *,
        engine_factory: Callable[[str], Any],
        realtime_factory: Callable[[str, threading.BoundedSemaphore], RealtimePoseSession] | None = None,
        storage_root: Path,
        max_sessions: int | None = None,
        max_active: int | None = None,
        per_ip_limit: int | None = None,
        session_ttl_seconds: int = REPORT_RETENTION_SECONDS,
        inference_concurrency: int | None = None,
    ) -> None:
        import secrets

        self._secrets = secrets
        self._engine_factory = engine_factory
        self._realtime_factory = realtime_factory
        self.storage_root = storage_root
        self.max_sessions = max_sessions or int(os.environ.get("POSE_MAX_SESSIONS", "100"))
        self.max_active = max_active or int(os.environ.get("POSE_MAX_ACTIVE", "50"))
        self.per_ip_limit = per_ip_limit or int(os.environ.get("POSE_MAX_SESSIONS_PER_IP", "10"))
        self.session_ttl_seconds = max(60, session_ttl_seconds)
        concurrency = inference_concurrency or int(os.environ.get("POSE_INFERENCE_CONCURRENCY", "1"))
        self._inference_gate = threading.BoundedSemaphore(max(1, concurrency))
        self._lock = threading.RLock()
        self._sessions: dict[str, BrowserSession] = {}

    def get_or_create(self, session_id: str | None, remote_ip: str) -> tuple[BrowserSession, bool]:
        now = time.monotonic()
        self.cleanup(now)
        with self._lock:
            if session_id and session_id in self._sessions:
                item = self._sessions[session_id]
                item.last_seen = now
                return item, False
            if len(self._sessions) >= self.max_sessions:
                raise SessionCapacityError("当前访问人数已满，请稍后重试")
            ip_count = sum(1 for item in self._sessions.values() if item.remote_ip == remote_ip)
            if ip_count >= self.per_ip_limit:
                raise SessionCapacityError("当前网络建立的会话过多，请稍后重试")
            new_id = self._secrets.token_urlsafe(32)
            realtime = (
                self._realtime_factory(new_id, self._inference_gate)
                if self._realtime_factory is not None
                else RealtimePoseSession(new_id, inference_gate=self._inference_gate)
            )
            item = BrowserSession(
                session_id=new_id,
                csrf_token=self._secrets.token_urlsafe(24),
                remote_ip=remote_ip,
                engine=self._engine_factory(new_id),
                realtime=realtime,
                uploads={},
                created_at=now,
                last_seen=now,
            )
            self._sessions[new_id] = item
            return item, True

    def can_activate(self, session_id: str) -> bool:
        with self._lock:
            active = sum(1 for item in self._sessions.values() if item.realtime.running)
            own_active = bool(self._sessions.get(session_id) and self._sessions[session_id].realtime.running)
            return own_active or active < self.max_active

    def delete(self, session_id: str) -> None:
        import shutil

        with self._lock:
            item = self._sessions.pop(session_id, None)
        if item is None:
            return
        item.realtime.stop(reason="deleted")
        item.engine.stop()
        for upload in item.uploads.values():
            Path(str(upload["path"])).unlink(missing_ok=True)
        target = (self.storage_root / session_id).resolve()
        root = self.storage_root.resolve()
        if target.parent == root and target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def cleanup(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        to_delete: list[str] = []
        with self._lock:
            items = list(self._sessions.items())
        for session_id, item in items:
            if item.realtime.should_release_after_disconnect(now):
                item.realtime.stop(reason="disconnected")
            stale = now - item.last_seen >= self.session_ttl_seconds
            if stale and not item.realtime.running:
                to_delete.append(session_id)
            elif item.realtime.report_expired(now):
                item.realtime.clear_results()
        for session_id in to_delete:
            self.delete(session_id)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "active_realtime": sum(1 for item in self._sessions.values() if item.realtime.running),
                "max_sessions": self.max_sessions,
                "max_active_realtime": self.max_active,
            }


__all__ = [
    "BrowserSession",
    "DISCONNECT_GRACE_SECONDS",
    "LatestFrameQueue",
    "RealtimePoseSession",
    "RealtimeProtocolError",
    "SessionCapacityError",
    "SessionManager",
    "unpack_frame",
    "validate_settings",
    "validate_manual_floor_points",
]
