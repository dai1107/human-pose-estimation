from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import queue
import struct
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from hyrox.view_policy import CAMERA_VIEWS
from src.biomechanics.kinematics_3d import (
    ThreeDKinematicsTracker,
    summarize_three_d_records,
)
from src.backends.mediapipe_backend import MediaPipeBackend
from src.backends.base import Keypoint, PoseResult
from src.backends.yolo_guided_mediapipe_backend import YoloGuidedMediaPipeBackend
from src.backends.yolo_pose_backend import YoloPoseBackend
from src.backends.yolo_rtmw_backend import YoloRtmwWholeBodyBackend
from src.utils.backend_policy import resolve_backend_choice
from src.utils.device import resolve_torch_device
from src.utils.smoothing import KeypointSmoother
from src.utils.keypoint_schema import MEDIAPIPE_33_NAMES, MEDIAPIPE_CONNECTIONS
from src.output_schema import artifact_metadata, versioned_csv_columns, versioned_csv_row
from src.product_pose import (
    RealtimeSmoothingConfig,
    ThreeDKinematicsConfig,
    ThreeDQualityConfig,
)
from src.latency_audit import derive_web_latencies, summarize_latency_samples
from webui.analysis import RepVoiceFeedbackTracker, assess_action, enrich_report, visible_feedback
from webui.hands import (
    WebHandOverlay,
    hand_overlay_visible,
    rtmw_hand_detections,
    serialize_hand_overlay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_HEADER = struct.Struct(">I")
FRAME_V2_PREFIX = struct.Struct(">4sI")
FRAME_V2_MAGIC = b"PSV2"
MAX_FRAME_METADATA_BYTES = 4096
MAX_FRAME_BYTES = 512 * 1024
MAX_FRAME_WIDTH = 1280
MAX_FRAME_HEIGHT = 720
MAX_FRAME_PIXELS = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT
DEFAULT_RECEIVE_FPS = 30.0
REPORT_RETENTION_SECONDS = 10 * 60
DISCONNECT_GRACE_SECONDS = 30
WEB_CLIENT_TIMING_FIELDS = {
    "camera_frame_presented_ms",
    "frame_copy_start_ms",
    "frame_copy_end_ms",
    "encode_start_ms",
    "encode_end_ms",
    "socket_send_ms",
    "expected_display_time_ms",
    "capture_time_ms",
    "media_time_s",
    "presented_frames",
}
WEB_LATENCY_AUDIT_FIELDS = WEB_CLIENT_TIMING_FIELDS | {
    "server_receive_ms",
    "inference_start_ms",
    "inference_end_ms",
    "socket_result_send_ms",
    "client_result_receive_ms",
    "pose_render_start_ms",
    "pose_render_end_ms",
    "video_frame_presented_at_render_ms",
    "source_video_frame_id",
    "current_video_frame_id",
    "current_video_presented_frames",
    "rvfc_lateness_ms",
    "rvfc_late_frame_count",
    "presented_frame_skip_count",
    "main_thread_long_task_count",
    "main_thread_long_task_duration_ms",
    "render_loop_p95_ms",
    "canvas_draw_p95_ms",
    "dom_update_p95_ms",
    "long_task_render_count",
    "long_task_dom_update_count",
    "long_task_frame_copy_count",
    "long_task_encode_count",
    "long_task_pose_transfer_count",
    "long_task_other_count",
    "local_pose_inference_ms",
    "local_pose_dropped_frames",
}
WEB_FRAME_META_FIELDS = {
    "sessionId",
    "frameId",
    "presentedFrames",
    "mediaTime",
    "presentationTime",
    "expectedDisplayTime",
    "captureTime",
    "processingDuration",
    "width",
    "height",
    "callbackSource",
}
CAMERA_DIAGNOSTIC_WARNINGS = {
    "fps_below_requested",
    "low_light",
    "frame_interval_unstable",
    "duplicate_frames",
}
CAMERA_DIAGNOSTIC_FIELDS = {
    "settings",
    "requestedFps",
    "actualPresentedFps",
    "frameIntervalP50Ms",
    "frameIntervalP95Ms",
    "frameIntervalAnomalyRatio",
    "brightnessMean",
    "duplicateFrameRatio",
    "sampleCount",
    "warnings",
}
CAMERA_SETTING_FIELDS = {
    "width",
    "height",
    "frameRate",
    "deviceId",
    "resizeMode",
    "facingMode",
}
FORBIDDEN_ANALYSIS_PREDICTION_FIELDS = {
    "predicted_landmarks",
    "display_landmarks",
    "prediction_landmarks",
    "prediction",
}

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


class StaleFrameContext(RuntimeError):
    """Internal control flow for an observation whose browser context changed."""


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    jpeg: bytes
    received_at: float
    client_capture_ms: float | None = None
    connection_generation: int = 0
    timing: dict[str, float] = field(default_factory=dict)
    frame_meta: dict[str, Any] = field(default_factory=dict)
    pose_result: PoseResult | None = None
    frame_width: int = 0
    frame_height: int = 0
    source: str = "server_mediapipe"


@dataclass(frozen=True, slots=True)
class FrameRequest:
    frame_id: int
    client_capture_ms: float | None
    session_id: str
    run_id: str
    action: str
    backend: str
    jpeg: bytes
    timing: dict[str, float] = field(default_factory=dict)
    frame_meta: dict[str, Any] = field(default_factory=dict)


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


def _validate_frame_media(jpeg: bytes) -> None:
    if len(jpeg) > MAX_FRAME_BYTES:
        raise RealtimeProtocolError("frame_too_large", "单帧不能超过 512 KB")
    is_jpeg = jpeg.startswith(b"\xff\xd8\xff")
    is_webp = len(jpeg) >= 12 and jpeg[:4] == b"RIFF" and jpeg[8:12] == b"WEBP"
    if not (is_jpeg or is_webp):
        raise RealtimeProtocolError("unsupported_frame", "仅接受 JPEG 或 WebP 摄像头帧")


def _browser_landmarks(values: Any, *, world: bool = False) -> list[Keypoint]:
    if not isinstance(values, list) or len(values) > len(MEDIAPIPE_33_NAMES):
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器姿态关键点数量无效")
    points: list[Keypoint] = []
    coordinate_limit = 100.0 if world else 10.0
    for index, name in enumerate(MEDIAPIPE_33_NAMES):
        if index >= len(values):
            break
        item = values[index]
        if not isinstance(item, Mapping):
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器姿态关键点格式无效")
        numbers: dict[str, float] = {}
        for field_name, default in (("x", 0.0), ("y", 0.0), ("z", 0.0), ("visibility", 1.0), ("presence", 1.0)):
            value = item.get(field_name, default)
            if isinstance(value, bool):
                raise RealtimeProtocolError("invalid_pose_frame", f"关键点 {field_name} 无效")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RealtimeProtocolError("invalid_pose_frame", f"关键点 {field_name} 无效") from exc
            if not math.isfinite(number):
                raise RealtimeProtocolError("invalid_pose_frame", f"关键点 {field_name} 无效")
            numbers[field_name] = number
        if any(abs(numbers[field_name]) > coordinate_limit for field_name in ("x", "y", "z")):
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器姿态坐标超出范围")
        visibility = min(1.0, max(0.0, numbers["visibility"]))
        presence = min(1.0, max(0.0, numbers["presence"]))
        points.append(Keypoint(
            name=name,
            x=numbers["x"],
            y=numbers["y"],
            z=numbers["z"],
            confidence=min(visibility, presence),
            source_model="browser-mediapipe-world" if world else "browser-mediapipe",
            visibility=visibility,
            presence=presence,
        ))
    return points


def _browser_model_benchmark(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器模型基准格式无效")
    selected = str(value.get("selectedModel", "")).strip().lower()
    if selected not in {"lite", "full"}:
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器模型基准档位无效")
    raw_stats = value.get("stats")
    if not isinstance(raw_stats, Mapping):
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器模型基准统计无效")
    stats: dict[str, dict[str, float | int]] = {}
    for model in ("lite", "full"):
        raw_model = raw_stats.get(model)
        if not isinstance(raw_model, Mapping):
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器模型基准统计缺失")
        parsed: dict[str, float | int] = {}
        for name in ("inferenceP50Ms", "inferenceP95Ms", "poseFps", "detectionRate"):
            raw_number = raw_model.get(name, 0)
            if isinstance(raw_number, bool):
                raise RealtimeProtocolError("invalid_pose_frame", "浏览器模型基准数值无效")
            try:
                number = float(raw_number)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RealtimeProtocolError("invalid_pose_frame", "浏览器模型基准数值无效") from exc
            limit = 1.0 if name == "detectionRate" else 60_000.0
            if not math.isfinite(number) or not 0 <= number <= limit:
                raise RealtimeProtocolError("invalid_pose_frame", "浏览器模型基准数值无效")
            parsed[name] = round(number, 4)
        samples = raw_model.get("samples", 0)
        if isinstance(samples, bool) or not isinstance(samples, int) or not 0 <= samples <= 10_000:
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器模型基准样本数无效")
        parsed["samples"] = samples
        stats[model] = parsed
    long_tasks = value.get("mainThreadLongTaskCount", 0)
    if isinstance(long_tasks, bool) or not isinstance(long_tasks, int) or not 0 <= long_tasks <= 100_000:
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器长任务计数无效")
    return {
        "selected_model": selected,
        "selection_reason": str(value.get("reason", ""))[:256],
        "main_thread_long_task_count": long_tasks,
        "stats": stats,
    }


def _browser_display_filter(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器显示滤波摘要无效")
    profile = str(value.get("profile", ""))
    if profile != "ultra_responsive":
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器显示滤波档位无效")
    prediction_enabled = value.get("predictionEnabled", False)
    raw_blend_enabled = value.get("rawBlendEnabled", False)
    if not isinstance(prediction_enabled, bool) or not isinstance(raw_blend_enabled, bool):
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器显示滤波标志无效")
    parsed: dict[str, float | int | bool | str] = {
        "profile": profile,
        "prediction_enabled": prediction_enabled,
        "raw_blend_enabled": raw_blend_enabled,
    }
    blended_count = value.get("blendedPointCount", 0)
    if isinstance(blended_count, bool) or not isinstance(blended_count, int) or not 0 <= blended_count <= 66:
        raise RealtimeProtocolError("invalid_pose_frame", "浏览器显示混合节点数无效")
    parsed["blended_point_count"] = blended_count
    for source_name, target_name in (("meanRawWeight", "mean_raw_weight"), ("maxRawWeight", "max_raw_weight")):
        raw_number = value.get(source_name, 0)
        if isinstance(raw_number, bool):
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器显示混合权重无效")
        try:
            number = float(raw_number)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器显示混合权重无效") from exc
        if not math.isfinite(number) or not 0 <= number <= 0.45 + 1e-6:
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器显示混合权重无效")
        parsed[target_name] = round(number, 6)
    return parsed


def _keypoint_bbox(points: list[Keypoint]) -> tuple[float, float, float, float] | None:
    usable = [point for point in points if point.confidence >= 0.2]
    if not usable:
        return None
    return (
        min(point.x for point in usable),
        min(point.y for point in usable),
        max(point.x for point in usable),
        max(point.y for point in usable),
    )


def unpack_frame_request(message: bytes) -> FrameRequest:
    if len(message) <= FRAME_HEADER.size:
        raise RealtimeProtocolError("frame_too_small", "摄像头帧数据不完整")
    if message.startswith(FRAME_V2_MAGIC):
        if len(message) <= FRAME_V2_PREFIX.size:
            raise RealtimeProtocolError("frame_too_small", "摄像头帧元数据不完整")
        _, metadata_size = FRAME_V2_PREFIX.unpack_from(message, 0)
        if metadata_size <= 0 or metadata_size > MAX_FRAME_METADATA_BYTES:
            raise RealtimeProtocolError("invalid_frame_metadata", "摄像头帧元数据长度无效")
        metadata_end = FRAME_V2_PREFIX.size + metadata_size
        if len(message) <= metadata_end:
            raise RealtimeProtocolError("frame_too_small", "摄像头帧数据不完整")
        try:
            metadata = json.loads(
                message[FRAME_V2_PREFIX.size:metadata_end].decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RealtimeProtocolError("invalid_frame_metadata", "摄像头帧元数据无效") from exc
        if not isinstance(metadata, Mapping):
            raise RealtimeProtocolError("invalid_frame_metadata", "摄像头帧元数据必须是对象")
        frame_id = metadata.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or not 0 < frame_id <= 0xFFFFFFFF:
            raise RealtimeProtocolError("invalid_frame_id", "frame_id 必须是正整数")
        client_capture_ms = metadata.get("client_capture_ms")
        if isinstance(client_capture_ms, bool):
            raise RealtimeProtocolError("invalid_capture_time", "client_capture_ms 无效")
        try:
            client_capture_ms = float(client_capture_ms)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RealtimeProtocolError("invalid_capture_time", "client_capture_ms 无效") from exc
        if not math.isfinite(client_capture_ms) or client_capture_ms < 0:
            raise RealtimeProtocolError("invalid_capture_time", "client_capture_ms 无效")
        identity: dict[str, str] = {}
        for name in ("session_id", "run_id", "action", "backend"):
            resolved = str(metadata.get(name, "")).strip()
            if not resolved or len(resolved) > 128:
                raise RealtimeProtocolError(
                    "invalid_frame_metadata",
                    f"{name} 缺失或过长",
                )
            identity[name] = resolved
        timing_value = metadata.get("timing", {})
        if not isinstance(timing_value, Mapping):
            raise RealtimeProtocolError("invalid_frame_metadata", "timing 必须是对象")
        timing: dict[str, float] = {}
        for name, value in timing_value.items():
            if name not in WEB_CLIENT_TIMING_FIELDS or isinstance(value, bool):
                continue
            try:
                resolved_value = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(resolved_value) and resolved_value >= 0:
                timing[name] = resolved_value
        frame_meta_value = metadata.get("frame_meta", {})
        if not isinstance(frame_meta_value, Mapping):
            raise RealtimeProtocolError("invalid_frame_metadata", "frame_meta 必须是对象")
        frame_meta = {
            str(name): value
            for name, value in frame_meta_value.items()
            if name in WEB_FRAME_META_FIELDS
            and not isinstance(value, bool)
            and (value is None or isinstance(value, (str, int, float)))
        }
        if frame_meta:
            if frame_meta.get("frameId") != frame_id:
                raise RealtimeProtocolError("invalid_frame_metadata", "frame_meta.frameId 与 frame_id 不一致")
            if str(frame_meta.get("sessionId", "")) != identity["session_id"]:
                raise RealtimeProtocolError("invalid_frame_metadata", "frame_meta.sessionId 与 session_id 不一致")
            for name in (
                "presentedFrames", "mediaTime", "presentationTime", "expectedDisplayTime",
                "captureTime", "processingDuration", "width", "height",
            ):
                value = frame_meta.get(name)
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0):
                    raise RealtimeProtocolError("invalid_frame_metadata", f"frame_meta.{name} 无效")
            if frame_meta.get("callbackSource") not in {
                "requestVideoFrameCallback", "requestAnimationFrame",
            }:
                raise RealtimeProtocolError("invalid_frame_metadata", "frame_meta.callbackSource 无效")
        jpeg = message[metadata_end:]
        _validate_frame_media(jpeg)
        return FrameRequest(
            frame_id=frame_id,
            client_capture_ms=client_capture_ms,
            session_id=identity["session_id"],
            run_id=identity["run_id"],
            action=identity["action"],
            backend=identity["backend"],
            jpeg=jpeg,
            timing=timing,
            frame_meta=frame_meta,
        )

    sequence = FRAME_HEADER.unpack_from(message, 0)[0]
    jpeg = message[FRAME_HEADER.size :]
    _validate_frame_media(jpeg)
    return FrameRequest(
        frame_id=sequence,
        client_capture_ms=None,
        session_id="",
        run_id="",
        action="",
        backend="",
        jpeg=jpeg,
    )


def unpack_frame(message: bytes) -> tuple[int, bytes]:
    """Compatibility wrapper for protocol-v1 callers and report tooling."""

    request = unpack_frame_request(message)
    return request.frame_id, request.jpeg


def validate_settings(
    values: Mapping[str, Any],
    *,
    allow_experimental_backends: bool = False,
) -> dict[str, Any]:
    action = str(values.get("action", "lunge"))
    view = str(values.get("camera_view", "side"))
    sensitivity = str(values.get("sensitivity", "medium"))
    backend = str(values.get("backend", "mediapipe"))
    profile = str(values.get("landmark_profile", "full"))
    if action not in {"none", *HYROX_ACTION_NAMES}:
        raise RealtimeProtocolError("invalid_action", "无效的 HYROX 动作")
    if view not in CAMERA_VIEWS:
        raise RealtimeProtocolError("invalid_view", "无效的拍摄视角")
    if sensitivity not in {"low", "medium", "high"}:
        raise RealtimeProtocolError("invalid_sensitivity", "无效的灵敏度")
    allowed_backends = {"auto", "mediapipe"}
    if allow_experimental_backends:
        allowed_backends.update(
            {"yolo-mediapipe", "yolo-pose", "rtmw-wholebody"}
        )
    if backend not in allowed_backends:
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
        "show_fingers": bool(values.get("show_fingers", False)),
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
    resolved = resolve_backend_choice(
        requested,
        action_type=action,
        input_video="",
        product_mode=True,
    )
    if resolved != "mediapipe":
        raise RealtimeProtocolError(
            "experimental_backend_disabled",
            "产品实时会话只支持 MediaPipe Pose",
        )
    return (
        MediaPipeBackend(
            PROJECT_ROOT / "models" / "pose_landmarker_full.task",
            output_segmentation_masks=False,
        ),
        "mediapipe",
    )


def experimental_backend_factory(requested: str, action: str) -> tuple[Any, str]:
    """Explicit research/benchmark factory; never selected by product auto."""
    resolved = resolve_backend_choice(
        requested,
        action_type=action,
        input_video="",
        product_mode=False,
    )
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
    if resolved == "yolo-mediapipe":
        return (
            YoloGuidedMediaPipeBackend(
                PROJECT_ROOT / "yolo11n-pose.pt",
                PROJECT_ROOT / "models" / "pose_landmarker_full.task",
                target_select="tracking",
                device=resolve_torch_device("auto"),
            ),
            "yolo-guided-mediapipe",
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
    if resolved == "rtmw-wholebody":
        try:
            return (
                YoloRtmwWholeBodyBackend(
                    PROJECT_ROOT
                    / "models"
                    / "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
                    PROJECT_ROOT / "yolo11n-pose.pt",
                    target_select="tracking",
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
                    target_select="tracking",
                    device=resolve_torch_device("auto"),
                ),
                "yolo-guided-mediapipe-fallback",
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
        allow_experimental_backends: bool = False,
        realtime_smoothing_config: RealtimeSmoothingConfig | None = None,
        three_d_kinematics_config: ThreeDKinematicsConfig | None = None,
        three_d_quality_config: ThreeDQualityConfig | None = None,
        max_pose_age_ms: float = 150.0,
    ) -> None:
        self.session_id = session_id
        self._allow_experimental_backends = bool(allow_experimental_backends)
        self._backend_factory = (
            experimental_backend_factory
            if backend_factory is default_backend_factory
            and self._allow_experimental_backends
            else backend_factory
        )
        self._inference_gate = inference_gate or threading.BoundedSemaphore(1)
        self._max_receive_fps = max(1.0, float(max_receive_fps))
        self._retention_seconds = max(60, int(report_retention_seconds))
        self._realtime_smoothing_config = (
            RealtimeSmoothingConfig()
            if realtime_smoothing_config is None
            else realtime_smoothing_config
        )
        self._three_d_kinematics_config = (
            ThreeDKinematicsConfig()
            if three_d_kinematics_config is None
            else three_d_kinematics_config
        )
        self._three_d_quality_config = (
            ThreeDQualityConfig()
            if three_d_quality_config is None
            else three_d_quality_config
        )
        self._max_pose_age_ms = max(0.0, float(max_pose_age_ms))
        self._frames = LatestFrameQueue()
        self._results: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._settings = validate_settings(
            {},
            allow_experimental_backends=self._allow_experimental_backends,
        )
        self._state: dict[str, Any] = {
            "session_id": session_id,
            "run_id": "",
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
            "last_submitted_frame_id": 0,
            "paused": False,
            "error": "",
            "queue_dropped": 0,
        }
        self._history: deque[dict[str, Any]] = deque(maxlen=9000)
        self._pending_latency_audits: dict[int, tuple[dict[str, float], dict[str, float | None]]] = {}
        self._camera_diagnostics: dict[str, Any] = {}
        self._last_submit_at = 0.0
        self._connected = False
        self._disconnected_at: float | None = None
        self._report_expires_at: float | None = None
        self._voice_feedback = RepVoiceFeedbackTracker()
        self._run_generation = 0
        self._run_id = ""
        self._settings_revision = 0
        self._last_submitted_sequence = 0
        self._connection_generation = 0
        self._client_clock_offsets_ms: dict[int, float] = {}

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
            self._connection_generation += 1
            self._client_clock_offsets_ms.clear()
            self._connected = True
            self._disconnected_at = None
            while True:
                try:
                    self._results.get_nowait()
                except queue.Empty:
                    break

    def mark_disconnected(self) -> None:
        with self._lock:
            self._connected = False
            self._disconnected_at = time.monotonic()
            self._settings_revision += 1
            self._frames.clear()

    def start(self, values: Mapping[str, Any]) -> dict[str, Any]:
        settings = validate_settings(
            values,
            allow_experimental_backends=self._allow_experimental_backends,
        )
        with self._lock:
            new_run = self._thread is None or not self._thread.is_alive()
            if new_run:
                self._run_generation += 1
                self._run_id = f"{self._run_generation:x}-{time.monotonic_ns():x}"
                self._settings_revision += 1
                self._last_submitted_sequence = 0
                self._last_submit_at = 0.0
                self._history.clear()
                self._pending_latency_audits.clear()
                self._camera_diagnostics.clear()
                self._frames = LatestFrameQueue()
                while True:
                    try:
                        self._results.get_nowait()
                    except queue.Empty:
                        break
                self._voice_feedback.reset()
            elif settings != self._settings:
                self._settings_revision += 1
            self._settings.update(settings)
            self._state.update(
                {
                    "session_id": self.session_id,
                    "run_id": self._run_id,
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
                            "last_submitted_frame_id": 0,
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
        settings = validate_settings(
            merged,
            allow_experimental_backends=self._allow_experimental_backends,
        )
        with self._lock:
            if settings != self._settings:
                self._settings_revision += 1
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
        request = unpack_frame_request(message)
        now = time.monotonic()
        with self._lock:
            if not self._state["running"]:
                raise RealtimeProtocolError("not_started", "请先发送 start 消息")
            if request.session_id and request.session_id != self.session_id:
                raise RealtimeProtocolError("stale_session", "摄像头帧属于其他会话")
            if request.run_id and request.run_id != self._run_id:
                raise RealtimeProtocolError("stale_run", "摄像头帧属于已停止的运行")
            if request.action and request.action != str(self._settings["action"]):
                raise RealtimeProtocolError("stale_action", "摄像头帧属于旧动作")
            if request.backend and request.backend != str(self._settings["backend"]):
                raise RealtimeProtocolError("stale_backend", "摄像头帧属于旧后端")
            if request.frame_id <= self._last_submitted_sequence:
                raise RealtimeProtocolError("stale_frame", "frame_id 必须严格递增")
            if now - self._last_submit_at < 1.0 / self._max_receive_fps:
                return False
            self._last_submit_at = now
            self._last_submitted_sequence = request.frame_id
            connection_generation = self._connection_generation
            self._state["last_submitted_frame_id"] = request.frame_id
        self._frames.put_latest(
            FramePacket(
                sequence=request.frame_id,
                jpeg=request.jpeg,
                received_at=now,
                client_capture_ms=request.client_capture_ms,
                connection_generation=connection_generation,
                timing={**request.timing, "server_receive_ms": now * 1000.0},
                frame_meta=request.frame_meta,
            )
        )
        return True

    def submit_pose_frame(self, values: Mapping[str, Any]) -> bool:
        """Accept browser-local MediaPipe landmarks without rerunning the model."""

        if set(values) & FORBIDDEN_ANALYSIS_PREDICTION_FIELDS:
            raise RealtimeProtocolError(
                "invalid_pose_frame",
                "显示预测数据不得进入分析协议",
            )
        if str(values.get("source", "")) != "browser_mediapipe":
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器姿态来源无效")
        pose_model = str(values.get("pose_model", "")).strip().lower()
        if pose_model not in {"lite", "full"}:
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器姿态模型档位无效")
        model_benchmark = _browser_model_benchmark(values.get("pose_model_benchmark"))
        display_filter = _browser_display_filter(values.get("display_filter"))
        frame_id = values.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or not 0 < frame_id <= 0xFFFFFFFF:
            raise RealtimeProtocolError("invalid_frame_id", "frame_id 必须是正整数")
        image_points = _browser_landmarks(values.get("image_landmarks"), world=False)
        world_points = _browser_landmarks(values.get("world_landmarks"), world=True)
        inference_ms = values.get("pose_inference_ms", 0.0)
        capture_ms = values.get("capture_timestamp_ms")
        try:
            inference_ms = float(inference_ms)
            capture_ms = float(capture_ms)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器姿态时间戳无效") from exc
        if not math.isfinite(inference_ms) or inference_ms < 0 or inference_ms > 60_000:
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器推理耗时无效")
        if not math.isfinite(capture_ms) or capture_ms < 0:
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器采集时间无效")
        frame_meta_value = values.get("frame_meta")
        if not isinstance(frame_meta_value, Mapping):
            raise RealtimeProtocolError("invalid_pose_frame", "frame_meta 缺失")
        frame_meta = {
            str(name): value
            for name, value in frame_meta_value.items()
            if name in WEB_FRAME_META_FIELDS and not isinstance(value, bool)
        }
        if frame_meta.get("frameId") != frame_id:
            raise RealtimeProtocolError("invalid_pose_frame", "frame_meta.frameId 与 frame_id 不一致")
        if str(frame_meta.get("sessionId", "")) != self.session_id:
            raise RealtimeProtocolError("stale_session", "浏览器姿态属于其他会话")
        width = frame_meta.get("width", 0)
        height = frame_meta.get("height", 0)
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器姿态尺寸无效")
        width, height = int(width), int(height)
        if width <= 0 or height <= 0 or width > MAX_FRAME_WIDTH * 4 or height > MAX_FRAME_HEIGHT * 4:
            raise RealtimeProtocolError("invalid_pose_frame", "浏览器姿态尺寸无效")
        timing_value = values.get("timing", {})
        timing: dict[str, float] = {}
        if isinstance(timing_value, Mapping):
            for name, value in list(timing_value.items())[:64]:
                if name not in WEB_LATENCY_AUDIT_FIELDS:
                    continue
                if isinstance(value, bool):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(number):
                    timing[str(name)[:64]] = number
        result = PoseResult(
            keypoints=image_points,
            connections=MEDIAPIPE_CONNECTIONS,
            model_name=f"browser-mediapipe-{pose_model}",
            num_keypoints=len(image_points),
            success=bool(image_points),
            inference_time_ms=inference_ms,
            bbox=_keypoint_bbox(image_points),
            timestamp_ms=int(round(capture_ms)),
            extra={
                "world_keypoints": world_points,
                "world_landmarks_available": bool(world_points),
                "source": "browser_mediapipe",
                "pose_model": pose_model,
                "pose_model_benchmark": model_benchmark,
                "display_filter": display_filter,
            },
        )
        now = time.monotonic()
        with self._lock:
            if not self._state["running"]:
                raise RealtimeProtocolError("not_started", "请先发送 start 消息")
            if str(values.get("session_id", "")) != self.session_id:
                raise RealtimeProtocolError("stale_session", "浏览器姿态属于其他会话")
            if str(values.get("run_id", "")) != self._run_id:
                raise RealtimeProtocolError("stale_run", "浏览器姿态属于已停止的运行")
            if str(values.get("action", "")) != str(self._settings["action"]):
                raise RealtimeProtocolError("stale_action", "浏览器姿态属于旧动作")
            if str(values.get("backend", "")) != str(self._settings["backend"]):
                raise RealtimeProtocolError("stale_backend", "浏览器姿态属于旧后端")
            if frame_id <= self._last_submitted_sequence:
                raise RealtimeProtocolError("stale_frame", "frame_id 必须严格递增")
            if now - self._last_submit_at < 1.0 / self._max_receive_fps:
                return False
            self._last_submit_at = now
            self._last_submitted_sequence = frame_id
            connection_generation = self._connection_generation
            self._state["last_submitted_frame_id"] = frame_id
        self._frames.put_latest(FramePacket(
            sequence=frame_id,
            jpeg=b"",
            received_at=now,
            client_capture_ms=capture_ms,
            connection_generation=connection_generation,
            timing={**timing, "server_receive_ms": now * 1000.0},
            frame_meta=frame_meta,
            pose_result=result,
            frame_width=width,
            frame_height=height,
            source="browser_mediapipe",
        ))
        return True

    def record_latency_audit(self, values: Mapping[str, Any]) -> bool:
        """Attach browser render timestamps to an already processed frame."""

        frame_id = values.get("frame_id")
        timing_value = values.get("timing")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or not isinstance(timing_value, Mapping):
            raise RealtimeProtocolError("invalid_latency_audit", "延迟审计数据无效")
        unknown_fields = set(timing_value) - WEB_LATENCY_AUDIT_FIELDS
        if unknown_fields or len(timing_value) > len(WEB_LATENCY_AUDIT_FIELDS):
            raise RealtimeProtocolError(
                "invalid_latency_audit",
                "延迟审计包含未知字段",
            )
        timing: dict[str, float] = {}
        for name, value in timing_value.items():
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(number):
                timing[str(name)] = number
        derived = derive_web_latencies(timing)
        with self._lock:
            for frame in reversed(self._history):
                if frame.get("frame_id") == frame_id:
                    frame["latency_timing"] = timing
                    frame["latency"] = derived
                    return True
            self._pending_latency_audits[frame_id] = (timing, derived)
            while len(self._pending_latency_audits) > 256:
                self._pending_latency_audits.pop(next(iter(self._pending_latency_audits)))
        return True

    def record_camera_diagnostics(self, values: Mapping[str, Any]) -> bool:
        """Store bounded browser camera telemetry outside the analysis frame stream."""

        diagnostics_value = values.get("diagnostics")
        if not isinstance(diagnostics_value, Mapping):
            raise RealtimeProtocolError(
                "invalid_camera_diagnostics",
                "摄像头诊断数据无效",
            )
        if set(diagnostics_value) - CAMERA_DIAGNOSTIC_FIELDS:
            raise RealtimeProtocolError(
                "invalid_camera_diagnostics",
                "摄像头诊断包含未知字段",
            )
        settings_value = diagnostics_value.get("settings")
        if not isinstance(settings_value, Mapping):
            raise RealtimeProtocolError(
                "invalid_camera_diagnostics",
                "摄像头实际设置无效",
            )
        if set(settings_value) - CAMERA_SETTING_FIELDS:
            raise RealtimeProtocolError(
                "invalid_camera_diagnostics",
                "摄像头实际设置包含未知字段",
            )

        def finite_number(
            name: str,
            *,
            minimum: float,
            maximum: float,
            source: Mapping[str, Any] = diagnostics_value,
        ) -> float:
            raw = source.get(name)
            if isinstance(raw, bool):
                raise RealtimeProtocolError(
                    "invalid_camera_diagnostics",
                    f"摄像头诊断字段 {name} 无效",
                )
            try:
                number = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RealtimeProtocolError(
                    "invalid_camera_diagnostics",
                    f"摄像头诊断字段 {name} 无效",
                ) from exc
            if not math.isfinite(number) or not minimum <= number <= maximum:
                raise RealtimeProtocolError(
                    "invalid_camera_diagnostics",
                    f"摄像头诊断字段 {name} 超出范围",
                )
            return number

        width = finite_number("width", minimum=0, maximum=16_384, source=settings_value)
        height = finite_number("height", minimum=0, maximum=16_384, source=settings_value)
        frame_rate = finite_number("frameRate", minimum=0, maximum=1_000, source=settings_value)
        if width != int(width) or height != int(height):
            raise RealtimeProtocolError(
                "invalid_camera_diagnostics",
                "摄像头宽高必须为整数",
            )

        settings: dict[str, Any] = {
            "width": int(width),
            "height": int(height),
            "frameRate": frame_rate,
        }
        for name, maximum_length in (
            ("deviceId", 256),
            ("resizeMode", 64),
            ("facingMode", 64),
        ):
            raw = settings_value.get(name, "")
            if not isinstance(raw, str) or len(raw) > maximum_length:
                raise RealtimeProtocolError(
                    "invalid_camera_diagnostics",
                    f"摄像头实际设置字段 {name} 无效",
                )
            settings[name] = raw

        sample_count = diagnostics_value.get("sampleCount")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or not 0 <= sample_count <= 10_000_000
        ):
            raise RealtimeProtocolError(
                "invalid_camera_diagnostics",
                "摄像头诊断采样数无效",
            )
        warnings_value = diagnostics_value.get("warnings")
        if (
            not isinstance(warnings_value, list)
            or len(warnings_value) > len(CAMERA_DIAGNOSTIC_WARNINGS)
            or any(
                not isinstance(item, str) or item not in CAMERA_DIAGNOSTIC_WARNINGS
                for item in warnings_value
            )
            or len(set(warnings_value)) != len(warnings_value)
        ):
            raise RealtimeProtocolError(
                "invalid_camera_diagnostics",
                "摄像头诊断警告无效",
            )

        diagnostics = {
            "settings": settings,
            "requestedFps": finite_number(
                "requestedFps",
                minimum=1,
                maximum=1_000,
            ),
            "actualPresentedFps": finite_number(
                "actualPresentedFps",
                minimum=0,
                maximum=1_000,
            ),
            "frameIntervalP50Ms": finite_number(
                "frameIntervalP50Ms",
                minimum=0,
                maximum=60_000,
            ),
            "frameIntervalP95Ms": finite_number(
                "frameIntervalP95Ms",
                minimum=0,
                maximum=60_000,
            ),
            "frameIntervalAnomalyRatio": finite_number(
                "frameIntervalAnomalyRatio",
                minimum=0,
                maximum=1,
            ),
            "brightnessMean": finite_number(
                "brightnessMean",
                minimum=0,
                maximum=255,
            ),
            "duplicateFrameRatio": finite_number(
                "duplicateFrameRatio",
                minimum=0,
                maximum=1,
            ),
            "sampleCount": sample_count,
            "warnings": list(warnings_value),
        }
        with self._lock:
            self._camera_diagnostics = diagnostics
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
            self._pending_latency_audits.clear()
            self._camera_diagnostics.clear()
            self._report_expires_at = None

    def report(self) -> dict[str, Any]:
        with self._lock:
            frames = list(self._history)
            state = dict(self._state)
            camera_diagnostics = dict(self._camera_diagnostics)
        latency_samples = [frame["latency"] for frame in frames if isinstance(frame.get("latency"), Mapping)]
        return enrich_report({
            **artifact_metadata("web_realtime_pose_report"),
            "generated_at_unix_ms": int(time.time() * 1000),
            "retention_minutes": self._retention_seconds // 60,
            "privacy": "报告不包含原始或标注视频帧",
            "summary": {
                "action": state["action"],
                "action_label": state["action_label"],
                "reps": state["reps"],
                "candidate_count": state["candidate_count"],
                "pose_valid_rep_count": state["pose_valid_rep_count"],
                "no_rep_count": state["no_rep_count"],
                "unsure_count": state["unsure_count"],
                "last_phase": state["phase"],
                "processed_frames": len(frames),
                "dropped_frames": state["queue_dropped"],
                "backend": state["backend"],
                "three_d_kinematics": summarize_three_d_records(frames),
                "latency_audit": summarize_latency_samples(latency_samples),
                "camera_diagnostics": camera_diagnostics,
            },
            "frames": frames,
        })

    def report_csv(self) -> str:
        report = self.report()
        output = io.StringIO(newline="")
        fieldnames = versioned_csv_columns([
            "sequence", "frame_id", "client_capture_ms", "timestamp_unix_ms", "action", "phase", "reps", "pose_detected",
            "inference_ms", "server_ms", "pose_age_ms", "width", "height",
            "three_d_available", "three_d_reliable_ratio", "three_d_conflict_ratio",
            "three_d_assist_status", "three_d_kinematics_json",
            "three_d_assist_json",
            "feedback", "keypoints_json",
            "frame_meta_json",
            "capture_to_submit_ms", "submit_to_result_ms", "result_to_render_ms",
            "render_to_expected_display_ms", "pose_age_at_render_ms",
            "video_frame_age_at_render_ms", "pose_video_age_difference_ms",
            "latency_timing_json",
        ])
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for frame in report["frames"]:
            metrics = frame.get("metrics", {})
            latency = frame.get("latency", {})
            writer.writerow(
                versioned_csv_row({
                    "sequence": frame.get("sequence"),
                    "frame_id": frame.get("frame_id"),
                    "client_capture_ms": frame.get("client_capture_ms"),
                    "timestamp_unix_ms": frame.get("timestamp_unix_ms"),
                    "action": frame.get("action"),
                    "phase": frame.get("phase"),
                    "reps": frame.get("reps"),
                    "pose_detected": frame.get("pose_detected"),
                    "inference_ms": metrics.get("inference_ms"),
                    "server_ms": metrics.get("server_ms"),
                    "pose_age_ms": metrics.get("pose_age_ms"),
                    "width": metrics.get("width"),
                    "height": metrics.get("height"),
                    "three_d_available": metrics.get("three_d_available"),
                    "three_d_reliable_ratio": metrics.get("three_d_reliable_ratio"),
                    "three_d_conflict_ratio": metrics.get("three_d_conflict_ratio"),
                    "three_d_assist_status": metrics.get("three_d_assist_status"),
                    "three_d_kinematics_json": json.dumps(
                        frame.get("three_d_kinematics", {}),
                        ensure_ascii=False,
                    ),
                    "three_d_assist_json": json.dumps(
                        frame.get("last_three_d_assist", {}),
                        ensure_ascii=False,
                    ),
                    "feedback": json.dumps(frame.get("feedback", []), ensure_ascii=False),
                    "keypoints_json": json.dumps(frame.get("keypoints", []), ensure_ascii=False),
                    "frame_meta_json": json.dumps(frame.get("frame_meta", {}), ensure_ascii=False),
                    "capture_to_submit_ms": latency.get("capture_to_submit_ms"),
                    "submit_to_result_ms": latency.get("submit_to_result_ms"),
                    "result_to_render_ms": latency.get("result_to_render_ms"),
                    "render_to_expected_display_ms": latency.get("render_to_expected_display_ms"),
                    "pose_age_at_render_ms": latency.get("pose_age_at_render_ms"),
                    "video_frame_age_at_render_ms": latency.get("video_frame_age_at_render_ms"),
                    "pose_video_age_difference_ms": latency.get("pose_video_age_difference_ms"),
                    "latency_timing_json": json.dumps(frame.get("latency_timing", {}), ensure_ascii=False),
                })
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

    def _context_is_current(
        self,
        *,
        settings_revision: int,
        run_id: str,
        connection_generation: int,
    ) -> bool:
        with self._lock:
            return bool(
                self._state["running"]
                and not self._stop_event.is_set()
                and run_id == self._run_id
                and settings_revision == self._settings_revision
                and connection_generation == self._connection_generation
            )

    def _run(self) -> None:
        backend: Any | None = None
        backend_name = "-"
        backend_request: tuple[str, str] | None = None
        smoother = self._new_smoother()
        three_d_tracker = self._new_three_d_tracker()
        analyzer: Any | None = None
        analyzer_key: tuple[str, str, str] | None = None
        hand_overlay = WebHandOverlay(PROJECT_ROOT / "models" / "hand_landmarker.task")
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
                    settings_revision = self._settings_revision
                    run_id = self._run_id
                if settings["paused"]:
                    continue
                try:
                    message_backend_name = (
                        packet.pose_result.model_name
                        if packet.pose_result is not None
                        else "browser-mediapipe"
                    )
                    if packet.pose_result is None:
                        requested_backend = (str(settings["backend"]), str(settings["action"]))
                        if backend is None or backend_request != requested_backend:
                            if backend is not None:
                                backend.close()
                            backend, backend_name = self._backend_factory(settings["backend"], settings["action"])
                            backend_request = requested_backend
                            smoother = self._new_smoother()
                            three_d_tracker.reset()
                        message_backend_name = backend_name
                    message = self._process_packet(
                        packet,
                        backend,
                        message_backend_name,
                        smoother,
                        three_d_tracker,
                        settings,
                        settings_revision,
                        run_id,
                        analyzer,
                        analyzer_key,
                        hand_overlay,
                    )
                    analyzer = message.pop("_analyzer")
                    analyzer_key = message.pop("_analyzer_key")
                    context_is_current = self._context_is_current(
                        settings_revision=settings_revision,
                        run_id=run_id,
                        connection_generation=packet.connection_generation,
                    )
                    if not context_is_current:
                        smoother.reset()
                        three_d_tracker.reset()
                        self._publish(
                            {
                                "type": "frame_dropped",
                                "reason": "stale_context",
                                "session_id": self.session_id,
                                "run_id": run_id,
                                "frame_id": packet.sequence,
                                "sequence": packet.sequence,
                            }
                        )
                        continue
                    now = time.perf_counter()
                    instant_fps = 1.0 / max(now - last_processed, 1e-6)
                    smooth_fps = instant_fps if smooth_fps <= 0 else smooth_fps * 0.8 + instant_fps * 0.2
                    last_processed = now
                    message["metrics"]["fps"] = round(smooth_fps, 1)
                    history_item = {key: value for key, value in message.items() if key != "type"}
                    with self._lock:
                        pending_audit = self._pending_latency_audits.pop(packet.sequence, None)
                        if pending_audit is not None:
                            history_item["latency_timing"], history_item["latency"] = pending_audit
                        self._history.append(history_item)
                        self._state.update(
                            {
                                "running": True,
                                "status": "running",
                                "status_text": "分析中",
                                "backend": message_backend_name,
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
                                "frame_index": packet.sequence,
                                "queue_dropped": self._frames.dropped,
                                "error": "",
                            }
                        )
                    self._publish(message)
                except StaleFrameContext:
                    smoother.reset()
                    three_d_tracker.reset()
                    self._publish(
                        {
                            "type": "frame_dropped",
                            "reason": "stale_context",
                            "session_id": self.session_id,
                            "run_id": run_id,
                            "frame_id": packet.sequence,
                            "sequence": packet.sequence,
                        }
                    )
                except RealtimeProtocolError as exc:
                    self._publish(
                        {
                            **exc.as_message(),
                            "session_id": self.session_id,
                            "run_id": run_id,
                            "frame_id": packet.sequence,
                        }
                    )
                except Exception as exc:
                    error = RealtimeProtocolError("inference_failed", f"姿态识别失败：{exc}")
                    with self._lock:
                        self._state.update({"status": "error", "status_text": "识别出错", "error": str(exc)})
                    self._publish(
                        {
                            **error.as_message(),
                            "session_id": self.session_id,
                            "run_id": run_id,
                            "frame_id": packet.sequence,
                        }
                    )
        finally:
            hand_overlay.close()
            if backend is not None:
                backend.close()
            with self._lock:
                self._state["running"] = False
                if self._state["status"] != "error":
                    self._state.update({"status": "idle", "status_text": "连接已释放"})
                self._report_expires_at = time.monotonic() + self._retention_seconds

    def _new_smoother(self) -> KeypointSmoother:
        return KeypointSmoother.from_config(
            self._realtime_smoothing_config,
            max_missing_frames=5,
            occlusion_guard=True,
        )

    def _new_three_d_tracker(self) -> ThreeDKinematicsTracker:
        return ThreeDKinematicsTracker(
            self._three_d_kinematics_config,
            self._three_d_quality_config,
            max_pose_age_ms=self._max_pose_age_ms,
        )

    def _process_packet(
        self,
        packet: FramePacket,
        backend: Any,
        backend_name: str,
        smoother: KeypointSmoother,
        three_d_tracker: ThreeDKinematicsTracker,
        settings: Mapping[str, Any],
        settings_revision: int,
        run_id: str,
        analyzer: Any | None,
        analyzer_key: tuple[str, str, str] | None,
        hand_overlay: WebHandOverlay,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        frame: np.ndarray | None = None
        if packet.pose_result is None:
            array = np.frombuffer(packet.jpeg, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is None or frame.ndim != 3:
                raise RealtimeProtocolError("decode_failed", "无法解码摄像头帧")
            height, width = frame.shape[:2]
            if width <= 0 or height <= 0 or width > MAX_FRAME_WIDTH or height > MAX_FRAME_HEIGHT or width * height > MAX_FRAME_PIXELS:
                raise RealtimeProtocolError("invalid_dimensions", "摄像头帧最大支持 1280×720")
        else:
            width, height = packet.frame_width, packet.frame_height

        if packet.client_capture_ms is not None:
            with self._lock:
                clock_offset_ms = self._client_clock_offsets_ms.get(
                    packet.connection_generation
                )
                if clock_offset_ms is None:
                    clock_offset_ms = packet.received_at * 1000.0 - packet.client_capture_ms
                    self._client_clock_offsets_ms[packet.connection_generation] = clock_offset_ms
            capture_timestamp_ms = packet.client_capture_ms + clock_offset_ms
            inference_timestamp_ms = int(round(capture_timestamp_ms))
            capture_timestamp_ns = int(round(capture_timestamp_ms * 1_000_000.0))
        else:
            capture_timestamp_ns = int(round(packet.received_at * 1_000_000_000.0))
            inference_timestamp_ms = int(round(capture_timestamp_ns / 1_000_000.0))
        with self._inference_gate:
            inference_start_ms = time.monotonic() * 1000.0
            if packet.pose_result is not None:
                detected = packet.pose_result
            elif backend is not None and frame is not None:
                detected = backend.detect(frame, timestamp_ms=inference_timestamp_ms)
            else:
                raise RuntimeError("server pose backend is unavailable")
            inference_end_ms = time.monotonic() * 1000.0
            result = smoother.smooth_result(
                detected,
                capture_timestamp_ns=capture_timestamp_ns,
            )
            result, three_d_result = three_d_tracker.attach(
                result,
                capture_timestamp_ns=capture_timestamp_ns,
                pose_age_ms=max(0.0, (time.perf_counter() - packet.received_at) * 1000.0),
            )
            show_hand_overlay = frame is not None and hand_overlay_visible(
                str(settings["landmark_profile"]),
                bool(settings["show_fingers"]),
            )
            hand_detections = (
                rtmw_hand_detections(result.extra)
                if show_hand_overlay
                else {}
            )
            hand_detections = hand_detections or (
                hand_overlay.update(
                    frame,
                    timestamp_ms=inference_timestamp_ms,
                    enabled=show_hand_overlay,
                )
                if frame is not None
                else {}
            )
            if hand_detections and result.extra.get("rtmw_hand_keypoints"):
                hand_overlay.update(  # type: ignore[arg-type]
                    frame,
                    timestamp_ms=inference_timestamp_ms,
                    enabled=False,
                )
        if not self._context_is_current(
            settings_revision=settings_revision,
            run_id=run_id,
            connection_generation=packet.connection_generation,
        ):
            raise StaleFrameContext
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
            if features is not None:
                features["three_d_kinematics"] = three_d_result.as_dict()
            action_state = analyzer.attach_view_context(
                analyzer.update(features, timestamp_ms=inference_timestamp_ms)
            )

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
        hand_keypoints, hand_connections = serialize_hand_overlay(hand_detections)
        keypoints.extend(hand_keypoints)
        connections.extend(hand_connections)
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
        last_three_d_assist = (
            dict(action_state.get("last_three_d_assist") or {})
            if isinstance(action_state, Mapping)
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
        pose_age_ms = (time.monotonic() - packet.received_at) * 1000.0
        latency_timing = {
            **packet.timing,
            "inference_start_ms": inference_start_ms,
            "inference_end_ms": inference_end_ms,
        }
        return {
            "type": "result",
            "session_id": self.session_id,
            "run_id": self._run_id,
            "frame_id": packet.sequence,
            "sequence": packet.sequence,
            "client_capture_ms": packet.client_capture_ms,
            "frame_meta": packet.frame_meta,
            "source": packet.source,
            "pose_model_benchmark": result.extra.get("pose_model_benchmark", {}),
            "display_filter": result.extra.get("display_filter", {}),
            "server_inference_ms": round(float(result.inference_time_ms), 1),
            "pose_age_ms": round(pose_age_ms, 1),
            "latency_timing": latency_timing,
            "timestamp_unix_ms": int(time.time() * 1000),
            "action": settings["action"],
            "camera_view": settings["camera_view"],
            "request_backend": settings["backend"],
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
            "last_rep_decision": last_rep_decision,
            "last_rep_observability": last_rep_observability,
            "last_three_d_assist": last_three_d_assist,
            "pose_detected": has_pose,
            "hands_detected": bool(hand_keypoints),
            "feedback": feedback,
            "detected_issues": detected_issues,
            "assessment": assessment,
            "voice_feedback": voice_feedback,
            "keypoints": keypoints,
            "connections": connections,
            "world_angles": three_d_result.as_dict()["angles_3d"],
            "three_d_kinematics": three_d_result.as_dict(),
            "counts": {
                "reps": reps,
                "candidate_count": candidate_count,
                "pose_valid_rep_count": pose_valid_rep_count,
                "no_rep_count": no_rep_count,
                "unsure_count": unsure_count,
            },
            "metrics": {
                "backend": backend_name,
                "inference_ms": round(float(result.inference_time_ms), 1),
                "server_ms": round(server_ms, 1),
                "pose_age_ms": round(pose_age_ms, 1),
                "queue_dropped": self._frames.dropped,
                "three_d_available": three_d_result.three_d_available,
                "three_d_reliable_ratio": round(
                    three_d_result.three_d_reliable_ratio,
                    4,
                ),
                "three_d_conflict_ratio": round(
                    three_d_result.three_d_conflict_ratio,
                    4,
                ),
                "three_d_assist_status": three_d_result.assist_status,
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
        realtime_smoothing_config: RealtimeSmoothingConfig | None = None,
        three_d_kinematics_config: ThreeDKinematicsConfig | None = None,
        three_d_quality_config: ThreeDQualityConfig | None = None,
        max_pose_age_ms: float = 150.0,
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
        self._realtime_smoothing_config = realtime_smoothing_config
        self._three_d_kinematics_config = three_d_kinematics_config
        self._three_d_quality_config = three_d_quality_config
        self._max_pose_age_ms = max(0.0, float(max_pose_age_ms))
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
                else RealtimePoseSession(
                    new_id,
                    inference_gate=self._inference_gate,
                    realtime_smoothing_config=self._realtime_smoothing_config,
                    three_d_kinematics_config=self._three_d_kinematics_config,
                    three_d_quality_config=self._three_d_quality_config,
                    max_pose_age_ms=self._max_pose_age_ms,
                )
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
    "experimental_backend_factory",
    "unpack_frame",
    "validate_settings",
    "validate_manual_floor_points",
]
