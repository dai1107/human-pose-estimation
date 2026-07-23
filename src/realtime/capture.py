"""Input capture lifecycle for camera and video sources."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

from src.camera.backend_benchmark import (
    DEFAULT_BACKEND_CACHE,
    backend_api_code,
    decode_fourcc,
    select_cached_backend,
    supported_backend_names,
)
from src.runtime_logging import AppError, ExitCode, InputSourceError

LOGGER = logging.getLogger("pose.desktop")


def open_capture(args: argparse.Namespace) -> tuple[cv2.VideoCapture, str, float]:
    if args.input_video:
        capture = cv2.VideoCapture(args.input_video)
        if not capture.isOpened():
            capture.release()
            raise InputSourceError(
                f"无法打开输入视频：{args.input_video}",
                hint="确认路径、文件权限和视频编码，或先运行 doctor",
            )
        fps = capture.get(cv2.CAP_PROP_FPS)
        return capture, "video", fps if fps > 0 else 30.0

    fourcc = args.camera_fourcc.strip().upper()
    if fourcc and len(fourcc) != 4:
        raise AppError(
            "CFG002",
            "--camera-fourcc 必须是 4 个字符或空字符串",
            exit_code=ExitCode.CONFIG_ERROR,
        )
    requested_api = str(getattr(args, "camera_api", "auto")).strip().lower()
    if requested_api not in {"auto", "default", "dshow", "msmf"}:
        raise AppError(
            "CFG002",
            "--camera-api 必须是 auto、default、dshow 或 msmf",
            exit_code=ExitCode.CONFIG_ERROR,
        )
    supported = supported_backend_names()
    if requested_api != "auto" and requested_api not in supported:
        raise AppError(
            "CFG002",
            f"当前平台不支持摄像头后端 {requested_api}",
            exit_code=ExitCode.CONFIG_ERROR,
        )
    cache_path = getattr(args, "camera_backend_cache", str(DEFAULT_BACKEND_CACHE))
    cached_api = (
        select_cached_backend(
            cache_path,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            fourcc=fourcc,
        )
        if requested_api == "auto"
        else None
    )
    primary_api = requested_api if requested_api != "auto" else (cached_api or "default")
    candidates = list(dict.fromkeys((primary_api, "default", *supported)))
    capture = None
    selected_api = ""
    for candidate in candidates:
        api_code = backend_api_code(candidate)
        candidate_capture = (
            cv2.VideoCapture(args.camera)
            if api_code is None
            else cv2.VideoCapture(args.camera, api_code)
        )
        if candidate_capture.isOpened():
            capture = candidate_capture
            selected_api = candidate
            break
        candidate_capture.release()
        LOGGER.warning(
            "Camera %s could not open with backend %s",
            args.camera,
            candidate,
        )
    if capture is None:
        raise InputSourceError(
            f"无法打开摄像头 {args.camera}",
            hint="检查设备占用、权限和摄像头编号，可运行 pose-camera-benchmark 复查后端",
        )
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    fps = capture.get(cv2.CAP_PROP_FPS)
    actual_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fourcc = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
    try:
        reported_backend = capture.getBackendName()
    except (AttributeError, cv2.error):
        reported_backend = selected_api
    LOGGER.info(
        "Camera %s opened (selection=%s, backend=%s, requested=%dx%d@%g %s, "
        "actual=%.0fx%.0f@%.1f %s, cache=%s)",
        args.camera,
        selected_api,
        reported_backend,
        args.width,
        args.height,
        args.camera_fps,
        fourcc or "unchanged",
        actual_width,
        actual_height,
        fps if fps > 0 else 0,
        actual_fourcc or "unknown",
        "hit" if cached_api else "miss",
    )
    return capture, "camera", fps if fps > 0 else 30.0


def read_capture_frame(
    capture: cv2.VideoCapture,
    *,
    input_mode: str,
    processed_frames: int,
) -> tuple[bool, object | None]:
    ok, frame = capture.read()
    if ok and frame is not None:
        return True, frame
    if input_mode == "camera":
        raise InputSourceError(
            "摄像头已断开或停止返回画面",
            hint="重新连接摄像头后再启动程序",
        )
    if processed_frames == 0:
        raise InputSourceError(
            "视频为空、损坏或没有可解码画面",
            hint="用播放器确认文件完整，并检查 OpenCV 是否支持该编码",
        )
    return False, None


def timestamp_for_frame(input_mode: str, started_ns: int, frame_index: int, fps: float) -> int:
    if input_mode == "video":
        return int(round(frame_index * 1000.0 / max(fps, 1.0)))
    return int((time.monotonic_ns() - started_ns) / 1_000_000)
