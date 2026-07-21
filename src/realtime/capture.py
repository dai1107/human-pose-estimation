"""Input capture lifecycle for camera and video sources."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

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

    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            LOGGER.warning(
                "Camera %s could not open with CAP_DSHOW; retrying the default OpenCV backend",
                args.camera,
            )
            capture = cv2.VideoCapture(args.camera)
    else:
        capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.camera_fourcc.strip():
        fourcc = args.camera_fourcc.strip().upper()
        if len(fourcc) != 4:
            capture.release()
            raise AppError(
                "CFG002",
                "--camera-fourcc 必须是 4 个字符或空字符串",
                exit_code=ExitCode.CONFIG_ERROR,
            )
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not capture.isOpened():
        capture.release()
        raise InputSourceError(
            f"无法打开摄像头 {args.camera}",
            hint="检查设备占用、权限和摄像头编号，可用 doctor --camera 复查",
        )
    fps = capture.get(cv2.CAP_PROP_FPS)
    LOGGER.info(
        "Camera %s opened (requested FPS: %g, reported FPS: %.1f, FourCC: %s)",
        args.camera,
        args.camera_fps,
        fps if fps > 0 else 0,
        args.camera_fourcc or "unchanged",
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
