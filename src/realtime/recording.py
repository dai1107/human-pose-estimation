"""Video and screenshot output helpers for the desktop runtime."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from src.runtime_logging import OutputWriteError


def create_writer(path: str, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    output = Path(path)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(
            f"无法创建输出目录：{output.parent}",
            hint="检查磁盘空间和目录权限",
        ) from exc
    height, width = frame_shape[:2]
    fourcc_name = "mp4v" if output.suffix.lower() == ".mp4" else "XVID"
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*fourcc_name),
        max(1.0, min(60.0, fps)),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise OutputWriteError(
            f"无法创建视频输出：{output}",
            hint="检查磁盘空间、扩展名和编码支持",
        )
    return writer


def make_output_path(directory_name: str, suffix: str, root: str | Path = "outputs") -> Path:
    output_dir = Path(root) / directory_name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputWriteError(
            f"无法创建输出目录：{output_dir}",
            hint="检查磁盘空间和目录权限",
        ) from exc
    stem = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    candidate = output_dir / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def save_screenshot(frame: object, root: str | Path = "outputs") -> Path:
    path = make_output_path("screenshots", ".png", root=root)
    if not cv2.imwrite(str(path), frame):
        raise OutputWriteError(
            f"无法保存截图：{path}",
            hint="检查磁盘空间和目录权限",
        )
    return path
