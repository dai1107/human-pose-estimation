from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ModuleNotFoundError:
    mp = None
    mp_python = None
    vision = None

from src.biomechanics.hand_landmarks import (
    SUPPLEMENTAL_FINGER_CONNECTIONS,
    SUPPLEMENTAL_FINGER_DISPLAY_INDICES,
    coerce_hand_landmarks,
)
from src.biomechanics.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, coerce_landmark, coerce_landmarks
from src.biomechanics.normalization import normalize_landmarks
from src.biomechanics.session_writer import SessionConfig, SessionWriter
from src.biomechanics.types import KinematicFrame, LandmarkPoint, PoseFrame
from src.biomechanics.velocity import KinematicsProcessor
from src.ui.metrics_overlay import draw_metrics_overlay


POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

BODY_JOINTS: frozenset[int] = frozenset(range(11, 33))
UPPER_BODY_JOINTS: frozenset[int] = frozenset({11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24})
LOWER_BODY_JOINTS: frozenset[int] = frozenset({23, 24, 25, 26, 27, 28, 29, 30, 31, 32})
LANDMARK_PROFILES: dict[str, frozenset[int]] = {
    "full": frozenset(range(len(LANDMARK_NAMES))),
    "no-face": BODY_JOINTS,
    "upper-body": UPPER_BODY_JOINTS,
    "lower-body": LOWER_BODY_JOINTS,
}
PROFILE_LABELS: dict[str, str] = {
    "full": "FULL",
    "no-face": "NO FACE",
    "upper-body": "UPPER BODY",
    "lower-body": "LOWER BODY",
}
HAND_TIP_INDICES: frozenset[int] = frozenset({4, 8, 12, 16, 20})
POSE_HAND_OCCLUSION_INDICES: frozenset[int] = frozenset({15, 16, 17, 18, 19, 20, 21, 22})
POSE_OCCLUSION_GUARD_INDICES: frozenset[int] = frozenset({11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32})
HAND_OCCLUSION_RADIUS = 0.065
HAND_OCCLUSION_JUMP_THRESHOLD = 0.035
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 60.0
DEFAULT_CAMERA_FOURCC = "MJPG"
DEFAULT_DETECT_WIDTH = 480
DEFAULT_MAX_DETECT_FPS = 30.0
DEFAULT_MAX_HAND_DETECT_FPS = 18.0


@dataclass(frozen=True)
class DrawLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


class PoseResultStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: vision.PoseLandmarkerResult | None = None
        self._timestamp_ms: int = -1

    def update(self, result: vision.PoseLandmarkerResult, timestamp_ms: int) -> None:
        with self._lock:
            if timestamp_ms < self._timestamp_ms:
                return
            self._result = result
            self._timestamp_ms = timestamp_ms

    def snapshot(self) -> tuple[vision.PoseLandmarkerResult | None, int]:
        with self._lock:
            return self._result, self._timestamp_ms


class HandResultStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: vision.HandLandmarkerResult | None = None
        self._timestamp_ms: int = -1

    def update(self, result: vision.HandLandmarkerResult, timestamp_ms: int) -> None:
        with self._lock:
            if timestamp_ms < self._timestamp_ms:
                return
            self._result = result
            self._timestamp_ms = timestamp_ms

    def snapshot(self) -> tuple[vision.HandLandmarkerResult | None, int]:
        with self._lock:
            return self._result, self._timestamp_ms


@dataclass(frozen=True)
class HandDetection:
    side: str
    score: float
    landmarks: Sequence[object]
    world_landmarks: Sequence[object] | None = None


class SlidingFps:
    def __init__(self, window: int = 30) -> None:
        self._samples: deque[float] = deque(maxlen=max(2, window))

    def tick(self) -> float:
        now = time.perf_counter()
        self._samples.append(now)
        if len(self._samples) < 2:
            return 0.0
        elapsed = self._samples[-1] - self._samples[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._samples) - 1) / elapsed


def is_near_hand_occlusion(landmark: DrawLandmark, occlusion_points: Sequence[DrawLandmark]) -> bool:
    if not occlusion_points:
        return False
    target = np.array([landmark.x, landmark.y], dtype=float)
    if not np.all(np.isfinite(target)):
        return False
    for hand_point in occlusion_points:
        if hand_point.visibility < 0.05 or hand_point.presence < 0.05:
            continue
        candidate = np.array([hand_point.x, hand_point.y], dtype=float)
        if np.all(np.isfinite(candidate)) and float(np.linalg.norm(target - candidate)) <= HAND_OCCLUSION_RADIUS:
            return True
    return False


class LandmarkSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._previous: list[DrawLandmark] | None = None
        self._previous_timestamp_ms: int | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_timestamp_ms = None

    def smooth(
        self,
        landmarks: Sequence[object],
        timestamp_ms: int | None = None,
        occlusion_points: Sequence[DrawLandmark] | None = None,
        occlusion_guard_indices: frozenset[int] | None = None,
        self_occlusion_indices: frozenset[int] | None = None,
    ) -> list[DrawLandmark]:
        current = [
            DrawLandmark(
                x=coerced.x,
                y=coerced.y,
                z=coerced.z,
                visibility=coerced.visibility,
                presence=coerced.presence,
            )
            for coerced in (coerce_landmark(point) for point in landmarks)
        ]
        all_occlusion_points = list(occlusion_points or [])
        if self_occlusion_indices:
            all_occlusion_points.extend(current[index] for index in sorted(self_occlusion_indices) if index < len(current))
        if self.alpha <= 0.0 or self._previous is None or len(self._previous) != len(current):
            self._previous = current
            self._previous_timestamp_ms = timestamp_ms
            return current

        smoothed: list[DrawLandmark] = []
        dt = 1.0 / 30.0
        if timestamp_ms is not None and self._previous_timestamp_ms is not None:
            delta_ms = timestamp_ms - self._previous_timestamp_ms
            if 0 < delta_ms <= 1000:
                dt = delta_ms / 1000.0

        for index, (old, new) in enumerate(zip(self._previous, current)):
            confidence = max(0.0, min(1.0, (new.visibility + new.presence) / 2.0))
            displacement = float(np.linalg.norm(np.array([new.x - old.x, new.y - old.y, new.z - old.z], dtype=float)))
            speed = displacement / max(dt, 1e-3)
            dynamic_alpha = min(0.95, self.alpha + min(0.3, speed * 0.08))
            if displacement < 0.006:
                dynamic_alpha *= 0.65
            if confidence < 0.55:
                dynamic_alpha *= max(0.15, confidence)
            if displacement > 0.30 and confidence < 0.55:
                dynamic_alpha = 0.08
            if (
                occlusion_guard_indices is not None
                and index in occlusion_guard_indices
                and displacement > HAND_OCCLUSION_JUMP_THRESHOLD
                and is_near_hand_occlusion(new, all_occlusion_points)
            ):
                dynamic_alpha = min(dynamic_alpha, 0.04)
            dynamic_alpha = max(0.03, min(0.95, dynamic_alpha))
            keep = 1.0 - dynamic_alpha
            smoothed.append(
                DrawLandmark(
                    x=old.x * keep + new.x * dynamic_alpha,
                    y=old.y * keep + new.y * dynamic_alpha,
                    z=old.z * keep + new.z * dynamic_alpha,
                    visibility=new.visibility,
                    presence=new.presence,
                )
            )
        self._previous = smoothed
        self._previous_timestamp_ms = timestamp_ms
        return smoothed


class VideoRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self.writer is not None and self.writer.isOpened()

    def start(self, frame_shape: tuple[int, int, int], fps_hint: float) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame_shape[:2]
        fps = fps_hint if fps_hint >= 1.0 else 30.0
        fps = max(1.0, min(60.0, fps))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidates = ((".mp4", "mp4v"), (".avi", "XVID"))

        for suffix, fourcc_name in candidates:
            path = self.output_dir / f"pose_{stamp}{suffix}"
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if writer.isOpened():
                self.writer = writer
                self.path = path
                return path
            writer.release()

        raise RuntimeError(f"Could not create a video file in {self.output_dir}")

    def write(self, frame: np.ndarray) -> None:
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)

    def stop(self) -> Path | None:
        path = self.path
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.path = None
        return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime single-person pose detection with MediaPipe Pose Landmarker."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH, help=f"Requested camera width. Default: {DEFAULT_CAMERA_WIDTH}.")
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT, help=f"Requested camera height. Default: {DEFAULT_CAMERA_HEIGHT}.")
    parser.add_argument("--camera-fps", type=float, default=DEFAULT_CAMERA_FPS, help=f"Requested camera capture FPS. Use 0 for backend default. Default: {DEFAULT_CAMERA_FPS:g}.")
    parser.add_argument("--camera-fourcc", default=DEFAULT_CAMERA_FOURCC, help=f"Requested camera FourCC codec. Use empty string to leave unchanged. Default: {DEFAULT_CAMERA_FOURCC}.")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="Path to .task model file. Default: models/pose_landmarker_full.task.")
    parser.add_argument("--landmark-profile", default="no-face", choices=tuple(LANDMARK_PROFILES), help="Pose landmarks to draw at startup. Default: no-face.")
    parser.add_argument("--include-landmarks", default="", help="Comma-separated pose landmark names or indices to draw instead of the selected profile.")
    parser.add_argument("--exclude-landmarks", default="", help="Comma-separated pose landmark names or indices to hide from the selected profile.")
    parser.add_argument("--show-hands", action="store_true", help="Show supplemental five-finger landmarks at startup. You can also toggle them with H.")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="Path to Hand Landmarker .task model file.")
    parser.add_argument("--hand-detect-width", type=int, default=416, help="Resize hand detector input to this width for lower latency. Default: 416.")
    parser.add_argument("--max-hand-detect-fps", type=float, default=DEFAULT_MAX_HAND_DETECT_FPS, help=f"Maximum hand detector submissions per second. Default: {DEFAULT_MAX_HAND_DETECT_FPS:g}.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect. Default: 2.")
    parser.add_argument("--record", action="store_true", help="Start recording immediately.")
    parser.add_argument("--save-dir", default="outputs", help="Directory for sessions, screenshots, and recordings.")
    parser.add_argument("--metrics-overlay", action="store_true", help="Show kinematic metrics panel at startup.")
    parser.add_argument("--session-autostart", action="store_true", help="Start a kinematic data session automatically.")
    parser.add_argument("--camera-view", default="unknown", choices=("side", "front", "front_left", "front_right", "unknown"), help="Camera view for view-sensitive analysis. Default: unknown.")
    parser.add_argument("--detect-width", type=int, default=DEFAULT_DETECT_WIDTH, help=f"Resize detector input to this width for lower latency. Use 0 for full frame. Default: {DEFAULT_DETECT_WIDTH}.")
    parser.add_argument("--max-detect-fps", type=float, default=DEFAULT_MAX_DETECT_FPS, help=f"Maximum pose detector submissions per second. Default: {DEFAULT_MAX_DETECT_FPS:g}.")
    parser.add_argument("--max-pending-ms", type=int, default=180, help="Timeout for one pending async detection before submitting a new frame. Default: 180.")
    parser.add_argument("--max-result-lag-ms", type=int, default=280, help="Hide stale pose results older than this many ms. Default: 280.")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--plot-on-save", dest="plot_on_save", action="store_true", help="Generate PNG plots when saving a session. Default.")
    plot_group.add_argument("--no-plot-on-save", dest="plot_on_save", action="store_false", help="Skip PNG plots when saving a session.")
    parser.add_argument(
        "--smoothing",
        nargs="?",
        type=float,
        const=0.65,
        default=0.65,
        help="Landmark smoothing alpha from 0 to 1. Use 0 to disable. Default: 0.65.",
    )
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", help="Mirror display. Default.")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable mirror display.")
    parser.set_defaults(mirror=True)
    parser.set_defaults(plot_on_save=True)
    return parser.parse_args(argv)


def resolve_model_path(raw_path: str) -> Path:
    model_path = Path(raw_path).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    return model_path


def parse_pose_landmark_selector(raw_value: str) -> set[int]:
    indices: set[int] = set()
    if not raw_value.strip():
        return indices
    tokens = raw_value.replace(";", ",").split(",")
    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue
        if token in LANDMARK_INDEX:
            indices.add(LANDMARK_INDEX[token])
            continue
        if token.startswith("landmark_") and token.removeprefix("landmark_").isdigit():
            indices.add(int(token.removeprefix("landmark_")))
            continue
        if token.isdigit():
            indices.add(int(token))
            continue
        raise ValueError(f"unknown pose landmark selector: {token}")
    invalid = sorted(index for index in indices if index < 0 or index >= len(LANDMARK_NAMES))
    if invalid:
        raise ValueError(f"pose landmark index out of range: {invalid[0]}")
    return indices


def resolve_pose_draw_indices(profile: str, include_raw: str, exclude_raw: str) -> frozenset[int]:
    include = parse_pose_landmark_selector(include_raw)
    exclude = parse_pose_landmark_selector(exclude_raw)
    base = set(include) if include else set(LANDMARK_PROFILES[profile])
    base.difference_update(exclude)
    return frozenset(sorted(base))


def normalize_camera_fourcc(raw_value: str) -> str | None:
    value = raw_value.strip().upper()
    if not value:
        return None
    if len(value) != 4:
        raise ValueError("--camera-fourcc must be exactly 4 characters, or an empty string")
    return value


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    camera_fps: float,
    camera_fourcc: str | None,
) -> tuple[cv2.VideoCapture | None, str | None]:
    attempts: list[tuple[str, int | None]] = []
    if os.name == "nt":
        attempts.append(("DirectShow", cv2.CAP_DSHOW))
    attempts.append(("default", None))

    for backend, api_preference in attempts:
        if api_preference is None:
            capture = cv2.VideoCapture(camera_index)
        else:
            capture = cv2.VideoCapture(camera_index, api_preference)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if camera_fourcc is not None:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_fourcc))
        if width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if camera_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, camera_fps)
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                return capture, backend
        capture.release()
    return None, None


def next_timestamp_ms(previous_ms: int) -> int:
    current_ms = int(time.monotonic_ns() / 1_000_000)
    if current_ms <= previous_ms:
        return previous_ms + 1
    return current_ms


def is_visible(landmark: DrawLandmark, threshold: float = 0.2) -> bool:
    return (
        landmark.visibility >= threshold
        and landmark.presence >= threshold
        and -0.25 <= landmark.x <= 1.25
        and -0.25 <= landmark.y <= 1.25
    )


def to_pixel(landmark: DrawLandmark, width: int, height: int) -> tuple[int, int]:
    x = min(width - 1, max(0, int(round(landmark.x * width))))
    y = min(height - 1, max(0, int(round(landmark.y * height))))
    return x, y


def draw_pose(frame: np.ndarray, landmarks: Sequence[DrawLandmark], mode: str, point_indices: Iterable[int]) -> None:
    height, width = frame.shape[:2]
    point_set = set(point_indices)
    connections: Iterable[tuple[int, int]] = POSE_CONNECTIONS

    for start, end in connections:
        if start not in point_set or end not in point_set:
            continue
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        if not is_visible(landmarks[start]) or not is_visible(landmarks[end]):
            continue
        start_xy = to_pixel(landmarks[start], width, height)
        end_xy = to_pixel(landmarks[end], width, height)
        cv2.line(frame, start_xy, end_xy, (80, 220, 120), 2, cv2.LINE_AA)

    for index in sorted(point_set):
        if index >= len(landmarks) or not is_visible(landmarks[index]):
            continue
        center = to_pixel(landmarks[index], width, height)
        if index in HIGHLIGHT_JOINTS:
            cv2.circle(frame, center, 7, (0, 170, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, center, 9, (10, 30, 30), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, center, 4, (255, 210, 80), -1, cv2.LINE_AA)


def draw_hands(frame: np.ndarray, hands: dict[str, Sequence[DrawLandmark]]) -> None:
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
            if not is_visible(landmarks[start], threshold=0.05) or not is_visible(landmarks[end], threshold=0.05):
                continue
            cv2.line(frame, to_pixel(landmarks[start], width, height), to_pixel(landmarks[end], width, height), line_color, 2, cv2.LINE_AA)
        for index in sorted(SUPPLEMENTAL_FINGER_DISPLAY_INDICES):
            if index >= len(landmarks):
                continue
            landmark = landmarks[index]
            if not is_visible(landmark, threshold=0.05):
                continue
            radius = 5 if index in HAND_TIP_INDICES else 3
            cv2.circle(frame, to_pixel(landmark, width, height), radius, point_color, -1, cv2.LINE_AA)


def put_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    top_left = (max(0, x - 6), max(0, y - text_height - 7))
    bottom_right = (min(frame.shape[1] - 1, x + text_width + 6), min(frame.shape[0] - 1, y + baseline + 7))
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (20, 22, 24), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    camera_index: int,
    has_pose: bool,
    mode: str,
    mirror: bool,
    recording: bool,
    session_active: bool,
    session_id: str | None,
    result_lag_ms: int | None,
    backend: str,
    hands_enabled: bool = False,
    hands_detected: bool = False,
    hand_lag_ms: int | None = None,
) -> None:
    lines = [
        f"FPS: {fps:4.1f}",
        f"CAM: {camera_index} ({backend})",
        f"POSE: {'YES' if has_pose else 'NO'}",
        f"LAG: {result_lag_ms} ms" if result_lag_ms is not None else "LAG: N/A",
        f"SESSION: {'RECORDING' if session_active else 'IDLE'}",
        f"MODE: {PROFILE_LABELS.get(mode, mode.upper())}",
        f"MIRROR: {'ON' if mirror else 'OFF'}",
        f"REC: {'ON' if recording else 'OFF'}",
    ]
    if hands_enabled:
        lines.append(f"HANDS: {'YES' if hands_detected else 'NO'}")
        lines.append(f"HAND LAG: {hand_lag_ms} ms" if hand_lag_ms is not None else "HAND LAG: N/A")
    for row, text in enumerate(lines):
        color = (245, 245, 245)
        if text.startswith("POSE:"):
            color = (80, 230, 120) if has_pose else (60, 80, 255)
        if text.startswith("HANDS:"):
            color = (80, 230, 120) if hands_detected else (0, 190, 255)
        if text.startswith("LAG:") and result_lag_ms is not None and result_lag_ms > 180:
            color = (0, 180, 255)
        if text.startswith("HAND LAG:") and hand_lag_ms is not None and hand_lag_ms > 180:
            color = (0, 190, 255)
        if text.startswith("SESSION:") and session_active:
            color = (70, 90, 255)
        if text.startswith("REC:") and recording:
            color = (70, 80, 255)
        put_text(frame, text, (14, 28 + row * 30), color)

    if session_active and session_id:
        put_text(frame, f"ID: {session_id}", (14, 28 + len(lines) * 30), (245, 245, 245))

    if not has_pose:
        offset = len(lines) + (1 if session_active and session_id else 0)
        put_text(frame, "No pose detected", (14, 28 + offset * 30), (60, 80, 255))


def save_screenshot(frame: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not write screenshot: {path}")
    return path


def prepare_detection_frame(frame: np.ndarray, detect_width: int) -> np.ndarray:
    if detect_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= detect_width:
        return frame
    detect_height = max(1, int(round(height * (detect_width / width))))
    return cv2.resize(frame, (detect_width, detect_height), interpolation=cv2.INTER_AREA)


def frame_to_mp_image(frame: np.ndarray, detect_width: int) -> mp.Image:
    detection_frame = prepare_detection_frame(frame, detect_width)
    rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))


def to_landmark_points(landmarks: Sequence[object] | None) -> list[LandmarkPoint]:
    return coerce_landmarks(landmarks)


def to_hand_landmark_points(landmarks: Sequence[object] | None) -> list[LandmarkPoint]:
    return coerce_hand_landmarks(landmarks)


def infer_hand_side(landmarks: Sequence[object], fallback_index: int) -> str:
    xs = [point.x for point in (coerce_landmark(point) for point in landmarks) if np.isfinite(point.x)]
    if xs:
        return "left" if sum(xs) / len(xs) < 0.5 else "right"
    return "left" if fallback_index == 0 else "right"


def hand_side_from_handedness(
    result: vision.HandLandmarkerResult,
    hand_index: int,
    landmarks: Sequence[object],
) -> tuple[str, float]:
    handedness = getattr(result, "handedness", None) or []
    if hand_index < len(handedness) and handedness[hand_index]:
        category = handedness[hand_index][0]
        raw_name = str(getattr(category, "category_name", "") or getattr(category, "display_name", "")).strip().lower()
        raw_score = getattr(category, "score", 0.5)
        try:
            score = float(raw_score) if raw_score is not None else 0.5
        except (TypeError, ValueError):
            score = 0.5
        if raw_name in {"left", "right"}:
            return raw_name, score
    return infer_hand_side(landmarks, hand_index), 0.5


def extract_hand_detections(result: vision.HandLandmarkerResult | None) -> dict[str, HandDetection]:
    if result is None or not getattr(result, "hand_landmarks", None):
        return {}
    world_landmarks = getattr(result, "hand_world_landmarks", None) or []
    detections: dict[str, HandDetection] = {}
    for index, landmarks in enumerate(result.hand_landmarks):
        side, score = hand_side_from_handedness(result, index, landmarks)
        world = world_landmarks[index] if index < len(world_landmarks) else None
        existing = detections.get(side)
        if existing is not None and existing.score >= score:
            continue
        detections[side] = HandDetection(side=side, score=score, landmarks=landmarks, world_landmarks=world)
    return detections


def build_pose_frame(
    frame_index: int,
    timestamp_ms: int,
    result: vision.PoseLandmarkerResult | None,
    smoothed_landmarks: Sequence[object] | None,
    hand_detections: dict[str, HandDetection] | None,
    smoothed_hand_landmarks: dict[str, Sequence[object]] | None,
    hands_detected: bool,
    pose_detected: bool,
    mirror: bool,
    frame_shape: tuple[int, int, int],
    fps: float,
) -> PoseFrame:
    image_source = result.pose_landmarks[0] if pose_detected and result is not None and result.pose_landmarks else None
    world_source = None
    if pose_detected and result is not None and getattr(result, "pose_world_landmarks", None):
        if result.pose_world_landmarks:
            world_source = result.pose_world_landmarks[0]

    image_landmarks = to_landmark_points(image_source)
    world_landmarks = to_landmark_points(world_source)
    smoothed_points = to_landmark_points(smoothed_landmarks)
    hand_detections = hand_detections or {}
    smoothed_hand_landmarks = smoothed_hand_landmarks or {}
    image_hand_landmarks = {
        side: to_hand_landmark_points(detection.landmarks)
        for side, detection in hand_detections.items()
    }
    world_hand_landmarks = {
        side: to_hand_landmark_points(detection.world_landmarks)
        for side, detection in hand_detections.items()
    }
    smoothed_hand_points = {
        side: to_hand_landmark_points(points)
        for side, points in smoothed_hand_landmarks.items()
    }
    normalization = normalize_landmarks(smoothed_points if pose_detected else None)
    height, width = frame_shape[:2]
    return PoseFrame(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        pose_detected=pose_detected,
        image_landmarks=image_landmarks,
        world_landmarks=world_landmarks,
        smoothed_landmarks=smoothed_points,
        normalized_landmarks=normalization.landmarks,
        hands_detected=hands_detected,
        hand_landmarks=image_hand_landmarks,
        hand_world_landmarks=world_hand_landmarks,
        smoothed_hand_landmarks=smoothed_hand_points,
        normalization_success=normalization.success,
        normalization_message=normalization.message,
        mirror=mirror,
        camera_width=width,
        camera_height=height,
        fps=fps,
    )


def print_controls() -> None:
    print("Controls: Q/ESC quit, S screenshot, R record, M mirror, 1 full, F face on/off, 3 metrics, 6 no-face, 7 upper, 8 lower, H hands, C session")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.smoothing = max(0.0, min(1.0, float(args.smoothing)))
    args.max_detect_fps = max(0.0, float(args.max_detect_fps))
    args.max_hand_detect_fps = max(0.0, float(args.max_hand_detect_fps))
    args.max_pending_ms = max(1, int(args.max_pending_ms))
    args.max_result_lag_ms = max(1, int(args.max_result_lag_ms))
    args.max_hands = max(1, int(args.max_hands))
    args.hand_detect_width = max(0, int(args.hand_detect_width))
    args.camera_fps = max(0.0, float(args.camera_fps))
    try:
        camera_fourcc = normalize_camera_fourcc(str(args.camera_fourcc))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 6
    min_detect_interval_ms = int(round(1000.0 / args.max_detect_fps)) if args.max_detect_fps > 0 else 0
    min_hand_detect_interval_ms = int(round(1000.0 / args.max_hand_detect_fps)) if args.max_hand_detect_fps > 0 else 0
    model_path = resolve_model_path(args.model)
    hand_model_path = resolve_model_path(args.hand_model)
    save_dir = Path(args.save_dir)
    try:
        pose_draw_indices = resolve_pose_draw_indices(args.landmark_profile, args.include_landmarks, args.exclude_landmarks)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 6

    if mp is None or mp_python is None or vision is None:
        print(
            "ERROR: mediapipe is not installed. Install project dependencies with "
            "'python -m pip install -r requirements.txt' before running realtime detection.",
            file=sys.stderr,
        )
        return 5

    if not model_path.exists():
        print(
            "ERROR: model file not found.\n"
            f"Path: {model_path}\n"
            "Download a Pose Landmarker .task file into the models directory before running.",
            file=sys.stderr,
        )
        return 2

    if args.show_hands and not hand_model_path.exists():
        print(
            "ERROR: hand model file not found.\n"
            f"Path: {hand_model_path}\n"
            "Download a Hand Landmarker .task file into the models directory or run without --show-hands.",
            file=sys.stderr,
        )
        return 2

    capture, backend = open_camera(args.camera, args.width, args.height, args.camera_fps, camera_fourcc)
    if capture is None or backend is None:
        print(
            f"ERROR: camera {args.camera} could not be opened. "
            "Check that the camera is connected and not used by another app.",
            file=sys.stderr,
        )
        return 3

    result_store = PoseResultStore()
    hand_result_store = HandResultStore()

    def on_result(
        result: vision.PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        del output_image
        result_store.update(result, timestamp_ms)

    def on_hand_result(
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        del output_image
        hand_result_store.update(result, timestamp_ms)

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        result_callback=on_result,
    )

    try:
        landmarker = vision.PoseLandmarker.create_from_options(options)
    except Exception as exc:
        capture.release()
        print(f"ERROR: failed to initialize PoseLandmarker: {exc}", file=sys.stderr)
        return 4

    hand_landmarker = None
    if hand_model_path.exists():
        try:
            hand_options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_model_path)),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=args.max_hands,
                min_hand_detection_confidence=0.45,
                min_hand_presence_confidence=0.45,
                min_tracking_confidence=0.45,
                result_callback=on_hand_result,
            )
            hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        except Exception as exc:
            if args.show_hands:
                capture.release()
                print(f"ERROR: failed to initialize HandLandmarker: {exc}", file=sys.stderr)
                return 4
            print(f"WARNING: hand hotkey disabled because HandLandmarker failed to initialize: {exc}", file=sys.stderr)

    window_name = "MediaPipe Pose Landmarker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fps_counter = SlidingFps(window=30)
    smoother = LandmarkSmoother(alpha=args.smoothing)
    hand_smoothers: dict[str, LandmarkSmoother] = {}
    recorder = VideoRecorder(save_dir / "recordings")
    session_writer = SessionWriter(save_dir)
    kinematics_processor = KinematicsProcessor()
    mode = args.landmark_profile
    mirror = bool(args.mirror)
    metrics_overlay_enabled = bool(args.metrics_overlay)
    hand_overlay_enabled = bool(args.show_hands and hand_landmarker is not None)
    session_autostart_pending = bool(args.session_autostart)
    last_input_timestamp_ms = -1
    last_submitted_timestamp_ms = -1
    last_hand_submitted_timestamp_ms = -1
    last_draw_timestamp_ms = -1
    last_hand_draw_timestamp_ms = -1
    last_processed_result_timestamp_ms = -1
    processed_frame_index = 0
    draw_landmarks: list[DrawLandmark] | None = None
    draw_hand_landmarks: dict[str, list[DrawLandmark]] = {}
    latest_hand_detections: dict[str, HandDetection] = {}
    kinematic_frame: KinematicFrame | None = None
    flash_text = ""
    flash_until = 0.0
    flash_color = (245, 245, 245)
    detection_error_printed = False
    hand_detection_error_printed = False

    def flash(text: str, color: tuple[int, int, int] = (245, 245, 245), seconds: float = 2.0) -> None:
        nonlocal flash_text, flash_until, flash_color
        flash_text = text
        flash_color = color
        flash_until = time.perf_counter() + seconds

    def start_session(frame_shape: tuple[int, int, int]) -> None:
        height, width = frame_shape[:2]
        config = SessionConfig(
            camera_index=args.camera,
            width=width,
            height=height,
            mirror=mirror,
            smoothing=args.smoothing,
            model_name=model_path.name,
            plot_on_save=args.plot_on_save,
            landmark_profile=mode,
            hands_enabled=hand_overlay_enabled,
            hand_model_name=hand_model_path.name if hand_landmarker is not None else None,
        )
        session_id = session_writer.start(config)
        flash(f"Session recording: {session_id}", (80, 230, 120))
        print(f"Session started: {session_id}")

    def stop_session() -> None:
        path = session_writer.stop(final_mirror=mirror)
        if path is not None:
            flash(f"Session saved: {path}", (80, 230, 120), seconds=3.0)
            print(f"Session saved: {path}")

    reported_camera_fps = capture.get(cv2.CAP_PROP_FPS)
    camera_fps_text = f"{reported_camera_fps:.1f}" if reported_camera_fps > 0 else "unknown"
    requested_camera_fps_text = f"{args.camera_fps:g}" if args.camera_fps > 0 else "backend default"
    requested_fourcc_text = camera_fourcc if camera_fourcc is not None else "unchanged"
    print(
        f"Camera {args.camera} opened with backend: {backend} "
        f"(requested FPS: {requested_camera_fps_text}, reported FPS: {camera_fps_text}, FourCC: {requested_fourcc_text})"
    )
    print_controls()

    try:
        with ExitStack() as stack:
            stack.enter_context(landmarker)
            if hand_landmarker is not None:
                stack.enter_context(hand_landmarker)
            while True:
                ok, raw_frame = capture.read()
                if not ok or raw_frame is None:
                    print("ERROR: failed to read a frame from the camera.", file=sys.stderr)
                    break

                frame = cv2.flip(raw_frame, 1) if mirror else raw_frame.copy()
                frame = np.ascontiguousarray(frame)
                fps_value = fps_counter.tick()
                if session_autostart_pending and not session_writer.is_active:
                    try:
                        start_session(frame.shape)
                    except Exception as exc:
                        flash(f"Session start failed: {exc}", (60, 80, 255), seconds=3.0)
                        print(f"ERROR: could not start session: {exc}", file=sys.stderr)
                    session_autostart_pending = False

                timestamp_ms = next_timestamp_ms(last_input_timestamp_ms)
                last_input_timestamp_ms = timestamp_ms

                _, result_timestamp_before_submit = result_store.snapshot()
                pending_detection = last_submitted_timestamp_ms > result_timestamp_before_submit
                pending_timed_out = timestamp_ms - last_submitted_timestamp_ms >= args.max_pending_ms
                detect_interval_elapsed = timestamp_ms - last_submitted_timestamp_ms >= min_detect_interval_ms

                if detect_interval_elapsed and (not pending_detection or pending_timed_out):
                    try:
                        mp_image = frame_to_mp_image(frame, args.detect_width)
                        landmarker.detect_async(mp_image, timestamp_ms)
                        last_submitted_timestamp_ms = timestamp_ms
                    except Exception as exc:
                        flash(f"Detection failed: {exc}", (60, 80, 255), seconds=3.0)
                        if not detection_error_printed:
                            print(f"ERROR: detect_async failed: {exc}", file=sys.stderr)
                            detection_error_printed = True

                needs_hand_detection = hand_landmarker is not None and hand_overlay_enabled
                if hand_landmarker is not None and not needs_hand_detection and draw_hand_landmarks:
                    for side_smoother in hand_smoothers.values():
                        side_smoother.reset()
                    latest_hand_detections = {}
                    draw_hand_landmarks = {}
                    last_hand_draw_timestamp_ms = -1

                if needs_hand_detection:
                    _, hand_timestamp_before_submit = hand_result_store.snapshot()
                    pending_hand_detection = last_hand_submitted_timestamp_ms > hand_timestamp_before_submit
                    pending_hand_timed_out = timestamp_ms - last_hand_submitted_timestamp_ms >= args.max_pending_ms
                    hand_interval_elapsed = timestamp_ms - last_hand_submitted_timestamp_ms >= min_hand_detect_interval_ms
                    if hand_interval_elapsed and (not pending_hand_detection or pending_hand_timed_out):
                        try:
                            hand_mp_image = frame_to_mp_image(frame, args.hand_detect_width)
                            hand_landmarker.detect_async(hand_mp_image, timestamp_ms)
                            last_hand_submitted_timestamp_ms = timestamp_ms
                        except Exception as exc:
                            flash(f"Hand detection failed: {exc}", (60, 80, 255), seconds=3.0)
                            if not hand_detection_error_printed:
                                print(f"ERROR: hand detect_async failed: {exc}", file=sys.stderr)
                                hand_detection_error_printed = True

                result, result_timestamp_ms = result_store.snapshot()
                result_lag_ms = timestamp_ms - result_timestamp_ms if result_timestamp_ms >= 0 else None
                result_is_fresh = result_lag_ms is not None and result_lag_ms <= args.max_result_lag_ms
                hand_result, hand_result_timestamp_ms = hand_result_store.snapshot() if needs_hand_detection else (None, -1)
                hand_lag_ms = timestamp_ms - hand_result_timestamp_ms if hand_result_timestamp_ms >= 0 else None
                hand_result_is_fresh = hand_lag_ms is not None and hand_lag_ms <= args.max_result_lag_ms
                has_pose = bool(result_is_fresh and result is not None and result.pose_landmarks)
                new_result = result_timestamp_ms >= 0 and result_timestamp_ms != last_processed_result_timestamp_ms
                if has_pose and result is not None:
                    if result_timestamp_ms != last_draw_timestamp_ms:
                        hand_occlusion_points = [
                            point
                            for hand_landmarks in draw_hand_landmarks.values()
                            for point in hand_landmarks
                        ]
                        draw_landmarks = smoother.smooth(
                            result.pose_landmarks[0],
                            result_timestamp_ms,
                            occlusion_points=hand_occlusion_points,
                            occlusion_guard_indices=POSE_OCCLUSION_GUARD_INDICES,
                            self_occlusion_indices=POSE_HAND_OCCLUSION_INDICES,
                        )
                        last_draw_timestamp_ms = result_timestamp_ms
                    if draw_landmarks is not None:
                        draw_pose(frame, draw_landmarks, mode, pose_draw_indices)
                else:
                    if result_timestamp_ms != last_draw_timestamp_ms or (
                        result_lag_ms is not None and result_lag_ms > args.max_result_lag_ms
                    ):
                        smoother.reset()
                        draw_landmarks = None
                        last_draw_timestamp_ms = result_timestamp_ms

                if needs_hand_detection and hand_result_is_fresh:
                    if hand_result_timestamp_ms != last_hand_draw_timestamp_ms:
                        latest_hand_detections = extract_hand_detections(hand_result)
                        next_draw_hand_landmarks: dict[str, list[DrawLandmark]] = {}
                        for side, detection in latest_hand_detections.items():
                            side_smoother = hand_smoothers.setdefault(side, LandmarkSmoother(alpha=args.smoothing))
                            next_draw_hand_landmarks[side] = side_smoother.smooth(detection.landmarks, hand_result_timestamp_ms)
                        for side, side_smoother in list(hand_smoothers.items()):
                            if side not in latest_hand_detections:
                                side_smoother.reset()
                        draw_hand_landmarks = next_draw_hand_landmarks
                        last_hand_draw_timestamp_ms = hand_result_timestamp_ms
                    if hand_overlay_enabled and draw_hand_landmarks:
                        draw_hands(frame, draw_hand_landmarks)
                elif needs_hand_detection:
                    if hand_result_timestamp_ms != last_hand_draw_timestamp_ms or (
                        hand_lag_ms is not None and hand_lag_ms > args.max_result_lag_ms
                    ):
                        for side_smoother in hand_smoothers.values():
                            side_smoother.reset()
                        latest_hand_detections = {}
                        draw_hand_landmarks = {}
                        last_hand_draw_timestamp_ms = hand_result_timestamp_ms

                if new_result and result_is_fresh:
                    processed_frame_index += 1
                    pose_frame = build_pose_frame(
                        frame_index=processed_frame_index,
                        timestamp_ms=result_timestamp_ms,
                        result=result,
                        smoothed_landmarks=draw_landmarks,
                        hand_detections=latest_hand_detections if hand_result_is_fresh else {},
                        smoothed_hand_landmarks=draw_hand_landmarks if hand_result_is_fresh else {},
                        hands_detected=bool(latest_hand_detections) if hand_result_is_fresh else False,
                        pose_detected=has_pose,
                        mirror=mirror,
                        frame_shape=frame.shape,
                        fps=fps_value,
                    )
                    kinematic_frame = kinematics_processor.process(pose_frame)
                    if session_writer.is_active:
                        session_writer.add_frame(pose_frame, kinematic_frame)
                    last_processed_result_timestamp_ms = result_timestamp_ms
                elif new_result:
                    last_processed_result_timestamp_ms = result_timestamp_ms

                draw_hud(
                    frame=frame,
                    fps=fps_value,
                    camera_index=args.camera,
                    has_pose=has_pose,
                    mode=mode,
                    mirror=mirror,
                    recording=recorder.is_recording,
                    session_active=session_writer.is_active,
                    session_id=session_writer.session_id,
                    result_lag_ms=result_lag_ms,
                    backend=backend,
                    hands_enabled=hand_overlay_enabled,
                    hands_detected=bool(draw_hand_landmarks) if hand_overlay_enabled else False,
                    hand_lag_ms=hand_lag_ms if hand_overlay_enabled else None,
                )

                if metrics_overlay_enabled:
                    draw_metrics_overlay(
                        frame,
                        {
                            "pose_detected": has_pose,
                            "fps": fps_value,
                            "session_state": "RECORDING" if session_writer.is_active else "IDLE",
                            "mirror": mirror,
                            "right_elbow_angle": getattr(kinematic_frame, "right_elbow_angle", None),
                            "right_knee_angle": getattr(kinematic_frame, "right_knee_angle", None),
                            "right_wrist_speed": getattr(kinematic_frame, "right_wrist_speed", None),
                            "pelvis_speed": getattr(kinematic_frame, "pelvis_speed", None),
                            "motion_energy_proxy": getattr(kinematic_frame, "motion_energy_proxy", None),
                        },
                        origin=(14, 260),
                    )

                if time.perf_counter() < flash_until and flash_text:
                    put_text(frame, flash_text, (14, frame.shape[0] - 20), flash_color)

                if args.record and not recorder.is_recording:
                    try:
                        path = recorder.start(frame.shape, fps_value)
                        flash(f"Recording: {path}", (80, 230, 120))
                    except Exception as exc:
                        flash(f"Record failed: {exc}", (60, 80, 255), seconds=3.0)
                        print(f"ERROR: could not start recording: {exc}", file=sys.stderr)
                    args.record = False

                if recorder.is_recording:
                    recorder.write(frame)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    try:
                        path = save_screenshot(frame, save_dir / "screenshots")
                        flash(f"Screenshot: {path}", (80, 230, 120))
                    except Exception as exc:
                        flash(f"Screenshot failed: {exc}", (60, 80, 255), seconds=3.0)
                        print(f"ERROR: could not save screenshot: {exc}", file=sys.stderr)
                elif key in (ord("r"), ord("R")):
                    if recorder.is_recording:
                        path = recorder.stop()
                        flash(f"Recording saved: {path}", (80, 230, 120))
                    else:
                        try:
                            path = recorder.start(frame.shape, fps_value)
                            flash(f"Recording: {path}", (80, 230, 120))
                        except Exception as exc:
                            flash(f"Record failed: {exc}", (60, 80, 255), seconds=3.0)
                            print(f"ERROR: could not start recording: {exc}", file=sys.stderr)
                elif key in (ord("m"), ord("M")):
                    mirror = not mirror
                    smoother.reset()
                    for side_smoother in hand_smoothers.values():
                        side_smoother.reset()
                    draw_landmarks = None
                    draw_hand_landmarks = {}
                    flash(f"Mirror: {'ON' if mirror else 'OFF'}")
                elif key == ord("1"):
                    mode = "full"
                    pose_draw_indices = resolve_pose_draw_indices(mode, args.include_landmarks, args.exclude_landmarks)
                    flash("Mode: full skeleton")
                elif key == ord("3"):
                    metrics_overlay_enabled = not metrics_overlay_enabled
                    flash(f"Metrics panel: {'ON' if metrics_overlay_enabled else 'OFF'}")
                elif key in (ord("f"), ord("F")):
                    mode = "full" if mode == "no-face" else "no-face"
                    pose_draw_indices = resolve_pose_draw_indices(mode, args.include_landmarks, args.exclude_landmarks)
                    flash("Face landmarks: ON" if mode == "full" else "Face landmarks: OFF")
                elif key == ord("6"):
                    mode = "no-face"
                    pose_draw_indices = resolve_pose_draw_indices(mode, args.include_landmarks, args.exclude_landmarks)
                    flash("Mode: no face")
                elif key == ord("7"):
                    mode = "upper-body"
                    pose_draw_indices = resolve_pose_draw_indices(mode, args.include_landmarks, args.exclude_landmarks)
                    flash("Mode: upper body")
                elif key == ord("8"):
                    mode = "lower-body"
                    pose_draw_indices = resolve_pose_draw_indices(mode, args.include_landmarks, args.exclude_landmarks)
                    flash("Mode: lower body")
                elif key in (ord("h"), ord("H")):
                    if hand_landmarker is not None:
                        hand_overlay_enabled = not hand_overlay_enabled
                        if not hand_overlay_enabled:
                            for side_smoother in hand_smoothers.values():
                                side_smoother.reset()
                            latest_hand_detections = {}
                            draw_hand_landmarks = {}
                            last_hand_draw_timestamp_ms = -1
                        flash(f"Hands overlay: {'ON' if hand_overlay_enabled else 'OFF'}")
                    else:
                        flash(f"Hand model not found: {hand_model_path}", (0, 190, 255), seconds=3.0)
                elif key in (ord("c"), ord("C")):
                    try:
                        if session_writer.is_active:
                            stop_session()
                        else:
                            start_session(frame.shape)
                    except Exception as exc:
                        flash(f"Session failed: {exc}", (60, 80, 255), seconds=3.0)
                        print(f"ERROR: session operation failed: {exc}", file=sys.stderr)
    finally:
        if session_writer.is_active:
            path = session_writer.stop(final_mirror=mirror)
            if path is not None:
                print(f"Session saved: {path}")
        saved_path = recorder.stop()
        if saved_path is not None:
            print(f"Recording saved: {saved_path}")
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
