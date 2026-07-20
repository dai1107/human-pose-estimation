from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from math import nan
from pathlib import Path
from typing import Sequence

import cv2

from hyrox.action_names import HYROX_ACTION_NAMES, action_from_menu_key, next_hyrox_action
from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from hyrox.view_policy import CAMERA_VIEWS, next_camera_view
from src.backends.base import PoseBackend, PoseResult
from src.backends.factory import create_backend
from src.biomechanics.landmarks import LANDMARK_NAMES
from src.biomechanics.normalization import normalize_landmarks
from src.biomechanics.session_writer import SessionConfig, SessionWriteError, SessionWriter
from src.biomechanics.types import KinematicFrame, LandmarkPoint, PoseFrame
from src.biomechanics.velocity import KinematicsProcessor
from src.detectors.yolo_person_detector import YoloPersonDetector
from src.fusion.yolo_roi_mediapipe import FusionFrameStats, YoloRoiMediaPipeFusion
from src.pose.adapters import format_normalized_pose_debug, normalize_backend_pose_result
from src.paths import resolve_asset
from src.realtime.feedback_engine import FeedbackEngine
from src.runtime_hand import DEFAULT_HAND_DETECT_WIDTH, DEFAULT_MAX_HAND_DETECT_FPS, HandDetection, MediaPipeHandTracker
from src.utils.angle_utils import body_angles
from src.utils.backend_policy import resolve_backend_choice
from src.utils.draw_utils import draw_bbox, draw_hand_landmarks, draw_hyrox_action_overlay, draw_hyrox_action_selector, draw_hyrox_debug_overlay, draw_pose_result_filtered, draw_realtime_overlay
from src.utils.device import resolve_torch_device
from src.utils.metrics import RealtimeMetrics
from src.utils.smoothing import KeypointSmoother
from src.ui.metrics_overlay import draw_metrics_overlay
from src.configuration import ConfigValidationError
from src.runtime_logging import (
    AppError,
    BackendInitializationError,
    ExitCode,
    InputSourceError,
    OutputWriteError,
    configure_logging,
    report_error,
    safe_cleanup,
)
from src.version import __version__


RUNTIME_BACKENDS = ("mediapipe", "yolo-pose")
LOGGER = logging.getLogger("pose.desktop")
POSE_NAME_TO_INDEX = {name: index for index, name in enumerate(LANDMARK_NAMES)}
FACE_KEYPOINT_NAMES = frozenset(LANDMARK_NAMES[:11])
NO_FACE_KEYPOINT_NAMES = frozenset(LANDMARK_NAMES[11:])
UPPER_BODY_KEYPOINT_NAMES = frozenset(
    {
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
)
LOWER_BODY_KEYPOINT_NAMES = frozenset(
    {
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
)
DRAW_MODE_KEYPOINT_NAMES: dict[str, frozenset[str] | None] = {
    "full": None,
    "no-face": NO_FACE_KEYPOINT_NAMES,
    "upper-body": UPPER_BODY_KEYPOINT_NAMES,
    "lower-body": LOWER_BODY_KEYPOINT_NAMES,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime-first human keypoint detection baseline.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--log-dir", default="outputs/logs", help="Directory for rolling application logs. Default: outputs/logs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and tracebacks.")
    parser.add_argument("--backend", default="auto", choices=("auto", "mediapipe", "yolo-pose"), help="Pose backend. Default: auto.")
    parser.add_argument("--action-type", default="auto", help="Action type for --backend auto, e.g. rowing, ski_erg, or HYROX video stem.")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="MediaPipe pose model path.")
    parser.add_argument("--yolo-pose-model", default="yolo11n-pose.pt", help="YOLO Pose model path. Default: yolo11n-pose.pt.")
    parser.add_argument("--yolo-device", default="auto", help="YOLO Pose device, e.g. auto, 0, cuda:0, cpu. Default: auto.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--input-video", default="", help="Read frames from a video instead of opening a camera.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width. Default: 640.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height. Default: 480.")
    parser.add_argument("--camera-fps", type=float, default=60.0, help="Requested camera FPS. Default: 60.")
    parser.add_argument("--camera-fourcc", default="MJPG", help="Requested camera FourCC. Empty string leaves it unchanged.")
    parser.add_argument("--landmark-profile", default="full", choices=("full", "no-face", "upper-body", "lower-body"), help="Landmark display profile. Default: full.")
    parser.add_argument("--show-hands", action="store_true", help="Show supplemental five-finger hand landmarks at startup.")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="MediaPipe hand model path.")
    parser.add_argument("--hand-detect-width", type=int, default=DEFAULT_HAND_DETECT_WIDTH, help=f"Resize hand detector input to this width. Default: {DEFAULT_HAND_DETECT_WIDTH}.")
    parser.add_argument("--max-hand-detect-fps", type=float, default=DEFAULT_MAX_HAND_DETECT_FPS, help=f"Maximum hand detector submissions per second. Default: {DEFAULT_MAX_HAND_DETECT_FPS:g}.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect. Default: 2.")
    parser.add_argument("--record", default="", help="Save annotated video to this path.")
    parser.add_argument("--record-raw", default="", help="Save raw input frames to this path.")
    parser.add_argument("--save-metrics", default="", help="Append final metrics to this CSV path.")
    parser.add_argument("--save-dir", default="outputs", help="Directory for sessions, screenshots, and recordings. Default: outputs.")
    parser.add_argument("--headless", action="store_true", help="Do not open an OpenCV window; useful for input-video batch evaluation.")
    parser.add_argument("--normalized-pose-debug", action="store_true", help="Print the unified NormalizedPose summary every 30 frames. Default: off.")
    parser.add_argument("--hyrox-debug", action="store_true", help="Show HYROX debug pose features overlay. Default: off.")
    parser.add_argument("--hyrox-action", default="none", choices=("none", *HYROX_ACTION_NAMES), help="Enable HYROX action analysis. Default: none.")
    parser.add_argument("--hyrox-sensitivity", default="medium", choices=("low", "medium", "high"), help="HYROX action sensitivity. Default: medium.")
    parser.add_argument("--hyrox-config", default="", help="HYROX analyzer config path. Empty selects the action-specific default.")
    parser.add_argument("--metrics-overlay", action="store_true", help="Show kinematic metrics panel at startup.")
    parser.add_argument("--session-autostart", action="store_true", help="Start a kinematic data session automatically.")
    parser.add_argument("--camera-view", default="unknown", choices=CAMERA_VIEWS, help="Camera view used by view-sensitive evaluation: front, side, front_left, front_right, or unknown. Default: unknown.")
    parser.add_argument("--person-detector", default="none", choices=("none", "yolo"), help="Optional person detector. Default: none.")
    parser.add_argument("--detector-model", default="yolo11n.pt", help="YOLO person detector model. Default: yolo11n.pt.")
    parser.add_argument("--detector-device", default="auto", help="YOLO person detector device, e.g. auto, 0, cuda:0, cpu. Default: auto.")
    parser.add_argument("--detector-every-n", type=int, default=5, help="Run YOLO every N frames. Default: 5.")
    parser.add_argument("--bbox-expand", type=float, default=1.25, help="Expand detector bbox by this scale. Default: 1.25.")
    parser.add_argument("--bbox-smoothing", type=float, default=0.6, help="BBox smoothing alpha. Default: 0.6.")
    parser.add_argument(
        "--target-select",
        default="tracking",
        choices=("tracking", "confidence", "area"),
        help="Select and track the athlete, or select each frame by confidence/area.",
    )
    parser.add_argument("--fusion", default="none", choices=("none", "yolo-roi-mediapipe"), help="Fusion strategy. Default: none.")
    parser.add_argument("--smoothing", default="one-euro", choices=("none", "ema", "one-euro"), help="Keypoint smoothing mode.")
    parser.add_argument("--ema-alpha", type=float, default=0.6, help="EMA alpha. Default: 0.6.")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.0, help="One Euro min cutoff. Default: 1.0.")
    parser.add_argument("--one-euro-beta", type=float, default=0.01, help="One Euro beta. Default: 0.01.")
    parser.add_argument("--one-euro-d-cutoff", type=float, default=1.0, help="One Euro derivative cutoff. Default: 1.0.")
    parser.add_argument("--pose-hold-frames", type=int, default=5, help="Hold the last valid pose for short tracking drops. Use 0 to disable. Default: 5.")
    occlusion_group = parser.add_mutually_exclusive_group()
    occlusion_group.add_argument("--occlusion-guard", dest="occlusion_guard", action="store_true", help="Suppress body-joint jumps near hands. Default.")
    occlusion_group.add_argument("--no-occlusion-guard", dest="occlusion_guard", action="store_false", help="Disable hand/body occlusion guard.")
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument("--mirror", dest="mirror", action="store_true", help="Mirror display for camera input. Default.")
    mirror_group.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable mirror display.")
    parser.set_defaults(mirror=True)
    parser.set_defaults(occlusion_guard=True)
    return parser.parse_args(argv)


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
        f"Camera {args.camera} opened "
        f"(requested FPS: {args.camera_fps:g}, reported FPS: {fps if fps > 0 else 0:.1f}, FourCC: {args.camera_fourcc or 'unchanged'})"
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
    suffix = output.suffix.lower()
    fourcc_name = "mp4v" if suffix == ".mp4" else "XVID"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*fourcc_name), max(1.0, min(60.0, fps)), (width, height))
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


def save_screenshot(frame, root: str | Path = "outputs") -> Path:
    path = make_output_path("screenshots", ".png", root=root)
    if not cv2.imwrite(str(path), frame):
        raise OutputWriteError(
            f"无法保存截图：{path}",
            hint="检查磁盘空间和目录权限",
        )
    return path


def visible_keypoint_names_for_mode(mode: str) -> set[str] | None:
    allowed = DRAW_MODE_KEYPOINT_NAMES.get(mode)
    if allowed is None:
        return None
    return set(allowed)


def highlight_keypoint_names_for_mode(mode: str) -> set[str]:
    return set()


def to_landmark_points(result: PoseResult) -> list[LandmarkPoint]:
    points = [LandmarkPoint(nan, nan, nan, 0.0, 0.0) for _ in range(len(LANDMARK_NAMES))]
    for point in result.keypoints:
        index = POSE_NAME_TO_INDEX.get(point.name)
        if index is None:
            continue
        confidence = max(0.0, min(1.0, float(point.confidence)))
        points[index] = LandmarkPoint(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
            visibility=confidence,
            presence=confidence,
        )
    return points


def build_pose_frame_from_result(
    result: PoseResult,
    *,
    frame_index: int,
    mirror: bool,
    frame_shape: tuple[int, int, int],
    fps: float,
    hand_detections: dict[str, HandDetection] | None = None,
) -> PoseFrame:
    image_landmarks = to_landmark_points(result)
    normalization = normalize_landmarks(image_landmarks if result.success else None)
    hand_detections = hand_detections or {}
    image_hand_landmarks = {
        side: list(detection.landmarks)
        for side, detection in hand_detections.items()
    }
    world_hand_landmarks = {
        side: list(detection.world_landmarks)
        for side, detection in hand_detections.items()
    }
    height, width = frame_shape[:2]
    return PoseFrame(
        frame_index=frame_index,
        timestamp_ms=int(result.timestamp_ms or 0),
        pose_detected=bool(result.success),
        image_landmarks=image_landmarks,
        world_landmarks=[],
        smoothed_landmarks=image_landmarks,
        normalized_landmarks=normalization.landmarks,
        hands_detected=bool(image_hand_landmarks),
        hand_landmarks=image_hand_landmarks,
        hand_world_landmarks=world_hand_landmarks,
        smoothed_hand_landmarks={side: list(points) for side, points in image_hand_landmarks.items()},
        normalization_success=normalization.success,
        normalization_message=normalization.message,
        mirror=mirror,
        camera_width=width,
        camera_height=height,
        fps=float(fps),
    )


def current_model_label(args: argparse.Namespace, backend_name: str) -> str:
    if backend_name == "yolo-pose":
        return Path(args.yolo_pose_model).name
    return Path(args.model).name


def timestamp_for_frame(input_mode: str, started_ns: int, frame_index: int, fps: float) -> int:
    if input_mode == "video":
        return int(round(frame_index * 1000.0 / max(fps, 1.0)))
    return int((time.monotonic_ns() - started_ns) / 1_000_000)


def validate_runtime_args(args: argparse.Namespace, resolved_backend: str) -> None:
    if args.fusion == "yolo-roi-mediapipe" and resolved_backend != "mediapipe":
        raise AppError(
            "CFG003",
            "--fusion yolo-roi-mediapipe 只支持 --backend mediapipe",
            exit_code=ExitCode.CONFIG_ERROR,
        )
    if args.fusion == "yolo-roi-mediapipe" and args.person_detector != "yolo":
        raise AppError(
            "CFG003",
            "--fusion yolo-roi-mediapipe 需要 --person-detector yolo",
            exit_code=ExitCode.CONFIG_ERROR,
        )
    if resolved_backend != "mediapipe" and args.person_detector != "none":
        raise AppError(
            "CFG003",
            "--person-detector yolo 仅用于 MediaPipe ROI；当前后端应使用 none",
            exit_code=ExitCode.CONFIG_ERROR,
        )


def backend_device_for(args: argparse.Namespace, backend_name: str) -> str:
    return resolve_torch_device(args.yolo_device) if backend_name == "yolo-pose" else "cpu"


def create_runtime_backend(args: argparse.Namespace, backend_name: str) -> tuple[PoseBackend, str]:
    return create_backend(args, backend_name=backend_name), backend_device_for(args, backend_name)


def create_runtime_smoother(args: argparse.Namespace) -> KeypointSmoother:
    return KeypointSmoother(
        mode=args.smoothing,
        ema_alpha=args.ema_alpha,
        one_euro_min_cutoff=args.one_euro_min_cutoff,
        one_euro_beta=args.one_euro_beta,
        one_euro_d_cutoff=args.one_euro_d_cutoff,
        max_missing_frames=args.pose_hold_frames,
        occlusion_guard=args.occlusion_guard,
    )


def next_runtime_backend(current_backend: str) -> str:
    if current_backend not in RUNTIME_BACKENDS:
        raise ValueError(f"backend switching only supports: {', '.join(RUNTIME_BACKENDS)}")
    return "yolo-pose" if current_backend == "mediapipe" else "mediapipe"


def runtime_backend_switch_allowed(args: argparse.Namespace) -> tuple[bool, str]:
    if args.fusion != "none":
        return False, "fusion must be none"
    if args.person_detector != "none":
        return False, "person_detector must be none"
    return True, ""


def runtime_hyrox_config_path(
    action_name: str,
    *,
    startup_action: str,
    startup_config: str | None,
) -> str | None:
    """Only reuse an explicit config for the action it was supplied for."""
    return startup_config if startup_config and action_name == startup_action else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        configure_logging(
            app_name="desktop",
            log_dir=args.log_dir,
            debug=bool(args.debug),
        )
    except OSError as exc:
        print(
            f"ERROR: [OUT003] 无法初始化日志目录 {args.log_dir}: {exc}",
            file=sys.stderr,
        )
        return int(ExitCode.OUTPUT_ERROR)
    LOGGER.info(
        "Pose Estimation %s starting backend=%s input=%s",
        __version__,
        args.backend,
        args.input_video or f"camera:{args.camera}",
    )
    for attribute in (
        "model",
        "hand_model",
        "yolo_pose_model",
        "detector_model",
    ):
        setattr(args, attribute, str(resolve_asset(getattr(args, attribute))))
    if args.hyrox_config:
        args.hyrox_config = str(resolve_asset(args.hyrox_config))
    backend = None
    fusion_runner = None
    capture = None
    record_writer = None
    raw_writer = None
    save_dir = Path(args.save_dir)
    record_output_path = args.record
    raw_record_output_path = args.record_raw
    recording_enabled = bool(record_output_path)
    raw_recording_enabled = bool(raw_record_output_path)
    mirror_enabled = bool(args.mirror)
    display_mode = args.landmark_profile
    metrics_overlay_enabled = bool(args.metrics_overlay)
    session_autostart_pending = bool(args.session_autostart)
    hand_model_path = Path(args.hand_model)
    hand_detect_interval_ms = int(round(1000.0 / float(args.max_hand_detect_fps))) if float(args.max_hand_detect_fps) > 0 else 0
    resolved_backend = resolve_backend_choice(args.backend, action_type=args.action_type, input_video=args.input_video)
    runtime_backend_device = backend_device_for(args, resolved_backend)
    runtime_detector_device = resolve_torch_device(args.detector_device) if args.person_detector == "yolo" else "none"
    session_writer = SessionWriter(save_dir)
    kinematics_processor = KinematicsProcessor()
    kinematic_frame: KinematicFrame | None = None
    hand_tracker: MediaPipeHandTracker | None = None
    hand_tracker_error: str | None = None
    hand_overlay_enabled = False
    hand_detections: dict[str, HandDetection] = {}
    hand_detection_error_printed = False
    last_hand_detection_timestamp_ms = -1
    startup_hyrox_action = args.hyrox_action
    startup_hyrox_config = args.hyrox_config or None
    hyrox_action_enabled = args.hyrox_action != "none"
    try:
        hyrox_analyzer = (
            create_action_analyzer(
                args.hyrox_action,
                runtime_hyrox_config_path(
                    args.hyrox_action,
                    startup_action=startup_hyrox_action,
                    startup_config=startup_hyrox_config,
                ),
                sensitivity=args.hyrox_sensitivity,
                camera_view=args.camera_view,
                live_mode=not bool(args.input_video),
            )
            if hyrox_action_enabled
            else None
        )
    except (FileNotFoundError, ConfigValidationError, ValueError) as exc:
        error = AppError(
            getattr(exc, "error_code", "CFG001"),
            str(exc),
            exit_code=ExitCode.CONFIG_ERROR,
            hint="修正配置后运行 python -m src.doctor 复查",
        )
        report_error(LOGGER, error, debug=bool(args.debug))
        return int(error.exit_code)
    metrics = RealtimeMetrics(
        backend=resolved_backend,
        smoothing=args.smoothing,
        input_name=args.input_video or "camera",
        person_detector=args.person_detector,
        fusion=args.fusion,
        detector_every_n=args.detector_every_n,
        backend_device=runtime_backend_device,
        detector_device=runtime_detector_device,
    )
    exit_code = ExitCode.SUCCESS
    try:
        validate_runtime_args(args, resolved_backend)
        LOGGER.info(
            "Resolved backend: %s (requested: %s, action_type: %s)",
            resolved_backend,
            args.backend,
            args.action_type,
        )
        try:
            backend, runtime_backend_device = create_runtime_backend(
                args,
                resolved_backend,
            )
        except Exception as exc:
            raise BackendInitializationError(
                f"姿态后端 {resolved_backend} 初始化失败：{exc}",
                hint="检查模型文件和后端依赖，可先运行 python -m src.doctor",
            ) from exc
        yolo_detector = None
        if args.fusion == "yolo-roi-mediapipe":
            yolo_detector = YoloPersonDetector(
                model_path=args.detector_model,
                every_n=args.detector_every_n,
                bbox_expand=args.bbox_expand,
                bbox_smoothing=args.bbox_smoothing,
                target_select=args.target_select,
                device=runtime_detector_device,
            )
            fusion_runner = YoloRoiMediaPipeFusion(backend, yolo_detector)
        capture, input_mode, source_fps = open_capture(args)
        smoother = create_runtime_smoother(args)
        feedback_engine = FeedbackEngine()
        started_ns = time.monotonic_ns()
        frame_index = 0
        hyrox_debug_error_printed = False
        hyrox_action_error_printed = False
        normalized_pose_error_printed = False
        window_name = "Realtime Keypoint Baseline"
        status_message = ""
        status_until = 0.0
        hyrox_action_selector_open = False

        def set_status(message: str, seconds: float = 2.5) -> None:
            nonlocal status_message, status_until
            status_message = message
            status_until = time.perf_counter() + seconds

        def switch_hyrox_action(action_name: str) -> bool:
            nonlocal hyrox_action_enabled, hyrox_analyzer, hyrox_action_error_printed
            if action_name == args.hyrox_action:
                set_status(f"action remains {action_name}")
                return True
            try:
                new_analyzer = (
                    create_action_analyzer(
                        action_name,
                        runtime_hyrox_config_path(
                            action_name,
                            startup_action=startup_hyrox_action,
                            startup_config=startup_hyrox_config,
                        ),
                        sensitivity=args.hyrox_sensitivity,
                        camera_view=args.camera_view,
                        live_mode=not bool(args.input_video),
                    )
                    if action_name != "none"
                    else None
                )
            except (FileNotFoundError, ValueError) as exc:
                set_status(f"action switch failed: {action_name}", seconds=3.0)
                LOGGER.error("[CFG001] Action switch failed: %s", exc)
                return False
            hyrox_analyzer = new_analyzer
            args.hyrox_action = action_name
            hyrox_action_enabled = action_name != "none"
            hyrox_action_error_printed = False
            set_status(f"action {action_name}", seconds=3.0)
            LOGGER.info("HYROX action: %s", action_name)
            return True

        def ensure_hand_tracker(*, required: bool) -> MediaPipeHandTracker | None:
            nonlocal hand_tracker, hand_tracker_error
            if hand_tracker is not None:
                return hand_tracker
            if hand_tracker_error is not None:
                if required:
                    raise RuntimeError(hand_tracker_error)
                return None
            if not hand_model_path.exists():
                hand_tracker_error = f"hand model not found: {hand_model_path}"
                if required:
                    raise FileNotFoundError(hand_tracker_error)
                return None
            try:
                hand_tracker = MediaPipeHandTracker(
                    hand_model_path,
                    detect_width=args.hand_detect_width,
                    max_hands=args.max_hands,
                )
            except Exception as exc:
                hand_tracker_error = f"hand tracker init failed: {exc}"
                if required:
                    raise RuntimeError(hand_tracker_error) from exc
                return None
            return hand_tracker

        def start_session(frame_shape: tuple[int, int, int]) -> None:
            height, width = frame_shape[:2]
            config = SessionConfig(
                camera_index=args.camera,
                width=width,
                height=height,
                mirror=mirror_enabled,
                smoothing=args.smoothing,
                model_name=current_model_label(args, resolved_backend),
                landmark_profile=display_mode,
                hands_enabled=hand_overlay_enabled,
                hand_model_name=hand_model_path.name if hand_tracker is not None else None,
                camera_view=args.camera_view,
            )
            session_id = session_writer.start(config)
            set_status(f"session {session_id}", seconds=3.0)
            LOGGER.info("Session started: %s", session_id)

        def stop_session() -> None:
            path = session_writer.stop(final_mirror=mirror_enabled)
            if path is not None:
                set_status("session saved", seconds=3.0)
                LOGGER.info("Session saved: %s", path)

        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        LOGGER.info("Controls: Q/ESC quit, A action menu, N next action, B backend, V camera view, S screenshot, R record, T raw record, M mirror, 1 full, 3 metrics, F face, 6 no-face, 7 upper, 8 lower, H hands, C session")
        if args.show_hands:
            ensure_hand_tracker(required=True)
            hand_overlay_enabled = True

        while True:
            ok, raw_frame = read_capture_frame(
                capture,
                input_mode=input_mode,
                processed_frames=frame_index,
            )
            if not ok or raw_frame is None:
                break
            frame_started = time.perf_counter()
            frame_index += 1
            display_frame = cv2.flip(raw_frame, 1) if input_mode == "camera" and mirror_enabled else raw_frame.copy()
            if session_autostart_pending and not session_writer.is_active:
                try:
                    start_session(display_frame.shape)
                except Exception as exc:
                    set_status("session start failed", seconds=3.0)
                    LOGGER.error("[OUT002] Session start failed: %s", exc)
                session_autostart_pending = False

            if raw_recording_enabled:
                if raw_writer is None:
                    if not raw_record_output_path:
                        raw_record_output_path = str(make_output_path("recordings", "_raw.mp4", root=save_dir))
                    raw_writer = create_writer(raw_record_output_path, source_fps, raw_frame.shape)
                    LOGGER.info("Raw recording started: %s", raw_record_output_path)
                raw_writer.write(raw_frame)

            timestamp_ms = timestamp_for_frame(input_mode, started_ns, frame_index, source_fps)
            if fusion_runner is not None:
                result, fusion_stats = fusion_runner.detect(display_frame, timestamp_ms=timestamp_ms)
            else:
                result = backend.detect(display_frame, timestamp_ms=timestamp_ms)
                fusion_stats = FusionFrameStats()
            normalized_pose = None
            try:
                frame_height, frame_width = display_frame.shape[:2]
                normalized_pose = normalize_backend_pose_result(
                    result,
                    image_width=frame_width,
                    image_height=frame_height,
                    timestamp_ms=result.timestamp_ms if result.timestamp_ms is not None else timestamp_ms,
                    frame_id=frame_index,
                    latency_ms=result.inference_time_ms,
                )
            except Exception as exc:
                if not normalized_pose_error_printed:
                    LOGGER.warning("normalized pose conversion failed: %s", exc)
                    normalized_pose_error_printed = True
            if args.normalized_pose_debug and (frame_index == 1 or frame_index % 30 == 0):
                LOGGER.debug("%s", format_normalized_pose_debug(normalized_pose))
            result = smoother.smooth_result(result)
            if hand_overlay_enabled:
                tracker = ensure_hand_tracker(required=False)
                if tracker is None:
                    hand_overlay_enabled = False
                    hand_detections = {}
                    last_hand_detection_timestamp_ms = -1
                    set_status(hand_tracker_error or "hand tracker unavailable", seconds=3.0)
                elif hand_detect_interval_ms <= 0 or timestamp_ms - last_hand_detection_timestamp_ms >= hand_detect_interval_ms:
                    try:
                        hand_detections = tracker.detect(display_frame, timestamp_ms=timestamp_ms)
                        last_hand_detection_timestamp_ms = timestamp_ms
                    except Exception as exc:
                        hand_detections = {}
                        hand_overlay_enabled = False
                        last_hand_detection_timestamp_ms = -1
                        set_status("hand detection failed", seconds=3.0)
                        if not hand_detection_error_printed:
                            LOGGER.warning("hand detection failed: %s", exc)
                            hand_detection_error_printed = True
            angles = body_angles(result)
            feedback = feedback_engine.update(result, angles)
            frame_finished = time.perf_counter()
            snapshot = metrics.update(
                result,
                angles,
                frame_started=frame_started,
                frame_finished=frame_finished,
                roi_enabled=fusion_stats.roi_enabled,
                roi_success=fusion_stats.roi_success,
                yolo_detection_time_ms=fusion_stats.yolo_detection_time_ms,
                bbox_reused=fusion_stats.bbox_reused,
                bbox_lost=fusion_stats.bbox_lost,
                fallback_to_full_frame=fusion_stats.fallback_to_full_frame,
                source_model_distribution=getattr(fusion_stats, "source_model_distribution", None),
            )
            pose_frame = build_pose_frame_from_result(
                result,
                frame_index=frame_index,
                mirror=mirror_enabled,
                frame_shape=display_frame.shape,
                fps=snapshot.realtime_fps if snapshot.realtime_fps > 0 else source_fps,
                hand_detections=hand_detections if hand_overlay_enabled else {},
            )
            kinematic_frame = kinematics_processor.process(pose_frame)
            if session_writer.is_active:
                session_writer.add_frame(pose_frame, kinematic_frame)
            hyrox_features = None
            hyrox_action_state = None
            has_pose = bool(result.success and result.keypoints)
            if (args.hyrox_debug or hyrox_action_enabled) and has_pose:
                try:
                    height, width = display_frame.shape[:2]
                    hyrox_features = extract_basic_pose_features(
                        result.keypoints,
                        image_width=width,
                        image_height=height,
                        segmentation_mask=result.extra.get("segmentation_mask"),
                    )
                except Exception as exc:
                    hyrox_features = None
                    if not hyrox_debug_error_printed:
                        LOGGER.warning("HYROX debug extraction failed: %s", exc)
                        hyrox_debug_error_printed = True
            if hyrox_analyzer is not None:
                try:
                    hyrox_action_state = hyrox_analyzer.attach_view_context(
                        hyrox_analyzer.update(hyrox_features if has_pose else None, timestamp_ms=timestamp_ms)
                    )
                except Exception as exc:
                    hyrox_action_state = None
                    if not hyrox_action_error_printed:
                        LOGGER.warning("HYROX action analysis failed: %s", exc)
                        hyrox_action_error_printed = True

            annotated = display_frame.copy()
            draw_pose_result_filtered(
                annotated,
                result,
                visible_names=visible_keypoint_names_for_mode(display_mode),
                highlight_names=highlight_keypoint_names_for_mode(display_mode),
            )
            if hand_overlay_enabled and hand_detections:
                draw_hand_landmarks(
                    annotated,
                    {side: detection.landmarks for side, detection in hand_detections.items()},
                )
            draw_bbox(annotated, result.bbox)
            overlay_status = status_message if time.perf_counter() < status_until else ""
            draw_realtime_overlay(
                annotated,
                backend=resolved_backend,
                fusion=args.fusion,
                person_detector=args.person_detector,
                detector_every_n=args.detector_every_n,
                smoothing=args.smoothing,
                input_mode=input_mode,
                result=result,
                metrics=snapshot,
                feedback=feedback,
                recording=recording_enabled,
                raw_recording=raw_recording_enabled,
                angles=angles,
                status_message=overlay_status,
            )
            right_panel_x = max(14, annotated.shape[1] - 344)
            detail_panel_x = max(14, annotated.shape[1] - 404)
            if metrics_overlay_enabled:
                draw_metrics_overlay(
                    annotated,
                    {
                        "pose_detected": has_pose,
                        "fps": snapshot.realtime_fps,
                        "session_state": "RECORDING" if session_writer.is_active else "IDLE",
                        "mirror": mirror_enabled,
                        "right_elbow_angle": getattr(kinematic_frame, "right_elbow_angle", None),
                        "right_knee_angle": getattr(kinematic_frame, "right_knee_angle", None),
                        "right_wrist_speed": getattr(kinematic_frame, "right_wrist_speed", None),
                        "pelvis_speed": getattr(kinematic_frame, "pelvis_speed", None),
                        "motion_energy_proxy": getattr(kinematic_frame, "motion_energy_proxy", None),
                    },
                    origin=(right_panel_x, 28),
                )
            if args.hyrox_debug:
                try:
                    draw_hyrox_debug_overlay(
                        annotated,
                        hyrox_features,
                        has_pose=has_pose,
                        action_state=hyrox_action_state,
                    )
                except Exception as exc:
                    if not hyrox_debug_error_printed:
                        LOGGER.warning("HYROX debug overlay failed: %s", exc)
                        hyrox_debug_error_printed = True
            if hyrox_action_enabled and not args.hyrox_debug:
                try:
                    draw_hyrox_action_overlay(
                        annotated,
                        hyrox_action_state,
                        origin=(250, 26),
                    )
                except Exception as exc:
                    if not hyrox_action_error_printed:
                        LOGGER.warning("HYROX action overlay failed: %s", exc)
                        hyrox_action_error_printed = True
            if hyrox_action_selector_open:
                draw_hyrox_action_selector(annotated, args.hyrox_action)

            if recording_enabled:
                if record_writer is None:
                    if not record_output_path:
                        record_output_path = str(make_output_path("recordings", ".mp4", root=save_dir))
                    record_writer = create_writer(record_output_path, source_fps, annotated.shape)
                    LOGGER.info("Recording started: %s", record_output_path)
                record_writer.write(annotated)

            if not args.headless:
                cv2.imshow(window_name, annotated)
                delay = 1 if input_mode == "camera" else max(1, int(round(1000.0 / max(source_fps, 1.0))))
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if hyrox_action_selector_open:
                    if key in (27, ord("a"), ord("A")):
                        hyrox_action_selector_open = False
                        set_status("action selection cancelled")
                    else:
                        selected_action = action_from_menu_key(key)
                        if selected_action is not None:
                            switch_hyrox_action(selected_action)
                            hyrox_action_selector_open = False
                    continue
                if key == 27:
                    break
                if key in (ord("a"), ord("A")):
                    hyrox_action_selector_open = True
                    set_status("select action 0-8", seconds=10.0)
                    continue
                if key in (ord("n"), ord("N")):
                    switch_hyrox_action(next_hyrox_action(args.hyrox_action))
                    continue
                if key in (ord("b"), ord("B")):
                    allowed, reason = runtime_backend_switch_allowed(args)
                    if not allowed:
                        set_status(f"switch disabled: {reason}")
                        LOGGER.info("Backend switch ignored: %s", reason)
                        continue

                    target_backend = next_runtime_backend(resolved_backend)
                    LOGGER.info(
                        "Switching backend: %s -> %s",
                        resolved_backend,
                        target_backend,
                    )
                    try:
                        new_backend, new_backend_device = create_runtime_backend(args, target_backend)
                    except Exception as exc:
                        set_status(f"switch failed: {target_backend}")
                        LOGGER.error("[BCK001] Backend switch failed: %s", exc)
                        continue

                    if backend is not None:
                        try:
                            backend.close()
                        except Exception as exc:
                            LOGGER.warning("[REC001] failed to close old backend: %s", exc)
                    backend = new_backend
                    resolved_backend = target_backend
                    runtime_backend_device = new_backend_device
                    metrics.set_backend(resolved_backend, runtime_backend_device)
                    smoother = create_runtime_smoother(args)
                    kinematics_processor.reset()
                    set_status(f"backend switched to {resolved_backend}")
                    LOGGER.info(
                        "Backend switched: %s (device: %s)",
                        resolved_backend,
                        runtime_backend_device,
                    )
                elif key in (ord("m"), ord("M")):
                    mirror_enabled = not mirror_enabled
                    smoother.reset()
                    kinematics_processor.reset()
                    hand_detections = {}
                    last_hand_detection_timestamp_ms = -1
                    set_status(f"mirror {'on' if mirror_enabled else 'off'}")
                    LOGGER.info("Mirror: %s", "ON" if mirror_enabled else "OFF")
                elif key in (ord("v"), ord("V")):
                    if session_writer.is_active:
                        set_status("stop session before changing view", seconds=3.0)
                        LOGGER.info("Camera view switch ignored while a session is active.")
                        continue
                    args.camera_view = next_camera_view(args.camera_view)
                    if hyrox_analyzer is not None:
                        hyrox_analyzer.set_camera_view(args.camera_view)
                        hyrox_analyzer.reset()
                    set_status(f"camera view {args.camera_view}", seconds=3.0)
                    LOGGER.info("Camera view: %s", args.camera_view)
                elif key in (ord("r"), ord("R")):
                    if recording_enabled:
                        recording_enabled = False
                        saved_path = record_output_path
                        if record_writer is not None:
                            record_writer.release()
                            record_writer = None
                        set_status("record off")
                        if saved_path:
                            LOGGER.info("Recording saved: %s", saved_path)
                        if not args.record:
                            record_output_path = ""
                    else:
                        if not record_output_path:
                            record_output_path = str(make_output_path("recordings", ".mp4", root=save_dir))
                        recording_enabled = True
                        set_status("record on")
                        LOGGER.info("Recording armed: %s", record_output_path)
                elif key in (ord("t"), ord("T")):
                    if raw_recording_enabled:
                        raw_recording_enabled = False
                        saved_path = raw_record_output_path
                        if raw_writer is not None:
                            raw_writer.release()
                            raw_writer = None
                        set_status("raw record off")
                        if saved_path:
                            LOGGER.info("Raw recording saved: %s", saved_path)
                        raw_record_output_path = ""
                    else:
                        if not raw_record_output_path:
                            raw_record_output_path = str(make_output_path("recordings", "_raw.mp4", root=save_dir))
                        raw_recording_enabled = True
                        set_status("raw record on")
                        LOGGER.info("Raw recording armed: %s", raw_record_output_path)
                elif key in (ord("s"), ord("S")):
                    try:
                        screenshot_path = save_screenshot(annotated, root=save_dir)
                        set_status("screenshot saved")
                        LOGGER.info("Screenshot saved: %s", screenshot_path)
                    except Exception as exc:
                        set_status("screenshot failed")
                        LOGGER.error("[OUT001] Screenshot failed: %s", exc)
                elif key == ord("1"):
                    display_mode = "full"
                    set_status("mode full skeleton")
                elif key == ord("3"):
                    metrics_overlay_enabled = not metrics_overlay_enabled
                    set_status(f"metrics {'on' if metrics_overlay_enabled else 'off'}")
                elif key in (ord("f"), ord("F")):
                    display_mode = "full" if display_mode == "no-face" else "no-face"
                    set_status("face landmarks on" if display_mode == "full" else "face landmarks off")
                elif key == ord("6"):
                    display_mode = "no-face"
                    set_status("mode no face")
                elif key == ord("7"):
                    display_mode = "upper-body"
                    set_status("mode upper body")
                elif key == ord("8"):
                    display_mode = "lower-body"
                    set_status("mode lower body")
                elif key in (ord("h"), ord("H")):
                    if hand_overlay_enabled:
                        hand_overlay_enabled = False
                        hand_detections = {}
                        last_hand_detection_timestamp_ms = -1
                        set_status("hands off")
                    else:
                        tracker = ensure_hand_tracker(required=False)
                        if tracker is None:
                            set_status(hand_tracker_error or "hand tracker unavailable", seconds=3.0)
                        else:
                            hand_overlay_enabled = True
                            hand_detections = {}
                            last_hand_detection_timestamp_ms = -1
                            set_status("hands on")
                elif key in (ord("c"), ord("C")):
                    try:
                        if session_writer.is_active:
                            stop_session()
                        else:
                            start_session(display_frame.shape)
                    except Exception as exc:
                        set_status("session failed", seconds=3.0)
                        LOGGER.error("[OUT002] Session operation failed: %s", exc)

        if args.save_metrics:
            try:
                metrics.write_csv(args.save_metrics)
            except Exception as exc:
                raise OutputWriteError(
                    f"无法保存指标 CSV：{args.save_metrics}",
                    hint="检查磁盘空间和输出目录权限",
                ) from exc
        LOGGER.info("Metrics summary:")
        for line in metrics.summary_lines():
            LOGGER.info("  %s", line)
        exit_code = ExitCode.SUCCESS
    except KeyboardInterrupt:
        LOGGER.warning("[RUN130] 用户中断，正在保存可恢复输出并关闭资源")
        exit_code = ExitCode.INTERRUPTED
    except ConfigValidationError as exc:
        error = AppError(
            exc.error_code,
            str(exc),
            exit_code=ExitCode.CONFIG_ERROR,
            hint="修正配置后运行 python -m src.doctor 复查",
        )
        report_error(LOGGER, error, debug=bool(args.debug))
        exit_code = error.exit_code
    except AppError as exc:
        report_error(LOGGER, exc, debug=bool(args.debug))
        exit_code = exc.exit_code
    except Exception as exc:
        error = AppError(
            "RUN001",
            f"运行时发生未预期错误：{exc}",
            exit_code=ExitCode.RUNTIME_ERROR,
            hint="使用 --debug 重现并查看日志中的 traceback",
        )
        report_error(LOGGER, error, debug=bool(args.debug))
        exit_code = error.exit_code
    finally:
        cleanup_errors: list[Exception] = []
        if session_writer.is_active:
            def save_active_session() -> None:
                path = session_writer.stop(final_mirror=mirror_enabled)
                if path is not None:
                    LOGGER.info("Session saved: %s", path)

            error = safe_cleanup(
                LOGGER,
                "active session",
                save_active_session,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if record_writer is not None:
            error = safe_cleanup(
                LOGGER,
                "annotated video writer",
                record_writer.release,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if raw_writer is not None:
            error = safe_cleanup(
                LOGGER,
                "raw video writer",
                raw_writer.release,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if fusion_runner is not None and hasattr(fusion_runner, "close"):
            error = safe_cleanup(
                LOGGER,
                "fusion runner",
                fusion_runner.close,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if hand_tracker is not None:
            error = safe_cleanup(
                LOGGER,
                "hand tracker",
                hand_tracker.close,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if backend is not None:
            error = safe_cleanup(
                LOGGER,
                "pose backend",
                backend.close,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if capture is not None:
            error = safe_cleanup(
                LOGGER,
                "capture",
                capture.release,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        error = safe_cleanup(
            LOGGER,
            "OpenCV windows",
            cv2.destroyAllWindows,
            debug=bool(args.debug),
        )
        if error is not None:
            cleanup_errors.append(error)
        if cleanup_errors and exit_code == ExitCode.SUCCESS:
            exit_code = (
                ExitCode.OUTPUT_ERROR
                if any(isinstance(error, SessionWriteError) for error in cleanup_errors)
                else ExitCode.RUNTIME_ERROR
            )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
