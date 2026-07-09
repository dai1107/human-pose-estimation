from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from math import nan
from pathlib import Path
from typing import Sequence

import cv2

from hyrox.actions import LungeAnalyzer
from hyrox.features import extract_basic_pose_features
from src.backends.base import PoseBackend, PoseResult
from src.backends.factory import create_backend
from src.biomechanics.landmarks import LANDMARK_NAMES
from src.biomechanics.normalization import normalize_landmarks
from src.biomechanics.session_writer import SessionConfig, SessionWriter
from src.biomechanics.types import KinematicFrame, LandmarkPoint, PoseFrame
from src.biomechanics.velocity import KinematicsProcessor
from src.detectors.yolo_person_detector import YoloPersonDetector
from src.fitness.squat.calibration import StandingCalibrationBuilder
from src.fitness.squat.phase_metrics import build_squat_frame_from_pose
from src.fitness.squat.realtime_overlay import draw_squat_overlay
from src.fitness.squat.rep_detector import SquatRepDetector
from src.fitness.squat.schema import load_squat_config
from src.fusion.yolo_roi_mediapipe import FusionFrameStats, YoloRoiMediaPipeFusion
from src.realtime.feedback_engine import FeedbackEngine
from src.runtime_hand import DEFAULT_HAND_DETECT_WIDTH, DEFAULT_MAX_HAND_DETECT_FPS, HandDetection, MediaPipeHandTracker
from src.sports.basketball.realtime_overlay import draw_basketball_overlay
from src.utils.angle_utils import body_angles
from src.utils.backend_policy import resolve_backend_choice
from src.utils.draw_utils import draw_bbox, draw_hand_landmarks, draw_hyrox_action_overlay, draw_hyrox_debug_overlay, draw_pose_result_filtered, draw_realtime_overlay
from src.utils.device import resolve_torch_device
from src.utils.metrics import RealtimeMetrics
from src.utils.smoothing import KeypointSmoother
from src.ui.metrics_overlay import draw_metrics_overlay


RUNTIME_BACKENDS = ("mediapipe", "yolo-pose")
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
SHOT_KEYPOINT_NAMES = frozenset(
    {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
)
DRAW_MODE_KEYPOINT_NAMES: dict[str, frozenset[str] | None] = {
    "full": None,
    "no-face": NO_FACE_KEYPOINT_NAMES,
    "upper-body": UPPER_BODY_KEYPOINT_NAMES,
    "lower-body": LOWER_BODY_KEYPOINT_NAMES,
    "shot": SHOT_KEYPOINT_NAMES,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime-first human keypoint detection baseline.")
    parser.add_argument("--backend", default="auto", choices=("auto", "mediapipe", "yolo-pose"), help="Pose backend. Default: auto.")
    parser.add_argument("--action-type", default="auto", help="Action type for --backend auto, e.g. squat, rowing, ski_erg, or HYROX video stem.")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="MediaPipe pose model path.")
    parser.add_argument("--yolo-pose-model", default="yolo11n-pose.pt", help="YOLO Pose model path. Default: yolo11n-pose.pt.")
    parser.add_argument("--yolo-device", default="auto", help="YOLO Pose device, e.g. auto, 0, cuda:0, cpu. Default: auto.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument("--input-video", default="", help="Read frames from a video instead of opening a camera.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width. Default: 640.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height. Default: 480.")
    parser.add_argument("--camera-fps", type=float, default=60.0, help="Requested camera FPS. Default: 60.")
    parser.add_argument("--camera-fourcc", default="MJPG", help="Requested camera FourCC. Empty string leaves it unchanged.")
    parser.add_argument("--landmark-profile", default="full", choices=("full", "no-face", "upper-body", "lower-body", "shot"), help="Landmark display profile. Default: full.")
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
    parser.add_argument("--hyrox-debug", action="store_true", help="Show HYROX debug pose features overlay. Default: off.")
    parser.add_argument("--hyrox-action", default="none", choices=("none", "lunge"), help="Enable HYROX action analysis. Default: none.")
    parser.add_argument("--hyrox-sensitivity", default="medium", choices=("low", "medium", "high"), help="HYROX action sensitivity. Default: medium.")
    parser.add_argument("--hyrox-config", default="configs/hyrox/lunge.yaml", help="HYROX analyzer config path. Missing file falls back to defaults.")
    parser.add_argument("--metrics-overlay", action="store_true", help="Show kinematic metrics panel at startup.")
    parser.add_argument("--session-autostart", action="store_true", help="Start a kinematic data session automatically.")
    parser.add_argument("--analysis-mode", default="pose", choices=("pose", "squat", "basketball"), help="Optional realtime analysis mode. Default: pose.")
    parser.add_argument("--camera-view", default="unknown", choices=("side", "front", "front_left", "front_right", "unknown"), help="Camera view for view-sensitive analysis. Default: unknown.")
    parser.add_argument("--shot-type", default="set_shot", choices=("set_shot", "jump_shot"), help="Basketball shot type for realtime panel. Default: set_shot.")
    parser.add_argument("--shooting-side", default="right", choices=("right", "left"), help="Basketball shooting side. Default: right.")
    parser.add_argument("--person-detector", default="none", choices=("none", "yolo"), help="Optional person detector. Default: none.")
    parser.add_argument("--detector-model", default="yolo11n.pt", help="YOLO person detector model. Default: yolo11n.pt.")
    parser.add_argument("--detector-device", default="auto", help="YOLO person detector device, e.g. auto, 0, cuda:0, cpu. Default: auto.")
    parser.add_argument("--detector-every-n", type=int, default=5, help="Run YOLO every N frames. Default: 5.")
    parser.add_argument("--bbox-expand", type=float, default=1.25, help="Expand detector bbox by this scale. Default: 1.25.")
    parser.add_argument("--bbox-smoothing", type=float, default=0.6, help="BBox smoothing alpha. Default: 0.6.")
    parser.add_argument("--target-select", default="confidence", choices=("confidence", "area"), help="Select person by confidence or area.")
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
            raise RuntimeError(f"could not open input video: {args.input_video}")
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
            raise RuntimeError("--camera-fourcc must be exactly 4 characters, or empty")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not capture.isOpened():
        raise RuntimeError(f"camera {args.camera} could not be opened")
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera {args.camera} opened "
        f"(requested FPS: {args.camera_fps:g}, reported FPS: {fps if fps > 0 else 0:.1f}, FourCC: {args.camera_fourcc or 'unchanged'})"
    )
    return capture, "camera", fps if fps > 0 else 30.0


def create_writer(path: str, fps: float, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    suffix = output.suffix.lower()
    fourcc_name = "mp4v" if suffix == ".mp4" else "XVID"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*fourcc_name), max(1.0, min(60.0, fps)), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create video writer: {output}")
    return writer


def make_output_path(directory_name: str, suffix: str, root: str | Path = "outputs") -> Path:
    output_dir = Path(root) / directory_name
    output_dir.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(f"could not save screenshot: {path}")
    return path


def visible_keypoint_names_for_mode(mode: str) -> set[str] | None:
    allowed = DRAW_MODE_KEYPOINT_NAMES.get(mode)
    if allowed is None:
        return None
    return set(allowed)


def highlight_keypoint_names_for_mode(mode: str) -> set[str]:
    return set(SHOT_KEYPOINT_NAMES) if mode == "shot" else set()


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
        raise RuntimeError("--fusion yolo-roi-mediapipe only supports --backend mediapipe")
    if args.fusion == "yolo-roi-mediapipe" and args.person_detector != "yolo":
        raise RuntimeError("--fusion yolo-roi-mediapipe requires --person-detector yolo")
    if resolved_backend != "mediapipe" and args.person_detector != "none":
        raise RuntimeError("--person-detector yolo is only for MediaPipe ROI; use --person-detector none with this backend")


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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
    hyrox_action_enabled = args.hyrox_action != "none"
    hyrox_analyzer = LungeAnalyzer.from_config_path(args.hyrox_config, sensitivity=args.hyrox_sensitivity) if args.hyrox_action == "lunge" else None
    squat_mode_enabled = args.analysis_mode == "squat"
    squat_config = load_squat_config() if squat_mode_enabled else None
    squat_detector = SquatRepDetector(squat_config, None, camera_view=args.camera_view) if squat_mode_enabled else None
    squat_calibration_builder = (
        StandingCalibrationBuilder(
            camera_view=args.camera_view,
            minimum_visibility=float(squat_config.get("data_quality", {}).get("minimum_landmark_visibility", 0.65)),
        )
        if squat_mode_enabled and squat_config is not None
        else None
    )
    squat_panel_enabled = squat_mode_enabled
    squat_active = squat_mode_enabled
    squat_calibration_status = "N/A"
    last_squat_measurement = None
    last_squat_displacement = None
    basketball_mode_enabled = args.analysis_mode == "basketball"
    basketball_panel_enabled = basketball_mode_enabled
    basketball_collecting = False
    basketball_manual_release_ms: int | None = None
    basketball_phase = "IDLE"
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
    try:
        validate_runtime_args(args, resolved_backend)
        print(f"Resolved backend: {resolved_backend} (requested: {args.backend}, action_type: {args.action_type})")
        backend, runtime_backend_device = create_runtime_backend(args, resolved_backend)
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
        window_name = "Realtime Keypoint Baseline"
        status_message = ""
        status_until = 0.0

        def set_status(message: str, seconds: float = 2.5) -> None:
            nonlocal status_message, status_until
            status_message = message
            status_until = time.perf_counter() + seconds

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
            )
            session_id = session_writer.start(config)
            set_status(f"session {session_id}", seconds=3.0)
            print(f"Session started: {session_id}")

        def stop_session() -> None:
            path = session_writer.stop(final_mirror=mirror_enabled)
            if path is not None:
                set_status("session saved", seconds=3.0)
                print(f"Session saved: {path}")

        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("Controls: Q/ESC quit, B backend, S screenshot, R record, T raw record, M mirror, 1 full, 2 shot, 3 metrics, F face, 6 no-face, 7 upper, 8 lower, H hands, C session, K squat calibration, P squat analysis, 4 squat panel, 5 basketball panel, J shot clip marker, L release proxy marker")
        if args.show_hands:
            ensure_hand_tracker(required=True)
            hand_overlay_enabled = True

        while True:
            ok, raw_frame = capture.read()
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
                    print(f"Session start failed: {exc}", file=sys.stderr)
                session_autostart_pending = False

            if raw_recording_enabled:
                if raw_writer is None:
                    if not raw_record_output_path:
                        raw_record_output_path = str(make_output_path("recordings", "_raw.mp4", root=save_dir))
                    raw_writer = create_writer(raw_record_output_path, source_fps, raw_frame.shape)
                    print(f"Raw recording started: {raw_record_output_path}")
                raw_writer.write(raw_frame)

            timestamp_ms = timestamp_for_frame(input_mode, started_ns, frame_index, source_fps)
            if fusion_runner is not None:
                result, fusion_stats = fusion_runner.detect(display_frame, timestamp_ms=timestamp_ms)
            else:
                result = backend.detect(display_frame, timestamp_ms=timestamp_ms)
                fusion_stats = FusionFrameStats()
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
                            print(f"WARN: hand detection failed: {exc}", file=sys.stderr)
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
                    hyrox_features = extract_basic_pose_features(result.keypoints, image_width=width, image_height=height)
                except Exception as exc:
                    hyrox_features = None
                    if not hyrox_debug_error_printed:
                        print(f"WARN: HYROX debug extraction failed: {exc}", file=sys.stderr)
                        hyrox_debug_error_printed = True
            if hyrox_analyzer is not None:
                try:
                    hyrox_action_state = hyrox_analyzer.update(hyrox_features if has_pose else None, timestamp_ms=timestamp_ms)
                except Exception as exc:
                    hyrox_action_state = None
                    if not hyrox_action_error_printed:
                        print(f"WARN: HYROX action analysis failed: {exc}", file=sys.stderr)
                        hyrox_action_error_printed = True

            if squat_mode_enabled and squat_detector is not None:
                squat_measurement = build_squat_frame_from_pose(pose_frame, kinematic_frame)
                last_squat_measurement = squat_measurement
                if squat_calibration_builder is not None and squat_calibration_builder.active:
                    calibration = squat_calibration_builder.add(squat_measurement)
                    if calibration is not None:
                        squat_detector.set_calibration(calibration)
                        squat_calibration_status = calibration.status
                        set_status(f"squat calibration {calibration.status}", seconds=3.0)
                elif squat_active:
                    completed_rep = squat_detector.update(squat_measurement)
                    if squat_detector.frame_states:
                        value = squat_detector.frame_states[-1].get("pelvis_displacement_normalized")
                        try:
                            last_squat_displacement = float(value) if value != "" else None
                        except (TypeError, ValueError):
                            last_squat_displacement = None
                    if completed_rep is not None:
                        set_status(f"squat rep {completed_rep.rep_index}", seconds=1.5)

            if basketball_mode_enabled:
                side = args.shooting_side
                elbow_angle = getattr(kinematic_frame, f"{side}_elbow_angle", None)
                wrist_speed = getattr(kinematic_frame, f"{side}_wrist_speed", None)
                knee_angle = getattr(kinematic_frame, f"{side}_knee_angle", None)
                try:
                    elbow_value = float(elbow_angle)
                except (TypeError, ValueError):
                    elbow_value = float("nan")
                try:
                    wrist_value = float(wrist_speed)
                except (TypeError, ValueError):
                    wrist_value = float("nan")
                try:
                    knee_value = float(knee_angle)
                except (TypeError, ValueError):
                    knee_value = float("nan")
                if not pose_frame.pose_detected:
                    basketball_phase = "IDLE"
                elif wrist_value == wrist_value and wrist_value > 0.7:
                    basketball_phase = "RELEASE_PROXY" if basketball_manual_release_ms is not None else "ARM_EXTENSION"
                elif elbow_value == elbow_value and elbow_value > 130:
                    basketball_phase = "FOLLOW_THROUGH"
                elif knee_value == knee_value and knee_value < 145:
                    basketball_phase = "DIP"
                else:
                    basketball_phase = "SETUP"

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
            if squat_mode_enabled and squat_panel_enabled and squat_detector is not None:
                if squat_calibration_builder is not None and squat_calibration_builder.active:
                    calibration_label = f"CALIBRATING {squat_calibration_builder.progress(pose_frame.timestamp_ms) * 100.0:.0f}%"
                else:
                    calibration_label = squat_calibration_status
                visibility = getattr(last_squat_measurement, "visibility_mean", None)
                quality_label = "GOOD" if visibility is not None and visibility >= 0.75 else ("WARNING" if visibility is not None and visibility >= 0.55 else "N/A")
                draw_squat_overlay(
                    annotated,
                    {
                        "camera_view": args.camera_view,
                        "calibration_status": calibration_label,
                        "state": squat_detector.machine.state,
                        "rep_count": squat_detector.machine.rep_count,
                        "left_knee_angle": getattr(last_squat_measurement, "left_knee_angle", None),
                        "right_knee_angle": getattr(last_squat_measurement, "right_knee_angle", None),
                        "trunk_tilt_proxy": getattr(last_squat_measurement, "trunk_tilt_proxy", None),
                        "pelvis_displacement": last_squat_displacement,
                        "data_quality": quality_label,
                    },
                    origin=(detail_panel_x, 260 if metrics_overlay_enabled else 120),
                )
            if basketball_mode_enabled and basketball_panel_enabled:
                visibility = getattr(kinematic_frame, "quality", {}).get("visibility_mean") if kinematic_frame is not None else None
                quality_label = "GOOD" if visibility is not None and visibility >= 0.75 else ("WARNING" if visibility is not None and visibility >= 0.55 else "N/A")
                release_label = f"{basketball_manual_release_ms} ms" if basketball_manual_release_ms is not None else "PENDING"
                side = args.shooting_side
                draw_basketball_overlay(
                    annotated,
                    {
                        "shot_type": args.shot_type,
                        "shooting_side": side,
                        "camera_view": args.camera_view,
                        "phase": basketball_phase,
                        "knee_angle": getattr(kinematic_frame, f"{side}_knee_angle", None),
                        "elbow_angle": getattr(kinematic_frame, f"{side}_elbow_angle", None),
                        "pelvis_speed": getattr(kinematic_frame, "pelvis_speed", None),
                        "wrist_speed": getattr(kinematic_frame, f"{side}_wrist_speed", None),
                        "release_proxy": release_label,
                        "data_quality": quality_label,
                    },
                    origin=(detail_panel_x, 260 if metrics_overlay_enabled else 120),
                )
            if args.hyrox_debug:
                try:
                    draw_hyrox_debug_overlay(annotated, hyrox_features, has_pose=has_pose)
                except Exception as exc:
                    if not hyrox_debug_error_printed:
                        print(f"WARN: HYROX debug overlay failed: {exc}", file=sys.stderr)
                        hyrox_debug_error_printed = True
            if hyrox_action_enabled:
                try:
                    draw_hyrox_action_overlay(
                        annotated,
                        hyrox_action_state,
                        origin=(250, 214 if args.hyrox_debug else 26),
                    )
                except Exception as exc:
                    if not hyrox_action_error_printed:
                        print(f"WARN: HYROX action overlay failed: {exc}", file=sys.stderr)
                        hyrox_action_error_printed = True

            if recording_enabled:
                if record_writer is None:
                    if not record_output_path:
                        record_output_path = str(make_output_path("recordings", ".mp4", root=save_dir))
                    record_writer = create_writer(record_output_path, source_fps, annotated.shape)
                    print(f"Recording started: {record_output_path}")
                record_writer.write(annotated)

            if not args.headless:
                cv2.imshow(window_name, annotated)
                delay = 1 if input_mode == "camera" else max(1, int(round(1000.0 / max(source_fps, 1.0))))
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("b"), ord("B")):
                    allowed, reason = runtime_backend_switch_allowed(args)
                    if not allowed:
                        set_status(f"switch disabled: {reason}")
                        print(f"Backend switch ignored: {reason}")
                        continue

                    target_backend = next_runtime_backend(resolved_backend)
                    print(f"Switching backend: {resolved_backend} -> {target_backend}")
                    try:
                        new_backend, new_backend_device = create_runtime_backend(args, target_backend)
                    except Exception as exc:
                        set_status(f"switch failed: {target_backend}")
                        print(f"Backend switch failed: {exc}", file=sys.stderr)
                        continue

                    if backend is not None:
                        try:
                            backend.close()
                        except Exception as exc:
                            print(f"WARN: failed to close old backend: {exc}", file=sys.stderr)
                    backend = new_backend
                    resolved_backend = target_backend
                    runtime_backend_device = new_backend_device
                    metrics.set_backend(resolved_backend, runtime_backend_device)
                    smoother = create_runtime_smoother(args)
                    kinematics_processor.reset()
                    set_status(f"backend switched to {resolved_backend}")
                    print(f"Backend switched: {resolved_backend} (device: {runtime_backend_device})")
                elif key in (ord("m"), ord("M")):
                    mirror_enabled = not mirror_enabled
                    smoother.reset()
                    kinematics_processor.reset()
                    hand_detections = {}
                    last_hand_detection_timestamp_ms = -1
                    set_status(f"mirror {'on' if mirror_enabled else 'off'}")
                    print(f"Mirror: {'ON' if mirror_enabled else 'OFF'}")
                elif key in (ord("r"), ord("R")):
                    if recording_enabled:
                        recording_enabled = False
                        saved_path = record_output_path
                        if record_writer is not None:
                            record_writer.release()
                            record_writer = None
                        set_status("record off")
                        if saved_path:
                            print(f"Recording saved: {saved_path}")
                        if not args.record:
                            record_output_path = ""
                    else:
                        if not record_output_path:
                            record_output_path = str(make_output_path("recordings", ".mp4", root=save_dir))
                        recording_enabled = True
                        set_status("record on")
                        print(f"Recording armed: {record_output_path}")
                elif key in (ord("t"), ord("T")):
                    if raw_recording_enabled:
                        raw_recording_enabled = False
                        saved_path = raw_record_output_path
                        if raw_writer is not None:
                            raw_writer.release()
                            raw_writer = None
                        set_status("raw record off")
                        if saved_path:
                            print(f"Raw recording saved: {saved_path}")
                        raw_record_output_path = ""
                    else:
                        if not raw_record_output_path:
                            raw_record_output_path = str(make_output_path("recordings", "_raw.mp4", root=save_dir))
                        raw_recording_enabled = True
                        set_status("raw record on")
                        print(f"Raw recording armed: {raw_record_output_path}")
                elif key in (ord("s"), ord("S")):
                    try:
                        screenshot_path = save_screenshot(annotated, root=save_dir)
                        set_status("screenshot saved")
                        print(f"Screenshot saved: {screenshot_path}")
                    except Exception as exc:
                        set_status("screenshot failed")
                        print(f"Screenshot failed: {exc}", file=sys.stderr)
                elif key == ord("1"):
                    display_mode = "full"
                    set_status("mode full skeleton")
                elif key == ord("2"):
                    display_mode = "shot"
                    set_status("mode shot joints")
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
                        print(f"Session operation failed: {exc}", file=sys.stderr)
                elif key in (ord("k"), ord("K")):
                    if squat_mode_enabled and squat_calibration_builder is not None:
                        squat_calibration_builder.start()
                        squat_calibration_status = "CALIBRATING"
                        if squat_detector is not None:
                            squat_detector.machine.reset(clear_calibration=True)
                        set_status("squat calibration started")
                    else:
                        set_status("squat mode not enabled")
                elif key in (ord("p"), ord("P")):
                    if squat_mode_enabled:
                        squat_active = not squat_active
                        set_status(f"squat analysis {'on' if squat_active else 'off'}")
                    else:
                        set_status("squat mode not enabled")
                elif key == ord("4"):
                    if squat_mode_enabled:
                        squat_panel_enabled = not squat_panel_enabled
                        set_status(f"squat panel {'on' if squat_panel_enabled else 'off'}")
                    else:
                        set_status("squat mode not enabled")
                elif key == ord("5"):
                    if basketball_mode_enabled:
                        basketball_panel_enabled = not basketball_panel_enabled
                        set_status(f"basketball panel {'on' if basketball_panel_enabled else 'off'}")
                    else:
                        set_status("basketball mode not enabled")
                elif key in (ord("j"), ord("J")):
                    if basketball_mode_enabled:
                        basketball_collecting = not basketball_collecting
                        set_status(f"shot candidate {'on' if basketball_collecting else 'off'}")
                    else:
                        set_status("basketball mode not enabled")
                elif key in (ord("l"), ord("L")):
                    if basketball_mode_enabled:
                        basketball_manual_release_ms = pose_frame.timestamp_ms
                        set_status(f"release proxy {basketball_manual_release_ms} ms")
                    else:
                        set_status("basketball mode not enabled")

        if args.save_metrics:
            metrics.write_csv(args.save_metrics)
        print("Metrics summary:")
        for line in metrics.summary_lines():
            print(f"  {line}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if session_writer.is_active:
            path = session_writer.stop(final_mirror=mirror_enabled)
            if path is not None:
                print(f"Session saved: {path}")
        if record_writer is not None:
            record_writer.release()
        if raw_writer is not None:
            raw_writer.release()
        if fusion_runner is not None and hasattr(fusion_runner, "close"):
            fusion_runner.close()
        if hand_tracker is not None:
            hand_tracker.close()
        if backend is not None:
            backend.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
