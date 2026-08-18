from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import cv2

from hyrox.action_names import action_from_menu_key, next_hyrox_action
from hyrox.view_policy import next_camera_view
from src.biomechanics.session_writer import SessionConfig, SessionWriteError, SessionWriter
from src.biomechanics.kinematics_3d import ThreeDKinematicsTracker
from src.biomechanics.types import KinematicFrame
from src.biomechanics.velocity import KinematicsProcessor
from src.backends.catalog import is_experimental_backend
from src.backends.base import PoseResult
from src.backends.mediapipe_backend import MediaPipeLiveStreamBackend
from src.detectors.yolo_person_detector import YoloPersonDetector
from src.fusion.yolo_roi_mediapipe import FusionFrameStats, YoloRoiMediaPipeFusion
from src.pose.adapters import format_normalized_pose_debug, normalize_backend_pose_result
from src.paths import resolve_asset
from src.product_pose import load_product_pose_config
from src.latency_audit import LatencyAuditRecorder, derive_desktop_latencies
from src.realtime.feedback_engine import FeedbackEngine, FeedbackState
from src.runtime_hand import HandDetection, MediaPipeHandTracker
from src.utils.angle_utils import body_angles
from src.utils.backend_policy import resolve_backend_choice
from src.utils.draw_utils import draw_bbox, draw_hand_landmarks, draw_hyrox_action_overlay, draw_hyrox_action_selector, draw_hyrox_debug_overlay, draw_landmark_lag_debug, draw_pose_result_filtered, draw_realtime_overlay
from src.utils.device import resolve_torch_device
from src.utils.metrics import RealtimeMetrics
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

from src.realtime.backend_runtime import (
    backend_device_for,
    create_display_smoother,
    create_runtime_backend,
    create_runtime_smoother,
    next_runtime_backend,
    runtime_backend_switch_allowed,
    validate_runtime_args,
)
from src.realtime.capture import open_capture, read_capture_frame, timestamp_for_frame
from src.realtime.cli import parse_args
from src.realtime.hyrox_analysis import HyroxAnalysisController
from src.realtime.presentation import (
    highlight_keypoint_names_for_mode,
    visible_keypoint_names_for_mode,
)
from src.realtime.recording import create_writer, make_output_path, save_screenshot
from src.realtime.latest_frame import LatestFrameCamera, LatestFrameVideo
from src.realtime.inference_resolution import InferenceResolutionController
from src.realtime.budget import RealtimeBudgetController
from src.realtime.scheduler import LatestOnlyMediaPipeScheduler, PoseAgeGate
from src.realtime.session import build_pose_frame_from_result, current_model_label
from src.realtime.types import CapturedFrame, TimedPoseResult
from src.realtime.camera_motion import LatestCameraMotionWorker
from src.realtime.temporal_pose import TemporalPoseBuffer
from src.realtime.validation import (
    LatestValidationWorker,
    ValidationBudgetGate,
    build_validation_task,
    realtime_layer_contract,
)

LOGGER = logging.getLogger("pose.desktop")


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
    latest_frame_source: LatestFrameCamera | LatestFrameVideo | None = None
    pose_scheduler: LatestOnlyMediaPipeScheduler | None = None
    pose_age_gate: PoseAgeGate | None = None
    budget_controller: RealtimeBudgetController | None = None
    camera_motion_worker: LatestCameraMotionWorker | None = None
    validation_worker: LatestValidationWorker | None = None
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
    startup_hyrox_config = args.hyrox_config or None
    try:
        product_pose_config = load_product_pose_config()
        realtime_latency_config = product_pose_config.realtime_latency
        realtime_smoothing_config = product_pose_config.analysis_smoothing
        use_latest_frame_pipeline = bool(
            realtime_latency_config.latest_frame_only
            and (not args.input_video or not args.headless)
            and resolved_backend == "mediapipe"
            and args.fusion == "none"
            and args.person_detector == "none"
        )
        hyrox_analysis = HyroxAnalysisController(
            action=args.hyrox_action,
            sensitivity=args.hyrox_sensitivity,
            camera_view=args.camera_view,
            startup_config=startup_hyrox_config,
            live_mode=not bool(args.input_video),
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
    latency_audit = LatencyAuditRecorder(mode="desktop")
    exit_code = ExitCode.SUCCESS
    try:
        validate_runtime_args(args, resolved_backend)
        if (
            is_experimental_backend(resolved_backend)
            or args.fusion != "none"
            or args.person_detector != "none"
        ):
            LOGGER.warning(
                "EXPERIMENTAL pose path enabled: backend=%s fusion=%s person_detector=%s; not part of the product runtime",
                resolved_backend,
                args.fusion,
                args.person_detector,
            )
        LOGGER.info(
            "Resolved backend: %s (requested: %s, action_type: %s)",
            resolved_backend,
            args.backend,
            args.action_type,
        )
        try:
            if use_latest_frame_pipeline:
                backend = MediaPipeLiveStreamBackend(
                    args.model,
                    output_segmentation_masks=False,
                    num_poses=1,
                )
                runtime_backend_device = "cpu"
            else:
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
        if use_latest_frame_pipeline:
            latest_frame_source = (
                LatestFrameCamera(capture, source=f"camera:{args.camera}").start()
                if input_mode == "camera"
                else LatestFrameVideo(
                    capture,
                    source_fps=source_fps,
                    source=args.input_video,
                ).start()
            )
            capture = None
            pose_scheduler = LatestOnlyMediaPipeScheduler(
                backend,
                target_pose_fps=realtime_latency_config.target_pose_fps,
                max_pose_fps=realtime_latency_config.max_pose_fps,
            )
            pose_age_gate = PoseAgeGate(
                max_pose_age_ms=realtime_latency_config.max_pose_age_ms,
                max_frame_gap=realtime_latency_config.max_frame_gap,
            )
            LOGGER.info(
                "Desktop latest-frame pipeline enabled (source=%s, buffer=1, pose_fps=%.0f/%.0f, max_pose_age_ms=%.0f, max_frame_gap=%d)",
                input_mode,
                realtime_latency_config.target_pose_fps,
                realtime_latency_config.max_pose_fps,
                realtime_latency_config.max_pose_age_ms,
                realtime_latency_config.max_frame_gap,
            )
        analysis_smoother = create_runtime_smoother(args, realtime_smoothing_config)
        display_smoother = create_display_smoother(product_pose_config.display_smoothing)
        inference_resolution = InferenceResolutionController(
            product_pose_config.pose.inference_width,
            adaptive=product_pose_config.pose.adaptive_resolution,
        )
        if pose_scheduler is not None:
            budget_controller = RealtimeBudgetController(
                target_pose_fps=realtime_latency_config.target_pose_fps,
                max_pose_fps=realtime_latency_config.max_pose_fps,
                min_pose_fps=min(12.0, realtime_latency_config.target_pose_fps),
                warning_pose_age_ms=realtime_latency_config.warning_pose_age_ms,
            )
        LOGGER.info(
            "Pose inference resolution width=%d adaptive=%s (display resolution unchanged)",
            inference_resolution.current_width,
            inference_resolution.adaptive,
        )
        three_d_tracker = ThreeDKinematicsTracker(
            product_pose_config.three_d_kinematics,
            product_pose_config.three_d_quality,
            max_pose_age_ms=realtime_latency_config.max_pose_age_ms,
        )
        temporal_pose_buffer = TemporalPoseBuffer()
        camera_motion_worker = LatestCameraMotionWorker() if input_mode == "camera" else None
        validation_worker = LatestValidationWorker()
        validation_budget_gate = ValidationBudgetGate(
            warning_pose_age_ms=realtime_latency_config.warning_pose_age_ms,
            validation_budget_ms=product_pose_config.three_d_quality.validation_budget_ms,
            maximum_stride=(
                product_pose_config.three_d_quality.validation_maximum_stride
            ),
        )
        feedback_engine = FeedbackEngine()
        started_ns = time.monotonic_ns()
        frame_index = 0
        normalized_pose_error_printed = False
        hyrox_debug_overlay_error_reported = False
        hyrox_action_overlay_error_reported = False
        window_name = "Realtime Keypoint Baseline"
        status_message = ""
        status_until = 0.0
        hyrox_action_selector_open = False
        empty_result = PoseResult(
            keypoints=[],
            connections=(),
            model_name=resolved_backend,
            num_keypoints=0,
            success=False,
            inference_time_ms=0.0,
            timestamp_ms=0,
        )
        result = empty_result
        latest_draw_result = empty_result
        latest_raw_draw_result = empty_result
        latest_draw_timed: TimedPoseResult | None = None
        angles: dict[str, float | None] = {}
        feedback = FeedbackState(
            person_lost=False,
            low_confidence=False,
            keypoints_unstable=False,
            angle_available=False,
            message="Waiting for pose",
        )
        snapshot = metrics.snapshot()
        hyrox_features = None
        hyrox_action_state = None

        def set_status(message: str, seconds: float = 2.5) -> None:
            nonlocal status_message, status_until
            status_message = message
            status_until = time.perf_counter() + seconds

        def switch_hyrox_action(action_name: str) -> bool:
            nonlocal hyrox_action_overlay_error_reported, latest_draw_result, latest_raw_draw_result, latest_draw_timed
            if action_name == args.hyrox_action:
                set_status(f"action remains {action_name}")
                return True
            try:
                hyrox_analysis.switch(action_name)
            except (FileNotFoundError, ValueError) as exc:
                set_status(f"action switch failed: {action_name}", seconds=3.0)
                LOGGER.error("[CFG001] Action switch failed: %s", exc)
                return False
            args.hyrox_action = action_name
            three_d_tracker.reset()
            temporal_pose_buffer.reset()
            if camera_motion_worker is not None:
                camera_motion_worker.reset()
            if validation_worker is not None:
                validation_worker.reset()
            if pose_scheduler is not None:
                pose_scheduler.invalidate()
                if pose_age_gate is not None:
                    pose_age_gate.reset()
                analysis_smoother.reset()
                display_smoother.reset()
                latest_draw_result = empty_result
                latest_raw_draw_result = empty_result
                latest_draw_timed = None
            hyrox_action_overlay_error_reported = False
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
            captured_frame: CapturedFrame | None = None
            capture_read_start_ns = 0
            capture_read_end_ns = 0
            if latest_frame_source is not None:
                captured_frame = latest_frame_source.get_latest(
                    after_frame_id=frame_index,
                    timeout=0.1,
                )
                if captured_frame is None:
                    if latest_frame_source.terminal_read_failure:
                        raise InputSourceError(
                            "摄像头已断开或停止返回画面",
                            hint="重新连接摄像头后再启动程序",
                        )
                    if input_mode == "video" and getattr(latest_frame_source, "exhausted", False):
                        break
                    continue
                raw_frame = captured_frame.image
                frame_index = captured_frame.frame_id
                capture_read_start_ns = captured_frame.capture_read_start_ns
                capture_read_end_ns = captured_frame.capture_read_end_ns or captured_frame.capture_timestamp_ns
            else:
                capture_read_start_ns = time.perf_counter_ns()
                ok, raw_frame = read_capture_frame(
                    capture,
                    input_mode=input_mode,
                    processed_frames=frame_index,
                )
                if not ok or raw_frame is None:
                    break
                capture_read_end_ns = time.perf_counter_ns()
                frame_index += 1
            frame_started = time.perf_counter()
            display_frame = cv2.flip(raw_frame, 1) if input_mode == "camera" and mirror_enabled else raw_frame
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

            timestamp_ms = (
                (
                    captured_frame.source_timestamp_ms
                    if captured_frame.source_timestamp_ms is not None
                    else int(captured_frame.capture_timestamp_ns // 1_000_000)
                )
                if captured_frame is not None
                else timestamp_for_frame(input_mode, started_ns, frame_index, source_fps)
            )
            metrics.record_frame_read(
                source_fps=source_fps,
                frame_timestamp_ms=timestamp_ms,
                capture_ms=max(0, capture_read_end_ns - capture_read_start_ns) / 1_000_000.0,
                wall_time=frame_started,
            )
            analysis_result: PoseResult | None = None
            display_result: PoseResult | None = None
            analysis_frame_id = frame_index
            analysis_timestamp_ms = timestamp_ms
            analysis_frame_started = frame_started
            accepted_timed: TimedPoseResult | None = None
            sync_inference_start_ns = 0
            sync_inference_end_ns = 0
            fusion_stats = FusionFrameStats()
            if pose_scheduler is not None:
                if captured_frame is None or pose_age_gate is None:
                    raise RuntimeError("latest-frame pipeline is not fully initialized")
                prepared_inference = inference_resolution.prepare(display_frame)
                inference_frame = replace(
                    captured_frame,
                    image=prepared_inference.image,
                    width=prepared_inference.width,
                    height=prepared_inference.height,
                    inference_width=prepared_inference.width,
                    inference_height=prepared_inference.height,
                    resize_ms=prepared_inference.resize_ms,
                )
                pose_scheduler.submit(inference_frame)
                candidate = pose_scheduler.latest_result
                now_ns = time.perf_counter_ns()
                if pose_age_gate.accept_for_analysis(
                    candidate,
                    current_frame_id=frame_index,
                    now_ns=now_ns,
                    current_frame_timestamp_ms=timestamp_ms,
                ):
                    accepted_timed = candidate
                    analysis_result = candidate.pose if candidate is not None else None
                    if candidate is not None:
                        decision = (
                            budget_controller.observe(
                                inference_ms=candidate.inference_ms,
                                pose_result_age_ms=max(
                                    0.0,
                                    candidate.source_age_ms(timestamp_ms),
                                ),
                                queue_depth=int(
                                    pose_scheduler.pending_frame_id is not None
                                ),
                                resolution_can_downgrade=(
                                    inference_resolution.can_step_down
                                ),
                            )
                            if budget_controller is not None
                            else None
                        )
                        if decision is not None and decision.action in {
                            "reduce_pose_fps",
                            "increase_pose_fps",
                        }:
                            pose_scheduler.set_target_pose_fps(
                                decision.target_pose_fps
                            )
                        if (
                            decision is not None
                            and decision.action == "reduce_inference_resolution"
                            and inference_resolution.force_step_down()
                        ):
                            LOGGER.info(
                                "Realtime budget reduced pose resolution to width=%d after inference/age P95 %.1f/%.1fms",
                                inference_resolution.current_width,
                                decision.p95_inference_ms,
                                decision.p95_pose_result_age_ms,
                            )
                        if decision is not None and decision.changed:
                            LOGGER.info(
                                "Realtime budget action=%s pose_fps=%.0f width=%d optional_stride=%d reason=%s queue_saturation=%.2f",
                                decision.action,
                                decision.target_pose_fps,
                                inference_resolution.current_width,
                                decision.extra_analysis_stride,
                                decision.reason,
                                decision.queue_saturation_ratio,
                            )
                        analysis_frame_id = candidate.frame_id
                        analysis_timestamp_ms = (
                            int(candidate.pose.timestamp_ms)
                            if candidate.pose is not None and candidate.pose.timestamp_ms is not None
                            else int(candidate.capture_timestamp_ns // 1_000_000)
                        )
                        analysis_frame_started = candidate.capture_timestamp_ns / 1_000_000_000.0
            elif fusion_runner is not None:
                sync_inference_start_ns = time.perf_counter_ns()
                analysis_result, fusion_stats = fusion_runner.detect(display_frame, timestamp_ms=timestamp_ms)
                sync_inference_end_ns = time.perf_counter_ns()
            else:
                sync_inference_start_ns = time.perf_counter_ns()
                if resolved_backend == "mediapipe":
                    prepared_inference = inference_resolution.prepare(display_frame)
                    analysis_result = backend.detect(
                        prepared_inference.image,
                        timestamp_ms=timestamp_ms,
                    )
                    performance = dict(analysis_result.extra.get("performance", {}))
                    performance["resize_ms"] = prepared_inference.resize_ms
                    extra = dict(analysis_result.extra)
                    extra["performance"] = performance
                    extra["inference_resolution"] = {
                        "width": prepared_inference.width,
                        "height": prepared_inference.height,
                    }
                    analysis_result = replace(analysis_result, extra=extra)
                else:
                    analysis_result = backend.detect(display_frame, timestamp_ms=timestamp_ms)
                sync_inference_end_ns = time.perf_counter_ns()
                if resolved_backend == "mediapipe" and inference_resolution.observe(
                    (sync_inference_end_ns - sync_inference_start_ns) / 1_000_000.0
                ):
                    LOGGER.info(
                        "Adaptive pose resolution reduced to width=%d after inference P95 %.1fms",
                        inference_resolution.current_width,
                        inference_resolution.last_p95_ms,
                    )

            if analysis_result is not None:
                analysis_result = replace(
                    analysis_result,
                    frame_id=analysis_frame_id,
                    timestamp_ms=(
                        analysis_result.timestamp_ms
                        if analysis_result.timestamp_ms is not None
                        else analysis_timestamp_ms
                    ),
                )
                normalized_pose = None
                try:
                    frame_height, frame_width = display_frame.shape[:2]
                    normalized_pose = normalize_backend_pose_result(
                        analysis_result,
                        image_width=frame_width,
                        image_height=frame_height,
                        timestamp_ms=(
                            analysis_result.timestamp_ms
                            if analysis_result.timestamp_ms is not None
                            else analysis_timestamp_ms
                        ),
                        frame_id=analysis_frame_id,
                        latency_ms=analysis_result.inference_time_ms,
                    )
                except Exception as exc:
                    if not normalized_pose_error_printed:
                        LOGGER.warning("normalized pose conversion failed: %s", exc)
                        normalized_pose_error_printed = True
                if args.normalized_pose_debug and (analysis_frame_id == 1 or analysis_frame_id % 30 == 0):
                    LOGGER.debug("%s", format_normalized_pose_debug(normalized_pose))
                postprocess_started_ns = time.perf_counter_ns()
                observation_timestamp_ns = (
                    accepted_timed.capture_timestamp_ns
                    if accepted_timed is not None
                    else int(analysis_timestamp_ms * 1_000_000)
                )
                result = analysis_smoother.smooth_result(
                    analysis_result,
                    capture_timestamp_ns=observation_timestamp_ns,
                )
                display_result = display_smoother.smooth_result(
                    analysis_result,
                    capture_timestamp_ns=observation_timestamp_ns,
                )
                pose_age_for_3d_ms = (
                    max(0.0, accepted_timed.source_age_ms(timestamp_ms))
                    if accepted_timed is not None
                    else 0.0
                )
                validation_snapshot = (
                    validation_worker.snapshot() if validation_worker is not None else {}
                )
                result, three_d_result = three_d_tracker.attach(
                    result,
                    capture_timestamp_ns=(
                        accepted_timed.capture_timestamp_ns
                        if accepted_timed is not None
                        else int(analysis_timestamp_ms * 1_000_000)
                    ),
                    pose_age_ms=pose_age_for_3d_ms,
                    image_width=frame_width,
                    image_height=frame_height,
                    raw_result=analysis_result,
                    camera_view=args.camera_view,
                    camera_motion=(
                        camera_motion_worker.snapshot()
                        if camera_motion_worker is not None
                        else None
                    ),
                    validation_enabled=False,
                    validation_result=validation_snapshot,
                )
                validation_stride = validation_budget_gate.observe(
                    pose_age_ms=pose_age_for_3d_ms,
                    validation_ms=float(validation_snapshot.get("validation_ms", 0.0)),
                )
                if budget_controller is not None:
                    validation_stride = max(
                        validation_stride, budget_controller.extra_analysis_stride
                    )
                if (
                    validation_worker is not None
                    and analysis_frame_id % validation_stride == 0
                ):
                    validation_worker.submit(
                        build_validation_task(
                            result,
                            three_d_result,
                            local_ground_minimum_confidence=(
                                product_pose_config.three_d_quality.local_ground_minimum_confidence
                            ),
                            biomech_minimum_confidence=(
                                product_pose_config.three_d_quality.biomech_minimum_confidence
                            ),
                        )
                    )
                result = replace(
                    result,
                    extra={
                        **result.extra,
                        "realtime_layers": realtime_layer_contract(
                            validation=validation_snapshot,
                            validation_stride=validation_stride,
                        ),
                    },
                )
                if camera_motion_worker is not None:
                    camera_motion_worker.submit(
                        display_frame,
                        timestamp_ms=analysis_timestamp_ms,
                        person_keypoints=result.keypoints,
                    )
                if hand_overlay_enabled:
                    tracker = ensure_hand_tracker(required=False)
                    if tracker is None:
                        hand_overlay_enabled = False
                        hand_detections = {}
                        last_hand_detection_timestamp_ms = -1
                        set_status(hand_tracker_error or "hand tracker unavailable", seconds=3.0)
                    elif hand_detect_interval_ms <= 0 or analysis_timestamp_ms - last_hand_detection_timestamp_ms >= hand_detect_interval_ms * (
                        budget_controller.extra_analysis_stride
                        if budget_controller is not None
                        else 1
                    ):
                        try:
                            hand_detections = tracker.detect(display_frame, timestamp_ms=analysis_timestamp_ms)
                            last_hand_detection_timestamp_ms = analysis_timestamp_ms
                        except Exception as exc:
                            hand_detections = {}
                            hand_overlay_enabled = False
                            last_hand_detection_timestamp_ms = -1
                            set_status("hand detection failed", seconds=3.0)
                            if not hand_detection_error_printed:
                                LOGGER.warning("hand detection failed: %s", exc)
                                hand_detection_error_printed = True
                angles = body_angles(result)
                postprocess_finished_ns = time.perf_counter_ns()
                rule_started_ns = postprocess_finished_ns
                feedback = feedback_engine.update(result, angles)
                frame_finished = time.perf_counter()
                snapshot = metrics.update(
                    result,
                    angles,
                    frame_started=analysis_frame_started,
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
                    frame_index=analysis_frame_id,
                    mirror=mirror_enabled,
                    frame_shape=display_frame.shape,
                    fps=snapshot.realtime_fps if snapshot.realtime_fps > 0 else source_fps,
                    hand_detections=hand_detections if hand_overlay_enabled else {},
                )
                kinematic_frame = kinematics_processor.process(pose_frame)
                if session_writer.is_active:
                    session_writer.add_frame(pose_frame, kinematic_frame)
                analyzed_has_pose = bool(result.success and result.keypoints)
                height, width = display_frame.shape[:2]
                hyrox_features, hyrox_action_state = hyrox_analysis.update(
                    result.keypoints,
                    has_pose=analyzed_has_pose,
                    timestamp_ms=analysis_timestamp_ms,
                    image_width=width,
                    image_height=height,
                    segmentation_mask=result.extra.get("segmentation_mask"),
                    three_d_kinematics=result.extra.get("three_d_kinematics"),
                    extract_when_disabled=bool(args.hyrox_debug),
                )
                temporal_context = temporal_pose_buffer.update(
                    result.keypoints,
                    timestamp_ms=analysis_timestamp_ms,
                    features=hyrox_features,
                    three_d_kinematics=result.extra.get("three_d_kinematics"),
                    phase=(
                        str(hyrox_action_state.get("phase", "unknown"))
                        if hyrox_action_state is not None
                        else "unknown"
                    ),
                )
                result_extra = dict(result.extra)
                result_extra["temporal_motion_context"] = temporal_context
                result = replace(result, extra=result_extra)
                if hyrox_action_state is not None:
                    hyrox_action_state = dict(hyrox_action_state)
                    hyrox_debug = dict(hyrox_action_state.get("debug") or {})
                    hyrox_debug["temporal_motion_context"] = temporal_context
                    hyrox_action_state["debug"] = hyrox_debug
                rule_finished_ns = time.perf_counter_ns()
                pose_age_ms = (
                    max(0.0, accepted_timed.source_age_ms(timestamp_ms))
                    if accepted_timed is not None
                    else max(0.0, (rule_finished_ns - capture_read_end_ns) / 1_000_000.0)
                )
                metrics.record_pose_timing(
                    preprocess_ms=max(
                        0.0,
                        (
                            (accepted_timed.inference_start_ns if accepted_timed is not None else sync_inference_start_ns)
                            - capture_read_end_ns
                        ) / 1_000_000.0,
                    ),
                    postprocess_ms=(postprocess_finished_ns - postprocess_started_ns) / 1_000_000.0,
                    rule_ms=(rule_finished_ns - rule_started_ns) / 1_000_000.0,
                    pose_timestamp_ms=analysis_timestamp_ms,
                    pose_result_age_ms=pose_age_ms,
                    queue_depth=int(pose_scheduler.pending_frame_id is not None) if pose_scheduler is not None else 0,
                    wall_time=rule_finished_ns / 1_000_000_000.0,
                )
                latest_draw_result = display_result
                latest_raw_draw_result = analysis_result
                latest_draw_timed = accepted_timed

            analysis_end_ns = time.perf_counter_ns()
            if pose_scheduler is not None:
                display_sync = (
                    pose_age_gate.synchronization(
                        latest_draw_timed,
                        current_frame_id=frame_index,
                        now_ns=time.perf_counter_ns(),
                        current_frame_timestamp_ms=timestamp_ms,
                    )
                    if pose_age_gate is not None
                    else {"fresh": False, "reason": "gate_missing"}
                )
                if bool(display_sync["fresh"]):
                    display_extra = dict(latest_draw_result.extra)
                    display_extra["display_sync"] = display_sync
                    display_extra["pose_result_age_ms"] = display_sync.get(
                        "pose_result_age_ms"
                    )
                    draw_result = replace(
                        latest_draw_result,
                        extra=display_extra,
                    )
                    raw_draw_result = latest_raw_draw_result
                else:
                    draw_result = replace(
                        empty_result,
                        timestamp_ms=timestamp_ms,
                        frame_id=frame_index,
                        extra={"display_sync": display_sync},
                    )
                    raw_draw_result = draw_result
                metrics.set_realtime_drop_counts(
                    busy=pose_scheduler.busy_drop_count,
                    stale=(
                        pose_scheduler.stale_drop_count
                        + (pose_age_gate.stale_drop_count if pose_age_gate is not None else 0)
                    ),
                    camera_overwrite=(
                        latest_frame_source.overwritten_frame_count
                        if latest_frame_source is not None
                        else 0
                    ),
                    queue_depth=int(pose_scheduler.pending_frame_id is not None),
                    frames_inferred=pose_scheduler.result_count,
                )
                snapshot = metrics.snapshot()
            else:
                draw_result = display_result or replace(
                    empty_result,
                    timestamp_ms=timestamp_ms,
                    frame_id=frame_index,
                )
                display_extra = dict(draw_result.extra)
                display_extra["display_sync"] = {
                    "fresh": True,
                    "reason": "synchronous",
                    "current_frame_id": frame_index,
                    "pose_frame_id": draw_result.frame_id,
                    "frame_gap": 0,
                    "current_frame_timestamp_ms": timestamp_ms,
                    "pose_timestamp_ms": draw_result.timestamp_ms,
                    "pose_result_age_ms": max(
                        0.0,
                        float(timestamp_ms) - float(draw_result.timestamp_ms or timestamp_ms),
                    ),
                }
                draw_result = replace(draw_result, extra=display_extra)
                raw_draw_result = analysis_result or draw_result
            has_pose = bool(draw_result.success and draw_result.keypoints)

            draw_start_ns = time.perf_counter_ns()
            annotated = display_frame.copy()
            visible_names = visible_keypoint_names_for_mode(display_mode)
            highlight_names = highlight_keypoint_names_for_mode(display_mode)
            if args.landmark_lag_debug:
                draw_landmark_lag_debug(
                    annotated,
                    raw_draw_result,
                    draw_result,
                    visible_names=visible_names,
                    highlight_names=highlight_names,
                )
            else:
                draw_pose_result_filtered(
                    annotated,
                    draw_result,
                    visible_names=visible_names,
                    highlight_names=highlight_names,
                )
            if hand_overlay_enabled and hand_detections:
                draw_hand_landmarks(
                    annotated,
                    {side: detection.landmarks for side, detection in hand_detections.items()},
                )
            draw_bbox(annotated, draw_result.bbox)
            overlay_status = status_message if time.perf_counter() < status_until else ""
            draw_realtime_overlay(
                annotated,
                backend=resolved_backend,
                fusion=args.fusion,
                person_detector=args.person_detector,
                detector_every_n=args.detector_every_n,
                smoothing=args.smoothing,
                input_mode=input_mode,
                result=draw_result,
                metrics=snapshot,
                feedback=feedback,
                recording=recording_enabled,
                raw_recording=raw_recording_enabled,
                angles=angles,
                status_message=overlay_status,
            )
            right_panel_x = max(14, annotated.shape[1] - 344)
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
                    if not hyrox_debug_overlay_error_reported:
                        LOGGER.warning("HYROX debug overlay failed: %s", exc)
                        hyrox_debug_overlay_error_reported = True
            if hyrox_analysis.enabled and not args.hyrox_debug:
                try:
                    draw_hyrox_action_overlay(
                        annotated,
                        hyrox_action_state,
                        origin=(250, 26),
                    )
                except Exception as exc:
                    if not hyrox_action_overlay_error_reported:
                        LOGGER.warning("HYROX action overlay failed: %s", exc)
                        hyrox_action_overlay_error_reported = True
            if hyrox_action_selector_open:
                draw_hyrox_action_selector(annotated, args.hyrox_action)

            if recording_enabled:
                if record_writer is None:
                    if not record_output_path:
                        record_output_path = str(make_output_path("recordings", ".mp4", root=save_dir))
                    record_writer = create_writer(record_output_path, source_fps, annotated.shape)
                    LOGGER.info("Recording started: %s", record_output_path)
                record_writer.write(annotated)

            draw_end_ns = time.perf_counter_ns()
            imshow_return_ns = draw_end_ns
            render_finished = time.perf_counter()
            metrics.record_render(
                render_ms=(draw_end_ns - draw_start_ns) / 1_000_000.0,
                wall_time=render_finished,
            )
            if metrics.periodic_snapshot_due(render_finished):
                current = metrics.snapshot()
                LOGGER.info(
                    "performance source=%.1f display=%.1f infer=%.1f fps playback=%.3f pose_age=%.1fms queue=%d read/inferred/skipped/rendered=%d/%d/%d/%d",
                    current.source_fps,
                    current.display_fps,
                    current.inference_fps,
                    current.playback_speed_ratio,
                    current.pose_result_age_ms,
                    current.queue_depth,
                    current.frames_read,
                    current.frames_inferred,
                    current.frames_skipped,
                    current.frames_rendered,
                )

            if not args.headless:
                cv2.imshow(window_name, annotated)
                imshow_return_ns = time.perf_counter_ns()
                delay = 1 if pose_scheduler is not None or input_mode == "camera" else max(1, int(round(1000.0 / max(source_fps, 1.0))))
                displayed_timed = latest_draw_timed if pose_scheduler is not None else None
                pose_capture_timestamp_ns = (
                    displayed_timed.capture_timestamp_ns
                    if displayed_timed is not None
                    else capture_read_end_ns
                )
                inference_start_ns = (
                    displayed_timed.inference_start_ns
                    if displayed_timed is not None
                    else sync_inference_start_ns
                )
                inference_end_ns = (
                    displayed_timed.inference_end_ns
                    if displayed_timed is not None
                    else sync_inference_end_ns
                )
                if (
                    draw_result.success
                    and capture_read_end_ns > 0
                    and inference_start_ns > 0
                    and inference_end_ns > 0
                ):
                    latency_timing = {
                        "frame_id": frame_index,
                        "pose_frame_id": displayed_timed.frame_id if displayed_timed is not None else frame_index,
                        "capture_read_start_ns": capture_read_start_ns,
                        "capture_read_end_ns": capture_read_end_ns,
                        "pose_capture_timestamp_ns": pose_capture_timestamp_ns,
                        "inference_start_ns": inference_start_ns,
                        "inference_end_ns": inference_end_ns,
                        "analysis_end_ns": analysis_end_ns,
                        "draw_start_ns": draw_start_ns,
                        "draw_end_ns": draw_end_ns,
                        "imshow_return_ns": imshow_return_ns,
                    }
                    latency_audit.add(latency_timing, derive_desktop_latencies(latency_timing))
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
                    if pose_scheduler is not None:
                        set_status("switch disabled: restart realtime pipeline")
                        LOGGER.info("Backend switch ignored while LIVE_STREAM pipeline is active.")
                        continue
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
                    analysis_smoother = create_runtime_smoother(args, realtime_smoothing_config)
                    display_smoother = create_display_smoother(product_pose_config.display_smoothing)
                    three_d_tracker.reset()
                    temporal_pose_buffer.reset()
                    if camera_motion_worker is not None:
                        camera_motion_worker.reset()
                    if validation_worker is not None:
                        validation_worker.reset()
                    kinematics_processor.reset()
                    set_status(f"backend switched to {resolved_backend}")
                    LOGGER.info(
                        "Backend switched: %s (device: %s)",
                        resolved_backend,
                        runtime_backend_device,
                    )
                elif key in (ord("m"), ord("M")):
                    mirror_enabled = not mirror_enabled
                    if pose_scheduler is not None:
                        pose_scheduler.invalidate()
                        if pose_age_gate is not None:
                            pose_age_gate.reset()
                        latest_draw_result = empty_result
                        latest_raw_draw_result = empty_result
                        latest_draw_timed = None
                    analysis_smoother.reset()
                    display_smoother.reset()
                    three_d_tracker.reset()
                    temporal_pose_buffer.reset()
                    if camera_motion_worker is not None:
                        camera_motion_worker.reset()
                    if validation_worker is not None:
                        validation_worker.reset()
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
                    hyrox_analysis.set_camera_view(args.camera_view)
                    if pose_scheduler is not None:
                        pose_scheduler.invalidate()
                        if pose_age_gate is not None:
                            pose_age_gate.reset()
                        latest_draw_result = empty_result
                        latest_raw_draw_result = empty_result
                        latest_draw_timed = None
                        analysis_smoother.reset()
                        display_smoother.reset()
                    three_d_tracker.reset()
                    temporal_pose_buffer.reset()
                    if camera_motion_worker is not None:
                        camera_motion_worker.reset()
                    if validation_worker is not None:
                        validation_worker.reset()
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
        if args.latency_audit and latency_audit.samples:
            try:
                audit_path = latency_audit.write_json(args.latency_audit)
                LOGGER.info("Latency audit saved: %s", audit_path)
            except Exception as exc:
                raise OutputWriteError(
                    f"无法保存完整延迟审计：{args.latency_audit}",
                    hint="检查磁盘空间和输出目录权限",
                ) from exc
        LOGGER.info("Metrics summary:")
        for line in metrics.summary_lines():
            LOGGER.info("  %s", line)
        if pose_scheduler is not None:
            LOGGER.info(
                "  latest-frame: submitted=%d results=%d busy_drops=%d rate_drops=%d stale_drops=%d unknown_callbacks=%d",
                pose_scheduler.submitted_count,
                pose_scheduler.result_count,
                pose_scheduler.busy_drop_count,
                pose_scheduler.rate_limit_drop_count,
                pose_scheduler.stale_drop_count + (pose_age_gate.stale_drop_count if pose_age_gate is not None else 0),
                pose_scheduler.unknown_callback_count,
            )
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
        if camera_motion_worker is not None:
            error = safe_cleanup(
                LOGGER,
                "camera motion worker",
                camera_motion_worker.close,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if validation_worker is not None:
            error = safe_cleanup(
                LOGGER,
                "validation worker",
                validation_worker.close,
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
        if latest_frame_source is not None:
            error = safe_cleanup(
                LOGGER,
                "latest-frame source",
                latest_frame_source.stop,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if pose_scheduler is not None:
            error = safe_cleanup(
                LOGGER,
                "live pose scheduler",
                pose_scheduler.close,
                debug=bool(args.debug),
            )
            if error is not None:
                cleanup_errors.append(error)
        if backend is not None and pose_scheduler is None:
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
