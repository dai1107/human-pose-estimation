from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.base import BaseActionAnalyzer
from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from hyrox.view_policy import CAMERA_VIEWS
from src.backends.mediapipe_backend import MediaPipeBackend
from src.configuration import ConfigValidationError
from src.paths import resolve_asset
from src.output_schema import versioned_csv_row
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
from src.utils.draw_utils import draw_hyrox_action_overlay, draw_pose_result_filtered
from src.utils.smoothing import KeypointSmoother
from src.version import __version__

LOGGER = logging.getLogger("pose.replay")

FEATURE_CSV_COLUMNS = (
    "visible_score",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "torso_angle",
    "shoulder_tilt",
    "hip_tilt",
    "body_center_x",
    "body_center_y",
    "body_height_norm",
    "left_wrist_to_hip_y",
    "right_wrist_to_hip_y",
    "wrist_distance_norm",
    "ankle_distance_norm",
    "min_knee_angle",
    "min_hip_angle",
    "max_hip_angle",
    "hip_center_y",
    "knee_center_y",
    "shoulder_center_y",
    "wrist_center_y",
    "hip_knee_depth",
    "wrist_above_shoulder",
    "hip_width",
    "knee_width",
    "ankle_width",
    "min_elbow_angle",
    "max_elbow_angle",
    "left_wrist_above_shoulder",
    "right_wrist_above_shoulder",
)

ANALYZER_DEBUG_CSV_COLUMNS = (
    "camera_view",
    "view_profile",
    "stand_hip_center_y",
    "hip_drop",
    "hip_motion_tolerance",
    "hip_drop_min",
    "raw_phase",
    "stable_phase",
    "frames_in_phase",
    "last_rep_time_ms",
    "bottom_seen",
    "bottom_depth_met",
    "stand_seen",
    "extension_seen",
    "extension_pending",
    "selected_hip_angle",
    "selected_elbow_angle",
    "just_completed_rep",
    "confirmation_frames",
    "rep_cooldown_ms",
    "sensitivity",
    "config_name",
    "wrist_to_hip",
    "carrying_score",
    "motion_detected",
    "stationary_ms",
    "stroke_count",
    "elbow_angle_mean",
    "phase_duration_ms",
    "pull_count",
    "wrist_height_mean",
    "wrist_asymmetry",
    "knee_angle_mean",
    "body_center_delta_x",
    "feet_stagger_score",
    "step_count",
    "ankle_delta",
    "drive_score",
)

DEBUG_CSV_COLUMNS = (
    "frame_index",
    "timestamp_ms",
    "pose_detected",
    *FEATURE_CSV_COLUMNS,
    "action",
    "phase",
    "rep_count",
    "feedback_codes",
    "feedback_texts",
    *ANALYZER_DEBUG_CSV_COLUMNS,
    "schema_version",
    "program_version",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a local HYROX video through MediaPipe pose and the shared HYROX analyzer.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--log-dir", default="outputs/logs", help="Directory for rolling application logs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and tracebacks.")
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument("--hyrox-action", required=True, choices=HYROX_ACTION_NAMES, help="HYROX action analyzer to run during replay.")
    parser.add_argument("--hyrox-sensitivity", default="medium", choices=("low", "medium", "high"), help="HYROX action sensitivity. Default: medium.")
    parser.add_argument("--hyrox-config", default="", help="HYROX analyzer config path. Empty selects the action-specific default.")
    parser.add_argument("--camera-view", default="unknown", choices=CAMERA_VIEWS, help="Camera view used for evaluation. Default: unknown.")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="MediaPipe pose model path.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier, e.g. 0.5, 1.0, 2.0. Default: 1.0.")
    parser.add_argument("--smoothing", default="one-euro", choices=("none", "ema", "one-euro"), help="Keypoint smoothing mode. Default: one-euro, matching main.py.")
    parser.add_argument("--pose-hold-frames", type=int, default=5, help="Hold the last valid pose across short detector drops. Default: 5.")
    parser.add_argument("--headless", action="store_true", help="Process without opening an OpenCV window; useful for batch validation.")
    parser.add_argument("--save-debug-csv", default="", help="Optional CSV path for per-frame features and analyzer outputs.")
    return parser


def create_analyzer(
    action_name: str,
    sensitivity: str,
    config_path: str | None = None,
    camera_view: str = "unknown",
) -> BaseActionAnalyzer:
    return create_action_analyzer(
        action_name,
        config_path,
        sensitivity=sensitivity,
        camera_view=camera_view,
    )


def frame_timestamp_ms(frame_index: int, source_fps: float) -> int:
    fps = source_fps if source_fps > 0 else 30.0
    return max(0, int(round((frame_index * 1000.0) / fps)))


def playback_delay_ms(source_fps: float, speed: float) -> int:
    fps = source_fps if source_fps > 0 else 30.0
    safe_speed = speed if speed > 0 else 1.0
    return max(1, int(round(1000.0 / (fps * safe_speed))))


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def serialize_feedback(messages: Iterable[object] | None, field: str) -> str:
    if messages is None:
        return ""
    values: list[str] = []
    for message in messages:
        if isinstance(message, Mapping):
            values.append(_safe_text(message.get(field, "")))
        else:
            values.append(_safe_text(getattr(message, field, "")))
    return " | ".join(value for value in values if value)


def build_debug_row(
    *,
    frame_index: int,
    timestamp_ms: int,
    has_pose: bool,
    features: Mapping[str, object] | None,
    state: Mapping[str, object] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "pose_detected": int(has_pose),
    }
    for field in FEATURE_CSV_COLUMNS:
        row[field] = None if features is None else features.get(field)
    row["action"] = "" if state is None else state.get("action", "")
    row["phase"] = "" if state is None else state.get("phase", "")
    row["rep_count"] = 0 if state is None else state.get("rep_count", 0)
    feedback_messages = None if state is None else state.get("feedback_messages")
    row["feedback_codes"] = serialize_feedback(feedback_messages, "code")
    row["feedback_texts"] = serialize_feedback(feedback_messages, "text")
    debug = {} if state is None else state.get("debug", {})
    if not isinstance(debug, Mapping):
        debug = {}
    for field in ANALYZER_DEBUG_CSV_COLUMNS:
        row[field] = debug.get(field)
    return versioned_csv_row(row)


def write_debug_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DEBUG_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configure_logging(
            app_name="replay",
            log_dir=args.log_dir,
            debug=bool(args.debug),
        )
    except OSError as exc:
        print(f"ERROR: [OUT003] cannot initialize log directory: {exc}", file=sys.stderr)
        return int(ExitCode.OUTPUT_ERROR)
    if args.speed <= 0:
        error = AppError(
            "CFG004",
            "--speed must be > 0",
            exit_code=ExitCode.CONFIG_ERROR,
        )
        report_error(LOGGER, error, debug=bool(args.debug))
        return int(error.exit_code)
    args.model = str(resolve_asset(args.model))
    if args.hyrox_config:
        args.hyrox_config = str(resolve_asset(args.hyrox_config))

    video_path = Path(args.video)
    if not video_path.exists():
        error = InputSourceError(f"video not found: {video_path}")
        report_error(LOGGER, error, debug=bool(args.debug))
        return int(error.exit_code)

    try:
        analyzer = create_analyzer(args.hyrox_action, args.hyrox_sensitivity, args.hyrox_config or None, args.camera_view)
    except (FileNotFoundError, ConfigValidationError, ValueError) as exc:
        error = AppError(
            getattr(exc, "error_code", "CFG001"),
            str(exc),
            exit_code=ExitCode.CONFIG_ERROR,
        )
        report_error(LOGGER, error, debug=bool(args.debug))
        return int(error.exit_code)
    smoother = KeypointSmoother(mode=args.smoothing, max_missing_frames=max(0, args.pose_hold_frames))
    try:
        backend = MediaPipeBackend(args.model)
    except Exception as exc:
        error = BackendInitializationError(
            f"MediaPipe backend initialization failed: {exc}",
        )
        report_error(LOGGER, error, debug=bool(args.debug))
        return int(error.exit_code)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        backend.close()
        error = InputSourceError(f"could not open video: {video_path}")
        report_error(LOGGER, error, debug=bool(args.debug))
        return int(error.exit_code)

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    delay_ms = playback_delay_ms(source_fps, args.speed)
    window_name = "HYROX Video Replay"
    csv_rows: list[dict[str, object]] | None = [] if args.save_debug_csv else None
    frame_index = 0
    final_state: Mapping[str, object] | None = None

    exit_code = ExitCode.SUCCESS
    try:
        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                if frame_index == 0:
                    raise InputSourceError(
                        "video is empty, damaged, or has no decodable frames",
                    )
                break
            frame_index += 1
            timestamp_ms = frame_timestamp_ms(frame_index, source_fps)
            result = smoother.smooth_result(backend.detect(frame, timestamp_ms=timestamp_ms))
            has_pose = bool(result.success and result.keypoints)
            features = None
            if has_pose:
                height, width = frame.shape[:2]
                features = extract_basic_pose_features(
                    result.keypoints,
                    image_width=width,
                    image_height=height,
                    segmentation_mask=result.extra.get("segmentation_mask"),
                )
            state = analyzer.attach_view_context(analyzer.update(features if has_pose else None, timestamp_ms=timestamp_ms))
            final_state = state

            if not args.headless:
                annotated = frame.copy()
                draw_pose_result_filtered(annotated, result)
                draw_hyrox_action_overlay(annotated, state, origin=(14, 26))
                cv2.putText(
                    annotated,
                    f"frame: {frame_index} speed: {args.speed:.2f}x",
                    (14, max(36, annotated.shape[0] - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, annotated)
            if csv_rows is not None:
                csv_rows.append(
                    build_debug_row(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        has_pose=has_pose,
                        features=features,
                        state=state,
                    )
                )
            if not args.headless:
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
    except KeyboardInterrupt:
        LOGGER.warning("[RUN130] replay interrupted by user")
        exit_code = ExitCode.INTERRUPTED
    except AppError as exc:
        report_error(LOGGER, exc, debug=bool(args.debug))
        exit_code = exc.exit_code
    except Exception as exc:
        error = AppError(
            "RUN001",
            f"replay failed: {exc}",
            exit_code=ExitCode.RUNTIME_ERROR,
            hint="rerun with --debug for a traceback",
        )
        report_error(LOGGER, error, debug=bool(args.debug))
        exit_code = error.exit_code
    finally:
        cleanup_errors = [
            safe_cleanup(LOGGER, "capture", capture.release, debug=bool(args.debug)),
            safe_cleanup(LOGGER, "backend", backend.close, debug=bool(args.debug)),
            safe_cleanup(LOGGER, "OpenCV windows", cv2.destroyAllWindows, debug=bool(args.debug)),
        ]
        if any(error is not None for error in cleanup_errors) and exit_code == ExitCode.SUCCESS:
            exit_code = ExitCode.RUNTIME_ERROR

    if exit_code not in {ExitCode.SUCCESS, ExitCode.INTERRUPTED}:
        return int(exit_code)
    if args.save_debug_csv:
        output_path = Path(args.save_debug_csv)
        try:
            write_debug_csv(output_path, csv_rows or [])
        except Exception as exc:
            error = OutputWriteError(
                f"could not save debug CSV: {output_path}",
            )
            report_error(LOGGER, error, debug=bool(args.debug))
            return int(error.exit_code)
        LOGGER.info("Debug CSV saved: %s", output_path)
    final_phase = "unknown" if final_state is None else str(final_state.get("phase", "unknown"))
    final_rep_count = 0 if final_state is None else int(final_state.get("rep_count", 0))
    LOGGER.info(
        "Replay finished. Processed %s frames from %s. Final phase: %s; reps: %s.",
        frame_index,
        video_path.name,
        final_phase,
        final_rep_count,
    )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
