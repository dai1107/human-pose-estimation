from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import cv2

from hyrox.actions import LungeAnalyzer
from hyrox.features import extract_basic_pose_features
from src.backends.mediapipe_backend import MediaPipeBackend
from src.utils.draw_utils import draw_hyrox_action_overlay, draw_pose_result_filtered


DEBUG_CSV_COLUMNS = (
    "frame_index",
    "timestamp_ms",
    "pose_detected",
    "visible_score",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "torso_angle",
    "min_knee_angle",
    "min_hip_angle",
    "hip_center_y",
    "knee_center_y",
    "action",
    "phase",
    "rep_count",
    "feedback_codes",
    "feedback_texts",
    "raw_phase",
    "stable_phase",
    "frames_in_phase",
    "last_rep_time_ms",
    "bottom_seen",
    "just_completed_rep",
    "confirmation_frames",
    "rep_cooldown_ms",
    "sensitivity",
    "config_name",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a local HYROX video through MediaPipe pose and the shared HYROX analyzer.")
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument("--hyrox-action", required=True, choices=("lunge",), help="HYROX action analyzer to run during replay.")
    parser.add_argument("--hyrox-sensitivity", default="medium", choices=("low", "medium", "high"), help="HYROX action sensitivity. Default: medium.")
    parser.add_argument("--hyrox-config", default="configs/hyrox/lunge.yaml", help="HYROX analyzer config path. Missing file falls back to defaults.")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="MediaPipe pose model path.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier, e.g. 0.5, 1.0, 2.0. Default: 1.0.")
    parser.add_argument("--save-debug-csv", default="", help="Optional CSV path for per-frame features and analyzer outputs.")
    return parser


def create_analyzer(action_name: str, sensitivity: str, config_path: str | None = None) -> LungeAnalyzer:
    if action_name != "lunge":
        raise ValueError(f"unsupported HYROX action: {action_name}")
    return LungeAnalyzer.from_config_path(config_path, sensitivity=sensitivity)


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
    for field in DEBUG_CSV_COLUMNS[3:13]:
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
    for field in DEBUG_CSV_COLUMNS[18:]:
        row[field] = debug.get(field)
    return row


def write_debug_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DEBUG_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.speed <= 0:
        raise SystemExit("--speed must be > 0")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    analyzer = create_analyzer(args.hyrox_action, args.hyrox_sensitivity, args.hyrox_config)
    backend = MediaPipeBackend(args.model)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        backend.close()
        raise SystemExit(f"could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    delay_ms = playback_delay_ms(source_fps, args.speed)
    window_name = "HYROX Video Replay"
    csv_rows: list[dict[str, object]] = []
    frame_index = 0

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            timestamp_ms = frame_timestamp_ms(frame_index, source_fps)
            result = backend.detect(frame, timestamp_ms=timestamp_ms)
            has_pose = bool(result.success and result.keypoints)
            features = None
            if has_pose:
                height, width = frame.shape[:2]
                features = extract_basic_pose_features(result.keypoints, image_width=width, image_height=height)
            state = analyzer.update(features if has_pose else None, timestamp_ms=timestamp_ms)

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
            csv_rows.append(
                build_debug_row(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    has_pose=has_pose,
                    features=features,
                    state=state,
                )
            )
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        backend.close()
        cv2.destroyAllWindows()

    if args.save_debug_csv:
        output_path = Path(args.save_debug_csv)
        write_debug_csv(output_path, csv_rows)
        print(f"Debug CSV saved: {output_path}")
    print(f"Replay finished. Processed {frame_index} frames from {video_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
