"""Measure the pre-optimization synchronous pose pipeline latency baseline.

This round-one tool intentionally mirrors the current sequential scheduling:
read one frame, run one backend inference, smooth it, and compute the basic
angles/feedback before reading the next frame.  It does not implement the
latest-frame scheduler planned for later rounds.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.backends.mediapipe_backend import MediaPipeBackend
from src.realtime.feedback_engine import FeedbackEngine
from src.utils.angle_utils import body_angles
from src.utils.smoothing import KeypointSmoother


def summarize_samples(values: Sequence[float]) -> dict[str, float]:
    samples = np.asarray(tuple(float(value) for value in values), dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "mean": float(np.mean(samples)),
    }


def run_baseline(
    input_video: str | Path,
    *,
    model: str | Path = "models/pose_landmarker_full.task",
    max_frames: int = 300,
    warmup_frames: int = 5,
) -> dict[str, Any]:
    source = Path(input_video)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open input video: {source}")

    backend = MediaPipeBackend(model, output_segmentation_masks=False)
    smoother = KeypointSmoother(
        mode="one-euro",
        max_missing_frames=5,
        occlusion_guard=True,
    )
    feedback = FeedbackEngine()
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_fps = source_fps if source_fps > 0 else 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    inference_ms: list[float] = []
    pose_pipeline_ms: list[float] = []
    read_ms: list[float] = []
    successful_poses = 0
    raw_world_frames = 0
    forwarded_world_frames = 0
    decoded_frames = 0

    try:
        while decoded_frames < max(1, int(max_frames)):
            read_started = time.perf_counter()
            ok, frame = capture.read()
            read_finished = time.perf_counter()
            if not ok or frame is None:
                break
            decoded_frames += 1
            timestamp_ms = int(round(decoded_frames * 1000.0 / source_fps))
            pipeline_started = time.perf_counter()
            result = backend.detect(frame, timestamp_ms=timestamp_ms)
            result = smoother.smooth_result(result)
            angles = body_angles(result)
            feedback.update(result, angles)
            pipeline_finished = time.perf_counter()

            raw_result = result.extra.get("raw_result")
            if getattr(raw_result, "pose_world_landmarks", None):
                raw_world_frames += 1
            if result.extra.get("world_landmarks"):
                forwarded_world_frames += 1
            if result.success:
                successful_poses += 1
            if decoded_frames > max(0, int(warmup_frames)):
                read_ms.append((read_finished - read_started) * 1000.0)
                inference_ms.append(float(result.inference_time_ms))
                pose_pipeline_ms.append((pipeline_finished - pipeline_started) * 1000.0)
    finally:
        capture.release()
        backend.close()

    measured_frames = len(pose_pipeline_ms)
    return {
        "schema_version": 1,
        "baseline_type": "round1_synchronous_sequential",
        "input_video": str(source),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "mediapipe": importlib.metadata.version("mediapipe"),
        },
        "source_fps": source_fps,
        "source_width": source_width,
        "source_height": source_height,
        "backend": "mediapipe",
        "running_mode": "VIDEO",
        "decoded_frames": decoded_frames,
        "warmup_frames": min(decoded_frames, max(0, int(warmup_frames))),
        "measured_frames": measured_frames,
        "successful_pose_frames": successful_poses,
        "camera_or_decode_read_ms": summarize_samples(read_ms),
        "pose_inference_ms": summarize_samples(inference_ms),
        "pose_pipeline_ms": summarize_samples(pose_pipeline_ms),
        "raw_world_landmark_frames": raw_world_frames,
        "forwarded_world_landmark_frames": forwarded_world_frames,
        "world_landmarks_currently_discarded": bool(
            raw_world_frames > 0 and forwarded_world_frames == 0
        ),
        "scheduling": "read -> infer -> smooth -> angles/feedback -> next read",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record the round-one synchronous MediaPipe latency baseline; "
            "this does not enable later latest-frame scheduling."
        )
    )
    parser.add_argument("--input-video", required=True, help="Recorded video used as the deterministic input.")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="MediaPipe pose task model.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum decoded frames. Default: 300.")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Frames excluded from latency percentiles. Default: 5.")
    parser.add_argument("--json-output", default="", help="Optional UTF-8 JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_baseline(
        args.input_video,
        model=args.model,
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
