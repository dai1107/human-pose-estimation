"""Benchmark round-two latest-frame playback without opening a GUI window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.backends.mediapipe_backend import MediaPipeLiveStreamBackend
from src.realtime.latest_frame import LatestFrameVideo
from src.realtime.inference_resolution import InferenceResolutionController
from src.realtime.scheduler import LatestOnlyMediaPipeScheduler
from src.utils.metrics import RealtimeMetrics


def run_benchmark(
    input_video: str | Path,
    *,
    model: str | Path = "models/pose_landmarker_full.task",
    target_pose_fps: float = 15.0,
    max_pose_fps: float = 20.0,
    inference_width: int = 640,
    adaptive_resolution: bool = True,
) -> dict[str, object]:
    source_path = Path(input_video)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open input video: {source_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_fps = source_fps if source_fps > 0.0 else 30.0
    source_frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    source_width = max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    backend = MediaPipeLiveStreamBackend(
        model,
        output_segmentation_masks=False,
        num_poses=1,
    )
    scheduler = LatestOnlyMediaPipeScheduler(
        backend,
        target_pose_fps=target_pose_fps,
        max_pose_fps=max_pose_fps,
    )
    resolution = InferenceResolutionController(
        inference_width,
        adaptive=adaptive_resolution,
    )
    video = LatestFrameVideo(
        capture,
        source_fps=source_fps,
        source=str(source_path),
    ).start()
    metrics = RealtimeMetrics(
        backend="mediapipe-live-stream",
        smoothing="none",
        input_name=str(source_path),
    )
    last_frame_id = 0
    last_result_frame_id = -1
    started = time.perf_counter()
    try:
        while True:
            frame = video.get_latest(after_frame_id=last_frame_id, timeout=0.1)
            if frame is None:
                if video.exhausted:
                    break
                continue
            last_frame_id = frame.frame_id
            now = time.perf_counter()
            frame_timestamp_ms = float(frame.source_timestamp_ms or 0)
            metrics.record_frame_read(
                source_fps=source_fps,
                frame_timestamp_ms=frame_timestamp_ms,
                capture_ms=max(
                    0.0,
                    (frame.capture_read_end_ns - frame.capture_read_start_ns) / 1_000_000.0,
                ),
                wall_time=now,
            )
            prepared = resolution.prepare(frame.image)
            scheduler.submit(
                replace(
                    frame,
                    image=prepared.image,
                    width=prepared.width,
                    height=prepared.height,
                    inference_width=prepared.width,
                    inference_height=prepared.height,
                    resize_ms=prepared.resize_ms,
                )
            )
            result = scheduler.latest_result
            if result is not None and result.frame_id > last_result_frame_id and result.pose is not None:
                resolution.observe(result.inference_ms)
                ready_time = time.perf_counter()
                metrics.update(
                    result.pose,
                    {},
                    frame_started=result.capture_timestamp_ns / 1_000_000_000.0,
                    frame_finished=ready_time,
                )
                metrics.record_pose_timing(
                    preprocess_ms=float(
                        result.pose.extra.get("performance", {}).get("resize_ms", 0.0)
                    ),
                    pose_timestamp_ms=float(result.pose.timestamp_ms or 0),
                    pose_result_age_ms=result.age_ms(time.perf_counter_ns()),
                    queue_depth=int(scheduler.pending_frame_id is not None),
                    wall_time=ready_time,
                )
                last_result_frame_id = result.frame_id
            metrics.set_realtime_drop_counts(
                busy=scheduler.busy_drop_count,
                stale=scheduler.stale_drop_count,
                camera_overwrite=video.overwritten_frame_count,
                queue_depth=int(scheduler.pending_frame_id is not None),
                frames_inferred=scheduler.result_count,
            )
            metrics.record_render(render_ms=0.0, wall_time=time.perf_counter())
    finally:
        video.stop()
        scheduler.close()

    snapshot = metrics.snapshot()
    elapsed_s = max(0.0, time.perf_counter() - started)
    return {
        "schema_version": 1,
        "benchmark_type": "latest_frame_adaptive_resolution_playback",
        "input_video": str(source_path),
        "source_frame_count": source_frame_count,
        "source_resolution": {"width": source_width, "height": source_height},
        "elapsed_seconds": elapsed_s,
        "target_pose_fps": target_pose_fps,
        "max_pose_fps": max_pose_fps,
        "buffer_capacity": 1,
        "pending_capacity": scheduler.pending_capacity,
        "inference_width_initial": int(inference_width),
        "inference_width_final": resolution.current_width,
        "adaptive_resolution": bool(adaptive_resolution),
        "resolution_downgrade_count": resolution.downgrade_count,
        "inference_p95_ms": resolution.last_p95_ms,
        "metrics": asdict(snapshot),
        "acceptance": {
            "playback_speed_ratio_in_range": 0.95 <= snapshot.playback_speed_ratio <= 1.05,
            "queue_depth_bounded": snapshot.queue_depth <= 1,
            "display_not_inference_clock": snapshot.display_fps > snapshot.inference_fps,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--target-pose-fps", type=float, default=15.0)
    parser.add_argument("--max-pose-fps", type=float, default=20.0)
    parser.add_argument("--inference-width", type=int, default=640)
    parser.add_argument("--no-adaptive-resolution", action="store_true")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.input_video,
        model=args.model,
        target_pose_fps=args.target_pose_fps,
        max_pose_fps=args.max_pose_fps,
        inference_width=args.inference_width,
        adaptive_resolution=not args.no_adaptive_resolution,
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
