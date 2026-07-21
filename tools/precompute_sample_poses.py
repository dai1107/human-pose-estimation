"""Generate versioned pose caches for fixed HYROX web samples."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.backends.mediapipe_backend import MediaPipeBackend
from src.backends.yolo_guided_mediapipe_backend import YoloGuidedMediaPipeBackend
from src.paths import resolve_asset
from src.utils.device import resolve_torch_device
from src.validation.golden_videos import load_manifest
from webui.sample_cache import (
    build_cache_payload,
    cache_path_for,
    expected_source_backend,
    serialize_hand_detections,
    serialize_pose_result,
    source_asset_fingerprints,
    write_cache_payload,
)
from webui.hands import WebHandOverlay


def create_backend(action: str):
    if action == "lunge":
        return YoloGuidedMediaPipeBackend(
            PROJECT_ROOT / "yolo11n-pose.pt",
            PROJECT_ROOT / "models" / "pose_landmarker_full.task",
            target_select="tracking",
            device=resolve_torch_device("auto"),
        )
    return MediaPipeBackend(
        PROJECT_ROOT / "models" / "pose_landmarker_full.task",
        output_segmentation_masks=False,
    )


def generate_case(action: str, video: str, *, output: Path | None = None) -> Path:
    video_path = resolve_asset(video)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open sample video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    backend = create_backend(action)
    hand_overlay = WebHandOverlay(PROJECT_ROOT / "models" / "hand_landmarker.task")
    frames = []
    connections = ()
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            result = backend.detect(frame, timestamp_ms=timestamp_ms)
            connections = result.connections
            cached = serialize_pose_result(result)
            cached["hands"] = serialize_hand_detections(
                hand_overlay.update(frame, timestamp_ms=timestamp_ms, enabled=True)
            )
            frames.append(cached)
    finally:
        capture.release()
        backend.close()
        hand_overlay.close()
    if not frames:
        raise RuntimeError(f"sample video contains no frames: {video_path}")
    payload = build_cache_payload(
        action=action,
        video_path=video_path,
        source_backend=expected_source_backend(action),
        fps=fps,
        width=width,
        height=height,
        connections=connections,
        frames=frames,
    )
    target = write_cache_payload(payload, output or cache_path_for(action))
    source_times = [float(frame["source_inference_ms"]) for frame in frames]
    print(
        f"{action}: {len(frames)} frames, "
        f"source inference p50={statistics.median(source_times):.1f} ms -> {target}"
    )
    return target


def augment_case_hands(action: str, video: str) -> Path:
    import gzip
    import json

    target = cache_path_for(action)
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"invalid pose cache: {target}")
    video_path = resolve_asset(video)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open sample video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    hand_overlay = WebHandOverlay(PROJECT_ROOT / "models" / "hand_landmarker.task")
    index = 0
    try:
        while index < len(frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp_ms = int(round((index + 1) * 1000.0 / fps))
            frames[index]["hands"] = serialize_hand_detections(
                hand_overlay.update(frame, timestamp_ms=timestamp_ms, enabled=True)
            )
            index += 1
    finally:
        capture.release()
        hand_overlay.close()
    if index != len(frames):
        raise RuntimeError(f"hand cache frame mismatch for {action}: {index}/{len(frames)}")
    payload["source_assets"] = source_asset_fingerprints(action)
    write_cache_payload(payload, target)
    print(f"{action}: added cached hands to {index} frames -> {target}")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/hyrox_golden_videos.json")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--hands-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _model, cases = load_manifest(args.manifest)
    selected = set(args.case)
    if selected:
        unknown = selected - {case.case_id for case in cases}
        if unknown:
            raise ValueError(f"unknown sample case(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.case_id in selected]
    for case in cases:
        if args.hands_only:
            augment_case_hands(case.action, case.video)
        else:
            generate_case(case.action, case.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
