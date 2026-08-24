"""Build a reviewed, marker-first wristband track for composite swim videos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def _band_candidates(frame: np.ndarray, panel: tuple[int, int, int, int], count: int = 30) -> np.ndarray:
    x0, y0, x1, y1 = panel
    roi = frame[y0:y1, x0:x1]
    value = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    small = cv2.GaussianBlur(value, (0, 0), 2.0)
    large = cv2.GaussianBlur(value, (0, 0), 10.0)
    base = large - small + 0.20 * np.maximum(0.0, 80.0 - small)
    maxima = cv2.dilate(base, np.ones((11, 11), dtype=np.uint8))
    ys, xs = np.where((base >= maxima - 1e-3) & (small < 105.0))
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    base_order = np.argsort(base[ys, xs])[::-1][:180]
    scored: list[tuple[float, float, float]] = []
    height, width = value.shape
    angles = np.linspace(0.0, math.pi, 12, endpoint=False)
    for candidate_index in base_order:
        x, y = int(xs[candidate_index]), int(ys[candidate_index])
        if x < 25 or x >= width - 25 or y < 25 or y >= height - 25:
            continue
        core = float(np.median(value[y - 4 : y + 5, x - 4 : x + 5]))
        two_sided_contrast = 0.0
        for angle in angles:
            dx = int(round(18.0 * math.cos(float(angle))))
            dy = int(round(18.0 * math.sin(float(angle))))
            first = value[y + dy - 4 : y + dy + 5, x + dx - 4 : x + dx + 5]
            second = value[y - dy - 4 : y - dy + 5, x - dx - 4 : x - dx + 5]
            contrast = min(float(np.median(first)), float(np.median(second))) - core
            two_sided_contrast = max(two_sided_contrast, contrast)
        score = float(base[y, x]) + 1.8 * max(0.0, two_sided_contrast)
        scored.append((score, float(x + x0), float(y + y0)))
    scored.sort(reverse=True)
    return np.asarray([[x, y, score] for score, x, y in scored[:count]], dtype=np.float32)


def extract_candidates(
    video_path: Path,
    panel: tuple[int, int, int, int],
    output_path: Path,
    *,
    count: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    candidates = np.full((frame_count, count, 3), np.nan, dtype=np.float32)
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                candidates = candidates[:index]
                break
            found = _band_candidates(frame, panel, count=count)
            candidates[index, : found.shape[0]] = found
            if (index + 1) % 300 == 0 or index + 1 == frame_count:
                print(f"band candidates: {index + 1}/{frame_count}", flush=True)
    finally:
        capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        candidates=candidates,
        fps=np.asarray(fps),
        width=np.asarray(width),
        height=np.asarray(height),
        panel=np.asarray(panel, dtype=np.int32),
        video=np.asarray(str(video_path.resolve())),
    )
    return candidates, fps, (width, height)


def render_review(
    video_path: Path,
    candidate_path: Path,
    output_dir: Path,
    *,
    start: int,
    end: int,
    stride: int,
    top: int = 20,
    crop_panel: bool = False,
) -> None:
    data = np.load(candidate_path)
    candidates = data["candidates"]
    panel = tuple(int(value) for value in data["panel"])
    capture = cv2.VideoCapture(str(video_path))
    frames = list(range(max(0, start), min(end, candidates.shape[0] - 1) + 1, max(1, stride)))
    output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[np.ndarray] = []
    try:
        for frame_index in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            for rank, (x, y, _score) in enumerate(candidates[frame_index, :top], start=1):
                if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                    continue
                point = (int(round(float(x))), int(round(float(y))))
                cv2.circle(frame, point, 12, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, str(rank), (point[0] + 9, point[1] - 8), 0, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_index}", (18, 42), 0, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            if crop_panel:
                x0, y0, x1, y1 = panel
                frame = frame[y0:y1, x0:x1]
            panels.append(cv2.resize(frame, (640, 360)))
    finally:
        capture.release()
    for sheet_index in range(0, len(panels), 9):
        group = panels[sheet_index : sheet_index + 9]
        while len(group) < 9:
            group.append(np.zeros_like(panels[0]))
        sheet = np.vstack([np.hstack(group[row : row + 3]) for row in range(0, 9, 3)])
        cv2.imwrite(str(output_dir / f"review_{sheet_index // 9 + 1:02d}.jpg"), sheet)


def _nearest_candidate(candidates: np.ndarray, x: float, y: float) -> int:
    finite = np.isfinite(candidates[:, 0]) & np.isfinite(candidates[:, 1])
    if not finite.any():
        return 0
    distances = np.full(candidates.shape[0], np.inf)
    distances[finite] = np.hypot(candidates[finite, 0] - x, candidates[finite, 1] - y)
    return int(np.argmin(distances))


def solve_reviewed_track(
    candidate_path: Path,
    anchors_path: Path,
    output_csv: Path,
) -> np.ndarray:
    data = np.load(candidate_path)
    candidates = data["candidates"].astype(np.float64)
    fps = float(data["fps"])
    anchors_data = json.loads(anchors_path.read_text(encoding="utf-8"))
    anchors = sorted((int(item["frame"]), float(item["x"]), float(item["y"])) for item in anchors_data["anchors"])
    if len(anchors) < 2:
        raise RuntimeError("at least two reviewed anchors are required")
    track = np.full((candidates.shape[0], 2), np.nan, dtype=np.float64)
    for (start, start_x, start_y), (end, end_x, end_y) in zip(anchors, anchors[1:]):
        if end <= start:
            continue
        segment = candidates[start : end + 1].copy()
        segment[0, 0] = (start_x, start_y, np.nanmax(segment[0, :, 2]) + 500.0)
        segment[-1, 0] = (end_x, end_y, np.nanmax(segment[-1, :, 2]) + 500.0)
        states = segment.shape[1]
        costs = np.full((segment.shape[0], states), np.inf)
        back = np.zeros((segment.shape[0], states), dtype=np.int16)
        costs[0, 0] = 0.0
        for local_index in range(1, segment.shape[0]):
            scores = segment[local_index, :, 2]
            finite_scores = scores[np.isfinite(scores)]
            score_top = float(np.max(finite_scores)) if finite_scores.size else 0.0
            score_scale = max(18.0, float(np.std(finite_scores)) if finite_scores.size else 18.0)
            for state in range(states):
                x, y, score = segment[local_index, state]
                if not all(math.isfinite(float(value)) for value in (x, y, score)):
                    continue
                emission = min(5.0, max(0.0, (score_top - score) / score_scale))
                previous_x = segment[local_index - 1, :, 0]
                previous_y = segment[local_index - 1, :, 1]
                distances = np.hypot(previous_x - x, previous_y - y)
                transition = np.minimum(14.0, (distances / 28.0) ** 2)
                total = costs[local_index - 1] + transition
                previous_state = int(np.argmin(total))
                costs[local_index, state] = total[previous_state] + emission
                back[local_index, state] = previous_state
        state = 0
        path_states = np.zeros(segment.shape[0], dtype=np.int16)
        path_states[-1] = state
        for local_index in range(segment.shape[0] - 1, 0, -1):
            state = int(back[local_index, state])
            path_states[local_index - 1] = state
        for local_index, state in enumerate(path_states):
            track[start + local_index] = segment[local_index, state, :2]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("frame_index,time_s,wristband_x,wristband_y\n")
        for index, (x, y) in enumerate(track):
            x_text = "" if not math.isfinite(x) else f"{x:.4f}"
            y_text = "" if not math.isfinite(y) else f"{y:.4f}"
            handle.write(f"{index},{index / fps:.6f},{x_text},{y_text}\n")
    return track


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("video", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument("--panel", nargs=4, type=int, required=True)
    extract.add_argument("--count", type=int, default=30)
    review = subparsers.add_parser("review")
    review.add_argument("video", type=Path)
    review.add_argument("candidates", type=Path)
    review.add_argument("output_dir", type=Path)
    review.add_argument("--start", type=int, default=0)
    review.add_argument("--end", type=int, default=10**9)
    review.add_argument("--stride", type=int, default=60)
    review.add_argument("--top", type=int, default=20)
    review.add_argument("--crop-panel", action="store_true")
    solve = subparsers.add_parser("solve")
    solve.add_argument("candidates", type=Path)
    solve.add_argument("anchors", type=Path)
    solve.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        extract_candidates(args.video.resolve(), tuple(args.panel), args.output.resolve(), count=args.count)
    elif args.command == "review":
        render_review(
            args.video.resolve(),
            args.candidates.resolve(),
            args.output_dir.resolve(),
            start=args.start,
            end=args.end,
            stride=args.stride,
            top=max(0, args.top),
            crop_panel=args.crop_panel,
        )
    else:
        solve_reviewed_track(args.candidates.resolve(), args.anchors.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
