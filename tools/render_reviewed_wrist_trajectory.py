"""Render a conservative reviewed wristband trajectory on the native timeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends.mediapipe_backend import MediaPipeBackend


def _point(points: object, name: str) -> object | None:
    for point in points or ():
        if getattr(point, "name", "") == name:
            return point
    return None


def _finite(point: object | None) -> bool:
    return point is not None and all(
        math.isfinite(float(getattr(point, axis, math.nan))) for axis in ("x", "y")
    )


def _rotated_crop_to_source(
    point: object,
    panel: tuple[int, int, int, int],
) -> tuple[float, float]:
    x0, y0, x1, y1 = panel
    rotated_x = float(getattr(point, "x"))
    rotated_y = float(getattr(point, "y"))
    return (
        x0 + rotated_y * (x1 - x0),
        y0 + (1.0 - rotated_x) * (y1 - y0),
    )


def extract_torso_reference(
    video_path: Path,
    model_path: Path,
    panel: tuple[int, int, int, int],
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.is_file():
        data = np.load(cache_path)
        return data["shoulder_x"], data["shoulder_y"], data["torso_pixels"]
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    shoulder_x = np.full(frame_count, np.nan, dtype=np.float64)
    shoulder_y = np.full(frame_count, np.nan, dtype=np.float64)
    torso_pixels = np.full(frame_count, np.nan, dtype=np.float64)
    backend = MediaPipeBackend(
        model_path,
        min_pose_detection_confidence=0.18,
        min_pose_presence_confidence=0.18,
        min_tracking_confidence=0.18,
    )
    x0, y0, x1, y1 = panel
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            crop = frame[y0:y1, x0:x1]
            rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            result = backend.detect(rotated, int(round(index * 1000.0 / fps)))
            shoulders = [_point(result.keypoints, name) for name in ("left_shoulder", "right_shoulder")]
            hips = [_point(result.keypoints, name) for name in ("left_hip", "right_hip")]
            valid_shoulders = [_rotated_crop_to_source(p, panel) for p in shoulders if _finite(p)]
            valid_hips = [_rotated_crop_to_source(p, panel) for p in hips if _finite(p)]
            if valid_shoulders:
                sx, sy = np.mean(np.asarray(valid_shoulders), axis=0)
                shoulder_x[index] = sx
                shoulder_y[index] = sy
            if valid_shoulders and valid_hips:
                hx, hy = np.mean(np.asarray(valid_hips), axis=0)
                torso_pixels[index] = math.hypot(float(sx - hx), float(sy - hy))
            if (index + 1) % 300 == 0:
                print(f"torso reference: {index + 1}/{frame_count}", flush=True)
    finally:
        capture.release()
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        shoulder_x=shoulder_x,
        shoulder_y=shoulder_y,
        torso_pixels=torso_pixels,
    )
    return shoulder_x, shoulder_y, torso_pixels


def _interpolate_reference(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    finite = np.isfinite(result)
    if finite.sum() < 2:
        return result
    indices = np.arange(result.size)
    result[~finite] = np.interp(indices[~finite], indices[finite], result[finite])
    kernel = min(31, result.size // 2 * 2 + 1)
    if kernel >= 5:
        result = cv2.GaussianBlur(result.reshape(-1, 1), (1, kernel), 0).reshape(-1)
    return result


def _short_gap_interpolation(values: np.ndarray, maximum_gap: int = 5) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    index = 0
    while index < result.size:
        if math.isfinite(float(result[index])):
            index += 1
            continue
        start = index
        while index < result.size and not math.isfinite(float(result[index])):
            index += 1
        end = index
        if (
            start > 0
            and end < result.size
            and end - start <= maximum_gap
            and math.isfinite(float(result[start - 1]))
            and math.isfinite(float(result[end]))
        ):
            result[start:end] = np.linspace(
                result[start - 1], result[end], end - start + 2
            )[1:-1]
    return result


def load_reviewed_track(
    csv_path: Path,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.full(frame_count, np.nan, dtype=np.float64)
    y = np.full(frame_count, np.nan, dtype=np.float64)
    confidence = np.zeros(frame_count, dtype=np.float64)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["frame_index"])
            if index >= frame_count:
                continue
            value = float(row["appearance_confidence"] or 0.0)
            confidence[index] = value
            if value < 0.20 or not row["wristband_x"] or not row["wristband_y"]:
                continue
            x[index] = float(row["wristband_x"])
            y[index] = float(row["wristband_y"])
    return _short_gap_interpolation(x), _short_gap_interpolation(y), confidence


def _finite_gradient(values: np.ndarray, fps: float) -> np.ndarray:
    result = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    start = 0
    while start < values.size:
        while start < values.size and not finite[start]:
            start += 1
        end = start
        while end < values.size and finite[end]:
            end += 1
        if end - start >= 3:
            result[start:end] = np.gradient(values[start:end], 1.0 / fps)
        start = end
    return result


def build_arrays(
    wrist_x: np.ndarray,
    wrist_y: np.ndarray,
    confidence: np.ndarray,
    shoulder_x: np.ndarray,
    shoulder_y: np.ndarray,
    torso_pixels: np.ndarray,
    fps: float,
) -> dict[str, np.ndarray]:
    shoulder_x = _interpolate_reference(shoulder_x)
    shoulder_y = _interpolate_reference(shoulder_y)
    torso_pixels = _interpolate_reference(torso_pixels)
    reliable_scale = 45.0 / np.clip(torso_pixels, 60.0, 360.0)
    forward = (shoulder_x - wrist_x) * reliable_scale
    depth = (wrist_y - shoulder_y) * reliable_scale
    # Seven-frame Gaussian denoising is chronological and does not normalize
    # the stroke phase or replace it with a constant-speed cycle.
    for values in (forward, depth):
        finite_indices = np.flatnonzero(np.isfinite(values))
        for index in finite_indices:
            left = max(0, index - 3)
            right = min(values.size, index + 4)
            local = values[left:right]
            finite = np.isfinite(local)
            if finite.sum() >= 3:
                weights = np.exp(-0.5 * ((np.arange(left, right) - index) / 1.35) ** 2)
                values[index] = float(np.sum(local[finite] * weights[finite]) / np.sum(weights[finite]))
    velocity_forward = _finite_gradient(forward, fps)
    velocity_depth = _finite_gradient(depth, fps)
    speed = np.hypot(velocity_forward, velocity_depth)
    acceleration_forward = _finite_gradient(velocity_forward, fps)
    acceleration_depth = _finite_gradient(velocity_depth, fps)
    acceleration = np.hypot(acceleration_forward, acceleration_depth)
    return {
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,
        "confidence": confidence,
        "forward_cm": forward,
        "depth_cm": depth,
        "forward_velocity_cm_s": velocity_forward,
        "depth_velocity_cm_s": velocity_depth,
        "speed_cm_s": speed,
        "forward_acceleration_cm_s2": acceleration_forward,
        "depth_acceleration_cm_s2": acceleration_depth,
        "acceleration_cm_s2": acceleration,
    }


def _map_point(
    forward: float,
    depth: float,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    x0, y0, x1, y1 = rect
    x = x0 + (forward - min_x) / max(1e-6, max_x - min_x) * (x1 - x0)
    y = y1 - (depth - min_y) / max(1e-6, max_y - min_y) * (y1 - y0)
    return int(round(x)), int(round(y))


def _robust_bounds(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 10:
        return (-80.0, 80.0, -80.0, 40.0)
    x0, x1 = np.percentile(x[finite], (1, 99))
    y0, y1 = np.percentile(y[finite], (1, 99))
    pad_x = max(8.0, 0.12 * (x1 - x0))
    pad_y = max(8.0, 0.12 * (y1 - y0))
    return float(x0 - pad_x), float(x1 + pad_x), float(y0 - pad_y), float(y1 + pad_y)


def _draw_series(
    canvas: np.ndarray,
    values: np.ndarray,
    frame_index: int,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (46, 55, 76), 2)
    finite = np.isfinite(values)
    scale_values = values[finite]
    maximum = max(1.0, float(np.percentile(scale_values, 97)) if scale_values.size else 1.0)
    start = max(0, frame_index - 300)
    points: list[tuple[int, int]] = []
    for index in range(start, frame_index + 1):
        if not math.isfinite(float(values[index])):
            if len(points) >= 2:
                cv2.polylines(canvas, [np.asarray(points)], False, color, 2, cv2.LINE_AA)
            points = []
            continue
        px = x0 + int((index - start) / max(1, frame_index - start) * (x1 - x0))
        py = y1 - int(min(1.0, max(0.0, float(values[index]) / maximum)) * (y1 - y0))
        points.append((px, py))
    if len(points) >= 2:
        cv2.polylines(canvas, [np.asarray(points)], False, color, 2, cv2.LINE_AA)
    current = values[frame_index]
    text = f"{label}: --" if not math.isfinite(float(current)) else f"{label}: {current:.1f}"
    cv2.putText(canvas, text, (x0 + 12, y0 + 30), 0, 0.72, color, 2, cv2.LINE_AA)


def render_reference(
    path: Path,
    arrays: dict[str, np.ndarray],
    fps: float,
    side: str,
) -> None:
    forward = arrays["forward_cm"]
    depth = arrays["depth_cm"]
    bounds = _robust_bounds(forward, depth)
    width, height = 720, 1280
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    plot_rect = (48, 150, 672, 690)
    finite_points = np.flatnonzero(np.isfinite(forward) & np.isfinite(depth))
    try:
        for frame_index in range(forward.size):
            canvas = np.full((height, width, 3), (17, 22, 34), dtype=np.uint8)
            cv2.putText(
                canvas,
                f"REVIEWED {side.upper()} WRISTBAND TRAJECTORY",
                (40, 62),
                0,
                0.82,
                (235, 242, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"native timeline  {frame_index / fps:06.2f}s  frame {frame_index}",
                (40, 103),
                0,
                0.62,
                (143, 169, 203),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(canvas, plot_rect[:2], plot_rect[2:], (46, 55, 76), 2)
            cv2.putText(canvas, "Side On", (48, 137), 0, 0.78, (235, 242, 255), 2, cv2.LINE_AA)
            for index in finite_points[::3]:
                point = _map_point(forward[index], depth[index], bounds, plot_rect)
                cv2.circle(canvas, point, 1, (75, 105, 125), -1, cv2.LINE_AA)
            trail = []
            for index in range(max(0, frame_index - 75), frame_index + 1):
                if math.isfinite(float(forward[index])) and math.isfinite(float(depth[index])):
                    trail.append(_map_point(forward[index], depth[index], bounds, plot_rect))
                elif len(trail) >= 2:
                    cv2.polylines(canvas, [np.asarray(trail)], False, (255, 210, 40), 3, cv2.LINE_AA)
                    trail = []
            if len(trail) >= 2:
                cv2.polylines(canvas, [np.asarray(trail)], False, (255, 210, 40), 3, cv2.LINE_AA)
            if math.isfinite(float(forward[frame_index])) and math.isfinite(float(depth[frame_index])):
                point = _map_point(forward[frame_index], depth[frame_index], bounds, plot_rect)
                cv2.circle(canvas, point, 12, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(canvas, point, 6, (0, 110, 255), -1, cv2.LINE_AA)
                status = f"DIRECT WRISTBAND  confidence {arrays['confidence'][frame_index]:.2f}"
                status_color = (90, 235, 140)
            else:
                status = "OCCLUDED / IDENTITY EVIDENCE INSUFFICIENT"
                status_color = (80, 180, 255)
            cv2.putText(canvas, status, (48, 735), 0, 0.66, status_color, 2, cv2.LINE_AA)
            _draw_series(
                canvas,
                arrays["speed_cm_s"],
                frame_index,
                (48, 790, 672, 965),
                (60, 220, 255),
                "Speed cm/s",
            )
            _draw_series(
                canvas,
                arrays["acceleration_cm_s2"],
                frame_index,
                (48, 1020, 672, 1195),
                (90, 135, 255),
                "Acceleration cm/s2",
            )
            writer.write(canvas)
    finally:
        writer.release()


def write_csv(path: Path, arrays: dict[str, np.ndarray], fps: float) -> None:
    fields = [
        "frame_index",
        "time_s",
        "wristband_x_px",
        "wristband_y_px",
        "appearance_confidence",
        "forward_cm",
        "depth_cm",
        "forward_velocity_cm_s",
        "depth_velocity_cm_s",
        "speed_cm_s",
        "forward_acceleration_cm_s2",
        "depth_acceleration_cm_s2",
        "acceleration_cm_s2",
        "direct_or_short_gap",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(arrays["wrist_x"].size):
            row: dict[str, object] = {
                "frame_index": index,
                "time_s": f"{index / fps:.6f}",
                "appearance_confidence": f"{arrays['confidence'][index]:.6f}",
                "direct_or_short_gap": int(math.isfinite(float(arrays["wrist_x"][index]))),
            }
            mapping = {
                "wristband_x_px": "wrist_x",
                "wristband_y_px": "wrist_y",
                "forward_cm": "forward_cm",
                "depth_cm": "depth_cm",
                "forward_velocity_cm_s": "forward_velocity_cm_s",
                "depth_velocity_cm_s": "depth_velocity_cm_s",
                "speed_cm_s": "speed_cm_s",
                "forward_acceleration_cm_s2": "forward_acceleration_cm_s2",
                "depth_acceleration_cm_s2": "depth_acceleration_cm_s2",
                "acceleration_cm_s2": "acceleration_cm_s2",
            }
            for output_name, array_name in mapping.items():
                value = arrays[array_name][index]
                row[output_name] = "" if not math.isfinite(float(value)) else f"{value:.6f}"
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("reviewed_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--panel", nargs=4, type=int, required=True)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models" / "pose_landmarker_full.task"
    )
    args = parser.parse_args()
    video = args.video.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    stem = video.stem
    side_cn = "右手" if args.side == "right" else "左手"
    torso_cache = output_dir / f"{stem}_{side_cn}_躯干参考.npz"
    shoulder_x, shoulder_y, torso_pixels = extract_torso_reference(
        video, args.model.resolve(), tuple(args.panel), torso_cache
    )
    wrist_x, wrist_y, confidence = load_reviewed_track(
        args.reviewed_csv.resolve(), frame_count
    )
    arrays = build_arrays(
        wrist_x,
        wrist_y,
        confidence,
        shoulder_x,
        shoulder_y,
        torso_pixels,
        fps,
    )
    reference_video = output_dir / f"{stem}_{side_cn}_腕带高精度轨迹_原速.mp4"
    csv_path = output_dir / f"{stem}_{side_cn}_腕带逐帧速度加速度.csv"
    summary_path = output_dir / f"{stem}_{side_cn}_腕带解析摘要.json"
    render_reference(reference_video, arrays, fps, args.side)
    write_csv(csv_path, arrays, fps)
    finite = np.isfinite(arrays["wrist_x"])
    summary = {
        "input_video": str(video),
        "selected_side": args.side,
        "identity_rule": "black wristband is the hard side-specific identity marker",
        "timebase": "one output frame per input frame at native fps; no phase normalization",
        "uncertain_policy": "no point for occluded or low-evidence frames; only gaps <=5 frames are interpolated",
        "video": {"fps": fps, "frame_count": frame_count, "duration_s": frame_count / fps},
        "quality": {
            "displayed_frame_count": int(finite.sum()),
            "displayed_frame_rate": float(finite.mean()),
            "occluded_or_rejected_frame_count": int((~finite).sum()),
        },
        "outputs": {
            "reference_video": str(reference_video),
            "trajectory_csv": str(csv_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
