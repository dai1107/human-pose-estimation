"""Joint wristband appearance learning plus acceleration-aware optical-flow DP."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from probe_wristband_classifier import _feature, _nearest, _patch


def _load_anchors(path: Path) -> list[tuple[int, float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        (int(item["frame"]), float(item["x"]), float(item["y"]))
        for item in data["anchors"]
    )


def _training_rows(
    video_path: Path,
    candidates_path: Path,
    anchors_path: Path,
) -> tuple[list[np.ndarray], list[int]]:
    candidates = np.load(candidates_path)["candidates"]
    capture = cv2.VideoCapture(str(video_path))
    features: list[np.ndarray] = []
    labels: list[int] = []
    try:
        for frame_index, anchor_x, anchor_y in _load_anchors(anchors_path):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            row = candidates[frame_index]
            positive_index = _nearest(row, anchor_x, anchor_y)
            positive_patch = _patch(
                frame, row[positive_index, 0], row[positive_index, 1]
            )
            for angle in range(0, 360, 30):
                matrix = cv2.getRotationMatrix2D((23.5, 23.5), angle, 1.0)
                rotated = cv2.warpAffine(
                    positive_patch,
                    matrix,
                    (48, 48),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                features.append(_feature(rotated))
                labels.append(1)
            for candidate_index, (x, y, _score) in enumerate(row):
                if candidate_index == positive_index or not (
                    math.isfinite(float(x)) and math.isfinite(float(y))
                ):
                    continue
                if math.hypot(float(x) - anchor_x, float(y) - anchor_y) < 34.0:
                    continue
                features.append(_feature(_patch(frame, x, y)))
                labels.append(0)
    finally:
        capture.release()
    return features, labels


def train_joint_classifier(
    training_sets: list[tuple[Path, Path, Path]],
) -> object:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for video, candidates, anchors in training_sets:
        set_features, set_labels = _training_rows(video, candidates, anchors)
        features.extend(set_features)
        labels.extend(set_labels)
    classifier = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            LinearSVC(C=0.06, class_weight="balanced", dual="auto"), cv=5
        ),
    )
    classifier.fit(np.asarray(features), np.asarray(labels))
    return classifier


def score_video(
    classifier: object,
    video_path: Path,
    candidates_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = np.load(candidates_path)
    candidates = data["candidates"].astype(np.float64)
    panel = tuple(int(value) for value in data["panel"])
    fps = float(data["fps"])
    probabilities = np.zeros(candidates.shape[:2], dtype=np.float64)
    flows = np.zeros((*candidates.shape[:2], 2), dtype=np.float64)
    x0, y0, x1, y1 = panel
    background_capture = cv2.VideoCapture(str(video_path))
    frame_count = candidates.shape[0]
    background_samples: list[np.ndarray] = []
    try:
        for frame_index in np.linspace(0, frame_count - 1, 61).astype(np.int64):
            background_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = background_capture.read()
            if ok:
                background_samples.append(frame[y0:y1, x0:x1])
    finally:
        background_capture.release()
    background = np.median(np.stack(background_samples), axis=0).astype(np.float32)
    capture = cv2.VideoCapture(str(video_path))
    previous_gray: np.ndarray | None = None
    previous_candidates: np.ndarray | None = None
    try:
        for frame_index in range(candidates.shape[0]):
            ok, frame = capture.read()
            if not ok:
                break
            row = candidates[frame_index]
            valid_indices = np.flatnonzero(
                np.isfinite(row[:, 0]) & np.isfinite(row[:, 1])
            )[:24]
            if valid_indices.size:
                batch = np.asarray(
                    [_feature(_patch(frame, row[i, 0], row[i, 1])) for i in valid_indices]
                )
                learned = classifier.predict_proba(batch)[:, 1]
                foreground_values = []
                roi = frame[y0:y1, x0:x1].astype(np.float32)
                difference = np.mean(np.abs(roi - background), axis=2)
                for candidate_index in valid_indices:
                    x, y = row[candidate_index, :2]
                    px = int(round(float(x))) - x0
                    py = int(round(float(y))) - y0
                    patch = difference[
                        max(0, py - 12) : min(difference.shape[0], py + 13),
                        max(0, px - 12) : min(difference.shape[1], px + 13),
                    ]
                    foreground_values.append(float(np.median(patch)) if patch.size else 0.0)
                foreground_values = np.asarray(foreground_values, dtype=np.float64)
                foreground_gate = 1.0 / (1.0 + np.exp(-(foreground_values - 15.0) / 3.5))
                probabilities[frame_index, valid_indices] = learned * foreground_gate
            gray_full = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(
                gray_full,
                (max(1, gray_full.shape[1] // 2), max(1, gray_full.shape[0] // 2)),
                interpolation=cv2.INTER_AREA,
            )
            if previous_gray is not None and previous_candidates is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    0.5,
                    3,
                    21,
                    3,
                    5,
                    1.2,
                    0,
                )
                for candidate_index, (x, y, _score) in enumerate(previous_candidates):
                    if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                        continue
                    px = max(
                        0,
                        min(flow.shape[1] - 1, int(round((float(x) - x0) * 0.5))),
                    )
                    py = max(
                        0,
                        min(flow.shape[0] - 1, int(round((float(y) - y0) * 0.5))),
                    )
                    flows[frame_index - 1, candidate_index] = 2.0 * flow[py, px]
            previous_gray = gray
            previous_candidates = row
            if (frame_index + 1) % 300 == 0:
                print(f"joint scoring: {frame_index + 1}/{candidates.shape[0]}", flush=True)
    finally:
        capture.release()
    return candidates, probabilities, flows, fps


def _solve_segment(
    positions: np.ndarray,
    probabilities: np.ndarray,
    flows: np.ndarray,
    start_state: int,
    end_state: int,
) -> np.ndarray:
    length, states = positions.shape[:2]
    if length == 1:
        return np.asarray([start_state], dtype=np.int16)
    raw_scores = positions[:, :, 2]
    emission = np.full((length, states), 20.0, dtype=np.float64)
    for index in range(length):
        finite = np.isfinite(raw_scores[index])
        if not finite.any():
            continue
        scale = max(12.0, float(np.nanstd(raw_scores[index, finite])))
        top = float(np.nanmax(raw_scores[index, finite]))
        appearance = 5.0 * (1.0 - probabilities[index])
        detector = 0.18 * np.maximum(0.0, (top - raw_scores[index]) / scale)
        emission[index, finite] = appearance[finite] + detector[finite]
    emission[0] = 1e6
    emission[0, start_state] = 0.0
    emission[-1] = 1e6
    emission[-1, end_state] = 0.0

    pair_cost = np.full((states, states), np.inf, dtype=np.float64)
    first_back = np.full((length, states, states), -1, dtype=np.int16)
    start = positions[0, start_state, :2]
    for current in range(states):
        point = positions[1, current, :2]
        if not np.isfinite(point).all():
            continue
        displacement = point - start
        predicted = start + flows[0, start_state]
        flow_error = float(np.linalg.norm(point - predicted))
        speed = float(np.linalg.norm(displacement))
        pair_cost[start_state, current] = (
            emission[1, current]
            + 0.015 * speed
            + 0.010 * max(0.0, speed - 42.0) ** 2
            + min(16.0, (flow_error / 14.0) ** 2)
        )

    for time_index in range(2, length):
        new_cost = np.full((states, states), np.inf, dtype=np.float64)
        previous_positions = positions[time_index - 2, :, :2]
        middle_positions = positions[time_index - 1, :, :2]
        current_positions = positions[time_index, :, :2]
        for middle in range(states):
            middle_point = middle_positions[middle]
            if not np.isfinite(middle_point).all():
                continue
            valid_previous = np.isfinite(previous_positions).all(axis=1) & np.isfinite(pair_cost[:, middle])
            if not valid_previous.any():
                continue
            previous_indices = np.flatnonzero(valid_previous)
            velocity_previous = middle_point - previous_positions[previous_indices]
            predicted_flow = middle_point + flows[time_index - 1, middle]
            for current in range(states):
                current_point = current_positions[current]
                if not np.isfinite(current_point).all():
                    continue
                velocity_current = current_point - middle_point
                acceleration = np.linalg.norm(
                    velocity_current[None, :] - velocity_previous, axis=1
                )
                speed = float(np.linalg.norm(velocity_current))
                flow_error = float(np.linalg.norm(current_point - predicted_flow))
                transition = (
                    0.045 * np.minimum(acceleration, 45.0) ** 2
                    + 0.012 * speed
                    + 0.009 * max(0.0, speed - 42.0) ** 2
                    + min(16.0, (flow_error / 14.0) ** 2)
                )
                totals = pair_cost[previous_indices, middle] + transition
                best_position = int(np.argmin(totals))
                best_previous = int(previous_indices[best_position])
                new_cost[middle, current] = totals[best_position] + emission[time_index, current]
                first_back[time_index, middle, current] = best_previous
        pair_cost = new_cost

    if length == 2:
        return np.asarray([start_state, end_state], dtype=np.int16)
    middle = int(np.argmin(pair_cost[:, end_state]))
    path = np.zeros(length, dtype=np.int16)
    path[-1] = end_state
    path[-2] = middle
    for time_index in range(length - 1, 1, -1):
        previous = int(first_back[time_index, path[time_index - 1], path[time_index]])
        if previous < 0:
            previous = path[time_index - 1]
        path[time_index - 2] = previous
    path[0] = start_state
    return path


def solve_track(
    candidates: np.ndarray,
    probabilities: np.ndarray,
    flows: np.ndarray,
    anchors_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = _load_anchors(anchors_path)
    track = np.full((candidates.shape[0], 2), np.nan, dtype=np.float64)
    confidence = np.zeros(candidates.shape[0], dtype=np.float64)
    selected = np.full(candidates.shape[0], -1, dtype=np.int16)
    for (start, sx, sy), (end, ex, ey) in zip(anchors, anchors[1:]):
        if end <= start:
            continue
        segment_positions = candidates[start : end + 1]
        segment_probabilities = probabilities[start : end + 1]
        segment_flows = flows[start : end + 1]
        start_state = _nearest(segment_positions[0], sx, sy)
        end_state = _nearest(segment_positions[-1], ex, ey)
        path = _solve_segment(
            segment_positions,
            segment_probabilities,
            segment_flows,
            start_state,
            end_state,
        )
        for offset, state in enumerate(path):
            frame_index = start + offset
            selected[frame_index] = state
            track[frame_index] = segment_positions[offset, state, :2]
            confidence[frame_index] = segment_probabilities[offset, state]
    return track, confidence, selected


def write_outputs(
    output_prefix: Path,
    video_path: Path,
    track: np.ndarray,
    confidence: np.ndarray,
    selected: np.ndarray,
    fps: float,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("frame_index,time_s,wristband_x,wristband_y,appearance_confidence,candidate_index\n")
        for index, (x, y) in enumerate(track):
            x_text = "" if not math.isfinite(float(x)) else f"{x:.4f}"
            y_text = "" if not math.isfinite(float(y)) else f"{y:.4f}"
            handle.write(
                f"{index},{index / fps:.6f},{x_text},{y_text},{confidence[index]:.6f},{int(selected[index])}\n"
            )
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_prefix.with_suffix(".mp4")),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame_index in range(track.shape[0]):
            ok, frame = capture.read()
            if not ok:
                break
            x, y = track[frame_index]
            if (
                math.isfinite(float(x))
                and math.isfinite(float(y))
                and confidence[frame_index] >= 0.20
            ):
                point = (int(round(float(x))), int(round(float(y))))
                cv2.circle(frame, point, 14, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, point, 4, (0, 80, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"wristband {confidence[frame_index]:.2f}",
                    (point[0] + 18, point[1] - 12),
                    0,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--training-set", nargs=3, action="append", required=True)
    args = parser.parse_args()
    training_sets = [tuple(Path(value) for value in item) for item in args.training_set]
    classifier = train_joint_classifier(training_sets)
    candidates, probabilities, flows, fps = score_video(
        classifier, args.video, args.candidates
    )
    track, confidence, selected = solve_track(
        candidates, probabilities, flows, args.anchors
    )
    write_outputs(
        args.output_prefix,
        args.video,
        track,
        confidence,
        selected,
        fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
