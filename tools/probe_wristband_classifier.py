"""Probe whether reviewed wristband patches generalize across the two clips."""

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


def _patch(frame: np.ndarray, x: float, y: float, size: int = 48) -> np.ndarray:
    half = size // 2
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_REFLECT_101)
    cx, cy = int(round(x)) + half, int(round(y)) + half
    return padded[cy - half : cy + half, cx - half : cx + half]


def _feature(patch: np.ndarray) -> np.ndarray:
    patch = cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    compact = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_AREA)
    compact = (compact - float(np.mean(compact))) / max(8.0, float(np.std(compact)))
    image_values = compact.reshape(-1).astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    color = []
    for image in (patch.astype(np.float32), hsv):
        for channel in range(3):
            values = image[:, :, channel]
            color.extend(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.percentile(values, 10)),
                    float(np.percentile(values, 50)),
                    float(np.percentile(values, 90)),
                ]
            )
    return np.concatenate((image_values, np.asarray(color, dtype=np.float32)))


def _frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {index}")
    return frame


def _nearest(candidates: np.ndarray, x: float, y: float) -> int:
    return int(np.argmin(np.hypot(candidates[:, 0] - x, candidates[:, 1] - y)))


def train(
    video: Path,
    candidates_path: Path,
    anchors_path: Path,
) -> object:
    candidates = np.load(candidates_path)["candidates"]
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))["anchors"]
    capture = cv2.VideoCapture(str(video))
    features: list[np.ndarray] = []
    labels: list[int] = []
    try:
        for anchor in anchors:
            index = int(anchor["frame"])
            frame = _frame(capture, index)
            row = candidates[index]
            positive = _nearest(row, float(anchor["x"]), float(anchor["y"]))
            positive_patch = _patch(frame, row[positive, 0], row[positive, 1])
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
                if candidate_index == positive or not (math.isfinite(x) and math.isfinite(y)):
                    continue
                if math.hypot(float(x) - float(anchor["x"]), float(y) - float(anchor["y"])) < 32.0:
                    continue
                negative = _patch(frame, x, y)
                features.append(_feature(negative))
                labels.append(0)
    finally:
        capture.release()
    classifier = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(LinearSVC(C=0.08, class_weight="balanced"), cv=4),
    )
    classifier.fit(np.asarray(features), np.asarray(labels))
    return classifier


def evaluate(
    classifier: object,
    video: Path,
    candidates_path: Path,
    anchors_path: Path,
) -> None:
    candidates = np.load(candidates_path)["candidates"]
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))["anchors"]
    capture = cv2.VideoCapture(str(video))
    try:
        for anchor in anchors:
            index = int(anchor["frame"])
            frame = _frame(capture, index)
            row = candidates[index]
            valid = np.isfinite(row[:, 0]) & np.isfinite(row[:, 1])
            valid_indices = np.flatnonzero(valid)
            features = np.asarray([_feature(_patch(frame, row[i, 0], row[i, 1])) for i in valid_indices])
            probabilities = classifier.predict_proba(features)[:, 1]
            target = _nearest(row, float(anchor["x"]), float(anchor["y"]))
            target_position = int(np.flatnonzero(valid_indices == target)[0])
            rank = int(np.sum(probabilities > probabilities[target_position])) + 1
            winner_position = int(np.argmax(probabilities))
            winner = int(valid_indices[winner_position])
            distance = math.hypot(
                float(row[winner, 0]) - float(anchor["x"]),
                float(row[winner, 1]) - float(anchor["y"]),
            )
            print(
                f"frame={index} target_candidate={target + 1} "
                f"classifier_rank={rank} target_p={probabilities[target_position]:.4f} "
                f"winner={winner + 1} winner_p={probabilities[winner_position]:.4f} "
                f"winner_anchor_distance={distance:.2f}"
            )
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_video", type=Path)
    parser.add_argument("train_candidates", type=Path)
    parser.add_argument("train_anchors", type=Path)
    parser.add_argument("test_video", type=Path)
    parser.add_argument("test_candidates", type=Path)
    parser.add_argument("test_anchors", type=Path)
    args = parser.parse_args()
    classifier = train(args.train_video, args.train_candidates, args.train_anchors)
    evaluate(classifier, args.test_video, args.test_candidates, args.test_anchors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
