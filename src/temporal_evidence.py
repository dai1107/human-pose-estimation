"""Small-data temporal evidence primitives for internal HYROX experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np


PHASE_FEATURE_NAMES: tuple[str, ...] = (
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "torso_angle",
    "hip_center_y",
    "knee_center_y",
    "shoulder_center_y",
    "wrist_center_y",
    "left_wrist_to_shoulder_y",
    "right_wrist_to_shoulder_y",
    "wrist_distance_norm",
    "ankle_distance_norm",
    "body_height_norm",
    "visible_score",
    "minimum_knee_extension_relative",
    "minimum_hip_extension_relative",
)

PHASE_VECTOR_NAMES: tuple[str, ...] = tuple(
    name
    for feature in PHASE_FEATURE_NAMES
    for name in (feature, f"{feature}_delta", f"{feature}_missing")
)


@dataclass(frozen=True, slots=True)
class StandingPoseBaseline:
    knee_angle: float
    hip_angle: float
    hip_center_y: float
    body_height: float
    sample_count: int
    reliable: bool
    quality_score: float = 0.0
    rejection_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "knee_angle": self.knee_angle,
            "hip_angle": self.hip_angle,
            "hip_center_y": self.hip_center_y,
            "body_height": self.body_height,
            "sample_count": self.sample_count,
            "reliable": self.reliable,
            "quality_score": self.quality_score,
            "rejection_reasons": list(self.rejection_reasons),
        }


def estimate_standing_baseline(
    frames: Sequence[Mapping[str, object]],
    *,
    maximum_calibration_frames: int = 120,
) -> StandingPoseBaseline:
    """Estimate an upright baseline without reading phase labels."""

    candidates: list[
        tuple[float, float, float, float, float, float, float, float]
    ] = []
    previous_hip: float | None = None
    for frame in frames[: max(1, int(maximum_calibration_frames))]:
        left_knee = _number(frame.get("left_knee_angle"))
        right_knee = _number(frame.get("right_knee_angle"))
        left_hip = _number(frame.get("left_hip_angle"))
        right_hip = _number(frame.get("right_hip_angle"))
        knees = [value for value in (left_knee, right_knee) if value is not None]
        hips = [value for value in (left_hip, right_hip) if value is not None]
        knee = min(knees) if knees else None
        hip = min(hips) if hips else None
        hip_y = _number(frame.get("hip_center_y"))
        height = _number(
            frame.get("body_height_norm")
            or frame.get("body_box_height_norm")
        )
        visible = _number(frame.get("visible_score")) or 0.0
        speed = (
            0.0
            if previous_hip is None or hip_y is None
            else abs(hip_y - previous_hip)
        )
        if hip_y is not None:
            previous_hip = hip_y
        if (
            knee is None
            or hip is None
            or hip_y is None
            or height is None
            or height <= 1e-6
            or visible < 0.65
        ):
            continue
        upright_score = min(knee, hip) - 400.0 * speed
        candidates.append(
            (
                upright_score,
                knee,
                hip,
                hip_y,
                height,
                speed,
                abs(left_knee - right_knee)
                if left_knee is not None and right_knee is not None
                else 0.0,
                abs(left_hip - right_hip)
                if left_hip is not None and right_hip is not None
                else 0.0,
            )
        )
    candidates.sort(reverse=True)
    selected = candidates[: max(5, min(30, len(candidates) // 3 or 1))]
    if not selected:
        return StandingPoseBaseline(
            165.0,
            165.0,
            0.5,
            0.5,
            0,
            False,
            0.0,
            ("NO_USABLE_UPRIGHT_SAMPLES",),
        )
    knee_angle = float(np.median([row[1] for row in selected]))
    hip_angle = float(np.median([row[2] for row in selected]))
    hip_center_y = float(np.median([row[3] for row in selected]))
    body_height = float(np.median([row[4] for row in selected]))
    median_speed = float(np.median([row[5] for row in selected]))
    knee_mad = float(
        np.median([abs(row[1] - knee_angle) for row in selected])
    )
    hip_mad = float(
        np.median([abs(row[2] - hip_angle) for row in selected])
    )
    height_mad_ratio = float(
        np.median([abs(row[4] - body_height) for row in selected])
        / max(body_height, 1e-6)
    )
    bilateral_gap = float(
        max(
            np.median([row[6] for row in selected]),
            np.median([row[7] for row in selected]),
        )
    )
    rejection_reasons: list[str] = []
    quality_checks = (
        (len(selected) >= 5, "TOO_FEW_UPRIGHT_SAMPLES"),
        (knee_angle >= 140.0, "IMPLAUSIBLE_STANDING_KNEE_ANGLE"),
        (hip_angle >= 135.0, "IMPLAUSIBLE_STANDING_HIP_ANGLE"),
        (0.15 <= body_height <= 1.20, "IMPLAUSIBLE_BODY_HEIGHT"),
        (median_speed <= 0.02, "UNSTABLE_HIP_POSITION"),
        (knee_mad <= 10.0 and hip_mad <= 10.0, "UNSTABLE_EXTENSION_ANGLES"),
        (height_mad_ratio <= 0.12, "UNSTABLE_BODY_HEIGHT"),
        (bilateral_gap <= 35.0, "BILATERAL_ANGLE_DISAGREEMENT"),
    )
    for passed, reason in quality_checks:
        if not passed:
            rejection_reasons.append(reason)
    quality_score = sum(passed for passed, _reason in quality_checks) / len(
        quality_checks
    )
    return StandingPoseBaseline(
        knee_angle=knee_angle,
        hip_angle=hip_angle,
        hip_center_y=hip_center_y,
        body_height=body_height,
        sample_count=len(selected),
        reliable=not rejection_reasons,
        quality_score=float(quality_score),
        rejection_reasons=tuple(rejection_reasons),
    )


def phase_feature_matrix(
    frames: Sequence[Mapping[str, object]],
    baseline: StandingPoseBaseline,
) -> np.ndarray:
    rows: list[list[float]] = []
    previous: dict[str, float] = {}
    for frame in frames:
        enriched = dict(frame)
        knee = _minimum(frame, "left_knee_angle", "right_knee_angle")
        hip = _minimum(frame, "left_hip_angle", "right_hip_angle")
        enriched["minimum_knee_extension_relative"] = (
            None if knee is None else knee - baseline.knee_angle
        )
        enriched["minimum_hip_extension_relative"] = (
            None if hip is None else hip - baseline.hip_angle
        )
        row: list[float] = []
        for name in PHASE_FEATURE_NAMES:
            value = _number(enriched.get(name))
            missing = value is None
            resolved = 0.0 if value is None else value
            delta = 0.0 if name not in previous else resolved - previous[name]
            if not missing:
                previous[name] = resolved
            row.extend((resolved, delta, 1.0 if missing else 0.0))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


@dataclass(slots=True)
class RidgePhaseModel:
    classes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    temperature: float
    transition: np.ndarray
    initial: np.ndarray

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        labels: Sequence[str],
        sequences: Sequence[Sequence[str]],
        *,
        classes: Sequence[str],
        l2: float = 2.0,
    ) -> RidgePhaseModel:
        values = np.asarray(matrix, dtype=np.float64)
        resolved_classes = tuple(str(name) for name in classes)
        index = {name: position for position, name in enumerate(resolved_classes)}
        targets = np.asarray([index[str(label)] for label in labels], dtype=np.int64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        normalized = (values - mean) / scale
        design = np.column_stack((normalized, np.ones(len(normalized))))
        one_hot = np.eye(len(resolved_classes), dtype=np.float64)[targets]
        counts = np.bincount(targets, minlength=len(resolved_classes)).astype(float)
        sample_weight = len(targets) / np.maximum(
            counts[targets] * len(resolved_classes),
            1.0,
        )
        weighted = design * np.sqrt(sample_weight[:, None])
        weighted_targets = one_hot * np.sqrt(sample_weight[:, None])
        regularizer = np.eye(design.shape[1], dtype=float) * float(l2)
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(
            weighted.T @ weighted + regularizer,
            weighted.T @ weighted_targets,
        )
        logits = design @ weights
        temperature = _fit_temperature(logits, targets)
        transition, initial = _transition_probabilities(
            sequences,
            resolved_classes,
        )
        return cls(
            resolved_classes,
            mean,
            scale,
            weights,
            temperature,
            transition,
            initial,
        )

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        normalized = (values - self.mean) / self.scale
        design = np.column_stack((normalized, np.ones(len(normalized))))
        return _softmax(design @ self.weights / self.temperature)

    def predict_linear(self, matrix: np.ndarray) -> list[str]:
        probabilities = self.predict_proba(matrix)
        return [self.classes[index] for index in probabilities.argmax(axis=1)]

    def predict_causal_hmm(self, matrix: np.ndarray) -> list[str]:
        emissions = np.maximum(self.predict_proba(matrix), 1e-12)
        if not len(emissions):
            return []
        posterior = self.initial * emissions[0]
        posterior /= max(float(posterior.sum()), 1e-12)
        output = [self.classes[int(np.argmax(posterior))]]
        for emission in emissions[1:]:
            posterior = (posterior @ self.transition) * emission
            posterior /= max(float(posterior.sum()), 1e-12)
            output.append(self.classes[int(np.argmax(posterior))])
        return output


def phase_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    tolerance_frames: int = 15,
) -> dict[str, float | int | None]:
    size = min(len(expected), len(predicted))
    accuracy = (
        sum(expected[index] == predicted[index] for index in range(size)) / size
        if size
        else None
    )
    expected_boundaries = _phase_boundaries(expected[:size])
    predicted_boundaries = _phase_boundaries(predicted[:size])
    used: set[int] = set()
    errors: list[int] = []
    for phase, frame in expected_boundaries:
        choices = [
            (abs(candidate_frame - frame), index, candidate_frame)
            for index, (candidate_phase, candidate_frame) in enumerate(
                predicted_boundaries
            )
            if index not in used
            and candidate_phase == phase
            and abs(candidate_frame - frame) <= tolerance_frames
        ]
        if not choices:
            continue
        _distance, index, candidate_frame = min(choices)
        used.add(index)
        errors.append(candidate_frame - frame)
    absolute = sorted(abs(value) for value in errors)
    return {
        "frame_count": size,
        "frame_accuracy": accuracy,
        "human_boundary_count": len(expected_boundaries),
        "predicted_boundary_count": len(predicted_boundaries),
        "matched_boundary_count": len(errors),
        "boundary_recall": (
            len(errors) / len(expected_boundaries)
            if expected_boundaries
            else None
        ),
        "boundary_precision": (
            len(errors) / len(predicted_boundaries)
            if predicted_boundaries
            else None
        ),
        "boundary_mae_frames": (
            sum(absolute) / len(absolute) if absolute else None
        ),
        "boundary_median_ae_frames": (
            absolute[len(absolute) // 2] if absolute else None
        ),
    }


def decode_phase_candidates(
    labels: Sequence[str],
    phase_order: Sequence[str],
    *,
    minimum_run_frames: int = 2,
    allow_single_phase_skip: bool = True,
    maximum_phase_skips: int | None = None,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Turn a phase stream into complete, de-duplicated cycle candidates."""

    runs = _runs(labels)
    runs = [
        run for run in runs if run[2] - run[1] + 1 >= minimum_run_frames
    ]
    order = tuple(str(phase) for phase in phase_order)
    allowed_skips = (
        int(bool(allow_single_phase_skip))
        if maximum_phase_skips is None
        else max(0, int(maximum_phase_skips))
    )
    candidates: list[tuple[int, int]] = []
    fragments = 0
    progress = 0
    start: int | None = None
    for phase, run_start, run_end in runs:
        if (
            start is not None
            and progress >= len(order) - 1 - allowed_skips
            and phase == order[-1]
        ):
            candidates.append((start, run_end))
            progress = 1
            start = run_start
            continue
        if phase == order[0]:
            if progress > 1:
                fragments += 1
            progress = 1
            start = run_start
            continue
        if start is None or progress <= 0:
            continue
        expected = order[progress] if progress < len(order) else None
        if phase == expected:
            progress += 1
        elif allowed_skips:
            matching_offset = next(
                (
                    offset
                    for offset in range(1, allowed_skips + 1)
                    if progress + offset < len(order)
                    and phase == order[progress + offset]
                ),
                None,
            )
            if matching_offset is not None:
                progress += matching_offset + 1
            elif phase not in order[:progress]:
                fragments += 1
                progress = 0
                start = None
        elif phase not in order[:progress]:
            fragments += 1
            progress = 0
            start = None
            continue
        if progress >= len(order) and start is not None:
            candidates.append((start, run_end))
            progress = 0
            start = None
    if start is not None:
        fragments += 1
    deduplicated: list[tuple[int, int]] = []
    duplicates = 0
    for candidate in candidates:
        if deduplicated and _interval_iou(candidate, deduplicated[-1]) >= 0.5:
            duplicates += 1
            if candidate[1] - candidate[0] > (
                deduplicated[-1][1] - deduplicated[-1][0]
            ):
                deduplicated[-1] = candidate
        else:
            deduplicated.append(candidate)
    return deduplicated, {
        "raw_complete_candidate_count": len(candidates),
        "deduplicated_candidate_count": len(deduplicated),
        "duplicate_settlement_count": duplicates,
        "incomplete_fragment_count": fragments,
    }


def candidate_metrics(
    expected: Sequence[tuple[int, int]],
    predicted: Sequence[tuple[int, int]],
) -> dict[str, float | int | None]:
    used: set[int] = set()
    terminal_errors: list[int] = []
    for expected_interval in expected:
        choices = [
            (
                _interval_iou(expected_interval, candidate),
                index,
                candidate,
            )
            for index, candidate in enumerate(predicted)
            if index not in used and _interval_iou(expected_interval, candidate) > 0.1
        ]
        if not choices:
            continue
        _iou, index, candidate = max(choices)
        used.add(index)
        terminal_errors.append(candidate[1] - expected_interval[1])
    absolute = sorted(abs(value) for value in terminal_errors)
    return {
        "human_candidate_count": len(expected),
        "predicted_candidate_count": len(predicted),
        "matched_candidate_count": len(terminal_errors),
        "candidate_recall": (
            len(terminal_errors) / len(expected) if expected else None
        ),
        "candidate_precision": (
            len(terminal_errors) / len(predicted) if predicted else None
        ),
        "missed_candidate_count": len(expected) - len(terminal_errors),
        "false_candidate_count": len(predicted) - len(terminal_errors),
        "terminal_mae_frames": (
            sum(absolute) / len(absolute) if absolute else None
        ),
    }


def rle_roi_foreground_ratio(
    mask: Mapping[str, object] | None,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float | None:
    """Measure a rectangular contour proxy directly from row-major RLE."""

    if not isinstance(mask, Mapping):
        return None
    size = mask.get("size")
    counts = mask.get("counts")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or not isinstance(counts, (list, tuple))
    ):
        return None
    height, width = int(size[0]), int(size[1])
    left = max(0, min(width, int(x0)))
    right = max(left, min(width, int(x1)))
    top = max(0, min(height, int(y0)))
    bottom = max(top, min(height, int(y1)))
    area = (right - left) * (bottom - top)
    if area <= 0:
        return None
    roi_foreground = 0
    cursor = 0
    foreground = False
    for raw_count in counts:
        count = max(0, int(raw_count))
        run_start = cursor
        run_end = min(height * width, cursor + count)
        if foreground and run_end > run_start:
            first_row = run_start // width
            last_row = (run_end - 1) // width
            for row in range(max(top, first_row), min(bottom - 1, last_row) + 1):
                row_start = row * width
                overlap_start = max(run_start, row_start + left)
                overlap_end = min(run_end, row_start + right)
                roi_foreground += max(0, overlap_end - overlap_start)
        cursor += count
        foreground = not foreground
        if cursor >= height * width:
            break
    return roi_foreground / area


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    candidates = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    losses = []
    for temperature in candidates:
        probabilities = _softmax(logits / temperature)
        selected = probabilities[np.arange(len(targets)), targets]
        losses.append(
            (
                float(-np.log(np.maximum(selected, 1e-12)).mean()),
                temperature,
            )
        )
    return min(losses)[1]


def _transition_probabilities(
    sequences: Sequence[Sequence[str]],
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    index = {name: position for position, name in enumerate(classes)}
    transition = np.ones((len(classes), len(classes)), dtype=float) * 0.25
    initial = np.ones(len(classes), dtype=float) * 0.25
    for sequence in sequences:
        if not sequence:
            continue
        initial[index[str(sequence[0])]] += 1.0
        for previous, current in zip(sequence, sequence[1:]):
            transition[index[str(previous)], index[str(current)]] += 1.0
    transition /= transition.sum(axis=1, keepdims=True)
    initial /= initial.sum()
    return transition, initial


def _phase_boundaries(labels: Sequence[str]) -> list[tuple[str, int]]:
    return [
        (str(labels[index]), index)
        for index in range(1, len(labels))
        if labels[index] != labels[index - 1]
    ]


def _runs(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    if not labels:
        return []
    output = []
    start = 0
    current = str(labels[0])
    for index, value in enumerate(labels[1:], start=1):
        if str(value) == current:
            continue
        output.append((current, start, index - 1))
        current = str(value)
        start = index
    output.append((current, start, len(labels) - 1))
    return output


def _interval_iou(
    first: tuple[int, int],
    second: tuple[int, int],
) -> float:
    intersection = max(
        0,
        min(first[1], second[1]) - max(first[0], second[0]) + 1,
    )
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return intersection / union if union > 0 else 0.0


def _number(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if isfinite(resolved) else None


def _minimum(
    values: Mapping[str, object],
    first: str,
    second: str,
) -> float | None:
    resolved = [
        number
        for name in (first, second)
        if (number := _number(values.get(name))) is not None
    ]
    return min(resolved) if resolved else None


__all__ = [
    "PHASE_FEATURE_NAMES",
    "PHASE_VECTOR_NAMES",
    "RidgePhaseModel",
    "StandingPoseBaseline",
    "candidate_metrics",
    "decode_phase_candidates",
    "estimate_standing_baseline",
    "phase_feature_matrix",
    "phase_metrics",
    "rle_roi_foreground_ratio",
]
