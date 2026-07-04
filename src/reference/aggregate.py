from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .features import ExtractedFeatures, FeatureConfig, extract_feature_matrix, load_feature_config
from .resample import resample_sequence
from .session_loader import write_csv_rows


def _coerce_clip(clip: Any, feature_config: FeatureConfig) -> ExtractedFeatures:
    if isinstance(clip, ExtractedFeatures):
        return clip
    if isinstance(clip, tuple) and len(clip) >= 3:
        matrix = np.asarray(clip[0], dtype=float)
        timestamps = np.asarray(clip[1], dtype=float)
        feature_names = list(clip[2])
        return ExtractedFeatures(matrix, timestamps, feature_names, np.isfinite(matrix), {"strategy": "provided"})
    if isinstance(clip, list):
        return extract_feature_matrix(clip, feature_config)
    raise TypeError("unsupported reference clip type")


def build_reference_template(
    reference_clips: list[Any],
    feature_config: FeatureConfig | None = None,
    target_length: int = 100,
) -> dict[str, Any]:
    config = feature_config or load_feature_config()
    feature_names = config.feature_names
    resampled: list[np.ndarray] = []
    for clip in reference_clips:
        extracted = _coerce_clip(clip, config)
        matrix = extracted.matrix
        if extracted.feature_names != feature_names:
            name_to_index = {name: index for index, name in enumerate(extracted.feature_names)}
            aligned = np.zeros((matrix.shape[0], len(feature_names)), dtype=float)
            for index, name in enumerate(feature_names):
                source_index = name_to_index.get(name)
                if source_index is not None:
                    aligned[:, index] = matrix[:, source_index]
            matrix = aligned
        sequence, axis = resample_sequence(matrix, extracted.timestamps, target_length=target_length)
        resampled.append(sequence)
    if not resampled:
        raise ValueError("at least one reference clip is required")
    stack = np.stack(resampled, axis=0)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    lower = mean - std
    upper = mean + std

    peak_events: dict[str, dict[str, float | None]] = {}
    for feature_index, feature_name in enumerate(feature_names):
        peak_times: list[float] = []
        for sequence in resampled:
            values = sequence[:, feature_index]
            if np.isfinite(values).any():
                peak_times.append(float(axis[int(np.nanargmax(values))]))
        finite = [value for value in peak_times if isfinite(value)]
        peak_events[feature_name] = {
            "mean_percent": float(np.mean(finite)) if finite else None,
            "std_percent": float(np.std(finite)) if len(finite) > 1 else (0.0 if finite else None),
        }

    return {
        "feature_names": feature_names,
        "normalized_time_percent": axis.tolist(),
        "mean_trajectory": mean.tolist(),
        "std_trajectory": std.tolist(),
        "confidence_band": {"lower": lower.tolist(), "upper": upper.tolist()},
        "peak_events": peak_events,
        "clip_count": len(resampled),
        "template_stability_status": "limited" if len(resampled) < 3 else "ok",
    }


def write_template_csv(reference_dir: str | Path, template: dict[str, Any]) -> None:
    path = Path(reference_dir)
    feature_names = list(template["feature_names"])
    axis = list(template["normalized_time_percent"])
    mean = np.asarray(template["mean_trajectory"], dtype=float)
    std = np.asarray(template["std_trajectory"], dtype=float)
    rows: list[dict[str, Any]] = []
    for index, percent in enumerate(axis):
        row: dict[str, Any] = {"normalized_time_percent": f"{float(percent):.8g}"}
        for feature_index, feature_name in enumerate(feature_names):
            row[f"{feature_name}_mean"] = f"{mean[index, feature_index]:.8g}"
            row[f"{feature_name}_std"] = f"{std[index, feature_index]:.8g}"
        rows.append(row)
    write_csv_rows(path / "template_features.csv", rows)

