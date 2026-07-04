from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from src.reference.dtw import constrained_dtw
from src.reference.features import extract_feature_matrix, load_feature_config
from src.reference.library import load_reference
from src.reference.resample import resample_sequence
from src.reference.session_loader import read_csv_rows, write_csv_rows


SHOT_REFERENCE_FEATURES = (
    "shooting_knee_angle",
    "shooting_hip_angle",
    "pelvis_speed",
    "trunk_tilt_proxy",
    "shooting_shoulder_angle",
    "shooting_elbow_angle",
    "shooting_elbow_angular_velocity",
    "shooting_wrist_speed",
)


def compare_shot_to_reference(
    feature_rows: list[dict[str, Any]],
    reference_dir: str | Path,
    output_dir: str | Path,
    target_length: int = 100,
) -> dict[str, Any]:
    ref_path = Path(reference_dir)
    reference = load_reference(ref_path)
    ref_rows = read_csv_rows(ref_path / "shot_features.csv")
    if not ref_rows:
        ref_rows = read_csv_rows(ref_path / "clip_kinematics.csv")
    if not ref_rows:
        return {"status": "WARNING", "message": "reference features unavailable"}

    feature_names = [name for name in SHOT_REFERENCE_FEATURES if any(name in row for row in feature_rows) and any(name in row for row in ref_rows)]
    if not feature_names:
        return {"status": "WARNING", "message": "no overlapping shot features"}
    ref_matrix, ref_ts = _matrix(ref_rows, feature_names)
    cand_matrix, cand_ts = _matrix(feature_rows, feature_names)
    ref_resampled, _ = resample_sequence(ref_matrix, ref_ts, target_length=target_length)
    cand_resampled, _ = resample_sequence(cand_matrix, cand_ts, target_length=target_length)
    result = constrained_dtw(ref_resampled, cand_resampled, window_ratio=0.2, feature_names=feature_names)
    top = sorted(
        [
            {"feature": name, "mean_absolute_error": value}
            for name, value in result.per_feature_error.items()
            if isfinite(float(value))
        ],
        key=lambda item: item["mean_absolute_error"],
        reverse=True,
    )[:5]
    _plot_alignment(Path(output_dir) / "reference_alignment.png", result.aligned_reference, result.aligned_candidate, feature_names)
    write_csv_rows(
        Path(output_dir) / "reference_feature_errors.csv",
        [{"feature": item["feature"], "mean_absolute_error": f"{item['mean_absolute_error']:.8g}"} for item in top],
    )
    return {
        "status": "PASS",
        "reference_id": reference.reference_id,
        "global_dtw_distance": result.total_distance,
        "normalized_dtw_distance": result.normalized_distance,
        "top_difference_features": top,
        "peak_timing_offsets": {},
        "phase_specific_differences": [],
        "data_quality_summary": "relative kinematic comparison only",
    }


def _matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.array([_num(row.get("timestamp_ms")) for row in rows], dtype=float)
    values = np.array([[_num(row.get(name)) for name in feature_names] for row in rows], dtype=float)
    for col in range(values.shape[1]):
        finite = np.isfinite(values[:, col])
        if not finite.any():
            values[:, col] = 0.0
        elif not finite.all():
            values[:, col] = np.interp(np.arange(values.shape[0]), np.flatnonzero(finite), values[finite, col])
    return values, timestamps


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _plot_alignment(path: Path, reference: np.ndarray, candidate: np.ndarray, feature_names: list[str]) -> None:
    if reference.size == 0 or candidate.size == 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = np.linspace(0, 100, reference.shape[0])
    plt.figure(figsize=(10, 5))
    for index, name in enumerate(feature_names[:4]):
        plt.plot(axis, reference[:, index], label=f"ref {name}")
        plt.plot(axis, candidate[:, index], linestyle="--", label=f"shot {name}")
    plt.title("Shot vs personal reference alignment")
    plt.xlabel("aligned time (%)")
    plt.ylabel("proxy value")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

