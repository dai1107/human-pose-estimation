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

from .phase_metrics import rep_kinematics_rows
from .schema import SquatRep


SQUAT_COMPARISON_FEATURES = {
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "trunk_tilt_proxy",
    "left_elbow_angle",
    "right_elbow_angle",
}


def compare_reps_to_reference(
    reps: list[SquatRep],
    reference_dir: str | Path,
    output_dir: str | Path,
    target_length: int = 100,
) -> dict[str, Any]:
    reference_path = Path(reference_dir)
    reference = load_reference(reference_path)
    reference_rows = read_csv_rows(reference_path / "clip_kinematics.csv")
    if not reference_rows:
        return {"status": "WARNING", "message": "reference clip_kinematics.csv not found", "rep_comparisons": []}

    feature_config = load_feature_config()
    reference_features = extract_feature_matrix(reference_rows, feature_config)
    selected_indices = [
        index
        for index, name in enumerate(reference_features.feature_names)
        if name in SQUAT_COMPARISON_FEATURES
    ]
    if not selected_indices:
        return {"status": "WARNING", "message": "no squat comparison features available", "rep_comparisons": []}

    selected_names = [reference_features.feature_names[index] for index in selected_indices]
    reference_matrix = reference_features.matrix[:, selected_indices]
    reference_resampled, _ = resample_sequence(reference_matrix, reference_features.timestamps, target_length=target_length)

    rows: list[dict[str, Any]] = []
    rep_summaries: list[dict[str, Any]] = []
    for rep in reps:
        rep_rows = rep_kinematics_rows(rep)
        if not rep_rows:
            continue
        candidate_features = extract_feature_matrix(rep_rows, feature_config)
        name_to_index = {name: index for index, name in enumerate(candidate_features.feature_names)}
        candidate_indices = [name_to_index[name] for name in selected_names if name in name_to_index]
        if len(candidate_indices) != len(selected_names):
            continue
        candidate_matrix = candidate_features.matrix[:, candidate_indices]
        candidate_resampled, _ = resample_sequence(candidate_matrix, candidate_features.timestamps, target_length=target_length)
        result = constrained_dtw(reference_resampled, candidate_resampled, window_ratio=0.2, feature_names=selected_names)
        top_features = sorted(
            [
                {"feature": name, "mean_absolute_error": value}
                for name, value in result.per_feature_error.items()
                if isfinite(float(value))
            ],
            key=lambda item: item["mean_absolute_error"],
            reverse=True,
        )[:3]
        rep_summary = {
            "rep_index": rep.rep_index,
            "normalized_dtw_distance": result.normalized_distance,
            "top_difference_features": top_features,
        }
        rep_summaries.append(rep_summary)
        rows.append(
            {
                "rep_index": rep.rep_index,
                "normalized_dtw_distance": f"{result.normalized_distance:.8g}",
                "top_difference_feature": top_features[0]["feature"] if top_features else "",
                "top_difference_error": f"{top_features[0]['mean_absolute_error']:.8g}" if top_features else "",
            }
        )

    output_path = Path(output_dir)
    write_csv_rows(output_path / "squat_reference_comparison.csv", rows)
    _plot_reference_comparison(output_path / "squat_reference_alignment.png", rep_summaries)
    return {
        "status": "PASS" if rep_summaries else "WARNING",
        "reference_id": reference.reference_id,
        "feature_names": selected_names,
        "rep_comparisons": rep_summaries,
        "message": "relative kinematic comparison only",
    }


def _plot_reference_comparison(path: Path, rep_summaries: list[dict[str, Any]]) -> None:
    if not rep_summaries:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [int(item["rep_index"]) for item in rep_summaries]
    distances = [float(item["normalized_dtw_distance"]) for item in rep_summaries]
    plt.figure(figsize=(8, 4))
    plt.plot(indices, distances, marker="o", linewidth=2.0)
    plt.title("Squat reps vs personal reference")
    plt.xlabel("rep index")
    plt.ylabel("normalized DTW distance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

