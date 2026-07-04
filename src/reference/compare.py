from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.time_utils import make_session_id, now_iso

from .canonicalize import canonicalize_feature_matrix
from .clipper import clip_session
from .dtw import DtwResult, constrained_dtw
from .features import extract_feature_matrix, load_feature_config
from .library import load_reference
from .quality import evaluate_quality, load_quality_rules
from .report import write_comparison_plots, write_markdown_report
from .resample import resample_sequence
from .session_loader import read_csv_rows, write_csv_rows


def _top_features(result: DtwResult, limit: int = 5) -> list[dict[str, Any]]:
    rows = [
        {"feature": name, "mean_absolute_error": value}
        for name, value in result.per_feature_error.items()
        if isfinite(float(value))
    ]
    return sorted(rows, key=lambda item: item["mean_absolute_error"], reverse=True)[:limit]


def _feature_error_rows(result: DtwResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (name, mean_error) in enumerate(result.per_feature_error.items()):
        ref_values = result.aligned_reference[:, index]
        cand_values = result.aligned_candidate[:, index]
        finite_mask = np.isfinite(ref_values) & np.isfinite(cand_values)
        max_error = float(np.max(np.abs(ref_values[finite_mask] - cand_values[finite_mask]))) if finite_mask.any() else float("nan")
        rows.append(
            {
                "feature": name,
                "mean_absolute_error": f"{mean_error:.8g}" if isfinite(float(mean_error)) else "",
                "max_absolute_error": f"{max_error:.8g}" if isfinite(max_error) else "",
                "used": int(finite_mask.any()),
            }
        )
    return rows


def _top_time_ranges(result: DtwResult, bins: int = 10, limit: int = 3) -> list[dict[str, float]]:
    if result.aligned_reference.size == 0:
        return []
    local_error = np.nanmean(np.abs(result.aligned_reference - result.aligned_candidate), axis=1)
    if not np.isfinite(local_error).any():
        return []
    ranges: list[dict[str, float]] = []
    edges = np.linspace(0, len(local_error), bins + 1, dtype=int)
    for start, end in zip(edges, edges[1:]):
        if end <= start:
            continue
        values = local_error[start:end]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        ranges.append(
            {
                "start_percent": float(start / len(local_error) * 100.0),
                "end_percent": float(end / len(local_error) * 100.0),
                "mean_error": float(np.mean(finite)),
            }
        )
    return sorted(ranges, key=lambda item: item["mean_error"], reverse=True)[:limit]


def _write_aligned_features(path: Path, result: DtwResult, feature_names: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    total = max(len(result.path) - 1, 1)
    for path_index, ((ref_index, cand_index), ref_values, cand_values) in enumerate(
        zip(result.path, result.aligned_reference, result.aligned_candidate)
    ):
        row: dict[str, Any] = {
            "path_index": path_index,
            "reference_index": ref_index,
            "candidate_index": cand_index,
            "aligned_percent": f"{path_index / total * 100.0:.8g}",
        }
        for index, feature_name in enumerate(feature_names):
            row[f"reference_{feature_name}"] = f"{ref_values[index]:.8g}" if isfinite(float(ref_values[index])) else ""
            row[f"candidate_{feature_name}"] = f"{cand_values[index]:.8g}" if isfinite(float(cand_values[index])) else ""
            error = abs(ref_values[index] - cand_values[index])
            row[f"error_{feature_name}"] = f"{error:.8g}" if isfinite(float(error)) else ""
        rows.append(row)
    write_csv_rows(path, rows)


def _write_dtw_path(path: Path, result: DtwResult) -> None:
    rows = [
        {"path_index": index, "reference_index": ref_index, "candidate_index": candidate_index}
        for index, (ref_index, candidate_index) in enumerate(result.path)
    ]
    write_csv_rows(path, rows)


def compare_reference_to_session(
    session_dir: str | Path,
    reference_dir: str | Path,
    output_dir: str | Path = "outputs/comparisons",
    start_ms: int | None = None,
    end_ms: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    target_length: int = 100,
    window_ratio: float = 0.15,
    canonical_side: str | None = None,
    candidate_movement_side: str = "unknown",
    plot: bool = True,
) -> Path:
    reference_path = Path(reference_dir)
    reference = load_reference(reference_path)
    reference_rows = read_csv_rows(reference_path / "clip_kinematics.csv")
    if not reference_rows:
        template_rows = read_csv_rows(reference_path / "template_features.csv")
        if not template_rows:
            raise FileNotFoundError("reference does not contain clip_kinematics.csv or template_features.csv")
        raise NotImplementedError("template reference comparison is not yet supported by the CLI")

    candidate_clip = clip_session(session_dir, start_ms=start_ms, end_ms=end_ms, start_frame=start_frame, end_frame=end_frame)
    quality = evaluate_quality(
        candidate_clip.kinematics,
        candidate_clip.landmarks,
        metadata=candidate_clip.session.metadata,
        rules=load_quality_rules(),
    )
    feature_config = load_feature_config()
    ref_features = extract_feature_matrix(reference_rows, feature_config)
    cand_features = extract_feature_matrix(candidate_clip.kinematics, feature_config)

    ref_matrix, ref_mirrored = canonicalize_feature_matrix(
        ref_features.matrix,
        ref_features.feature_names,
        reference.movement_side,
        canonical_side if reference.mirror_canonicalization_enabled or canonical_side else None,
    )
    cand_matrix, cand_mirrored = canonicalize_feature_matrix(
        cand_features.matrix,
        cand_features.feature_names,
        candidate_movement_side,
        canonical_side,
    )

    ref_resampled, _ = resample_sequence(ref_matrix, ref_features.timestamps, target_length=target_length)
    cand_resampled, _ = resample_sequence(cand_matrix, cand_features.timestamps, target_length=target_length)
    result = constrained_dtw(
        ref_resampled,
        cand_resampled,
        window_ratio=window_ratio,
        feature_names=ref_features.feature_names,
    )

    comparison_id = make_session_id()
    root = Path(output_dir)
    output_path = root / comparison_id
    suffix = 1
    while output_path.exists():
        output_path = root / f"{comparison_id}_{suffix}"
        suffix += 1
    output_path.mkdir(parents=True, exist_ok=False)

    top_features = _top_features(result)
    top_ranges = _top_time_ranges(result)
    summary: dict[str, Any] = {
        "reference_id": reference.reference_id,
        "candidate_session_id": candidate_clip.clip_range.session_id,
        "candidate_clip_range": candidate_clip.clip_range.to_dict(),
        "alignment_method": "constrained_dtw",
        "feature_set": feature_config.name,
        "features_used": ref_features.processing["features_used"],
        "global_dtw_distance": result.total_distance,
        "normalized_dtw_distance": result.normalized_distance,
        "top_difference_features": top_features,
        "top_difference_time_ranges": top_ranges,
        "data_quality_summary": quality.to_dict(),
        "mirror_canonicalization_applied": bool(ref_mirrored or cand_mirrored),
    }
    metadata = {
        "comparison_id": output_path.name,
        "created_at": now_iso(),
        "reference_dir": str(reference_path),
        "candidate_session_dir": str(Path(session_dir)),
        "target_length": target_length,
        "window_ratio": window_ratio,
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_aligned_features(output_path / "aligned_features.csv", result, ref_features.feature_names)
    write_csv_rows(output_path / "feature_errors.csv", _feature_error_rows(result))
    _write_dtw_path(output_path / "dtw_path.csv", result)
    if plot:
        write_comparison_plots(output_path, result.aligned_reference, result.aligned_candidate, ref_features.feature_names)
    write_markdown_report(output_path / "report.md", summary)
    return output_path

