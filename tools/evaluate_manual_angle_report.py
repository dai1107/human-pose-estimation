from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyrox.geometry import calculate_angle_2d, calculate_angle_3d
from tools.angle_validation import MANUAL_ANGLE_DEFINITIONS


SCHEMA_VERSION = 1
SOURCE_SPECS: Mapping[str, tuple[str, str]] = {
    "raw_lite": ("mediapipe_lite/raw_pose.jsonl.gz", "raw"),
    "raw_full": ("mediapipe_full/raw_pose.jsonl.gz", "raw"),
    "causal_full": ("causal_analysis_pose.jsonl.gz", "filtered"),
    "offline_full": ("offline_annotation_assist.jsonl.gz", "filtered"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate reviewed 2D angles against cached image and world "
            "landmarks for every annotated action."
        )
    )
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/hyrox"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/angle_validation/reviewed_v1"),
    )
    return parser


def load_manual_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("manual angle report must be a JSON object")
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("manual angle report must contain an annotations list")
    return dict(payload)


def evaluate_report(
    report: Mapping[str, Any],
    *,
    dataset_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(dataset_root)
    annotations = [
        dict(item)
        for item in report.get("annotations", ())
        if isinstance(item, Mapping)
    ]
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        record_id = str(annotation.get("record_id", "")).strip()
        if not record_id:
            raise ValueError("every annotation must contain record_id")
        by_record[record_id].append(annotation)

    frame_sources: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    missing_sources: list[dict[str, str]] = []
    for record_id, record_annotations in sorted(by_record.items()):
        wanted = {int(item["frame_index"]) for item in record_annotations}
        cache_root = root / "pose_cache" / record_id
        for source_name, (relative_path, _) in SOURCE_SPECS.items():
            source_path = cache_root / relative_path
            if not source_path.is_file():
                missing_sources.append(
                    {
                        "record_id": record_id,
                        "source": source_name,
                        "path": str(source_path),
                    }
                )
                frame_sources[(record_id, source_name)] = {}
                continue
            frame_sources[(record_id, source_name)] = _read_selected_frames(
                source_path,
                wanted,
            )

    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        manual = _finite(annotation.get("manual_angle_deg"))
        if manual is None:
            continue
        record_id = str(annotation["record_id"])
        frame_index = int(annotation["frame_index"])
        joint = str(annotation.get("joint", "")).strip().lower()
        if joint not in MANUAL_ANGLE_DEFINITIONS:
            continue
        width, height = _frame_size(annotation)
        row = {
            "annotation_id": annotation.get("annotation_id"),
            "record_id": record_id,
            "action": annotation.get("action"),
            "frame_index": frame_index,
            "timestamp_ms": annotation.get("timestamp_ms"),
            "joint": joint,
            "camera_view": annotation.get("camera_view"),
            "event": annotation.get("event"),
            "visibility": annotation.get("visibility"),
            "manual_angle_deg": manual,
        }
        for source_name, (_, source_type) in SOURCE_SPECS.items():
            frame = frame_sources[(record_id, source_name)].get(frame_index)
            image_points, world_points = _landmark_maps(frame, source_type)
            mirror_joint = _opposite_side_joint(joint)
            angle_2d = _joint_angle(
                image_points,
                joint,
                dimensions=(width, height),
                spatial=False,
            )
            uncorrected_2d = _joint_angle(
                image_points,
                joint,
                dimensions=None,
                spatial=False,
            )
            angle_3d = _joint_angle(
                world_points,
                joint,
                dimensions=None,
                spatial=True,
            )
            row[f"{source_name}_2d_deg"] = angle_2d
            row[f"{source_name}_2d_error_deg"] = _absolute_error(
                angle_2d,
                manual,
                joint=joint,
            )
            row[f"{source_name}_uncorrected_2d_deg"] = uncorrected_2d
            row[
                f"{source_name}_uncorrected_2d_error_deg"
            ] = _absolute_error(uncorrected_2d, manual, joint=joint)
            row[f"{source_name}_3d_deg"] = angle_3d
            row[f"{source_name}_3d_projection_gap_deg"] = _absolute_error(
                angle_3d,
                manual,
                joint=joint,
            )
            mirror_2d = _joint_angle(
                image_points,
                mirror_joint,
                dimensions=(width, height),
                spatial=False,
            )
            mirror_3d = _joint_angle(
                world_points,
                mirror_joint,
                dimensions=None,
                spatial=True,
            )
            row[f"{source_name}_mirror_2d_deg"] = mirror_2d
            row[f"{source_name}_mirror_2d_error_deg"] = _absolute_error(
                mirror_2d,
                manual,
                joint=joint,
            )
            row[f"{source_name}_mirror_3d_deg"] = mirror_3d
            row[
                f"{source_name}_mirror_3d_projection_gap_deg"
            ] = _absolute_error(mirror_3d, manual, joint=joint)
        # Round 12 channel contract.  The formal selected-rule angle remains
        # causal filtered 2D; 3D/canonical values stay validation-only.
        channel_sources = {
            "raw_2d": "raw_full_2d",
            "filtered_2d": "causal_full_2d",
            "raw_3d": "raw_full_3d",
            "filtered_3d": "causal_full_3d",
            # A rigid body-coordinate transform preserves a three-point angle.
            "canonical_3d": "causal_full_3d",
            "selected_rule": "causal_full_2d",
        }
        for channel, source in channel_sources.items():
            value_suffix = "_deg"
            error_suffix = (
                "_projection_gap_deg" if source.endswith("_3d") else "_error_deg"
            )
            row[f"round12_{channel}_deg"] = row.get(f"{source}{value_suffix}")
            row[f"round12_{channel}_error_deg"] = row.get(
                f"{source}{error_suffix}"
            )
        row["round12_selected_rule_source"] = (
            "causal_full_image_landmarks_2d"
        )
        rows.append(row)

    summary = _build_summary(rows)
    summary["findings"] = _findings(rows)
    summary["round12_validation"] = _round12_validation(rows)
    summary["unmatched_primary_annotations"] = [
        {
            "annotation_id": row.get("annotation_id"),
            "record_id": row.get("record_id"),
            "frame_index": row.get("frame_index"),
            "joint": row.get("joint"),
        }
        for row in rows
        if _finite(row.get("raw_full_2d_deg")) is None
    ]
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "reviewed_angle_model_evaluation_v1",
            "record_count": len(by_record),
            "annotation_count": len(rows),
            "missing_sources": missing_sources,
            "manual_reference": {
                "coordinate_system": report.get("coordinate_system"),
                "angle_definition": report.get("angle_definition"),
            },
            "three_d_interpretation": (
                "Manual labels are projected 2D pixel angles. A world-3D "
                "difference is therefore a projection-consistency gap, not "
                "a direct 3D ground-truth error."
            ),
        }
    )
    return summary, rows


def write_artifacts(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    matched_report: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path | None]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "reviewed_angle_evaluation.json"
    rows_path = target / "reviewed_angle_rows.csv"
    matched_path = (
        target / "angle_validation_report_v1_model_matched.json"
        if matched_report is not None
        else None
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0]) if rows else []
    with rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if matched_path is not None and matched_report is not None:
        matched_path.write_text(
            json.dumps(
                matched_report,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return summary_path, rows_path, matched_path


def build_model_matched_report(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_annotation = {
        str(row.get("annotation_id")): row
        for row in rows
        if row.get("annotation_id")
    }
    output = dict(report)
    matched_annotations: list[dict[str, Any]] = []
    matched_count = 0
    for source in report.get("annotations", ()):
        if not isinstance(source, Mapping):
            continue
        annotation = dict(source)
        row = by_annotation.get(str(annotation.get("annotation_id", "")))
        if row is None:
            matched_annotations.append(annotation)
            continue
        model_value = _finite(row.get("raw_full_2d_deg"))
        model_error = _finite(row.get("raw_full_2d_error_deg"))
        annotation.update(
            {
                "model_angle_deg": _rounded(model_value),
                "angle_error_deg": _rounded(model_error),
                "comparison_status": (
                    "matched_raw_full_aspect_corrected"
                    if model_value is not None
                    else "model_frame_unavailable"
                ),
                "model_angle_source": (
                    "mediapipe_full_image_landmarks_aspect_corrected"
                ),
                "model_angles": {
                    source_name: {
                        "angle_2d_deg": _rounded(
                            _finite(row.get(f"{source_name}_2d_deg"))
                        ),
                        "error_2d_deg": _rounded(
                            _finite(row.get(f"{source_name}_2d_error_deg"))
                        ),
                        "angle_3d_deg": _rounded(
                            _finite(row.get(f"{source_name}_3d_deg"))
                        ),
                        "projection_gap_3d_deg": _rounded(
                            _finite(
                                row.get(
                                    f"{source_name}_3d_projection_gap_deg"
                                )
                            )
                        ),
                    }
                    for source_name in SOURCE_SPECS
                },
                "three_d_comparison_status": (
                    "projection_consistency_only_no_3d_manual_truth"
                ),
            }
        )
        matched_count += int(model_value is not None)
        matched_annotations.append(annotation)
    output["annotations"] = matched_annotations
    output["model_match"] = {
        "schema_version": SCHEMA_VERSION,
        "primary_source": "raw_full_2d_aspect_corrected",
        "matched_annotation_count": matched_count,
        "annotation_count": len(matched_annotations),
        "source_report_unchanged": True,
        "three_d_reference_scope": "projection_consistency_only",
    }
    return output


def _read_selected_frames(
    path: Path,
    wanted: set[int],
) -> dict[int, Mapping[str, Any]]:
    selected: dict[int, Mapping[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            frame_index = int(item.get("frame_index", -1))
            if frame_index in wanted:
                selected[frame_index] = item
                if len(selected) == len(wanted):
                    break
    return selected


def _landmark_maps(
    frame: Mapping[str, Any] | None,
    source_type: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if frame is None:
        return {}, {}
    if source_type == "raw":
        raw = frame.get("raw_native")
        if not isinstance(raw, Mapping):
            return {}, {}
        image = raw.get("image_landmarks")
        world = raw.get("world_landmarks")
    else:
        image = frame.get("image_normalized_2d")
        world = frame.get("mp_world_body_3d")
    return _point_map(image), _point_map(world)


def _point_map(value: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {
        str(point["name"]): point
        for point in value
        if isinstance(point, Mapping) and point.get("name")
    }


def _joint_angle(
    points: Mapping[str, Mapping[str, Any]],
    joint: str,
    *,
    dimensions: tuple[float, float] | None,
    spatial: bool,
) -> float | None:
    if joint == "torso":
        shoulder = _midpoint(points, "left_shoulder", "right_shoulder")
        hip = _midpoint(points, "left_hip", "right_hip")
        if shoulder is None or hip is None:
            return None
        vertical = np.asarray(hip, dtype=float).copy()
        vertical[1] -= 1.0
        if spatial:
            return calculate_angle_3d(shoulder, hip, vertical)
        width, height = dimensions or (1.0, 1.0)
        return calculate_angle_2d(shoulder, hip, vertical, width, height)

    definition = MANUAL_ANGLE_DEFINITIONS[joint]
    triplet = [points.get(name) for name in definition]
    if any(point is None for point in triplet):
        return None
    if spatial:
        return calculate_angle_3d(*triplet)
    width, height = dimensions or (1.0, 1.0)
    return calculate_angle_2d(*triplet, width, height)


def _midpoint(
    points: Mapping[str, Mapping[str, Any]],
    first_name: str,
    second_name: str,
) -> np.ndarray | None:
    first = points.get(first_name)
    second = points.get(second_name)
    if first is None or second is None:
        return None
    values = []
    for axis in ("x", "y", "z"):
        left = _finite(first.get(axis))
        right = _finite(second.get(axis))
        if left is None or right is None:
            return None
        values.append((left + right) / 2.0)
    return np.asarray(values, dtype=float)


def _frame_size(annotation: Mapping[str, Any]) -> tuple[float, float]:
    value = annotation.get("native_frame_size")
    if not isinstance(value, Mapping):
        return 1.0, 1.0
    width = _finite(value.get("width")) or 1.0
    height = _finite(value.get("height")) or 1.0
    return width, height


def _opposite_side_joint(joint: str) -> str:
    if joint.startswith("left_"):
        candidate = f"right_{joint.removeprefix('left_')}"
        return candidate if candidate in MANUAL_ANGLE_DEFINITIONS else joint
    if joint.startswith("right_"):
        candidate = f"left_{joint.removeprefix('right_')}"
        return candidate if candidate in MANUAL_ANGLE_DEFINITIONS else joint
    return joint


def _absolute_error(
    value: float | None,
    manual: float,
    *,
    joint: str,
) -> float | None:
    if value is None:
        return None
    normalized = abs(value) if joint == "torso" else value
    return abs(float(normalized) - float(manual))


def _build_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    group_fields = ("action", "joint", "camera_view", "visibility", "record_id")
    output: dict[str, Any] = {"overall": _metrics(rows)}
    for field in group_fields:
        labels = sorted({str(row.get(field, "")) for row in rows})
        output[f"by_{field}"] = {
            label: _metrics(
                [row for row in rows if str(row.get(field, "")) == label]
            )
            for label in labels
        }
    return output


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"count": len(rows), "sources": {}}
    for source_name in SOURCE_SPECS:
        source: dict[str, Any] = {}
        for dimension, suffix in (
            ("2d", "2d_error_deg"),
            ("uncorrected_2d", "uncorrected_2d_error_deg"),
            ("3d_projection_gap", "3d_projection_gap_deg"),
            ("mirror_2d", "mirror_2d_error_deg"),
            (
                "mirror_3d_projection_gap",
                "mirror_3d_projection_gap_deg",
            ),
        ):
            values = [
                value
                for row in rows
                if (
                    value := _finite(row.get(f"{source_name}_{suffix}"))
                )
                is not None
            ]
            source[dimension] = _distribution(values)
        output["sources"][source_name] = source
    return output


def _findings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_metrics = _metrics(rows)["sources"]
    aspect_ratio_correction: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for source_name in SOURCE_SPECS:
        corrected = source_metrics[source_name]["2d"]
        uncorrected = source_metrics[source_name]["uncorrected_2d"]
        corrected_mae = _finite(corrected.get("mae_deg"))
        uncorrected_mae = _finite(uncorrected.get("mae_deg"))
        reduction = (
            uncorrected_mae - corrected_mae
            if corrected_mae is not None and uncorrected_mae is not None
            else None
        )
        aspect_ratio_correction[source_name] = {
            "uncorrected_mae_deg": uncorrected_mae,
            "corrected_mae_deg": corrected_mae,
            "absolute_mae_reduction_deg": _rounded(reduction),
            "relative_mae_reduction": _rounded(
                reduction / uncorrected_mae
                if reduction is not None and uncorrected_mae
                else None
            ),
        }
        if corrected_mae is not None:
            ranking.append(
                {"source": source_name, "mae_deg": corrected_mae}
            )
    ranking.sort(key=lambda item: (float(item["mae_deg"]), str(item["source"])))
    return {
        "aspect_ratio_correction": aspect_ratio_correction,
        "two_d_source_ranking": ranking,
        "recommended_primary_validation_source": "raw_full",
        "recommended_offline_review_source": (
            ranking[0]["source"] if ranking else None
        ),
        "three_d_policy": (
            "Keep world-3D as shadow/confidence-only evidence until spatial "
            "3D ground truth is collected; do not fit 3D angles to these 2D labels."
        ),
    }


def _round12_validation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    channels = (
        "raw_2d",
        "filtered_2d",
        "raw_3d",
        "filtered_3d",
        "canonical_3d",
        "selected_rule",
    )
    channel_metrics = {
        channel: _distribution(
            [
                value
                for row in rows
                if (
                    value := _finite(
                        row.get(f"round12_{channel}_error_deg")
                    )
                )
                is not None
            ]
        )
        for channel in channels
    }
    required_actions = (
        "lunge",
        "wall_ball",
        "burpee_broad_jump",
        "rowing",
    )
    required_views = ("side", "oblique_30", "oblique_45", "front")
    observed_actions = {
        str(row.get("action", "")).strip().lower() for row in rows
    }
    observed_views = {
        _round12_view(str(row.get("camera_view", ""))) for row in rows
    }
    action_metrics = {
        action: {
            channel: _distribution(
                [
                    value
                    for row in rows
                    if str(row.get("action", "")).strip().lower() == action
                    and (
                        value := _finite(
                            row.get(f"round12_{channel}_error_deg")
                        )
                    )
                    is not None
                ]
            )
            for channel in channels
        }
        for action in required_actions
    }
    baseline = _distribution(
        [
            value
            for row in rows
            if (value := _finite(row.get("raw_lite_2d_error_deg")))
            is not None
        ]
    )
    candidate = channel_metrics["selected_rule"]
    comparisons: dict[str, Any] = {}
    checks: list[bool] = []
    for field in (
        "mae_deg",
        "median_absolute_error_deg",
        "p90_absolute_error_deg",
        "p95_absolute_error_deg",
    ):
        old = _finite(baseline.get(field))
        new = _finite(candidate.get(field))
        passed = new <= old if old is not None and new is not None else None
        comparisons[field] = {
            "old_version": old,
            "new_version": new,
            "delta": _rounded(
                new - old if old is not None and new is not None else None
            ),
            "non_regression": passed,
        }
        if passed is not None:
            checks.append(passed)
    return {
        "channel_contract": {
            "raw_2d": "raw_full image landmarks",
            "filtered_2d": "causal_full filtered image landmarks",
            "raw_3d": "raw_full world landmarks",
            "filtered_3d": "causal_full filtered world landmarks",
            "canonical_3d": (
                "causal_full world angle after rigid body-coordinate transform; "
                "equal for three-point joint angles by rotational invariance"
            ),
            "selected_rule": (
                "causal_full filtered 2D; formal HYROX thresholds unchanged"
            ),
        },
        "overall": channel_metrics,
        "by_required_action": action_metrics,
        "coverage": {
            "required_actions": list(required_actions),
            "required_camera_views": list(required_views),
            "observed_actions": sorted(observed_actions),
            "observed_camera_views": sorted(observed_views),
            "missing_actions": [
                item for item in required_actions if item not in observed_actions
            ],
            "missing_camera_views": [
                item for item in required_views if item not in observed_views
            ],
        },
        "old_vs_new": {
            "old_version": "raw_lite unfiltered 2D baseline",
            "new_version": "causal_full filtered 2D selected rule angle",
            "metrics": comparisons,
            "non_regression_pass": all(checks) if checks else None,
        },
        "event_timing_error": {
            "available": False,
            "reason": (
                "reviewed cache contains manual event labels but no matched "
                "old/new program event frames; use compare_manual_angles.py "
                "with --baseline-report for lowest-point/full-extension timing"
            ),
        },
        "manual_reference_limitation": (
            "3D values are projection-consistency gaps against manual 2D labels, "
            "not spatial 3D ground-truth MAE."
        ),
    }


def _round12_view(value: str) -> str:
    normalized = value.strip().lower().replace("°", "").replace("deg", "")
    return {
        "side_view": "side",
        "30": "oblique_30",
        "30_oblique": "oblique_30",
        "oblique30": "oblique_30",
        "45": "oblique_45",
        "45_oblique": "oblique_45",
        "oblique45": "oblique_45",
        "front_view": "front",
    }.get(normalized, normalized)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mae_deg": None,
            "median_absolute_error_deg": None,
            "p90_absolute_error_deg": None,
            "p95_absolute_error_deg": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mae_deg": round(float(np.mean(array)), 4),
        "median_absolute_error_deg": round(float(np.median(array)), 4),
        "p90_absolute_error_deg": round(float(np.percentile(array, 90)), 4),
        "p95_absolute_error_deg": round(float(np.percentile(array, 95)), 4),
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = load_manual_report(args.report)
    summary, rows = evaluate_report(report, dataset_root=args.dataset_root)
    matched_report = build_model_matched_report(report, rows)
    summary_path, rows_path, matched_path = write_artifacts(
        args.output_dir,
        summary,
        rows,
        matched_report=matched_report,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "rows": str(rows_path),
                "matched_report": str(matched_path),
                "annotation_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
