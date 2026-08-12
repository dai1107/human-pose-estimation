from __future__ import annotations

import csv
import json
import math
import os
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

from hyrox.geometry import calculate_angle_2d
from src.biomechanics.kinematics_3d import ANGLE_DEFINITIONS_3D


ANGLE_VALIDATION_SCHEMA_VERSION = 2
REQUIRED_ROUND12_ACTIONS = (
    "lunge",
    "wall_ball",
    "burpee_broad_jump",
    "rowing",
)
REQUIRED_ROUND12_VIEWS = (
    "side",
    "oblique_30",
    "oblique_45",
    "front",
)
MANUAL_ANGLE_DEFINITIONS = {
    **{
        key.removesuffix("_angle"): value
        for key, value in ANGLE_DEFINITIONS_3D.items()
    },
    "torso": (
        "shoulder_center",
        "hip_center",
        "vertical_reference",
    ),
}
ANGLE_FIELDS = (
    "angle_2d_raw_deg",
    "angle_2d_smoothed_deg",
    "angle_3d_raw_deg",
    "angle_3d_smoothed_deg",
    "angle_canonical_3d_deg",
    "rule_angle_deg",
)
MODEL_FIELD_BY_OBSERVATION = {
    "model_2d_raw_deg": "angle_2d_raw_deg",
    "model_2d_smoothed_deg": "angle_2d_smoothed_deg",
    "model_3d_raw_deg": "angle_3d_raw_deg",
    "model_3d_smoothed_deg": "angle_3d_smoothed_deg",
    "model_canonical_3d_deg": "angle_canonical_3d_deg",
    "model_rule_angle_deg": "rule_angle_deg",
}
ERROR_FIELD_BY_MODEL = {
    "model_2d_raw_deg": "error_2d_raw_deg",
    "model_2d_smoothed_deg": "error_2d_smoothed_deg",
    "model_3d_raw_deg": "error_3d_raw_deg",
    "model_3d_smoothed_deg": "error_3d_smoothed_deg",
    "model_canonical_3d_deg": "error_canonical_3d_deg",
    "model_rule_angle_deg": "error_rule_deg",
}


def normalize_joint_name(value: str) -> str:
    normalized = str(value).strip().lower().removesuffix("_angle")
    if normalized not in MANUAL_ANGLE_DEFINITIONS:
        raise ValueError(f"unsupported joint: {value}")
    return normalized


def joint_point_names(joint: str) -> tuple[str, str, str]:
    normalized = normalize_joint_name(joint)
    return MANUAL_ANGLE_DEFINITIONS[normalized]


def load_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("frames"),
        list,
    ):
        raise ValueError("angle report must contain a frames list")
    return dict(payload)


def iter_angle_observations(
    report: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    for frame in report.get("frames", ()):
        if not isinstance(frame, Mapping):
            continue
        for observation in frame.get("angle_observations", ()):
            if not isinstance(observation, Mapping):
                continue
            item = dict(observation)
            item.setdefault("frame_index", frame.get("frame_index"))
            item.setdefault("timestamp_ms", frame.get("timestamp_ms"))
            item.setdefault("action", frame.get("action", report.get("action", "")))
            item.setdefault(
                "camera_view",
                frame.get("camera_view", report.get("camera_view", "")),
            )
            try:
                item["joint"] = observation_joint_name(item)
            except ValueError:
                continue
            yield item


def observation_joint_name(observation: Mapping[str, Any]) -> str:
    side = str(observation.get("side", "")).strip().lower()
    joint = str(observation.get("joint_name", "")).strip().lower()
    combined = f"{side}_{joint}" if side in {"left", "right"} else joint
    return normalize_joint_name(combined)


def find_observation(
    report: Mapping[str, Any] | None,
    *,
    frame_index: int,
    joint: str,
) -> dict[str, Any] | None:
    if report is None:
        return None
    normalized = normalize_joint_name(joint)
    for observation in iter_angle_observations(report):
        if (
            int(observation.get("frame_index", -1)) == int(frame_index)
            and observation["joint"] == normalized
        ):
            return observation
    return None


def video_metadata(path: str | Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps if fps > 0.0 else 30.0,
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()


def read_video_frame(path: str | Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"unable to read frame {frame_index}")
        return frame
    finally:
        capture.release()


def build_manual_annotation(
    *,
    video_path: str | Path,
    frame_index: int,
    joint: str,
    camera_view: str,
    points: Sequence[Sequence[float]],
    report: Mapping[str, Any] | None = None,
    event: str = "",
    annotator: str = "",
) -> dict[str, Any]:
    normalized_joint = normalize_joint_name(joint)
    if len(points) != 3:
        raise ValueError("manual angle annotation requires exactly three points")
    resolved_points: list[tuple[float, float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("each manual point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("manual points must be finite")
        resolved_points.append((x, y))
    manual_angle = calculate_angle_2d(*resolved_points)
    if manual_angle is None:
        raise ValueError("manual points do not form a valid angle")
    metadata = video_metadata(video_path)
    observation = find_observation(
        report,
        frame_index=frame_index,
        joint=normalized_joint,
    )
    timestamp_ms = (
        _finite(observation.get("timestamp_ms"))
        if observation is not None
        else None
    )
    if timestamp_ms is None:
        timestamp_ms = int(frame_index) * 1000.0 / metadata["fps"]
    point_names = joint_point_names(normalized_joint)
    payload: dict[str, Any] = {
        "schema_version": ANGLE_VALIDATION_SCHEMA_VERSION,
        "video_id": Path(video_path).stem,
        "video_path": str(Path(video_path)),
        "frame_index": int(frame_index),
        "timestamp_ms": round(timestamp_ms, 3),
        "joint": normalized_joint,
        "camera_view": str(camera_view),
        "manual_points": {
            _short_point_name(name): [
                round(resolved_points[index][0], 3),
                round(resolved_points[index][1], 3),
            ]
            for index, name in enumerate(point_names)
        },
        "point_order": list(point_names),
        "manual_angle_deg": round(manual_angle, 4),
        "landmark_visibility": (
            _finite(observation.get("landmark_visibility"))
            if observation is not None
            else None
        ),
        "event": str(event),
        "annotator": str(annotator),
        "image_width": metadata["width"],
        "image_height": metadata["height"],
        "source_fps": metadata["fps"],
    }
    for model_field, observation_field in MODEL_FIELD_BY_OBSERVATION.items():
        payload[model_field] = (
            _finite(observation.get(observation_field))
            if observation is not None
            else None
        )
    return payload


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping) and isinstance(
            payload.get("annotations"),
            list,
        ):
            records = payload["annotations"]
        elif isinstance(payload, Mapping):
            records = [payload]
        else:
            raise ValueError("unsupported manual annotation JSON structure")
    if not all(isinstance(item, Mapping) for item in records):
        raise ValueError("manual annotations must be JSON objects")
    return [dict(item) for item in records]


def append_annotation(
    path: str | Path,
    annotation: Mapping[str, Any],
) -> Path:
    target = Path(path)
    existing = load_annotations(target) if target.is_file() else []
    replacement = [
        item
        for item in existing
        if not (
            item.get("video_id") == annotation.get("video_id")
            and item.get("frame_index") == annotation.get("frame_index")
            and item.get("joint") == annotation.get("joint")
        )
    ]
    replacement.append(dict(annotation))
    replacement.sort(
        key=lambda item: (
            str(item.get("video_id", "")),
            int(item.get("frame_index", -1)),
            str(item.get("joint", "")),
        )
    )
    payload = {
        "schema_version": ANGLE_VALIDATION_SCHEMA_VERSION,
        "annotations": replacement,
    }
    _atomic_json(target, payload)
    return target


def export_angle_curves(
    report: Mapping[str, Any],
    output_path: str | Path,
    *,
    joint: str | None = None,
) -> Path:
    normalized = normalize_joint_name(joint) if joint else None
    rows = [
        observation
        for observation in iter_angle_observations(report)
        if normalized is None or observation["joint"] == normalized
    ]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "frame_index",
        "timestamp_ms",
        "joint",
        *ANGLE_FIELDS,
        "display_angle_deg",
        "drawn_landmarks_angle_deg",
        "landmark_visibility",
        "geometry_valid",
    )
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return target


def compare_manual_annotations(
    annotations: Sequence[Mapping[str, Any]],
    *,
    report: Mapping[str, Any] | None = None,
    baseline_report: Mapping[str, Any] | None = None,
    max_lag_frames: int = 15,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for source in annotations:
        row = dict(source)
        joint = normalize_joint_name(str(row.get("joint", "")))
        frame_index = int(row.get("frame_index", -1))
        observation = find_observation(
            report,
            frame_index=frame_index,
            joint=joint,
        )
        manual = _finite(row.get("manual_angle_deg"))
        if manual is None:
            raise ValueError(
                f"manual_angle_deg is missing for frame {frame_index}"
            )
        row["joint"] = joint
        if observation is not None:
            row.setdefault("action", observation.get("action", ""))
            row.setdefault("camera_view", observation.get("camera_view", ""))
        if observation is not None:
            row.setdefault(
                "landmark_visibility",
                observation.get("landmark_visibility"),
            )
            for model_field, observation_field in (
                MODEL_FIELD_BY_OBSERVATION.items()
            ):
                if _finite(row.get(model_field)) is None:
                    row[model_field] = _finite(
                        observation.get(observation_field)
                    )
        for model_field, error_field in ERROR_FIELD_BY_MODEL.items():
            model_value = _finite(row.get(model_field))
            if joint == "torso" and model_value is not None:
                model_value = abs(model_value)
            row[error_field] = (
                abs(model_value - manual)
                if model_value is not None
                else None
            )
        rows.append(row)

    overall = _error_statistics(rows)
    by_joint = {
        joint: _error_statistics(
            [row for row in rows if row["joint"] == joint]
        )
        for joint in sorted({str(row["joint"]) for row in rows})
    }
    by_action = _grouped_statistics(rows, "action")
    by_camera_view = _grouped_statistics(rows, "camera_view")
    high_visibility_side = [
        row
        for row in rows
        if str(row.get("camera_view", "")).lower() == "side"
        and (_finite(row.get("landmark_visibility")) or 0.0) >= 0.75
    ]
    curve_latency = (
        analyze_curve_latency(report, max_lag_frames=max_lag_frames)
        if report is not None
        else {}
    )
    event_offsets = _event_offsets(rows, curve_latency)
    summary = {
        "schema_version": ANGLE_VALIDATION_SCHEMA_VERSION,
        "annotation_count": len(rows),
        "overall": overall,
        "by_joint": by_joint,
        "by_action": by_action,
        "by_camera_view": by_camera_view,
        "side_high_visibility": _error_statistics(high_visibility_side),
        "curve_latency": curve_latency,
        "event_offsets": event_offsets,
        "round12_coverage": _round12_coverage(rows),
        "targets": {
            "manual_side_high_visibility_median_error_deg": 5.0,
            "manual_side_high_visibility_p90_error_deg": 10.0,
            "smoothed_angle_event_lag_frames": 2,
        },
        "interpretation": (
            "2D values are projected image angles. Front-view values must not "
            "be interpreted as true 3D joint angles."
        ),
    }
    if baseline_report is not None:
        baseline_annotations = [
            {
                key: value
                for key, value in annotation.items()
                if key not in MODEL_FIELD_BY_OBSERVATION
                and key not in ERROR_FIELD_BY_MODEL.values()
            }
            for annotation in annotations
        ]
        baseline_summary, baseline_rows = compare_manual_annotations(
            baseline_annotations,
            report=baseline_report,
            max_lag_frames=max_lag_frames,
        )
        _attach_baseline_rows(rows, baseline_rows)
        summary["version_comparison"] = _compare_versions(
            baseline_summary,
            summary,
        )
    return summary, rows


def analyze_curve_latency(
    report: Mapping[str, Any],
    *,
    max_lag_frames: int = 15,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in iter_angle_observations(report):
        grouped[observation["joint"]].append(observation)
    source_fps = _report_fps(report)
    output: dict[str, Any] = {}
    for joint, observations in sorted(grouped.items()):
        observations.sort(key=lambda item: int(item["frame_index"]))
        raw = [_finite(item.get("angle_2d_raw_deg")) for item in observations]
        smooth = [
            _finite(item.get("angle_2d_smoothed_deg"))
            for item in observations
        ]
        frames = [int(item["frame_index"]) for item in observations]
        lag, lag_mae, pair_count = estimate_curve_lag(
            raw,
            smooth,
            max_lag_frames=max_lag_frames,
        )
        raw_min = _extreme_frame(frames, raw, minimum=True)
        smooth_min = _extreme_frame(frames, smooth, minimum=True)
        raw_max = _extreme_frame(frames, raw, minimum=False)
        smooth_max = _extreme_frame(frames, smooth, minimum=False)
        output[joint] = {
            "raw_smoothed_lag_frames": lag,
            "raw_smoothed_lag_ms": (
                round(lag * 1000.0 / source_fps, 3)
                if lag is not None and source_fps > 0.0
                else None
            ),
            "lag_alignment_mae_deg": lag_mae,
            "lag_pair_count": pair_count,
            "raw_minimum_frame": raw_min,
            "smoothed_minimum_frame": smooth_min,
            "lowest_point_offset_frames": _difference(
                smooth_min,
                raw_min,
            ),
            "raw_maximum_frame": raw_max,
            "smoothed_maximum_frame": smooth_max,
            "full_extension_offset_frames": _difference(
                smooth_max,
                raw_max,
            ),
        }
    return output


def estimate_curve_lag(
    raw: Sequence[float | None],
    smoothed: Sequence[float | None],
    *,
    max_lag_frames: int = 15,
) -> tuple[int | None, float | None, int]:
    if len(raw) != len(smoothed):
        raise ValueError("raw and smoothed curves must have equal length")
    best: tuple[float, int, int] | None = None
    maximum = max(0, int(max_lag_frames))
    for lag in range(-maximum, maximum + 1):
        differences: list[float] = []
        for raw_index, raw_value in enumerate(raw):
            smooth_index = raw_index + lag
            if smooth_index < 0 or smooth_index >= len(smoothed):
                continue
            smooth_value = smoothed[smooth_index]
            if raw_value is None or smooth_value is None:
                continue
            differences.append(abs(float(raw_value) - float(smooth_value)))
        if len(differences) < 3:
            continue
        mae = sum(differences) / len(differences)
        candidate = (mae, abs(lag), lag)
        if best is None or candidate < (best[0], abs(best[1]), best[1]):
            best = (mae, lag, len(differences))
    if best is None:
        return None, None, 0
    return best[1], round(best[0], 4), best[2]


def write_comparison_artifacts(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "angle_validation_summary.json"
    rows_path = target_dir / "angle_validation_rows.csv"
    _atomic_json(summary_path, summary)
    fields = (
        "video_id",
        "frame_index",
        "timestamp_ms",
        "joint",
        "camera_view",
        "event",
        "action",
        "landmark_visibility",
        "manual_angle_deg",
        *MODEL_FIELD_BY_OBSERVATION,
        *ERROR_FIELD_BY_MODEL.values(),
        *(f"baseline_{field}" for field in MODEL_FIELD_BY_OBSERVATION),
        *(f"baseline_{field}" for field in ERROR_FIELD_BY_MODEL.values()),
        *(f"change_{field}" for field in ERROR_FIELD_BY_MODEL.values()),
    )
    with rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return summary_path, rows_path


def _error_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"count": len(rows)}
    for error_field in ERROR_FIELD_BY_MODEL.values():
        values = [
            value
            for row in rows
            if (value := _finite(row.get(error_field))) is not None
        ]
        output[error_field.removeprefix("error_").removesuffix("_deg")] = (
            _distribution(values)
        )
    return output


def _grouped_statistics(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = sorted(
        {
            str(row.get(field, "")).strip()
            for row in rows
            if str(row.get(field, "")).strip()
        }
    )
    return {
        value: _error_statistics(
            [row for row in rows if str(row.get(field, "")).strip() == value]
        )
        for value in values
    }


def _round12_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actions = {
        str(row.get("action", "")).strip().lower()
        for row in rows
        if str(row.get("action", "")).strip()
    }
    views = {
        _normalize_camera_view(str(row.get("camera_view", "")))
        for row in rows
        if str(row.get("camera_view", "")).strip()
    }
    pairs = {
        (
            str(row.get("action", "")).strip().lower(),
            _normalize_camera_view(str(row.get("camera_view", ""))),
        )
        for row in rows
    }
    missing_actions = [item for item in REQUIRED_ROUND12_ACTIONS if item not in actions]
    missing_views = [item for item in REQUIRED_ROUND12_VIEWS if item not in views]
    missing_pairs = [
        {"action": action, "camera_view": view}
        for action in REQUIRED_ROUND12_ACTIONS
        for view in REQUIRED_ROUND12_VIEWS
        if (action, view) not in pairs
    ]
    return {
        "required_actions": list(REQUIRED_ROUND12_ACTIONS),
        "required_camera_views": list(REQUIRED_ROUND12_VIEWS),
        "observed_actions": sorted(actions),
        "observed_camera_views": sorted(views),
        "missing_actions": missing_actions,
        "missing_camera_views": missing_views,
        "missing_action_view_pairs": missing_pairs,
        "action_coverage_complete": not missing_actions,
        "camera_view_coverage_complete": not missing_views,
        "action_view_matrix_complete": not missing_pairs,
    }


def _normalize_camera_view(value: str) -> str:
    normalized = value.strip().lower().replace("°", "").replace("deg", "")
    aliases = {
        "side_view": "side",
        "30": "oblique_30",
        "30_oblique": "oblique_30",
        "oblique30": "oblique_30",
        "45": "oblique_45",
        "45_oblique": "oblique_45",
        "oblique45": "oblique_45",
        "front_view": "front",
    }
    return aliases.get(normalized, normalized)


def _attach_baseline_rows(
    rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> None:
    baseline_by_key = {
        (
            str(row.get("video_id", "")),
            int(row.get("frame_index", -1)),
            str(row.get("joint", "")),
        ): row
        for row in baseline_rows
    }
    for row in rows:
        baseline = baseline_by_key.get(
            (
                str(row.get("video_id", "")),
                int(row.get("frame_index", -1)),
                str(row.get("joint", "")),
            )
        )
        if baseline is None:
            continue
        for field in MODEL_FIELD_BY_OBSERVATION:
            row[f"baseline_{field}"] = baseline.get(field)
        for field in ERROR_FIELD_BY_MODEL.values():
            baseline_error = _finite(baseline.get(field))
            candidate_error = _finite(row.get(field))
            row[f"baseline_{field}"] = baseline_error
            row[f"change_{field}"] = (
                candidate_error - baseline_error
                if candidate_error is not None and baseline_error is not None
                else None
            )


def _compare_versions(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    angle_metrics: dict[str, Any] = {}
    comparable_checks: list[bool] = []
    baseline_overall = baseline.get("overall", {})
    candidate_overall = candidate.get("overall", {})
    for name in (
        "2d_raw",
        "2d_smoothed",
        "3d_raw",
        "3d_smoothed",
        "canonical_3d",
        "rule",
    ):
        old = baseline_overall.get(name, {}) if isinstance(baseline_overall, Mapping) else {}
        new = candidate_overall.get(name, {}) if isinstance(candidate_overall, Mapping) else {}
        fields: dict[str, Any] = {}
        field_checks: list[bool] = []
        for field in (
            "mae_deg",
            "median_absolute_error_deg",
            "p90_absolute_error_deg",
            "p95_absolute_error_deg",
        ):
            old_value = _finite(old.get(field)) if isinstance(old, Mapping) else None
            new_value = _finite(new.get(field)) if isinstance(new, Mapping) else None
            passed = (
                new_value <= old_value
                if old_value is not None and new_value is not None
                else None
            )
            fields[field] = {
                "baseline": old_value,
                "candidate": new_value,
                "delta": (
                    round(new_value - old_value, 4)
                    if old_value is not None and new_value is not None
                    else None
                ),
                "non_regression": passed,
            }
            if passed is not None:
                field_checks.append(passed)
                comparable_checks.append(passed)
        fields["non_regression"] = all(field_checks) if field_checks else None
        angle_metrics[name] = fields

    event_timing: dict[str, Any] = {}
    for event in ("all", "lowest_point", "full_extension"):
        old_values = _event_absolute_offsets(baseline, event)
        new_values = _event_absolute_offsets(candidate, event)
        old_stats = _frame_distribution(old_values)
        new_stats = _frame_distribution(new_values)
        old_mae = _finite(old_stats.get("mean_absolute_error_frames"))
        new_mae = _finite(new_stats.get("mean_absolute_error_frames"))
        passed = (
            new_mae <= old_mae
            if old_mae is not None and new_mae is not None
            else None
        )
        if passed is not None:
            comparable_checks.append(passed)
        event_timing[event] = {
            "baseline": old_stats,
            "candidate": new_stats,
            "non_regression": passed,
        }
    return {
        "baseline_label": "old_version",
        "candidate_label": "new_version",
        "angle_error": angle_metrics,
        "event_timing_error": event_timing,
        "comparable_check_count": len(comparable_checks),
        "non_regression_pass": (
            all(comparable_checks) if comparable_checks else None
        ),
    }


def _event_absolute_offsets(
    summary: Mapping[str, Any],
    event: str,
) -> list[float]:
    output: list[float] = []
    for item in summary.get("event_offsets", ()):
        if not isinstance(item, Mapping):
            continue
        if event != "all" and str(item.get("event", "")) != event:
            continue
        value = _finite(item.get("offset_frames"))
        if value is not None:
            output.append(abs(value))
    return output


def _frame_distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_absolute_error_frames": None,
            "median_absolute_error_frames": None,
            "p90_absolute_error_frames": None,
            "p95_absolute_error_frames": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean_absolute_error_frames": round(float(np.mean(array)), 4),
        "median_absolute_error_frames": round(float(np.median(array)), 4),
        "p90_absolute_error_frames": round(float(np.percentile(array, 90)), 4),
        "p95_absolute_error_frames": round(float(np.percentile(array, 95)), 4),
    }


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
        "p90_absolute_error_deg": round(
            float(np.percentile(array, 90)),
            4,
        ),
        "p95_absolute_error_deg": round(
            float(np.percentile(array, 95)),
            4,
        ),
    }


def _event_offsets(
    rows: Sequence[Mapping[str, Any]],
    curve_latency: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        event = str(row.get("event", "")).strip().lower()
        if event in {"lowest_point", "full_extension"}:
            grouped[(str(row["joint"]), event)].append(
                int(row["frame_index"])
            )
    output: list[dict[str, Any]] = []
    for (joint, event), frames in sorted(grouped.items()):
        curve = curve_latency.get(joint)
        if not isinstance(curve, Mapping):
            continue
        field = (
            "smoothed_minimum_frame"
            if event == "lowest_point"
            else "smoothed_maximum_frame"
        )
        program_frame = curve.get(field)
        manual_frame = int(round(median(frames)))
        output.append(
            {
                "joint": joint,
                "event": event,
                "manual_event_frame": manual_frame,
                "program_event_frame": program_frame,
                "offset_frames": _difference(program_frame, manual_frame),
            }
        )
    return output


def _extreme_frame(
    frames: Sequence[int],
    values: Sequence[float | None],
    *,
    minimum: bool,
) -> int | None:
    candidates = [
        (float(value), int(frame))
        for frame, value in zip(frames, values, strict=True)
        if value is not None
    ]
    if not candidates:
        return None
    selected = min(candidates) if minimum else max(candidates)
    return selected[1]


def _report_fps(report: Mapping[str, Any]) -> float:
    performance = report.get("performance")
    if isinstance(performance, Mapping):
        value = _finite(performance.get("source_fps"))
        if value is not None and value > 0.0:
            return value
    frames = [
        item
        for item in report.get("frames", ())
        if isinstance(item, Mapping)
    ]
    timestamps = [
        _finite(item.get("timestamp_ms"))
        for item in frames[:30]
    ]
    valid = [
        right - left
        for left, right in zip(timestamps, timestamps[1:])
        if left is not None and right is not None and right > left
    ]
    return 1000.0 / median(valid) if valid else 30.0


def _difference(
    left: object,
    right: object,
) -> int | None:
    if left is None or right is None:
        return None
    return int(left) - int(right)


def _short_point_name(name: str) -> str:
    return name.removeprefix("left_").removeprefix("right_")


def _finite(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ANGLE_VALIDATION_SCHEMA_VERSION",
    "MANUAL_ANGLE_DEFINITIONS",
    "append_annotation",
    "analyze_curve_latency",
    "build_manual_annotation",
    "compare_manual_annotations",
    "estimate_curve_lag",
    "export_angle_curves",
    "find_observation",
    "iter_angle_observations",
    "joint_point_names",
    "load_annotations",
    "load_report",
    "normalize_joint_name",
    "read_video_frame",
    "video_metadata",
    "write_comparison_artifacts",
]
