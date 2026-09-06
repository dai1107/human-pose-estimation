from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from types import MappingProxyType
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.angle_v2 import (  # noqa: E402
    AngleV2Config,
    AngleV2ShadowProcessor,
    EndpointEvent,
    JointAngleSmoothingConfig,
    joint_group,
    load_angle_v2_config,
)
from src.biomechanics.kinematics_3d import (  # noqa: E402
    ThreeDKinematicsTracker,
)
from src.product_pose import (  # noqa: E402
    ThreeDKinematicsConfig,
    ThreeDQualityConfig,
)
from src.utils.smoothing import OneEuroValueFilter  # noqa: E402
from tools.angle_validation import (  # noqa: E402
    MANUAL_ANGLE_DEFINITIONS,
    load_annotations,
)
from tools.dataset.round8_temporal import pose_result_from_raw  # noqa: E402
from tools.evaluate_manual_angle_report import (  # noqa: E402
    _joint_angle,
    _landmark_maps,
)


GRID_MIN_CUTOFF = (1.0, 1.3, 1.6, 2.0)
GRID_BETA = (0.03, 0.06, 0.10, 0.15)
CALIBRATED_EVENTS = frozenset({"lowest_point", "full_extension"})
ENDPOINT_DIRECTION_TOLERANCE_DEG = 5.0
DEFAULT_MANUAL_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "angle_validation"
    / "round12"
    / "angle_validation_report_v1_model_matched.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay existing phone pose caches through the shadow-only HYROX "
            "Angle V2 quality, filtering and endpoint pipeline."
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/hyrox")
    )
    parser.add_argument(
        "--annotations", type=Path, default=DEFAULT_MANUAL_REPORT
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/angle_v2_shadow.yaml")
    )
    parser.add_argument(
        "--scope",
        choices=("all", "annotated"),
        default="all",
        help="Replay all phone RGB caches or only manually angle-annotated records.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/angle_validation/angle_v2_round2"),
    )
    return parser


def load_phone_manifest(dataset_root: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    payload = json.loads(
        (root / "manifests" / "phone_records.json").read_text(encoding="utf-8")
    )
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("phone manifest must contain a records list")
    return [dict(item) for item in records if isinstance(item, Mapping)]


def load_record_angle_curves(
    dataset_root: str | Path,
    manifest_record: Mapping[str, Any],
    *,
    joints: Sequence[str],
    config: AngleV2Config,
) -> dict[str, Any]:
    root = Path(dataset_root)
    record_id = str(manifest_record.get("record_id", ""))
    pose_cache = manifest_record.get("pose_cache")
    paths = (
        pose_cache.get("raw_pose_paths")
        if isinstance(pose_cache, Mapping)
        else None
    )
    relative = paths.get("mediapipe_full") if isinstance(paths, Mapping) else None
    if not isinstance(relative, str):
        raise ValueError(f"{record_id}: MediaPipe Full raw pose cache is missing")
    source = root / relative
    video = manifest_record.get("video")
    video = video if isinstance(video, Mapping) else {}
    width = int(video.get("width", 0) or 0)
    height = int(video.get("height", 0) or 0)
    fps = float(video.get("fps", 0.0) or 0.0)
    tracker = ThreeDKinematicsTracker(
        ThreeDKinematicsConfig(enabled=True, decision_mode="shadow"),
        ThreeDQualityConfig(
            max_bone_length_change_ratio=config.bone_length_deviation_ratio,
            max_2d_3d_difference_deg=180.0,
        ),
    )
    curves: dict[str, list[dict[str, Any]]] = {
        joint: [] for joint in joints
    }
    frame_count = 0
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            pose = pose_result_from_raw(raw)
            frame_index = int(raw.get("frame_index", frame_count))
            timestamp_ms = float(
                raw.get(
                    "source_timestamp_ms",
                    frame_index * 1000.0 / max(fps, 1.0),
                )
            )
            result = tracker.update(
                pose,
                raw_result=pose,
                image_width=width,
                image_height=height,
                validation_enabled=False,
            )
            image_points, world_points = _landmark_maps(raw, "raw")
            for joint in joints:
                measurement = result.measurements.get(f"{joint}_angle")
                if measurement is not None:
                    raw_2d = _finite(measurement.raw_2d)
                    raw_3d = _finite(measurement.raw_3d)
                    confidence = float(measurement.confidence)
                    reasons = list(measurement.quality_reasons)
                else:
                    raw_2d = _joint_angle(
                        image_points,
                        joint,
                        dimensions=(width, height),
                        spatial=False,
                    )
                    raw_3d = _joint_angle(
                        world_points,
                        joint,
                        dimensions=None,
                        spatial=True,
                    )
                    confidence = _manual_joint_confidence(image_points, joint)
                    reasons = []
                curves[joint].append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "raw_2d_angle_deg": raw_2d,
                        "raw_3d_angle_deg": raw_3d,
                        "confidence": confidence,
                        "quality_reasons": reasons,
                    }
                )
            frame_count += 1
    return {
        "record_id": record_id,
        "action": str(manifest_record.get("action", "unknown")),
        "camera_view": str(manifest_record.get("camera_view", "unknown")),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "source": str(source),
        "curves": curves,
    }


def tune_joint_smoothing(
    record_data: Mapping[str, Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_group[joint_group(str(annotation.get("joint", "")))].append(annotation)
    output: dict[str, Any] = {}
    for group, group_annotations in sorted(by_group.items()):
        candidates = []
        for min_cutoff in GRID_MIN_CUTOFF:
            for beta in GRID_BETA:
                filtered = _filtered_lookup(
                    record_data,
                    group_annotations,
                    min_cutoff=min_cutoff,
                    beta=beta,
                )
                errors: list[float] = []
                event_offsets: list[float] = []
                event_offsets_ms: list[float] = []
                for annotation in group_annotations:
                    key = _annotation_key(annotation)
                    value = filtered.get(key)
                    manual = _finite(annotation.get("manual_angle_deg"))
                    if value is not None and manual is not None:
                        errors.append(abs(value - manual))
                    event = str(annotation.get("event", ""))
                    if event in CALIBRATED_EVENTS:
                        offset = _event_offset(
                            record_data,
                            annotation,
                            filtered,
                            minimum=event == "lowest_point",
                        )
                        if offset is not None:
                            event_offsets.append(abs(offset[0]))
                            event_offsets_ms.append(abs(offset[1]))
                distribution = _distribution(errors)
                lag_frames = fmean(event_offsets) if event_offsets else None
                lag_ms = fmean(event_offsets_ms) if event_offsets_ms else None
                score = _smoothing_score(distribution, lag_frames)
                candidates.append(
                    {
                        "min_cutoff": min_cutoff,
                        "beta": beta,
                        "d_cutoff": 1.0,
                        **distribution,
                        "event_count": len(event_offsets),
                        "event_lag_mae_frames": _rounded(lag_frames),
                        "event_lag_mae_ms": _rounded(lag_ms),
                        "selection_score": _rounded(score),
                    }
                )
        eligible = [
            row
            for row in candidates
            if row["count"] > 0 and row["selection_score"] is not None
        ]
        selected = min(
            eligible,
            key=lambda row: (
                float(row["selection_score"]),
                float(row["p95_absolute_error_deg"]),
                float(row["mae_deg"]),
            ),
        )
        output[group] = {
            "annotation_count": len(group_annotations),
            "selected": selected,
            "grid": sorted(
                candidates,
                key=lambda row: (
                    float(row["selection_score"] or math.inf),
                    row["min_cutoff"],
                    row["beta"],
                ),
            ),
            "selection_policy": (
                "MAE + 0.25*P90 + 0.25*P95 + 0.5*event_lag_frames; "
                "3D channels are not fitted to projected 2D labels"
            ),
        }
    return output


def config_with_tuned_smoothing(
    config: AngleV2Config,
    tuning: Mapping[str, Mapping[str, Any]],
) -> AngleV2Config:
    profiles = dict(config.joint_smoothing)
    for group, payload in tuning.items():
        selected = payload.get("selected")
        if not isinstance(selected, Mapping):
            continue
        profiles[group] = JointAngleSmoothingConfig(
            min_cutoff=float(selected["min_cutoff"]),
            beta=float(selected["beta"]),
            d_cutoff=float(selected.get("d_cutoff", 1.0)),
        )
    return replace(config, joint_smoothing=MappingProxyType(profiles))


def replay_angle_v2(
    record_data: Mapping[str, Mapping[str, Any]],
    *,
    config: AngleV2Config,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    totals: Counter[str] = Counter()
    by_joint: dict[str, Counter[str]] = defaultdict(Counter)
    lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    endpoint_rows: list[dict[str, Any]] = []
    record_summaries = []
    for record_id, record in sorted(record_data.items()):
        processor = AngleV2ShadowProcessor(config)
        record_counts: Counter[str] = Counter()
        curves = record.get("curves", {})
        for joint, samples in sorted(curves.items()):
            for sample in samples:
                result = processor.observe(
                    joint=joint,
                    frame_index=int(sample["frame_index"]),
                    timestamp_ms=float(sample["timestamp_ms"]),
                    raw_2d_angle_deg=_finite(sample.get("raw_2d_angle_deg")),
                    raw_3d_angle_deg=_finite(sample.get("raw_3d_angle_deg")),
                    confidence=float(sample.get("confidence", 0.0)),
                    quality_reasons=sample.get("quality_reasons", ()),
                )
                row = result.as_dict()
                lookup[(record_id, joint, result.frame_index)] = row
                record_counts["sample_count"] += 1
                by_joint[joint]["sample_count"] += 1
                if result.angle_valid:
                    record_counts["angle_valid_count"] += 1
                    by_joint[joint]["angle_valid_count"] += 1
                if not result.bone_length_valid:
                    record_counts["bone_length_rejection_count"] += 1
                    by_joint[joint]["bone_length_rejection_count"] += 1
                if result.temporal_outlier:
                    record_counts["temporal_outlier_count"] += 1
                    by_joint[joint]["temporal_outlier_count"] += 1
                if result.two_d_three_d_conflict:
                    record_counts["2d_3d_conflict_count"] += 1
                    by_joint[joint]["2d_3d_conflict_count"] += 1
                for endpoint in result.endpoints:
                    payload = {
                        "record_id": record_id,
                        "action": record.get("action"),
                        **endpoint.as_dict(),
                    }
                    endpoint_rows.append(payload)
                    record_counts[f"endpoint_{endpoint.kind}_count"] += 1
                    by_joint[joint][f"endpoint_{endpoint.kind}_count"] += 1
        totals.update(record_counts)
        record_summaries.append(
            {
                "record_id": record_id,
                "action": record.get("action"),
                "camera_view": record.get("camera_view"),
                **dict(record_counts),
            }
        )
    return (
        {
            "record_count": len(record_data),
            "totals": dict(totals),
            "by_joint": {
                joint: dict(values) for joint, values in sorted(by_joint.items())
            },
            "records": record_summaries,
        },
        lookup,
        endpoint_rows,
    )


def compare_annotations(
    annotations: Sequence[Mapping[str, Any]],
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    *,
    round12_rows: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    endpoint_index: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in endpoints:
        endpoint_index[(str(item.get("record_id")), str(item.get("joint")))].append(item)
    baseline = _load_round12_lookup(round12_rows) if round12_rows else {}
    rows = []
    for annotation in annotations:
        record_id = str(
            annotation.get("record_id", annotation.get("video_id", ""))
        )
        joint = str(annotation.get("joint", ""))
        frame_index = int(annotation.get("frame_index", -1))
        manual = _finite(annotation.get("manual_angle_deg"))
        shadow = lookup.get((record_id, joint, frame_index), {})
        raw = _finite(shadow.get("raw_2d_angle_deg"))
        filtered = _finite(shadow.get("filtered_2d_angle_deg"))
        event_value = None
        event_frame = None
        event = str(annotation.get("event", ""))
        kind = "minimum" if event == "lowest_point" else "maximum"
        if event in CALIBRATED_EVENTS:
            reference_raw = raw
            candidates = [
                item
                for item in endpoint_index.get((record_id, joint), ())
                if item.get("kind") == kind
                and abs(int(item.get("frame_index", -10_000)) - frame_index) <= 15
                and _endpoint_direction_agrees(
                    item,
                    kind=kind,
                    reference_raw=reference_raw,
                )
            ]
            if candidates:
                selected = min(
                    candidates,
                    key=lambda item: abs(int(item["frame_index"]) - frame_index),
                )
                event_value = _finite(selected.get("raw_extremum_angle_deg"))
                event_frame = int(selected["frame_index"])
        old = baseline.get(str(annotation.get("annotation_id", "")), {})
        row = {
            "annotation_id": annotation.get("annotation_id"),
            "record_id": record_id,
            "action": annotation.get("action"),
            "frame_index": frame_index,
            "joint": joint,
            "joint_group": joint_group(joint),
            "camera_view": annotation.get("camera_view"),
            "event": event,
            "manual_angle_deg": manual,
            "raw_2d_angle_deg": raw,
            "raw_2d_error_deg": _error(raw, manual),
            "round12_filtered_2d_angle_deg": _finite(
                old.get("round12_filtered_2d_deg")
            ),
            "round12_filtered_2d_error_deg": _error(
                _finite(old.get("round12_filtered_2d_deg")), manual
            ),
            "angle_v2_filtered_2d_angle_deg": filtered,
            "angle_v2_filtered_2d_error_deg": _error(filtered, manual),
            "angle_v2_rule_event_angle_deg": event_value,
            "angle_v2_rule_event_error_deg": _error(event_value, manual),
            "angle_v2_event_frame": event_frame,
            "angle_v2_event_offset_frames": (
                event_frame - frame_index if event_frame is not None else None
            ),
            "angle_valid": shadow.get("angle_valid"),
            "evidence_valid": shadow.get("evidence_valid"),
            "reason_codes": ",".join(shadow.get("reason_codes", ())),
        }
        rows.append(row)
    paired_rows = [
        row
        for row in rows
        if _finite(row["raw_2d_error_deg"]) is not None
        and _finite(row["round12_filtered_2d_error_deg"]) is not None
        and _finite(row["angle_v2_filtered_2d_error_deg"]) is not None
    ]
    rejected_rows = [
        row
        for row in rows
        if _finite(row["angle_v2_filtered_2d_error_deg"]) is None
    ]
    rejection_reasons: Counter[str] = Counter()
    for row in rejected_rows:
        reasons = [
            item for item in str(row.get("reason_codes", "")).split(",") if item
        ]
        if not reasons:
            reasons = ["ANGLE_V2_SAMPLE_MISSING"]
        rejection_reasons.update(reasons)
    v2_vs_round12 = [
        (
            float(row["angle_v2_filtered_2d_error_deg"]),
            float(row["round12_filtered_2d_error_deg"]),
        )
        for row in paired_rows
    ]
    endpoint_paired_rows = [
        row
        for row in rows
        if _finite(row["raw_2d_error_deg"]) is not None
        and _finite(row["angle_v2_rule_event_error_deg"]) is not None
    ]
    endpoint_raw_errors = [
        float(row["raw_2d_error_deg"]) for row in endpoint_paired_rows
    ]
    endpoint_errors = [
        float(row["angle_v2_rule_event_error_deg"])
        for row in endpoint_paired_rows
    ]
    endpoint_clear_improvement = (
        len(endpoint_paired_rows) >= 30
        and fmean(endpoint_errors) < fmean(endpoint_raw_errors)
        and float(np.percentile(endpoint_errors, 95))
        <= float(np.percentile(endpoint_raw_errors, 95))
    )
    summary = {
        "annotation_count": len(rows),
        "raw_2d": _distribution(
            [value for row in rows if (value := _finite(row["raw_2d_error_deg"])) is not None]
        ),
        "round12_filtered_2d": _distribution(
            [
                value
                for row in rows
                if (
                    value := _finite(row["round12_filtered_2d_error_deg"])
                )
                is not None
            ]
        ),
        "angle_v2_filtered_2d": _distribution(
            [
                value
                for row in rows
                if (
                    value := _finite(row["angle_v2_filtered_2d_error_deg"])
                )
                is not None
            ]
        ),
        "angle_v2_rule_event": _distribution(
            [
                value
                for row in rows
                if (value := _finite(row["angle_v2_rule_event_error_deg"]))
                is not None
            ]
        ),
        "event_offset_frames": _distribution(
            [
                abs(float(row["angle_v2_event_offset_frames"]))
                for row in rows
                if row["angle_v2_event_offset_frames"] is not None
            ]
        ),
        "angle_valid_count": sum(row.get("angle_valid") is True for row in rows),
        "evidence_valid_count": sum(row.get("evidence_valid") is True for row in rows),
        "paired_available_comparison": {
            "count": len(paired_rows),
            "raw_2d": _distribution(
                [float(row["raw_2d_error_deg"]) for row in paired_rows]
            ),
            "round12_filtered_2d": _distribution(
                [float(row["round12_filtered_2d_error_deg"]) for row in paired_rows]
            ),
            "angle_v2_filtered_2d": _distribution(
                [float(row["angle_v2_filtered_2d_error_deg"]) for row in paired_rows]
            ),
            "angle_v2_vs_round12": {
                "angle_v2_better_count": sum(v2 < old for v2, old in v2_vs_round12),
                "round12_better_count": sum(old < v2 for v2, old in v2_vs_round12),
                "tied_count": sum(abs(v2 - old) <= 1e-9 for v2, old in v2_vs_round12),
                "mean_error_delta_deg": _rounded(
                    fmean(v2 - old for v2, old in v2_vs_round12)
                    if v2_vs_round12
                    else None
                ),
            },
        },
        "angle_v2_filtered_rejections": {
            "count": len(rejected_rows),
            "reason_counts": dict(sorted(rejection_reasons.items())),
        },
        "endpoint_shadow_assessment": {
            "paired_count": len(endpoint_paired_rows),
            "annotation_frame_raw_2d": _distribution(endpoint_raw_errors),
            "angle_v2_rule_event": _distribution(endpoint_errors),
            "endpoint_better_count": sum(
                endpoint < raw
                for endpoint, raw in zip(endpoint_errors, endpoint_raw_errors)
            ),
            "annotation_frame_better_count": sum(
                raw < endpoint
                for endpoint, raw in zip(endpoint_errors, endpoint_raw_errors)
            ),
            "tied_count": sum(
                abs(endpoint - raw) <= 1e-9
                for endpoint, raw in zip(endpoint_errors, endpoint_raw_errors)
            ),
            "minimum_required_pairs": 30,
            "clear_shadow_improvement": endpoint_clear_improvement,
            "formal_use_allowed": False,
            "decision": (
                "insufficient endpoint evidence; keep diagnostic shadow-only"
                if not endpoint_clear_improvement
                else "shadow improvement observed; formal use still requires independent validation"
            ),
        },
    }
    return summary, rows


def _endpoint_direction_agrees(
    endpoint: Mapping[str, Any],
    *,
    kind: str,
    reference_raw: float | None,
) -> bool:
    value = _finite(endpoint.get("raw_extremum_angle_deg"))
    if value is None:
        return False
    if reference_raw is None:
        return True
    if kind == "minimum":
        return value <= reference_raw + ENDPOINT_DIRECTION_TOLERANCE_DEG
    return value >= reference_raw - ENDPOINT_DIRECTION_TOLERANCE_DEG


def write_artifacts(
    output_dir: str | Path,
    payload: Mapping[str, Any],
    annotation_rows: Sequence[Mapping[str, Any]],
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "angle_v2_shadow_replay.json"
    annotations_path = target / "angle_v2_annotation_rows.csv"
    endpoints_path = target / "angle_v2_endpoints.csv"
    report_path = target / "ANGLE_V2_ROUND2_REPORT.md"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(annotations_path, annotation_rows)
    _write_csv(endpoints_path, endpoint_rows)
    report_path.write_text(_markdown(payload), encoding="utf-8")
    return summary_path, annotations_path, endpoints_path, report_path


def _filtered_lookup(
    record_data: Mapping[str, Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    min_cutoff: float,
    beta: float,
) -> dict[tuple[str, str, int], float | None]:
    wanted = {
        (
            str(item.get("record_id", item.get("video_id", ""))),
            str(item.get("joint", "")),
        )
        for item in annotations
    }
    output: dict[tuple[str, str, int], float | None] = {}
    for record_id, joint in sorted(wanted):
        record = record_data.get(record_id)
        curves = record.get("curves") if isinstance(record, Mapping) else None
        samples = curves.get(joint, ()) if isinstance(curves, Mapping) else ()
        filter_ = OneEuroValueFilter(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=1.0,
            max_gap_ms_before_reset=250.0,
        )
        for sample in samples:
            frame = int(sample["frame_index"])
            value = _finite(sample.get("raw_2d_angle_deg"))
            filtered = (
                filter_.apply(value, timestamp_ms=float(sample["timestamp_ms"]))
                if value is not None
                else None
            )
            output[(record_id, joint, frame)] = filtered
    return output


def _event_offset(
    record_data: Mapping[str, Mapping[str, Any]],
    annotation: Mapping[str, Any],
    filtered: Mapping[tuple[str, str, int], float | None],
    *,
    minimum: bool,
) -> tuple[int, float] | None:
    record_id, joint, human_frame = _annotation_key(annotation)
    record = record_data.get(record_id)
    curves = record.get("curves") if isinstance(record, Mapping) else None
    samples = curves.get(joint, ()) if isinstance(curves, Mapping) else ()
    candidates = []
    for sample in samples:
        frame = int(sample["frame_index"])
        if abs(frame - human_frame) > 15:
            continue
        value = filtered.get((record_id, joint, frame))
        if value is not None:
            candidates.append((float(value), frame, float(sample["timestamp_ms"])))
    if not candidates:
        return None
    selected = min(candidates) if minimum else max(candidates)
    human_timestamp = _finite(annotation.get("timestamp_ms"))
    offset_ms = (
        selected[2] - human_timestamp
        if human_timestamp is not None
        else (selected[1] - human_frame) * 1000.0 / max(float(record.get("fps", 30.0)), 1.0)
    )
    return selected[1] - human_frame, offset_ms


def _smoothing_score(metrics: Mapping[str, Any], lag_frames: float | None) -> float | None:
    values = [
        _finite(metrics.get("mae_deg")),
        _finite(metrics.get("p90_absolute_error_deg")),
        _finite(metrics.get("p95_absolute_error_deg")),
    ]
    if any(value is None for value in values):
        return None
    return (
        float(values[0])
        + 0.25 * float(values[1])
        + 0.25 * float(values[2])
        + 0.5 * float(lag_frames or 0.0)
    )


def _manual_joint_confidence(
    points: Mapping[str, Any], joint: str
) -> float:
    names = [
        name
        for name in MANUAL_ANGLE_DEFINITIONS.get(joint, ())
        if name != "vertical_reference"
    ]
    values = []
    for name in names:
        point = points.get(name)
        if point is None:
            return 0.0
        raw = (
            point.get("visibility", point.get("confidence", 0.0))
            if isinstance(point, Mapping)
            else getattr(point, "visibility", getattr(point, "confidence", 0.0))
        )
        value = _finite(raw)
        values.append(0.0 if value is None else value)
    return min(values, default=0.0)


def _annotation_key(annotation: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(annotation.get("record_id", annotation.get("video_id", ""))),
        str(annotation.get("joint", "")),
        int(annotation.get("frame_index", -1)),
    )


def _load_round12_lookup(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        source = source / "reviewed_angle_rows.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("annotation_id", "")): dict(row)
            for row in csv.DictReader(handle)
        }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


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
        "mae_deg": _rounded(float(np.mean(array))),
        "median_absolute_error_deg": _rounded(float(np.median(array))),
        "p90_absolute_error_deg": _rounded(float(np.percentile(array, 90))),
        "p95_absolute_error_deg": _rounded(float(np.percentile(array, 95))),
    }


def _error(value: float | None, manual: float | None) -> float | None:
    return abs(value - manual) if value is not None and manual is not None else None


def _finite(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if math.isfinite(resolved) else None


def _rounded(value: float | None) -> float | None:
    return round(float(value), 4) if value is not None else None


def _markdown(payload: Mapping[str, Any]) -> str:
    comparison = payload.get("manual_angle_comparison", {})
    replay = payload.get("replay", {})
    totals = replay.get("totals", {}) if isinstance(replay, Mapping) else {}
    lines = [
        "# 第 2 轮：HYROX Angle V2 Shadow 回放",
        "",
        f"- 回放记录：{replay.get('record_count', 0)}",
        f"- 角度样本：{totals.get('sample_count', 0)}",
        f"- 骨长质量拒绝：{totals.get('bone_length_rejection_count', 0)}",
        f"- 时序异常拒绝：{totals.get('temporal_outlier_count', 0)}",
        f"- 2D/3D 冲突降级：{totals.get('2d_3d_conflict_count', 0)}",
        "- 正式规则变化：无",
        "",
        "## 人工角度对照",
        "",
        "| 通道 | 数量 | MAE | Median | P90 | P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("raw_2d", "round12_filtered_2d", "angle_v2_filtered_2d", "angle_v2_rule_event"):
        stats = comparison.get(name, {}) if isinstance(comparison, Mapping) else {}
        lines.append(
            f"| {name} | {stats.get('count', 0)} | {_degree(stats.get('mae_deg'))} | "
            f"{_degree(stats.get('median_absolute_error_deg'))} | "
            f"{_degree(stats.get('p90_absolute_error_deg'))} | "
            f"{_degree(stats.get('p95_absolute_error_deg'))} |"
        )
    lines.extend(
        [
            "",
            "## 同样本配对对照",
            "",
        ]
    )
    paired = comparison.get("paired_available_comparison", {})
    lines.extend(
        [
            f"Angle V2 有效且三个 2D 通道均可用的人工标注：{paired.get('count', 0)}。",
            "",
            "| 通道 | 数量 | MAE | P90 | P95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("raw_2d", "round12_filtered_2d", "angle_v2_filtered_2d"):
        stats = paired.get(name, {}) if isinstance(paired, Mapping) else {}
        lines.append(
            f"| {name} | {stats.get('count', 0)} | {_degree(stats.get('mae_deg'))} | "
            f"{_degree(stats.get('p90_absolute_error_deg'))} | "
            f"{_degree(stats.get('p95_absolute_error_deg'))} |"
        )
    rejection = comparison.get("angle_v2_filtered_rejections", {})
    reason_counts = rejection.get("reason_counts", {}) if isinstance(rejection, Mapping) else {}
    lines.extend(
        [
            "",
            f"Angle V2 filtered 缺失：{rejection.get('count', 0)}；原因计数："
            + (", ".join(f"{name}={count}" for name, count in reason_counts.items()) or "无"),
            "",
            "## 端点采用结论",
            "",
        ]
    )
    endpoint_assessment = comparison.get("endpoint_shadow_assessment", {})
    lines.extend(
        [
            f"- 配对端点：{endpoint_assessment.get('paired_count', 0)} / "
            f"最低要求 {endpoint_assessment.get('minimum_required_pairs', 30)}",
            f"- clear shadow improvement："
            f"{str(endpoint_assessment.get('clear_shadow_improvement', False)).lower()}",
            "- 正式使用允许：false",
            "- 结论：端点角继续仅作诊断，不接入正式动作规则。",
            "",
            "## 安全边界",
            "",
            "所有新结果均为 shadow evidence。显示角仍使用现有链路；端点角只记录为 "
            "rule_event_angle 候选；骨长、时序异常和 2D/3D 冲突只会降级证据，"
            "不会产生新的正式 PASS。",
            "",
        ]
    )
    return "\n".join(lines)


def _degree(value: object) -> str:
    numeric = _finite(value)
    return "—" if numeric is None else f"{numeric:.4f}°"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_angle_v2_config(args.config)
    annotations = load_annotations(args.annotations)
    manifest = load_phone_manifest(args.dataset_root)
    annotated_ids = {
        str(item.get("record_id", item.get("video_id", "")))
        for item in annotations
    }
    selected_records = [
        item
        for item in manifest
        if args.scope == "all" or str(item.get("record_id")) in annotated_ids
    ]
    joints = sorted(
        {
            str(item.get("joint", ""))
            for item in annotations
            if str(item.get("joint", ""))
        }
        | {
            f"{side}_{joint}"
            for side in ("left", "right")
            for joint in ("knee", "hip", "elbow", "shoulder", "ankle")
        }
    )
    record_data = {
        str(record["record_id"]): load_record_angle_curves(
            args.dataset_root,
            record,
            joints=joints,
            config=config,
        )
        for record in selected_records
    }
    tuning = tune_joint_smoothing(record_data, annotations)
    tuned_config = config_with_tuned_smoothing(config, tuning)
    replay, lookup, endpoint_rows = replay_angle_v2(
        record_data, config=tuned_config
    )
    comparison, annotation_rows = compare_annotations(
        annotations,
        lookup,
        endpoint_rows,
        round12_rows=(
            PROJECT_ROOT
            / "outputs"
            / "angle_validation"
            / "round12"
            / "reviewed_angle_rows.csv"
        ),
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "hyrox_angle_v2_round2_shadow_replay_v1",
        "shadow_only": True,
        "formal_rules_changed": False,
        "config_source": str(args.config),
        "base_config": config.as_dict(),
        "tuned_shadow_config": tuned_config.as_dict(),
        "smoothing_parameter_search": tuning,
        "replay": replay,
        "manual_angle_comparison": comparison,
        "limitations": [
            "manual angles are projected 2D labels, not spatial 3D truth",
            "wrist and ankle groups have no direct manual-angle calibration labels",
            "all Round-2 outputs remain shadow-only",
        ],
    }
    paths = write_artifacts(
        args.output_dir,
        payload,
        annotation_rows,
        endpoint_rows,
    )
    print(
        json.dumps(
            {
                "summary": str(paths[0]),
                "annotations": str(paths[1]),
                "endpoints": str(paths[2]),
                "report": str(paths[3]),
                "record_count": replay["record_count"],
                "formal_rules_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compare_annotations",
    "config_with_tuned_smoothing",
    "load_phone_manifest",
    "load_record_angle_curves",
    "main",
    "replay_angle_v2",
    "tune_joint_smoothing",
    "write_artifacts",
]
