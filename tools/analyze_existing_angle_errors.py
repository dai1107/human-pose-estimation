from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.angle_validation import (  # noqa: E402
    find_observation,
    load_annotations,
    load_report,
)


SCHEMA_VERSION = 1
CHANNELS = (
    "raw_2d",
    "filtered_2d",
    "raw_3d",
    "filtered_3d",
    "canonical_3d",
    "selected_rule",
)
REPORT_FIELDS = {
    "raw_2d": "angle_2d_raw_deg",
    "filtered_2d": "angle_2d_smoothed_deg",
    "raw_3d": "angle_3d_raw_deg",
    "filtered_3d": "angle_3d_smoothed_deg",
    "canonical_3d": "angle_canonical_3d_deg",
    "selected_rule": "rule_angle_deg",
}
ANNOTATION_FIELDS = {
    "raw_2d": "model_2d_raw_deg",
    "filtered_2d": "model_2d_smoothed_deg",
    "raw_3d": "model_3d_raw_deg",
    "filtered_3d": "model_3d_smoothed_deg",
    "canonical_3d": "model_canonical_3d_deg",
    "selected_rule": "model_rule_angle_deg",
}
ROUND12_FIELDS = {
    channel: f"round12_{channel}_deg" for channel in CHANNELS
}
DIMENSIONS = (
    "action",
    "joint",
    "side",
    "camera_view",
    "movement_phase",
    "angle_range",
    "landmark_confidence",
    "2d_3d_disagreement",
)
INTERACTIONS = {
    "action_joint_angle_range": ("action", "joint", "angle_range"),
    "action_joint_movement_phase": (
        "action",
        "joint",
        "movement_phase",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-analyze existing manual HYROX angles across six angle "
            "channels and stratify their errors by action, joint, side, "
            "view, phase, angle range, confidence and 2D/3D disagreement."
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help=(
            "manual_angles.json or another JSON file containing an "
            "annotations list"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Frame report.json containing angle_observations.",
    )
    parser.add_argument(
        "--round12",
        type=Path,
        help=(
            "Round 12 reviewed_angle_rows.csv, a directory containing it, "
            "or a JSON list of equivalent rows."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/angle_validation/round1_error_analysis"),
    )
    parser.add_argument(
        "--minimum-finding-count",
        type=int,
        default=3,
        help="Minimum bucket size used when selecting headline findings.",
    )
    return parser


def load_round12_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        source = source / "reviewed_angle_rows.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Round 12 rows not found: {source}")
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("Round 12 JSON must be a row list or contain a rows list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("Round 12 rows must be JSON objects")
    return [dict(row) for row in rows]


def build_analysis_rows(
    annotations: Sequence[Mapping[str, Any]],
    *,
    report: Mapping[str, Any] | None = None,
    round12_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    annotation_rows = [dict(item) for item in annotations]
    round12_by_key = {
        _row_key(row): row for row in round12_rows if _row_key(row) is not None
    }
    if not annotation_rows:
        annotation_rows = [dict(item) for item in round12_rows]

    output: list[dict[str, Any]] = []
    for source in annotation_rows:
        matched = round12_by_key.get(_row_key(source))
        merged = dict(source)
        if matched is not None:
            merged.update(
                {
                    key: value
                    for key, value in matched.items()
                    if value not in (None, "") or key not in merged
                }
            )
        manual = _finite(merged.get("manual_angle_deg"))
        if manual is None:
            raise ValueError(
                "manual_angle_deg is missing for "
                f"{merged.get('annotation_id') or _row_key(merged)}"
            )
        joint_key = str(merged.get("joint", "")).strip().lower()
        frame_index = _integer(merged.get("frame_index"), default=-1)
        observation = None
        if report is not None and joint_key and frame_index >= 0:
            observation = find_observation(
                report,
                frame_index=frame_index,
                joint=joint_key,
            )

        side, joint = _split_joint(joint_key)
        action = _text(
            _first_present(
                merged.get("action"),
                observation.get("action") if observation else None,
            )
        )
        camera_view = _normalize_camera_view(
            _text(
                _first_present(
                    merged.get("camera_view"),
                    observation.get("camera_view") if observation else None,
                )
            )
        )
        phase = _normalize_token(
            _text(_first_present(merged.get("event"), merged.get("movement_phase")))
        )
        confidence_value, confidence_bucket = _landmark_confidence(
            _first_present(
                merged.get("landmark_visibility"),
                observation.get("landmark_visibility") if observation else None,
                merged.get("visibility"),
            )
        )
        row: dict[str, Any] = {
            "annotation_id": _text(merged.get("annotation_id"), unknown=""),
            "video_id": _text(
                _first_present(merged.get("video_id"), merged.get("record_id")),
                unknown="",
            ),
            "frame_index": frame_index,
            "timestamp_ms": _finite(merged.get("timestamp_ms")),
            "action": _normalize_token(action),
            "joint_key": joint_key or "unknown",
            "joint": joint,
            "side": side,
            "camera_view": camera_view,
            "movement_phase": phase,
            "manual_angle_deg": manual,
            "angle_range": _angle_range(manual),
            "landmark_confidence_value": confidence_value,
            "landmark_confidence": confidence_bucket,
        }
        for channel in CHANNELS:
            value = _channel_value(
                channel,
                annotation=merged,
                observation=observation,
                round12=matched,
            )
            if joint == "torso" and value is not None:
                value = abs(value)
            row[f"{channel}_deg"] = value
            row[f"{channel}_error_deg"] = (
                abs(value - manual) if value is not None else None
            )

        raw_disagreement = _absolute_difference(
            row["raw_2d_deg"], row["raw_3d_deg"]
        )
        filtered_disagreement = _absolute_difference(
            row["filtered_2d_deg"], row["filtered_3d_deg"]
        )
        disagreement = (
            filtered_disagreement
            if filtered_disagreement is not None
            else raw_disagreement
        )
        row["raw_2d_3d_disagreement_deg"] = raw_disagreement
        row["filtered_2d_3d_disagreement_deg"] = filtered_disagreement
        row["2d_3d_disagreement_deg"] = disagreement
        row["2d_3d_disagreement"] = _disagreement_range(disagreement)
        output.append(row)
    return output


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_finding_count: int = 3,
) -> dict[str, Any]:
    normalized = [dict(row) for row in rows]
    by_dimension = {
        dimension: _group_statistics(normalized, (dimension,))
        for dimension in DIMENSIONS
    }
    interactions = {
        name: _group_statistics(normalized, fields)
        for name, fields in INTERACTIONS.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "existing_angle_error_analysis_v1",
        "annotation_count": len(normalized),
        "matched_annotation_count": sum(
            1
            for row in normalized
            if any(_finite(row.get(f"{channel}_deg")) is not None for channel in CHANNELS)
        ),
        "channel_contract": {
            "raw_2d": "raw 2D image-landmark angle",
            "filtered_2d": "causal filtered 2D image-landmark angle",
            "raw_3d": "raw world-landmark angle; projection consistency only",
            "filtered_3d": (
                "causal filtered world-landmark angle; projection consistency only"
            ),
            "canonical_3d": (
                "canonical world-landmark angle; projection consistency only"
            ),
            "selected_rule": "angle selected by the current formal 2D rule path",
        },
        "overall": _statistics(normalized),
        "by_dimension": by_dimension,
        "interactions": interactions,
        "findings": _findings(
            normalized,
            interactions,
            minimum_count=max(1, int(minimum_finding_count)),
        ),
        "interpretation": {
            "manual_reference": "projected 2D pixel angle",
            "three_d_limitation": (
                "3D errors are 2D-projection consistency gaps, not spatial "
                "3D ground-truth errors."
            ),
            "formal_rules_changed": False,
        },
    }


def write_artifacts(
    output_dir: str | Path,
    analysis: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "angle_error_analysis.json"
    rows_path = target / "angle_error_rows.csv"
    report_path = target / "ANGLE_ERROR_ANALYSIS.md"
    summary_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = (
        "annotation_id",
        "video_id",
        "frame_index",
        "timestamp_ms",
        "action",
        "joint_key",
        "joint",
        "side",
        "camera_view",
        "movement_phase",
        "manual_angle_deg",
        "angle_range",
        "landmark_confidence_value",
        "landmark_confidence",
        *(f"{channel}_deg" for channel in CHANNELS),
        *(f"{channel}_error_deg" for channel in CHANNELS),
        "raw_2d_3d_disagreement_deg",
        "filtered_2d_3d_disagreement_deg",
        "2d_3d_disagreement_deg",
        "2d_3d_disagreement",
    )
    with rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    report_path.write_text(_markdown_report(analysis), encoding="utf-8")
    return summary_path, rows_path, report_path


def _statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    channels: dict[str, Any] = {}
    for channel in CHANNELS:
        values = [
            value
            for row in rows
            if (value := _finite(row.get(f"{channel}_error_deg"))) is not None
        ]
        channels[channel] = _distribution(values)
    paired = [
        row
        for row in rows
        if _finite(row.get("raw_2d_error_deg")) is not None
        and _finite(row.get("filtered_2d_error_deg")) is not None
    ]
    raw_mae = _mean_field(paired, "raw_2d_error_deg")
    filtered_mae = _mean_field(paired, "filtered_2d_error_deg")
    winners: Counter[str] = Counter()
    for row in rows:
        candidates = [
            (value, channel)
            for channel in CHANNELS
            if (value := _finite(row.get(f"{channel}_error_deg"))) is not None
        ]
        if candidates:
            winners[min(candidates)[1]] += 1
    return {
        "count": len(rows),
        "channels": channels,
        "paired_raw_filtered_2d": {
            "count": len(paired),
            "raw_2d_mae_deg": _rounded(raw_mae),
            "filtered_2d_mae_deg": _rounded(filtered_mae),
            "filtered_minus_raw_mae_deg": _rounded(
                filtered_mae - raw_mae
                if raw_mae is not None and filtered_mae is not None
                else None
            ),
            "filtered_better_count": sum(
                1
                for row in paired
                if float(row["filtered_2d_error_deg"])
                < float(row["raw_2d_error_deg"])
            ),
            "raw_better_count": sum(
                1
                for row in paired
                if float(row["raw_2d_error_deg"])
                < float(row["filtered_2d_error_deg"])
            ),
            "tie_count": sum(
                1
                for row in paired
                if math.isclose(
                    float(row["raw_2d_error_deg"]),
                    float(row["filtered_2d_error_deg"]),
                    abs_tol=1e-9,
                )
            ),
        },
        "best_channel_count": dict(sorted(winners.items())),
    }


def _group_statistics(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_text(row.get(field)) for field in fields)
        grouped[key].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item["statistics"] = _statistics(group)
        output.append(item)
    return output


def _findings(
    rows: Sequence[Mapping[str, Any]],
    interactions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_count: int,
) -> dict[str, Any]:
    buckets = [
        item
        for name in INTERACTIONS
        for item in interactions.get(name, ())
        if int(item.get("statistics", {}).get("count", 0)) >= minimum_count
    ]
    comparisons: list[dict[str, Any]] = []
    three_d_contexts: list[dict[str, Any]] = []
    for item in buckets:
        stats = item["statistics"]
        paired = stats["paired_raw_filtered_2d"]
        delta = _finite(paired.get("filtered_minus_raw_mae_deg"))
        descriptor = {
            key: value for key, value in item.items() if key != "statistics"
        }
        if delta is not None:
            comparisons.append(
                {
                    **descriptor,
                    "count": stats["count"],
                    "filtered_minus_raw_2d_mae_deg": delta,
                }
            )
        filtered = _metric(stats, "filtered_2d", "mae_deg")
        raw_3d = _metric(stats, "raw_3d", "mae_deg")
        if filtered is not None and raw_3d is not None and raw_3d < filtered:
            three_d_contexts.append(
                {
                    **descriptor,
                    "count": stats["count"],
                    "raw_3d_projection_gap_mae_deg": raw_3d,
                    "filtered_2d_mae_deg": filtered,
                    "raw_3d_minus_filtered_2d_deg": _rounded(raw_3d - filtered),
                }
            )
    comparisons.sort(key=lambda item: item["filtered_minus_raw_2d_mae_deg"])
    three_d_contexts.sort(key=lambda item: item["raw_3d_minus_filtered_2d_deg"])
    worst_rows = sorted(
        (
            {
                "annotation_id": row.get("annotation_id"),
                "video_id": row.get("video_id"),
                "frame_index": row.get("frame_index"),
                "action": row.get("action"),
                "joint_key": row.get("joint_key"),
                "movement_phase": row.get("movement_phase"),
                "manual_angle_deg": row.get("manual_angle_deg"),
                "selected_rule_error_deg": _finite(
                    row.get("selected_rule_error_deg")
                ),
            }
            for row in rows
            if _finite(row.get("selected_rule_error_deg")) is not None
        ),
        key=lambda item: float(item["selected_rule_error_deg"]),
        reverse=True,
    )
    return {
        "minimum_bucket_count": minimum_count,
        "largest_filtered_2d_improvements": comparisons[:10],
        "largest_filtered_2d_regressions": list(reversed(comparisons[-10:])),
        "raw_3d_contexts_better_than_filtered_2d": three_d_contexts[:10],
        "largest_selected_rule_errors": worst_rows[:10],
    }


def _markdown_report(analysis: Mapping[str, Any]) -> str:
    overall = analysis.get("overall", {})
    channels = overall.get("channels", {}) if isinstance(overall, Mapping) else {}
    lines = [
        "# 第一轮：现有人工角度分层误差分析",
        "",
        f"- 人工标注：{analysis.get('annotation_count', 0)} 条",
        f"- 六路角度可匹配：{analysis.get('matched_annotation_count', 0)} 条",
        "- 正式 HYROX 规则：未修改",
        "",
        "## 六路总体误差",
        "",
        "| 通道 | 可用数 | MAE | Median AE | P90 | P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for channel in CHANNELS:
        stats = channels.get(channel, {}) if isinstance(channels, Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                (
                    channel,
                    str(stats.get("count", 0)),
                    _degrees(stats.get("mae_deg")),
                    _degrees(stats.get("median_absolute_error_deg")),
                    _degrees(stats.get("p90_absolute_error_deg")),
                    _degrees(stats.get("p95_absolute_error_deg")),
                )
            )
            + " |"
        )
    paired = overall.get("paired_raw_filtered_2d", {})
    lines.extend(
        [
            "",
            "## Raw 与 filtered 2D",
            "",
            (
                f"在 {paired.get('count', 0)} 条成对样本中，filtered 2D 更准 "
                f"{paired.get('filtered_better_count', 0)} 条，raw 2D 更准 "
                f"{paired.get('raw_better_count', 0)} 条，持平 "
                f"{paired.get('tie_count', 0)} 条。filtered 相对 raw 的 MAE "
                f"变化为 {_signed_degrees(paired.get('filtered_minus_raw_mae_deg'))}。"
            ),
            "",
            "## 主要分桶发现",
            "",
        ]
    )
    findings = analysis.get("findings", {})
    sections = (
        ("filtered 2D 改善最大的上下文", "largest_filtered_2d_improvements"),
        ("filtered 2D 退化最大的上下文", "largest_filtered_2d_regressions"),
        (
            "raw 3D 投影差距小于 filtered 2D 的上下文",
            "raw_3d_contexts_better_than_filtered_2d",
        ),
    )
    for title, key in sections:
        lines.extend((f"### {title}", ""))
        items = findings.get(key, ()) if isinstance(findings, Mapping) else ()
        if not items:
            lines.extend(("没有满足最小样本数的分桶。", ""))
            continue
        for item in items[:5]:
            context = " / ".join(
                str(item[field])
                for field in (
                    "action",
                    "joint",
                    "movement_phase",
                    "angle_range",
                )
                if field in item
            )
            if "filtered_minus_raw_2d_mae_deg" in item:
                result = _signed_degrees(item["filtered_minus_raw_2d_mae_deg"])
            else:
                result = _signed_degrees(item.get("raw_3d_minus_filtered_2d_deg"))
            lines.append(f"- {context}（n={item.get('count', 0)}）：{result}")
        lines.append("")
    minimum_count = int(findings.get("minimum_bucket_count", 1))
    by_dimension = analysis.get("by_dimension", {})
    lines.extend(
        [
            "## 各维度 selected-rule 高误差分桶",
            "",
            "| 维度 | 分桶 | 样本数 | selected-rule MAE | raw 2D MAE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dimension in DIMENSIONS:
        items = (
            by_dimension.get(dimension, ())
            if isinstance(by_dimension, Mapping)
            else ()
        )
        candidates = [
            item
            for item in items
            if int(item.get("statistics", {}).get("count", 0)) >= minimum_count
            and _metric(item.get("statistics", {}), "selected_rule", "mae_deg")
            is not None
        ]
        if not candidates:
            continue
        worst = max(
            candidates,
            key=lambda item: float(
                _metric(item["statistics"], "selected_rule", "mae_deg") or 0.0
            ),
        )
        stats = worst["statistics"]
        lines.append(
            "| "
            + " | ".join(
                (
                    dimension,
                    str(worst.get(dimension, "unknown")),
                    str(stats.get("count", 0)),
                    _degrees(_metric(stats, "selected_rule", "mae_deg")),
                    _degrees(_metric(stats, "raw_2d", "mae_deg")),
                )
            )
            + " |"
        )
    lines.extend(("", "## selected-rule 最大逐点误差", ""))
    worst_rows = (
        findings.get("largest_selected_rule_errors", ())
        if isinstance(findings, Mapping)
        else ()
    )
    for item in worst_rows[:5]:
        lines.append(
            "- "
            f"{item.get('action')} / {item.get('joint_key')} / "
            f"{item.get('movement_phase')} / frame {item.get('frame_index')}："
            f"{_degrees(item.get('selected_rule_error_deg'))}"
        )
    lines.append("")
    lines.extend(
        [
            "## 解读边界",
            "",
            (
                "人工标签是视频像素上的投影 2D 角度，因此 3D 通道的数字只表示"
                "投影一致性差距，不是空间 3D 真值误差。本报告只用于定位误差来源，"
                "没有修改正式规则或阈值。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _channel_value(
    channel: str,
    *,
    annotation: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
    round12: Mapping[str, Any] | None,
) -> float | None:
    candidates = (
        round12.get(ROUND12_FIELDS[channel]) if round12 else None,
        observation.get(REPORT_FIELDS[channel]) if observation else None,
        annotation.get(ROUND12_FIELDS[channel]),
        annotation.get(ANNOTATION_FIELDS[channel]),
        annotation.get(f"{channel}_deg"),
    )
    return _finite(_first_present(*candidates))


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    annotation_id = _text(row.get("annotation_id"), unknown="")
    if annotation_id:
        return ("annotation_id", annotation_id)
    video_id = _text(
        _first_present(row.get("video_id"), row.get("record_id")), unknown=""
    )
    joint = _text(row.get("joint"), unknown="")
    frame = _integer(row.get("frame_index"), default=-1)
    if video_id and joint and frame >= 0:
        return ("point", video_id, frame, joint)
    return None


def _split_joint(value: str) -> tuple[str, str]:
    for side in ("left", "right"):
        prefix = f"{side}_"
        if value.startswith(prefix):
            return side, value.removeprefix(prefix) or "unknown"
    if value in {"torso", "trunk", "spine"}:
        return "center", value
    return "unknown", value or "unknown"


def _landmark_confidence(value: object) -> tuple[float | None, str]:
    numeric = _finite(value)
    if numeric is not None:
        if numeric >= 0.75:
            return numeric, "high"
        if numeric >= 0.5:
            return numeric, "medium"
        return numeric, "low"
    token = str(value or "").strip().lower()
    if token in {"high", "medium", "low"}:
        return None, token
    return None, "unknown"


def _angle_range(value: float) -> str:
    clipped = min(180.0, max(0.0, float(value)))
    lower = 170 if math.isclose(clipped, 180.0, abs_tol=1e-9) else int(clipped // 10) * 10
    upper = lower + 10
    return f"{lower:03d}–{upper:03d}°"


def _disagreement_range(value: float | None) -> str:
    if value is None:
        return "unavailable"
    for lower, upper in ((0, 5), (5, 10), (10, 20), (20, 30)):
        if value < upper:
            return f"{lower:02d}–{upper:02d}°"
    return "30°+"


def _normalize_camera_view(value: str) -> str:
    normalized = _normalize_token(value.replace("°", "").replace("deg", ""))
    return {
        "side_view": "side",
        "front_view": "front",
        "30": "oblique_30",
        "30_oblique": "oblique_30",
        "oblique30": "oblique_30",
        "45": "oblique_45",
        "45_oblique": "oblique_45",
        "oblique45": "oblique_45",
    }.get(normalized, normalized)


def _normalize_token(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    return normalized or "unknown"


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mae_deg": None,
            "median_absolute_error_deg": None,
            "p90_absolute_error_deg": None,
            "p95_absolute_error_deg": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mae_deg": _rounded(fmean(ordered)),
        "median_absolute_error_deg": _rounded(_percentile(ordered, 50)),
        "p90_absolute_error_deg": _rounded(_percentile(ordered, 90)),
        "p95_absolute_error_deg": _rounded(_percentile(ordered, 95)),
    }


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower]) * (1.0 - fraction) + float(ordered[upper]) * fraction


def _metric(
    statistics: Mapping[str, Any], channel: str, metric: str
) -> float | None:
    channels = statistics.get("channels")
    if not isinstance(channels, Mapping):
        return None
    item = channels.get(channel)
    return _finite(item.get(metric)) if isinstance(item, Mapping) else None


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        value for row in rows if (value := _finite(row.get(field))) is not None
    ]
    return fmean(values) if values else None


def _absolute_difference(left: object, right: object) -> float | None:
    left_value = _finite(left)
    right_value = _finite(right)
    if left_value is None or right_value is None:
        return None
    return abs(left_value - right_value)


def _first_present(*values: object) -> object | None:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _finite(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _integer(value: object, *, default: int) -> int:
    numeric = _finite(value)
    return int(numeric) if numeric is not None else default


def _text(value: object, *, unknown: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or unknown


def _rounded(value: float | None) -> float | None:
    return round(float(value), 4) if value is not None else None


def _degrees(value: object) -> str:
    numeric = _finite(value)
    return f"{numeric:.4f}°" if numeric is not None else "—"


def _signed_degrees(value: object) -> str:
    numeric = _finite(value)
    return f"{numeric:+.4f}°" if numeric is not None else "—"


def _default_inputs() -> tuple[Path | None, Path | None]:
    round12_dir = PROJECT_ROOT / "outputs" / "angle_validation" / "round12"
    annotations = round12_dir / "angle_validation_report_v1_model_matched.json"
    rows = round12_dir / "reviewed_angle_rows.csv"
    return (
        annotations if annotations.is_file() else None,
        rows if rows.is_file() else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_annotations, default_round12 = _default_inputs()
    annotations_path = args.annotations or default_annotations
    round12_path = args.round12 or default_round12
    if annotations_path is None and round12_path is None:
        raise SystemExit(
            "No input found. Pass --annotations and/or --round12."
        )
    annotations = (
        load_annotations(annotations_path) if annotations_path is not None else []
    )
    report = load_report(args.report) if args.report is not None else None
    round12_rows = (
        load_round12_rows(round12_path) if round12_path is not None else []
    )
    rows = build_analysis_rows(
        annotations,
        report=report,
        round12_rows=round12_rows,
    )
    analysis = analyze_rows(
        rows,
        minimum_finding_count=args.minimum_finding_count,
    )
    analysis["inputs"] = {
        "annotations": str(annotations_path) if annotations_path else None,
        "report": str(args.report) if args.report else None,
        "round12": str(round12_path) if round12_path else None,
    }
    summary_path, rows_path, report_path = write_artifacts(
        args.output_dir,
        analysis,
        rows,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "rows": str(rows_path),
                "report": str(report_path),
                "annotation_count": len(rows),
                "matched_annotation_count": analysis["matched_annotation_count"],
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
    "CHANNELS",
    "DIMENSIONS",
    "analyze_rows",
    "build_analysis_rows",
    "load_round12_rows",
    "main",
    "write_artifacts",
]
