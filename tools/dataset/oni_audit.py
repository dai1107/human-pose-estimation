"""Aggregation and validation helpers for the round-three ONI audit."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CLASS_DESCRIPTIONS = {
    "A": "color_and_depth",
    "B": "depth_without_color",
    "C": "color_without_depth",
    "D": "ir_only_or_incomplete_stream_set",
    "E": "corrupt_or_unreadable",
}
STREAM_NAMES = ("color", "depth", "ir")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_inventory(report: Mapping[str, Any]) -> list[str]:
    """Return schema/consistency errors for one oni-inspect report."""

    errors: list[str] = []
    if report.get("artifact_type") != "oni_inventory":
        errors.append("artifact_type must be oni_inventory")
    tool = report.get("tool")
    if not isinstance(tool, Mapping):
        errors.append("tool must be an object")
    elif tool.get("offline_file_only") is not True:
        errors.append("tool.offline_file_only must be true")
    file_report = report.get("file")
    if not isinstance(file_report, Mapping):
        errors.append("file must be an object")
    classification = report.get("classification")
    if not isinstance(classification, Mapping):
        errors.append("classification must be an object")
        classification_code = None
    else:
        classification_code = classification.get("code")
        if classification_code not in CLASS_DESCRIPTIONS:
            errors.append("classification.code must be A, B, C, D or E")
        if (
            classification_code in CLASS_DESCRIPTIONS
            and classification.get("description")
            != CLASS_DESCRIPTIONS[classification_code]
        ):
            errors.append("classification description does not match code")
        if classification.get("qualified_for_rgbd") is not (
            classification_code == "A"
        ):
            errors.append("qualified_for_rgbd is inconsistent with code")
    streams = report.get("streams")
    if not isinstance(streams, Mapping):
        errors.append("streams must be an object")
        return errors
    for stream_name in STREAM_NAMES:
        stream = streams.get(stream_name)
        if not isinstance(stream, Mapping):
            errors.append(f"streams.{stream_name} must be an object")
            continue
        for key in (
            "exists",
            "complete",
            "actual_frame_count",
            "decode_error_count",
            "interval_p50_us",
            "interval_p95_us",
        ):
            if key not in stream:
                errors.append(f"streams.{stream_name}.{key} is missing")
    depth = streams.get("depth")
    if (
        isinstance(depth, Mapping)
        and depth.get("exists") is True
        and not isinstance(depth.get("depth_quality"), Mapping)
    ):
        errors.append("depth stream exists but depth_quality is missing")
    elif (
        isinstance(depth, Mapping)
        and depth.get("exists") is True
        and isinstance(depth.get("depth_quality"), Mapping)
    ):
        for key in (
            "zero_value_ratio",
            "invalid_pixel_ratio",
            "min_depth_raw",
            "max_depth_raw",
            "center_region",
        ):
            if key not in depth["depth_quality"]:
                errors.append(f"depth_quality.{key} is missing")
    return errors


def enrich_inventory(
    report: dict[str, Any],
    record: Mapping[str, Any],
    *,
    inspector_return_code: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Attach immutable manifest identity to a raw inventory report."""

    report["dataset_record"] = {
        "record_id": record["record_id"],
        "source_type": record["source_type"],
        "source_filename": record["source_filename"],
        "source_file": record["source_file"],
        "manifest_sha256": record["sha256"],
        "action": record["action"],
        "recording_intent": record["recording_intent"],
        "recording_intent_verified": record["recording_intent_verified"],
        "target_athlete_selection_status": record["target_athlete"][
            "selection_status"
        ],
        "other_people_present": record["other_people_present"],
    }
    report["execution"] = {
        "inspector_return_code": inspector_return_code,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }
    report["validation_errors"] = validate_inventory(report)
    if report.get("file", {}).get("size_bytes") != record.get("size_bytes"):
        report["validation_errors"].append(
            "inspected file size does not match manifest"
        )
    return report


def record_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    record = report["dataset_record"]
    classification = report["classification"]
    file_report = report["file"]
    streams = report["streams"]
    return {
        "record_id": record["record_id"],
        "source_filename": record["source_filename"],
        "source_file": record["source_file"],
        "manifest_sha256": record["manifest_sha256"],
        "size_bytes": file_report["size_bytes"],
        "classification": classification["code"],
        "classification_description": classification["description"],
        "qualified_for_rgbd": classification["qualified_for_rgbd"],
        "open_success": file_report["open_success"],
        "complete_playback": file_report["complete_playback"],
        "duration_us": file_report["duration_us"],
        "decode_error_count": file_report["decode_error_count"],
        "streams": {
            name: {
                "exists": streams[name]["exists"],
                "complete": streams[name]["complete"],
                "width": streams[name]["width"],
                "height": streams[name]["height"],
                "pixel_format": streams[name]["pixel_format"],
                "nominal_fps": streams[name]["nominal_fps"],
                "actual_frame_count": streams[name]["actual_frame_count"],
                "first_timestamp_us": streams[name]["first_timestamp_us"],
                "last_timestamp_us": streams[name]["last_timestamp_us"],
                "first_frame_index": streams[name]["first_frame_index"],
                "last_frame_index": streams[name]["last_frame_index"],
                "timestamps_strictly_increasing": streams[name][
                    "timestamps_strictly_increasing"
                ],
                "frame_indices_continuous": streams[name][
                    "frame_indices_continuous"
                ],
                "non_increasing_timestamp_count": streams[name][
                    "non_increasing_timestamp_count"
                ],
                "frame_index_discontinuity_count": streams[name][
                    "frame_index_discontinuity_count"
                ],
                "interval_p50_us": streams[name]["interval_p50_us"],
                "interval_p95_us": streams[name]["interval_p95_us"],
                "actual_fps": streams[name]["actual_fps"],
                "estimated_dropped_frames": streams[name][
                    "estimated_dropped_frames"
                ],
                "abnormal_interval_count": streams[name][
                    "abnormal_interval_count"
                ],
                "decode_error_count": streams[name]["decode_error_count"],
            }
            for name in STREAM_NAMES
        },
        "depth_quality": streams["depth"].get("depth_quality"),
        "target_athlete_selection_status": record[
            "target_athlete_selection_status"
        ],
        "other_people_present": record["other_people_present"],
        "inspector_return_code": report["execution"][
            "inspector_return_code"
        ],
        "validation_errors": list(report.get("validation_errors", [])),
    }


def build_batch_report(
    manifest: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    *,
    started_at: str,
    completed_at: str | None = None,
) -> dict[str, Any]:
    records = [record_summary(report) for report in reports]
    class_counts = Counter(item["classification"] for item in records)
    expected_count = len(manifest.get("records", []))
    validation_error_count = sum(
        len(item["validation_errors"]) for item in records
    )
    unreadable_count = class_counts["E"]
    qualified_count = class_counts["A"]
    completed_count = len(records)
    execution_error_count = sum(
        item["inspector_return_code"] not in (0, 3) for item in records
    )
    durations = [int(item["duration_us"]) for item in records]
    depth_frames = [
        int(item["streams"]["depth"]["actual_frame_count"])
        for item in records
    ]
    ir_frames = [
        int(item["streams"]["ir"]["actual_frame_count"])
        for item in records
    ]
    depth_zero_ratios = [
        float(item["depth_quality"]["zero_value_ratio"])
        for item in records
        if item["streams"]["depth"]["exists"]
        and isinstance(item["depth_quality"], Mapping)
        and item["depth_quality"].get("zero_value_ratio") is not None
    ]
    stream_presence_counts = {
        name: sum(item["streams"][name]["exists"] for item in records)
        for name in STREAM_NAMES
    }
    total_decode_errors = sum(
        int(item["decode_error_count"]) for item in records
    )
    non_increasing_timestamps = sum(
        int(item["streams"][name]["non_increasing_timestamp_count"])
        for item in records
        for name in STREAM_NAMES
    )
    frame_index_discontinuities = sum(
        int(item["streams"][name]["frame_index_discontinuity_count"])
        for item in records
        for name in STREAM_NAMES
    )
    abnormal_intervals = sum(
        int(item["streams"][name]["abnormal_interval_count"])
        for item in records
        for name in STREAM_NAMES
    )
    estimated_dropped_frames = sum(
        int(item["streams"][name]["estimated_dropped_frames"])
        for item in records
        for name in STREAM_NAMES
    )
    execution_complete = (
        completed_count == expected_count
        and validation_error_count == 0
        and execution_error_count == 0
    )
    return {
        "schema_version": 1,
        "artifact_type": "hyrox_oni_batch_audit",
        "started_at": started_at,
        "completed_at": completed_at or utc_now(),
        "manifest": {
            "artifact_type": manifest.get("artifact_type"),
            "record_count": expected_count,
            "phone_pairing_policy": manifest.get(
                "phone_pairing_policy"
            ),
        },
        "status": "passed" if execution_complete else "failed",
        "summary": {
            "expected_record_count": expected_count,
            "audited_record_count": completed_count,
            "classification_counts": {
                code: class_counts[code] for code in CLASS_DESCRIPTIONS
            },
            "rgbd_qualified_count": qualified_count,
            "non_rgbd_count": completed_count - qualified_count,
            "unreadable_or_corrupt_count": unreadable_count,
            "validation_error_count": validation_error_count,
            "inspector_execution_error_count": execution_error_count,
            "all_records_audited": completed_count == expected_count,
            "all_reports_schema_valid": validation_error_count == 0,
            "stream_presence_counts": stream_presence_counts,
            "total_decode_error_count": total_decode_errors,
            "duration_us": {
                "minimum": min(durations, default=0),
                "maximum": max(durations, default=0),
                "sum": sum(durations),
            },
            "depth_frame_count": {
                "minimum": min(depth_frames, default=0),
                "maximum": max(depth_frames, default=0),
                "sum": sum(depth_frames),
            },
            "ir_frame_count": {
                "minimum": min(ir_frames, default=0),
                "maximum": max(ir_frames, default=0),
                "sum": sum(ir_frames),
            },
            "timeline_anomalies": {
                "non_increasing_timestamp_count": (
                    non_increasing_timestamps
                ),
                "frame_index_discontinuity_count": (
                    frame_index_discontinuities
                ),
                "abnormal_interval_count": abnormal_intervals,
                "estimated_dropped_frames": estimated_dropped_frames,
            },
            "depth_zero_value_ratio": {
                "minimum": min(depth_zero_ratios, default=0.0),
                "maximum": max(depth_zero_ratios, default=0.0),
            },
        },
        "records": records,
    }


def invalid_records(batch_report: Mapping[str, Any]) -> dict[str, Any]:
    records = [
        record
        for record in batch_report["records"]
        if not record["qualified_for_rgbd"]
    ]
    return {
        "schema_version": 1,
        "artifact_type": "hyrox_oni_invalid_records",
        "definition": (
            "Records not classified A (complete Color + Depth). "
            "B-D are readable but incomplete for RGB-D; E is unreadable."
        ),
        "count": len(records),
        "records": records,
    }


def audit_summary_markdown(batch_report: Mapping[str, Any]) -> str:
    summary = batch_report["summary"]
    counts = summary["classification_counts"]
    lines = [
        "# 第 3 轮 ONI 独立审计",
        "",
        f"- 执行状态：`{batch_report['status']}`",
        (
            f"- 已审计：`{summary['audited_record_count']}/"
            f"{summary['expected_record_count']}`"
        ),
        f"- A（Color + Depth）：`{counts['A']}`",
        f"- B（仅 Depth）：`{counts['B']}`",
        f"- C（仅 Color）：`{counts['C']}`",
        f"- D（仅 IR/其他不完整组合）：`{counts['D']}`",
        f"- E（损坏或无法完整读取）：`{counts['E']}`",
        f"- 可用于后续 RGB-D 流程：`{summary['rgbd_qualified_count']}`",
        (
            "- 流存在记录数："
            f"Color `{summary['stream_presence_counts']['color']}`，"
            f"Depth `{summary['stream_presence_counts']['depth']}`，"
            f"IR `{summary['stream_presence_counts']['ir']}`"
        ),
        f"- 解码错误总数：`{summary['total_decode_error_count']}`",
        (
            "- 时间线异常："
            f"时间戳不递增 "
            f"`{summary['timeline_anomalies']['non_increasing_timestamp_count']}`，"
            f"帧索引不连续 "
            f"`{summary['timeline_anomalies']['frame_index_discontinuity_count']}`，"
            f"异常间隔 "
            f"`{summary['timeline_anomalies']['abnormal_interval_count']}`，"
            f"估算丢帧 "
            f"`{summary['timeline_anomalies']['estimated_dropped_frames']}`"
        ),
        (
            "- Depth 总帧数："
            f"`{summary['depth_frame_count']['sum']}`；"
            f"单条范围 `{summary['depth_frame_count']['minimum']}`–"
            f"`{summary['depth_frame_count']['maximum']}`"
        ),
        (
            "- Depth 0 值比例范围："
            f"`{summary['depth_zero_value_ratio']['minimum']:.6f}`–"
            f"`{summary['depth_zero_value_ratio']['maximum']:.6f}`"
        ),
        f"- 单条报告 schema 错误：`{summary['validation_error_count']}`",
        (
            "- 审计器执行错误："
            f"`{summary['inspector_execution_error_count']}`"
        ),
        "",
        "## 数据边界",
        "",
        "- 本轮只读取当前 ONI，不连接摄像头，也不使用独立手机视频补齐分类。",
        "- 文件名中的“标准/错误”仍是未验证录制意图，不是训练真值。",
        "- 主体运动者仍待后续人工锁定；本轮结果不得绕过多人污染审计。",
        "",
        "## 非 A 类记录",
        "",
    ]
    invalid = [
        item
        for item in batch_report["records"]
        if not item["qualified_for_rgbd"]
    ]
    if not invalid:
        lines.append("- 无。")
    else:
        for item in invalid:
            lines.append(
                f"- `{item['record_id']}`：{item['classification']}，"
                f"`{item['source_filename']}`"
            )
    lines.append("")
    return "\n".join(lines)


def total_manifest_bytes(records: Iterable[Mapping[str, Any]]) -> int:
    return sum(int(record["size_bytes"]) for record in records)
