"""Round-five CLI: synchronize Color and Depth inside each ONI only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.oni_audit import utc_now
from tools.dataset.oni_export import write_json
from tools.dataset.oni_sync import (
    build_sync_report,
    read_timeline,
    write_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize ONI-internal Color and Depth timelines. Current "
            "phone data is independent and is never paired."
        )
    )
    parser.add_argument("--dataset-root", default="datasets/hyrox")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--extracted-root",
        default=None,
        help="Default: <dataset-root>/extracted.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Default: <dataset-root>/synchronized.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Default: <dataset-root>/reports/oni_sync.",
    )
    parser.add_argument("--record-id", action="append", dest="record_ids")
    return parser


def _phone_timeline_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "phone_rgb_timeline_schema",
        "source_type": "phone_rgb",
        "status": "managed_by_independent_phone_rgb_ingest",
        "pairing_with_current_oni": "forbidden",
        "required_fields": {
            "record_id": "string",
            "source_frame_index": "integer",
            "timestamp_ms": "number",
            "source_fps": "number",
            "timestamp_source": (
                "container_pts|capture_timestamp|derived_from_fps"
            ),
        },
        "paired_group_id_policy": (
            "null unless a separately designed synchronized acquisition protocol and "
            "verifiable sync events are explicitly recorded"
        ),
    }


def _ensure_phone_timeline_schema(path: Path) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if (
            existing.get("artifact_type") == "phone_rgb_timeline_schema"
            and existing.get("status") == "active_independent_timelines"
        ):
            return
    write_json(path, _phone_timeline_schema())


def _summary_markdown(batch: dict[str, Any]) -> str:
    summary = batch["summary"]
    counts = summary["sync_quality_counts"]
    lines = [
        "# 第 5 轮 ONI 内部同步",
        "",
        f"- 状态：`{batch['status']}`",
        (
            f"- 处理记录：`{summary['record_count']}/"
            f"{summary['expected_record_count']}`"
        ),
        f"- 可执行 Color–Depth 同步：`{summary['applicable_count']}`",
        f"- 缺少所需流：`{summary['not_applicable_count']}`",
        f"- good：`{counts.get('good', 0)}`",
        f"- usable：`{counts.get('usable', 0)}`",
        f"- video_level_only：`{counts.get('video_level_only', 0)}`",
        f"- invalid：`{counts.get('invalid', 0)}`",
        f"- 精细事件训练可用：`{summary['fine_event_eligible_count']}`",
        f"- 校验错误：`{summary['validation_error_count']}`",
        "",
        "## 当前结论",
        "",
        "- 当前 32 个 ONI 均缺少 Color，因此没有伪造 Color–Depth 帧配对。",
        "- Depth 与 IR 没有被当作 RGB–Depth 配对。",
        "- 手机时间轴仅保留独立接口，未与当前 ONI 建立偏移、漂移或互相关同步。",
        "- 全部记录只能保持 video_level_only，不能进入需要 RGB–Depth 精确对齐的训练。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else dataset_root / "manifests" / "oni_records.json"
    )
    extracted_root = (
        Path(args.extracted_root).resolve()
        if args.extracted_root
        else dataset_root / "extracted"
    )
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else dataset_root / "synchronized"
    )
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else dataset_root / "reports" / "oni_sync"
    )
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phone_pairing_policy") != "forbidden_for_current_oni":
        raise SystemExit("manifest does not forbid current ONI-phone pairing")
    if any(
        record.get("paired_group_id") is not None
        for record in manifest["records"]
    ):
        raise SystemExit("current ONI paired_group_id must remain null")
    records = manifest["records"]
    selected_ids = set(args.record_ids or [])
    if selected_ids:
        known = {record["record_id"] for record in records}
        unknown = selected_ids - known
        if unknown:
            raise SystemExit(
                "unknown record ID(s): " + ", ".join(sorted(unknown))
            )
        records = [
            record
            for record in records
            if record["record_id"] in selected_ids
        ]
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        record_id = record["record_id"]
        print(
            f"[{index}/{len(records)}] synchronizing {record_id}",
            flush=True,
        )
        export_root = extracted_root / record_id
        metadata_path = export_root / "metadata.json"
        if not metadata_path.is_file():
            raise SystemExit(f"export metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        color = read_timeline(export_root / "color" / "index.csv")
        depth = read_timeline(export_root / "depth" / "index.csv")
        report, pairs = build_sync_report(
            record, metadata, color, depth
        )
        record_output = output_root / record_id
        write_pairs(record_output / "color_depth_pairs.csv", pairs)
        write_json(record_output / "sync_report.json", report)
        reports.append(report)

    quality_counts = Counter(
        report["sync_quality"] for report in reports
    )
    validation_error_count = sum(
        len(report["validation_errors"]) for report in reports
    )
    summary = {
        "expected_record_count": len(records),
        "record_count": len(reports),
        "applicable_count": sum(
            report["applicable"] for report in reports
        ),
        "not_applicable_count": sum(
            not report["applicable"] for report in reports
        ),
        "fine_event_eligible_count": sum(
            report["fine_event_training_eligible"]
            for report in reports
        ),
        "sync_quality_counts": dict(sorted(quality_counts.items())),
        "validation_error_count": validation_error_count,
    }
    passed = (
        len(reports) == len(records)
        and validation_error_count == 0
        and all(report["phone_pairing_used"] is False for report in reports)
    )
    batch = {
        "schema_version": 1,
        "artifact_type": "hyrox_oni_internal_sync_batch",
        "generated_at": utc_now(),
        "status": "passed" if passed else "failed",
        "sync_scope": "oni_internal_color_depth",
        "phone_pairing_generated": False,
        "summary": summary,
        "records": reports,
    }
    excluded = {
        "schema_version": 1,
        "artifact_type": "oni_fine_sync_exclusions",
        "definition": (
            "Records excluded from fine RGB-Depth event training because "
            "sync is not good/usable or required streams are absent."
        ),
        "records": [
            report
            for report in reports
            if not report["fine_event_training_eligible"]
        ],
    }
    write_json(report_dir / "batch_report.json", batch)
    write_json(report_dir / "excluded_records.json", excluded)
    _ensure_phone_timeline_schema(
        dataset_root / "manifests" / "phone_timeline_schema.json"
    )
    (report_dir / "sync_summary.md").write_text(
        _summary_markdown(batch),
        encoding="utf-8",
    )
    print(
        f"Round 5 {batch['status']}: {len(reports)} records, "
        f"applicable={summary['applicable_count']}, "
        f"excluded={len(excluded['records'])}, "
        f"phone_pairs=0"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
