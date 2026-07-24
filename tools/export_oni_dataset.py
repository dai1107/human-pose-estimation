"""Round-four CLI: losslessly export every manifest ONI record."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.oni_export import (
    STREAM_NAMES,
    aggregate_counts,
    create_color_preview,
    create_depth_preview,
    fingerprints_match,
    stream_fingerprints,
    total_export_payload_bytes,
    validate_export,
    write_json,
)
from tools.dataset.oni_audit import utc_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Losslessly export ONI streams with source frame indices and "
            "timestamps. Recorded files only; no camera or phone pairing."
        )
    )
    parser.add_argument("--dataset-root", default="datasets/hyrox")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--audit-dir",
        default=None,
        help="Default: <dataset-root>/reports/oni_audit/records.",
    )
    parser.add_argument(
        "--exporter",
        default="tools/oni_bridge/oni-export.exe",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Default: <dataset-root>/extracted.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Default: <dataset-root>/reports/oni_export.",
    )
    parser.add_argument("--record-id", action="append", dest="record_ids")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and summarize existing exports without replaying ONI.",
    )
    parser.add_argument(
        "--skip-previews",
        action="store_true",
        help="Skip derived MP4 previews; lossless frames are still exported.",
    )
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        help="Skip the repeat export/hash comparison.",
    )
    return parser


def _safe_reset_record_dir(record_dir: Path, output_root: Path) -> None:
    resolved_root = output_root.resolve()
    resolved_record = record_dir.resolve()
    if (
        resolved_record.parent != resolved_root
        or not resolved_record.name.startswith("oni_")
    ):
        raise RuntimeError(f"unsafe export reset target: {resolved_record}")
    if record_dir.exists():
        shutil.rmtree(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)


def _run_exporter(
    exporter: Path,
    input_path: Path,
    output_path: Path,
    *,
    timeout_seconds: float,
) -> tuple[int, float, str]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [
                str(exporter),
                str(input_path),
                "--output",
                str(output_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return 124, time.perf_counter() - started, str(error)
    details = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    return completed.returncode, time.perf_counter() - started, details


def _load_audit(audit_dir: Path, record_id: str) -> dict[str, Any]:
    path = audit_dir / f"{record_id}.json"
    if not path.is_file():
        raise RuntimeError(f"round-three audit not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _finalize_record(
    record: dict[str, Any],
    audit: dict[str, Any],
    record_dir: Path,
    *,
    return_code: int,
    elapsed_seconds: float,
    create_previews: bool,
) -> dict[str, Any]:
    metadata_path = record_dir / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"export metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation_errors = validate_export(metadata, audit, record_dir)
    if return_code != 0:
        validation_errors.append(f"oni-export return code: {return_code}")
    fingerprints = stream_fingerprints(record_dir)
    previews: dict[str, Any] = {}
    if create_previews and not validation_errors:
        color_preview = create_color_preview(record_dir, metadata)
        depth_preview = create_depth_preview(record_dir, metadata)
        if color_preview is not None:
            previews["color"] = color_preview
        if depth_preview is not None:
            previews["depth"] = depth_preview
    metadata["dataset_record"] = {
        "record_id": record["record_id"],
        "source_type": record["source_type"],
        "source_filename": record["source_filename"],
        "source_file": record["source_file"],
        "manifest_sha256": record["sha256"],
        "paired_group_id": record["paired_group_id"],
        "target_athlete_selection_status": record["target_athlete"][
            "selection_status"
        ],
        "other_people_present": record["other_people_present"],
    }
    metadata["execution"] = {
        "return_code": return_code,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }
    metadata["stream_fingerprints"] = fingerprints
    metadata["derived_previews"] = previews
    metadata["validation_errors"] = validation_errors
    metadata["export_payload_bytes"] = total_export_payload_bytes(record_dir)
    write_json(metadata_path, metadata)
    return metadata


def _summary_record(metadata: dict[str, Any]) -> dict[str, Any]:
    record = metadata["dataset_record"]
    return {
        "record_id": record["record_id"],
        "source_filename": record["source_filename"],
        "manifest_sha256": record["manifest_sha256"],
        "complete": metadata["complete"],
        "streams": metadata["streams"],
        "stream_fingerprints": metadata["stream_fingerprints"],
        "derived_previews": metadata["derived_previews"],
        "validation_errors": metadata["validation_errors"],
        "export_payload_bytes": metadata["export_payload_bytes"],
    }


def _determinism_check(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    exporter: Path,
    output_root: Path,
    report_dir: Path,
    formal_metadata: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    repeat_root = report_dir / "determinism_repeat"
    repeat_dir = repeat_root / record["record_id"]
    repeat_root.mkdir(parents=True, exist_ok=True)
    if repeat_dir.exists():
        shutil.rmtree(repeat_dir)
    repeat_dir.mkdir(parents=True)
    return_code, elapsed, details = _run_exporter(
        exporter,
        dataset_root / record["source_file"],
        repeat_dir,
        timeout_seconds=timeout_seconds,
    )
    repeat_fingerprints = (
        stream_fingerprints(repeat_dir)
        if (repeat_dir / "metadata.json").is_file()
        else {}
    )
    formal_fingerprints = formal_metadata["stream_fingerprints"]
    passed = (
        return_code == 0
        and bool(repeat_fingerprints)
        and fingerprints_match(
            formal_fingerprints, repeat_fingerprints
        )
    )
    result = {
        "schema_version": 1,
        "artifact_type": "oni_export_determinism_report",
        "record_id": record["record_id"],
        "source_filename": record["source_filename"],
        "checks": {
            "repeat_export_return_code_zero": return_code == 0,
            "frame_counts_equal": all(
                formal_fingerprints[name]["frame_file_count"]
                == repeat_fingerprints.get(name, {}).get("frame_file_count")
                for name in STREAM_NAMES
            ),
            "index_hashes_equal": all(
                formal_fingerprints[name]["index_sha256"]
                == repeat_fingerprints.get(name, {}).get("index_sha256")
                for name in STREAM_NAMES
            ),
            "frame_content_hashes_equal": all(
                formal_fingerprints[name]["frames_aggregate_sha256"]
                == repeat_fingerprints.get(name, {}).get(
                    "frames_aggregate_sha256"
                )
                for name in STREAM_NAMES
            ),
        },
        "formal_fingerprints": formal_fingerprints,
        "repeat_fingerprints": repeat_fingerprints,
        "repeat_elapsed_seconds": round(elapsed, 6),
        "exporter_output": details,
        "status": "passed" if passed else "failed",
    }
    if repeat_dir.exists():
        shutil.rmtree(repeat_dir)
    if repeat_root.exists() and not any(repeat_root.iterdir()):
        repeat_root.rmdir()
    return result


def _summary_markdown(
    report: dict[str, Any],
    determinism: dict[str, Any] | None,
) -> str:
    summary = report["summary"]
    lines = [
        "# 第 4 轮 ONI 无损导出",
        "",
        f"- 状态：`{report['status']}`",
        (
            f"- 完成记录：`{summary['complete_count']}/"
            f"{summary['record_count']}`"
        ),
        f"- Color 帧：`{summary['color_frame_count']}`",
        f"- 16 位 Depth 帧：`{summary['depth_frame_count']}`",
        f"- 16 位 IR 帧：`{summary['ir_frame_count']}`",
        f"- 校验错误：`{summary['validation_error_count']}`",
        f"- 导出载荷字节（不含 metadata）：`{summary['total_export_payload_bytes']}`",
        "",
        "## 确定性",
        "",
    ]
    if determinism is None:
        lines.append("- 本次显式跳过重复导出检查。")
    else:
        lines.append(
            f"- `{determinism['record_id']}` 重复导出："
            f"`{determinism['status']}`"
        )
        for name, passed in determinism["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "## 数据边界",
            "",
            "- Depth/IR 使用逐帧 uint16 NPY，保留原始数值；MP4 仅为派生预览。",
            "- 当前 ONI 没有 Color，因此没有生成 color.mp4；Color 导出接口仍保留。",
            "- 未使用手机视频，也未建立 ONI—手机配对。",
            "- 导出完成不代表录制意图、主体运动者或使用授权已经人工确认。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else dataset_root / "manifests" / "oni_records.json"
    )
    audit_dir = (
        Path(args.audit_dir).resolve()
        if args.audit_dir
        else dataset_root / "reports" / "oni_audit" / "records"
    )
    exporter = Path(args.exporter).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else dataset_root / "extracted"
    )
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else dataset_root / "reports" / "oni_export"
    )
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not exporter.is_file():
        raise SystemExit(f"exporter not found: {exporter}")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    summaries: list[dict[str, Any]] = []
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        record_id = record["record_id"]
        record_dir = output_root / record_id
        print(
            f"[{index}/{len(records)}] exporting {record_id}: "
            f"{record['source_filename']}",
            flush=True,
        )
        if args.reuse_existing:
            return_code = 0
            elapsed = 0.0
        else:
            _safe_reset_record_dir(record_dir, output_root)
            return_code, elapsed, details = _run_exporter(
                exporter,
                dataset_root / record["source_file"],
                record_dir,
                timeout_seconds=args.timeout_seconds,
            )
            if return_code != 0:
                print(details, file=sys.stderr)
        audit = _load_audit(audit_dir, record_id)
        metadata = _finalize_record(
            record,
            audit,
            record_dir,
            return_code=return_code,
            elapsed_seconds=elapsed,
            create_previews=not args.skip_previews,
        )
        metadata_by_id[record_id] = metadata
        summaries.append(_summary_record(metadata))

    counts = aggregate_counts(summaries)
    counts["total_export_payload_bytes"] = sum(
        summary["export_payload_bytes"] for summary in summaries
    )
    passed = (
        counts["record_count"] == len(records)
        and counts["complete_count"] == len(records)
        and counts["validation_error_count"] == 0
    )
    batch_report = {
        "schema_version": 1,
        "artifact_type": "hyrox_oni_export_batch",
        "generated_at": utc_now(),
        "status": "passed" if passed else "failed",
        "source_manifest": str(manifest_path),
        "phone_pairing_used": False,
        "summary": counts,
        "records": summaries,
    }

    determinism: dict[str, Any] | None = None
    if not args.skip_determinism_check and records:
        determinism_record = min(
            records, key=lambda item: int(item["size_bytes"])
        )
        determinism = _determinism_check(
            determinism_record,
            dataset_root=dataset_root,
            exporter=exporter,
            output_root=output_root,
            report_dir=report_dir,
            formal_metadata=metadata_by_id[determinism_record["record_id"]],
            timeout_seconds=args.timeout_seconds,
        )
        if determinism["status"] != "passed":
            batch_report["status"] = "failed"
    write_json(report_dir / "batch_report.json", batch_report)
    if determinism is not None:
        write_json(
            report_dir / "determinism_report.json", determinism
        )
    (report_dir / "export_summary.md").write_text(
        _summary_markdown(batch_report, determinism),
        encoding="utf-8",
    )
    print(
        f"Round 4 {batch_report['status']}: "
        f"{counts['complete_count']}/{counts['record_count']} complete, "
        f"depth={counts['depth_frame_count']}, "
        f"ir={counts['ir_frame_count']}, "
        f"errors={counts['validation_error_count']}"
    )
    return 0 if batch_report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
