"""Round-three CLI: inspect all recorded ONI files without a camera."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.oni_audit import (
    audit_summary_markdown,
    build_batch_report,
    enrich_inventory,
    invalid_records,
    utc_now,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated C++/OpenNI2 inspector over the ONI manifest. "
            "No camera is opened and phone video is not paired or consulted."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/hyrox",
        help="Dataset root. Default: datasets/hyrox.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest path. Default: <dataset-root>/manifests/oni_records.json.",
    )
    parser.add_argument(
        "--inspector",
        default="tools/oni_bridge/oni-inspect.exe",
        help="Path to the isolated oni-inspect executable.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <dataset-root>/reports/oni_audit.",
    )
    parser.add_argument(
        "--record-id",
        action="append",
        dest="record_ids",
        help="Audit only this record ID. May be supplied more than once.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="Per-record timeout. Default: 900 seconds.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Rebuild aggregate reports from existing per-record JSON "
            "without replaying ONI files."
        ),
    )
    return parser


def _synthetic_failure_report(
    input_path: Path,
    *,
    size_bytes: int,
    message: str,
) -> dict[str, Any]:
    empty_stream = {
        "exists": False,
        "create_success": False,
        "start_success": False,
        "complete": False,
        "width": 0,
        "height": 0,
        "pixel_format": "UNKNOWN",
        "nominal_fps": 0,
        "expected_frame_count": 0,
        "actual_frame_count": 0,
        "first_timestamp_us": None,
        "last_timestamp_us": None,
        "first_frame_index": None,
        "last_frame_index": None,
        "timestamps_strictly_increasing": True,
        "frame_indices_continuous": True,
        "non_increasing_timestamp_count": 0,
        "frame_index_discontinuity_count": 0,
        "interval_p50_us": 0.0,
        "interval_p95_us": 0.0,
        "actual_fps": 0.0,
        "estimated_dropped_frames": 0,
        "abnormal_interval_count": 0,
        "decode_error_count": 1,
        "errors": [message],
    }
    depth_stream = dict(empty_stream)
    depth_stream["errors"] = [message]
    depth_stream["depth_quality"] = {
        "depth_scale_mm": 0.0,
        "total_pixel_count": 0,
        "valid_pixel_count": 0,
        "zero_pixel_count": 0,
        "zero_value_ratio": 0.0,
        "invalid_pixel_ratio": 0.0,
        "min_depth_raw": None,
        "max_depth_raw": None,
        "all_depth_invalid": True,
        "center_region": {
            "definition": "central_50_percent_width_and_height",
            "total_pixel_count": 0,
            "valid_pixel_count": 0,
            "zero_value_ratio": 0.0,
            "min_depth_raw": None,
            "p05_depth_raw": None,
            "p50_depth_raw": None,
            "p95_depth_raw": None,
            "max_depth_raw": None,
        },
    }
    return {
        "schema_version": 1,
        "artifact_type": "oni_inventory",
        "tool": {
            "name": "oni-inspect",
            "version": "unknown",
            "openni_version": "unknown",
            "offline_file_only": True,
        },
        "file": {
            "path": str(input_path),
            "size_bytes": size_bytes,
            "open_success": False,
            "complete_playback": False,
            "duration_us": 0,
            "decode_error_count": 1,
            "errors": [message],
        },
        "classification": {
            "code": "E",
            "description": "corrupt_or_unreadable",
            "qualified_for_rgbd": False,
        },
        "streams": {
            "color": dict(empty_stream),
            "depth": depth_stream,
            "ir": dict(empty_stream),
        },
    }


def _audit_record(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    inspector: Path,
    records_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    input_path = (dataset_root / record["source_file"]).resolve()
    output_path = records_dir / f"{record['record_id']}.json"
    started = time.perf_counter()
    return_code = -1
    try:
        completed = subprocess.run(
            [
                str(inspector),
                str(input_path),
                "--output",
                str(output_path.resolve()),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        return_code = completed.returncode
        if output_path.is_file():
            report = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            details = completed.stderr.strip() or completed.stdout.strip()
            message = (
                f"inspector_exit_{return_code}_without_report"
                + (f": {details}" if details else "")
            )
            report = _synthetic_failure_report(
                input_path,
                size_bytes=int(record["size_bytes"]),
                message=message,
            )
    except subprocess.TimeoutExpired:
        return_code = 124
        report = _synthetic_failure_report(
            input_path,
            size_bytes=int(record["size_bytes"]),
            message=f"inspector_timeout_after_{timeout_seconds}_seconds",
        )
    except (OSError, json.JSONDecodeError) as error:
        return_code = 125
        report = _synthetic_failure_report(
            input_path,
            size_bytes=int(record["size_bytes"]),
            message=f"inspector_execution_or_report_error: {error}",
        )
    elapsed = time.perf_counter() - started
    enrich_inventory(
        report,
        record,
        inspector_return_code=return_code,
        elapsed_seconds=elapsed,
    )
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else dataset_root / "manifests" / "oni_records.json"
    )
    inspector = Path(args.inspector).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else dataset_root / "reports" / "oni_audit"
    )
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not inspector.is_file():
        raise SystemExit(f"inspector not found: {inspector}")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    selected_ids = set(args.record_ids or [])
    if selected_ids:
        known_ids = {record["record_id"] for record in records}
        unknown_ids = selected_ids - known_ids
        if unknown_ids:
            raise SystemExit(
                "unknown record ID(s): " + ", ".join(sorted(unknown_ids))
            )
        records = [
            record
            for record in records
            if record["record_id"] in selected_ids
        ]
        manifest = dict(manifest)
        manifest["records"] = records

    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    reports: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if args.reuse_existing:
            record_report = records_dir / f"{record['record_id']}.json"
            if not record_report.is_file():
                raise SystemExit(
                    f"existing record report not found: {record_report}"
                )
            report = json.loads(
                record_report.read_text(encoding="utf-8")
            )
            if (
                report.get("dataset_record", {}).get("record_id")
                != record["record_id"]
            ):
                raise SystemExit(
                    f"record identity mismatch: {record_report}"
                )
            reports.append(report)
        else:
            print(
                f"[{index}/{len(records)}] auditing "
                f"{record['record_id']}: {record['source_filename']}",
                flush=True,
            )
            reports.append(
                _audit_record(
                    record,
                    dataset_root=dataset_root,
                    inspector=inspector,
                    records_dir=records_dir,
                    timeout_seconds=args.timeout_seconds,
                )
            )

    batch = build_batch_report(
        manifest,
        reports,
        started_at=started_at,
    )
    (output_dir / "batch_report.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    invalid = invalid_records(batch)
    (output_dir / "invalid_records.json").write_text(
        json.dumps(invalid, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "audit_summary.md").write_text(
        audit_summary_markdown(batch),
        encoding="utf-8",
    )
    summary = batch["summary"]
    print(
        f"Round 3 {batch['status']}: "
        f"{summary['audited_record_count']}/"
        f"{summary['expected_record_count']} audited; "
        f"classes={summary['classification_counts']}; "
        f"invalid={invalid['count']}"
    )
    return 0 if batch["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
