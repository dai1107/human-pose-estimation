"""Round-two CLI: back up ONI files and build validated record manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.manifest import (
    build_dataset_manifest,
    validation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an independent read-only ONI backup, stable record IDs, "
            "SHA256 hashes, without modifying the independent phone manifest."
        )
    )
    parser.add_argument(
        "--source-dir",
        default="hyrox动作视频集（oni）",
        help="Directory containing the original ONI files.",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets/hyrox",
        help="Dataset root. Default: datasets/hyrox.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Do not copy ONI files. Intended only for dry-run diagnostics.",
    )
    parser.add_argument(
        "--leave-source-writable",
        action="store_true",
        help="Do not mark source ONI files read-only.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing manifest without rebuilding or copying.",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="With --validate-only, recompute every backup SHA256 and read-only check.",
    )
    return parser


def _write_summary(path: Path, report: dict[str, object]) -> None:
    summary = report["summary"]
    checks = report["checks"]
    lines = [
        "# 第 2 轮 ONI 数据清单验收",
        "",
        f"- 状态：`{report['status']}`",
        f"- ONI 记录：`{summary['record_count']}`",
        f"- 总字节数：`{summary['total_bytes']}`",
        f"- ONI manifest 内手机记录：`{summary['phone_record_count']}`（手机数据由独立 manifest 管理）",
        f"- ONI—手机配对：`{summary['paired_group_count']}`",
        "",
        "## 自动检查",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "## 待人工补充",
            "",
            "- `subject_id`：确认真实受试者后填写；",
            "- `usage_authorization`：确认数据授权范围；",
            "- `target_athlete`：在第 7 轮多人污染审计中锁定真正运动者；",
            "- 文件名中的标准/错误描述仍是未验证录制意图，不是训练真值。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root)
    if args.verify_files and not args.validate_only:
        raise SystemExit("--verify-files requires --validate-only")
    if args.validate_only:
        manifest_path = dataset_root / "manifests" / "oni_records.json"
        if not manifest_path.is_file():
            raise SystemExit(f"manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        payload = build_dataset_manifest(
            args.source_dir,
            dataset_root,
            project_root=PROJECT_ROOT,
            copy_files=not args.reference_only,
            mark_source_read_only=not args.leave_source_writable,
        )
    report = validation_report(
        payload,
        dataset_root=dataset_root,
        verify_files=bool(args.verify_files),
    )
    report_path = dataset_root / "reports" / "round2_dataset_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary(
        dataset_root / "reports" / "round2_test_summary.md",
        report,
    )
    summary = report["summary"]
    print(
        f"Round 2 {report['status']}: {summary['record_count']} ONI records, "
        f"{summary['backup_hash_verified_count']} backups hash-verified; "
        f"manifest={dataset_root / 'manifests' / 'oni_records.json'}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
