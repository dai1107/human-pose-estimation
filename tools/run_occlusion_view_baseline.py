"""Build phase-A occlusion/cross-view diagnostics and hard-case inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.occlusion_view import (
    build_hard_case_inventory,
    run_golden_diagnostics,
    write_failure_inventory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw/filtered landmarks, confidence, normalized motion, bone "
            "length, raw/stable phases, golden counts, and reviewed hard cases."
        )
    )
    parser.add_argument("--output-dir", default="outputs/occlusion_view_phase_a")
    parser.add_argument("--manifest", default="configs/hyrox_golden_videos.json")
    parser.add_argument("--model", default="")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--error-library",
        default="datasets/hyrox/reviews/error_library_v1/index.json",
    )
    parser.add_argument(
        "--phone-manifest", default="datasets/hyrox/manifests/phone_records.json"
    )
    parser.add_argument(
        "--review-root", default="datasets/hyrox/reviews/human_v1/reviewer_a/records"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_hard_case_inventory(
        args.error_library, args.phone_manifest, args.review_root
    )
    inventory = write_failure_inventory(output_dir, rows)
    print(f"hard-case inventory: {inventory['case_count']} -> {output_dir}")
    if args.inventory_only:
        return 0
    report = run_golden_diagnostics(
        args.manifest,
        output_dir,
        model_override=args.model,
        selected_cases=set(args.case) if args.case else None,
    )
    print(
        f"diagnostic baseline: {report['case_count']} cases; "
        f"golden={report['golden_regression_status']} -> {output_dir}"
    )
    return 0 if report["golden_regression_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
