from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.angle_validation import (
    compare_manual_annotations,
    load_annotations,
    load_report,
    write_comparison_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare manual joint angles with raw/smoothed 2D, 3D and rule "
            "angles, including curve and event lag."
        )
    )
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Old-version frame report used for explicit non-regression comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/angle_validation"),
    )
    parser.add_argument("--max-lag-frames", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    annotations = load_annotations(args.annotations)
    report = load_report(args.report) if args.report else None
    baseline_report = (
        load_report(args.baseline_report) if args.baseline_report else None
    )
    summary, rows = compare_manual_annotations(
        annotations,
        report=report,
        baseline_report=baseline_report,
        max_lag_frames=args.max_lag_frames,
    )
    summary_path, rows_path = write_comparison_artifacts(
        args.output_dir,
        summary,
        rows,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "rows": str(rows_path),
                "annotation_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
