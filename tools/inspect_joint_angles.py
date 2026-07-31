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
    analyze_curve_latency,
    export_angle_curves,
    find_observation,
    load_report,
    normalize_joint_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect exported raw/smoothed 2D, raw/smoothed 3D and rule angles."
        )
    )
    parser.add_argument("report", type=Path, help="Web JSON report")
    parser.add_argument("--joint", required=True, help="Example: left_knee")
    parser.add_argument("--frame", type=int, help="Inspect one exact frame")
    parser.add_argument(
        "--csv",
        type=Path,
        help="Export the selected joint's complete angle curves",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the inspection result as JSON",
    )
    parser.add_argument("--max-lag-frames", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    joint = normalize_joint_name(args.joint)
    report = load_report(args.report)
    if args.csv is not None:
        export_angle_curves(report, args.csv, joint=joint)
    if args.frame is not None:
        result = find_observation(
            report,
            frame_index=args.frame,
            joint=joint,
        )
        if result is None:
            raise SystemExit(
                f"No {joint} observation at frame {args.frame}"
            )
    else:
        result = analyze_curve_latency(
            report,
            max_lag_frames=args.max_lag_frames,
        ).get(joint)
        if result is None:
            raise SystemExit(f"No observations for {joint}")
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
