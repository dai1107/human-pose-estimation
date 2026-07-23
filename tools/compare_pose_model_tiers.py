"""Compare MediaPipe Pose Lite and Full on all configured golden videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends.mediapipe_backend import MediaPipeBackend
from src.paths import resolve_asset
from src.validation.golden_videos import load_manifest
from src.validation.model_tiers import build_model_tier_report, compare_model_tiers_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Pose Landmarker Lite and Full on golden videos.")
    parser.add_argument("--manifest", default="configs/hyrox_golden_videos.json")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--full-model", default="models/pose_landmarker_full.task")
    parser.add_argument("--lite-model", default="models/pose_landmarker_lite.task")
    parser.add_argument("--report", default="outputs/validation/pose_model_tiers.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, cases = load_manifest(args.manifest)
    if args.case:
        selected = set(args.case)
        unknown = selected - {case.case_id for case in cases}
        if unknown:
            raise ValueError(f"unknown golden case(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.case_id in selected]
    full_backend = MediaPipeBackend(resolve_asset(args.full_model), output_segmentation_masks=False)
    lite_backend = MediaPipeBackend(resolve_asset(args.lite_model), output_segmentation_masks=False)
    records = []
    try:
        for case in cases:
            print(f"comparing {case.case_id}: {case.video}")
            records.append(compare_model_tiers_case(case, full_backend=full_backend, lite_backend=lite_backend))
    finally:
        full_backend.close()
        lite_backend.close()
    report = build_model_tier_report(records)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{report['status']}: {report['passed_count']}/{report['case_count']} cases; "
        f"lite_auto_approved={report['lite_auto_approved']}; report={report_path}"
    )
    return 0 if report["lite_auto_approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
