"""CLI for the reproducible eight-video HYROX golden regression."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.golden_videos import build_report, evaluate_case, load_manifest
from src.backends.mediapipe_backend import MediaPipeBackend
from src.paths import resolve_asset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate bundled HYROX videos against fixed expected intervals.")
    parser.add_argument("--manifest", default="configs/hyrox_golden_videos.json")
    parser.add_argument("--case", action="append", default=[], help="Only run this case id; repeatable.")
    parser.add_argument("--model", default="", help="Override the pose model in the manifest.")
    parser.add_argument("--report", default="outputs/validation/hyrox_golden_report.json")
    parser.add_argument("--list", action="store_true", help="List configured cases without running inference.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, cases = load_manifest(args.manifest)
    if args.case:
        selected = set(args.case)
        unknown = selected - {case.case_id for case in cases}
        if unknown:
            raise ValueError(f"unknown golden case(s): {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.case_id in selected]
    if args.list:
        for case in cases:
            print(f"{case.case_id}: {case.video} -> {case.action} ({case.camera_view})")
        return 0
    observations = []
    selected_model = args.model or model
    backend = MediaPipeBackend(
        resolve_asset(selected_model),
        output_segmentation_masks=False,
    )
    try:
        for case in cases:
            print(f"running {case.case_id}: {case.video}")
            observations.append(evaluate_case(case, selected_model, backend=backend))
    finally:
        backend.close()
    report = build_report(cases, observations)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['status']}: {report['passed_count']}/{report['case_count']} cases; report={report_path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
