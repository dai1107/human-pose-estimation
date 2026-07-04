from __future__ import annotations

import argparse
from pathlib import Path

from src.fitness.squat.report import analyze_squat_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze squat repetitions from a saved kinematic session.")
    parser.add_argument("--session", required=True, help="Path to outputs/sessions/<session_id>.")
    parser.add_argument("--camera-view", default="unknown", choices=["side", "front", "front_left", "front_right", "unknown"])
    parser.add_argument("--output-dir", default="outputs/squat_reports")
    parser.add_argument("--config", default=None, help="Optional squat config YAML path.")
    parser.add_argument("--reference", default=None, help="Optional outputs/references/<squat_reference_id> path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = analyze_squat_session(
        session_dir=Path(args.session),
        camera_view=args.camera_view,
        output_dir=Path(args.output_dir),
        config_path=Path(args.config) if args.config else None,
        reference_dir=Path(args.reference) if args.reference else None,
    )
    print(f"Squat report written: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

