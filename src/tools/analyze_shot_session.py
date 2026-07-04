from __future__ import annotations

import argparse
from pathlib import Path

from src.sports.basketball.report import analyze_shot_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a basketball shot clip from a saved pose session.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--shot-type", required=True, choices=["set_shot", "jump_shot"])
    parser.add_argument("--shooting-side", required=True, choices=["right", "left"])
    parser.add_argument("--camera-view", required=True, choices=["side", "front_left", "front_right", "front", "unknown"])
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--release-ms", type=int, default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output-dir", default="outputs/basketball/reports")
    parser.add_argument("--config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = analyze_shot_session(
        session_dir=Path(args.session),
        shot_type=args.shot_type,
        shooting_side=args.shooting_side,
        camera_view=args.camera_view,
        output_dir=Path(args.output_dir),
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        release_ms=args.release_ms,
        reference_dir=Path(args.reference) if args.reference else None,
        config_path=Path(args.config) if args.config else None,
    )
    print(f"Basketball shot report written: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

