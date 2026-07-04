from __future__ import annotations

import argparse
from pathlib import Path

from src.reference.compare import compare_reference_to_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a saved session clip with a personal reference action.")
    parser.add_argument("--session", required=True, help="Path to outputs/sessions/<candidate_session_id>.")
    parser.add_argument("--reference", required=True, help="Path to outputs/references/<reference_id>.")
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/comparisons")
    parser.add_argument("--target-length", type=int, default=100)
    parser.add_argument("--window-ratio", type=float, default=0.15)
    parser.add_argument("--canonical-side", choices=["left", "right"], default=None)
    parser.add_argument("--candidate-movement-side", choices=["left", "right", "bilateral", "unknown"], default="unknown")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = compare_reference_to_session(
        session_dir=Path(args.session),
        reference_dir=Path(args.reference),
        output_dir=Path(args.output_dir),
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        target_length=args.target_length,
        window_ratio=args.window_ratio,
        canonical_side=args.canonical_side,
        candidate_movement_side=args.candidate_movement_side,
        plot=not args.no_plot,
    )
    print(f"Comparison written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

