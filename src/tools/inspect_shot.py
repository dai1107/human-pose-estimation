from __future__ import annotations

import argparse
from pathlib import Path

from src.sports.basketball.shot_clipper import detect_shot_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a saved session for possible basketball shot candidate clips.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--shooting-side", required=True, choices=["right", "left"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = detect_shot_candidates(Path(args.session), args.shooting_side)
    if not candidates:
        print("No shot candidates found. Manual clipping is recommended.")
        return 0
    for index, candidate in enumerate(candidates, start=1):
        print(f"SHOT CANDIDATE {index}: {candidate.start_ms / 1000.0:.2f}s - {candidate.end_ms / 1000.0:.2f}s confidence={candidate.confidence:.2f}")
    print("Candidates are suggestions only; confirm or modify clip boundaries before analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

