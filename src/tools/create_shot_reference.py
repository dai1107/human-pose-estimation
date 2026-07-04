from __future__ import annotations

import argparse
from pathlib import Path

from src.reference.library import create_reference_from_session
from src.reference.session_loader import write_csv_rows
from src.sports.basketball.chain_features import extract_chain_feature_rows
from src.sports.basketball.shot_clipper import clip_shot_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a personal basketball shooting reference clip.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--shot-type", required=True, choices=["set_shot", "jump_shot"])
    parser.add_argument("--shooting-side", required=True, choices=["right", "left"])
    parser.add_argument("--camera-view", required=True, choices=["side", "front_left", "front_right", "front", "unknown"])
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/basketball/references")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference_dir = create_reference_from_session(
        args.session,
        output_root=args.output_dir,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        name=args.name,
        action_type=f"basketball_{args.shot_type}",
        camera_view=args.camera_view,
        movement_side=args.shooting_side,
    )
    clip = clip_shot_session(
        args.session,
        args.shooting_side,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    write_csv_rows(reference_dir / "shot_features.csv", extract_chain_feature_rows(clip.frames))
    print(f"Shot reference created: {reference_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

