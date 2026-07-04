from __future__ import annotations

import argparse
from pathlib import Path

from src.reference.library import create_reference_from_reference_clips, create_reference_from_session, load_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a personal reference action from saved session clips.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", help="Path to outputs/sessions/<session_id>.")
    source.add_argument("--from-clips", nargs="+", help="Existing reference clip directories to aggregate.")
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--action-type", default="generic_motion")
    parser.add_argument("--camera-view", default="unknown", choices=["side", "front", "front_left", "front_right", "unknown"])
    parser.add_argument("--movement-side", default="unknown", choices=["left", "right", "bilateral", "unknown"])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-dir", default="outputs/references")
    parser.add_argument("--enable-mirror-canonicalization", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.from_clips:
        reference_dir = create_reference_from_reference_clips(
            [Path(path) for path in args.from_clips],
            output_root=args.output_dir,
            name=args.name,
            action_type=args.action_type,
            camera_view=args.camera_view,
            movement_side=args.movement_side,
        )
    else:
        reference_dir = create_reference_from_session(
            Path(args.session),
            output_root=args.output_dir,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            name=args.name,
            description=args.description,
            action_type=args.action_type,
            camera_view=args.camera_view,
            movement_side=args.movement_side,
            tags=args.tag,
            notes=args.notes,
            mirror_canonicalization_enabled=args.enable_mirror_canonicalization,
        )
    reference = load_reference(reference_dir)
    quality = reference.quality_summary.get("status", "UNKNOWN") if isinstance(reference.quality_summary, dict) else "UNKNOWN"
    print(f"Reference created: {reference_dir}")
    print(f"Reference ID: {reference.reference_id}")
    print(f"Data quality: {quality}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

