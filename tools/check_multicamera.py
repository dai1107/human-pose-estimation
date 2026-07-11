from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.multiview import MultiCameraCapture, MultiCameraPlan, parse_camera_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open one or more cameras together and verify timestamp skew.")
    parser.add_argument("--camera", action="append", required=True, help="Camera source as INDEX:VIEW[:mirror|no-mirror]. Repeat for multiple cameras.")
    parser.add_argument("--primary", type=int, default=None, help="Primary camera index. Default: first source.")
    parser.add_argument("--frames", type=int, default=30, help="Number of bundles to read. Default: 30.")
    parser.add_argument("--sync-tolerance-ms", type=int, default=50, help="Maximum acceptable per-bundle timestamp skew. Default: 50.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    try:
        sources = [parse_camera_source(spec) for spec in args.camera]
        plan = MultiCameraPlan.from_sources(
            sources,
            primary_camera_index=args.primary,
            synchronization_tolerance_ms=args.sync_tolerance_ms,
        )
        skews: list[int] = []
        synchronized = 0
        with MultiCameraCapture(plan, width=args.width, height=args.height, fps=args.fps) as capture:
            for _ in range(args.frames):
                bundle = capture.read()
                skews.append(bundle.skew_ms)
                synchronized += int(bundle.synchronized)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    payload = {
        "camera_count": len(plan.sources),
        "primary_camera_index": plan.primary_camera_index,
        "views": [source.camera_view for source in plan.sources],
        "has_front_and_side": plan.has_front_and_side,
        "frames": args.frames,
        "synchronized_frames": synchronized,
        "max_skew_ms": max(skews, default=0),
        "average_skew_ms": sum(skews) / len(skews) if skews else 0.0,
        "sync_tolerance_ms": plan.synchronization_tolerance_ms,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Cameras: {payload['camera_count']} views={','.join(payload['views'])}")
        print(f"Synchronized: {synchronized}/{args.frames}; max skew={payload['max_skew_ms']} ms")
    return 0 if synchronized == args.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
