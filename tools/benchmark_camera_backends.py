"""Benchmark local camera backends and persist the best device-specific choices."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.camera.backend_benchmark import (
    CAMERA_BENCHMARK_CONFIGS,
    DEFAULT_BACKEND_CACHE,
    benchmark_camera_matrix,
    save_backend_selections,
    supported_backend_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenCV camera backends on this device. This command opens the camera.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0.")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("default", "dshow", "msmf"),
        default=None,
        help="Backends to test. Windows default: default dshow msmf.",
    )
    parser.add_argument("--duration", type=float, default=3.0, help="Measured seconds per combination.")
    parser.add_argument("--warmup", type=float, default=0.5, help="Warm-up seconds per combination.")
    parser.add_argument(
        "--output",
        default="outputs/benchmarks/camera_backend_benchmark.json",
        help="Benchmark JSON output path.",
    )
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_BACKEND_CACHE),
        help="Device-local backend selection cache path.",
    )
    parser.add_argument(
        "--no-cache-update",
        action="store_true",
        help="Write the report without updating the startup selection cache.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration <= 0 or args.warmup < 0:
        raise SystemExit("--duration must be positive and --warmup must be non-negative")
    backends = tuple(args.backends or supported_backend_names())
    unsupported = set(backends) - set(supported_backend_names())
    if unsupported:
        raise SystemExit(f"unsupported camera backends on {sys.platform}: {sorted(unsupported)}")
    results = benchmark_camera_matrix(
        camera_index=args.camera,
        backends=backends,
        configs=CAMERA_BENCHMARK_CONFIGS,
        duration_seconds=args.duration,
        warmup_seconds=args.warmup,
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": sys.platform,
        "camera_index": args.camera,
        "sensor_to_photon_note": (
            "sensor_to_photon_ms is null unless supplied by an external physical measurement; "
            "software read timing is not a substitute."
        ),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.no_cache_update:
        save_backend_selections(results, args.cache)
    successes = sum(bool(item["opened"]) and item["successful_reads"] > 0 for item in results)
    print(f"Camera backend benchmark: {successes}/{len(results)} combinations returned frames")
    print(f"Report: {output_path}")
    if not args.no_cache_update:
        print(f"Selection cache: {args.cache}")
    return 0 if successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
