"""Run short smoke or formal 30/60 minute pose endurance validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.view_policy import CAMERA_VIEWS
from src.validation.endurance import (
    EnduranceThresholds,
    build_endurance_report,
    run_video_endurance,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loop a video and measure realtime endurance telemetry.")
    parser.add_argument("--video", default="HYROX视频/农夫行走.mp4")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--minutes", type=int, choices=(30, 60), help="Formal endurance duration.")
    duration.add_argument("--duration-seconds", type=float, help="Short smoke duration; overrides the 30-minute default.")
    parser.add_argument("--action", default="none", choices=("none", *HYROX_ACTION_NAMES))
    parser.add_argument("--camera-view", default="unknown", choices=CAMERA_VIEWS)
    parser.add_argument("--min-fps", type=float, default=1.0)
    parser.add_argument("--max-p95-latency-ms", type=float, default=1000.0)
    parser.add_argument("--max-memory-growth-mb", type=float, default=512.0)
    parser.add_argument("--max-read-failure-rate", type=float, default=0.01)
    parser.add_argument("--report", default="outputs/validation/endurance_report.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    duration_seconds = args.duration_seconds if args.duration_seconds is not None else float((args.minutes or 30) * 60)
    thresholds = EnduranceThresholds(
        min_fps=args.min_fps,
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_memory_growth_mb=args.max_memory_growth_mb,
        max_read_failure_rate=args.max_read_failure_rate,
    )
    observation = run_video_endurance(
        video=args.video,
        model=args.model,
        duration_seconds=duration_seconds,
        action=args.action,
        camera_view=args.camera_view,
    )
    report = build_endurance_report(observation, thresholds)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    report_path.write_text(serialized, encoding="utf-8")
    roundtrip = json.loads(report_path.read_text(encoding="utf-8"))
    if roundtrip.get("artifact_type") != "pose_endurance_report":
        print(f"failed: report roundtrip validation failed; report={report_path}")
        return 1
    observation_payload = report["observation"]
    print(
        f"{report['status']}: frames={observation_payload['total_frames']} "
        f"fps={observation_payload['average_fps']:.2f} "
        f"p95={observation_payload['p95_latency_ms']:.2f}ms "
        f"memory_growth={observation_payload['memory_growth_mb']:.2f}MB; report={report_path}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
