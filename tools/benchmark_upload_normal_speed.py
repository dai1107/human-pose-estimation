from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webui.app import PoseStreamEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production upload-analysis path and fail when its "
            "processed FPS is below the source video FPS."
        )
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--action", required=True)
    parser.add_argument("--camera-view", default="side")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/upload_normal_speed"),
    )
    return parser


def run_benchmark(
    video: Path,
    *,
    action: str,
    camera_view: str,
    timeout_seconds: float,
    output_dir: Path,
) -> dict[str, object]:
    source = video.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    run_id = str(time.time_ns())
    run_dir = output_dir / run_id
    engine = PoseStreamEngine(
        output_dir=run_dir,
        pose_cache_root=run_dir / "cold_pose_cache",
    )
    engine.start(
        {
            "source_mode": "upload",
            "source_name": source.name,
            "video_path": str(source),
            "backend": "mediapipe",
            "action": action,
            "camera_view": camera_view,
            "sensitivity": "medium",
            "mirror": False,
            "landmark_profile": "full",
            "show_fingers": False,
            "generate_annotated_video": False,
            "bundled_sample_tracking": False,
            "delete_source_after": False,
        }
    )
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    try:
        while time.monotonic() < deadline:
            state = engine.snapshot()
            if not state.get("running"):
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(
                f"upload analysis exceeded {timeout_seconds:.1f} seconds"
            )
        state = engine.snapshot()
        analysis_report = engine.report()
    finally:
        engine.stop()
    if state.get("status") != "completed":
        raise RuntimeError(str(state.get("error") or state.get("status")))
    performance = dict(state.get("performance_summary") or {})
    angle_source_counts: Counter[str] = Counter()
    displayed_angle_count = 0
    for frame in analysis_report.get("frames", ()):
        if not isinstance(frame, dict):
            continue
        assessment = frame.get("assessment")
        if not isinstance(assessment, dict):
            continue
        for angle in assessment.get("angles", ()):
            if not isinstance(angle, dict):
                continue
            displayed_angle_count += 1
            angle_source_counts[str(angle.get("source", "unknown"))] += 1
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "upload_normal_speed_benchmark_v1",
        "video": str(source),
        "action": action,
        "camera_view": camera_view,
        "cold_pose_cache": True,
        "performance": performance,
        "displayed_angle_count": displayed_angle_count,
        "angle_source_counts": dict(sorted(angle_source_counts.items())),
        "passed": performance.get("normal_speed_analysis_passed") is True,
    }
    report_path = run_dir / "normal_speed_benchmark.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        args.video,
        action=args.action,
        camera_view=args.camera_view,
        timeout_seconds=args.timeout_seconds,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
