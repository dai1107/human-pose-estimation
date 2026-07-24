"""Freeze the current RGB rule/DTW/3D Assist implementation baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends.mediapipe_backend import MediaPipeBackend
from src.validation.baseline import (
    DEFAULT_TAG_SUGGESTION,
    GoldenTraceCollector,
    build_schema_snapshot,
    collect_environment,
    golden_video_inventory,
    snapshot_configs,
    utc_now,
    write_json,
)
from src.validation.golden_videos import (
    build_report,
    evaluate_case,
    load_manifest,
)
from tools.benchmark_latency_baseline import run_baseline


def run_golden_baseline(
    manifest_path: str | Path,
    *,
    model_override: str = "",
) -> dict[str, Any]:
    model, cases = load_manifest(manifest_path)
    selected_model = model_override or model
    observations = []
    traces: dict[str, dict[str, Any]] = {}
    backend = MediaPipeBackend(selected_model, output_segmentation_masks=False)
    try:
        for case in cases:
            print(f"baseline golden {case.case_id}: {case.video}", flush=True)
            collector = GoldenTraceCollector()
            observations.append(
                evaluate_case(
                    case,
                    selected_model,
                    backend=backend,
                    frame_observer=collector.observe,
                )
            )
            traces[case.case_id] = collector.report()
    finally:
        backend.close()
    report = build_report(cases, observations)
    for record in report["cases"]:
        record["trace"] = traces[record["id"]]
    report.update(
        {
            "baseline_schema_version": 1,
            "generated_at": utc_now(),
            "model": selected_model,
            "video_inventory": golden_video_inventory(
                PROJECT_ROOT, [case.video for case in cases]
            ),
        }
    )
    return report


def build_latency_report(
    input_video: str | Path,
    *,
    model: str | Path,
    max_frames: int,
    warmup_frames: int,
    camera_report: str | Path | None = None,
) -> dict[str, Any]:
    deterministic = run_baseline(
        input_video,
        model=model,
        max_frames=max_frames,
        warmup_frames=warmup_frames,
    )
    camera_payload: object = None
    if camera_report and Path(camera_report).is_file():
        camera_payload = json.loads(Path(camera_report).read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "artifact_type": "phase_zero_latency_baseline",
        "generated_at": utc_now(),
        "deterministic_video_pipeline": deterministic,
        "camera_benchmark": camera_payload,
        "realtime_display_latency": {
            "pose_age_ms": None,
            "display_latency_ms": None,
            "sensor_to_photon_ms": None,
            "status": "external_measurement_required",
            "note": (
                "Camera exposure, driver buffering, compositor and physical display "
                "latency cannot be inferred from an offline video. Use the existing "
                "--latency-audit session plus a 120/240 FPS external recording on "
                "each target device."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze environment, configs, output schema, 8-video results and "
            "MediaPipe latency without enabling ONI or neural inference."
        )
    )
    parser.add_argument("--output-dir", default="reports/baseline")
    parser.add_argument(
        "--manifest", default="configs/hyrox_golden_videos.json"
    )
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--latency-video",
        default="HYROX视频/负重箭步蹲.mp4",
    )
    parser.add_argument("--latency-max-frames", type=int, default=133)
    parser.add_argument("--latency-warmup-frames", type=int, default=5)
    parser.add_argument("--camera-report", default="")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument(
        "--tag-suggestion", default=DEFAULT_TAG_SUGGESTION
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment(
        PROJECT_ROOT, tag_suggestion=args.tag_suggestion
    )
    environment["configuration_snapshot"] = snapshot_configs(
        PROJECT_ROOT, output_dir
    )
    write_json(output_dir / "environment.json", environment)

    golden_report: dict[str, Any] | None = None
    if not args.skip_golden:
        golden_report = run_golden_baseline(
            args.manifest, model_override=args.model
        )
        write_json(output_dir / "golden_results.json", golden_report)

    schema = build_schema_snapshot(golden_report)
    write_json(output_dir / "output_schema.json", schema)

    if not args.skip_latency:
        manifest_model, _ = load_manifest(args.manifest)
        latency_report = build_latency_report(
            args.latency_video,
            model=args.model or manifest_model,
            max_frames=args.latency_max_frames,
            warmup_frames=args.latency_warmup_frames,
            camera_report=args.camera_report or None,
        )
        write_json(output_dir / "latency_report.json", latency_report)

    status = "passed"
    if golden_report is not None and golden_report["status"] != "passed":
        status = "failed"
    print(f"Baseline freeze {status}: {output_dir}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
