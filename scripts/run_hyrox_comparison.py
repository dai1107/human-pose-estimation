from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.output_schema import versioned_csv_row, versioned_payload

SCHEMES: dict[str, list[str]] = {
    "mediapipe": [
        "--backend",
        "mediapipe",
        "--smoothing",
        "one-euro",
    ],
    "yolo_roi_mediapipe": [
        "--backend",
        "mediapipe",
        "--person-detector",
        "yolo",
        "--fusion",
        "yolo-roi-mediapipe",
        "--detector-every-n",
        "5",
        "--bbox-expand",
        "1.25",
        "--bbox-smoothing",
        "0.6",
        "--smoothing",
        "one-euro",
    ],
    "yolo_roi_mediapipe_cpu": [
        "--backend",
        "mediapipe",
        "--person-detector",
        "yolo",
        "--fusion",
        "yolo-roi-mediapipe",
        "--detector-device",
        "cpu",
        "--detector-every-n",
        "5",
        "--bbox-expand",
        "1.25",
        "--bbox-smoothing",
        "0.6",
        "--smoothing",
        "one-euro",
    ],
    "yolo_roi_mediapipe_gpu": [
        "--backend",
        "mediapipe",
        "--person-detector",
        "yolo",
        "--fusion",
        "yolo-roi-mediapipe",
        "--detector-device",
        "0",
        "--detector-every-n",
        "5",
        "--bbox-expand",
        "1.25",
        "--bbox-smoothing",
        "0.6",
        "--smoothing",
        "one-euro",
    ],
    "yolo_pose": [
        "--backend",
        "yolo-pose",
        "--yolo-pose-model",
        "yolo11n-pose.pt",
        "--smoothing",
        "one-euro",
    ],
    "yolo_pose_cpu": [
        "--backend",
        "yolo-pose",
        "--yolo-pose-model",
        "yolo11n-pose.pt",
        "--yolo-device",
        "cpu",
        "--smoothing",
        "one-euro",
    ],
    "yolo_pose_gpu": [
        "--backend",
        "yolo-pose",
        "--yolo-pose-model",
        "yolo11n-pose.pt",
        "--yolo-device",
        "0",
        "--smoothing",
        "one-euro",
    ],
}

SUMMARY_FIELDS = [
    "timestamp",
    "video",
    "scheme",
    "status",
    "returncode",
    "duration_sec",
    "metrics_path",
    "record_path",
    "log_path",
    "total_frames",
    "success_rate",
    "num_keypoints",
    "avg_fps",
    "avg_inference_time_ms",
    "p95_inference_time_ms",
    "avg_end_to_end_latency_ms",
    "avg_keypoint_confidence",
    "missing_rate_shoulder",
    "missing_rate_hip",
    "missing_rate_knee",
    "missing_rate_ankle",
    "keypoint_jitter",
    "angle_jitter",
    "roi_success_rate",
    "avg_yolo_detection_time_ms",
    "fallback_to_full_frame_count",
    "bbox_reuse_count",
    "bbox_lost_count",
    "person_lost_count",
    "backend_device",
    "detector_device",
    "source_model_distribution",
    "error_tail",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HYROX backend comparison over input videos.")
    parser.add_argument("--video-dir", default="HYROX视频", help="Folder containing HYROX mp4 files.")
    parser.add_argument("--output-root", default="", help="Output root. Default: outputs/comparisons/hyrox_batch_<timestamp>.")
    parser.add_argument("--video", action="append", default=[], help="Specific video filename to run. Can be repeated.")
    parser.add_argument("--scheme", action="append", choices=tuple(SCHEMES), default=[], help="Specific scheme to run. Can be repeated.")
    parser.add_argument("--skip-record", action="store_true", help="Do not save annotated mp4 outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path("outputs") / "comparisons" / f"hyrox_batch_{timestamp}"
    metrics_dir = output_root / "metrics"
    records_dir = output_root / "records"
    logs_dir = output_root / "logs"
    for directory in (metrics_dir, records_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    video_dir = Path(args.video_dir)
    requested = set(args.video)
    videos = [video_dir / name for name in args.video] if requested else sorted(video_dir.glob("*.mp4"))
    schemes = args.scheme or list(SCHEMES)
    if not videos:
        raise RuntimeError(f"no mp4 videos found in {video_dir}")

    rows = read_existing_summary(output_root / "summary.csv")
    manifest: dict[str, Any] = versioned_payload("hyrox_comparison_manifest", {
        "created_at": timestamp,
        "video_dir": str(video_dir),
        "output_root": str(output_root),
        "schemes": schemes,
        "videos": [str(path) for path in videos],
    })
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for video in videos:
        for scheme in schemes:
            row = run_one(video, scheme, metrics_dir, records_dir, logs_dir, skip_record=args.skip_record)
            rows = [existing for existing in rows if not (existing.get("video") == row["video"] and existing.get("scheme") == row["scheme"])]
            rows.append(row)
            write_summary(output_root / "summary.csv", rows)
            print(f"{row['status']}: {row['video']} / {row['scheme']} ({row['duration_sec']}s)")
    return 0


def run_one(video: Path, scheme: str, metrics_dir: Path, records_dir: Path, logs_dir: Path, skip_record: bool) -> dict[str, str]:
    started = datetime.now()
    stem = video.stem
    metrics_path = metrics_dir / f"{stem}_{scheme}.csv"
    record_path = records_dir / f"{stem}_{scheme}.mp4"
    log_path = logs_dir / f"{stem}_{scheme}.log"
    if metrics_path.exists():
        metrics_path.unlink()
    if record_path.exists():
        record_path.unlink()

    command = [
        sys.executable,
        "main.py",
        "--headless",
        "--input-video",
        str(video),
        "--save-metrics",
        str(metrics_path),
        *SCHEMES[scheme],
    ]
    if not skip_record:
        command.extend(["--record", str(record_path)])

    completed = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True)
    output = completed.stdout + "\n" + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    duration = (datetime.now() - started).total_seconds()

    metrics = read_metrics(metrics_path)
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(metrics)
    row.update(
        {
            "timestamp": started.strftime("%Y-%m-%d %H:%M:%S"),
            "video": video.name,
            "scheme": scheme,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": str(completed.returncode),
            "duration_sec": f"{duration:.1f}",
            "metrics_path": str(metrics_path),
            "record_path": str(record_path) if not skip_record else "",
            "log_path": str(log_path),
            "error_tail": tail(output, 600) if completed.returncode != 0 else "",
        }
    )
    return row


def read_metrics(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[-1] if rows else {}


def read_existing_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[*SUMMARY_FIELDS, "schema_version", "program_version"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(versioned_csv_row(row))


def tail(text: str, max_chars: int) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return compact[-max_chars:]


if __name__ == "__main__":
    raise SystemExit(main())
