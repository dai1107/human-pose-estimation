"""Summarize internal timing captures and external high-speed video checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.latency_audit import external_sensor_to_photon_ms, summarize_latency_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a sensor-to-photon latency bottleneck report.")
    parser.add_argument("--mode", required=True, choices=("web-local", "web-server", "desktop"))
    parser.add_argument("--input", default="", help="JSON report containing samples or frames.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--recording-fps", type=float, default=0.0, help="External slow-motion recording FPS.")
    parser.add_argument("--physical-motion-frame", type=int, default=-1)
    parser.add_argument("--display-motion-frame", type=int, default=-1)
    return parser


def _samples(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("samples"), list):
        return [item for item in payload["samples"] if isinstance(item, dict)]
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []
    return [
        dict(item.get("latency", {}))
        for item in frames
        if isinstance(item, dict) and isinstance(item.get("latency"), dict)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples: list[dict[str, Any]] = []
    if args.input:
        samples = _samples(json.loads(Path(args.input).read_text(encoding="utf-8-sig")))
    external = None
    if args.recording_fps > 0 or args.physical_motion_frame >= 0 or args.display_motion_frame >= 0:
        external = {
            "recording_fps": args.recording_fps,
            "physical_motion_frame": args.physical_motion_frame,
            "display_motion_frame": args.display_motion_frame,
            "sensor_to_photon_ms": external_sensor_to_photon_ms(
                recording_fps=args.recording_fps,
                physical_motion_frame=args.physical_motion_frame,
                display_motion_frame=args.display_motion_frame,
            ),
        }
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "summary": summarize_latency_samples(samples),
        "external_sensor_to_photon": external,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
