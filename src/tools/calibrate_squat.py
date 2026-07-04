from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fitness.squat.calibration import calibrate_standing
from src.fitness.squat.phase_metrics import build_squat_frames_from_session
from src.fitness.squat.schema import load_squat_config
from src.reference.session_loader import load_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate standing squat calibration from the beginning of a saved session.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--camera-view", default="unknown", choices=["side", "front", "front_left", "front_right", "unknown"])
    parser.add_argument("--duration-ms", type=int, default=2500)
    parser.add_argument("--config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = load_session(Path(args.session))
    config = load_squat_config(args.config)
    frames = build_squat_frames_from_session(session)
    if frames:
        start_ts = frames[0].timestamp_ms
        frames = [frame for frame in frames if frame.timestamp_ms - start_ts <= args.duration_ms]
    calibration = calibrate_standing(
        frames,
        camera_view=args.camera_view,
        minimum_visibility=float(config.get("data_quality", {}).get("minimum_landmark_visibility", 0.65)),
    )
    print(json.dumps(calibration.to_dict(), indent=2, ensure_ascii=False))
    return 0 if calibration.status != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())

