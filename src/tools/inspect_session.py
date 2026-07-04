from __future__ import annotations

import argparse
from pathlib import Path

from src.reference.quality import evaluate_quality, load_quality_rules
from src.reference.session_loader import available_numeric_fields, load_session, numeric_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a saved kinematic session.")
    parser.add_argument("--session", required=True, help="Path to outputs/sessions/<session_id>.")
    parser.add_argument("--quality-config", default=None, help="Optional quality YAML path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = load_session(Path(args.session))
    rules = load_quality_rules(args.quality_config)
    quality = evaluate_quality(session.kinematics, session.landmarks, session.metadata, rules)
    fields = available_numeric_fields(session.kinematics)
    summary = numeric_summary(session.kinematics, fields[:12])
    metrics = quality.metrics

    print(f"Session: {session.session_id}")
    print(f"Duration: {metrics['motion_duration_ms'] / 1000.0:.2f} s")
    print(f"Frames: {metrics['frame_count']}")
    print(f"Pose valid ratio: {metrics['pose_valid_ratio']:.1%}")
    print(f"Landmark missing ratio: {metrics['landmark_missing_ratio']:.1%}")
    print(f"Estimated FPS: {metrics['frame_rate_estimate']:.2f}")
    print(f"Mirror known: {metrics['mirror_state_known']}")
    print(f"Resolution: {session.metadata.get('actual_resolution', 'unknown')}")
    print(f"Data quality: {quality.status}")
    if quality.warnings:
        for warning in quality.warnings:
            print(f"WARNING: {warning}")
    print("Available kinematic fields:")
    for field in fields:
        print(f"- {field}")
    print("Curve summary:")
    for field, values in summary.items():
        print(f"- {field}: mean={values['mean']} min={values['min']} max={values['max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

