"""Execute the round-six phone RGB ingestion and baseline freeze."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.manifest import utc_now
from tools.dataset.phone_rgb import (
    _atomic_json,
    build_phone_rgb_manifest,
    validate_phone_manifest,
    write_round6_governance_reports,
)
from tools.dataset.round6_baselines import write_round6_baselines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest independent phone RGB videos and freeze round-six reports."
    )
    parser.add_argument("--source-dir", default="hyrox手机录像数据")
    parser.add_argument("--dataset-root", default="datasets/hyrox")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--latency-max-frames", type=int, default=300)
    parser.add_argument("--expected-videos", type=int, default=30)
    parser.add_argument("--expected-appledouble", type=int, default=30)
    parser.add_argument("--expected-total-frames", type=int, default=15515)
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Create copies, manifest, decode audit and governance reports without pose baselines.",
    )
    parser.add_argument(
        "--keep-source-writable",
        action="store_true",
        help="Do not mark original source videos read-only (backups are always read-only).",
    )
    return parser


def _acceptance_checks(
    manifest: dict[str, object],
    audit: dict[str, object],
    roles: dict[str, object],
    *,
    expected_videos: int,
    expected_appledouble: int,
    expected_total_frames: int,
) -> dict[str, bool]:
    summary = audit["summary"]
    assert isinstance(summary, dict)
    records = manifest["records"]
    assert isinstance(records, list)
    role_checks = roles["checks"]
    assert isinstance(role_checks, dict)
    return {
        "real_video_count": summary["real_video_count"] == expected_videos,
        "appledouble_count": summary["appledouble_file_count"] == expected_appledouble,
        "appledouble_record_count_zero": summary["appledouble_training_record_count"] == 0,
        "all_decode_to_declared_last_frame": summary["decoded_to_declared_last_frame_count"] == expected_videos,
        "total_frame_count": summary["decoded_frame_count_total"] == expected_total_frames,
        "canonical_source_type_only": all(record.get("source_type") == "phone_rgb" for record in records),
        "no_phone_oni_pairing": all(record.get("paired_group_id") is None for record in records),
        "all_three_way_hashes_match": summary["three_way_sha256_match_count"] == expected_videos,
        "data_roles_disjoint": bool(role_checks.get("no_silent_overlap")),
        "example_candidate_count": bool(role_checks.get("example_candidate_count_is_eight")),
        "training_and_golden_disabled": bool(role_checks.get("all_training_and_golden_disabled")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = project_root / source_dir
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    manifest, audit = build_phone_rgb_manifest(
        source_dir,
        dataset_root,
        project_root=project_root,
        mark_source_read_only=not args.keep_source_writable,
    )
    validation_errors = validate_phone_manifest(
        manifest, dataset_root=dataset_root, verify_files=True
    )
    roles, coverage, observability = write_round6_governance_reports(
        manifest, dataset_root
    )
    baselines: dict[str, object] = {"status": "skipped_by_ingest_only"}
    if not args.ingest_only:
        phone, coordinate, latency, intervals = write_round6_baselines(
            manifest,
            dataset_root,
            project_root=project_root,
            model=args.model,
            frame_stride=args.frame_stride,
            latency_max_frames=args.latency_max_frames,
        )
        roles, coverage, observability = write_round6_governance_reports(
            manifest, dataset_root, interval_candidates=intervals
        )
        baselines = {
            "phone_rgb_data_baseline": phone["summary"],
            "coordinate_baseline": coordinate["coordinate_spaces"],
            "realtime_latency_baseline": {
                "model_tiers": list(
                    latency["deterministic_phone_video"]["model_tiers"]
                ),
                "camera_chain_status": latency["camera_chain_snapshot"]["status"],
            },
        }

    checks = _acceptance_checks(
        manifest,
        audit,
        roles,
        expected_videos=args.expected_videos,
        expected_appledouble=args.expected_appledouble,
        expected_total_frames=args.expected_total_frames,
    )
    checks["manifest_validation"] = not validation_errors
    checks["decode_audit_status"] = audit["status"] == "passed"
    report = {
        "schema_version": 1,
        "artifact_type": "round6_implementation_summary",
        "generated_at": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "validation_errors": validation_errors,
        "summary": audit["summary"],
        "outputs": {
            "phone_manifest": "datasets/hyrox/manifests/phone_records.json",
            "data_roles": "datasets/hyrox/manifests/data_roles_v1.json",
            "decode_audit": "datasets/hyrox/reports/phone_rgb_decode_audit.json",
            "coverage": "datasets/hyrox/reports/coverage_gap_matrix_v1.json",
            "observability": "datasets/hyrox/reports/observability_gap_v1.json",
            "phone_baseline": "reports/baseline/phone_rgb_data_baseline_v1.json",
            "latency_baseline": "reports/baseline/realtime_latency_baseline_v1.json",
            "coordinate_baseline": "reports/baseline/coordinate_baseline_v1.json",
        },
        "baselines": baselines,
        "default_runtime_changed": False,
        "oni_phone_pair_count": 0,
    }
    output = dataset_root / "reports" / "round6_implementation_summary.json"
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
