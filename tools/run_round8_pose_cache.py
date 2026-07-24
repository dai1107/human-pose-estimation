"""Execute round-eight pose caches, coordinate layers and temporal audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.manifest import utc_now
from tools.dataset.phone_rgb import _atomic_json
from tools.dataset.round8_pose_cache import (
    approve_event_anchors,
    benchmark_model_mode_matrix,
    build_cache_reports,
    build_temporal_streams_and_report,
    cache_record,
    implementation_summary,
    propose_event_anchors,
    validate_round8_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Round 8: cache target-bound MediaPipe Lite/Full raw poses, build "
            "coordinate layers, separate causal/display/offline streams and "
            "audit joint latency/jitter."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dataset-root", default="datasets/hyrox")
    parser.add_argument(
        "--manifest", default="datasets/hyrox/manifests/phone_records.json"
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--anchors-only", action="store_true")
    parser.add_argument("--derive-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--skip-benchmark-matrix", action="store_true")
    parser.add_argument(
        "--approve-anchor-review",
        action="store_true",
        help=(
            "Approve the 30 initial event-time anchor review sheets for the "
            "round-eight provisional latency audit. Round-nine independent "
            "double review remains mandatory."
        ),
    )
    parser.add_argument("--reviewer-id", default="")
    parser.add_argument(
        "--reviewer-type",
        choices=("human", "ai_assisted_manual_visual_review"),
        default="ai_assisted_manual_visual_review",
    )
    return parser


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_summaries(
    manifest: dict[str, Any],
    *,
    dataset_root: Path,
    project_root: Path,
    force: bool,
) -> list[dict[str, Any]]:
    summaries = []
    records = manifest.get("records") or []
    for index, record in enumerate(records, start=1):
        summary_path = (
            dataset_root
            / "pose_cache"
            / str(record["record_id"])
            / "cache_summary.json"
        )
        if summary_path.is_file() and not force:
            print(
                f"[{index:02d}/{len(records):02d}] reuse round8 cache: "
                f"{record['record_id']}",
                flush=True,
            )
            summary = _load(summary_path)
        else:
            print(
                f"[{index:02d}/{len(records):02d}] cache Lite/Full: "
                f"{record['record_id']}",
                flush=True,
            )
            summary = cache_record(
                record,
                dataset_root=dataset_root,
                project_root=project_root,
            )
        summaries.append(summary)
    return summaries


def _update_manifest(manifest: dict[str, Any], path: Path, status: str) -> None:
    manifest["generated_at"] = utc_now()
    manifest["round8_pose_cache_status"] = status
    for record in manifest.get("records") or []:
        record["pose_cache"] = {
            "status": "complete",
            "backends": ["mediapipe_lite", "mediapipe_full"],
            "target_track_id": record.get("target_athlete", {}).get("track_id"),
            "raw_pose_paths": {
                backend: (
                    f"pose_cache/{record['record_id']}/{backend}/"
                    "raw_pose.jsonl.gz"
                )
                for backend in ("mediapipe_lite", "mediapipe_full")
            },
            "causal_analysis_pose": (
                f"pose_cache/{record['record_id']}/"
                "causal_analysis_pose.jsonl.gz"
            ),
            "low_latency_display_pose": (
                f"pose_cache/{record['record_id']}/"
                "low_latency_display_pose.jsonl.gz"
            ),
        }
    _atomic_json(path, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    dataset_root = _resolve(project_root, args.dataset_root).resolve()
    manifest_path = _resolve(project_root, args.manifest).resolve()
    manifest = _load(manifest_path)
    anchor_path = (
        dataset_root / "annotations" / "round8_event_time_anchors_v1.json"
    )
    backend_path = dataset_root / "reports" / "backend_agreement_v1.json"
    coordinate_path = dataset_root / "reports" / "coordinate_quality_v1.json"
    observability_path = (
        dataset_root / "reports" / "pose_observability_v1.json"
    )
    temporal_path = (
        dataset_root / "reports" / "joint_latency_jitter_v1.json"
    )

    if args.summary_only:
        backend = _load(backend_path)
        coordinate = _load(coordinate_path)
        observability = _load(observability_path)
        temporal = _load(temporal_path)
        integrity = validate_round8_artifacts(
            manifest, dataset_root=dataset_root
        )
        summary = implementation_summary(
            backend,
            coordinate,
            temporal,
            observability,
            dataset_root=dataset_root,
            integrity=integrity,
        )
        _update_manifest(manifest, manifest_path, str(summary["status"]))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] != "failed" else 1

    if not args.derive_only:
        summaries = _cache_summaries(
            manifest,
            dataset_root=dataset_root,
            project_root=project_root,
            force=args.force_cache,
        )
        backend, coordinate, observability = build_cache_reports(
            manifest,
            summaries,
            dataset_root=dataset_root,
        )
        if args.cache_only:
            print(
                json.dumps(
                    {
                        "status": "cache_complete",
                        "record_count": len(summaries),
                        "frame_count": backend["frame_count"],
                        "next_command": (
                            "python -m tools.run_round8_pose_cache "
                            "--anchors-only"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
    else:
        backend = _load(backend_path)
        coordinate = _load(coordinate_path)
        observability = _load(observability_path)

    if not anchor_path.is_file():
        anchors = propose_event_anchors(
            manifest,
            dataset_root=dataset_root,
        )
    else:
        anchors = _load(anchor_path)
    if args.anchors_only and not args.approve_anchor_review:
        print(
            json.dumps(
                {
                    "status": "anchor_visual_review_required",
                    "anchor_count": len(anchors.get("records") or []),
                    "review_sheets": (
                        "datasets/hyrox/reports/"
                        "round8_anchor_review_sheets/overview_*.jpg"
                    ),
                    "next_command": (
                        "python -m tools.run_round8_pose_cache --derive-only "
                        "--approve-anchor-review --reviewer-id <id>"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.approve_anchor_review:
        anchors = approve_event_anchors(
            anchors,
            reviewer_id=args.reviewer_id,
            reviewer_type=args.reviewer_type,
        )
        _atomic_json(anchor_path, anchors)
    if not str(anchors.get("status", "")).startswith(
        "first_visual_review_completed"
    ):
        print(
            json.dumps(
                {
                    "status": "anchor_visual_review_required",
                    "review_sheets": (
                        "datasets/hyrox/reports/"
                        "round8_anchor_review_sheets/overview_*.jpg"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    benchmark = (
        {}
        if args.skip_benchmark_matrix
        else benchmark_model_mode_matrix(
            manifest,
            dataset_root=dataset_root,
            project_root=project_root,
        )
    )
    temporal = build_temporal_streams_and_report(
        manifest,
        anchors,
        dataset_root=dataset_root,
        benchmark_matrix=benchmark,
    )
    integrity = validate_round8_artifacts(
        manifest, dataset_root=dataset_root
    )
    summary = implementation_summary(
        backend,
        coordinate,
        temporal,
        observability,
        dataset_root=dataset_root,
        integrity=integrity,
    )
    _update_manifest(manifest, manifest_path, str(summary["status"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
