"""Replay reviewed Rowing/SkiErg pose caches and save deterministic count evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ACTIONS = {"rowing", "skierg"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_counting_report(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    manifest = _load(root / "manifests" / "phone_records.json")
    overlay = _load(root / "reviews" / "human_quick_review_application_v1.json")
    reviewed = {
        item["record_id"]: item
        for item in overlay.get("records", [])
        if isinstance(item, dict)
    }
    results: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        if not isinstance(record, dict) or record.get("action") not in TARGET_ACTIONS:
            continue
        record_id = str(record["record_id"])
        review = reviewed.get(record_id)
        if not isinstance(review, dict):
            continue
        pose_relative = (record.get("pose_cache") or {}).get("causal_analysis_pose")
        if not isinstance(pose_relative, str):
            raise ValueError(f"{record_id}: causal pose cache is missing")
        pose_path = root / pose_relative
        video = record.get("video") or {}
        analyzer = create_action_analyzer(
            str(record["action"]),
            camera_view=str(record.get("camera_view", "unknown")),
            live_mode=False,
        )
        count_events: list[dict[str, Any]] = []
        previous_count = 0
        final_state: dict[str, Any] = {}
        analyzed_frames = 0
        with gzip.open(pose_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                frame = json.loads(line)
                frame_index = int(frame.get("frame_index", -1))
                if not (
                    int(review["segments"][0]["start_frame"])
                    <= frame_index
                    <= int(review["segments"][0]["end_frame"])
                ):
                    continue
                eligible = bool(frame.get("formal_pose_eligible")) and bool(
                    frame.get("may_drive_rules_or_training")
                )
                features = (
                    extract_basic_pose_features(
                        frame.get("image_normalized_2d"),
                        int(video.get("width", 1) or 1),
                        int(video.get("height", 1) or 1),
                    )
                    if eligible
                    else None
                )
                final_state = analyzer.update(
                    features,
                    int(round(float(frame.get("source_timestamp_ms", 0.0)))),
                )
                analyzed_frames += 1
                current_count = int(final_state.get("rep_count", 0) or 0)
                if current_count > previous_count:
                    debug = final_state.get("debug") or {}
                    count_events.append(
                        {
                            "count": current_count,
                            "frame_index": frame_index,
                            "timestamp_ms": int(
                                round(float(frame.get("source_timestamp_ms", 0.0)))
                            ),
                            "terminal_phase": str(final_state.get("phase", "")),
                            "selected_pose_side": debug.get("selected_pose_side"),
                            "analysis_visible_score": debug.get(
                                "analysis_visible_score"
                            ),
                        }
                    )
                previous_count = current_count
        results.append(
            {
                "record_id": record_id,
                "action": record["action"],
                "camera_view": record.get("camera_view"),
                "human_review_overall_result": review.get("overall_result"),
                "human_review_notes": review.get("notes"),
                "usable_interval": review.get("segments", [None])[0],
                "analyzed_frame_count": analyzed_frames,
                "cycle_count": int(final_state.get("cycle_count", 0) or 0),
                "candidate_count": int(final_state.get("candidate_count", 0) or 0),
                "valid_count": int(final_state.get("rep_count", 0) or 0),
                "unsure_count": int(final_state.get("unsure_count", 0) or 0),
                "no_rep_count": int(final_state.get("no_rep_count", 0) or 0),
                "count_events": count_events,
                "official_rep_count_supported": bool(
                    final_state.get("official_rep_count_supported", False)
                ),
                "count_semantics": final_state.get("count_semantics"),
                "pose_cache": pose_relative,
                "pose_cache_sha256": _sha256(pose_path),
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "human_review_counting_regression_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "reviewed_video_cycle_count_evidence_available",
        "actions": sorted(TARGET_ACTIONS),
        "record_count": len(results),
        "truth_boundary": {
            "human_exact_rep_counts_supplied": False,
            "counts_claimed_as_ground_truth": False,
            "use": (
                "Regression and failure diagnosis only until a human supplies "
                "per-rep boundaries or exact counts."
            ),
        },
        "records": results,
    }
    output = root / "reports" / "human_review_counting_regression_v1.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay reviewed Rowing/SkiErg causal pose caches."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/hyrox"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.dataset_root
    if not root.is_absolute():
        root = args.project_root.resolve() / root
    output = build_counting_report(root)
    print(json.dumps({"report": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
