"""Execute round-seven people tracking, review, objects and ROI ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.manifest import utc_now
from tools.dataset.phone_rgb import _atomic_json
from tools.dataset.round7_objects import build_object_scene_visibility_report
from tools.dataset.round7_roi import build_roi_ablation_report
from tools.dataset.round7_tracking import (
    YoloPoseCandidateDetector,
    build_overview_sheets,
    build_record_review_sheet,
    build_target_lock_audit,
    create_initialization_proposals,
    scan_record_people,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Round 7: cache all person candidates, review target initialization, "
            "build object/scene candidates and compare target ROI with full-frame pose."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dataset-root", default="datasets/hyrox")
    parser.add_argument(
        "--manifest", default="datasets/hyrox/manifests/phone_records.json"
    )
    parser.add_argument("--data-roles", default="datasets/hyrox/manifests/data_roles_v1.json")
    parser.add_argument("--person-model", default="yolo11n-pose.pt")
    parser.add_argument("--pose-model", default="models/pose_landmarker_full.task")
    parser.add_argument("--device", default="")
    parser.add_argument("--detection-interval", type=int, default=5)
    parser.add_argument("--roi-frame-stride", type=int, default=1)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--roi-only", action="store_true")
    parser.add_argument("--skip-roi", action="store_true")
    parser.add_argument(
        "--approve-reviewed-proposals",
        action="store_true",
        help=(
            "Mark current proposals reviewed. Use only after viewing every review "
            "sheet; the reviewer identity and type are persisted."
        ),
    )
    parser.add_argument("--reviewer-id", default="")
    parser.add_argument(
        "--reviewer-type",
        default="ai_assisted_manual_visual_review",
        choices=("human", "ai_assisted_manual_visual_review"),
    )
    parser.add_argument(
        "--target-override",
        action="append",
        default=[],
        metavar="RECORD_ID=TRACK_ID",
        help="Override a proposed track after visual review; repeatable.",
    )
    parser.add_argument(
        "--target-segment",
        action="append",
        default=[],
        metavar="RECORD_ID=TRACK_ID:START-END",
        help=(
            "Map a source candidate segment to the canonical target after visual "
            "review; repeat for every contiguous segment of a split track."
        ),
    )
    return parser


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_overrides(values: Sequence[str]) -> dict[str, str]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"target override must be RECORD_ID=TRACK_ID: {value}")
        record_id, track_id = value.split("=", 1)
        if not record_id.strip() or not track_id.strip():
            raise ValueError(f"target override must be RECORD_ID=TRACK_ID: {value}")
        output[record_id.strip()] = track_id.strip()
    return output


def _parse_segment_overrides(
    values: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        try:
            record_id, remainder = value.split("=", 1)
            track_id, frame_range = remainder.rsplit(":", 1)
            start_text, end_text = frame_range.split("-", 1)
            segment = {
                "source_track_id": track_id.strip(),
                "start_frame": int(start_text),
                "end_frame": int(end_text),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "target segment must be RECORD_ID=TRACK_ID:START-END: "
                f"{value}"
            ) from exc
        if (
            not record_id.strip()
            or not segment["source_track_id"]
            or segment["start_frame"] < 0
            or segment["end_frame"] < segment["start_frame"]
        ):
            raise ValueError(
                "target segment must be RECORD_ID=TRACK_ID:START-END: "
                f"{value}"
            )
        output.setdefault(record_id.strip(), []).append(segment)
    return output


def _approve_initializations(
    payload: dict[str, Any],
    *,
    dataset_root: Path,
    reviewer_id: str,
    reviewer_type: str,
    overrides: Mapping[str, str],
    segment_overrides: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if not reviewer_id.strip():
        raise ValueError("--reviewer-id is required with --approve-reviewed-proposals")
    reviewed_at = utc_now()
    for item in payload.get("records") or []:
        record_id = str(item["record_id"])
        if record_id in overrides:
            item["selected_track_id"] = overrides[record_id]
        summary = _load_json(
            dataset_root / "tracks" / record_id / "candidate_scan_summary.json"
        )
        frame_count = int(summary["frame_count"])
        item["canonical_target_track_id"] = str(
            item.get("canonical_target_track_id") or "target_athlete_001"
        )
        if record_id in segment_overrides:
            segments = sorted(
                (dict(segment) for segment in segment_overrides[record_id]),
                key=lambda segment: int(segment["start_frame"]),
            )
            expected_start = 0
            for segment in segments:
                if int(segment["start_frame"]) != expected_start:
                    raise ValueError(
                        f"{record_id}: target segments must be contiguous from frame 0"
                    )
                expected_start = int(segment["end_frame"]) + 1
            if expected_start != frame_count:
                raise ValueError(
                    f"{record_id}: target segments must end at frame {frame_count - 1}"
                )
            item["source_track_segments"] = segments
            item["selected_track_id"] = str(segments[0]["source_track_id"])
            item["manual_reinitializations"] = [
                {
                    "frame": int(after["start_frame"]),
                    "before_source_track_id": str(before["source_track_id"]),
                    "after_source_track_id": str(after["source_track_id"]),
                    "before_frame_range": [
                        int(before["start_frame"]),
                        int(before["end_frame"]),
                    ],
                    "after_frame_range": [
                        int(after["start_frame"]),
                        int(after["end_frame"]),
                    ],
                    "reason": (
                        "same athlete manually reacquired after occlusion, scale "
                        "change, or candidate-track fragmentation"
                    ),
                }
                for before, after in zip(segments, segments[1:])
            ]
        else:
            selected_track_id = str(item["selected_track_id"])
            item["source_track_segments"] = [
                {
                    "source_track_id": selected_track_id,
                    "start_frame": 0,
                    "end_frame": max(0, frame_count - 1),
                }
            ]
            item["manual_reinitializations"] = []
        reviewed_frames = sorted(
            {
                0,
                max(0, frame_count // 2),
                max(0, frame_count - 1),
                *[int(value) for value in summary.get("review_event_frames") or []],
                *[
                    int(segment["start_frame"])
                    for segment in item.get("source_track_segments") or []
                ],
                *[
                    int(segment["end_frame"])
                    for segment in item.get("source_track_segments") or []
                ],
            }
        )
        item["selection_status"] = "manual_visual_review_completed"
        item["selected_by"] = reviewer_id
        item["reviewer_type"] = reviewer_type
        item["selected_at"] = reviewed_at
        item["reason"] = (
            "reviewed beginning, middle, end and all detector-proposed crossing/"
            "occlusion frames; selected the visible action-performing athlete"
        )
        item["reviewed_segments"] = [
            {"frame_index": frame, "review_sheet_checked": True}
            for frame in reviewed_frames
        ]
        item["automatic_proposal_accepted"] = (
            len(item.get("source_track_segments") or []) == 1
            and item["selected_track_id"]
            == summary.get("proposed_target_track_id")
        )
    payload["generated_at"] = reviewed_at
    payload["proposal_is_ground_truth"] = False
    payload["review_completed"] = True
    payload["reviewer_disclosure"] = {
        "reviewer_id": reviewer_id,
        "reviewer_type": reviewer_type,
        "independent_second_human_review_completed": False,
    }
    return payload


def _update_phone_manifest(
    manifest: dict[str, Any], target_audit: Mapping[str, Any], path: Path
) -> None:
    audit_by_id = {
        str(item["record_id"]): item for item in target_audit.get("records") or []
    }
    for record in manifest.get("records") or []:
        audit = audit_by_id[str(record["record_id"])]
        initialization = audit["initialization"]
        record["target_athlete"] = {
            "track_id": audit["target_track_id"],
            "selection_status": "selected",
            "selection_method": "round7_manual_visual_review_of_all_person_candidates",
            "selected_by": initialization.get("selected_by"),
            "reviewer_type": initialization.get("reviewer_type"),
            "selected_at": initialization.get("selected_at"),
            "selection_reason": initialization.get("reason"),
            "initialization_frame": initialization.get("initialization_frame"),
            "initialization_bbox_xyxy": initialization.get(
                "initialization_bbox_xyxy"
            ),
            "identity_switch_segments": [
                event
                for event in audit.get("events") or []
                if event.get("event_type") == "TARGET_IDENTITY_SWITCH"
            ],
            "manual_reinitializations": audit.get("manual_reinitializations") or [],
        }
        record["other_people_present"] = (
            "yes" if int(audit["other_people_frame_count"]) > 0 else "no_detected_candidate"
        )
        record["full_body_visible"] = (
            "target_bound_tracking_complete_pose_visibility_in_round8"
        )
        record["review_status"]["subject_identity"] = (
            "target_track_selected_subject_identity_pending"
        )
    manifest["generated_at"] = utc_now()
    manifest["round7_target_lock_status"] = (
        "complete_with_independent_second_human_review_pending"
    )
    _atomic_json(path, manifest)


def _scan(
    manifest: Mapping[str, Any],
    *,
    dataset_root: Path,
    model_path: Path,
    device: str,
    detection_interval: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detector = YoloPoseCandidateDetector(model_path, device=device)
    summaries = []
    sheets: list[Path] = []
    for index, record in enumerate(manifest.get("records") or [], start=1):
        print(f"[{index:02d}/30] scan people: {record['record_id']}", flush=True)
        summary = scan_record_people(
            record,
            dataset_root=dataset_root,
            detector=detector,
            detection_interval=detection_interval,
        )
        summaries.append(summary)
        sheet = build_record_review_sheet(
            record,
            summary,
            dataset_root=dataset_root,
            output_path=dataset_root
            / "reports"
            / "round7_review_sheets"
            / f"{record['record_id']}.jpg",
        )
        sheets.append(sheet)
    overview = build_overview_sheets(
        sheets,
        output_dir=dataset_root / "reports" / "round7_review_sheets",
    )
    proposals = create_initialization_proposals(
        manifest, summaries, dataset_root=dataset_root
    )
    proposals["overview_sheets"] = [
        path.relative_to(dataset_root).as_posix() for path in overview
    ]
    _atomic_json(
        dataset_root / "annotations" / "target_initializations_v1.json",
        proposals,
    )
    return summaries, proposals


def _summary_report(
    target: Mapping[str, Any] | None,
    objects: Mapping[str, Any] | None,
    roi: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "target_lock_report_present": target is not None,
        "object_scene_report_present": objects is not None,
        "roi_ablation_report_present": roi is not None,
        "all_30_target_initializations": bool(
            target and target.get("manual_visual_initialization_count") == 30
        ),
        "all_30_people_tracks": bool(
            target and target.get("checks", {}).get("all_30_records_scanned")
        ),
        "all_30_object_tracks": bool(
            objects
            and objects.get("checks", {}).get(
                "all_records_have_object_candidate_files"
            )
        ),
        "roi_default_safety_gate": bool(
            roi
            and roi.get("checks", {}).get(
                "roi_stays_disabled_unless_both_gates_pass"
            )
        ),
    }
    second_human_pending = bool(
        target
        and target.get("summary", {}).get("manual_review_disagreement_rate") is None
    )
    return {
        "schema_version": 1,
        "artifact_type": "round7_implementation_summary",
        "generated_at": utc_now(),
        "status": (
            "passed_with_independent_second_human_review_pending"
            if all(checks.values()) and second_human_pending
            else "passed"
            if all(checks.values())
            else "failed"
        ),
        "checks": checks,
        "target_lock_summary": target.get("summary") if target else None,
        "object_scene_summary": objects.get("summary") if objects else None,
        "roi_summary": roi.get("summary") if roi else None,
        "remaining_manual_gate": (
            "An independent second human reviewer is required to estimate identity-switch miss rate and inter-reviewer disagreement."
            if second_human_pending
            else None
        ),
        "default_runtime_changed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    dataset_root = _resolve(project_root, args.dataset_root).resolve()
    manifest_path = _resolve(project_root, args.manifest).resolve()
    roles_path = _resolve(project_root, args.data_roles).resolve()
    manifest = _load_json(manifest_path)
    roles = _load_json(roles_path)
    initialization_path = (
        dataset_root / "annotations" / "target_initializations_v1.json"
    )

    if args.roi_only:
        target = _load_json(dataset_root / "reports" / "target_lock_audit_v1.json")
        objects = _load_json(
            dataset_root / "reports" / "object_scene_visibility_v1.json"
        )
        roi = build_roi_ablation_report(
            manifest,
            roles,
            dataset_root=dataset_root,
            project_root=project_root,
            model=args.pose_model,
            frame_stride=args.roi_frame_stride,
        )
        summary = _summary_report(target, objects, roi)
        _atomic_json(dataset_root / "reports" / "round7_implementation_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] != "failed" else 1

    should_scan = not args.finalize_only and not initialization_path.is_file()
    if args.scan_only or should_scan:
        _, initialization = _scan(
            manifest,
            dataset_root=dataset_root,
            model_path=_resolve(project_root, args.person_model),
            device=args.device,
            detection_interval=args.detection_interval,
        )
        print(
            json.dumps(
                {
                    "status": "review_required",
                    "initializations": initialization_path.as_posix(),
                    "overview_sheets": initialization["overview_sheets"],
                    "next_command": (
                        "python -m tools.run_round7_tracking "
                        "--approve-reviewed-proposals --reviewer-id <id>"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    initialization = _load_json(initialization_path)
    if args.approve_reviewed_proposals:
        initialization = _approve_initializations(
            initialization,
            dataset_root=dataset_root,
            reviewer_id=args.reviewer_id,
            reviewer_type=args.reviewer_type,
            overrides=_parse_overrides(args.target_override),
            segment_overrides=_parse_segment_overrides(args.target_segment),
        )
        _atomic_json(initialization_path, initialization)
    if not initialization.get("review_completed"):
        raise SystemExit(
            "Target proposals require visual review. Inspect round7_review_sheets "
            "then pass --approve-reviewed-proposals --reviewer-id <id>."
        )

    target = build_target_lock_audit(
        manifest, initialization, dataset_root=dataset_root
    )
    _update_phone_manifest(manifest, target, manifest_path)
    objects = build_object_scene_visibility_report(
        manifest, dataset_root=dataset_root
    )
    roi = None
    if not args.skip_roi:
        roi = build_roi_ablation_report(
            manifest,
            roles,
            dataset_root=dataset_root,
            project_root=project_root,
            model=args.pose_model,
            frame_stride=args.roi_frame_stride,
        )
    summary = _summary_report(target, objects, roi)
    _atomic_json(dataset_root / "reports" / "round7_implementation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
