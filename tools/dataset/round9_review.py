from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from .round9_annotations import (
    CORE_ACTIONS,
    ERROR_TO_CORRECTION,
    EVENTS,
    PHASES,
    _active_bounds,
    _hash_payload,
    _load_pose,
    _phase_segments,
    _read_json,
    _rep_anchors,
    _source_error_codes,
    _utc_now,
    _write_json,
)


def _nearest_anchor_deltas(left: Sequence[int], right: Sequence[int]) -> list[int]:
    available = list(right)
    deltas: list[int] = []
    for anchor in left:
        if not available:
            break
        match = min(available, key=lambda candidate: abs(candidate - anchor))
        deltas.append(abs(match - anchor))
        available.remove(match)
    return deltas


def _labels_for_record(
    record: dict[str, Any],
    core: dict[str, Any] | None,
    offline_anchors: Sequence[int],
) -> dict[int, list[str]]:
    labels: dict[int, list[str]] = {}
    for frame in (0, record["frame_count"] - 1):
        labels.setdefault(frame, []).append("video_boundary")
    for segment in record["segments"]:
        labels.setdefault(segment["start_frame"], []).append(
            f"{segment['timeline_label']}:start"
        )
        labels.setdefault(segment["end_frame"], []).append(
            f"{segment['timeline_label']}:end"
        )
    if core:
        for rep in core["reps"]:
            labels.setdefault(rep["start_frame"], []).append(f"{rep['rep_id']}:start")
            labels.setdefault(rep["end_frame"], []).append(f"{rep['rep_id']}:end")
            anchor = rep["phases"][0]["rep_anchor_frame"]
            labels.setdefault(anchor, []).append(f"{rep['rep_id']}:causal_anchor")
    for index, anchor in enumerate(offline_anchors, 1):
        labels.setdefault(anchor, []).append(f"offline_anchor_{index:02d}")
    return labels


def _read_frame(video_path: Path, frame_index: int) -> Any:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _tile(frame: Any, title: str, frame_index: int, width: int = 220) -> Any:
    import cv2
    import numpy as np

    if frame is None:
        canvas = np.zeros((410, width, 3), dtype=np.uint8)
        cv2.putText(canvas, "DECODE FAILED", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return canvas
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    banner = np.zeros((54, width, 3), dtype=np.uint8)
    cv2.putText(
        banner,
        f"f={frame_index}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    compact = title[:42]
    cv2.putText(
        banner,
        compact,
        (6, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((banner, resized))


def _sheet(
    video_path: Path,
    labels: dict[int, list[str]],
    output_path: Path,
    *,
    columns: int = 4,
) -> None:
    import cv2
    import numpy as np

    tiles = [
        _tile(_read_frame(video_path, frame), "|".join(labels[frame]), frame)
        for frame in sorted(labels)
    ]
    if not tiles:
        return
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * height : row * height + tile.shape[0], column * width : column * width + tile.shape[1]] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write review sheet: {output_path}")


def _overview(
    record_rows: Sequence[dict[str, Any]],
    project_root: Path,
    output_path: Path,
) -> None:
    import cv2
    import numpy as np

    width = 170
    blocks = []
    for row in record_rows:
        frames = [
            row["causal_active_bounds"][0],
            round(sum(row["causal_active_bounds"]) / 2),
            row["causal_active_bounds"][1],
        ]
        video_path = project_root / row["source_original"]
        tiles = [
            _tile(_read_frame(video_path, frame), row["record_id"], frame, width=width)
            for frame in frames
        ]
        height = max(tile.shape[0] for tile in tiles)
        block = np.zeros((height, width * 3, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            block[: tile.shape[0], index * width : (index + 1) * width] = tile
        blocks.append(block)
    total_height = sum(block.shape[0] for block in blocks)
    canvas = np.zeros((total_height, width * 3, 3), dtype=np.uint8)
    cursor = 0
    for block in blocks:
        canvas[cursor : cursor + block.shape[0], : block.shape[1]] = block
        cursor += block.shape[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write overview: {output_path}")


def build_multimethod_review(
    project_root: Path,
    dataset_root: Path,
    *,
    write_sheets: bool = True,
) -> dict[str, Any]:
    manifest = _read_json(dataset_root / "manifests" / "phone_records.json")
    action = _read_json(dataset_root / "annotations" / "action_segments_v1.json")
    core = _read_json(dataset_root / "annotations" / "core_rep_phase_event_error_v1.json")
    action_by_id = {row["record_id"]: row for row in action["records"]}
    core_by_id = {row["record_id"]: row for row in core["records"]}
    records: list[dict[str, Any]] = []
    active_deltas: list[int] = []
    anchor_deltas: list[int] = []
    exact_rep_count = 0
    sheets_root = dataset_root / "reports" / "round9_review_sheets"

    for source in manifest["records"]:
        record_id = source["record_id"]
        proposal = action_by_id[record_id]
        target_segment = next(
            segment for segment in proposal["segments"] if segment["timeline_label"] == "target_action"
        )
        causal_bounds = (target_segment["start_frame"], target_segment["end_frame"])
        offline = _load_pose(
            dataset_root / "pose_cache" / record_id / "offline_annotation_assist.jsonl.gz"
        )
        offline_start, offline_end, offline_gaps = _active_bounds(offline)
        offline_bounds = (offline_start, offline_end)
        bound_delta = abs(causal_bounds[0] - offline_start) + abs(causal_bounds[1] - offline_end)
        active_deltas.extend(
            [abs(causal_bounds[0] - offline_start), abs(causal_bounds[1] - offline_end)]
        )

        core_record = core_by_id.get(record_id)
        causal_anchors: list[int] = []
        offline_anchors: list[int] = []
        deltas: list[int] = []
        if core_record:
            causal_anchors = [
                rep["phases"][0]["rep_anchor_frame"] for rep in core_record["reps"]
            ]
            offline_anchors = _rep_anchors(
                source["action"], offline, offline_start, offline_end
            )
            deltas = _nearest_anchor_deltas(causal_anchors, offline_anchors)
            anchor_deltas.extend(deltas)
            if len(causal_anchors) == len(offline_anchors):
                exact_rep_count += 1

        row = {
            "record_id": record_id,
            "source_filename": source["source_filename"],
            "source_original": source["source_original"],
            "action": source["action"],
            "frame_count": source["video"]["decoded_frame_count"],
            "causal_active_bounds": list(causal_bounds),
            "offline_active_bounds": list(offline_bounds),
            "active_boundary_total_delta_frames": bound_delta,
            "offline_target_out_of_frame_candidates": offline_gaps,
            "causal_rep_anchor_count": len(causal_anchors),
            "offline_rep_anchor_count": len(offline_anchors),
            "matched_anchor_absolute_deltas_frames": deltas,
            "multimethod_review_priority": (
                "high"
                if bound_delta > 12
                or len(causal_anchors) != len(offline_anchors)
                or (deltas and max(deltas) > 10)
                else "normal"
            ),
            "visual_review_status": "pending",
        }
        records.append(row)
        if write_sheets:
            labels = _labels_for_record(proposal, core_record, offline_anchors)
            video_path = project_root / source["source_original"]
            _sheet(video_path, labels, sheets_root / f"{record_id}.jpg")

    if write_sheets:
        for index in range(0, len(records), 5):
            _overview(
                records[index : index + 5],
                project_root,
                sheets_root / f"overview_{index // 5 + 1:02d}.jpg",
            )

    return {
        "schema_version": 1,
        "artifact_type": "round9_multimethod_review_v1",
        "generated_at": _utc_now(),
        "status": "pose_crosscheck_complete_visual_review_pending",
        "methods": [
            "strictly_causal_pose_motion_and_cycle_proposals",
            "offline_centered_smoothing_independent_motion_and_cycle_proposals",
            "original_rgb_keyframe_visual_review",
            "mediapipe_lite_full_backend_disagreement_priority",
        ],
        "record_count": len(records),
        "core_record_count": len(core_by_id),
        "metrics": {
            "active_boundary_mean_absolute_delta_frames": (
                statistics.fmean(active_deltas) if active_deltas else None
            ),
            "active_boundary_max_absolute_delta_frames": max(active_deltas, default=None),
            "core_exact_rep_count_agreement_records": exact_rep_count,
            "core_anchor_mean_absolute_delta_frames": (
                statistics.fmean(anchor_deltas) if anchor_deltas else None
            ),
            "core_anchor_max_absolute_delta_frames": max(anchor_deltas, default=None),
        },
        "records": records,
    }


def write_multimethod_review(
    project_root: Path,
    dataset_root: Path,
    *,
    write_sheets: bool = True,
) -> dict[str, Any]:
    report = build_multimethod_review(
        project_root, dataset_root, write_sheets=write_sheets
    )
    _write_json(dataset_root / "reports" / "round9_multimethod_review_v1.json", report)
    return report


def _reviewed_segments(
    frame_count: int, start: int, end: int, reviewer_id: str
) -> list[dict[str, Any]]:
    rows = []
    if start:
        rows.append(("setup", 0, start - 1, 0.8))
    rows.append(("target_action", start, end, 0.95))
    if end < frame_count - 1:
        rows.append(("transition", end + 1, frame_count - 1, 0.85))
    return [
        {
            "timeline_label": label,
            "start_frame": segment_start,
            "end_frame": segment_end,
            "action_type": None,
            "action_confidence": confidence,
            "annotator_id": reviewer_id,
            "annotator_type": "ai_assisted_multimethod_visual_review",
            "review_status": "ai_second_pass_complete_human_confirmation_pending",
            "human_confirmed": False,
        }
        for label, segment_start, segment_end, confidence in rows
    ]


def _event_frame(
    action: str,
    event_index: int,
    rep_start: int,
    anchor: int,
    rep_end: int,
) -> int:
    if action == "lunge":
        return (rep_start, anchor, anchor, rep_end)[event_index]
    if action == "burpee_broad_jump":
        span = max(1, rep_end - anchor)
        return (
            max(rep_start, anchor - 12),
            anchor,
            min(rep_end, anchor + round(span * 0.45)),
            min(rep_end, anchor + round(span * 0.72)),
        )[event_index]
    return (
        rep_start,
        anchor,
        min(rep_end, anchor + max(8, round((rep_end - anchor) * 0.4))),
        rep_end,
    )[event_index]


def _reviewed_reps(
    source: dict[str, Any],
    anchors: Sequence[int],
    start: int,
    end: int,
    reviewer_id: str,
    intent_assessment: str,
) -> list[dict[str, Any]]:
    boundaries = [start]
    boundaries.extend(round((left + right) / 2) for left, right in zip(anchors, anchors[1:]))
    boundaries.append(end + 1)
    errors = _source_error_codes(source)
    fps = float(source["video"]["fps"])
    reps: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        rep_start, rep_end = boundaries[index], boundaries[index + 1] - 1
        phases = _phase_segments(PHASES[source["action"]], rep_start, anchor, rep_end)
        for phase in phases:
            phase["annotation_source"] = "ai_multimethod_review"
            phase["review_status"] = "ai_second_pass_complete_human_confirmation_pending"
        events = []
        for event_index, event_type in enumerate(EVENTS[source["action"]]):
            frame = _event_frame(
                source["action"], event_index, rep_start, anchor, rep_end
            )
            events.append(
                {
                    "event_type": event_type,
                    "frame_index": frame,
                    "timestamp_ms": round(frame * 1000.0 / fps, 3),
                    "observability": "RGB_AND_POSE_AI_REVIEW",
                    "evidence_source": "ai_multimethod_review",
                    "review_status": "ai_second_pass_complete_independent_human_review_pending",
                    "is_ground_truth": False,
                }
            )
        error_rows = []
        for error_code in errors:
            correction = ERROR_TO_CORRECTION.get(error_code)
            error_rows.append(
                {
                    "error_code": error_code,
                    "start_frame": rep_start,
                    "end_frame": rep_end,
                    "severity": "UNSURE",
                    "affected_side": "unknown",
                    "confidence": 0.65 if "visual_support" in intent_assessment else 0.5,
                    "criterion_id": f"{source['action']}.{error_code.lower()}.round9_trial_v1",
                    "phase": correction[2] if correction else "unknown",
                    "measured_value": None,
                    "unit": None,
                    "pass_range": None,
                    "fail_range": None,
                    "observability": "AI_REVIEWED_NOT_HUMAN_CONFIRMED",
                    "evidence_source": "filename_prior_plus_rgb_pose_review",
                    "annotator_id": reviewer_id,
                    "review_status": "ai_second_pass_complete_human_adjudication_pending",
                    "intent_assessment": intent_assessment,
                    "is_ground_truth": False,
                }
            )
        reps.append(
            {
                "rep_id": f"{source['record_id']}_rep_{index + 1:03d}",
                "start_frame": rep_start,
                "end_frame": rep_end,
                "validity": "UNSURE",
                "target_track_id": source["target_athlete"]["track_id"],
                "phases": phases,
                "events": events,
                "errors": error_rows,
                "review_status": "ai_second_pass_complete_human_frame_review_pending",
                "training_eligible": False,
            }
        )
    return reps


def apply_ai_review_decisions(
    project_root: Path,
    dataset_root: Path,
    decisions_path: Path,
) -> dict[str, Any]:
    decisions = _read_json(decisions_path)
    if decisions.get("human_reviewer") is not False:
        raise ValueError("This importer is only for the declared AI review pass")
    manifest = _read_json(dataset_root / "manifests" / "phone_records.json")
    action = _read_json(dataset_root / "annotations" / "action_segments_v1.json")
    core = _read_json(dataset_root / "annotations" / "core_rep_phase_event_error_v1.json")
    objects = _read_json(dataset_root / "annotations" / "object_scene_evidence_v1.json")
    scoring = _read_json(dataset_root / "annotations" / "scoring_correction_v1.json")
    review = _read_json(dataset_root / "reports" / "round9_multimethod_review_v1.json")
    source_by_id = {row["record_id"]: row for row in manifest["records"]}
    action_by_id = {row["record_id"]: row for row in action["records"]}
    core_by_id = {row["record_id"]: row for row in core["records"]}
    object_by_id = {row["record_id"]: row for row in objects["records"]}
    scoring_by_id = {row["record_id"]: row for row in scoring["records"]}
    reviewer_id = decisions["reviewer_id"]
    modified_boundaries = 0
    modified_core_anchors = 0

    for decision in decisions["records"]:
        record_id = decision["record_id"]
        source = source_by_id[record_id]
        action_row = action_by_id[record_id]
        start, end = map(int, decision["target_action_bounds"])
        if not (0 <= start <= end < action_row["frame_count"]):
            raise ValueError(f"{record_id}: invalid reviewed action bounds")
        old_target = next(
            segment
            for segment in action_row["segments"]
            if segment["timeline_label"] == "target_action"
        )
        if [old_target["start_frame"], old_target["end_frame"]] != [start, end]:
            modified_boundaries += 1
        action_row["segments"] = _reviewed_segments(
            action_row["frame_count"], start, end, reviewer_id
        )
        for segment in action_row["segments"]:
            segment["action_type"] = source["action"]
        action_row["action_type_ai_confirmed"] = bool(decision["action_confirmed"])
        action_row["intent_assessment"] = decision["intent_assessment"]
        action_row["video_action_review_status"] = (
            "ai_multimethod_second_pass_complete_human_confirmation_pending"
        )
        action_row["training_eligible"] = False

        if source["action"] in CORE_ACTIONS:
            anchors = [int(frame) for frame in decision["rep_anchors"]]
            old = core_by_id[record_id]
            old_anchors = [
                rep["phases"][0]["rep_anchor_frame"] for rep in old["reps"]
            ]
            if old_anchors != anchors:
                modified_core_anchors += 1
            old["reps"] = _reviewed_reps(
                source,
                anchors,
                start,
                end,
                reviewer_id,
                decision["intent_assessment"],
            )
            old["core_annotation_status"] = (
                "ai_multimethod_second_pass_complete_human_ground_truth_pending"
            )

        visibility = decisions["object_visibility"].get(source["action"], {})
        for evidence in object_by_id[record_id]["evidence"]:
            status = visibility.get(evidence["object_class"], "unknown")
            evidence["object_visible"] = status
            evidence["observability"] = (
                "OBSERVABLE_AI_REVIEWED"
                if status == "visible"
                else "UNOBSERVABLE"
                if status == "unobservable"
                else "PARTIALLY_OBSERVABLE_NO_READING"
                if status == "visible_reading_unobservable"
                else "UNKNOWN"
            )
            evidence["evidence_source"] = "original_rgb_ai_multimethod_review"
            evidence["review_status"] = (
                "ai_second_pass_complete_human_confirmation_pending"
            )
            evidence["rule_truth_generated"] = False

        if record_id in scoring_by_id:
            scoring_by_id[record_id]["review_status"] = (
                "ai_multimethod_schema_review_complete_two_human_experts_pending"
            )

    action["status"] = "ai_multimethod_second_pass_complete_human_confirmation_pending"
    action["ai_confirmed_record_count"] = len(decisions["records"])
    action["human_confirmed_record_count"] = 0
    core["status"] = "ai_multimethod_second_pass_complete_ground_truth_pending"
    core["rep_proposal_count"] = sum(len(row["reps"]) for row in core["records"])
    objects["status"] = "ai_multimethod_visibility_review_complete_human_confirmation_pending"
    scoring["status"] = "ai_schema_review_complete_dual_human_expert_scoring_pending"
    now = _utc_now()
    for payload in (action, core, objects, scoring):
        payload["updated_at"] = now

    _write_json(dataset_root / "annotations" / "action_segments_v1.json", action)
    _write_json(
        dataset_root / "annotations" / "core_rep_phase_event_error_v1.json", core
    )
    _write_json(dataset_root / "annotations" / "object_scene_evidence_v1.json", objects)
    _write_json(dataset_root / "annotations" / "scoring_correction_v1.json", scoring)

    agreement = _read_json(dataset_root / "reports" / "annotation_agreement_v1.json")
    agreement.update(
        {
            "updated_at": now,
            "status": "two_ai_method_passes_complete_independent_human_review_pending",
            "ai_assisted_reviewer_count": 2,
            "eligible_reviewer_count": 0,
            "ai_method_agreement": review["metrics"],
            "event_anchor_agreement": None,
            "error_label_agreement": None,
            "scoring_agreement": None,
            "release_gate_passed": False,
        }
    )
    _write_json(dataset_root / "reports" / "annotation_agreement_v1.json", agreement)

    proposal = _read_json(
        dataset_root / "reports" / "proposal_acceptance_bias_v1.json"
    )
    proposal.update(
        {
            "updated_at": now,
            "status": "ai_second_pass_decisions_recorded_human_decisions_pending",
            "ai_action_boundary_modified_records": modified_boundaries,
            "ai_core_anchor_modified_records": modified_core_anchors,
            "accepted": None,
            "modified": None,
            "rejected": None,
            "performance_evaluation_allowed": False,
        }
    )
    _write_json(
        dataset_root / "reports" / "proposal_acceptance_bias_v1.json", proposal
    )

    queue = _read_json(
        dataset_root / "reports" / "round9_active_review_queue_v1.json"
    )
    queue.update(
        {
            "updated_at": now,
            "status": "ai_multimethod_review_complete_human_review_pending",
            "ai_reviewed_record_count": len(decisions["records"]),
        }
    )
    for row in queue["records"]:
        row["ai_review_status"] = "complete"
        row["review_status"] = "independent_human_review_pending"
    _write_json(
        dataset_root / "reports" / "round9_active_review_queue_v1.json", queue
    )

    review.update(
        {
            "updated_at": now,
            "status": "ai_multimethod_visual_review_complete",
            "visual_reviewed_record_count": len(decisions["records"]),
            "core_visual_reviewed_record_count": sum(
                1 for row in decisions["records"] if "rep_anchors" in row
            ),
            "action_boundary_modified_records": modified_boundaries,
            "core_anchor_modified_records": modified_core_anchors,
            "reviewer_id": reviewer_id,
            "reviewer_type": decisions["reviewer_type"],
            "counts_as_independent_human_review": False,
        }
    )
    for row in review["records"]:
        row["visual_review_status"] = "ai_multimethod_second_pass_complete"
    _write_json(
        dataset_root / "reports" / "round9_multimethod_review_v1.json", review
    )

    summary = _read_json(
        dataset_root / "reports" / "round9_implementation_summary.json"
    )
    summary.update(
        {
            "updated_at": now,
            "status": "round9_ai_assisted_pilot_complete_human_release_gate_pending",
            "core_rep_proposal_count": core["rep_proposal_count"],
            "ai_multimethod_reviewed_record_count": len(decisions["records"]),
            "ai_multimethod_reviewed_core_record_count": sum(
                1 for row in decisions["records"] if "rep_anchors" in row
            ),
            "action_boundary_modified_records": modified_boundaries,
            "core_anchor_modified_records": modified_core_anchors,
            "human_confirmed_record_count": 0,
            "independent_human_reviewer_count": 0,
            "training_eligible_record_count": 0,
            "release_gate_passed": False,
            "release_blockers": [
                "two independent human reviewers have not supplied annotations",
                "usage authorization and subject identity remain pending",
            ],
        }
    )
    hashed_artifacts = [
        dataset_root / "annotations" / "action_segments_v1.json",
        dataset_root / "annotations" / "core_rep_phase_event_error_v1.json",
        dataset_root / "annotations" / "object_scene_evidence_v1.json",
        dataset_root / "annotations" / "scoring_correction_v1.json",
        dataset_root / "reports" / "annotation_agreement_v1.json",
        dataset_root / "reports" / "proposal_acceptance_bias_v1.json",
        dataset_root / "reports" / "continuous_ood_gap_v1.json",
        dataset_root / "reports" / "round9_active_review_queue_v1.json",
        dataset_root / "reports" / "round9_multimethod_review_v1.json",
    ]
    summary["artifact_hashes"] = {
        path.name: _hash_payload(_read_json(path)) for path in hashed_artifacts
    }
    _write_json(
        dataset_root / "reports" / "round9_implementation_summary.json", summary
    )
    return summary
