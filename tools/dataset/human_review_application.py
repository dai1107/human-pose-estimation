"""Validate and apply a single-reviewer quick-review bundle without promoting it to ground truth."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALL_AUTHORIZED_USES = (
    "internal_storage",
    "manual_annotation",
    "model_training",
    "model_validation",
    "model_testing",
    "internal_demo",
    "public_example",
    "external_reviewer_access",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"] if isinstance(item, dict)]
    return []


def _validate_review_record(
    review: dict[str, Any],
    manifest_record: dict[str, Any],
) -> dict[str, Any]:
    record_id = str(review.get("record_id", ""))
    if str(review.get("reviewer_type", "")).lower() != "human":
        raise ValueError(f"{record_id}: reviewer_type must be human")
    if str(review.get("reviewer_role", "")) != "reviewer_a":
        raise ValueError(f"{record_id}: this importer expects reviewer_a")
    quick = ((review.get("review") or {}).get("quick_review") or {})
    if not isinstance(quick, dict) or quick.get("status") != "complete":
        raise ValueError(f"{record_id}: quick review is not complete")
    if str(quick.get("action", "")) != str(manifest_record.get("action", "")):
        raise ValueError(f"{record_id}: reviewed action does not match the manifest")
    video = manifest_record.get("video") or {}
    frame_count = int(video.get("decoded_frame_count", 0) or 0)
    start = int(quick.get("usable_start_frame", -1))
    end = int(quick.get("usable_end_frame", -1))
    if frame_count <= 0 or not (0 <= start <= end < frame_count):
        raise ValueError(
            f"{record_id}: closed usable interval [{start}, {end}] is outside "
            f"0..{max(-1, frame_count - 1)}"
        )
    if str(quick.get("authorization", "")).lower() != "confirmed":
        raise ValueError(f"{record_id}: authorization is not confirmed")
    audit_log = review.get("audit_log")
    if not isinstance(audit_log, list) or not audit_log:
        raise ValueError(f"{record_id}: audit log is empty")
    revisions = [int(item.get("revision", -1)) for item in audit_log if isinstance(item, dict)]
    if revisions != list(range(1, len(revisions) + 1)):
        raise ValueError(f"{record_id}: audit revisions are not contiguous")
    saved_revision = int(review.get("revision", -1))
    if saved_revision not in {revisions[-1], revisions[-1] + 1}:
        raise ValueError(f"{record_id}: saved revision is inconsistent with the audit log")
    return quick


def _authorization_block(
    *,
    reviewer_id: str,
    applied_at: str,
    source_bundle: str,
) -> dict[str, Any]:
    return {
        "status": "confirmed",
        "recorded_by": reviewer_id,
        "authorized_uses": list(ALL_AUTHORIZED_USES),
        "confirmed_at": applied_at,
        "source": source_bundle,
        "notes": (
            "All enumerated uses confirmed by the user on 2026-07-26. "
            "Authorization does not waive annotation quality or identity gates."
        ),
    }


def _apply_manifest_authorization(
    path: Path,
    *,
    reviewer_id: str,
    applied_at: str,
    source_bundle: str,
) -> int:
    payload = load_json(path)
    raw_records = payload.get("records") if isinstance(payload, dict) else None
    records = (
        [record for record in raw_records if isinstance(record, dict)]
        if isinstance(raw_records, list)
        else []
    )
    for record in records:
        record["usage_authorization"] = _authorization_block(
            reviewer_id=reviewer_id,
            applied_at=applied_at,
            source_bundle=source_bundle,
        )
    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary["authorization_pending_count"] = 0
        summary["authorization_confirmed_count"] = len(records)
    write_json(path, payload)
    return len(records)


def _apply_data_role_authorization(path: Path) -> None:
    payload = load_json(path)
    assignments = payload.get("assignments")
    if isinstance(assignments, list):
        for assignment in assignments:
            if isinstance(assignment, dict):
                assignment["authorization_status"] = "confirmed_all_enumerated_uses"
    write_json(path, payload)


def apply_quick_review_bundle(
    dataset_root: str | Path,
    review_bundle: str | Path,
    *,
    authorize_oni: bool = True,
    waive_subject_identity_for_internal_record_grouping: bool = True,
) -> dict[str, Path]:
    root = Path(dataset_root)
    bundle = Path(review_bundle)
    handoff_path = bundle / "human_v1" / "reviewer_a" / "codex_handoff.json"
    record_dir = bundle / "human_v1" / "reviewer_a" / "records"
    protocol_path = bundle / "human_v1" / "protocol_frozen.json"
    handoff = load_json(handoff_path)
    protocol = load_json(protocol_path)
    manifest_path = root / "manifests" / "phone_records.json"
    manifest = load_json(manifest_path)
    manifest_records = _records(manifest)
    manifest_by_id = {
        str(record.get("record_id", "")): record for record in manifest_records
    }

    review_paths = sorted(record_dir.glob("*.json"))
    if len(review_paths) != len(manifest_records):
        raise ValueError(
            f"review bundle has {len(review_paths)} records; "
            f"phone manifest has {len(manifest_records)}"
        )
    if int(handoff.get("saved_record_count", -1)) != len(review_paths):
        raise ValueError("handoff saved_record_count does not match record files")

    applied_at = utc_now()
    applied_records: list[dict[str, Any]] = []
    for path in review_paths:
        review = load_json(path)
        record_id = str(review.get("record_id", ""))
        manifest_record = manifest_by_id.get(record_id)
        if manifest_record is None:
            raise ValueError(f"{record_id}: missing from phone manifest")
        quick = _validate_review_record(review, manifest_record)
        usable = str(quick.get("video_usability", "")).lower() == "usable"
        target_correct = str(quick.get("target_status", "")).lower() == "correct"
        interval_confirmed = usable and target_correct
        segment = {
            "timeline_label": "target_action",
            "action_type": str(quick["action"]),
            "start_frame": int(quick["usable_start_frame"]),
            "end_frame": int(quick["usable_end_frame"]),
            "interval_semantics": "closed",
            "annotator_id": str(review.get("reviewer_id", "quick_reviewer")),
            "annotator_type": "human_single_reviewer_quick_review",
            "review_status": "single_human_quick_review_complete",
            "human_confirmed": interval_confirmed,
            "reviewer_count": 1,
            "training_eligible": False,
        }
        applied_records.append(
            {
                "record_id": record_id,
                "action_type": str(quick["action"]),
                "source_filename": str(manifest_record.get("source_filename", "")),
                "reviewer_role": str(review.get("reviewer_role", "")),
                "reviewer_id": str(review.get("reviewer_id", "")),
                "review_revision": int(review.get("revision", 0)),
                "review_record_sha256": file_sha256(path),
                "human_confirmed_action_type": target_correct,
                "human_confirmed_usable_interval": interval_confirmed,
                "video_usability": str(quick.get("video_usability", "")),
                "overall_result": str(quick.get("overall_result", "")),
                "observability": str(quick.get("observability", "")),
                "equipment_visibility": str(
                    quick.get("equipment_visibility", "")
                ),
                "proposal_decision": str(quick.get("proposal_decision", "")),
                "notes": str(quick.get("notes", "")),
                "segments": [segment],
                "rep_annotations_supplied": bool(quick.get("reps")),
                "phase_error_intervals_supplied": bool(
                    quick.get("phase_error_intervals")
                ),
                "event_annotations_supplied": bool(quick.get("events")),
                "fine_annotation_complete": bool(quick.get("reps"))
                and bool(quick.get("phase_error_intervals"))
                and bool(quick.get("events")),
                "counts_as_reviewed_ground_truth": False,
                "training_eligible": False,
            }
        )

    applied_ids = {item["record_id"] for item in applied_records}
    if applied_ids != set(manifest_by_id):
        raise ValueError("review bundle and phone manifest record IDs differ")

    source_label = str(bundle.resolve())
    phone_authorized = _apply_manifest_authorization(
        manifest_path,
        reviewer_id="user_confirmation_via_reviewer_a_bundle",
        applied_at=applied_at,
        source_bundle=source_label,
    )
    oni_authorized = 0
    oni_manifest_path = root / "manifests" / "oni_records.json"
    if authorize_oni and oni_manifest_path.is_file():
        oni_authorized = _apply_manifest_authorization(
            oni_manifest_path,
            reviewer_id="user_confirmation_2026-07-26",
            applied_at=applied_at,
            source_bundle=source_label,
        )
    data_roles_path = root / "manifests" / "data_roles_v1.json"
    if data_roles_path.is_file():
        _apply_data_role_authorization(data_roles_path)

    review_dir = root / "reviews"
    overlay = {
        "schema_version": 1,
        "artifact_type": "human_quick_review_application_v1",
        "generated_at": applied_at,
        "protocol_version": protocol.get("protocol_version"),
        "source_bundle": source_label,
        "source_handoff_sha256": file_sha256(handoff_path),
        "source_protocol_sha256": file_sha256(protocol_path),
        "interval_semantics": "closed",
        "review_level": "single_human_review_sufficient_for_current_stage",
        "record_count": len(applied_records),
        "human_confirmed_action_record_count": sum(
            bool(item["human_confirmed_action_type"]) for item in applied_records
        ),
        "human_confirmed_interval_record_count": sum(
            bool(item["human_confirmed_usable_interval"])
            for item in applied_records
        ),
        "fine_annotation_complete_record_count": sum(
            bool(item["fine_annotation_complete"]) for item in applied_records
        ),
        "independent_human_reviewer_count": 1,
        "authorization": {
            "authorized_uses": list(ALL_AUTHORIZED_USES),
            "phone_record_count": phone_authorized,
            "oni_record_count": oni_authorized,
        },
        "subject_identity_policy": {
            "status": (
                "temporarily_waived_for_internal_record_grouped_experiments"
                if waive_subject_identity_for_internal_record_grouping
                else "pending"
            ),
            "manifest_subject_ids_modified": False,
            "release_identity_gate_passed": False,
            "grouping_fallback": "record_id",
        },
        "promotion_policy": {
            "reviewed_ground_truth": False,
            "training_eligible": False,
            "reason": (
                "Single review is sufficient for the current stage, but only "
                "records with complete rep/phase-error/event annotations may "
                "advance to the next annotation checks."
            ),
        },
        "source_proposal_artifacts_modified": False,
        "records": applied_records,
    }
    overlay_path = write_json(
        review_dir / "human_quick_review_application_v1.json",
        overlay,
    )
    progress = {
        "schema_version": 1,
        "artifact_type": "round9_human_review_progress_v1",
        "generated_at": applied_at,
        "status": "single_human_review_active_core_fine_annotation_and_oni_subject_review_pending",
        "record_count": len(applied_records),
        "human_confirmed_action_record_count": overlay[
            "human_confirmed_action_record_count"
        ],
        "human_confirmed_interval_record_count": overlay[
            "human_confirmed_interval_record_count"
        ],
        "fine_annotation_complete_record_count": overlay[
            "fine_annotation_complete_record_count"
        ],
        "independent_human_reviewer_count": 1,
        "authorization_confirmed_phone_record_count": phone_authorized,
        "authorization_confirmed_oni_record_count": oni_authorized,
        "subject_identity_temporarily_waived_for_internal_experiments": (
            waive_subject_identity_for_internal_record_grouping
        ),
        "release_gate_passed": False,
        "training_gate_passed": False,
        "review_policy": "single_human_review_sufficient_for_current_stage",
        "second_reviewer_required_for_current_stage": False,
        "remaining_blockers": [
            "core_rep_phase_event_error_fine_annotations_pending",
            "oni_depth_ir_subject_review_not_present_in_phone_review_bundle",
        ],
        "deferred_tasks": [
            "second_independent_human_review_and_agreement",
            "idle_transition_unknown_and_continuous_switch_data_collection",
            "automatic_action_recognition_training_and_gating",
        ],
        "review_overlay": str(overlay_path.relative_to(root)),
        "review_overlay_sha256": file_sha256(overlay_path),
    }
    progress_path = write_json(
        root / "reports" / "round9_human_review_progress_v1.json",
        progress,
    )
    return {
        "review_overlay": overlay_path,
        "round9_progress": progress_path,
        "phone_manifest": manifest_path,
        "oni_manifest": oni_manifest_path,
        "data_roles": data_roles_path,
    }


__all__ = ["ALL_AUTHORIZED_USES", "apply_quick_review_bundle"]
