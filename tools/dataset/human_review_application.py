"""Validate and apply a single-reviewer quick-review bundle without promoting it to ground truth."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection


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

NONCORE_EVENT_FROM_PHASE = {
    "farmers_carry": {
        "carrying": ("carry_start", "start"),
        "rest": ("carry_stop", "start"),
    },
    "rowing": {
        "catch": ("catch_reached", "start"),
        "drive": ("drive_start", "start"),
        "finish": ("finish_reached", "start"),
        "recovery": ("recovery_end", "end"),
    },
    "skierg": {
        "top": ("top_reached", "start"),
        "pull_down": ("pull_start", "start"),
        "bottom": ("bottom_reached", "start"),
        "return": ("return_end", "end"),
    },
    "sled_push": {
        "drive": ("drive_start", "start"),
        "step": ("step_contact", "start"),
        "reset": ("drive_end", "start"),
    },
    "sled_pull": {
        "reach": ("reach_reached", "start"),
        "pull": ("pull_start", "start"),
        "recover": ("pull_finish", "start"),
    },
}

TEMPORARY_SUBJECT_SECOND_RECORDS = {
    "phone_skierg_002",
    "phone_sled_push_004",
    "phone_sled_push_005",
}


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
    if not isinstance(quick, dict) or quick.get("status") not in {
        "complete",
        "blocked",
    }:
        raise ValueError(f"{record_id}: quick review has no terminal decision")
    if quick.get("status") == "blocked" and str(
        quick.get("video_usability", "")
    ).lower() != "unusable":
        raise ValueError(
            f"{record_id}: blocked review must record video_usability=unusable"
        )
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
    if (
        not revisions
        or revisions[0] != 1
        or any(current <= previous for previous, current in zip(revisions, revisions[1:]))
    ):
        raise ValueError(f"{record_id}: audit revisions are not strictly increasing from 1")
    saved_revision = int(review.get("revision", -1))
    if saved_revision not in {revisions[-1], revisions[-1] + 1}:
        raise ValueError(f"{record_id}: saved revision is inconsistent with the audit log")
    return quick


def _closed_interval(
    item: dict[str, Any],
    *,
    record_id: str,
    label: str,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    try:
        start = int(item["start_frame"])
        end = int(item["end_frame"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{record_id}: {label} has invalid frame bounds") from exc
    if not (minimum <= start <= end <= maximum):
        raise ValueError(
            f"{record_id}: {label} closed interval [{start}, {end}] is outside "
            f"{minimum}..{maximum}"
        )
    return start, end


def _validate_fine_annotations(
    quick: dict[str, Any],
    *,
    record_id: str,
    maximum_frame: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate reviewer-supplied rep, phase/error and event annotations.

    The exported review UI stores closed intervals.  This validator deliberately
    checks the human result itself rather than trusting the AI proposal notes
    that may still be retained for audit provenance.
    """

    reps = [
        deepcopy(item)
        for item in (quick.get("reps") or [])
        if isinstance(item, dict)
    ]
    intervals = [
        deepcopy(item)
        for item in (quick.get("phase_error_intervals") or [])
        if isinstance(item, dict)
    ]
    events = [
        deepcopy(item)
        for item in (quick.get("events") or [])
        if isinstance(item, dict)
    ]
    if not (reps or intervals or events):
        return [], [], []
    if not (reps and intervals and events):
        raise ValueError(
            f"{record_id}: fine annotation requires reps, "
            "phase_error_intervals and events together"
        )

    rep_bounds: dict[str, tuple[int, int]] = {}
    previous_end: int | None = None
    for index, rep in enumerate(reps, start=1):
        rep_id = str(rep.get("rep_id", "")).strip()
        if not rep_id or rep_id in rep_bounds:
            raise ValueError(f"{record_id}: rep {index} has a missing or duplicate rep_id")
        start, end = _closed_interval(
            rep,
            record_id=record_id,
            label=rep_id,
            minimum=0,
            maximum=maximum_frame,
        )
        if previous_end is not None and start <= previous_end:
            raise ValueError(f"{record_id}: rep intervals overlap or are out of order")
        validity = str(rep.get("validity", "")).upper()
        if validity not in {"VALID", "NO_REP", "UNSURE"}:
            raise ValueError(f"{record_id}: {rep_id} has invalid validity {validity!r}")
        rep["validity"] = validity
        rep_bounds[rep_id] = (start, end)
        previous_end = end

    intervals_by_rep: dict[str, list[tuple[int, int]]] = {
        rep_id: [] for rep_id in rep_bounds
    }
    for index, interval in enumerate(intervals, start=1):
        rep_id = str(interval.get("rep_id", "")).strip()
        if rep_id not in rep_bounds:
            raise ValueError(
                f"{record_id}: phase/error interval {index} references unknown rep {rep_id!r}"
            )
        rep_start, rep_end = rep_bounds[rep_id]
        start, end = _closed_interval(
            interval,
            record_id=record_id,
            label=f"{rep_id} phase/error interval {index}",
            minimum=rep_start,
            maximum=rep_end,
        )
        existing = intervals_by_rep[rep_id]
        if existing and start <= existing[-1][1]:
            raise ValueError(f"{record_id}: {rep_id} phase/error intervals overlap")
        existing.append((start, end))
        if not str(interval.get("phase", "")).strip():
            raise ValueError(f"{record_id}: {rep_id} phase/error interval has no phase")
        if not str(interval.get("error_code", "")).strip():
            raise ValueError(f"{record_id}: {rep_id} phase/error interval has no error_code")

    for rep_id, bounds in rep_bounds.items():
        covered = intervals_by_rep[rep_id]
        if not covered:
            raise ValueError(f"{record_id}: {rep_id} has no phase/error intervals")
        has_boundary_gap = (
            covered[0][0] != bounds[0] or covered[-1][1] != bounds[1]
        )
        has_internal_gap = any(
            right[0] != left[1] + 1
            for left, right in zip(covered, covered[1:])
        )
        if (
            (has_boundary_gap or has_internal_gap)
            and not str(
                next(
                    item.get("phase_gap_reason", "")
                    for item in reps
                    if str(item.get("rep_id", "")) == rep_id
                )
            ).strip()
        ):
            raise ValueError(
                f"{record_id}: {rep_id} phase/error intervals contain an "
                "unexplained gap"
            )

    event_count_by_rep = {rep_id: 0 for rep_id in rep_bounds}
    for index, event in enumerate(events, start=1):
        rep_id = str(event.get("rep_id", "")).strip()
        if rep_id not in rep_bounds:
            raise ValueError(
                f"{record_id}: event {index} references unknown rep {rep_id!r}"
            )
        try:
            frame_index = int(event["frame_index"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{record_id}: event {index} has invalid frame_index") from exc
        rep_start, rep_end = rep_bounds[rep_id]
        if not rep_start <= frame_index <= rep_end:
            raise ValueError(
                f"{record_id}: event {index} frame {frame_index} is outside {rep_id}"
            )
        if not str(event.get("event_type", "")).strip():
            raise ValueError(f"{record_id}: event {index} has no event_type")
        event_count_by_rep[rep_id] += 1
    missing_events = sorted(
        rep_id for rep_id, count in event_count_by_rep.items() if count == 0
    )
    if missing_events:
        raise ValueError(
            f"{record_id}: reps without events: {', '.join(missing_events)}"
        )
    return reps, intervals, events


def _derive_noncore_events(
    action: str,
    reps: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive canonical events from human-reviewed phase boundaries."""

    rules = NONCORE_EVENT_FROM_PHASE.get(action)
    if not rules:
        return []
    intervals_by_rep: dict[str, list[dict[str, Any]]] = {}
    for interval in intervals:
        intervals_by_rep.setdefault(str(interval.get("rep_id", "")), []).append(
            interval
        )
    events: list[dict[str, Any]] = []
    for rep in reps:
        rep_id = str(rep.get("rep_id", ""))
        candidates: list[tuple[str, int]] = []
        if action == "farmers_carry":
            candidates.append(("monitor_start", int(rep["start_frame"])))
        for interval in intervals_by_rep.get(rep_id, []):
            rule = rules.get(str(interval.get("phase", "")))
            if not rule:
                continue
            event_type, position = rule
            frame_key = "start_frame" if position == "start" else "end_frame"
            candidates.append((event_type, int(interval[frame_key])))
        if action == "farmers_carry":
            candidates.append(("monitor_end", int(rep["end_frame"])))
        seen: set[tuple[str, int]] = set()
        for event_type, frame_index in candidates:
            fingerprint = (event_type, frame_index)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            events.append(
                {
                    "rep_id": rep_id,
                    "event_type": event_type,
                    "frame_index": frame_index,
                    "observability": "UNKNOWN",
                    "notes": "由已人工核对的阶段边界确定",
                    "annotation_origin": (
                        "deterministic_human_reviewed_phase_boundary_v1"
                    ),
                }
            )
    return events


def _repair_overlapping_rep_bounds_from_reviewed_intervals(
    reps: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Repair a stale proposal end bound when reviewed phases prove the split."""

    intervals_by_rep: dict[str, list[dict[str, Any]]] = {}
    for interval in intervals:
        intervals_by_rep.setdefault(str(interval.get("rep_id", "")), []).append(
            interval
        )
    repairs: list[dict[str, Any]] = []
    ordered = sorted(reps, key=lambda item: int(item.get("start_frame", -1)))
    for current, following in zip(ordered, ordered[1:]):
        current_end = int(current.get("end_frame", -1))
        following_start = int(following.get("start_frame", -1))
        if current_end < following_start:
            continue
        current_intervals = intervals_by_rep.get(
            str(current.get("rep_id", "")), []
        )
        if not current_intervals:
            continue
        reviewed_end = max(int(item["end_frame"]) for item in current_intervals)
        if reviewed_end != following_start - 1:
            continue
        previous_end = current_end
        current["end_frame"] = reviewed_end
        repairs.append(
            {
                "kind": "stale_proposal_rep_end_clamped_to_reviewed_phases",
                "rep_id": str(current.get("rep_id", "")),
                "previous_end_frame": previous_end,
                "resolved_end_frame": reviewed_end,
                "next_rep_start_frame": following_start,
                "source_review_record_preserved_unchanged": True,
            }
        )
    return repairs


def _flatten_confirmed_noncore_proposal(
    proposal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten a non-core proposal explicitly confirmed by the user."""

    reps: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for candidate in proposal.get("reps", []):
        if not isinstance(candidate, dict):
            continue
        rep_id = str(candidate.get("rep_id", "")).strip()
        reps.append(
            {
                "rep_id": rep_id,
                "start_frame": int(candidate["start_frame"]),
                "end_frame": int(candidate["end_frame"]),
                "validity": "VALID",
                "notes": "AI 候选，待人工核对",
                "annotation_origin": "explicitly_confirmed_noncore_ai_proposal",
                "count_semantics": proposal.get("proposal_semantics"),
                "official_rep_count_supported": bool(
                    proposal.get("official_rep_count_supported", False)
                ),
            }
        )
        for phase in candidate.get("phases", []):
            if not isinstance(phase, dict):
                continue
            intervals.append(
                {
                    "rep_id": rep_id,
                    "start_frame": int(phase["start_frame"]),
                    "end_frame": int(phase["end_frame"]),
                    "phase": str(phase["phase"]),
                    "error_code": "NO_ERROR",
                    "observability": "UNKNOWN",
                    "notes": "AI 候选，待人工核对",
                    "annotation_origin": (
                        "explicitly_confirmed_noncore_ai_proposal"
                    ),
                }
            )
        for event in candidate.get("events", []):
            if not isinstance(event, dict):
                continue
            events.append(
                {
                    "rep_id": rep_id,
                    "event_type": str(event["event_type"]),
                    "frame_index": int(event["frame_index"]),
                    "observability": "UNKNOWN",
                    "notes": "AI 候选，待人工核对",
                    "annotation_origin": (
                        "explicitly_confirmed_noncore_ai_proposal"
                    ),
                }
            )
    return reps, intervals, events


def _force_observable(
    items: list[dict[str, Any]],
    *,
    source: str,
) -> None:
    for item in items:
        previous = str(item.get("observability", "") or "UNSPECIFIED")
        item["observability"] = "OBSERVABLE"
        item["observability_override"] = {
            "previous_value": previous,
            "source": source,
        }


def _confirm_fine_human_review(
    items: list[dict[str, Any]],
    *,
    source: str,
) -> int:
    """Mark retained AI-proposal notes as reviewed provenance, not pending work."""

    updated = 0
    for item in items:
        for field in ("notes", "phase_gap_reason"):
            previous = str(item.get(field, ""))
            resolved = (
                previous.replace("AI 候选，待人工核对", "AI 候选，已人工核对")
                .replace("必须人工核对", "已人工核对")
                .replace("待人工核对", "已人工核对")
                .replace("非人工真值", "已人工核对的候选来源")
            )
            if resolved != previous:
                item[field] = resolved
                updated += 1
        item["human_review_status"] = "complete"
        item["human_review_confirmation_source"] = source
    return updated


def _fine_consistency_warnings(
    reps: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> list[dict[str, str]]:
    errors_by_rep: dict[str, set[str]] = {}
    for interval in intervals:
        rep_id = str(interval.get("rep_id", ""))
        error = str(interval.get("error_code", ""))
        if error and error != "NO_ERROR":
            errors_by_rep.setdefault(rep_id, set()).add(error)
    warnings: list[dict[str, str]] = []
    for rep in reps:
        rep_id = str(rep.get("rep_id", ""))
        validity = str(rep.get("validity", ""))
        errors = errors_by_rep.get(rep_id, set())
        if validity == "NO_REP" and not errors:
            warnings.append(
                {
                    "rep_id": rep_id,
                    "code": "NO_REP_WITHOUT_LABELED_ERROR",
                }
            )
        elif validity == "VALID" and errors:
            warnings.append(
                {
                    "rep_id": rep_id,
                    "code": "VALID_WITH_LABELED_ERROR",
                }
            )
    return warnings


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


def _temporary_subject_group(record_id: str, action: str) -> tuple[str, str]:
    suffix = "02" if record_id in TEMPORARY_SUBJECT_SECOND_RECORDS else "01"
    return (
        f"subject_group_{action}_{suffix}",
        "validation" if suffix == "02" else "development",
    )


def _apply_reviewed_data_roles(
    path: Path,
    records: list[dict[str, Any]],
    *,
    applied_at: str,
    source_bundle: str,
) -> None:
    payload = load_json(path)
    reviewed_by_id = {
        str(record.get("record_id", "")): record for record in records
    }
    assignments = [
        assignment
        for assignment in (payload.get("assignments") or [])
        if isinstance(assignment, dict)
    ]
    for assignment in assignments:
        record = reviewed_by_id.get(str(assignment.get("record_id", "")))
        assignment["authorization_status"] = "confirmed_all_enumerated_uses"
        if record is None:
            continue
        role = str(record.get("dataset_role", "unassigned_pending_review"))
        label_ready = bool(record.get("training_eligible"))
        usable = bool(record.get("fine_annotation_complete"))
        assignment.update(
            {
                "subject_group": str(record.get("subject_group", "")),
                "subject_group_is_temporary": True,
                "role": role,
                "role_source": "documented_temporary_subject_group_policy",
                "role_assigned_at": applied_at,
                "role_source_bundle": source_bundle,
                "expert_review_status": (
                    "complete" if usable else "complete_excluded_unusable"
                ),
                "training_eligible": bool(
                    usable and label_ready and role == "development"
                ),
                "evaluation_eligible": bool(
                    usable and label_ready and role in {"validation", "test"}
                ),
                "golden_eligible": False,
                "template_eligible": False,
                "example_eligible": False,
            }
        )

    role_sets = {
        key: sorted(
            str(item.get("record_id", ""))
            for item in assignments
            if bool(item.get(key))
        )
        for key in (
            "training_eligible",
            "golden_eligible",
            "evaluation_eligible",
            "template_eligible",
            "example_eligible",
        )
    }
    training = set(role_sets["training_eligible"])
    evaluation = set(role_sets["evaluation_eligible"])
    overlaps = sorted(training & evaluation)
    subject_roles: dict[str, set[str]] = {}
    for assignment in assignments:
        subject = str(assignment.get("subject_group", ""))
        role = str(assignment.get("role", ""))
        if subject and role in {"development", "validation", "test"}:
            subject_roles.setdefault(subject, set()).add(role)
    subject_role_conflicts = {
        subject: sorted(roles)
        for subject, roles in subject_roles.items()
        if len(roles) > 1
    }
    payload["generated_at"] = applied_at
    payload["assignments"] = assignments
    payload["role_sets"] = role_sets
    payload["overlaps"] = overlaps
    payload["subject_role_conflicts"] = subject_role_conflicts
    payload["checks"] = {
        "all_records_assigned": all(
            str(item.get("role", ""))
            in {"development", "validation", "test"}
            for item in assignments
        ),
        "no_training_evaluation_overlap": not overlaps,
        "no_subject_role_conflicts": not subject_role_conflicts,
        "test_role_intentionally_empty_pending_new_subjects": not any(
            str(item.get("role", "")) == "test" for item in assignments
        ),
        "temporary_groups_are_not_real_identity_claims": True,
    }
    write_json(path, payload)


def _apply_manifest_review_status(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    payload = load_json(path)
    reviewed_by_id = {
        str(record.get("record_id", "")): record for record in records
    }
    manifest_records = [
        item
        for item in (payload.get("records") or [])
        if isinstance(item, dict)
    ]
    for manifest_record in manifest_records:
        record = reviewed_by_id.get(str(manifest_record.get("record_id", "")))
        if record is None:
            continue
        usable = bool(record.get("fine_annotation_complete"))
        label_ready = bool(record.get("training_eligible"))
        role = str(record.get("dataset_role", ""))
        manifest_record["temporary_subject_group"] = str(
            record.get("subject_group", "")
        )
        manifest_record["temporary_subject_group_is_real_identity"] = False
        manifest_record["review_status"] = {
            "subject_identity": (
                "temporary_group_assigned_real_identity_pending"
            ),
            "action_expert": "single_human_review_complete",
            "data_role": f"temporary_{role}_assigned",
        }
        manifest_record["eligibility"] = {
            "training_eligible": bool(
                usable and label_ready and role == "development"
            ),
            "golden_eligible": False,
            "evaluation_eligible": bool(
                usable and label_ready and role in {"validation", "test"}
            ),
            "example_eligible": False,
        }
    write_json(path, payload)


def apply_quick_review_bundle(
    dataset_root: str | Path,
    review_bundle: str | Path,
    *,
    authorize_oni: bool = True,
    waive_subject_identity_for_internal_record_grouping: bool = True,
    accept_fine_annotations_for_internal_rgb_calibration: bool = False,
    force_phone_observable: bool = False,
    confirm_fine_rgb_human_reviewed: bool = False,
    derive_missing_noncore_events: bool = False,
    confirmed_valid_proposal_record_ids: Collection[str] = (),
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
    confirmed_proposal_ids = {
        str(record_id) for record_id in confirmed_valid_proposal_record_ids
    }
    unknown_confirmed_proposal_ids = confirmed_proposal_ids - set(manifest_by_id)
    if unknown_confirmed_proposal_ids:
        raise ValueError(
            "confirmed proposal records missing from phone manifest: "
            + ", ".join(sorted(unknown_confirmed_proposal_ids))
        )

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
    fine_records: list[dict[str, Any]] = []
    observability_override_source = (
        "user_statement_2026-07-28_all_phone_rgb_annotations_observable"
    )
    fine_review_confirmation_source = (
        "user_statement_2026-07-28_all_exported_phone_rgb_results_manually_reviewed"
    )
    proposal_confirmation_source = (
        "user_statement_2026-07-28_all_candidates_valid"
    )
    confirmed_note_count = 0
    resolved_confirmed_proposal_ids: set[str] = set()
    for path in review_paths:
        review = load_json(path)
        record_id = str(review.get("record_id", ""))
        manifest_record = manifest_by_id.get(record_id)
        if manifest_record is None:
            raise ValueError(f"{record_id}: missing from phone manifest")
        quick = _validate_review_record(review, manifest_record)
        fine_annotation_resolution: list[dict[str, Any]] = []
        if record_id in confirmed_proposal_ids:
            if any(
                quick.get(key)
                for key in ("reps", "phase_error_intervals", "events")
            ):
                raise ValueError(
                    f"{record_id}: confirmed proposal resolution expects empty "
                    "fine-annotation arrays"
                )
            # The proposal builder is imported lazily so the ordinary dataset
            # importer remains independent from the review web application.
            from webui.review import _build_noncore_candidate_annotations

            action_segments_path = root / "annotations" / "action_segments_v1.json"
            counting_path = (
                root / "reports" / "human_review_counting_regression_v1.json"
            )
            action_segments = load_json(action_segments_path)
            counting_report = load_json(counting_path)
            action_proposal = next(
                (
                    item
                    for item in _records(action_segments)
                    if str(item.get("record_id", "")) == record_id
                ),
                None,
            )
            counting_row = next(
                (
                    item
                    for item in _records(counting_report)
                    if str(item.get("record_id", "")) == record_id
                ),
                None,
            )
            proposal = _build_noncore_candidate_annotations(
                root,
                manifest_record,
                action_proposal,
                counting_row,
            )
            proposal_reps, proposal_intervals, proposal_events = (
                _flatten_confirmed_noncore_proposal(proposal)
            )
            if not (proposal_reps and proposal_intervals and proposal_events):
                raise ValueError(
                    f"{record_id}: confirmed proposal produced incomplete annotations"
                )
            quick["reps"] = proposal_reps
            quick["phase_error_intervals"] = proposal_intervals
            quick["events"] = proposal_events
            fine_annotation_resolution.append(
                {
                    "kind": "explicit_user_confirmation_of_all_ai_candidates",
                    "source": proposal_confirmation_source,
                    "source_proposal": str(
                        action_segments_path.relative_to(root)
                    ),
                    "source_proposal_sha256": file_sha256(action_segments_path),
                    "source_counting_report": str(counting_path.relative_to(root)),
                    "source_counting_report_sha256": file_sha256(counting_path),
                    "rep_count": len(proposal_reps),
                    "phase_error_interval_count": len(proposal_intervals),
                    "event_count": len(proposal_events),
                    "source_review_record_preserved_unchanged": True,
                }
            )
            resolved_confirmed_proposal_ids.add(record_id)
        if (
            derive_missing_noncore_events
            and quick.get("reps")
            and quick.get("phase_error_intervals")
            and not quick.get("events")
        ):
            derived_events = _derive_noncore_events(
                str(quick.get("action", "")),
                [
                    dict(item)
                    for item in quick.get("reps", [])
                    if isinstance(item, dict)
                ],
                [
                    dict(item)
                    for item in quick.get("phase_error_intervals", [])
                    if isinstance(item, dict)
                ],
            )
            if not derived_events:
                raise ValueError(
                    f"{record_id}: missing events could not be derived from "
                    "reviewed non-core phase boundaries"
                )
            quick["events"] = derived_events
            fine_annotation_resolution.append(
                {
                    "kind": "deterministic_events_from_human_reviewed_phases",
                    "derivation": (
                        "noncore_phase_boundary_to_canonical_event_v1"
                    ),
                    "event_count": len(derived_events),
                    "source_review_record_preserved_unchanged": True,
                }
            )
        if (
            derive_missing_noncore_events
            and str(quick.get("action", "")) in NONCORE_EVENT_FROM_PHASE
        ):
            rep_bound_repairs = (
                _repair_overlapping_rep_bounds_from_reviewed_intervals(
                    [
                        item
                        for item in quick.get("reps", [])
                        if isinstance(item, dict)
                    ],
                    [
                        item
                        for item in quick.get("phase_error_intervals", [])
                        if isinstance(item, dict)
                    ],
                )
            )
            fine_annotation_resolution.extend(rep_bound_repairs)
        audit_revisions = [
            int(item.get("revision", -1))
            for item in (review.get("audit_log") or [])
            if isinstance(item, dict)
        ]
        audit_revision_gaps = [
            revision
            for revision in range(1, max(audit_revisions, default=0) + 1)
            if revision not in audit_revisions
        ]
        reps, phase_error_intervals, events = _validate_fine_annotations(
            quick,
            record_id=record_id,
            maximum_frame=int(
                (manifest_record.get("video") or {}).get(
                    "decoded_frame_count",
                    0,
                )
            )
            - 1,
        )
        if force_phone_observable:
            _force_observable(
                phase_error_intervals,
                source=observability_override_source,
            )
            _force_observable(events, source=observability_override_source)
        if confirm_fine_rgb_human_reviewed:
            confirmed_note_count += _confirm_fine_human_review(
                reps,
                source=fine_review_confirmation_source,
            )
            confirmed_note_count += _confirm_fine_human_review(
                phase_error_intervals,
                source=fine_review_confirmation_source,
            )
            confirmed_note_count += _confirm_fine_human_review(
                events,
                source=fine_review_confirmation_source,
            )
        usable = str(quick.get("video_usability", "")).lower() == "usable"
        target_correct = str(quick.get("target_status", "")).lower() == "correct"
        interval_confirmed = usable and target_correct
        review_outcome_terminal = str(quick.get("status", "")) in {
            "complete",
            "blocked",
        }
        fine_complete = bool(
            str(quick.get("status", "")) == "complete"
            and reps
            and phase_error_intervals
            and events
        )
        consistency_warnings = _fine_consistency_warnings(
            reps,
            phase_error_intervals,
        )
        original_usable_start = int(quick["usable_start_frame"])
        original_usable_end = int(quick["usable_end_frame"])
        effective_usable_start = (
            min(original_usable_start, *(int(item["start_frame"]) for item in reps))
            if reps
            else original_usable_start
        )
        effective_usable_end = (
            max(original_usable_end, *(int(item["end_frame"]) for item in reps))
            if reps
            else original_usable_end
        )
        calibration_eligible = bool(
            accept_fine_annotations_for_internal_rgb_calibration
            and interval_confirmed
            and fine_complete
        )
        supervised_training_eligible = bool(
            calibration_eligible and confirm_fine_rgb_human_reviewed
        )
        reviewed_subject = str(quick.get("subject_id", "")).strip()
        reviewed_dataset_role = str(quick.get("dataset_role", "")).strip()
        fallback_subject, fallback_role = _temporary_subject_group(
            record_id,
            str(quick["action"]),
        )
        temporary_subject_group = (
            reviewed_subject
            if reviewed_subject
            not in {"", "subject_pending", "pending", "unknown"}
            else fallback_subject
        )
        dataset_role = (
            reviewed_dataset_role
            if reviewed_dataset_role in {"development", "validation", "test"}
            else fallback_role
        )
        resolved_observability = (
            "OBSERVABLE"
            if force_phone_observable
            else str(quick.get("observability", ""))
        )
        segment = {
            "timeline_label": "target_action",
            "action_type": str(quick["action"]),
            "start_frame": effective_usable_start,
            "end_frame": effective_usable_end,
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
                "subject_group": temporary_subject_group,
                "subject_group_is_temporary": True,
                "dataset_role": dataset_role,
                "temporary_grouping_source": (
                    "review_bundle"
                    if reviewed_subject
                    not in {"", "subject_pending", "pending", "unknown"}
                    and reviewed_dataset_role
                    in {"development", "validation", "test"}
                    else "documented_temporary_subject_group_policy"
                ),
                "source_filename": str(manifest_record.get("source_filename", "")),
                "reviewer_role": str(review.get("reviewer_role", "")),
                "reviewer_id": str(review.get("reviewer_id", "")),
                "review_revision": int(review.get("revision", 0)),
                "audit_revision_gaps": audit_revision_gaps,
                "review_record_sha256": file_sha256(path),
                "human_confirmed_action_type": target_correct,
                "human_confirmed_usable_interval": interval_confirmed,
                "video_usability": str(quick.get("video_usability", "")),
                "overall_result": str(quick.get("overall_result", "")),
                "observability": resolved_observability,
                "observability_override": (
                    {
                        "previous_value": str(quick.get("observability", "")),
                        "source": observability_override_source,
                    }
                    if force_phone_observable
                    else None
                ),
                "equipment_visibility": str(
                    quick.get("equipment_visibility", "")
                ),
                "proposal_decision": str(quick.get("proposal_decision", "")),
                "notes": str(quick.get("notes", "")),
                "segments": [segment],
                "rep_annotations_supplied": bool(reps),
                "phase_error_intervals_supplied": bool(phase_error_intervals),
                "event_annotations_supplied": bool(events),
                "fine_annotation_complete": fine_complete,
                "review_outcome_terminal": review_outcome_terminal,
                "fine_annotation_resolution": fine_annotation_resolution,
                "counts_as_reviewed_ground_truth": calibration_eligible,
                "internal_rgb_rule_calibration_eligible": calibration_eligible,
                "fine_rgb_human_review_status": (
                    "complete"
                    if confirm_fine_rgb_human_reviewed and fine_complete
                    else "not_confirmed"
                ),
                "training_eligible": supervised_training_eligible,
            }
        )
        if fine_complete:
            fine_records.append(
                {
                    "record_id": record_id,
                    "action": str(quick["action"]),
                    "subject_group": temporary_subject_group,
                    "subject_group_is_temporary": True,
                    "dataset_role": dataset_role,
                    "temporary_grouping_source": (
                        "review_bundle"
                        if reviewed_subject
                        not in {"", "subject_pending", "pending", "unknown"}
                        and reviewed_dataset_role
                        in {"development", "validation", "test"}
                        else "documented_temporary_subject_group_policy"
                    ),
                    "camera_view": str(
                        manifest_record.get("camera_view", "")
                    ),
                    "source_filename": str(manifest_record.get("source_filename", "")),
                    "source_type": "phone_rgb",
                    "reviewer_role": str(review.get("reviewer_role", "")),
                    "reviewer_id": str(review.get("reviewer_id", "")),
                    "review_revision": int(review.get("revision", 0)),
                    "audit_revision_gaps": audit_revision_gaps,
                    "review_record_sha256": file_sha256(path),
                    "interval_semantics": "closed",
                    "usable_start_frame": effective_usable_start,
                    "usable_end_frame": effective_usable_end,
                    "review_export_usable_start_frame": original_usable_start,
                    "review_export_usable_end_frame": original_usable_end,
                    "usable_interval_expanded_to_cover_fine_reps": bool(
                        effective_usable_start != original_usable_start
                        or effective_usable_end != original_usable_end
                    ),
                    "overall_result": str(quick.get("overall_result", "")),
                    "observability": resolved_observability,
                    "observability_override_source": (
                        observability_override_source
                        if force_phone_observable
                        else None
                    ),
                    "reps": reps,
                    "phase_error_intervals": phase_error_intervals,
                    "events": events,
                    "fine_annotation_resolution": fine_annotation_resolution,
                    "consistency_warnings": consistency_warnings,
                    "reviewed_ground_truth": calibration_eligible,
                    "internal_rgb_rule_calibration_eligible": calibration_eligible,
                    "fine_rgb_human_review_status": (
                        "complete"
                        if confirm_fine_rgb_human_reviewed
                        else "not_confirmed"
                    ),
                    "fine_rgb_human_review_confirmation_source": (
                        fine_review_confirmation_source
                        if confirm_fine_rgb_human_reviewed
                        else None
                    ),
                    "supervised_model_training_eligible": (
                        supervised_training_eligible
                    ),
                }
            )

    if resolved_confirmed_proposal_ids != confirmed_proposal_ids:
        unresolved = confirmed_proposal_ids - resolved_confirmed_proposal_ids
        raise ValueError(
            "confirmed proposal records were not resolved: "
            + ", ".join(sorted(unresolved))
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
        _apply_reviewed_data_roles(
            data_roles_path,
            applied_records,
            applied_at=applied_at,
            source_bundle=source_label,
        )
    _apply_manifest_review_status(manifest_path, applied_records)

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
        "terminal_review_decision_record_count": sum(
            bool(item["review_outcome_terminal"]) for item in applied_records
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
    fine_overlay = {
        "schema_version": 1,
        "artifact_type": "human_rgb_fine_annotations_v1",
        "generated_at": applied_at,
        "protocol_version": protocol.get("protocol_version"),
        "source_bundle": source_label,
        "source_handoff_sha256": file_sha256(handoff_path),
        "source_type": "phone_rgb",
        "oni_records_included": False,
        "interval_semantics": "closed",
        "single_human_review_sufficient_for_current_stage": True,
        "fine_rgb_human_review": {
            "status": (
                "complete"
                if confirm_fine_rgb_human_reviewed
                and len(fine_records) == len(applied_records)
                else "complete_with_unusable_records_excluded"
                if confirm_fine_rgb_human_reviewed
                else "not_confirmed"
            ),
            "confirmation_source": (
                fine_review_confirmation_source
                if confirm_fine_rgb_human_reviewed
                else None
            ),
            "retained_ai_proposal_notes_updated_count": confirmed_note_count,
            "source_bundle_preserved_for_audit": True,
            "confirmed_valid_proposal_record_ids": sorted(
                resolved_confirmed_proposal_ids
            ),
            "missing_noncore_events_derived": derive_missing_noncore_events,
        },
        "observability_policy": (
            {
                "all_phone_rgb_annotations": "OBSERVABLE",
                "source": observability_override_source,
                "original_values_preserved_in_override_audit": True,
            }
            if force_phone_observable
            else {"all_phone_rgb_annotations": "preserve_reviewer_values"}
        ),
        "record_count": len(fine_records),
        "rep_count": sum(len(item["reps"]) for item in fine_records),
        "phase_error_interval_count": sum(
            len(item["phase_error_intervals"]) for item in fine_records
        ),
        "event_count": sum(len(item["events"]) for item in fine_records),
        "consistency_warning_count": sum(
            len(item["consistency_warnings"]) for item in fine_records
        ),
        "internal_rgb_rule_calibration_eligible_record_count": sum(
            bool(item["internal_rgb_rule_calibration_eligible"])
            for item in fine_records
        ),
        "supervised_model_training_eligible_record_count": 0,
        "records": fine_records,
    }
    fine_overlay["supervised_model_training_eligible_record_count"] = sum(
        bool(item["supervised_model_training_eligible"])
        for item in fine_records
    )
    fine_overlay_path = write_json(
        review_dir / "human_rgb_fine_annotations_v1.json",
        fine_overlay,
    )
    fine_complete_count = overlay["fine_annotation_complete_record_count"]
    terminal_review_count = overlay["terminal_review_decision_record_count"]
    expected_phone_count = len(applied_records)
    progress = {
        "schema_version": 1,
        "artifact_type": "round9_human_review_progress_v1",
        "generated_at": applied_at,
        "status": (
            "all_phone_rgb_review_complete_usable_fine_annotations_ready"
            if terminal_review_count == expected_phone_count
            else "single_human_review_active_core_fine_annotation_and_oni_subject_review_pending"
        ),
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
        "terminal_review_decision_record_count": terminal_review_count,
        "unusable_or_blocked_record_count": (
            expected_phone_count - fine_complete_count
        ),
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
            *(
                []
                if terminal_review_count == expected_phone_count
                else ["phone_rgb_review_decisions_pending"]
            ),
            "oni_depth_ir_subject_review_not_present_in_phone_review_bundle",
        ],
        "deferred_tasks": [
            "second_independent_human_review_and_agreement",
            "high_disagreement_active_learning_clip_review",
            "idle_transition_unknown_and_continuous_switch_data_collection",
            "automatic_action_recognition_training_and_gating",
        ],
        "review_overlay": str(overlay_path.relative_to(root)),
        "review_overlay_sha256": file_sha256(overlay_path),
        "fine_annotation_overlay": str(fine_overlay_path.relative_to(root)),
        "fine_annotation_overlay_sha256": file_sha256(fine_overlay_path),
        "phone_rgb_internal_rule_calibration_ready": bool(
            fine_overlay["internal_rgb_rule_calibration_eligible_record_count"]
        ),
        "phone_rgb_supervised_experiment_labels_ready": bool(
            fine_overlay[
                "supervised_model_training_eligible_record_count"
            ]
        ),
        "phone_rgb_fine_human_review_complete": bool(
            confirm_fine_rgb_human_reviewed
            and terminal_review_count == expected_phone_count
        ),
        "oni_required_for_phone_rgb_internal_rule_calibration": False,
    }
    progress_path = write_json(
        root / "reports" / "round9_human_review_progress_v1.json",
        progress,
    )
    return {
        "review_overlay": overlay_path,
        "fine_annotation_overlay": fine_overlay_path,
        "round9_progress": progress_path,
        "phone_manifest": manifest_path,
        "oni_manifest": oni_manifest_path,
        "data_roles": data_roles_path,
    }


__all__ = ["ALL_AUTHORIZED_USES", "apply_quick_review_bundle"]
