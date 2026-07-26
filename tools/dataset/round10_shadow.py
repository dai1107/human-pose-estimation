"""Round 10 shadow-ablation reporting with strict human-ground-truth gates."""

from __future__ import annotations

import hashlib
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hyrox.features import extract_basic_pose_features
from src.action_gating import (
    ACTION_FEATURE_NAMES,
    ActionFeatureWindow,
    LogisticActionModel,
    grouped_cross_validate_logistic,
)
from src.contracts import ContractBundle, load_contract_bundle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    return []


def _human_segment_confirmed(record: dict[str, Any]) -> bool:
    if not bool(record.get("training_eligible")):
        return False
    segments = record.get("segments")
    if not isinstance(segments, list) or not segments:
        return False
    return all(
        isinstance(segment, dict)
        and bool(segment.get("human_confirmed"))
        and "human" in str(segment.get("annotator_type", "")).lower()
        for segment in segments
    )


def _quick_review_overlay(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "reviews" / "human_quick_review_application_v1.json"
    if not path.is_file():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def evaluate_data_readiness(
    dataset_root: Path,
    bundle: ContractBundle,
    *,
    allow_pending_subjects: bool | None = None,
) -> dict[str, Any]:
    manifest = load_json(dataset_root / "manifests" / "phone_records.json")
    action_segments = load_json(dataset_root / "annotations" / "action_segments_v1.json")
    agreement = load_json(dataset_root / "reports" / "annotation_agreement_v1.json")
    data_roles = load_json(dataset_root / "manifests" / "data_roles_v1.json")
    quick_review = _quick_review_overlay(dataset_root)
    quick_review_records = {
        item.get("record_id"): item
        for item in _records(quick_review)
    }
    if allow_pending_subjects is None:
        subject_policy = quick_review.get("subject_identity_policy")
        allow_pending_subjects = bool(
            isinstance(subject_policy, dict)
            and subject_policy.get("status")
            == "temporarily_waived_for_internal_record_grouped_experiments"
        )
    manifest_records = _records(manifest)
    segment_records = _records(action_segments)
    roles = {item.get("record_id"): item for item in data_roles.get("assignments", []) if isinstance(item, dict)}
    segments = {item.get("record_id"): item for item in segment_records}

    authorized: list[str] = []
    identified: list[str] = []
    single_human_reviewed: list[str] = []
    single_human_intervals: list[str] = []
    human_confirmed: list[str] = []
    training_eligible: list[str] = []
    eligible_class_records: Counter[str] = Counter()
    record_failures: list[dict[str, Any]] = []
    for record in manifest_records:
        record_id = str(record.get("record_id", ""))
        reasons: list[str] = []
        authorization = record.get("usage_authorization") or {}
        authorized_uses = set(authorization.get("authorized_uses") or []) if isinstance(authorization, dict) else set()
        authorization_ok = (
            isinstance(authorization, dict)
            and str(authorization.get("status", "")).lower() in {"confirmed", "approved", "authorized"}
            and "model_training" in authorized_uses
        )
        if authorization_ok:
            authorized.append(record_id)
        else:
            reasons.append("training_usage_authorization_pending")
        subject_id = str(record.get("subject_id", ""))
        identity_ok = bool(subject_id and subject_id not in {"subject_pending", "unknown", "pending"})
        if identity_ok:
            identified.append(record_id)
        elif not allow_pending_subjects:
            reasons.append("subject_identity_pending")
        quick_record = quick_review_records.get(record_id, {})
        if (
            isinstance(quick_record, dict)
            and bool(quick_record.get("human_confirmed_action_type"))
        ):
            single_human_reviewed.append(record_id)
            if bool(quick_record.get("human_confirmed_usable_interval")):
                single_human_intervals.append(record_id)
        segment = segments.get(record_id, {})
        segment_ok = _human_segment_confirmed(segment)
        if segment_ok:
            human_confirmed.append(record_id)
        else:
            reasons.append("independent_human_action_segment_review_pending")
        role = roles.get(record_id, {})
        role_ok = bool(role.get("training_eligible")) and str(role.get("role", "")).startswith("train")
        if not role_ok:
            reasons.append("training_role_not_assigned")
        identity_gate_ok = identity_ok or bool(allow_pending_subjects)
        eligible = authorization_ok and identity_gate_ok and segment_ok and role_ok
        if eligible:
            training_eligible.append(record_id)
            action = str(segment.get("action_type", "unknown"))
            eligible_class_records[action] += 1
        record_failures.append(
            {
                "record_id": record_id,
                "action": str(record.get("action", "unknown")),
                "training_eligible": eligible,
                "blockers": reasons,
            }
        )

    class_readiness = {}
    for class_name in bundle.action_gating.classes:
        count = int(eligible_class_records[class_name])
        class_readiness[class_name] = {
            "independent_record_count": count,
            "required_record_count": bundle.action_gating.minimum_records_per_class,
            "ready": count >= bundle.action_gating.minimum_records_per_class,
        }
    class_gaps = [name for name, status in class_readiness.items() if not status["ready"]]
    independent_humans = max(
        int(agreement.get("eligible_reviewer_count", 0) or 0),
        int(quick_review.get("independent_human_reviewer_count", 0) or 0),
    )
    agreement_ready = independent_humans >= 1
    blockers = []
    if len(authorized) != len(manifest_records):
        blockers.append("usage_authorization_incomplete")
    if len(identified) != len(manifest_records) and not allow_pending_subjects:
        blockers.append("subject_identity_incomplete")
    if len(human_confirmed) != len(manifest_records):
        blockers.append("human_action_segments_incomplete")
    if not agreement_ready:
        blockers.append("single_human_reviewer_pending")
    if class_gaps:
        blockers.append("required_action_idle_transition_unknown_class_coverage_incomplete")
    return {
        "ready": not blockers,
        "record_count": len(manifest_records),
        "authorized_record_count": len(authorized),
        "identified_subject_record_count": len(identified),
        "single_human_reviewed_action_record_count": len(single_human_reviewed),
        "single_human_reviewed_interval_record_count": len(single_human_intervals),
        "human_confirmed_record_count": len(human_confirmed),
        "training_eligible_record_count": len(training_eligible),
        "independent_human_reviewer_count": independent_humans,
        "agreement_gate_passed": agreement_ready,
        "subject_identity_waiver_applied": bool(allow_pending_subjects),
        "subject_identity_waiver_scope": (
            "internal_record_id_grouped_experiments_only"
            if allow_pending_subjects
            else None
        ),
        "group_split_key": bundle.action_gating.group_split_key,
        "fallback_group_split_key": bundle.action_gating.fallback_group_split_key,
        "class_readiness": class_readiness,
        "class_gaps": class_gaps,
        "blockers": blockers,
        "records": record_failures,
    }


def _offline_contract_records(dataset_root: Path) -> list[dict[str, Any]]:
    manifest = load_json(dataset_root / "manifests" / "phone_records.json")
    segments = {
        item.get("record_id"): item
        for item in _records(load_json(dataset_root / "annotations" / "action_segments_v1.json"))
    }
    object_scene = {
        item.get("record_id"): item
        for item in _records(load_json(dataset_root / "annotations" / "object_scene_evidence_v1.json"))
    }
    quick_review = _quick_review_overlay(dataset_root)
    quick_records = {
        item.get("record_id"): item for item in _records(quick_review)
    }
    records = []
    for record in _records(manifest):
        record_id = str(record.get("record_id", ""))
        pose_cache = record.get("pose_cache") if isinstance(record.get("pose_cache"), dict) else {}
        segment = segments.get(record_id, {})
        scene = object_scene.get(record_id, {})
        quick = quick_records.get(record_id, {})
        records.append(
            {
                "record_id": record_id,
                "subject": {
                    "status": (record.get("review_status") or {}).get("subject_identity", "pending"),
                    "training_gate_passed": str(record.get("subject_id", "")) not in {"", "subject_pending", "unknown"},
                },
                "pose": {
                    "status": pose_cache.get("status", "missing"),
                    "target_track_id": pose_cache.get("target_track_id"),
                },
                "coordinates": {
                    "status": "contract_available",
                    "metric_phone_3d_ground_truth": False,
                    "fallback": "image_normalized_2d",
                },
                "action_segment": {
                    "status": segment.get("video_action_review_status", "missing"),
                    "human_confirmed": bool(segment.get("training_eligible")),
                    "single_human_quick_review_confirmed": bool(
                        isinstance(quick, dict)
                        and quick.get("human_confirmed_action_type")
                        and quick.get("human_confirmed_usable_interval")
                    ),
                },
                "evidence": {
                    "status": "proposal_only" if scene else "missing",
                    "unobservable_is_pass": False,
                },
                "product_output": {
                    "status": "round10_contract_serializable",
                    "default_runtime_enabled": False,
                },
                "engineering_closed_loop": bool(pose_cache) and bool(segment) and bool(scene),
                "reviewed_truth_closed_loop": False,
            }
        )
    return records


def _segment_class(segment: dict[str, Any], record: dict[str, Any]) -> str | None:
    label = str(segment.get("timeline_label", ""))
    if label == "target_action":
        return str(segment.get("action_type") or record.get("action_type") or "unknown")
    if label == "idle":
        return "idle"
    if label in {"setup", "transition"}:
        return "transition"
    if label in {"unknown_motion", "target_out_of_frame"}:
        return "unknown"
    return None


def extract_action_training_samples(
    dataset_root: Path,
    bundle: ContractBundle,
) -> tuple[np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    """Extract only human-confirmed, causal, target-bound pose windows."""
    manifest = {item.get("record_id"): item for item in _records(load_json(dataset_root / "manifests" / "phone_records.json"))}
    segments_by_record = {
        item.get("record_id"): item
        for item in _records(load_json(dataset_root / "annotations" / "action_segments_v1.json"))
    }
    rows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    metadata: list[dict[str, Any]] = []
    for record_id, segment_record in sorted(segments_by_record.items()):
        record = manifest.get(record_id)
        if not isinstance(record, dict) or not _human_segment_confirmed(segment_record):
            continue
        pose_cache = record.get("pose_cache") if isinstance(record.get("pose_cache"), dict) else {}
        relative_pose_path = pose_cache.get("causal_analysis_pose")
        if not isinstance(relative_pose_path, str):
            continue
        pose_path = dataset_root / relative_pose_path
        if not pose_path.is_file():
            continue
        video = record.get("video") if isinstance(record.get("video"), dict) else {}
        width = int(video.get("width", 1) or 1)
        height = int(video.get("height", 1) or 1)
        fps = float(video.get("fps", 30.0) or 30.0)
        sampling_stride = max(1, int(round(fps * 0.25)))
        frame_labels: dict[int, str] = {}
        for segment in segment_record.get("segments", []):
            if not isinstance(segment, dict) or not bool(segment.get("human_confirmed")):
                continue
            class_name = _segment_class(segment, segment_record)
            if class_name not in bundle.action_gating.classes:
                continue
            for frame_index in range(int(segment.get("start_frame", 0)), int(segment.get("end_frame", -1)) + 1):
                frame_labels[frame_index] = class_name
        if not frame_labels:
            continue
        group = str(record.get(bundle.action_gating.group_split_key) or record_id)
        window = ActionFeatureWindow(bundle.action_gating)
        previous_label: str | None = None
        previous_frame: int | None = None
        with gzip.open(pose_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                frame = json.loads(line)
                frame_index = int(frame.get("frame_index", -1))
                class_name = frame_labels.get(frame_index)
                formal = bool(frame.get("formal_pose_eligible")) and bool(frame.get("may_drive_rules_or_training"))
                if class_name is None or not formal:
                    window.reset()
                    previous_label = None
                    previous_frame = None
                    continue
                if class_name != previous_label or (previous_frame is not None and frame_index != previous_frame + 1):
                    window.reset()
                previous_label = class_name
                previous_frame = frame_index
                features = extract_basic_pose_features(
                    frame.get("image_normalized_2d"),
                    image_width=width,
                    image_height=height,
                )
                window.update(features, timestamp_ms=int(round(float(frame.get("source_timestamp_ms", 0.0)))))
                if window.ready and frame_index % sampling_stride == 0:
                    rows.append(window.vector())
                    labels.append(class_name)
                    groups.append(group)
                    metadata.append(
                        {
                            "record_id": record_id,
                            "subject_id": str(record.get("subject_id")),
                            "frame_index": frame_index,
                            "class": class_name,
                            "pose_source": relative_pose_path,
                            "future_frames_used": False,
                            "human_confirmed": True,
                        }
                    )
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.size == 0:
        matrix = np.empty((0, len(ACTION_FEATURE_NAMES)), dtype=np.float64)
    return matrix, labels, groups, metadata


def train_action_gate_if_ready(
    dataset_root: Path,
    bundle: ContractBundle,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    if not readiness.get("ready"):
        return {
            "performed": False,
            "status": "blocked_by_data_readiness",
            "model_path": None,
            "model_hash": None,
            "metrics": None,
        }
    matrix, labels, groups, metadata = extract_action_training_samples(dataset_root, bundle)
    class_counts = Counter(labels)
    missing = [name for name in bundle.action_gating.classes if class_counts[name] == 0]
    if missing:
        return {
            "performed": False,
            "status": "blocked_feature_windows_missing_classes",
            "missing_classes": missing,
            "sample_count": len(labels),
            "model_path": None,
            "model_hash": None,
            "metrics": None,
        }
    try:
        metrics = grouped_cross_validate_logistic(
            matrix,
            labels,
            groups,
            classes=bundle.action_gating.classes,
            maximum_folds=5,
        )
    except ValueError as exc:
        return {
            "performed": False,
            "status": "blocked_grouped_cross_validation",
            "reason": str(exc),
            "sample_count": len(labels),
            "model_path": None,
            "model_hash": None,
            "metrics": None,
        }
    model = LogisticActionModel.fit(
        matrix,
        labels,
        classes=bundle.action_gating.classes,
        model_version="round10_action_gate_logreg_v1",
        training_metadata={
            "data_provenance": "reviewed_ground_truth_only",
            "group_split_key": bundle.action_gating.group_split_key,
            "record_ids": sorted({item["record_id"] for item in metadata}),
            "sample_count": len(metadata),
            "class_counts": dict(sorted(class_counts.items())),
            "feature_schema_version": bundle.action_gating.feature_schema_version,
        },
    )
    model_path = model.save(dataset_root / "models" / "round10_action_gate_logreg_v1.json")
    sample_manifest = write_json(
        dataset_root / "reports" / "round10_action_gate_samples_v1.json",
        {
            "schema_version": 1,
            "artifact_type": "round10_action_gate_samples_v1",
            "generated_at": utc_now(),
            "feature_names": list(ACTION_FEATURE_NAMES),
            "sample_count": len(metadata),
            "samples": metadata,
        },
    )
    return {
        "performed": True,
        "status": "trained_internal_shadow_only",
        "model_path": str(model_path.relative_to(dataset_root)),
        "model_hash": model.model_hash,
        "sample_manifest": str(sample_manifest.relative_to(dataset_root)),
        "sample_count": len(labels),
        "metrics": metrics,
    }


def build_round10_reports(
    dataset_root: str | Path,
    *,
    contract_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(dataset_root)
    bundle = load_contract_bundle(contract_dir)
    readiness = evaluate_data_readiness(root, bundle)
    training = train_action_gate_if_ready(root, bundle, readiness)
    training_blockers = list(readiness["blockers"])
    if readiness["ready"] and not training["performed"]:
        training_blockers.append(str(training["status"]))
    round7 = load_json(root / "reports" / "round7_implementation_summary.json")
    round8 = load_json(root / "reports" / "round8_implementation_summary.json")
    queue = load_json(root / "reports" / "round9_active_review_queue_v1.json")
    generated_at = utc_now()
    offline_records = _offline_contract_records(root)
    all_engineering_closed = bool(offline_records) and all(item["engineering_closed_loop"] for item in offline_records)

    experiment_definitions = [
        {
            "id": "A",
            "name": "full_frame_mediapipe_current_rules",
            "implementation_status": "available",
            "evaluation_status": "blocked_human_ground_truth" if not readiness["ready"] else "ready",
            "metrics": None,
        },
        {
            "id": "B",
            "name": "target_locked_roi_mediapipe",
            "implementation_status": "available_default_off",
            "evaluation_status": "not_selected_roi_precision_and_latency_gates_failed",
            "metrics": None,
        },
        {
            "id": "C",
            "name": "B_plus_body_canonical_2d_3d_quality_gates",
            "implementation_status": "coordinate_contract_available",
            "evaluation_status": "blocked_by_B_and_human_ground_truth",
            "metrics": None,
        },
        {
            "id": "D",
            "name": "C_plus_human_reviewed_multibackend_pose_events",
            "implementation_status": "pipeline_available",
            "evaluation_status": "blocked_independent_human_pose_event_review",
            "metrics": None,
        },
        {
            "id": "E",
            "name": "D_plus_logistic_regression_action_gate",
            "implementation_status": "baseline_and_shadow_runtime_available_default_off",
            "evaluation_status": training["status"],
            "metrics": training["metrics"],
        },
        {
            "id": "F",
            "name": "E_plus_display_only_prediction",
            "implementation_status": "display_prediction_contract_available",
            "evaluation_status": (
                "blocked_double_reviewed_event_anchors_and_E"
                if not training["performed"]
                else "blocked_double_reviewed_event_anchors"
            ),
            "metrics": None,
        },
    ]
    ablation = {
        "schema_version": 1,
        "artifact_type": "round10_shadow_ablation_v1",
        "generated_at": generated_at,
        "status": (
            "trained_internal_shadow_ablation_available"
            if training["performed"]
            else "data_readiness_failed_no_metrics_fabricated"
        ),
        "contract_versions": bundle.versions,
        "default_runtime": {
            "pose_mode": "mediapipe_rgb",
            "automatic_action_gating_enabled": False,
            "neural_model_enabled": False,
            "roi_enabled": False,
        },
        "data_readiness": readiness,
        "experiments": experiment_definitions,
        "action_gate_training": training,
        "offline_contract_closure": {
            "record_count": len(offline_records),
            "engineering_closed_loop_count": sum(bool(item["engineering_closed_loop"]) for item in offline_records),
            "reviewed_truth_closed_loop_count": sum(bool(item["reviewed_truth_closed_loop"]) for item in offline_records),
            "all_engineering_closed": all_engineering_closed,
            "records": offline_records,
        },
        "required_future_metrics": [
            "action_macro_f1", "unknown_rejection", "switch_latency_ms",
            "core_event_error_frames", "rule_decision_metrics", "three_d_stability",
            "per_joint_latency_jitter", "inference_p50_ms", "inference_p95_ms",
        ],
        "truthfulness": {
            "ai_proposals_used_as_ground_truth": False,
            "filename_intent_used_as_ground_truth": False,
            "unobservable_equipment_treated_as_pass": False,
            "test_thresholds_tuned_to_pass": False,
            "production_improvement_claimed": False,
        },
    }

    active_priorities = {
        item.get("record_id"): item.get("priority", 0)
        for item in queue.get("records", [])
        if isinstance(item, dict)
    }
    record_failures = sorted(
        (
            {
                **item,
                "priority": int(active_priorities.get(item["record_id"], 0) or 0),
            }
            for item in readiness["records"]
            if item["blockers"]
        ),
        key=lambda item: (-item["priority"], item["record_id"]),
    )
    failure_pool = {
        "schema_version": 1,
        "artifact_type": "round10_failure_pool_v1",
        "generated_at": generated_at,
        "status": "open",
        "global_blockers": training_blockers,
        "class_gaps": readiness["class_readiness"],
        "record_failures": record_failures,
        "engineering_failures": [
            {
                "component": "round7_roi",
                "status": "disabled",
                "reasons": [
                    "precision_gate_failed" if not round7.get("roi_summary", {}).get("precision_gate_passed") else "",
                    "latency_gate_failed" if not round7.get("roi_summary", {}).get("latency_gate_passed") else "",
                ],
            },
            {
                "component": "sensor_to_photon",
                "status": round8.get("temporal_summary", {}).get("sensor_to_photon", {}).get("status", "not_measured"),
                "reason": round8.get("temporal_summary", {}).get("sensor_to_photon", {}).get("reason"),
            },
        ],
        "highest_value_collection_tasks": [
            "continuous_mixed_actions_with_real_transitions",
            "idle_and_unknown_motion",
            "unseen_subjects",
            "unsupported_views_and_equipment_occlusion",
            "skierg_sled_pull_sled_push_errors_and_boundaries",
        ],
    }

    report_dir = root / "reports"
    ablation_path = write_json(report_dir / "round10_shadow_ablation_v1.json", ablation)
    failure_path = write_json(report_dir / "round10_failure_pool_v1.json", failure_pool)
    summary = {
        "schema_version": 1,
        "artifact_type": "round10_implementation_summary",
        "generated_at": generated_at,
        "status": (
            "ready_for_shadow_evaluation"
            if training["performed"]
            else (
                "engineering_complete_single_human_review_applied_remaining_training_gates_pending"
                if readiness.get("single_human_reviewed_action_record_count", 0)
                else "engineering_complete_human_training_gate_pending"
            )
        ),
        "contract_versions": bundle.versions,
        "required_artifacts_present": True,
        "automatic_action_gating_default_enabled": False,
        "auto_action_cli_enabled": False,
        "automatic_action_recognition_status": "deferred_by_product_decision",
        "formal_action_selection": "manual_only",
        "review_policy": "single_human_review_sufficient_for_current_stage",
        "shadow_runtime_available": True,
        "unknown_ood_supported": True,
        "switch_protection_available": True,
        "logistic_regression_baseline_implemented": True,
        "data_readiness_gate_passed": readiness["ready"],
        "authorization_confirmed_record_count": readiness[
            "authorized_record_count"
        ],
        "single_human_reviewed_action_record_count": readiness[
            "single_human_reviewed_action_record_count"
        ],
        "single_human_reviewed_interval_record_count": readiness[
            "single_human_reviewed_interval_record_count"
        ],
        "independent_human_reviewer_count": readiness[
            "independent_human_reviewer_count"
        ],
        "subject_identity_waiver_applied": readiness[
            "subject_identity_waiver_applied"
        ],
        "training_performed": False,
        "production_improvement_claimed": False,
        "offline_engineering_closed_loop_count": sum(bool(item["engineering_closed_loop"]) for item in offline_records),
        "offline_reviewed_truth_closed_loop_count": 0,
        "artifact_hashes": {
            ablation_path.name: file_sha256(ablation_path),
            failure_path.name: file_sha256(failure_path),
        },
        "current_manual_workflow_blockers": [
            *(
                []
                if readiness["authorized_record_count"] == readiness["record_count"]
                else ["usage_authorization_incomplete"]
            ),
            *(
                []
                if readiness["single_human_reviewed_action_record_count"]
                == readiness["record_count"]
                else ["single_human_manual_action_review_incomplete"]
            ),
        ],
        "deferred_tasks": [
            "automatic_action_recognition_training_and_gating",
            "idle_transition_unknown_and_continuous_switch_data_collection",
            "unknown_ood_rejection_and_switch_latency_evaluation",
        ],
        "deferred_action_gate_training_blockers": training_blockers,
        "release_blockers": [
            *(
                []
                if readiness["authorized_record_count"] == readiness["record_count"]
                else ["usage_authorization_incomplete"]
            ),
            *(
                []
                if readiness["single_human_reviewed_action_record_count"]
                == readiness["record_count"]
                else ["single_human_manual_action_review_incomplete"]
            ),
        ],
    }
    summary["training_performed"] = bool(training["performed"])
    summary["action_gate_training_status"] = training["status"]
    summary["action_gate_model_path"] = training.get("model_path")
    summary["action_gate_model_hash"] = training.get("model_hash")
    summary_path = write_json(report_dir / "round10_implementation_summary.json", summary)
    return {
        "ablation": ablation_path,
        "failure_pool": failure_path,
        "implementation_summary": summary_path,
    }


__all__ = [
    "build_round10_reports",
    "evaluate_data_readiness",
    "extract_action_training_samples",
    "train_action_gate_if_ready",
]
