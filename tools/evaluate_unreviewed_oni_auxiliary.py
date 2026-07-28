"""Evaluate existing ONI Depth/IR as an isolated unreviewed auxiliary source.

The experiment deliberately does not turn recording intent, automatic subject
proposals, or IR imagery into ground truth. It measures coarse evidence and
scenario coverage alongside the reviewed phone-RGB regression while leaving
the RGB runtime and its reported accuracy unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMS = ("depth", "ir")
PHONE_METRICS = (
    "human_rep_count",
    "predicted_candidate_count",
    "exact_count_record_count",
    "exact_count_and_status_record_count",
    "matched_rep_status_count",
    "detected_supported_error_record_count",
    "supported_expected_error_record_count",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _candidate(row: Mapping[str, object]) -> bool:
    return row.get("target_lock_status") == "automated_candidate"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def _stream_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    candidates = [row for row in rows if _candidate(row)]
    centers_x: list[float] = []
    centers_y: list[float] = []
    heights: list[float] = []
    for row in candidates:
        bbox = row.get("bbox_normalized")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        centers_x.append((x1 + x2) / 2.0)
        centers_y.append((y1 + y2) / 2.0)
        heights.append(y2 - y1)
    return {
        "sampled_checkpoint_count": len(rows),
        "automatic_candidate_count": len(candidates),
        "candidate_rate": (
            round(len(candidates) / len(rows), 6) if rows else 0.0
        ),
        "candidate_confidence_p50": _median(
            [float(row.get("confidence", 0.0) or 0.0) for row in candidates]
        ),
        "coarse_motion_ranges": {
            "bbox_center_x": (
                round(max(centers_x) - min(centers_x), 6)
                if centers_x
                else None
            ),
            "bbox_center_y": (
                round(max(centers_y) - min(centers_y), 6)
                if centers_y
                else None
            ),
            "bbox_height": (
                round(max(heights) - min(heights), 6)
                if heights
                else None
            ),
        },
    }


def _record_summary(
    dataset_root: Path,
    manifest_record: Mapping[str, object],
    audit_record: Mapping[str, object],
) -> dict[str, object]:
    record_id = str(manifest_record["record_id"])
    if manifest_record.get("paired_group_id") is not None:
        raise ValueError(f"{record_id}: phone-ONI pairing is forbidden")
    if manifest_record.get("recording_intent_verified") is not False:
        raise ValueError(f"{record_id}: experiment requires unverified intent")

    metadata_path = dataset_root / "extracted" / record_id / "metadata.json"
    metadata = _load(metadata_path)
    streams = metadata.get("streams")
    if not isinstance(streams, Mapping):
        raise ValueError(f"{record_id}: missing exported stream metadata")
    color = streams.get("color")
    if not isinstance(color, Mapping) or color.get("exists") is not False:
        raise ValueError(f"{record_id}: current experiment must not read Color")

    rows_by_stream: dict[str, list[dict[str, Any]]] = {}
    stream_results: dict[str, dict[str, object]] = {}
    track_sources: dict[str, dict[str, str]] = {}
    for modality in STREAMS:
        track_path = (
            dataset_root
            / "oni_tracks"
            / record_id
            / f"{modality}_target_proposals.jsonl"
        )
        rows = _load_jsonl(track_path)
        if any(row.get("modality") != modality for row in rows):
            raise ValueError(f"{record_id}: mixed modality in {track_path}")
        if any(row.get("human_confirmed") is not False for row in rows):
            raise ValueError(f"{record_id}: expected unreviewed proposals only")
        rows_by_stream[modality] = rows
        stream_results[modality] = _stream_summary(rows)
        track_sources[modality] = {
            "path": _relative(track_path, dataset_root),
            "sha256": _sha256(track_path),
        }

    depth_by_frame = {
        int(row["source_frame_index"]): row
        for row in rows_by_stream["depth"]
    }
    ir_by_frame = {
        int(row["source_frame_index"]): row for row in rows_by_stream["ir"]
    }
    common_frames = sorted(depth_by_frame.keys() & ir_by_frame.keys())
    agreements = [
        _candidate(depth_by_frame[index]) == _candidate(ir_by_frame[index])
        for index in common_frames
    ]
    dual_candidates = [
        _candidate(depth_by_frame[index]) and _candidate(ir_by_frame[index])
        for index in common_frames
    ]
    weak_errors = sorted(
        {
            str(code)
            for code in (manifest_record.get("expected_errors_unverified") or [])
            if code
        }
    )
    audit_modalities = audit_record.get("modalities")
    if not isinstance(audit_modalities, Mapping):
        raise ValueError(f"{record_id}: missing ONI audit modalities")

    return {
        "record_id": record_id,
        "action": str(manifest_record.get("action", "")),
        "recording_intent": str(
            manifest_record.get("recording_intent_code", "")
        ),
        "recording_intent_verified": False,
        "expected_errors_unverified": weak_errors,
        "subject_identity_confirmed": False,
        "human_review_complete": all(
            bool(
                isinstance(audit_modalities.get(modality), Mapping)
                and audit_modalities[modality].get("human_review_complete")
            )
            for modality in STREAMS
        ),
        "metadata": {
            "path": _relative(metadata_path, dataset_root),
            "sha256": _sha256(metadata_path),
            "color_present": False,
            "depth_present": bool(
                isinstance(streams.get("depth"), Mapping)
                and streams["depth"].get("exists")
            ),
            "ir_present": bool(
                isinstance(streams.get("ir"), Mapping)
                and streams["ir"].get("exists")
            ),
        },
        "track_sources": track_sources,
        "modalities": stream_results,
        "cross_modality_temporal_presence": {
            "matched_checkpoint_count": len(common_frames),
            "candidate_presence_agreement_rate": (
                round(sum(agreements) / len(agreements), 6)
                if agreements
                else 0.0
            ),
            "dual_candidate_checkpoint_count": sum(dual_candidates),
            "dual_candidate_rate": (
                round(sum(dual_candidates) / len(dual_candidates), 6)
                if dual_candidates
                else 0.0
            ),
            "scope": (
                "timestamp/index-level independent proposal presence only; "
                "no pixel registration or identity confirmation"
            ),
        },
        "unreviewed": True,
        "training_eligible": False,
        "release_eligible": False,
        "runtime_eligible": False,
    }


def build_experiment_report(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    phone_path = (
        root
        / "reports"
        / "reviewed_phone_rgb_guidance_evaluation_optimized_v1.json"
    )
    manifest_path = root / "manifests" / "oni_records.json"
    audit_path = root / "reports" / "oni_subject_audit_v1.json"
    observability_path = (
        root / "reports" / "oni_modality_observability_v1.json"
    )
    fine_path = root / "reviews" / "human_rgb_fine_annotations_v1.json"

    phone = _load(phone_path)
    manifest = _load(manifest_path)
    audit = _load(audit_path)
    observability = _load(observability_path)
    fine = _load(fine_path)
    if phone.get("oni_used") is not False:
        raise ValueError("phone benchmark must remain ONI-free")
    if fine.get("oni_records_included") is not False:
        raise ValueError("formal RGB annotations must remain ONI-free")
    if audit.get("release_or_training_eligible_record_count") != 0:
        raise ValueError("unreviewed ONI records must not be promoted")

    manifest_records = {
        str(record["record_id"]): record
        for record in manifest.get("records") or []
        if isinstance(record, Mapping)
    }
    audit_records = {
        str(record["record_id"]): record
        for record in audit.get("records") or []
        if isinstance(record, Mapping)
    }
    if manifest_records.keys() != audit_records.keys():
        raise ValueError("ONI manifest and subject-audit records differ")
    records = [
        _record_summary(root, manifest_records[record_id], audit_records[record_id])
        for record_id in sorted(manifest_records)
    ]

    phone_actions = {
        str(record.get("action"))
        for record in fine.get("records") or []
        if isinstance(record, Mapping)
    }
    phone_error_codes = {
        str(interval.get("error_code"))
        for record in fine.get("records") or []
        if isinstance(record, Mapping)
        for interval in record.get("phase_error_intervals") or []
        if isinstance(interval, Mapping)
        and interval.get("error_code") not in (None, "NO_ERROR")
    }
    oni_actions = {str(record["action"]) for record in records}
    oni_weak_error_codes = {
        code
        for record in records
        for code in record["expected_errors_unverified"]
    }

    sampled_checkpoints = sum(
        int(record["modalities"][modality]["sampled_checkpoint_count"])
        for record in records
        for modality in STREAMS
    )
    automatic_candidates = sum(
        int(record["modalities"][modality]["automatic_candidate_count"])
        for record in records
        for modality in STREAMS
    )
    presence_agreement_rates = [
        float(
            record["cross_modality_temporal_presence"][
                "candidate_presence_agreement_rate"
            ]
        )
        for record in records
    ]
    dual_candidate_rates = [
        float(
            record["cross_modality_temporal_presence"]["dual_candidate_rate"]
        )
        for record in records
    ]
    phone_metrics = {
        metric: int(phone[metric]) for metric in PHONE_METRICS
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "unreviewed_oni_auxiliary_experiment_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "reviewed_phone_rgb_plus_unreviewed_oni_depth_ir",
        "oni_used": True,
        "oni_use_scope": (
            "offline coarse-motion and scenario-coverage auxiliary experiment"
        ),
        "safety_gates": {
            "unreviewed": True,
            "training_eligible": False,
            "release_eligible": False,
            "runtime_eligible": False,
            "runtime_defaults_changed": False,
            "ir_treated_as_rgb": False,
            "phone_oni_pairing_created": False,
            "automatic_subject_proposals_promoted_to_truth": False,
            "recording_intent_promoted_to_truth": False,
        },
        "sources": {
            "reviewed_phone_evaluation": {
                "path": _relative(phone_path, root),
                "sha256": _sha256(phone_path),
            },
            "reviewed_phone_annotations": {
                "path": _relative(fine_path, root),
                "sha256": _sha256(fine_path),
            },
            "oni_manifest": {
                "path": _relative(manifest_path, root),
                "sha256": _sha256(manifest_path),
            },
            "oni_subject_audit": {
                "path": _relative(audit_path, root),
                "sha256": _sha256(audit_path),
            },
            "oni_modality_observability": {
                "path": _relative(observability_path, root),
                "sha256": _sha256(observability_path),
                "conclusion": observability.get("conclusion"),
            },
        },
        "oni_summary": {
            "record_count": len(records),
            "stream_count": len(records) * len(STREAMS),
            "sampled_checkpoint_count": sampled_checkpoints,
            "automatic_candidate_count": automatic_candidates,
            "human_reviewed_modality_count": int(
                audit.get("human_reviewed_modality_count", 0)
            ),
            "human_confirmed_target_count": int(
                audit.get("human_confirmed_target_count", 0)
            ),
            "weak_error_intent_record_count": sum(
                bool(record["expected_errors_unverified"]) for record in records
            ),
            "action_count": len(oni_actions),
            "unverified_error_code_count": len(oni_weak_error_codes),
            "cross_modality_candidate_presence_agreement": {
                "p50": _median(presence_agreement_rates),
                "minimum": (
                    round(min(presence_agreement_rates), 6)
                    if presence_agreement_rates
                    else None
                ),
                "maximum": (
                    round(max(presence_agreement_rates), 6)
                    if presence_agreement_rates
                    else None
                ),
            },
            "dual_candidate_rate_p50": _median(dual_candidate_rates),
        },
        "comparison": {
            "measured_phone_guidance_metrics": {
                metric: {
                    "reviewed_rgb_only": value,
                    "rgb_plus_unreviewed_oni": value,
                    "delta": 0,
                }
                for metric, value in phone_metrics.items()
            },
            "research_scenario_coverage": {
                "action_count": {
                    "reviewed_rgb_only": len(phone_actions),
                    "rgb_plus_unreviewed_oni": len(
                        phone_actions | oni_actions
                    ),
                    "delta": len((phone_actions | oni_actions) - phone_actions),
                },
                "distinct_error_code_count": {
                    "reviewed_rgb_only": len(phone_error_codes),
                    "rgb_plus_unreviewed_oni": len(
                        phone_error_codes | oni_weak_error_codes
                    ),
                    "delta": len(
                        (phone_error_codes | oni_weak_error_codes)
                        - phone_error_codes
                    ),
                    "oni_only_codes_unverified": sorted(
                        oni_weak_error_codes - phone_error_codes
                    ),
                },
            },
        },
        "verdict": {
            "measured_guidance_accuracy_improved": False,
            "research_coverage_improved": bool(
                (oni_actions - phone_actions)
                or (oni_weak_error_codes - phone_error_codes)
            ),
            "safe_to_enable_in_production": False,
            "reason": (
                "Existing ONI adds independent Depth/IR coarse-motion and weak "
                "scenario coverage, but it has no Color, confirmed subject "
                "identity, reviewed rep boundaries, or reviewed error truth. "
                "It therefore cannot provide a defensible phone-RGB accuracy "
                "gain in this experiment."
            ),
            "next_evidence_needed": [
                "complete independent Depth and IR subject review",
                "review rep boundaries and record-level intended errors",
                "collect synchronized calibrated RGB-D for direct RGB guidance calibration",
            ],
        },
        "records": records,
    }
    if automatic_candidates != int(
        audit.get("accepted_automatic_candidate_count", -1)
    ):
        raise ValueError(
            "automatic candidate count does not match ONI subject audit"
        )
    return _write(
        root / "reports" / "unreviewed_oni_auxiliary_experiment_v1.json",
        payload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare reviewed phone RGB with isolated unreviewed ONI "
            "Depth/IR auxiliary evidence."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/hyrox"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.dataset_root
    if not root.is_absolute():
        root = args.project_root.resolve() / root
    output = build_experiment_report(root)
    payload = _load(output)
    print(
        json.dumps(
            {
                "report": str(output),
                "oni_summary": payload["oni_summary"],
                "comparison": payload["comparison"],
                "verdict": payload["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
