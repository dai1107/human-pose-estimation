"""Build and validate the round-two ONI record manifest.

The builder preserves the original Chinese filenames, creates an independent
full backup by default, and never assumes a phone-video pairing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
ONI_SOURCE_TYPE = "oni"
PHONE_SOURCE_TYPE = "phone_rgb"
PHONE_SOURCE_TYPE_ALIASES = ("phone_rgb_future",)

ACTION_NAMES = {
    "负重箭步蹲": "lunge",
    "波比跳": "burpee_broad_jump",
    "墙球": "wall_ball",
    "划船机": "rowing",
    "滑雪机": "skierg",
    "推雪橇": "sled_push",
    "拉雪橇": "sled_pull",
    "农夫行走": "farmers_carry",
}

CAMERA_VIEWS = {
    "斜前方": "oblique_front",
    "斜后方": "oblique_rear",
    "侧方": "side",
    "正面": "front",
    "后方": "rear",
}

INTENT_CODES = {
    "甩壶铃": "kettlebell_swing",
    "桨把绕膝": "handle_around_knees",
    "身体后仰": "lean_back",
    "下蹲不足": "not_deep_enough",
    "踮脚": "heel_rise",
    "双脚不同步": "foot_desynchronization",
    "手脚过远": "hands_feet_too_far",
    "胸部未触地": "no_chest_contact",
    "落地补步、碎步": "extra_landing_steps",
    "同腿连续": "same_leg_consecutive",
    "后膝未触地": "no_knee_contact",
    "补步、碎步": "extra_steps",
    "髋部未伸展": "hip_not_extended",
}

UNVERIFIED_ERROR_CODES = {
    "甩壶铃": ("KETTLEBELL_SWING",),
    "桨把绕膝": ("HANDLE_AROUND_KNEES",),
    "身体后仰": ("LEAN_TOO_MUCH",),
    "下蹲不足": ("NOT_DEEP_ENOUGH",),
    "踮脚": ("HEEL_RISE",),
    "双脚不同步": ("FOOT_DESYNCHRONIZED",),
    "手脚过远": ("HANDS_FEET_TOO_FAR",),
    "胸部未触地": ("NO_CHEST_CONTACT",),
    "落地补步、碎步": ("EXTRA_STEP",),
    "同腿连续": ("SAME_LEG_CONSECUTIVE",),
    "后膝未触地": ("NO_KNEE_CONTACT",),
    "补步、碎步": ("EXTRA_STEP",),
    "髋部未伸展": ("HIP_NOT_EXTENDED",),
}

DATASET_DIRECTORIES = (
    "raw/oni",
    "raw/phone_rgb",
    "manifests",
    "extracted",
    "synchronized",
    "registered",
    "pose_cache",
    "annotations",
    "clips",
    "splits",
    "reports",
)

RECORD_ID_PATTERN = re.compile(r"^oni_([a-z0-9_]+)_(\d{3,})$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TRAILING_TAKE_PATTERN = re.compile(r"^(.*?)(\d+)$")
VARIANT_PATTERN = re.compile(r"^标准（(.+)）$")


@dataclass(frozen=True)
class ParsedOniFilename:
    source_filename: str
    action: str
    action_raw: str
    camera_view: str
    camera_view_raw: str
    recording_intent: str
    recording_intent_code: str
    recording_intent_raw: str
    recording_intent_base_raw: str
    recording_variant_raw: str | None
    take_index: int | None
    expected_errors_unverified: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_oni_filename(filename: str) -> ParsedOniFilename:
    path = Path(filename)
    if path.suffix.casefold() != ".oni":
        raise ValueError(f"ONI filename must end in .oni: {filename}")
    parts = path.stem.split("-")
    if len(parts) < 3:
        raise ValueError(
            "ONI filename must follow 动作-录制意图-视角.oni: "
            f"{filename}"
        )
    action_raw = parts[0].strip()
    camera_view_raw = parts[-1].strip()
    intent_raw = "-".join(parts[1:-1]).strip()
    if action_raw not in ACTION_NAMES:
        raise ValueError(f"unknown ONI action prefix {action_raw!r}: {filename}")
    if camera_view_raw not in CAMERA_VIEWS:
        raise ValueError(
            f"unknown ONI camera-view suffix {camera_view_raw!r}: {filename}"
        )
    if not intent_raw:
        raise ValueError(f"empty ONI recording intent: {filename}")

    take_index: int | None = None
    intent_base = intent_raw
    trailing = TRAILING_TAKE_PATTERN.fullmatch(intent_raw)
    if trailing and trailing.group(1):
        intent_base = trailing.group(1)
        take_index = int(trailing.group(2))

    variant_match = VARIANT_PATTERN.fullmatch(intent_base)
    variant = variant_match.group(1) if variant_match else None
    is_standard = intent_base == "标准" or variant_match is not None
    if is_standard:
        intent_code = "standard"
        expected_errors: tuple[str, ...] = ()
    else:
        intent_code = INTENT_CODES.get(intent_base, "unmapped_error_intent")
        expected_errors = UNVERIFIED_ERROR_CODES.get(intent_base, ())

    return ParsedOniFilename(
        source_filename=path.name,
        action=ACTION_NAMES[action_raw],
        action_raw=action_raw,
        camera_view=CAMERA_VIEWS[camera_view_raw],
        camera_view_raw=camera_view_raw,
        recording_intent="standard" if is_standard else "error",
        recording_intent_code=intent_code,
        recording_intent_raw=intent_raw,
        recording_intent_base_raw=intent_base,
        recording_variant_raw=variant,
        take_index=take_index,
        expected_errors_unverified=expected_errors,
    )


def set_read_only(path: str | Path) -> None:
    target = Path(path)
    target.chmod(target.stat().st_mode & ~stat.S_IWRITE)


def is_read_only(path: str | Path) -> bool:
    return not bool(Path(path).stat().st_mode & stat.S_IWRITE)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_existing_ids(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        return {}
    output: dict[str, str] = {}
    for record in records:
        if isinstance(record, Mapping):
            filename = str(record.get("source_filename", ""))
            record_id = str(record.get("record_id", ""))
            if filename and RECORD_ID_PATTERN.fullmatch(record_id):
                output[filename] = record_id
    return output


def _assign_record_ids(
    parsed_files: Sequence[ParsedOniFilename],
    existing_ids: Mapping[str, str],
) -> dict[str, str]:
    assigned: dict[str, str] = {}
    used = set(existing_ids.values())
    next_index: Counter[str] = Counter()
    for record_id in used:
        match = RECORD_ID_PATTERN.fullmatch(record_id)
        if match:
            next_index[match.group(1)] = max(
                next_index[match.group(1)], int(match.group(2))
            )
    for parsed in sorted(
        parsed_files, key=lambda item: (item.action, item.source_filename)
    ):
        previous = existing_ids.get(parsed.source_filename)
        if previous:
            assigned[parsed.source_filename] = previous
            continue
        index = next_index[parsed.action] + 1
        candidate = f"oni_{parsed.action}_{index:03d}"
        while candidate in used:
            index += 1
            candidate = f"oni_{parsed.action}_{index:03d}"
        next_index[parsed.action] = index
        used.add(candidate)
        assigned[parsed.source_filename] = candidate
    return assigned


def _ensure_dataset_directories(dataset_root: Path) -> None:
    for relative in DATASET_DIRECTORIES:
        (dataset_root / relative).mkdir(parents=True, exist_ok=True)


def _backup_and_verify(
    source: Path,
    destination: Path,
    *,
    copy_files: bool,
    mark_source_read_only: bool,
) -> dict[str, Any]:
    source_hash = sha256_file(source)
    destination_hash: str | None = None
    copy_status = "reference_only"
    if copy_files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size:
                raise RuntimeError(
                    f"existing backup size mismatch; refusing overwrite: {destination}"
                )
            copy_status = "existing_verified"
        else:
            shutil.copy2(source, destination)
            copy_status = "copied"
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                f"backup hash mismatch; source preserved and backup not overwritten: "
                f"{source.name}"
            )
        set_read_only(destination)
    if mark_source_read_only:
        set_read_only(source)
    return {
        "sha256": source_hash,
        "backup_sha256": destination_hash,
        "backup_status": copy_status,
        "backup_verified": bool(copy_files and destination_hash == source_hash),
        "source_read_only": is_read_only(source),
        "backup_read_only": is_read_only(destination)
        if copy_files and destination.exists()
        else None,
    }


def record_schema() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "hyrox_record_schema",
        "source_types": [ONI_SOURCE_TYPE, PHONE_SOURCE_TYPE],
        "read_compatibility_aliases": {
            "phone_rgb_future": PHONE_SOURCE_TYPE,
        },
        "required_record_fields": [
            "record_id",
            "source_type",
            "source_filename",
            "source_file",
            "action",
            "action_raw",
            "subject_id",
            "camera_view",
            "camera_view_raw",
            "recording_intent",
            "recording_intent_raw",
            "recording_intent_verified",
            "confirmed_labels",
            "paired_group_id",
            "target_athlete",
            "other_people_present",
            "sha256",
            "usage_authorization",
        ],
        "phone_interface": {
            "currently_required": True,
            "paired_with_current_oni": False,
            "independent_record_id_required": True,
            "independent_timeline_required": True,
        },
        "label_contract": {
            "recording_intent_is_ground_truth": False,
            "confirmed_labels_require_annotation": True,
        },
    }


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    expected_source_type: str = ONI_SOURCE_TYPE,
    dataset_root: str | Path | None = None,
    verify_files: bool = False,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        return [*errors, "records must be a list"]
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    required = set(record_schema()["required_record_fields"])
    root = Path(dataset_root) if dataset_root is not None else None
    for index, raw in enumerate(records):
        prefix = f"records[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(field for field in required if field not in raw)
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")
            continue
        record_id = str(raw.get("record_id"))
        filename = str(raw.get("source_filename"))
        if record_id in seen_ids:
            errors.append(f"{prefix} duplicate record_id: {record_id}")
        if filename in seen_names:
            errors.append(f"{prefix} duplicate source_filename: {filename}")
        seen_ids.add(record_id)
        seen_names.add(filename)
        if raw.get("source_type") != expected_source_type:
            errors.append(
                f"{prefix} source_type must be {expected_source_type!r}"
            )
        if raw.get("paired_group_id") is not None:
            errors.append(
                f"{prefix} paired_group_id must be null for current independent data"
            )
        if raw.get("recording_intent_verified") is not False:
            errors.append(
                f"{prefix} recording_intent_verified must remain false before annotation"
            )
        if raw.get("confirmed_labels") != []:
            errors.append(
                f"{prefix} confirmed_labels must be empty before annotation"
            )
        target = raw.get("target_athlete")
        if not isinstance(target, Mapping):
            errors.append(f"{prefix} target_athlete must be an object")
        elif target.get("selection_status") not in {
            "pending",
            "selected",
            "ambiguous",
        }:
            errors.append(f"{prefix} invalid target athlete selection_status")
        digest = str(raw.get("sha256", ""))
        if not SHA256_PATTERN.fullmatch(digest):
            errors.append(f"{prefix} sha256 must contain 64 lowercase hex digits")
        if expected_source_type == ONI_SOURCE_TYPE:
            if not RECORD_ID_PATTERN.fullmatch(record_id):
                errors.append(f"{prefix} invalid ONI record_id: {record_id}")
            if not filename.casefold().endswith(".oni"):
                errors.append(f"{prefix} ONI source_filename must end in .oni")
            if not str(raw.get("source_file", "")).startswith("raw/oni/"):
                errors.append(f"{prefix} ONI source_file must be below raw/oni")
        if root is not None and verify_files:
            file_path = root / str(raw.get("source_file"))
            if not file_path.is_file():
                errors.append(f"{prefix} backup missing: {file_path}")
            else:
                if file_path.stat().st_size != int(raw.get("size_bytes", -1)):
                    errors.append(f"{prefix} backup size mismatch")
                if sha256_file(file_path) != digest:
                    errors.append(f"{prefix} backup hash mismatch")
                if not is_read_only(file_path):
                    errors.append(f"{prefix} backup is not read-only")
    return errors


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records),
        "actions": dict(sorted(Counter(str(record["action"]) for record in records).items())),
        "camera_views": dict(
            sorted(Counter(str(record["camera_view"]) for record in records).items())
        ),
        "recording_intents": dict(
            sorted(
                Counter(str(record["recording_intent"]) for record in records).items()
            )
        ),
        "source_read_only_count": sum(
            bool(record["integrity"]["source_read_only"]) for record in records
        ),
        "backup_read_only_count": sum(
            bool(record["integrity"]["backup_read_only"]) for record in records
        ),
        "backup_hash_verified_count": sum(
            bool(record["integrity"]["backup_verified"]) for record in records
        ),
        "phone_record_count": 0,
        "paired_group_count": 0,
        "subject_pending_count": sum(
            record["subject_id"] == "subject_pending" for record in records
        ),
        "authorization_pending_count": sum(
            record["usage_authorization"]["status"] == "pending_confirmation"
            for record in records
        ),
        "target_selection_pending_count": sum(
            record["target_athlete"]["selection_status"] == "pending"
            for record in records
        ),
    }


def build_dataset_manifest(
    source_dir: str | Path,
    dataset_root: str | Path,
    *,
    project_root: str | Path | None = None,
    copy_files: bool = True,
    mark_source_read_only: bool = True,
) -> dict[str, Any]:
    source_root = Path(source_dir).resolve()
    output_root = Path(dataset_root).resolve()
    repository_root = (
        Path(project_root).resolve()
        if project_root is not None
        else source_root.parent.resolve()
    )
    if not source_root.is_dir():
        raise FileNotFoundError(f"ONI source directory not found: {source_root}")
    source_files = sorted(
        (path for path in source_root.iterdir() if path.is_file() and path.suffix.casefold() == ".oni"),
        key=lambda path: path.name,
    )
    if not source_files:
        raise FileNotFoundError(f"no ONI files found: {source_root}")

    _ensure_dataset_directories(output_root)
    manifest_path = output_root / "manifests" / "oni_records.json"
    existing_ids = _load_existing_ids(manifest_path)
    parsed_files = [parse_oni_filename(path.name) for path in source_files]
    assigned_ids = _assign_record_ids(parsed_files, existing_ids)
    parsed_by_name = {item.source_filename: item for item in parsed_files}
    records: list[dict[str, Any]] = []

    for source in source_files:
        parsed = parsed_by_name[source.name]
        destination = output_root / "raw" / "oni" / source.name
        integrity = _backup_and_verify(
            source,
            destination,
            copy_files=copy_files,
            mark_source_read_only=mark_source_read_only,
        )
        stat_result = source.stat()
        record = {
            "record_id": assigned_ids[source.name],
            "source_type": ONI_SOURCE_TYPE,
            "source_filename": source.name,
            "source_original": _relative_or_absolute(source, repository_root),
            "source_file": f"raw/oni/{source.name}",
            "action": parsed.action,
            "action_raw": parsed.action_raw,
            "subject_id": "subject_pending",
            "camera_view": parsed.camera_view,
            "camera_view_raw": parsed.camera_view_raw,
            "recording_intent": parsed.recording_intent,
            "recording_intent_code": parsed.recording_intent_code,
            "recording_intent_raw": parsed.recording_intent_raw,
            "recording_intent_base_raw": parsed.recording_intent_base_raw,
            "recording_variant_raw": parsed.recording_variant_raw,
            "take_index": parsed.take_index,
            "recording_intent_verified": False,
            "expected_errors_unverified": list(
                parsed.expected_errors_unverified
            ),
            "confirmed_labels": [],
            "paired_group_id": None,
            "target_athlete": {
                "track_id": None,
                "selection_status": "pending",
                "selection_method": None,
                "identity_switch_segments": [],
                "other_people_risk_acknowledged": True,
            },
            "other_people_present": "yes_or_possible",
            "full_body_visible": "unknown",
            "floor_visible": "unknown",
            "size_bytes": stat_result.st_size,
            "mtime_utc": datetime.fromtimestamp(
                stat_result.st_mtime, timezone.utc
            ).isoformat(),
            "sha256": integrity["sha256"],
            "integrity": integrity,
            "usage_authorization": {
                "status": "pending_confirmation",
                "recorded_by": "",
                "authorized_uses": [],
                "notes": "Must be confirmed before external distribution or training release.",
            },
            "notes": "",
        }
        records.append(record)

    records.sort(key=lambda item: str(item["record_id"]))
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "hyrox_oni_record_manifest",
        "generated_at": utc_now(),
        "source_directory": _relative_or_absolute(source_root, repository_root),
        "dataset_root": _relative_or_absolute(output_root, repository_root),
        "copy_policy": "independent_full_copy" if copy_files else "reference_only",
        "phone_data_status": "managed_by_independent_phone_rgb_manifest",
        "phone_pairing_policy": "forbidden_for_current_oni",
        "records": records,
        "summary": _summary(records),
    }
    errors = validate_manifest(
        payload,
        expected_source_type=ONI_SOURCE_TYPE,
        dataset_root=output_root,
        verify_files=False,
    )
    if errors:
        raise ValueError("generated ONI manifest is invalid: " + "; ".join(errors))

    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    phone_manifest_path = output_root / "manifests" / "phone_records.json"
    if not phone_manifest_path.exists():
        phone_payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_type": "hyrox_phone_record_manifest",
            "generated_at": utc_now(),
            "source_type": PHONE_SOURCE_TYPE,
            "data_status": "awaiting_phone_rgb_ingest",
            "paired_with_current_oni": False,
            "records": [],
        }
        phone_manifest_path.write_text(
            json.dumps(phone_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_root / "manifests" / "record_schema.json").write_text(
        json.dumps(record_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def validation_report(
    payload: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    verify_files: bool = False,
) -> dict[str, Any]:
    errors = validate_manifest(
        payload,
        expected_source_type=ONI_SOURCE_TYPE,
        dataset_root=dataset_root,
        verify_files=verify_files,
    )
    records = payload.get("records")
    record_list = records if isinstance(records, list) else []
    summary = _summary(record_list)
    warnings: list[str] = []
    if summary["subject_pending_count"]:
        warnings.append("subject_id remains pending and must be confirmed before dataset splitting")
    if summary["authorization_pending_count"]:
        warnings.append("usage authorization remains pending confirmation")
    if summary["target_selection_pending_count"]:
        warnings.append("target athlete selection is deferred to the multi-person audit round")
    checks = {
        "unique_record_ids": not any("duplicate record_id" in item for item in errors),
        "unique_source_filenames": not any("duplicate source_filename" in item for item in errors),
        "all_sha256_present": all(
            SHA256_PATTERN.fullmatch(str(record.get("sha256", "")))
            for record in record_list
            if isinstance(record, Mapping)
        ),
        "all_backups_hash_verified": summary["backup_hash_verified_count"] == len(record_list),
        "all_sources_read_only": summary["source_read_only_count"] == len(record_list),
        "all_backups_read_only": summary["backup_read_only_count"] == len(record_list),
        "oni_manifest_contains_no_phone_records": summary["phone_record_count"] == 0,
        "no_phone_oni_pairing": summary["paired_group_count"] == 0,
        "recording_intent_unverified": all(
            record.get("recording_intent_verified") is False
            and record.get("confirmed_labels") == []
            for record in record_list
            if isinstance(record, Mapping)
        ),
        "target_athlete_fields_present": all(
            isinstance(record.get("target_athlete"), Mapping)
            for record in record_list
            if isinstance(record, Mapping)
        ),
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "hyrox_round2_validation_report",
        "generated_at": utc_now(),
        "status": "passed" if not errors and all(checks.values()) else "failed",
        "file_verification_performed": bool(verify_files),
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "checks": checks,
    }


__all__ = [
    "ACTION_NAMES",
    "CAMERA_VIEWS",
    "DATASET_DIRECTORIES",
    "MANIFEST_SCHEMA_VERSION",
    "ONI_SOURCE_TYPE",
    "PHONE_SOURCE_TYPE",
    "PHONE_SOURCE_TYPE_ALIASES",
    "ParsedOniFilename",
    "build_dataset_manifest",
    "is_read_only",
    "parse_oni_filename",
    "record_schema",
    "set_read_only",
    "sha256_file",
    "validate_manifest",
    "validation_report",
]
