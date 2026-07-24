"""Round-six ingestion and governance reports for independent phone RGB data."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2

from .manifest import (
    ACTION_NAMES,
    INTENT_CODES,
    MANIFEST_SCHEMA_VERSION,
    PHONE_SOURCE_TYPE,
    PHONE_SOURCE_TYPE_ALIASES,
    SHA256_PATTERN,
    UNVERIFIED_ERROR_CODES,
    is_read_only,
    record_schema,
    set_read_only,
    sha256_file,
    utc_now,
)


PHONE_RECORD_ID_PATTERN = re.compile(r"^phone_([a-z0-9_]+)_(\d{3,})$")
TRAILING_TAKE_PATTERN = re.compile(r"^(.*?)-(\d+)$")
STANDARD_PATTERN = re.compile(r"^标准(?:（.+）)?$")
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v"})

PHONE_CAMERA_VIEWS = {
    "正前方": "front",
    "前方": "front",
    "正后方": "rear",
    "后方": "rear",
    "斜前方": "oblique_front",
    "斜后方": "oblique_rear",
    "侧后方": "oblique_rear",
    "侧方": "side",
}

PHONE_INTENT_CODES = {
    **INTENT_CODES,
    "落地补步": "extra_landing_steps",
}

PHONE_UNVERIFIED_ERRORS = {
    **UNVERIFIED_ERROR_CODES,
    "落地补步": ("EXTRA_STEP",),
}

EXAMPLE_CANDIDATE_FILENAMES = frozenset(
    {
        "波比跳-标准（双脚起）-正前方-1.mp4",
        "负重箭步蹲-标准-正前方-1.mp4",
        "滑雪机-标准-后方-1.mp4",
        "划船机-标准-侧后方-1.mp4",
        "拉雪橇-标准-侧方-2.mp4",
        "农夫行走-标准-侧方-1.mp4",
        "墙球-标准-正后方-1.mp4",
        "推雪橇-标准-侧方-3.mp4",
    }
)

ACTION_EQUIPMENT = {
    "burpee_broad_jump": ("floor_or_lane",),
    "lunge": ("kettlebell_or_sandbag", "floor_or_lane"),
    "skierg": ("ergometer", "handle", "display_region"),
    "rowing": ("ergometer", "handle", "display_region"),
    "sled_pull": ("sled", "rope", "floor_or_lane"),
    "farmers_carry": ("kettlebell_or_sandbag", "floor_or_lane"),
    "wall_ball": ("ball", "target_board", "floor_or_lane"),
    "sled_push": ("sled", "floor_or_lane"),
}

POSE_ONLY_UNOBSERVABLE_RULES = {
    "burpee_broad_jump": [
        "chest_floor_contact_without_floor/contact evidence",
        "measured broad-jump distance",
    ],
    "lunge": ["actual carried load", "rear-knee floor contact without floor/contact evidence"],
    "skierg": ["ergometer resistance", "displayed distance/calories"],
    "rowing": ["ergometer resistance", "displayed distance/calories"],
    "sled_pull": ["actual sled load", "rope/handle state", "sled crossing the finish line"],
    "farmers_carry": ["actual carried load", "measured course distance"],
    "wall_ball": ["actual ball weight", "target hit", "target height"],
    "sled_push": ["actual sled load", "sled crossing the finish line"],
}


@dataclass(frozen=True)
class ParsedPhoneFilename:
    source_filename: str
    action: str
    action_raw: str
    camera_view: str
    camera_view_raw: str
    recording_intent: str
    recording_intent_code: str
    recording_intent_raw: str
    take_index: int | None
    expected_errors_unverified: tuple[str, ...]


def is_appledouble(path: str | Path) -> bool:
    return Path(path).name.startswith("._")


def parse_phone_filename(filename: str) -> ParsedPhoneFilename:
    path = Path(filename)
    if path.suffix.casefold() not in VIDEO_SUFFIXES:
        raise ValueError(f"unsupported phone RGB video suffix: {filename}")
    take_index: int | None = None
    stem = path.stem
    take_match = TRAILING_TAKE_PATTERN.fullmatch(stem)
    if take_match:
        stem = take_match.group(1)
        take_index = int(take_match.group(2))
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"phone filename must follow 动作-录制意图-视角-序号: {filename}")
    action_raw = parts[0].strip()
    camera_view_raw = parts[-1].strip()
    intent_raw = "-".join(parts[1:-1]).strip()
    if action_raw not in ACTION_NAMES:
        raise ValueError(f"unknown phone action prefix {action_raw!r}: {filename}")
    if camera_view_raw not in PHONE_CAMERA_VIEWS:
        raise ValueError(f"unknown phone camera view {camera_view_raw!r}: {filename}")
    if not intent_raw:
        raise ValueError(f"empty phone recording intent: {filename}")
    is_standard = bool(STANDARD_PATTERN.fullmatch(intent_raw))
    return ParsedPhoneFilename(
        source_filename=path.name,
        action=ACTION_NAMES[action_raw],
        action_raw=action_raw,
        camera_view=PHONE_CAMERA_VIEWS[camera_view_raw],
        camera_view_raw=camera_view_raw,
        recording_intent="standard" if is_standard else "error",
        recording_intent_code=(
            "standard" if is_standard else PHONE_INTENT_CODES.get(intent_raw, "unmapped_error_intent")
        ),
        recording_intent_raw=intent_raw,
        take_index=take_index,
        expected_errors_unverified=(
            () if is_standard else PHONE_UNVERIFIED_ERRORS.get(intent_raw, ())
        ),
    )


def discover_phone_files(source_dir: str | Path) -> tuple[list[Path], list[Path]]:
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"phone RGB source directory not found: {root}")
    files = [path for path in root.rglob("*") if path.is_file()]
    appledouble = sorted(
        (path for path in files if is_appledouble(path) and path.suffix.casefold() in VIDEO_SUFFIXES),
        key=lambda item: item.as_posix(),
    )
    videos = sorted(
        (
            path
            for path in files
            if not is_appledouble(path) and path.suffix.casefold() in VIDEO_SUFFIXES
        ),
        key=lambda item: item.name,
    )
    return videos, appledouble


def _fourcc(value: float) -> str | None:
    if not math.isfinite(value) or value <= 0:
        return None
    integer = int(value)
    rendered = "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))
    return rendered if all(32 <= ord(char) < 127 for char in rendered) else None


def _capture_property(capture: cv2.VideoCapture, name: str) -> float | None:
    prop = getattr(cv2, name, None)
    if prop is None:
        return None
    try:
        value = float(capture.get(prop))
    except (cv2.error, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def decode_phone_video(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        return {
            "status": "failed",
            "error": "opencv_video_capture_open_failed",
            "declared_frame_count": 0,
            "decoded_frame_count": 0,
            "decoded_to_declared_last_frame": False,
            "timestamps_ms": [],
            "abnormal_frames": [],
        }

    declared_frames = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if math.isfinite(fps) and fps > 0 else 0.0
    width = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    codec = _fourcc(capture.get(cv2.CAP_PROP_FOURCC))
    pixel_format = _fourcc(_capture_property(capture, "CAP_PROP_CODEC_PIXEL_FORMAT") or 0.0)
    rotation = _capture_property(capture, "CAP_PROP_ORIENTATION_META")
    audio_streams = _capture_property(capture, "CAP_PROP_AUDIO_TOTAL_STREAMS")
    backend_name = capture.getBackendName() if hasattr(capture, "getBackendName") else "unknown"
    timestamps: list[float] = []
    abnormal_frames: list[dict[str, Any]] = []
    timestamp_fallbacks = 0
    raw_timestamp_previous = -1.0
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index = decoded
            decoded += 1
            if frame.ndim != 3 or frame.shape[2] != 3:
                abnormal_frames.append(
                    {"frame_index": frame_index, "reason": "decoded_frame_not_bgr_3_channel"}
                )
            elif frame.shape[1] != width or frame.shape[0] != height:
                abnormal_frames.append(
                    {
                        "frame_index": frame_index,
                        "reason": "decoded_dimensions_changed",
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
            raw_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            expected = (frame_index * 1000.0 / fps) if fps > 0 else float(frame_index)
            if (
                not math.isfinite(raw_timestamp)
                or raw_timestamp < 0
                or (frame_index > 0 and raw_timestamp <= raw_timestamp_previous)
            ):
                timestamp = expected
                timestamp_fallbacks += 1
            else:
                timestamp = raw_timestamp
                raw_timestamp_previous = raw_timestamp
            timestamps.append(round(float(timestamp), 6))
    finally:
        capture.release()

    reached_declared_last = declared_frames > 0 and decoded == declared_frames
    status = "passed" if reached_declared_last and not abnormal_frames else "failed"
    return {
        "status": status,
        "error": None if status == "passed" else "decode_or_frame_integrity_failure",
        "container": source.suffix.lstrip(".").lower(),
        "opencv_backend": backend_name,
        "declared_frame_count": declared_frames,
        "decoded_frame_count": decoded,
        "declared_last_frame_index": declared_frames - 1 if declared_frames else None,
        "decoded_last_frame_index": decoded - 1 if decoded else None,
        "decoded_to_declared_last_frame": reached_declared_last,
        "fps": fps,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "duration_seconds": round(decoded / fps, 6) if fps > 0 else None,
        "rotation_degrees": int(round(rotation)) if rotation is not None else None,
        "codec_fourcc": codec,
        "codec_pixel_format": pixel_format,
        "source_color_space": "unknown_without_container_color_metadata_parser",
        "decoded_color_space": "BGR8",
        "model_input_color_space": "sRGB_after_explicit_BGR_to_RGB_conversion",
        "audio_track_count": int(round(audio_streams)) if audio_streams is not None and audio_streams >= 0 else None,
        "audio_audit_status": (
            "reported_by_opencv" if audio_streams is not None and audio_streams >= 0 else "unavailable_in_opencv_backend"
        ),
        "timestamp_source": (
            "container_pts" if timestamps and timestamp_fallbacks == 0 else "container_pts_with_fps_fallback"
        ),
        "timestamp_fallback_count": timestamp_fallbacks,
        "timestamps_ms": timestamps,
        "abnormal_frames": abnormal_frames,
    }


def _load_existing_phone_ids(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    output: dict[str, str] = {}
    for record in payload.get("records") or []:
        if not isinstance(record, Mapping):
            continue
        record_id = str(record.get("record_id", ""))
        filename = str(record.get("source_filename", ""))
        if filename and PHONE_RECORD_ID_PATTERN.fullmatch(record_id):
            output[filename] = record_id
    return output


def _assign_phone_ids(
    parsed: Sequence[ParsedPhoneFilename], existing: Mapping[str, str]
) -> dict[str, str]:
    next_indices: Counter[str] = Counter()
    used = set(existing.values())
    for record_id in used:
        match = PHONE_RECORD_ID_PATTERN.fullmatch(record_id)
        if match:
            next_indices[match.group(1)] = max(next_indices[match.group(1)], int(match.group(2)))
    assigned: dict[str, str] = {}
    for item in sorted(parsed, key=lambda value: (value.action, value.source_filename)):
        if item.source_filename in existing:
            assigned[item.source_filename] = existing[item.source_filename]
            continue
        index = next_indices[item.action] + 1
        candidate = f"phone_{item.action}_{index:03d}"
        while candidate in used:
            index += 1
            candidate = f"phone_{item.action}_{index:03d}"
        next_indices[item.action] = index
        used.add(candidate)
        assigned[item.source_filename] = candidate
    return assigned


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def phone_timeline_schema(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "phone_rgb_timeline_schema",
        "source_type": PHONE_SOURCE_TYPE,
        "status": "active_independent_timelines",
        "record_count": len(records),
        "pairing_with_current_oni": "forbidden",
        "required_fields": {
            "record_id": "string",
            "source_frame_index": "integer",
            "timestamp_ms": "number",
            "source_fps": "number",
            "timestamp_source": "container_pts|capture_timestamp|derived_from_fps",
        },
        "timestamp_storage": "datasets/hyrox/reports/phone_rgb_decode_audit.json records[].decode.timestamps_ms",
        "paired_group_id_policy": (
            "null unless a separately designed synchronized acquisition protocol and verifiable sync events are recorded"
        ),
    }


def _copy_and_verify(source: Path, destination: Path, *, mark_source_read_only: bool) -> dict[str, Any]:
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing phone backup size mismatch; refusing overwrite: {destination}")
        copy_status = "existing_verified"
    else:
        shutil.copy2(source, destination)
        copy_status = "copied"
    backup_hash = sha256_file(destination)
    if backup_hash != source_hash:
        raise RuntimeError(f"phone backup hash mismatch: {source.name}")
    set_read_only(destination)
    if mark_source_read_only:
        set_read_only(source)
    return {
        "source_sha256": source_hash,
        "backup_sha256": backup_hash,
        "manifest_sha256": source_hash,
        "three_way_match": source_hash == backup_hash,
        "copy_status": copy_status,
        "source_read_only": is_read_only(source),
        "backup_read_only": is_read_only(destination),
    }


def build_phone_rgb_manifest(
    source_dir: str | Path,
    dataset_root: str | Path,
    *,
    project_root: str | Path | None = None,
    mark_source_read_only: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = Path(source_dir).resolve()
    output_root = Path(dataset_root).resolve()
    repository_root = Path(project_root).resolve() if project_root is not None else source_root.parent
    videos, appledouble = discover_phone_files(source_root)
    if not videos:
        raise FileNotFoundError(f"no real phone RGB videos found: {source_root}")
    parsed = [parse_phone_filename(path.name) for path in videos]
    manifest_path = output_root / "manifests" / "phone_records.json"
    assigned = _assign_phone_ids(parsed, _load_existing_phone_ids(manifest_path))
    parsed_by_name = {item.source_filename: item for item in parsed}
    records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []

    for source in videos:
        item = parsed_by_name[source.name]
        destination = output_root / "raw" / "phone_rgb" / source.name
        integrity = _copy_and_verify(source, destination, mark_source_read_only=mark_source_read_only)
        decode = decode_phone_video(destination)
        stat = source.stat()
        record_id = assigned[source.name]
        record = {
            "record_id": record_id,
            "source_type": PHONE_SOURCE_TYPE,
            "source_filename": source.name,
            "source_original": _relative(source, repository_root),
            "source_file": f"raw/phone_rgb/{source.name}",
            "action": item.action,
            "action_raw": item.action_raw,
            "subject_id": "subject_pending",
            "camera_view": item.camera_view,
            "camera_view_raw": item.camera_view_raw,
            "recording_intent": item.recording_intent,
            "recording_intent_code": item.recording_intent_code,
            "recording_intent_raw": item.recording_intent_raw,
            "take_index": item.take_index,
            "recording_intent_verified": False,
            "expected_errors_unverified": list(item.expected_errors_unverified),
            "confirmed_labels": [],
            "paired_group_id": None,
            "timeline_id": f"{record_id}_source_timeline_v1",
            "target_athlete": {
                "track_id": None,
                "selection_status": "pending",
                "selection_method": None,
                "identity_switch_segments": [],
            },
            "other_people_present": "unknown",
            "full_body_visible": "pending_pose_baseline_and_target_lock",
            "floor_or_lane_visible": "unknown",
            "equipment_visible": "unknown",
            "device_id": "device_pending",
            "distance_bin": "unknown",
            "lighting": "unknown",
            "clothing": "unknown",
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": integrity["manifest_sha256"],
            "integrity": integrity,
            "video": {
                key: decode.get(key)
                for key in (
                    "declared_frame_count",
                    "decoded_frame_count",
                    "fps",
                    "width",
                    "height",
                    "resolution",
                    "duration_seconds",
                    "rotation_degrees",
                    "codec_fourcc",
                    "source_color_space",
                    "decoded_color_space",
                    "audio_track_count",
                )
            },
            "review_status": {
                "subject_identity": "pending_confirmation",
                "action_expert": "pending_review",
                "data_role": "pending_review",
            },
            "usage_authorization": {
                "status": "pending_confirmation",
                "recorded_by": "",
                "authorized_uses": [],
                "notes": "No training, evaluation or publication use before confirmation.",
            },
            "eligibility": {
                "training_eligible": False,
                "golden_eligible": False,
                "evaluation_eligible": False,
                "example_eligible": False,
            },
            "example_candidate": source.name in EXAMPLE_CANDIDATE_FILENAMES,
            "notes": "Filename describes recording intent only and is not ground truth.",
        }
        records.append(record)
        audit_records.append(
            {
                "record_id": record_id,
                "source_filename": source.name,
                "source_file": record["source_file"],
                "integrity": integrity,
                "decode": decode,
            }
        )

    records.sort(key=lambda value: str(value["record_id"]))
    audit_records.sort(key=lambda value: str(value["record_id"]))
    summary = {
        "real_video_count": len(records),
        "appledouble_file_count": len(appledouble),
        "appledouble_training_record_count": 0,
        "decoded_to_declared_last_frame_count": sum(
            item["decode"]["decoded_to_declared_last_frame"] for item in audit_records
        ),
        "declared_frame_count_total": sum(
            int(item["decode"]["declared_frame_count"]) for item in audit_records
        ),
        "decoded_frame_count_total": sum(
            int(item["decode"]["decoded_frame_count"]) for item in audit_records
        ),
        "duration_seconds_total": round(
            sum(float(item["decode"].get("duration_seconds") or 0.0) for item in audit_records), 6
        ),
        "three_way_sha256_match_count": sum(item["integrity"]["three_way_match"] for item in audit_records),
        "source_read_only_count": sum(item["integrity"]["source_read_only"] for item in audit_records),
        "backup_read_only_count": sum(item["integrity"]["backup_read_only"] for item in audit_records),
        "paired_group_count": sum(record["paired_group_id"] is not None for record in records),
        "example_candidate_count": sum(record["example_candidate"] for record in records),
        "actions": dict(sorted(Counter(str(record["action"]) for record in records).items())),
        "recording_intents": dict(
            sorted(Counter(str(record["recording_intent"]) for record in records).items())
        ),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "hyrox_phone_rgb_record_manifest",
        "generated_at": utc_now(),
        "source_type": PHONE_SOURCE_TYPE,
        "read_compatibility_aliases": list(PHONE_SOURCE_TYPE_ALIASES),
        "new_records_may_use_aliases": False,
        "data_status": "ingested_pending_human_governance",
        "source_directory": _relative(source_root, repository_root),
        "dataset_root": _relative(output_root, repository_root),
        "paired_with_current_oni": False,
        "pairing_policy": "forbidden_for_independent_current_collections",
        "appledouble_policy": "excluded_before_record_id_assignment",
        "records": records,
        "summary": summary,
    }
    audit = {
        "schema_version": 1,
        "artifact_type": "phone_rgb_decode_and_integrity_audit",
        "generated_at": utc_now(),
        "status": "passed" if (
            summary["decoded_to_declared_last_frame_count"] == len(records)
            and summary["three_way_sha256_match_count"] == len(records)
        ) else "failed",
        "summary": summary,
        "appledouble_files": [
            {"source_filename": path.name, "size_bytes": path.stat().st_size, "excluded": True}
            for path in appledouble
        ],
        "records": audit_records,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(output_root / "reports" / "phone_rgb_decode_audit.json", audit)
    _atomic_json(output_root / "manifests" / "record_schema.json", record_schema())
    _atomic_json(
        output_root / "manifests" / "phone_timeline_schema.json",
        phone_timeline_schema(records),
    )
    return manifest, audit


def validate_phone_manifest(
    payload: Mapping[str, Any], *, dataset_root: str | Path | None = None, verify_files: bool = False
) -> list[str]:
    errors: list[str] = []
    records = payload.get("records")
    if not isinstance(records, list):
        return ["records must be a list"]
    root = Path(dataset_root) if dataset_root is not None else None
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"records[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        record_id = str(record.get("record_id", ""))
        filename = str(record.get("source_filename", ""))
        if not PHONE_RECORD_ID_PATTERN.fullmatch(record_id):
            errors.append(f"{prefix} invalid phone record_id: {record_id}")
        if record_id in seen_ids:
            errors.append(f"{prefix} duplicate record_id: {record_id}")
        if filename in seen_names:
            errors.append(f"{prefix} duplicate source_filename: {filename}")
        seen_ids.add(record_id)
        seen_names.add(filename)
        if record.get("source_type") != PHONE_SOURCE_TYPE:
            errors.append(f"{prefix} source_type must be {PHONE_SOURCE_TYPE!r}")
        if is_appledouble(filename):
            errors.append(f"{prefix} AppleDouble metadata must not be a record")
        if record.get("paired_group_id") is not None:
            errors.append(f"{prefix} paired_group_id must be null")
        if not SHA256_PATTERN.fullmatch(str(record.get("sha256", ""))):
            errors.append(f"{prefix} invalid sha256")
        if record.get("recording_intent_verified") is not False or record.get("confirmed_labels") != []:
            errors.append(f"{prefix} filename intent must remain unverified")
        if root is not None and verify_files:
            path = root / str(record.get("source_file", ""))
            if not path.is_file():
                errors.append(f"{prefix} backup missing")
            elif sha256_file(path) != record.get("sha256"):
                errors.append(f"{prefix} backup hash mismatch")
            elif not is_read_only(path):
                errors.append(f"{prefix} backup must be read-only")
    return errors


def build_data_roles(manifest: Mapping[str, Any]) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    for record in manifest.get("records") or []:
        candidate = bool(record.get("example_candidate"))
        assignments.append(
            {
                "record_id": record["record_id"],
                "source_filename": record["source_filename"],
                "role": "example_candidate" if candidate else "unassigned_pending_review",
                "example_candidate": candidate,
                "training_eligible": False,
                "golden_eligible": False,
                "evaluation_eligible": False,
                "template_eligible": False,
                "example_eligible": False,
                "authorization_status": record["usage_authorization"]["status"],
                "expert_review_status": record["review_status"]["action_expert"],
            }
        )
    role_sets = {
        role: sorted(item["record_id"] for item in assignments if item[role])
        for role in (
            "training_eligible",
            "golden_eligible",
            "evaluation_eligible",
            "template_eligible",
            "example_eligible",
        )
    }
    memberships: defaultdict[str, list[str]] = defaultdict(list)
    for role, ids in role_sets.items():
        for record_id in ids:
            memberships[record_id].append(role)
    overlaps = [
        {"record_id": record_id, "roles": roles}
        for record_id, roles in sorted(memberships.items())
        if len(roles) > 1
    ]
    return {
        "schema_version": 1,
        "artifact_type": "hyrox_data_roles_v1",
        "generated_at": utc_now(),
        "policy": {
            "filename_is_ground_truth": False,
            "examples_are_training_templates_or_golden": False,
            "authorization_and_expert_review_required": True,
            "silent_role_overlap_forbidden": True,
        },
        "assignments": assignments,
        "role_sets": role_sets,
        "overlaps": overlaps,
        "checks": {
            "all_records_assigned": len(assignments) == len(manifest.get("records") or []),
            "example_candidate_count_is_eight": sum(item["example_candidate"] for item in assignments) == 8,
            "no_silent_overlap": not overlaps,
            "all_training_and_golden_disabled": not role_sets["training_eligible"] and not role_sets["golden_eligible"],
        },
    }


def _coverage_value(record: Mapping[str, Any], field: str) -> str:
    if field in {"resolution", "fps"}:
        return str(record.get("video", {}).get(field, "unknown"))
    if field == "floor_or_lane_visible":
        return str(record.get(field, "unknown"))
    return str(record.get(field, "unknown"))


def build_coverage_gap_matrix(
    manifest: Mapping[str, Any], *, interval_candidates: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
) -> dict[str, Any]:
    fields = (
        "subject_id",
        "camera_view",
        "device_id",
        "resolution",
        "fps",
        "distance_bin",
        "lighting",
        "clothing",
        "other_people_present",
        "equipment_visible",
        "floor_or_lane_visible",
        "full_body_visible",
        "recording_intent",
    )
    records_by_action: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in manifest.get("records") or []:
        records_by_action[str(record["action"])].append(record)
    actions: dict[str, Any] = {}
    interval_candidates = interval_candidates or {}
    for action, records in sorted(records_by_action.items()):
        distributions = {
            field: dict(sorted(Counter(_coverage_value(record, field) for record in records).items()))
            for field in fields
        }
        intent_counts = Counter(str(record["recording_intent"]) for record in records)
        action_candidates = [
            item
            for record in records
            for item in interval_candidates.get(str(record["record_id"]), ())
        ]
        actions[action] = {
            "record_count": len(records),
            "field_distributions": distributions,
            "expected_error_unverified": dict(
                sorted(
                    Counter(
                        str(code)
                        for record in records
                        for code in record.get("expected_errors_unverified") or []
                    ).items()
                )
            ),
            "sample_role_gaps": {
                "compliant": "present_unverified" if intent_counts["standard"] else "missing",
                "error": "present_unverified" if intent_counts["error"] else "missing",
                "boundary": "missing",
                "unknown_ood": "missing",
            },
            "diversity_gaps": [
                "subject identities pending; no unseen-subject claim",
                "single device identity pending",
                "distance, lighting and clothing not yet audited",
                "speed labels unavailable before target-bound temporal annotation",
            ],
            "equipment_visibility_requirements": list(ACTION_EQUIPMENT[action]),
            "equipment_visibility_gap": "pending_round7_object_scene_audit",
            "idle_setup_exit_transition_candidates": action_candidates,
            "candidate_policy": (
                "low-pose intervals are review proposals only; background people are never automatic negatives"
            ),
        }
    return {
        "schema_version": 1,
        "artifact_type": "hyrox_coverage_gap_matrix_v1",
        "generated_at": utc_now(),
        "source_type": PHONE_SOURCE_TYPE,
        "records_are_independent_from_oni": True,
        "actions": actions,
        "global_gaps": [
            "all subject identities require confirmation",
            "target athlete tracks are not locked",
            "continuous mixed-action and unknown/OOD recordings are absent",
            "boundary severity and repeated error takes are insufficient",
            "background-person, occlusion and identity-switch intervals await round 7",
        ],
    }


def build_observability_gap(manifest: Mapping[str, Any]) -> dict[str, Any]:
    actions = {}
    present_actions = sorted({str(record["action"]) for record in manifest.get("records") or []})
    for action in present_actions:
        actions[action] = {
            "required_equipment_or_scene_evidence": list(ACTION_EQUIPMENT[action]),
            "pose_only_unobservable_rules": POSE_ONLY_UNOBSERVABLE_RULES[action],
            "current_status": "UNOBSERVABLE_until_target_and_object_scene_audit",
            "product_policy": "do_not_infer_PASS_from_missing_equipment_or_scene_evidence",
        }
    return {
        "schema_version": 1,
        "artifact_type": "hyrox_observability_gap_v1",
        "generated_at": utc_now(),
        "coordinate_warning": (
            "MediaPipe World is body-centred relative 3D and is not camera coordinates, metric venue geometry or measured equipment distance."
        ),
        "actions": actions,
    }


def write_round6_governance_reports(
    manifest: Mapping[str, Any], dataset_root: str | Path, *, interval_candidates: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(dataset_root)
    roles = build_data_roles(manifest)
    coverage = build_coverage_gap_matrix(manifest, interval_candidates=interval_candidates)
    observability = build_observability_gap(manifest)
    _atomic_json(root / "manifests" / "data_roles_v1.json", roles)
    _atomic_json(root / "reports" / "coverage_gap_matrix_v1.json", coverage)
    _atomic_json(root / "reports" / "observability_gap_v1.json", observability)
    return roles, coverage, observability


def migrate_source_type(value: str) -> str:
    """Read compatibility for the retired alias; writers only emit phone_rgb."""
    return PHONE_SOURCE_TYPE if value in PHONE_SOURCE_TYPE_ALIASES else value


__all__ = [
    "EXAMPLE_CANDIDATE_FILENAMES",
    "PHONE_CAMERA_VIEWS",
    "PHONE_RECORD_ID_PATTERN",
    "ParsedPhoneFilename",
    "build_coverage_gap_matrix",
    "build_data_roles",
    "build_observability_gap",
    "build_phone_rgb_manifest",
    "decode_phone_video",
    "discover_phone_files",
    "is_appledouble",
    "migrate_source_type",
    "parse_phone_filename",
    "phone_timeline_schema",
    "validate_phone_manifest",
    "write_round6_governance_reports",
]
