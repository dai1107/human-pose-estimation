from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.audit_oni_dataset import build_parser
from tools.dataset.oni_audit import (
    audit_summary_markdown,
    build_batch_report,
    enrich_inventory,
    invalid_records,
    validate_inventory,
)


def _inventory(code: str = "A") -> dict:
    stream = {
        "exists": True,
        "complete": True,
        "width": 640,
        "height": 480,
        "pixel_format": "RGB888",
        "nominal_fps": 30,
        "actual_frame_count": 10,
        "first_timestamp_us": 0,
        "last_timestamp_us": 300000,
        "first_frame_index": 1,
        "last_frame_index": 10,
        "timestamps_strictly_increasing": True,
        "frame_indices_continuous": True,
        "non_increasing_timestamp_count": 0,
        "frame_index_discontinuity_count": 0,
        "decode_error_count": 0,
        "interval_p50_us": 33333.0,
        "interval_p95_us": 33334.0,
        "actual_fps": 30.0,
        "estimated_dropped_frames": 0,
        "abnormal_interval_count": 0,
    }
    depth = dict(stream)
    depth["pixel_format"] = "DEPTH_1_MM"
    depth["depth_quality"] = {
        "valid_pixel_count": 100,
        "zero_value_ratio": 0.1,
        "invalid_pixel_ratio": 0.1,
        "min_depth_raw": 1000,
        "max_depth_raw": 4000,
        "center_region": {"p50_depth_raw": 2000},
    }
    descriptions = {
        "A": "color_and_depth",
        "B": "depth_without_color",
    }
    color = dict(stream)
    if code == "B":
        color["exists"] = False
        color["complete"] = False
        color["actual_frame_count"] = 0
    return {
        "schema_version": 1,
        "artifact_type": "oni_inventory",
        "tool": {"offline_file_only": True},
        "file": {
            "size_bytes": 123,
            "open_success": True,
            "complete_playback": True,
            "duration_us": 300000,
            "decode_error_count": 0,
        },
        "classification": {
            "code": code,
            "description": descriptions[code],
            "qualified_for_rgbd": code == "A",
        },
        "streams": {
            "color": color,
            "depth": depth,
            "ir": dict(stream),
        },
    }


def _record(record_id: str) -> dict:
    return {
        "record_id": record_id,
        "source_type": "oni",
        "source_filename": f"{record_id}.oni",
        "source_file": f"raw/oni/{record_id}.oni",
        "size_bytes": 123,
        "sha256": "0" * 64,
        "action": "lunge",
        "recording_intent": "standard",
        "recording_intent_verified": False,
        "target_athlete": {"selection_status": "pending"},
        "other_people_present": "yes_or_possible",
    }


def test_inventory_validation_and_manifest_identity_enrichment() -> None:
    report = enrich_inventory(
        _inventory("A"),
        _record("oni_lunge_001"),
        inspector_return_code=0,
        elapsed_seconds=1.25,
    )

    assert validate_inventory(report) == []
    assert report["validation_errors"] == []
    assert report["dataset_record"]["manifest_sha256"] == "0" * 64
    assert (
        report["dataset_record"]["target_athlete_selection_status"]
        == "pending"
    )


def test_batch_separates_non_rgbd_records() -> None:
    first = enrich_inventory(
        _inventory("A"),
        _record("oni_lunge_001"),
        inspector_return_code=0,
        elapsed_seconds=1.0,
    )
    second = enrich_inventory(
        _inventory("B"),
        _record("oni_lunge_002"),
        inspector_return_code=0,
        elapsed_seconds=1.0,
    )
    manifest = {
        "artifact_type": "hyrox_oni_record_manifest",
        "phone_pairing_policy": "forbidden_for_current_oni",
        "records": [_record("oni_lunge_001"), _record("oni_lunge_002")],
    }

    batch = build_batch_report(
        manifest,
        [first, second],
        started_at="2026-07-24T00:00:00+00:00",
        completed_at="2026-07-24T00:00:02+00:00",
    )
    rejected = invalid_records(batch)

    assert batch["status"] == "passed"
    assert batch["summary"]["classification_counts"]["A"] == 1
    assert batch["summary"]["classification_counts"]["B"] == 1
    assert rejected["count"] == 1
    assert rejected["records"][0]["record_id"] == "oni_lunge_002"
    assert "主体运动者仍待后续人工锁定" in audit_summary_markdown(batch)


def test_cli_has_focused_record_and_timeout_options() -> None:
    args = build_parser().parse_args(
        [
            "--record-id",
            "oni_lunge_001",
            "--timeout-seconds",
            "30",
        ]
    )

    assert args.record_ids == ["oni_lunge_001"]
    assert args.timeout_seconds == 30.0
    assert args.reuse_existing is False


def test_cpp_bridge_contract_and_product_dependency_isolation() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "oni_bridge" / "oni_inspect.cpp").read_text(
        encoding="utf-8"
    )
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8").lower()

    for contract in (
        '\\"offline_file_only\\": true',
        "interval_p50_us",
        "interval_p95_us",
        "estimated_dropped_frames",
        "depth_quality",
        "classification",
        "CreateHardLinkW",
    ):
        assert contract in source
    dependency_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "openni" not in dependency_block
    assert "orbbec" not in dependency_block


def test_oni_bridge_runtime_manifest_matches_deployed_files() -> None:
    root = Path(__file__).resolve().parents[1]
    bridge = root / "tools" / "oni_bridge"
    manifest = json.loads(
        (bridge / "runtime_manifest.json").read_text(encoding="utf-8-sig")
    )

    assert manifest["architecture"] == "x86_64"
    assert manifest["offline_file_only"] is True
    assert manifest["camera_driver_included"] is False
    assert any(
        "MSVCR120.dll" in dependency
        for dependency in manifest["system_dependencies"]
    )
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert {
        "oni-inspect.exe",
        "OpenNI2.dll",
        "libwinpthread-1.dll",
        "OpenNI2/Drivers/OniFile.dll",
        "OpenNI2/Drivers/OniFile.ini",
        "OPENNI2_LICENSE.txt",
        "MINGW_W64_RUNTIME_LICENSE.txt",
    } <= set(entries)
    assert not any("orbbec" in path.lower() for path in entries)
    for relative_path, entry in entries.items():
        deployed = bridge / relative_path
        assert deployed.is_file()
        assert deployed.stat().st_size == entry["bytes"]
        assert hashlib.sha256(deployed.read_bytes()).hexdigest() == entry["sha256"]
