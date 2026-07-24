from __future__ import annotations

import json
from pathlib import Path

from tools.dataset.manifest import (
    PHONE_SOURCE_TYPE,
    build_dataset_manifest,
    is_read_only,
    parse_oni_filename,
    validate_manifest,
    validation_report,
)
from tools.build_hyrox_dataset_manifest import build_parser


def test_parse_oni_filename_preserves_chinese_intent_and_take_index() -> None:
    standard = parse_oni_filename("推雪橇-标准11-斜前方.oni")
    error = parse_oni_filename("负重箭步蹲-后膝未触地-斜前方.oni")
    variant = parse_oni_filename("波比跳-标准（双腿起）-斜前方.oni")

    assert standard.action == "sled_push"
    assert standard.recording_intent_raw == "标准11"
    assert standard.recording_intent_base_raw == "标准"
    assert standard.take_index == 11
    assert standard.recording_intent == "standard"
    assert error.recording_intent == "error"
    assert error.expected_errors_unverified == ("NO_KNEE_CONTACT",)
    assert variant.recording_variant_raw == "双腿起"
    assert variant.take_index is None


def test_unknown_filename_parts_fail_instead_of_being_guessed() -> None:
    try:
        parse_oni_filename("未知动作-标准-未知视角.oni")
    except ValueError as exc:
        assert "unknown ONI action" in str(exc)
    else:
        raise AssertionError("unknown action should fail")


def test_build_manifest_copies_hashes_and_marks_files_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = source / "负重箭步蹲-标准-斜前方.oni"
    second = source / "负重箭步蹲-后膝未触地-斜前方.oni"
    first.write_bytes(b"oni-standard")
    second.write_bytes(b"oni-error")
    dataset = tmp_path / "datasets" / "hyrox"

    payload = build_dataset_manifest(
        source,
        dataset,
        project_root=tmp_path,
    )

    assert payload["summary"]["record_count"] == 2
    assert payload["phone_pairing_policy"] == "forbidden_for_current_oni"
    assert all(record["paired_group_id"] is None for record in payload["records"])
    assert all(record["recording_intent_verified"] is False for record in payload["records"])
    assert all(record["confirmed_labels"] == [] for record in payload["records"])
    assert all(record["subject_id"] == "subject_pending" for record in payload["records"])
    assert all(record["target_athlete"]["selection_status"] == "pending" for record in payload["records"])
    assert all(record["integrity"]["backup_verified"] for record in payload["records"])
    assert is_read_only(first)
    assert is_read_only(dataset / "raw" / "oni" / first.name)
    phone = json.loads(
        (dataset / "manifests" / "phone_records.json").read_text(
            encoding="utf-8"
        )
    )
    assert phone["source_type"] == PHONE_SOURCE_TYPE
    assert phone["records"] == []
    report = validation_report(payload, dataset_root=dataset)
    assert report["status"] == "passed"
    assert all(report["checks"].values())


def test_existing_record_ids_remain_stable_when_new_file_is_added(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original = source / "墙球-标准-斜后方.oni"
    original.write_bytes(b"one")
    dataset = tmp_path / "dataset"
    first = build_dataset_manifest(source, dataset, project_root=tmp_path)
    first_id = first["records"][0]["record_id"]

    added = source / "墙球-下蹲不足-斜后方.oni"
    added.write_bytes(b"two")
    second = build_dataset_manifest(source, dataset, project_root=tmp_path)
    by_name = {
        record["source_filename"]: record["record_id"]
        for record in second["records"]
    }

    assert by_name[original.name] == first_id
    assert len(set(by_name.values())) == 2


def test_validation_rejects_phone_pairing_and_confirmed_labels() -> None:
    payload = {
        "schema_version": 1,
        "records": [
            {
                "record_id": "oni_lunge_001",
                "source_type": "oni",
                "source_filename": "负重箭步蹲-标准-斜前方.oni",
                "source_file": "raw/oni/负重箭步蹲-标准-斜前方.oni",
                "action": "lunge",
                "action_raw": "负重箭步蹲",
                "subject_id": "subject_pending",
                "camera_view": "oblique_front",
                "camera_view_raw": "斜前方",
                "recording_intent": "standard",
                "recording_intent_raw": "标准",
                "recording_intent_verified": True,
                "confirmed_labels": ["VALID"],
                "paired_group_id": "phone_pair",
                "target_athlete": {"selection_status": "pending"},
                "other_people_present": "yes_or_possible",
                "sha256": "0" * 64,
                "usage_authorization": {"status": "pending_confirmation"},
            }
        ],
    }

    errors = validate_manifest(payload)

    assert any("paired_group_id must be null" in item for item in errors)
    assert any("recording_intent_verified must remain false" in item for item in errors)
    assert any("confirmed_labels must be empty" in item for item in errors)


def test_dataset_cli_supports_non_mutating_full_validation() -> None:
    args = build_parser().parse_args(["--validate-only", "--verify-files"])

    assert args.validate_only is True
    assert args.verify_files is True


def test_rebuilding_oni_manifest_does_not_overwrite_ingested_phone_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oni"
    source.mkdir()
    (source / "墙球-标准-斜后方.oni").write_bytes(b"oni")
    dataset = tmp_path / "dataset"
    phone_path = dataset / "manifests" / "phone_records.json"
    phone_path.parent.mkdir(parents=True)
    expected = {
        "schema_version": 1,
        "source_type": "phone_rgb",
        "records": [{"record_id": "phone_wall_ball_001"}],
    }
    phone_path.write_text(json.dumps(expected), encoding="utf-8")

    build_dataset_manifest(source, dataset, project_root=tmp_path)

    assert json.loads(phone_path.read_text(encoding="utf-8")) == expected
