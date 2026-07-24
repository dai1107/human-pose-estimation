from __future__ import annotations

import json
from pathlib import Path

from tools.dataset import phone_rgb
from tools.dataset.phone_rgb import (
    build_data_roles,
    build_phone_rgb_manifest,
    discover_phone_files,
    migrate_source_type,
    parse_phone_filename,
    validate_phone_manifest,
)


def _decode_stub(frame_count: int = 3) -> dict[str, object]:
    return {
        "status": "passed",
        "error": None,
        "container": "mp4",
        "opencv_backend": "test",
        "declared_frame_count": frame_count,
        "decoded_frame_count": frame_count,
        "declared_last_frame_index": frame_count - 1,
        "decoded_last_frame_index": frame_count - 1,
        "decoded_to_declared_last_frame": True,
        "fps": 30.0,
        "width": 720,
        "height": 1280,
        "resolution": "720x1280",
        "duration_seconds": frame_count / 30.0,
        "rotation_degrees": 0,
        "codec_fourcc": "mp4v",
        "codec_pixel_format": None,
        "source_color_space": "unknown_without_container_color_metadata_parser",
        "decoded_color_space": "BGR8",
        "model_input_color_space": "sRGB_after_explicit_BGR_to_RGB_conversion",
        "audio_track_count": None,
        "audio_audit_status": "unavailable_in_opencv_backend",
        "timestamp_source": "container_pts",
        "timestamp_fallback_count": 0,
        "timestamps_ms": [0.0, 33.333333, 66.666667][:frame_count],
        "abnormal_frames": [],
    }


def test_phone_filename_parser_preserves_chinese_and_canonicalizes_fields() -> None:
    standard = parse_phone_filename("波比跳-标准（双脚起）-正前方-1.mp4")
    error = parse_phone_filename("划船机-桨把绕膝-侧后方-1.mp4")

    assert standard.action == "burpee_broad_jump"
    assert standard.recording_intent == "standard"
    assert standard.recording_intent_raw == "标准（双脚起）"
    assert standard.camera_view == "front"
    assert standard.take_index == 1
    assert error.action == "rowing"
    assert error.recording_intent == "error"
    assert error.expected_errors_unverified == ("HANDLE_AROUND_KNEES",)


def test_discovery_excludes_appledouble_before_record_assignment(tmp_path: Path) -> None:
    source = tmp_path / "phone"
    source.mkdir()
    real = source / "墙球-标准-正后方-1.mp4"
    metadata = source / "._墙球-标准-正后方-1.mp4"
    real.write_bytes(b"video")
    metadata.write_bytes(b"metadata")

    videos, appledouble = discover_phone_files(source)

    assert videos == [real]
    assert appledouble == [metadata]


def test_build_phone_manifest_copies_hashes_and_never_emits_future_alias(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "phone"
    source.mkdir()
    first = source / "墙球-标准-正后方-1.mp4"
    second = source / "墙球-踮脚-侧方-1.mp4"
    metadata = source / "._墙球-标准-正后方-1.mp4"
    first.write_bytes(b"first-video")
    second.write_bytes(b"second-video")
    metadata.write_bytes(b"apple-double")
    monkeypatch.setattr(phone_rgb, "decode_phone_video", lambda path: _decode_stub())
    dataset = tmp_path / "datasets" / "hyrox"

    manifest, audit = build_phone_rgb_manifest(
        source, dataset, project_root=tmp_path, mark_source_read_only=False
    )

    assert manifest["source_type"] == "phone_rgb"
    assert len(manifest["records"]) == 2
    assert all(record["source_type"] == "phone_rgb" for record in manifest["records"])
    assert all(record["paired_group_id"] is None for record in manifest["records"])
    assert all(record["integrity"]["three_way_match"] for record in manifest["records"])
    assert audit["summary"]["appledouble_file_count"] == 1
    assert audit["summary"]["appledouble_training_record_count"] == 0
    assert not validate_phone_manifest(manifest, dataset_root=dataset, verify_files=True)
    stored = json.loads(
        (dataset / "manifests" / "phone_records.json").read_text(encoding="utf-8")
    )
    assert all(record["source_type"] != "phone_rgb_future" for record in stored["records"])


def test_data_roles_keep_examples_out_of_training_golden_and_templates() -> None:
    manifest = {
        "records": [
            {
                "record_id": "phone_wall_ball_001",
                "source_filename": "墙球-标准-正后方-1.mp4",
                "example_candidate": True,
                "usage_authorization": {"status": "pending_confirmation"},
                "review_status": {"action_expert": "pending_review"},
            }
        ]
    }

    roles = build_data_roles(manifest)
    assignment = roles["assignments"][0]

    assert assignment["role"] == "example_candidate"
    assert assignment["training_eligible"] is False
    assert assignment["golden_eligible"] is False
    assert assignment["template_eligible"] is False
    assert roles["overlaps"] == []


def test_phone_source_alias_is_read_only_compatibility() -> None:
    assert migrate_source_type("phone_rgb_future") == "phone_rgb"
    assert migrate_source_type("phone_rgb") == "phone_rgb"
