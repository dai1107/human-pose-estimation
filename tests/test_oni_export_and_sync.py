from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tools.dataset.oni_export import (
    fingerprint_stream,
    fingerprints_match,
    validate_export,
)
from tools.dataset.oni_sync import (
    build_sync_report,
    can_pair_by_frame_index,
    pair_by_nearest_timestamp,
    sync_statistics,
)
from tools.export_oni_dataset import build_parser as export_parser
from tools.synchronize_oni_dataset import build_parser as sync_parser


def _write_index(path: Path, stream: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if stream == "depth":
        fields = (
            "output_frame",
            "source_frame_index",
            "timestamp_us",
            "depth_scale",
            "invalid_pixel_ratio",
        )
        rows = [
            {
                "output_frame": 0,
                "source_frame_index": 1,
                "timestamp_us": 100,
                "depth_scale": 1.0,
                "invalid_pixel_ratio": 0.25,
            },
            {
                "output_frame": 1,
                "source_frame_index": 2,
                "timestamp_us": 200,
                "depth_scale": 1.0,
                "invalid_pixel_ratio": 0.5,
            },
        ]
    else:
        fields = (
            "output_frame",
            "source_frame_index",
            "timestamp_us",
            "width",
            "height",
            "pixel_format",
        )
        rows = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stream(
    *,
    exists: bool,
    frame_count: int,
    pixel_format: str,
) -> dict:
    return {
        "exists": exists,
        "complete": exists,
        "width": 2 if exists else 0,
        "height": 2 if exists else 0,
        "pixel_format": pixel_format if exists else "UNKNOWN",
        "nominal_fps": 30 if exists else 0,
        "frame_encoding": (
            "npy_uint16_little_endian" if exists else "none"
        ),
        "expected_frame_count": frame_count,
        "actual_frame_count": frame_count,
        "first_timestamp_us": 100 if exists else None,
        "last_timestamp_us": 200 if exists else None,
        "first_frame_index": 1 if exists else None,
        "last_frame_index": 2 if exists else None,
        "decode_error_count": 0,
        "errors": [],
    }


def test_lossless_export_validation_and_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "record"
    for name in ("color", "depth", "ir"):
        (root / name / "frames").mkdir(parents=True)
        _write_index(root / name / "index.csv", name)
    np.save(
        root / "depth" / "frames" / "00000000.npy",
        np.array([[0, 1000], [2000, 3000]], dtype=np.uint16),
        allow_pickle=False,
    )
    np.save(
        root / "depth" / "frames" / "00000001.npy",
        np.array([[0, 1001], [2001, 3001]], dtype=np.uint16),
        allow_pickle=False,
    )
    metadata = {
        "artifact_type": "oni_lossless_export",
        "complete": True,
        "lossless_depth": True,
        "playback_speed_independent": True,
        "input": {"size_bytes": 123},
        "streams": {
            "color": _stream(
                exists=False, frame_count=0, pixel_format="UNKNOWN"
            ),
            "depth": _stream(
                exists=True, frame_count=2, pixel_format="DEPTH_1_MM"
            ),
            "ir": _stream(
                exists=False, frame_count=0, pixel_format="UNKNOWN"
            ),
        },
    }
    audit = {
        "file": {"size_bytes": 123},
        "streams": {
            name: {
                key: value
                for key, value in metadata["streams"][name].items()
                if key
                in {
                    "exists",
                    "width",
                    "height",
                    "pixel_format",
                    "nominal_fps",
                    "actual_frame_count",
                    "first_timestamp_us",
                    "last_timestamp_us",
                    "first_frame_index",
                    "last_frame_index",
                }
            }
            for name in ("color", "depth", "ir")
        },
    }

    assert validate_export(metadata, audit, root) == []
    first = fingerprint_stream(root / "depth")
    second = fingerprint_stream(root / "depth")
    all_streams_first = {
        "color": fingerprint_stream(root / "color"),
        "depth": first,
        "ir": fingerprint_stream(root / "ir"),
    }
    all_streams_second = {
        "color": fingerprint_stream(root / "color"),
        "depth": second,
        "ir": fingerprint_stream(root / "ir"),
    }
    assert first["frame_file_count"] == 2
    assert fingerprints_match(all_streams_first, all_streams_second)


def test_sync_prefers_nearest_timestamp_without_capture_confirmation() -> None:
    color = [
        {"output_frame": 0, "source_frame_index": 1, "timestamp_us": 1000},
        {"output_frame": 1, "source_frame_index": 2, "timestamp_us": 34000},
    ]
    depth = [
        {"output_frame": 0, "source_frame_index": 1, "timestamp_us": 1200},
        {"output_frame": 1, "source_frame_index": 2, "timestamp_us": 34500},
    ]

    assert not can_pair_by_frame_index(
        color, depth, capture_sync_confirmed=False
    )
    pairs = pair_by_nearest_timestamp(color, depth)
    stats = sync_statistics(pairs, rgb_frame_count=2)

    assert [pair["delta_us"] for pair in pairs] == [200, 500]
    assert all(pair["sync_method"] == "nearest_timestamp" for pair in pairs)
    assert stats["sync_quality"] == "good"
    assert stats["p95_error_ms"] < 1.0


def test_missing_color_is_video_level_only_and_never_phone_paired() -> None:
    record = {
        "record_id": "oni_lunge_001",
        "source_filename": "负重箭步蹲-标准-斜前方.oni",
        "sha256": "0" * 64,
        "paired_group_id": None,
    }
    metadata = {
        "streams": {
            "color": {"exists": False},
            "depth": {"exists": True},
        }
    }
    depth = [
        {"output_frame": 0, "source_frame_index": 1, "timestamp_us": 0}
    ]

    report, pairs = build_sync_report(record, metadata, [], depth)

    assert pairs == []
    assert report["applicable"] is False
    assert report["reason"] == "missing_color"
    assert report["sync_quality"] == "video_level_only"
    assert report["fine_event_training_eligible"] is False
    assert report["phone_pairing_used"] is False


def test_round_four_and_five_cli_contracts() -> None:
    export_args = export_parser().parse_args(
        ["--record-id", "oni_lunge_001", "--reuse-existing"]
    )
    sync_args = sync_parser().parse_args(
        ["--record-id", "oni_lunge_001"]
    )

    assert export_args.reuse_existing is True
    assert export_args.record_ids == ["oni_lunge_001"]
    assert sync_args.record_ids == ["oni_lunge_001"]


def test_cpp_exporter_and_runtime_manifest_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "tools" / "oni_bridge" / "oni_export.cpp"
    ).read_text(encoding="utf-8")
    manifest = json.loads(
        (
            root / "tools" / "oni_bridge" / "runtime_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    paths = {entry["path"] for entry in manifest["files"]}

    for token in (
        "npy_uint16_little_endian",
        "source_frame_index",
        "timestamp_us",
        "invalid_pixel_ratio",
        "playback_speed_independent",
        "CreateHardLinkW",
    ):
        assert token in source
    assert "oni-export.exe" in paths
    assert manifest["offline_file_only"] is True
    assert manifest["camera_driver_included"] is False
