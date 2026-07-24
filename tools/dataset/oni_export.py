"""Validation, hashing and preview helpers for lossless ONI exports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

STREAM_NAMES = ("color", "depth", "ir")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def frame_files(frames_dir: Path) -> list[Path]:
    if not frames_dir.is_dir():
        return []
    return sorted(path for path in frames_dir.iterdir() if path.is_file())


def fingerprint_stream(stream_root: Path) -> dict[str, Any]:
    index_path = stream_root / "index.csv"
    frames = frame_files(stream_root / "frames")
    aggregate = hashlib.sha256()
    total_bytes = 0
    for frame in frames:
        frame_hash = sha256_file(frame)
        size = frame.stat().st_size
        total_bytes += size
        aggregate.update(frame.name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(frame_hash))
        aggregate.update(size.to_bytes(8, "little"))
    return {
        "index_sha256": (
            sha256_file(index_path) if index_path.is_file() else None
        ),
        "frame_file_count": len(frames),
        "frame_total_bytes": total_bytes,
        "frames_aggregate_sha256": aggregate.hexdigest(),
    }


def _validate_index(
    stream_name: str,
    rows: list[dict[str, str]],
    stream: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_count = int(stream["actual_frame_count"])
    if len(rows) != expected_count:
        errors.append(
            f"{stream_name} index rows {len(rows)} != "
            f"actual frames {expected_count}"
        )
        return errors
    if not rows:
        return errors
    output_frames = [int(row["output_frame"]) for row in rows]
    source_indices = [int(row["source_frame_index"]) for row in rows]
    timestamps = [int(row["timestamp_us"]) for row in rows]
    if output_frames != list(range(len(rows))):
        errors.append(f"{stream_name} output_frame is not zero-based continuous")
    if any(
        current <= previous
        for previous, current in zip(source_indices, source_indices[1:])
    ):
        errors.append(f"{stream_name} source frame indices are not increasing")
    if any(
        current <= previous
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        errors.append(f"{stream_name} timestamps are not increasing")
    if source_indices[0] != stream["first_frame_index"]:
        errors.append(f"{stream_name} first source frame index mismatch")
    if source_indices[-1] != stream["last_frame_index"]:
        errors.append(f"{stream_name} last source frame index mismatch")
    if timestamps[0] != stream["first_timestamp_us"]:
        errors.append(f"{stream_name} first timestamp mismatch")
    if timestamps[-1] != stream["last_timestamp_us"]:
        errors.append(f"{stream_name} last timestamp mismatch")
    return errors


def validate_export(
    metadata: Mapping[str, Any],
    audit: Mapping[str, Any],
    output_root: Path,
) -> list[str]:
    errors: list[str] = []
    if metadata.get("artifact_type") != "oni_lossless_export":
        errors.append("artifact_type must be oni_lossless_export")
    if metadata.get("complete") is not True:
        errors.append("export is not complete")
    if metadata.get("lossless_depth") is not True:
        errors.append("lossless_depth must be true")
    if metadata.get("playback_speed_independent") is not True:
        errors.append("playback_speed_independent must be true")
    if metadata.get("input", {}).get("size_bytes") != audit["file"]["size_bytes"]:
        errors.append("input size does not match audit")
    for stream_name in STREAM_NAMES:
        stream = metadata["streams"][stream_name]
        audited = audit["streams"][stream_name]
        if stream["exists"] is not audited["exists"]:
            errors.append(f"{stream_name} presence differs from audit")
        if not stream["exists"]:
            if stream["actual_frame_count"] != 0:
                errors.append(f"absent {stream_name} has exported frames")
            continue
        for key in (
            "width",
            "height",
            "pixel_format",
            "nominal_fps",
            "actual_frame_count",
            "first_timestamp_us",
            "last_timestamp_us",
            "first_frame_index",
            "last_frame_index",
        ):
            if stream[key] != audited[key]:
                errors.append(f"{stream_name}.{key} differs from audit")
        if stream["complete"] is not True:
            errors.append(f"{stream_name} export is incomplete")
        stream_root = output_root / stream_name
        rows = read_index(stream_root / "index.csv")
        errors.extend(_validate_index(stream_name, rows, stream))
        frames = frame_files(stream_root / "frames")
        if len(frames) != stream["actual_frame_count"]:
            errors.append(
                f"{stream_name} frame files {len(frames)} != "
                f"{stream['actual_frame_count']}"
            )
        if stream_name in {"depth", "ir"} and frames:
            for frame_path in (frames[0], frames[-1]):
                if frame_path.suffix != ".npy":
                    errors.append(
                        f"{stream_name} frame is not lossless NPY: "
                        f"{frame_path.name}"
                    )
                    continue
                array = np.load(frame_path, mmap_mode="r", allow_pickle=False)
                if array.dtype != np.dtype("<u2"):
                    errors.append(
                        f"{stream_name} frame dtype is not uint16"
                    )
                if array.shape != (stream["height"], stream["width"]):
                    errors.append(
                        f"{stream_name} frame shape does not match metadata"
                    )
    return errors


def create_depth_preview(
    output_root: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    stream = metadata["streams"]["depth"]
    if not stream["exists"]:
        return None
    frames = frame_files(output_root / "depth" / "frames")
    if not frames:
        return None
    preview_dir = output_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "depth_preview.mp4"
    fps = float(stream["nominal_fps"] or 30)
    size = (int(stream["width"]), int(stream["height"]))
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open depth preview writer: {preview_path}")
    try:
        for frame_path in frames:
            depth = np.load(frame_path, allow_pickle=False)
            valid = depth > 0
            clipped = np.clip(depth.astype(np.float32), 500.0, 8000.0)
            normalized = (
                (8000.0 - clipped) * (255.0 / (8000.0 - 500.0))
            ).astype(np.uint8)
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            colored[~valid] = 0
            writer.write(colored)
    finally:
        writer.release()
    if not preview_path.is_file() or preview_path.stat().st_size == 0:
        raise RuntimeError(f"depth preview was not created: {preview_path}")
    return {
        "path": "preview/depth_preview.mp4",
        "derivative_only": True,
        "source": "depth/frames/*.npy",
        "frame_count": len(frames),
        "fps": fps,
        "sha256": sha256_file(preview_path),
        "bytes": preview_path.stat().st_size,
    }


def create_color_preview(
    output_root: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    stream = metadata["streams"]["color"]
    if not stream["exists"]:
        return None
    frames = frame_files(output_root / "color" / "frames")
    if not frames:
        return None
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        return {
            "path": None,
            "available": False,
            "reason": "color_frame_encoding_not_supported_by_opencv",
        }
    color_path = output_root / "color" / "color.mp4"
    fps = float(stream["nominal_fps"] or 30)
    writer = cv2.VideoWriter(
        str(color_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(stream["width"]), int(stream["height"])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open color preview writer: {color_path}")
    try:
        for frame_path in frames:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(
                    f"cannot decode exported color frame: {frame_path}"
                )
            writer.write(image)
    finally:
        writer.release()
    return {
        "path": "color/color.mp4",
        "derivative_only": True,
        "frame_count": len(frames),
        "fps": fps,
        "sha256": sha256_file(color_path),
        "bytes": color_path.stat().st_size,
    }


def stream_fingerprints(output_root: Path) -> dict[str, dict[str, Any]]:
    return {
        stream_name: fingerprint_stream(output_root / stream_name)
        for stream_name in STREAM_NAMES
    }


def total_export_payload_bytes(output_root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "metadata.json"
    )


def fingerprints_match(
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
) -> bool:
    keys = (
        "index_sha256",
        "frame_file_count",
        "frame_total_bytes",
        "frames_aggregate_sha256",
    )
    return all(
        first[stream][key] == second[stream][key]
        for stream in STREAM_NAMES
        for key in keys
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def aggregate_counts(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    record_list = list(records)
    return {
        "record_count": len(record_list),
        "complete_count": sum(
            record["complete"] is True for record in record_list
        ),
        "validation_error_count": sum(
            len(record["validation_errors"]) for record in record_list
        ),
        "color_frame_count": sum(
            record["streams"]["color"]["actual_frame_count"]
            for record in record_list
        ),
        "depth_frame_count": sum(
            record["streams"]["depth"]["actual_frame_count"]
            for record in record_list
        ),
        "ir_frame_count": sum(
            record["streams"]["ir"]["actual_frame_count"]
            for record in record_list
        ),
    }
