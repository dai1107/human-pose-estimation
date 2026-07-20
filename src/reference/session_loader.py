from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import isfinite, nan
from pathlib import Path
from typing import Any, Iterable

from src.output_schema import (
    ensure_supported_schema,
    versioned_csv_columns,
    versioned_csv_row,
)

@dataclass(frozen=True)
class SessionData:
    path: Path
    session_id: str
    metadata: dict[str, Any]
    kinematics: list[dict[str, Any]]
    landmarks: list[dict[str, Any]]


def parse_number(value: Any) -> float:
    if value is None:
        return nan
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return nan
    try:
        return float(text)
    except ValueError:
        return nan


def finite_number(value: Any) -> bool:
    return isfinite(parse_number(value))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or [])
    if not columns:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    columns = versioned_csv_columns(columns)
    versioned_rows = [versioned_csv_row(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(versioned_rows)


def load_session(session_dir: str | Path) -> SessionData:
    path = Path(session_dir)
    if not path.exists():
        raise FileNotFoundError(f"session directory not found: {path}")
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    ensure_supported_schema(metadata, artifact_type="pose_session")
    session_id = str(metadata.get("session_id") or path.name)
    return SessionData(
        path=path,
        session_id=session_id,
        metadata=metadata,
        kinematics=read_csv_rows(path / "kinematics.csv"),
        landmarks=read_csv_rows(path / "landmarks.csv"),
    )


def timestamps_from_rows(rows: list[dict[str, Any]]) -> list[float]:
    return [parse_number(row.get("timestamp_ms")) for row in rows]


def frame_indices_from_rows(rows: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = parse_number(row.get("frame_index"))
        values.append(int(value) if isfinite(value) else -1)
    return values


def filter_rows(
    rows: list[dict[str, Any]],
    start_ms: int | None = None,
    end_ms: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        timestamp = parse_number(row.get("timestamp_ms"))
        frame_index = parse_number(row.get("frame_index"))
        if start_ms is not None and (not isfinite(timestamp) or timestamp < start_ms):
            continue
        if end_ms is not None and (not isfinite(timestamp) or timestamp > end_ms):
            continue
        if start_frame is not None and (not isfinite(frame_index) or frame_index < start_frame):
            continue
        if end_frame is not None and (not isfinite(frame_index) or frame_index > end_frame):
            continue
        filtered.append(dict(row))
    return filtered


def available_numeric_fields(rows: list[dict[str, Any]]) -> list[str]:
    ignored = {"frame_index", "timestamp_ms", "pose_detected"}
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in ignored or key in fields:
                continue
            if finite_number(value):
                fields.append(key)
    return fields


def numeric_summary(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for field in fields:
        values = [parse_number(row.get(field)) for row in rows]
        finite = [value for value in values if isfinite(value)]
        if not finite:
            result[field] = {"mean": None, "min": None, "max": None}
            continue
        result[field] = {
            "mean": float(sum(finite) / len(finite)),
            "min": float(min(finite)),
            "max": float(max(finite)),
        }
    return result
