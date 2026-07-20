from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.version import __version__


OUTPUT_SCHEMA_VERSION = 1
LEGACY_SCHEMA_VERSION = 0
CSV_VERSION_FIELDS = ("schema_version", "program_version")


class UnsupportedSchemaVersion(ValueError):
    error_code = "SCH001"

    def __init__(self, found: int, supported: int, artifact_type: str) -> None:
        self.found = found
        self.supported = supported
        self.artifact_type = artifact_type
        super().__init__(
            f"{artifact_type} schema version {found} is newer than supported "
            f"version {supported}; upgrade pose-estimation-hyrox"
        )


def artifact_metadata(
    artifact_type: str,
    *,
    schema_version: int = OUTPUT_SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "schema_version": int(schema_version),
        "program_version": __version__,
        "artifact_type": str(artifact_type),
    }


def versioned_payload(
    artifact_type: str,
    payload: Mapping[str, Any],
    *,
    schema_version: int = OUTPUT_SCHEMA_VERSION,
) -> dict[str, Any]:
    output = artifact_metadata(artifact_type, schema_version=schema_version)
    output.update(dict(payload))
    return output


def versioned_csv_columns(columns: list[str] | tuple[str, ...]) -> list[str]:
    output = list(columns)
    for field in CSV_VERSION_FIELDS:
        if field not in output:
            output.append(field)
    return output


def versioned_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output.setdefault("schema_version", OUTPUT_SCHEMA_VERSION)
    output.setdefault("program_version", __version__)
    return output


def schema_version_of(payload: Mapping[str, Any]) -> int:
    raw = payload.get("schema_version", LEGACY_SCHEMA_VERSION)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid schema_version: {raw!r}") from exc


def ensure_supported_schema(
    payload: Mapping[str, Any],
    *,
    artifact_type: str,
    supported_version: int = OUTPUT_SCHEMA_VERSION,
) -> int:
    version = schema_version_of(payload)
    if version > supported_version:
        raise UnsupportedSchemaVersion(version, supported_version, artifact_type)
    if version < LEGACY_SCHEMA_VERSION:
        raise ValueError(f"invalid negative schema_version: {version}")
    return version


__all__ = [
    "CSV_VERSION_FIELDS",
    "LEGACY_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "UnsupportedSchemaVersion",
    "artifact_metadata",
    "ensure_supported_schema",
    "schema_version_of",
    "versioned_csv_columns",
    "versioned_csv_row",
    "versioned_payload",
]
