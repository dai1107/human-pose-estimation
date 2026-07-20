from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import pytest

from src.biomechanics.session_writer import SessionConfig, SessionWriter
from src.output_management import execute_cleanup, plan_cleanup
from src.output_schema import (
    OUTPUT_SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    ensure_supported_schema,
    versioned_payload,
)
from src.paths import SOURCE_ROOT, resolve_asset
from src.reference.session_loader import load_session


def test_versioned_payload_has_common_identity_fields() -> None:
    payload = versioned_payload("unit_test", {"value": 3})

    assert payload["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert payload["program_version"]
    assert payload["artifact_type"] == "unit_test"
    assert payload["value"] == 3


def test_schema_reader_accepts_legacy_and_rejects_future_versions() -> None:
    assert ensure_supported_schema({}, artifact_type="legacy") == 0
    with pytest.raises(UnsupportedSchemaVersion, match="upgrade"):
        ensure_supported_schema(
            {"schema_version": OUTPUT_SCHEMA_VERSION + 1},
            artifact_type="future",
        )


def test_session_outputs_have_json_and_csv_versions(tmp_path: Path) -> None:
    writer = SessionWriter(tmp_path)
    writer.start(
        SessionConfig(
            camera_index=0,
            width=640,
            height=480,
            mirror=False,
            smoothing=0.0,
            model_name="test",
            plot_on_save=False,
        ),
        session_id="versioned",
    )
    session_dir = writer.stop()
    assert session_dir is not None

    for filename in ("metadata.json", "summary.json", "sequence_summary.json"):
        payload = json.loads((session_dir / filename).read_text(encoding="utf-8"))
        assert payload["schema_version"] == OUTPUT_SCHEMA_VERSION
        assert payload["program_version"]
        assert payload["artifact_type"]

    for filename in ("landmarks.csv", "kinematics.csv"):
        with (session_dir / filename).open(encoding="utf-8", newline="") as handle:
            assert {"schema_version", "program_version"} <= set(
                csv.DictReader(handle).fieldnames or []
            )


def test_session_loader_rejects_newer_metadata(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "metadata.json").write_text(
        json.dumps({"schema_version": OUTPUT_SCHEMA_VERSION + 1}),
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedSchemaVersion):
        load_session(session_dir)


def test_cleanup_is_preview_only_until_apply_and_preserves_recent_items(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    old_item = root / "sessions" / "old"
    recent_item = root / "sessions" / "recent"
    old_item.mkdir(parents=True)
    recent_item.mkdir(parents=True)
    (old_item / "metadata.json").write_text("old", encoding="utf-8")
    (recent_item / "metadata.json").write_text("recent", encoding="utf-8")
    old_time = time.time() - 10 * 86400
    os.utime(old_item, (old_time, old_time))

    candidates = plan_cleanup(
        root,
        categories=["sessions"],
        older_than_days=5,
    )
    preview = execute_cleanup(root, candidates, apply=False)

    assert preview.candidate_count == 1
    assert old_item.exists()
    assert recent_item.exists()

    applied = execute_cleanup(root, candidates, apply=True)
    assert applied.deleted_count == 1
    assert not old_item.exists()
    assert recent_item.exists()


def test_cleanup_rejects_filesystem_root_and_unknown_categories(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="filesystem root"):
        plan_cleanup(Path(tmp_path.anchor))
    with pytest.raises(ValueError, match="unknown output categories"):
        plan_cleanup(tmp_path / "outputs", categories=["models"])


def test_asset_resolution_falls_back_to_bundled_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_asset("configs/hyrox/lunge.yaml")

    assert resolved == SOURCE_ROOT / "configs" / "hyrox" / "lunge.yaml"
