"""Reproducible phase-zero baseline artifacts.

This module only observes the existing RGB/MediaPipe pipeline.  It does not
import OpenNI, load ONI files, enable neural inference, or change runtime
configuration.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.output_schema import (
    CSV_VERSION_FIELDS,
    LEGACY_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
)
from src.paths import resolve_asset
from src.product_pose import load_product_pose_config
from src.validation.golden_videos import GoldenObservation
from src.version import __version__


BASELINE_SCHEMA_VERSION = 1
DEFAULT_TAG_SUGGESTION = "v0.9-rule-dtw-baseline"
CORE_DISTRIBUTIONS = (
    "pose-estimation-hyrox",
    "mediapipe",
    "opencv-python",
    "numpy",
    "Flask",
    "Flask-Sock",
)
CONFIG_PATTERNS = ("*.yaml", "*.yml", "*.json")
MODEL_PATTERNS = ("*.task", "*.onnx", "*.pt")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    source = Path(path)
    base = Path(root)
    stat = source.stat()
    return {
        "path": source.relative_to(base).as_posix(),
        "bytes": stat.st_size,
        "sha256": sha256_file(source),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _run_git(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(
    project_root: str | Path,
    *,
    tag_suggestion: str = DEFAULT_TAG_SUGGESTION,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status_text = _run_git(root, "status", "--porcelain") or ""
    head = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "branch", "--show-current")
    config_files = sorted(
        {
            path
            for pattern in CONFIG_PATTERNS
            for path in (root / "configs").rglob(pattern)
            if "sample_pose_cache" not in path.parts
        }
    )
    model_files = sorted(
        {
            path
            for pattern in MODEL_PATTERNS
            for base in (root / "models", root)
            for path in base.glob(pattern)
            if path.is_file()
        }
    )
    requirements = sorted(root.glob("requirements*.txt"))
    dependency_declarations = "\n".join(
        [
            (root / "pyproject.toml").read_text(encoding="utf-8"),
            *[path.read_text(encoding="utf-8") for path in requirements],
        ]
    ).casefold()
    product_config = load_product_pose_config(root / "configs" / "product_pose.yaml")
    installed = {
        distribution: _distribution_version(distribution)
        for distribution in CORE_DISTRIBUTIONS
    }
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "artifact_type": "phase_zero_environment",
        "generated_at": utc_now(),
        "program_version": __version__,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "dependencies": installed,
        "requirements": [file_record(path, root=root) for path in requirements],
        "models": [file_record(path, root=root) for path in model_files],
        "configuration_files": [
            file_record(path, root=root) for path in config_files
        ],
        "git": {
            "head": head,
            "branch": branch,
            "working_tree_clean": not bool(status_text),
            "status_porcelain": status_text.splitlines(),
            "tag_suggestion": tag_suggestion,
            "tag_created": False,
            "tag_command_after_commit": (
                f'git tag -a {tag_suggestion} -m "Frozen rule/DTW/3D Assist baseline"'
            ),
            "tag_note": (
                "The baseline tool never mutates Git. Create the annotated tag only "
                "after reviewing and committing a clean baseline."
            ),
        },
        "isolation": {
            "default_pose_backend": product_config.backend,
            "oni_runtime_required": False,
            "openni_dependency_declared": any(
                name in dependency_declarations
                for name in ("openni", "orbbec", "pyrealsense")
            ),
            "neural_prediction_default_enabled": (
                product_config.local_first.neural_prediction_enabled
            ),
            "oni_files_read": False,
        },
        "host": {
            "logical_cpu_count": os.cpu_count(),
        },
    }


def snapshot_configs(
    project_root: str | Path,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    target_root = Path(output_dir) / "configs"
    records: list[dict[str, Any]] = []
    for pattern in CONFIG_PATTERNS:
        for source in sorted((root / "configs").rglob(pattern)):
            if "sample_pose_cache" in source.parts:
                continue
            relative = source.relative_to(root / "configs")
            destination = target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record = file_record(source, root=root)
            record["snapshot_path"] = destination.relative_to(
                Path(output_dir)
            ).as_posix()
            records.append(record)
    records.sort(key=lambda item: str(item["path"]))
    return records


def _nested_keys(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _nested_keys(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        examples = [_nested_keys(item) for item in value[:3]]
        return {"type": "array", "item_examples": examples}
    return type(value).__name__


class GoldenTraceCollector:
    """Collect additive evidence without changing analyzer decisions."""

    def __init__(self) -> None:
        self.phase_segments: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.count_events: list[dict[str, Any]] = []
        self.rule_status_totals: Counter[str] = Counter()
        self.output_schema: object = {}
        self._last_phase: str | None = None
        self._last_candidate_count = 0
        self._last_cycle_count = 0

    def observe(
        self,
        frame_index: int,
        timestamp_ms: int,
        state: Mapping[str, object],
    ) -> None:
        phase = str(state.get("phase", "unknown"))
        if phase != self._last_phase:
            if self.phase_segments:
                self.phase_segments[-1]["end_frame"] = frame_index - 1
                self.phase_segments[-1]["end_ms"] = timestamp_ms
            self.phase_segments.append(
                {
                    "phase": phase,
                    "start_frame": frame_index,
                    "end_frame": frame_index,
                    "start_ms": timestamp_ms,
                    "end_ms": timestamp_ms,
                }
            )
            self._last_phase = phase
        else:
            self.phase_segments[-1]["end_frame"] = frame_index
            self.phase_segments[-1]["end_ms"] = timestamp_ms

        candidate_count = int(state.get("candidate_count", 0) or 0)
        cycle_count = int(state.get("cycle_count", 0) or 0)
        if candidate_count > self._last_candidate_count:
            decision = copy.deepcopy(state.get("last_rep_decision"))
            candidate = copy.deepcopy(state.get("last_rep_candidate"))
            three_d = copy.deepcopy(state.get("last_three_d_assist"))
            record = {
                "candidate_index": candidate_count,
                "observed_frame": frame_index,
                "observed_timestamp_ms": timestamp_ms,
                "candidate": candidate,
                "decision": decision,
                "three_d_assist": three_d,
            }
            self.candidates.append(record)
            if isinstance(decision, Mapping):
                for rule in decision.get("rules") or []:
                    if isinstance(rule, Mapping):
                        self.rule_status_totals[
                            f"{rule.get('rule_id')}:{rule.get('status')}"
                        ] += 1
            self.count_events.append(
                {
                    "type": "candidate",
                    "frame": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "value": candidate_count,
                }
            )
        if cycle_count > self._last_cycle_count:
            self.count_events.append(
                {
                    "type": "cycle",
                    "frame": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "value": cycle_count,
                }
            )
        self._last_candidate_count = candidate_count
        self._last_cycle_count = cycle_count
        self.output_schema = _nested_keys(state)

    def report(self) -> dict[str, Any]:
        return {
            "phase_segments": self.phase_segments,
            "candidates": self.candidates,
            "count_events": self.count_events,
            "rule_status_totals": dict(sorted(self.rule_status_totals.items())),
            "action_output_schema": self.output_schema,
            "dtw": {
                "status": "not_configured",
                "normalized_distance": None,
                "reason": (
                    "No versioned reference action is configured for the bundled "
                    "golden videos; the standalone DTW implementation remains covered "
                    "by unit tests."
                ),
            },
        }


def build_schema_snapshot(
    golden_report: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    action_schemas: dict[str, object] = {}
    if golden_report:
        for case in golden_report.get("cases") or []:
            if not isinstance(case, Mapping):
                continue
            trace = case.get("trace")
            observation = case.get("observation")
            if isinstance(trace, Mapping) and isinstance(observation, Mapping):
                action_schemas[str(observation.get("action", "unknown"))] = trace.get(
                    "action_output_schema", {}
                )
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "artifact_type": "phase_zero_output_schema_snapshot",
        "generated_at": utc_now(),
        "program_version": __version__,
        "output_schema_constants": {
            "current": OUTPUT_SCHEMA_VERSION,
            "legacy": LEGACY_SCHEMA_VERSION,
            "csv_version_fields": list(CSV_VERSION_FIELDS),
        },
        "golden_observation_fields": [
            field.name for field in fields(GoldenObservation)
        ],
        "action_output_schemas": action_schemas,
        "compatibility_contract": {
            "existing_fields_may_be_removed": False,
            "existing_fields_may_be_renamed": False,
            "new_optional_fields_may_be_added": True,
        },
    }


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def golden_video_inventory(
    project_root: str | Path,
    video_paths: Sequence[str],
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    return [
        file_record(resolve_asset(video), root=root)
        for video in video_paths
    ]


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "DEFAULT_TAG_SUGGESTION",
    "GoldenTraceCollector",
    "build_schema_snapshot",
    "collect_environment",
    "file_record",
    "golden_video_inventory",
    "sha256_file",
    "snapshot_configs",
    "utc_now",
    "write_json",
]
