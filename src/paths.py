from __future__ import annotations

import os
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def installation_root() -> Path:
    candidates = (SOURCE_ROOT, Path(sys.prefix))
    for candidate in candidates:
        if (candidate / "configs" / "hyrox").is_dir():
            return candidate
    return SOURCE_ROOT


def resolve_asset(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    candidates = (
        Path.cwd() / value,
        SOURCE_ROOT / value,
        Path(sys.prefix) / value,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def runtime_output_root() -> Path:
    configured = os.environ.get("POSE_OUTPUT_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.cwd() / "outputs"


__all__ = [
    "SOURCE_ROOT",
    "installation_root",
    "resolve_asset",
    "runtime_output_root",
]
