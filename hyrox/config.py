from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_LUNGE_CONFIG: dict[str, Any] = {
    "config_name": "lunge_default",
    "visibility_min": 0.45,
    "stand_knee_angle_min": 150.0,
    "stand_hip_angle_min": 145.0,
    "full_extension_knee_angle_min": 165.0,
    "full_extension_hip_angle_min": 160.0,
    "bottom_knee_angle_max": 115.0,
    "deep_knee_angle_max": 100.0,
    "torso_lean_warn": 20.0,
    "motion_tolerance": 3.0,
    "stable_frames": 3,
    "rep_cooldown_ms": 400,
}


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "null":
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def load_lunge_config(path: str | Path | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_LUNGE_CONFIG)
    config_path = Path(path) if path is not None else Path("configs/hyrox/lunge.yaml")
    if not config_path.exists():
        return config

    parsed: dict[str, Any] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parsed[key] = _parse_scalar(value)
    config.update(parsed)
    return config


__all__ = ["DEFAULT_LUNGE_CONFIG", "load_lunge_config"]
