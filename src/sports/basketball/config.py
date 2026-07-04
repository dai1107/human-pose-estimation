from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "analysis_name": "basketball_shot_v1",
    "template_name": "set_shot_chain_v1",
    "data_quality": {
        "minimum_pose_valid_ratio": 0.70,
        "minimum_landmark_visibility": 0.55,
    },
    "phase_detection": {
        "dip_displacement_threshold": 0.06,
        "arm_extension_elbow_delta": 8.0,
        "wrist_above_shoulder_threshold": -0.05,
        "follow_through_ms": 250,
    },
    "release_proxy": {
        "minimum_confidence": 0.35,
        "wrist_peak_weight": 0.35,
        "elbow_extension_weight": 0.25,
        "wrist_height_weight": 0.25,
        "follow_through_weight": 0.15,
    },
    "sequence_template": {
        "events": [
            "pelvis_upward_speed_peak",
            "shooting_side_knee_extension_peak",
            "shooting_side_hip_extension_peak",
            "shoulder_elevation_peak",
            "shooting_side_elbow_extension_peak",
            "shooting_side_wrist_speed_peak",
            "release_proxy_time",
        ],
        "rules": {
            "allow_event_overlap_ms": 80,
            "allow_missing_events": True,
            "require_release_proxy": False,
        },
    },
    "interpretation": {
        "mode": "descriptive_only",
        "allow_standard_verdict": False,
        "evidence_source": None,
        "evidence_note": "Do not enable until externally verified.",
    },
}


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw == "null":
        return None
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip('"').strip("'")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_basketball_config(path: str | Path | None = None) -> dict[str, Any]:
    config = _merge({}, DEFAULT_CONFIG)
    config_path = Path(path) if path else Path("configs/basketball_shot_v1.yaml")
    if not config_path.exists():
        return config
    parsed: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, parsed)]
    last_key_at_indent: dict[int, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if isinstance(parent, list):
                parent.append(_parse_scalar(stripped[2:]))
            continue
        if ":" not in stripped or not isinstance(parent, dict):
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] | list[Any]
            child = [] if key == "events" else {}
            parent[key] = child
            stack.append((indent, child))
            last_key_at_indent[indent] = key
    return _merge(config, parsed)

