from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any

from .session_loader import parse_number


DEFAULT_RULES: dict[str, float] = {
    "minimum_pose_valid_ratio": 0.75,
    "maximum_timestamp_gap_ms": 200.0,
    "minimum_clip_duration_ms": 300.0,
    "minimum_visible_landmark_ratio": 0.70,
}


@dataclass(frozen=True)
class QualityReport:
    status: str
    metrics: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_quality_rules(path: str | Path | None = None) -> dict[str, float]:
    rules = dict(DEFAULT_RULES)
    config_path = Path(path) if path is not None else Path("configs/reference_quality.yaml")
    if not config_path.exists():
        return rules
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        try:
            rules[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return rules


def _truthy_pose(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(parse_number(value))


def _timestamp_deltas(rows: list[dict[str, Any]]) -> list[float]:
    timestamps = [parse_number(row.get("timestamp_ms")) for row in rows]
    timestamps = [value for value in timestamps if isfinite(value)]
    return [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]


def evaluate_quality(
    kinematics_rows: list[dict[str, Any]],
    landmarks_rows: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    rules: dict[str, float] | None = None,
    camera_view: str | None = None,
) -> QualityReport:
    active_rules = dict(DEFAULT_RULES)
    active_rules.update(rules or {})
    metadata = metadata or {}
    total_frames = len(kinematics_rows)
    valid_frames = sum(1 for row in kinematics_rows if _truthy_pose(row.get("pose_detected")))
    pose_valid_ratio = valid_frames / total_frames if total_frames else 0.0

    visibility_values: list[float] = []
    missing_values: list[float] = []
    if landmarks_rows:
        for row in landmarks_rows:
            visibility = parse_number(row.get("visibility"))
            presence = parse_number(row.get("presence"))
            if isfinite(visibility):
                visibility_values.append(visibility)
            missing_values.append(0.0 if isfinite(visibility) and visibility >= 0.2 and (not isfinite(presence) or presence >= 0.2) else 1.0)
    else:
        for row in kinematics_rows:
            visibility = parse_number(row.get("visibility_mean"))
            missing = parse_number(row.get("missing_ratio"))
            if isfinite(visibility):
                visibility_values.append(visibility)
            if isfinite(missing):
                missing_values.append(missing)

    visibility_mean = sum(visibility_values) / len(visibility_values) if visibility_values else 0.0
    missing_ratio = sum(missing_values) / len(missing_values) if missing_values else 1.0
    visible_landmark_ratio = max(0.0, min(1.0, 1.0 - missing_ratio))

    deltas = _timestamp_deltas(kinematics_rows)
    max_gap = max(deltas) if deltas else 0.0
    gap_limit = active_rules["maximum_timestamp_gap_ms"]
    timestamp_gap_ratio = sum(1 for value in deltas if value > gap_limit) / len(deltas) if deltas else 0.0
    frame_rate_estimate = 1000.0 / median(deltas) if deltas and median(deltas) > 0 else 0.0
    timestamps = [parse_number(row.get("timestamp_ms")) for row in kinematics_rows]
    finite_timestamps = [value for value in timestamps if isfinite(value)]
    duration_ms = max(finite_timestamps) - min(finite_timestamps) if len(finite_timestamps) >= 2 else 0.0

    view = camera_view or str(metadata.get("camera_view", "unknown"))
    metrics: dict[str, Any] = {
        "pose_valid_ratio": pose_valid_ratio,
        "landmark_visibility_mean": visibility_mean,
        "landmark_missing_ratio": missing_ratio,
        "visible_landmark_ratio": visible_landmark_ratio,
        "timestamp_gap_ratio": timestamp_gap_ratio,
        "maximum_timestamp_gap_ms": max_gap,
        "frame_rate_estimate": frame_rate_estimate,
        "motion_duration_ms": duration_ms,
        "camera_view_known": view != "unknown",
        "mirror_state_known": "mirror" in metadata or "mirror_at_end" in metadata,
        "frame_count": total_frames,
        "pose_detected_frame_count": valid_frames,
    }

    warnings: list[str] = []
    if pose_valid_ratio < active_rules["minimum_pose_valid_ratio"]:
        warnings.append("pose valid ratio is below the comparison threshold")
    if max_gap > active_rules["maximum_timestamp_gap_ms"]:
        warnings.append("timestamp gap is larger than the configured threshold")
    if duration_ms < active_rules["minimum_clip_duration_ms"]:
        warnings.append("motion clip duration is shorter than the configured threshold")
    if visible_landmark_ratio < active_rules["minimum_visible_landmark_ratio"]:
        warnings.append("visible landmark ratio is below the comparison threshold")
    if view == "unknown":
        warnings.append("camera view is unknown")
    if not metrics["mirror_state_known"]:
        warnings.append("mirror state is unknown")

    return QualityReport(status="WARNING" if warnings else "PASS", metrics=metrics, warnings=warnings)

