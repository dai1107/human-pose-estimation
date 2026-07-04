from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .session_loader import parse_number


DEFAULT_FEATURE_GROUPS: dict[str, list[str]] = {
    "angles": [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_shoulder_angle",
        "right_shoulder_angle",
        "trunk_tilt_proxy",
    ],
    "velocities": [
        "pelvis_speed",
        "left_wrist_speed",
        "right_wrist_speed",
        "left_ankle_speed",
        "right_ankle_speed",
    ],
    "angular_velocities": [
        "left_elbow_angular_velocity",
        "right_elbow_angular_velocity",
        "left_knee_angular_velocity",
        "right_knee_angular_velocity",
    ],
    "stability": [
        "pelvis_stability_proxy",
        "trunk_stability_proxy",
        "motion_energy_proxy",
    ],
}


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    groups: dict[str, list[str]]

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for group_names in self.groups.values():
            for name in group_names:
                if name not in names:
                    names.append(name)
        return names


@dataclass(frozen=True)
class ExtractedFeatures:
    matrix: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]
    valid_mask: np.ndarray
    processing: dict[str, Any]


def load_feature_config(path: str | Path | None = None) -> FeatureConfig:
    config_path = Path(path) if path is not None else Path("configs/reference_features.yaml")
    if not config_path.exists():
        return FeatureConfig("default_kinematics_v1", dict(DEFAULT_FEATURE_GROUPS))

    name = "default_kinematics_v1"
    groups: dict[str, list[str]] = {}
    current_group: str | None = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "name":
                name = value
                current_group = None
            else:
                current_group = key
                groups.setdefault(current_group, [])
            continue
        if current_group and line.strip().startswith("-"):
            groups[current_group].append(line.strip()[1:].strip())

    return FeatureConfig(name=name, groups=groups or dict(DEFAULT_FEATURE_GROUPS))


def _interpolate_series(values: np.ndarray) -> tuple[np.ndarray, str]:
    if values.size == 0:
        return values.astype(float), "empty"
    indices = np.arange(values.size)
    finite_mask = np.isfinite(values)
    if finite_mask.all():
        return values.astype(float), "unchanged"
    if not finite_mask.any():
        return np.zeros(values.size, dtype=float), "filled_zero_all_missing"
    if finite_mask.sum() == 1:
        return np.full(values.size, values[finite_mask][0], dtype=float), "filled_single_value"
    return np.interp(indices, indices[finite_mask], values[finite_mask]).astype(float), "interpolated"


def extract_feature_matrix(
    kinematics_rows: list[dict[str, Any]],
    feature_config: FeatureConfig | None = None,
) -> ExtractedFeatures:
    config = feature_config or load_feature_config()
    feature_names = config.feature_names
    timestamps = np.array([parse_number(row.get("timestamp_ms")) for row in kinematics_rows], dtype=float)
    columns: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    missing_features: list[str] = []
    handling: dict[str, str] = {}
    missing_counts: dict[str, int] = {}
    used_features: list[str] = []

    for name in feature_names:
        raw_values = np.array([parse_number(row.get(name)) for row in kinematics_rows], dtype=float)
        finite_mask = np.isfinite(raw_values)
        if not finite_mask.any():
            missing_features.append(name)
        else:
            used_features.append(name)
        filled, method = _interpolate_series(raw_values)
        handling[name] = method
        missing_counts[name] = int((~finite_mask).sum())
        columns.append(filled)
        masks.append(finite_mask)

    if columns:
        matrix = np.column_stack(columns).astype(float)
        valid_mask = np.column_stack(masks)
    else:
        matrix = np.empty((len(kinematics_rows), 0), dtype=float)
        valid_mask = np.empty((len(kinematics_rows), 0), dtype=bool)

    processing = {
        "strategy": "linear_interpolation_with_zero_for_all_missing",
        "features_requested": len(feature_names),
        "features_used": len(used_features),
        "missing_features": missing_features,
        "missing_counts": missing_counts,
        "handling": handling,
    }
    return ExtractedFeatures(matrix=matrix, timestamps=timestamps, feature_names=feature_names, valid_mask=valid_mask, processing=processing)

