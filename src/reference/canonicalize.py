from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SIDE_FEATURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_elbow_angle", "right_elbow_angle"),
    ("left_knee_angle", "right_knee_angle"),
    ("left_hip_angle", "right_hip_angle"),
    ("left_shoulder_angle", "right_shoulder_angle"),
    ("left_wrist_speed", "right_wrist_speed"),
    ("left_ankle_speed", "right_ankle_speed"),
    ("left_elbow_angular_velocity", "right_elbow_angular_velocity"),
    ("left_knee_angular_velocity", "right_knee_angular_velocity"),
    ("left_hip_angular_velocity", "right_hip_angular_velocity"),
)


@dataclass(frozen=True)
class CanonicalizationResult:
    rows: list[dict[str, Any]]
    applied: bool
    original_side: str
    canonical_side: str | None
    message: str


def should_mirror(movement_side: str | None, canonical_side: str | None) -> bool:
    if canonical_side not in {"left", "right"}:
        return False
    return movement_side in {"left", "right"} and movement_side != canonical_side


def mirror_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mirrored: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        for left_name, right_name in SIDE_FEATURE_PAIRS:
            if left_name in row or right_name in row:
                new_row[left_name] = row.get(right_name, "")
                new_row[right_name] = row.get(left_name, "")
        mirrored.append(new_row)
    return mirrored


def canonicalize_feature_rows(
    rows: list[dict[str, Any]],
    movement_side: str | None,
    canonical_side: str | None = None,
) -> CanonicalizationResult:
    if not should_mirror(movement_side, canonical_side):
        return CanonicalizationResult(list(rows), False, movement_side or "unknown", canonical_side, "not applied")
    return CanonicalizationResult(
        mirror_feature_rows(rows),
        True,
        movement_side or "unknown",
        canonical_side,
        "left/right kinematic features swapped",
    )


def canonicalize_feature_matrix(
    matrix: np.ndarray,
    feature_names: list[str],
    movement_side: str | None,
    canonical_side: str | None = None,
) -> tuple[np.ndarray, bool]:
    if not should_mirror(movement_side, canonical_side):
        return np.asarray(matrix, dtype=float).copy(), False
    mirrored = np.asarray(matrix, dtype=float).copy()
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    for left_name, right_name in SIDE_FEATURE_PAIRS:
        left_index = name_to_index.get(left_name)
        right_index = name_to_index.get(right_name)
        if left_index is None or right_index is None:
            continue
        mirrored[:, [left_index, right_index]] = mirrored[:, [right_index, left_index]]
    return mirrored, True


def mirror_landmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mirrored: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        name = str(row.get("landmark_name", ""))
        if name.startswith("left_"):
            new_row["landmark_name"] = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
            new_row["landmark_name"] = "left_" + name.removeprefix("right_")
        for field in ("world_x", "smoothed_x"):
            try:
                new_row[field] = str(-float(row[field]))
            except (KeyError, TypeError, ValueError):
                pass
        try:
            value = float(row["image_x"])
            new_row["image_x"] = str(1.0 - value) if 0.0 <= value <= 1.0 else str(-value)
        except (KeyError, TypeError, ValueError):
            pass
        mirrored.append(new_row)
    return mirrored

