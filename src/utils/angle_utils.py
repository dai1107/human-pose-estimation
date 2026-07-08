from __future__ import annotations

from math import acos, degrees, isfinite
from typing import Mapping, Sequence

import numpy as np

from src.backends.base import Keypoint, PoseResult


def keypoint_map(source: PoseResult | Sequence[Keypoint] | Mapping[str, Keypoint]) -> dict[str, Keypoint]:
    if isinstance(source, PoseResult):
        return {point.name: point for point in source.keypoints}
    if isinstance(source, Mapping):
        return dict(source)
    return {point.name: point for point in source}


def usable_point(point: Keypoint | None, min_confidence: float = 0.2) -> np.ndarray | None:
    if point is None or point.confidence < min_confidence:
        return None
    if not all(isfinite(value) for value in (point.x, point.y, point.z)):
        return None
    return np.array([point.x, point.y, point.z], dtype=float)


def angle(a: Keypoint | None, b: Keypoint | None, c: Keypoint | None, min_confidence: float = 0.2) -> float | None:
    pa = usable_point(a, min_confidence)
    pb = usable_point(b, min_confidence)
    pc = usable_point(c, min_confidence)
    if pa is None or pb is None or pc is None:
        return None
    ab = pa - pb
    cb = pc - pb
    norm = float(np.linalg.norm(ab) * np.linalg.norm(cb))
    if norm <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(ab, cb) / norm, -1.0, 1.0))
    value = degrees(float(acos(cosine)))
    return value if isfinite(value) else None


def joint_angle(
    points: PoseResult | Sequence[Keypoint] | Mapping[str, Keypoint],
    first: str,
    middle: str,
    third: str,
    min_confidence: float = 0.2,
) -> float | None:
    by_name = keypoint_map(points)
    return angle(by_name.get(first), by_name.get(middle), by_name.get(third), min_confidence=min_confidence)


def knee_angle(points: PoseResult | Sequence[Keypoint] | Mapping[str, Keypoint], side: str = "left") -> float | None:
    return joint_angle(points, f"{side}_hip", f"{side}_knee", f"{side}_ankle")


def hip_angle(points: PoseResult | Sequence[Keypoint] | Mapping[str, Keypoint], side: str = "left") -> float | None:
    return joint_angle(points, f"{side}_shoulder", f"{side}_hip", f"{side}_knee")


def elbow_angle(points: PoseResult | Sequence[Keypoint] | Mapping[str, Keypoint], side: str = "left") -> float | None:
    return joint_angle(points, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")


def shoulder_angle(points: PoseResult | Sequence[Keypoint] | Mapping[str, Keypoint], side: str = "left") -> float | None:
    return joint_angle(points, f"{side}_hip", f"{side}_shoulder", f"{side}_elbow")


def trunk_lean_angle(points: PoseResult | Sequence[Keypoint] | Mapping[str, Keypoint], min_confidence: float = 0.2) -> float | None:
    by_name = keypoint_map(points)
    left_hip = usable_point(by_name.get("left_hip"), min_confidence)
    right_hip = usable_point(by_name.get("right_hip"), min_confidence)
    left_shoulder = usable_point(by_name.get("left_shoulder"), min_confidence)
    right_shoulder = usable_point(by_name.get("right_shoulder"), min_confidence)
    if left_hip is None or right_hip is None or left_shoulder is None or right_shoulder is None:
        return None
    pelvis = (left_hip + right_hip) / 2.0
    shoulder = (left_shoulder + right_shoulder) / 2.0
    trunk = (shoulder - pelvis)[:2]
    norm = float(np.linalg.norm(trunk))
    if norm <= 1e-12:
        return None
    vertical_up = np.array([0.0, -1.0], dtype=float)
    cosine = float(np.clip(np.dot(trunk, vertical_up) / norm, -1.0, 1.0))
    value = degrees(float(acos(cosine)))
    return value if isfinite(value) else None


def body_angles(points: PoseResult | Sequence[Keypoint] | Mapping[str, Keypoint]) -> dict[str, float | None]:
    return {
        "left_knee_angle": knee_angle(points, "left"),
        "right_knee_angle": knee_angle(points, "right"),
        "left_hip_angle": hip_angle(points, "left"),
        "right_hip_angle": hip_angle(points, "right"),
        "left_elbow_angle": elbow_angle(points, "left"),
        "right_elbow_angle": elbow_angle(points, "right"),
        "left_shoulder_angle": shoulder_angle(points, "left"),
        "right_shoulder_angle": shoulder_angle(points, "right"),
        "trunk_lean_angle": trunk_lean_angle(points),
    }
