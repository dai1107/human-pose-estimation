"""Round-eight coordinate contracts and relative-3D quality checks."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.backends.base import Keypoint
from src.biomechanics.kinematics_3d import (
    ANGLE_DEFINITIONS_3D,
    calculate_angle_2d,
    calculate_angle_3d,
)
from src.utils.keypoint_schema import MEDIAPIPE_33_NAMES, MEDIAPIPE_CONNECTIONS


COORDINATE_CONTRACT_VERSION = "hyrox_coordinate_spaces_v1"
BODY_CANONICAL_VERSION = "hip_torso_axes_v1"
ESTIMATED_INTRINSICS_VERSION = "uncalibrated_60deg_horizontal_fov_v1"


def _finite(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def point_payload(
    point: Keypoint,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    quality_reasons: Sequence[str] = (),
    covariance_scale: float = 1.0,
) -> dict[str, Any]:
    confidence = max(0.0, min(1.0, float(point.confidence)))
    variance = ((1.0 - confidence) ** 2 + 1e-6) * float(covariance_scale)
    return {
        "name": point.name,
        "x": _finite(point.x if x is None else x),
        "y": _finite(point.y if y is None else y),
        "z": _finite(point.z if z is None else z),
        "confidence": confidence,
        "visibility": _finite(point.visibility),
        "presence": _finite(point.presence),
        "source_backend": point.source_model,
        "covariance_proxy_diagonal": [variance, variance, variance],
        "quality_reasons": list(quality_reasons),
    }


def estimated_intrinsics(width: int, height: int) -> dict[str, Any]:
    width = max(1, int(width))
    height = max(1, int(height))
    horizontal_fov_deg = 60.0
    focal = (width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
    return {
        "status": "estimated_intrinsics",
        "version": ESTIMATED_INTRINSICS_VERSION,
        "source": (
            "generic 60 degree horizontal FOV assumption; no device identity or "
            "checkerboard/ChArUco calibration is available"
        ),
        "camera_matrix": [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": None,
        "resolution": [width, height],
        "device_id": "device_pending",
        "horizontal_fov_assumption_deg": horizontal_fov_deg,
        "focal_length_relative_uncertainty": 0.25,
        "principal_point_uncertainty_pixels": 0.05 * max(width, height),
        "absolute_depth_available": False,
        "metric_position_available": False,
    }


def camera_ray_directions(
    points: Sequence[Keypoint],
    *,
    width: int,
    height: int,
    intrinsics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=float)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    output: list[dict[str, Any]] = []
    for point in points:
        x = _finite(point.x)
        y = _finite(point.y)
        if x is None or y is None:
            output.append(
                {
                    **point_payload(
                        point,
                        x=float("nan"),
                        y=float("nan"),
                        z=float("nan"),
                        quality_reasons=("image_joint_missing",),
                    ),
                    "absolute_depth": None,
                }
            )
            continue
        pixel_x, pixel_y = x * width, y * height
        ray = np.asarray([(pixel_x - cx) / fx, (pixel_y - cy) / fy, 1.0])
        norm = float(np.linalg.norm(ray))
        ray = ray / max(norm, 1e-12)
        output.append(
            {
                **point_payload(
                    point,
                    x=float(ray[0]),
                    y=float(ray[1]),
                    z=float(ray[2]),
                    quality_reasons=("estimated_intrinsics", "direction_only"),
                    covariance_scale=2.0,
                ),
                "absolute_depth": None,
            }
        )
    return output


def _usable_map(points: Sequence[Keypoint], confidence: float = 0.2) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for point in points:
        values = np.asarray([point.x, point.y, point.z], dtype=float)
        if point.confidence >= confidence and np.all(np.isfinite(values)):
            output[point.name] = values
    return output


def body_canonical_transform(
    world_points: Sequence[Keypoint],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points = _usable_map(world_points)
    required = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if any(name not in points for name in required):
        return [], {
            "status": "unavailable",
            "version": BODY_CANONICAL_VERSION,
            "reason": "hip_or_shoulder_axis_missing",
            "world_to_body_canonical": None,
            "body_canonical_to_world": None,
        }
    hip_center = (points["left_hip"] + points["right_hip"]) / 2.0
    shoulder_center = (
        points["left_shoulder"] + points["right_shoulder"]
    ) / 2.0
    lateral = points["right_hip"] - points["left_hip"]
    up = shoulder_center - hip_center
    scale = float(np.linalg.norm(up))
    lateral_norm = float(np.linalg.norm(lateral))
    if scale <= 1e-8 or lateral_norm <= 1e-8:
        return [], {
            "status": "unavailable",
            "version": BODY_CANONICAL_VERSION,
            "reason": "degenerate_torso_or_hip_axis",
            "world_to_body_canonical": None,
            "body_canonical_to_world": None,
        }
    lateral = lateral / lateral_norm
    up = up - lateral * float(np.dot(up, lateral))
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-8:
        return [], {
            "status": "unavailable",
            "version": BODY_CANONICAL_VERSION,
            "reason": "degenerate_orthogonal_axes",
            "world_to_body_canonical": None,
            "body_canonical_to_world": None,
        }
    up = up / up_norm
    forward = np.cross(lateral, up)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-12)
    rotation = np.vstack([lateral, up, forward]) / scale
    translation = -rotation @ hip_center
    forward_matrix = np.eye(4)
    forward_matrix[:3, :3] = rotation
    forward_matrix[:3, 3] = translation
    inverse_matrix = np.linalg.inv(forward_matrix)
    canonical: list[dict[str, Any]] = []
    by_name = {point.name: point for point in world_points}
    for name in MEDIAPIPE_33_NAMES:
        point = by_name.get(name)
        if point is None:
            continue
        vector = np.asarray([point.x, point.y, point.z, 1.0], dtype=float)
        if not np.all(np.isfinite(vector)):
            canonical.append(
                point_payload(
                    point,
                    x=float("nan"),
                    y=float("nan"),
                    z=float("nan"),
                    quality_reasons=("world_joint_missing",),
                )
            )
            continue
        transformed = forward_matrix @ vector
        canonical.append(
            point_payload(
                point,
                x=float(transformed[0]),
                y=float(transformed[1]),
                z=float(transformed[2]),
            )
        )
    roundtrip = inverse_matrix @ forward_matrix
    return canonical, {
        "status": "available",
        "version": BODY_CANONICAL_VERSION,
        "origin": "midpoint(left_hip,right_hip)",
        "scale": "hip_center_to_shoulder_center",
        "axes": {
            "x": "left_hip_to_right_hip",
            "y": "hip_center_to_shoulder_center_orthogonalized",
            "z": "cross(x,y); relative body-facing axis, not camera extrinsics",
        },
        "torso_scale_mp_world_units": scale,
        "world_to_body_canonical": forward_matrix.tolist(),
        "body_canonical_to_world": inverse_matrix.tolist(),
        "roundtrip_matrix_max_abs_error": float(
            np.max(np.abs(roundtrip - np.eye(4)))
        ),
    }


def coordinate_layers(
    image_points: Sequence[Keypoint],
    world_points: Sequence[Keypoint],
    *,
    width: int,
    height: int,
    intrinsics: Mapping[str, Any],
    per_joint_reasons: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    reasons = per_joint_reasons or {}
    image_normalized = [
        point_payload(point, quality_reasons=reasons.get(point.name, ()))
        for point in image_points
    ]
    image_pixel = [
        point_payload(
            point,
            x=float(point.x) * width,
            y=float(point.y) * height,
            z=float(point.z) * width,
            quality_reasons=reasons.get(point.name, ()),
        )
        for point in image_points
    ]
    world = [
        point_payload(
            point,
            quality_reasons=reasons.get(point.name, ()),
            covariance_scale=1.5,
        )
        for point in world_points
    ]
    canonical, canonical_transform = body_canonical_transform(world_points)
    return {
        "contract_version": COORDINATE_CONTRACT_VERSION,
        "image_normalized_2d": {
            "status": "available" if image_points else "unavailable",
            "units": "normalized_image_fraction",
            "points": image_normalized,
        },
        "image_pixel_2d": {
            "status": "available" if image_points else "unavailable",
            "units": "source_image_pixels",
            "source_resolution": [width, height],
            "points": image_pixel,
        },
        "camera_ray_direction_3d": {
            "status": "estimated_intrinsics" if image_points else "unavailable",
            "units": "unit_direction_vector_no_depth",
            "intrinsics": dict(intrinsics),
            "points": camera_ray_directions(
                image_points,
                width=width,
                height=height,
                intrinsics=intrinsics,
            ),
            "absolute_metric_accuracy": "not_applicable",
        },
        "mp_world_body_3d": {
            "status": "available" if world_points else "unavailable",
            "units": "mediapipe_body_relative_world_units",
            "is_camera_coordinate_system": False,
            "is_absolute_metric_ground_truth": False,
            "points": world,
        },
        "body_canonical_3d": {
            "status": canonical_transform["status"],
            "units": "torso_scale_normalized_body_axes",
            "points": canonical,
            "transform": canonical_transform,
            "is_absolute_metric_ground_truth": False,
        },
        "oni_surface_metric_3d": {
            "status": "unavailable_phone_rgb_round8",
            "points": [],
            "reason": (
                "reserved for the same ONI record and reliable depth intervals "
                "in round 11; phone-to-ONI pairing is forbidden"
            ),
        },
        "coordinate_mixing_policy": {
            "equipment_and_floor_space": "image_pixel_2d_or_calibrated_camera_only",
            "body_canonical_must_not_mix_with_scene_geometry": True,
        },
    }


@dataclass
class CoordinateQualityTracker:
    max_bone_change_ratio: float = 0.20
    max_z_change_body_scale: float = 0.35
    max_2d_3d_angle_difference_deg: float = 25.0
    previous_world: dict[str, np.ndarray] = field(default_factory=dict)
    previous_bones: dict[tuple[str, str], float] = field(default_factory=dict)
    previous_forward: np.ndarray | None = None
    flag_counts: Counter[str] = field(default_factory=Counter)

    def update(
        self,
        image_points: Sequence[Keypoint],
        world_points: Sequence[Keypoint],
    ) -> tuple[list[str], dict[str, list[str]]]:
        image = _usable_map(image_points)
        world = _usable_map(world_points)
        frame_flags: set[str] = set()
        joint_reasons: dict[str, set[str]] = {
            name: set() for name in MEDIAPIPE_33_NAMES
        }
        bones: dict[tuple[str, str], float] = {}
        for first_index, second_index in MEDIAPIPE_CONNECTIONS:
            first = MEDIAPIPE_33_NAMES[first_index]
            second = MEDIAPIPE_33_NAMES[second_index]
            if first not in world or second not in world:
                continue
            length = float(np.linalg.norm(world[first] - world[second]))
            bones[(first, second)] = length
            previous = self.previous_bones.get((first, second))
            if (
                previous is not None
                and previous > 1e-8
                and abs(length - previous) / previous
                > self.max_bone_change_ratio
            ):
                frame_flags.add("bone_length_jump")
                joint_reasons[first].add("bone_length_jump")
                joint_reasons[second].add("bone_length_jump")
        torso_scale = None
        if all(name in world for name in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")):
            hip = (world["left_hip"] + world["right_hip"]) / 2.0
            shoulder = (world["left_shoulder"] + world["right_shoulder"]) / 2.0
            torso_scale = float(np.linalg.norm(shoulder - hip))
            lateral = world["right_hip"] - world["left_hip"]
            up = shoulder - hip
            forward = np.cross(lateral, up)
            norm = float(np.linalg.norm(forward))
            if norm > 1e-8:
                forward = forward / norm
                if (
                    self.previous_forward is not None
                    and float(np.dot(forward, self.previous_forward)) < -0.25
                ):
                    frame_flags.add("body_orientation_flip")
                self.previous_forward = forward
        if torso_scale is not None and torso_scale > 1e-8:
            for name, point in world.items():
                previous = self.previous_world.get(name)
                if (
                    previous is not None
                    and abs(float(point[2] - previous[2])) / torso_scale
                    > self.max_z_change_body_scale
                ):
                    frame_flags.add("z_jump")
                    joint_reasons[name].add("z_jump")
        if all(name in world for name in ("left_hip", "right_hip")) and all(
            name in self.previous_world for name in ("left_hip", "right_hip")
        ):
            keep = np.linalg.norm(world["left_hip"] - self.previous_world["left_hip"])
            keep += np.linalg.norm(world["right_hip"] - self.previous_world["right_hip"])
            swap = np.linalg.norm(world["left_hip"] - self.previous_world["right_hip"])
            swap += np.linalg.norm(world["right_hip"] - self.previous_world["left_hip"])
            if swap < 0.75 * keep:
                frame_flags.add("left_right_swap_candidate")
                joint_reasons["left_hip"].add("left_right_swap_candidate")
                joint_reasons["right_hip"].add("left_right_swap_candidate")
        angle_conflicts = 0
        for definition in ANGLE_DEFINITIONS_3D.values():
            if not all(name in image and name in world for name in definition):
                continue
            angle_2d = calculate_angle_2d(
                image[definition[0]][:2],
                image[definition[1]][:2],
                image[definition[2]][:2],
            )
            angle_3d = calculate_angle_3d(
                world[definition[0]],
                world[definition[1]],
                world[definition[2]],
            )
            if (
                angle_2d is not None
                and angle_3d is not None
                and abs(angle_2d - angle_3d)
                > self.max_2d_3d_angle_difference_deg
            ):
                angle_conflicts += 1
                for name in definition:
                    joint_reasons[name].add("two_d_three_d_angle_conflict")
        if angle_conflicts:
            frame_flags.add("two_d_three_d_conflict")
        self.previous_world = world
        self.previous_bones = bones
        self.flag_counts.update(frame_flags)
        return sorted(frame_flags), {
            name: sorted(values) for name, values in joint_reasons.items() if values
        }


__all__ = [
    "BODY_CANONICAL_VERSION",
    "COORDINATE_CONTRACT_VERSION",
    "CoordinateQualityTracker",
    "body_canonical_transform",
    "camera_ray_directions",
    "coordinate_layers",
    "estimated_intrinsics",
    "point_payload",
]
