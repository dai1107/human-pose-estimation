from __future__ import annotations

import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.utils.keypoint_schema import MEDIAPIPE_CONNECTIONS
from tools.dataset.round8_coordinates import (
    body_canonical_transform,
    camera_ray_directions,
    coordinate_layers,
    estimated_intrinsics,
)
from tools.dataset.round8_pose_cache import (
    binary_mask_rle,
    select_target_pose_candidate,
)
from tools.dataset.round8_temporal import predict_display_pose
from tools.run_round8_pose_cache import build_parser


def _point(name: str, x: float, y: float, z: float = 0.0) -> Keypoint:
    return Keypoint(
        name=name,
        x=x,
        y=y,
        z=z,
        confidence=0.95,
        visibility=0.95,
        presence=0.95,
        source_model="mediapipe",
    )


def test_body_canonical_transform_is_invertible_and_hip_centered() -> None:
    world = [
        _point("left_hip", -0.2, 0.0, 0.0),
        _point("right_hip", 0.2, 0.0, 0.0),
        _point("left_shoulder", -0.2, 0.6, 0.0),
        _point("right_shoulder", 0.2, 0.6, 0.0),
    ]

    canonical, transform = body_canonical_transform(world)
    by_name = {point["name"]: point for point in canonical}

    assert transform["status"] == "available"
    assert transform["roundtrip_matrix_max_abs_error"] <= 1e-12
    assert by_name["left_hip"]["x"] == -by_name["right_hip"]["x"]
    assert by_name["left_hip"]["y"] == by_name["right_hip"]["y"] == 0.0


def test_estimated_intrinsics_only_produce_unit_rays_without_depth() -> None:
    intrinsics = estimated_intrinsics(720, 1280)
    rays = camera_ray_directions(
        [_point("nose", 0.5, 0.5)],
        width=720,
        height=1280,
        intrinsics=intrinsics,
    )
    ray = rays[0]

    assert intrinsics["status"] == "estimated_intrinsics"
    assert intrinsics["absolute_depth_available"] is False
    assert np.linalg.norm([ray["x"], ray["y"], ray["z"]]) == pytest.approx(
        1.0
    )
    assert ray["absolute_depth"] is None


def test_coordinate_layers_keep_monocular_world_and_oni_metric_distinct() -> None:
    points = [
        _point("left_hip", 0.4, 0.6, -0.1),
        _point("right_hip", 0.6, 0.6, -0.1),
        _point("left_shoulder", 0.4, 0.3, -0.1),
        _point("right_shoulder", 0.6, 0.3, -0.1),
    ]
    layers = coordinate_layers(
        points,
        points,
        width=720,
        height=1280,
        intrinsics=estimated_intrinsics(720, 1280),
    )

    assert layers["mp_world_body_3d"]["is_absolute_metric_ground_truth"] is False
    assert layers["camera_ray_direction_3d"]["absolute_metric_accuracy"] == "not_applicable"
    assert layers["oni_surface_metric_3d"]["status"] == "unavailable_phone_rgb_round8"
    assert layers["coordinate_mixing_policy"]["body_canonical_must_not_mix_with_scene_geometry"]


def test_target_bbox_selects_second_pose_candidate_and_matching_mask() -> None:
    first = [_point("left_hip", 0.05, 0.05), _point("right_hip", 0.15, 0.15)]
    second = [_point("left_hip", 0.55, 0.55), _point("right_hip", 0.75, 0.80)]
    result = PoseResult(
        keypoints=first,
        connections=MEDIAPIPE_CONNECTIONS,
        model_name="mediapipe",
        num_keypoints=len(first),
        success=True,
        inference_time_ms=1.0,
        bbox=(0.05, 0.05, 0.15, 0.15),
        timestamp_ms=0,
        extra={
            "pose_candidates": [first, second],
            "world_pose_candidates": [first, second],
            "segmentation_masks": [
                np.zeros((4, 4), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32),
            ],
        },
    )

    selected, binding, mask = select_target_pose_candidate(
        result,
        target_bbox_pixels=(50, 50, 90, 95),
        width=100,
        height=100,
    )

    assert binding["selected_candidate_index"] == 1
    assert binding["target_binding_passed"] is True
    assert selected.keypoints == second
    assert mask is not None and np.all(mask == 1)


def test_binary_mask_rle_roundtrips_row_major_mask() -> None:
    mask = np.asarray([[0, 0, 1], [1, 0, 0]], dtype=np.float32)
    payload = binary_mask_rle(mask)
    values = []
    state = 0
    for count in payload["counts"]:
        values.extend([state] * count)
        state = 1 - state

    assert np.asarray(values, dtype=np.uint8).reshape(2, 3).tolist() == mask.astype(
        np.uint8
    ).tolist()


def test_display_prediction_is_bounded_and_support_foot_does_not_drift() -> None:
    first = PoseResult(
        keypoints=[
            _point("left_ankle", 0.40, 0.80),
            _point("left_hip", 0.45, 0.60),
            _point("right_hip", 0.55, 0.60),
            _point("left_shoulder", 0.45, 0.30),
            _point("right_shoulder", 0.55, 0.30),
        ],
        connections=MEDIAPIPE_CONNECTIONS,
        model_name="mediapipe_full",
        num_keypoints=5,
        success=True,
        inference_time_ms=1.0,
        timestamp_ms=0,
    )
    second = PoseResult(
        keypoints=[
            _point("left_ankle", 0.45, 0.78),
            _point("left_hip", 0.46, 0.59),
            _point("right_hip", 0.56, 0.59),
            _point("left_shoulder", 0.46, 0.29),
            _point("right_shoulder", 0.56, 0.29),
        ],
        connections=MEDIAPIPE_CONNECTIONS,
        model_name="mediapipe_full",
        num_keypoints=5,
        success=True,
        inference_time_ms=1.0,
        timestamp_ms=33,
    )

    predicted = predict_display_pose([first, second], fps=30.0, horizon_ms=45.0)
    ankle = {point.name: point for point in predicted[1].keypoints}["left_ankle"]

    assert ankle.x == second.keypoints[0].x
    assert predicted[1].extra["display_only"] is True
    assert predicted[1].extra["prediction_horizon_ms"] == 45.0


def test_round8_cli_has_explicit_anchor_review_gate() -> None:
    args = build_parser().parse_args(
        [
            "--derive-only",
            "--approve-anchor-review",
            "--reviewer-id",
            "reviewer-1",
        ]
    )

    assert args.derive_only is True
    assert args.approve_anchor_review is True
    assert args.reviewer_id == "reviewer-1"
