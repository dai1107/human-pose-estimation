from __future__ import annotations

from math import cos, pi, sin

import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.biomechanics.body_coordinates import build_body_coordinate_system
from src.biomechanics.ground_estimation import GroundEstimator, GroundEstimatorConfig
from src.biomechanics.kinematics_3d import ThreeDKinematicsTracker
from src.contracts import build_coordinate_output, load_contract_bundle
from src.product_pose import ThreeDKinematicsConfig


def _point(name: str, xyz: tuple[float, float, float]) -> Keypoint:
    return Keypoint(
        name=name,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        confidence=0.95,
        visibility=0.95,
        presence=0.95,
        source_model="mediapipe-world",
    )


def _body_points(*, rotation_rad: float = 0.0) -> list[Keypoint]:
    coordinates = {
        "left_shoulder": (-1.0, 2.0, 0.0),
        "right_shoulder": (1.0, 2.0, 0.0),
        "left_hip": (-1.0, 0.0, 0.0),
        "right_hip": (1.0, 0.0, 0.0),
        "left_knee": (-1.0, -1.0, 0.0),
        "right_knee": (1.0, -1.0, 0.0),
        "left_ankle": (-1.0, -2.0, 0.0),
        "right_ankle": (1.0, -2.0, 0.0),
        "left_heel": (-1.1, -2.0, 0.0),
        "right_heel": (0.9, -2.0, 0.0),
        "left_foot_index": (-0.8, -2.0, 0.0),
        "right_foot_index": (1.2, -2.0, 0.0),
    }
    rotation = np.array(
        [
            [cos(rotation_rad), -sin(rotation_rad), 0.0],
            [sin(rotation_rad), cos(rotation_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return [
        _point(name, tuple(float(value) for value in rotation @ np.asarray(xyz)))
        for name, xyz in coordinates.items()
    ]


def _canonical_map(payload: dict[str, object]) -> dict[str, np.ndarray]:
    points = payload["canonical_landmarks"]
    assert isinstance(points, list)
    return {
        str(point["name"]): np.array(
            [point["x"], point["y"], point["z"]],
            dtype=float,
        )
        for point in points
    }


def test_body_coordinate_system_is_orthonormal_and_rotation_invariant() -> None:
    baseline = build_body_coordinate_system(_body_points()).as_dict()
    rotated = build_body_coordinate_system(
        _body_points(rotation_rad=pi / 3.0)
    ).as_dict()

    assert baseline["reliable"] is True
    assert rotated["reliable"] is True
    assert baseline["orthogonality_error"] == pytest.approx(0.0, abs=1e-8)
    baseline_points = _canonical_map(baseline)
    rotated_points = _canonical_map(rotated)
    for name in baseline_points:
        assert rotated_points[name] == pytest.approx(
            baseline_points[name],
            abs=1e-7,
        )
    axes = baseline["axes_world"]
    assert isinstance(axes, dict)
    assert axes["x_left_to_right"] == pytest.approx((1.0, 0.0, 0.0))
    assert axes["y_up"] == pytest.approx((0.0, 1.0, 0.0))
    assert axes["z_forward"] == pytest.approx((0.0, 0.0, 1.0))


def test_body_coordinate_system_rejects_degenerate_hip_axis() -> None:
    points = _body_points()
    by_name = {point.name: point for point in points}
    left_hip = by_name["left_hip"]
    by_name["right_hip"] = _point(
        "right_hip",
        (left_hip.x, left_hip.y, left_hip.z),
    )

    result = build_body_coordinate_system(list(by_name.values()))

    assert not result.available
    assert not result.reliable
    assert "body_left_right_axis_degenerate" in result.quality_reasons


def _pose_result(timestamp_ms: int = 0) -> PoseResult:
    world = _body_points()
    image = [
        Keypoint(
            name=point.name,
            x=point.x,
            y=point.y,
            z=point.z,
            confidence=point.confidence,
            visibility=point.visibility,
            presence=point.presence,
            source_model="mediapipe",
        )
        for point in world
    ]
    return PoseResult(
        keypoints=image,
        connections=(),
        model_name="mediapipe",
        num_keypoints=len(image),
        success=True,
        inference_time_ms=1.0,
        timestamp_ms=timestamp_ms,
        extra={"world_keypoints": world},
    )


def test_tracker_outputs_legacy_and_canonical_angles_without_rule_promotion() -> None:
    tracker = ThreeDKinematicsTracker(
        ThreeDKinematicsConfig(decision_mode="assist")
    )

    result = tracker.update(_pose_result(), camera_view="side")
    payload = result.as_dict()
    knee = payload["measurements"]["left_knee_angle"]

    assert payload["body_coordinate_system"]["reliable"] is True
    assert knee["legacy_angle"] == pytest.approx(180.0)
    assert knee["canonical_3d_angle"] == pytest.approx(180.0)
    assert payload["body_coordinate_system"][
        "formal_threshold_replacement_allowed"
    ] is False
    assert knee["selected_angle"] == knee["legacy_angle"]

    coordinate_output = build_coordinate_output(
        load_contract_bundle().coordinate_spaces,
        three_d_kinematics=payload,
    )
    assert "body_canonical_3d" in coordinate_output["coordinate_spaces"]
    assert coordinate_output["body_canonical_landmarks"]


def _foot_points(y: float, *, world: bool) -> list[Keypoint]:
    points: list[Keypoint] = []
    for side, x in (("left", 0.4), ("right", 0.6)):
        for suffix, delta_x in (("ankle", 0.0), ("heel", -0.02), ("foot_index", 0.03)):
            points.append(
                _point(
                    f"{side}_{suffix}",
                    (x + delta_x, y, 0.0),
                )
            )
    return points


def _foot_confidence(value: float) -> dict[str, object]:
    return {
        side: {
            "foot_contact_confidence": value,
            "observable": True,
        }
        for side in ("left", "right")
    }


def test_ground_estimator_uses_stable_history_and_fuses_contact_evidence() -> None:
    estimator = GroundEstimator(
        GroundEstimatorConfig(minimum_samples=3)
    )
    result = None
    for _ in range(2):
        result = estimator.update(
            _foot_points(0.90, world=False),
            _foot_points(0.50, world=True),
            _foot_confidence(0.90),
        )
    assert result is not None
    assert result["status"] == "READY"
    assert result["ground_y_image_normalized"] == pytest.approx(0.90)
    assert result["ground_confidence"] >= 0.80
    assert result["contact_evidence"]["left"]["status"] == "CONTACT_EVIDENCE"

    lifted = estimator.update(
        _foot_points(0.70, world=False),
        _foot_points(0.20, world=True),
        _foot_confidence(0.0),
    )
    assert lifted["ground_y_image_normalized"] == pytest.approx(0.90)
    assert lifted["contact_evidence"]["left"]["status"] == "NO_CONTACT_EVIDENCE"
    assert lifted["formal_floor_replacement_allowed"] is False
    assert lifted["formal_contact_replacement_allowed"] is False


def test_tracker_ground_layer_remains_auxiliary() -> None:
    tracker = ThreeDKinematicsTracker()
    result = None
    for timestamp_ms in (0, 100, 200, 300, 400):
        result = tracker.update(
            _pose_result(timestamp_ms),
            capture_timestamp_ns=timestamp_ms * 1_000_000,
        )
    assert result is not None

    ground = result.ground_estimation
    assert ground["evidence_only"] is True
    assert ground["formal_floor_replacement_allowed"] is False
    assert ground["formal_contact_replacement_allowed"] is False
    payload = result.as_dict()
    assert payload["ground_confidence"] == ground["ground_confidence"]
    assert payload["contact_evidence"] == ground["contact_evidence"]
    # The 3D evidence payload must not masquerade as the established 2D floor.
    assert "floor_reference_status" not in payload
