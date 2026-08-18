from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.biomechanics.kinematics_3d import ThreeDKinematicsTracker
from src.product_pose import ThreeDQualityConfig
from src.realtime.camera_motion import CameraMotionEstimator
from src.realtime.temporal_pose import TemporalPoseBuffer


def _point(name: str, x: float, y: float, z: float = 0.0) -> Keypoint:
    return Keypoint(
        name=name,
        x=x,
        y=y,
        z=z,
        confidence=0.98,
        visibility=0.98,
        presence=0.98,
        source_model="mediapipe",
    )


def _pose(timestamp_ms: int, *, left_ankle_x: float = -1.0) -> PoseResult:
    coordinates = {
        "left_shoulder": (-1.0, 2.0, 0.0),
        "right_shoulder": (1.0, 2.0, 0.0),
        "left_hip": (-1.0, 0.0, 0.0),
        "right_hip": (1.0, 0.0, 0.0),
        "left_knee": (-1.0, -1.0, 0.0),
        "right_knee": (1.0, -1.0, 0.0),
        "left_ankle": (left_ankle_x, -2.0, 0.0),
        "right_ankle": (1.0, -2.0, 0.0),
        "left_heel": (left_ankle_x - 0.1, -2.0, 0.0),
        "right_heel": (0.9, -2.0, 0.0),
        "left_foot_index": (left_ankle_x + 0.2, -2.0, 0.0),
        "right_foot_index": (1.2, -2.0, 0.0),
    }
    world = [_point(name, *values) for name, values in coordinates.items()]
    image = [replace(item, source_model="mediapipe") for item in world]
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


def test_temporal_pose_buffer_is_causal_bounded_and_tracks_trends() -> None:
    buffer = TemporalPoseBuffer()
    snapshots = []
    for timestamp, angle, phase in (
        (0, 170.0, "stand"),
        (100, 150.0, "descent"),
        (300, 110.0, "descent"),
        (700, 90.0, "bottom"),
    ):
        snapshots.append(
            buffer.update(
                _pose(timestamp).keypoints,
                timestamp_ms=timestamp,
                features={"left_knee_angle": angle},
                three_d_kinematics={
                    "three_d_reliable_ratio": 0.8,
                    "foot_contact_evidence": {
                        "left": {"foot_contact_confidence": 0.9},
                        "right": {"foot_contact_confidence": 0.2},
                    },
                },
                phase=phase,
            )
        )

    current = snapshots[-1]
    assert current["causal_only"] is True
    assert current["future_frames_used"] is False
    assert current["formal_rule_replacement_allowed"] is False
    assert current["window_limit_ms"] == pytest.approx(500.0)
    assert current["window_duration_ms"] <= 500.0
    assert current["sample_count"] == 2
    assert current["angle_trends"]["left_knee_angle"] == "decreasing"
    assert current["phase"] == "bottom"


def test_contact_drift_degrades_auxiliary_3d_without_mutating_landmarks() -> None:
    tracker = ThreeDKinematicsTracker(
        quality_config=ThreeDQualityConfig(
            foot_stationary_speed_m_s=10.0,
            foot_contact_stable_frames=2,
            contact_constraint_drift_body_ratio=0.05,
        )
    )
    for timestamp in (0, 100, 200):
        tracker.update(
            _pose(timestamp),
            capture_timestamp_ns=timestamp * 1_000_000,
        )
    drifted = _pose(300, left_ankle_x=-0.65)
    original_world = list(drifted.extra["world_keypoints"])
    result = tracker.update(
        drifted,
        capture_timestamp_ns=300_000_000,
    )
    payload = result.as_dict()

    assert "left" in payload["constrained_3d"]["drifting_contact_sides"]
    assert payload["constrained_3d"]["constraint_applied"] is False
    assert payload["constrained_3d"]["raw_landmarks_modified"] is False
    assert payload["constrained_3d"]["canonical_landmarks_modified"] is False
    assert "contact_foot_drift" in payload["measurements"]["left_knee_angle"]["quality_reasons"]
    assert drifted.extra["world_keypoints"] == original_world
    assert drifted.keypoints[-1] == _pose(300, left_ankle_x=-0.65).keypoints[-1]


def _feature_frame(dx: int = 0, dy: int = 0) -> np.ndarray:
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    for y in range(15, 170, 25):
        for x in range(15, 230, 25):
            cv2.circle(frame, (x + dx, y + dy), 2, (255, 255, 255), -1)
    return frame


def test_camera_motion_estimator_classifies_global_motion() -> None:
    estimator = CameraMotionEstimator()
    warmup = estimator.update(_feature_frame(), timestamp_ms=0)
    static = estimator.update(_feature_frame(), timestamp_ms=250)
    moved = estimator.update(_feature_frame(dx=8, dy=4), timestamp_ms=500)

    assert warmup["available"] is False
    assert static["state"] == "camera_static"
    assert moved["available"] is True
    assert moved["state"] in {"camera_small_motion", "camera_unstable"}
    assert moved["camera_motion_score"] > static["camera_motion_score"]
    assert moved["modifies_body_3d"] is False


def test_camera_motion_reduces_global_ground_and_contact_reliability_only() -> None:
    tracker = ThreeDKinematicsTracker()
    result = tracker.update(
        _pose(0),
        camera_motion={
            "available": True,
            "camera_motion_score": 0.8,
            "state": "camera_unstable",
        },
    ).as_dict()

    reliability = result["reliability"]
    assert reliability["camera_unstable"] is True
    assert reliability["global_position_reliability"] == pytest.approx(0.35)
    assert result["camera_motion"]["modifies_body_3d"] is False
    assert result["measurements"]["left_knee_angle"]["legacy_angle"] == pytest.approx(180.0)
