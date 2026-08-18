from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.biomechanics.kinematics_3d import ThreeDKinematicsTracker, summarize_three_d_records
from src.biomechanics.local_ground_frame import build_local_ground_frame
from src.biomechanics.session_writer import SessionConfig, SessionWriter
from src.contracts import build_coordinate_output, load_contract_bundle
from src.realtime.session import build_pose_frame_from_result


def _point(name: str, xyz: tuple[float, float, float]) -> Keypoint:
    return Keypoint(
        name=name,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        confidence=0.98,
        visibility=0.98,
        presence=0.98,
        source_model="mediapipe-world",
    )


def _body(timestamp_ms: int = 0) -> PoseResult:
    coordinates = {
        "left_shoulder": (-0.25, 0.65, 0.00),
        "right_shoulder": (0.25, 0.65, 0.00),
        "left_elbow": (-0.28, 0.30, 0.00),
        "right_elbow": (0.28, 0.30, 0.00),
        "left_wrist": (-0.28, 0.00, 0.00),
        "right_wrist": (0.28, 0.00, 0.00),
        "left_hip": (-0.18, 0.00, 0.00),
        "right_hip": (0.18, 0.00, 0.00),
        "left_knee": (-0.18, -0.45, 0.00),
        "right_knee": (0.18, -0.45, 0.00),
        "left_ankle": (-0.18, -0.90, 0.00),
        "right_ankle": (0.18, -0.90, 0.00),
        "left_heel": (-0.18, -0.90, -0.12),
        "right_heel": (0.18, -0.90, -0.12),
        "left_foot_index": (-0.18, -0.90, 0.18),
        "right_foot_index": (0.18, -0.90, 0.18),
    }
    world = [_point(name, xyz) for name, xyz in coordinates.items()]
    image = [
        Keypoint(
            name=point.name,
            x=point.x + 0.5,
            y=0.9 - point.y * 0.5,
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


def _contact(value: float = 0.95) -> dict[str, object]:
    return {
        side: {"foot_contact_confidence": value}
        for side in ("left", "right")
    }


def test_local_ground_frame_is_orthonormal_and_explicitly_not_true_world() -> None:
    pose = _body()
    body_axes = {
        "reliable": True,
        "confidence": 0.95,
        "axes_world": {
            "x_left_to_right": (1.0, 0.0, 0.0),
            "y_up": (0.0, 1.0, 0.0),
            "z_forward": (0.0, 0.0, 1.0),
        },
    }
    result = build_local_ground_frame(
        pose.extra["world_keypoints"],
        body_coordinate_system=body_axes,
        ground_estimation={"status": "READY", "ground_confidence": 0.95},
        foot_contact_evidence=_contact(),
    )

    assert result["name"] == "local_ground_frame"
    assert result["available"] is True
    assert result["reliable"] is True
    assert result["true_world_coordinate"] is False
    assert result["camera_extrinsics_recovered"] is False
    axes = result["axes_world_relative"]
    matrix = np.asarray(
        [
            axes["x_body_left_to_right"],
            axes["y_estimated_vertical"],
            axes["z_forward"],
        ]
    )
    assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-7)
    assert axes["y_estimated_vertical"] == pytest.approx((0.0, 1.0, 0.0))
    support_y = {
        round(float(point["y"]), 8)
        for point in result["landmarks"]
        if point["name"] in result["support_landmarks"]
    }
    assert support_y == {0.0}


def test_local_ground_frame_refuses_single_foot_support() -> None:
    result = build_local_ground_frame(
        _body().extra["world_keypoints"],
        body_coordinate_system={
            "reliable": True,
            "confidence": 0.95,
            "axes_world": {
                "x_left_to_right": (1.0, 0.0, 0.0),
                "y_up": (0.0, 1.0, 0.0),
            },
        },
        ground_estimation={"status": "READY", "ground_confidence": 0.95},
        foot_contact_evidence={
            "left": {"foot_contact_confidence": 0.95},
            "right": {"foot_contact_confidence": 0.1},
        },
    )
    assert result["reliable"] is False
    assert "insufficient_contact_support" in result["quality_reasons"]


def test_tracker_emits_segment_coordinates_and_validation_only_biomech_metrics() -> None:
    tracker = ThreeDKinematicsTracker()
    result = None
    for timestamp in (0, 100, 200, 300, 400):
        result = tracker.update(
            _body(timestamp),
            capture_timestamp_ns=timestamp * 1_000_000,
            image_width=640,
            image_height=480,
            camera_view="side",
        )
    assert result is not None
    payload = result.as_dict()
    segments = payload["segment_coordinates"]
    metrics = payload["biomech_metrics"]

    assert segments["validation_only"] is True
    assert segments["segments"]["pelvis"]["reliable"] is True
    assert segments["segments"]["torso"]["reliable"] is True
    assert segments["segments"]["left_thigh"]["reliable"] is True
    knee = metrics["left_knee_flexion_proxy"]
    assert knee["legacy_angle"] == pytest.approx(180.0)
    assert knee["canonical_3d_angle"] == pytest.approx(180.0)
    assert knee["biomech_angle"] == pytest.approx(0.0, abs=1e-7)
    assert knee["observable"] is True
    assert knee["validation_only"] is True
    assert knee["formal_threshold_replacement_allowed"] is False
    assert metrics["trunk_flexion"]["biomech_angle"] == pytest.approx(0.0, abs=1e-7)
    # The established formal angle remains the legacy 2D value.
    assert payload["measurements"]["left_knee_angle"]["selected_angle"] == pytest.approx(180.0)


def test_coordinate_contract_exposes_named_auxiliary_ground_frame() -> None:
    tracker = ThreeDKinematicsTracker()
    payload = tracker.update(_body()).as_dict()
    coordinates = build_coordinate_output(
        load_contract_bundle().coordinate_spaces,
        three_d_kinematics=payload,
    )
    assert coordinates["analysis_space"] == "image_normalized_2d"
    assert coordinates["local_ground_frame"]["name"] == "local_ground_frame"
    assert coordinates["local_ground_frame_is_true_world"] is False


def test_session_csv_and_summary_include_biomech_validation_fields(tmp_path: Path) -> None:
    tracker = ThreeDKinematicsTracker()
    attached = None
    for timestamp in (0, 100, 200, 300):
        attached, _ = tracker.attach(
            _body(timestamp),
            capture_timestamp_ns=timestamp * 1_000_000,
        )
    assert attached is not None
    writer = SessionWriter(tmp_path)
    writer.start(
        SessionConfig(
            camera_index=0,
            width=640,
            height=480,
            mirror=False,
            smoothing=0.0,
            model_name="mediapipe",
            plot_on_save=False,
        )
    )
    pose_frame = build_pose_frame_from_result(
        attached,
        frame_index=1,
        mirror=False,
        frame_shape=(480, 640, 3),
        fps=30.0,
    )
    writer.pose_frames.append(pose_frame)
    session_dir = writer.stop()
    assert session_dir is not None
    with (session_dir / "kinematics_3d.csv").open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert "local_ground_confidence" in row
    assert "left_knee_flexion_proxy_biomech_angle" in row
    summary = summarize_three_d_records([pose_frame])
    assert "local_ground_available_ratio" in summary
    assert "left_knee_flexion_proxy" in summary["biomech_metrics"]
