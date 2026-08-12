from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from hyrox.features import extract_basic_pose_features
from src.backends.base import Keypoint, PoseResult
from src.biomechanics.kinematics_3d import (
    ThreeDKinematicsTracker,
    calculate_angle_3d,
    summarize_three_d_records,
)
from src.biomechanics.session_writer import SessionConfig, SessionWriter
from src.biomechanics.velocity import KinematicsProcessor
from src.configuration import ConfigValidationError
from src.product_pose import ThreeDQualityConfig, load_product_pose_config
from src.realtime.session import POSE_NAME_TO_INDEX, build_pose_frame_from_result


def _point(
    name: str,
    xyz: tuple[float, float, float],
    *,
    visibility: float = 0.95,
    presence: float = 0.95,
    world: bool = False,
) -> Keypoint:
    return Keypoint(
        name=name,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        confidence=min(visibility, presence),
        source_model="mediapipe-world" if world else "mediapipe",
        visibility=visibility,
        presence=presence,
    )


def _knee_result(
    *,
    timestamp_ms: int = 0,
    hip: tuple[float, float, float] = (0.0, 1.0, 0.0),
    knee: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ankle: tuple[float, float, float] = (1.0, 0.0, 0.0),
    visibility: float = 0.95,
    presence: float = 0.95,
    include_world: bool = True,
) -> PoseResult:
    coordinates = {
        "left_hip": hip,
        "left_knee": knee,
        "left_ankle": ankle,
    }
    image = [
        _point(
            name,
            xyz,
            visibility=visibility,
            presence=presence,
        )
        for name, xyz in coordinates.items()
    ]
    world = [
        _point(
            name,
            xyz,
            visibility=visibility,
            presence=presence,
            world=True,
        )
        for name, xyz in coordinates.items()
    ]
    return PoseResult(
        keypoints=image,
        connections=(),
        model_name="mediapipe",
        num_keypoints=len(image),
        success=True,
        inference_time_ms=1.0,
        timestamp_ms=timestamp_ms,
        extra={
            "world_keypoints": world if include_world else [],
            "world_landmarks_available": include_world,
        },
    )


def test_calculate_angle_3d_handles_standard_zero_and_non_finite_vectors() -> None:
    assert calculate_angle_3d(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ) == pytest.approx(90.0)
    assert calculate_angle_3d(
        np.zeros(3),
        np.zeros(3),
        np.ones(3),
    ) is None
    assert calculate_angle_3d(
        np.array([np.nan, 0.0, 0.0]),
        np.zeros(3),
        np.ones(3),
    ) is None


def test_shadow_mode_calculates_3d_but_always_selects_2d() -> None:
    tracker = ThreeDKinematicsTracker()
    original = _knee_result()
    original_keypoints = list(original.keypoints)

    attached, shadow = tracker.attach(original, capture_timestamp_ns=1_000_000)
    measurement = shadow.measurements["left_knee_angle"]

    assert measurement.angle_2d == pytest.approx(90.0)
    assert measurement.angle_3d == pytest.approx(90.0)
    assert measurement.selected_angle == measurement.angle_2d
    assert measurement.selected_source == "2d_shadow"
    assert measurement.three_d_reliable
    assert attached.keypoints == original_keypoints
    assert attached.extra["three_d_kinematics"]["decision_mode"] == "shadow"


def test_shadow_2d_angle_uses_frame_aspect_ratio() -> None:
    result = _knee_result(
        hip=(0.5, 0.359375, 0.0),
        knee=(0.5, 0.5, 0.0),
        ankle=(0.7165064, 0.4296875, 0.0),
    )

    uncorrected = ThreeDKinematicsTracker().update(result)
    corrected = ThreeDKinematicsTracker().update(
        result,
        image_width=720,
        image_height=1280,
    )

    assert corrected.measurements["left_knee_angle"].angle_2d == pytest.approx(
        60.0,
        abs=0.1,
    )
    assert abs(
        float(uncorrected.measurements["left_knee_angle"].angle_2d) - 60.0
    ) > 5.0


def test_missing_world_landmarks_fall_back_to_2d_without_error() -> None:
    _, shadow = ThreeDKinematicsTracker().attach(
        _knee_result(include_world=False),
        capture_timestamp_ns=1_000_000,
    )
    measurement = shadow.measurements["left_knee_angle"]

    assert measurement.angle_2d == pytest.approx(90.0)
    assert measurement.angle_3d is None
    assert measurement.selected_angle == measurement.angle_2d
    assert not measurement.three_d_reliable
    assert "world_landmarks_missing" in measurement.quality_reasons


@pytest.mark.parametrize(
    ("visibility", "presence", "reason"),
    (
        (0.69, 0.95, "low_visibility"),
        (0.95, 0.69, "low_presence"),
    ),
)
def test_low_image_landmark_quality_rejects_3d(
    visibility: float,
    presence: float,
    reason: str,
) -> None:
    shadow = ThreeDKinematicsTracker().update(
        _knee_result(visibility=visibility, presence=presence),
        capture_timestamp_ns=1_000_000,
    )

    measurement = shadow.measurements["left_knee_angle"]
    assert not measurement.three_d_reliable
    assert reason in measurement.quality_reasons


def test_bone_length_and_z_jumps_reject_3d_without_poisoning_2d() -> None:
    tracker = ThreeDKinematicsTracker(
        quality_config=ThreeDQualityConfig(max_2d_3d_difference_deg=180.0)
    )
    tracker.update(_knee_result(timestamp_ms=0), capture_timestamp_ns=0)
    changed = tracker.update(
        _knee_result(timestamp_ms=100, ankle=(2.0, 0.0, 1.0)),
        capture_timestamp_ns=100_000_000,
    )
    measurement = changed.measurements["left_knee_angle"]

    assert measurement.angle_2d is not None
    assert not measurement.three_d_reliable
    assert "bone_length_jump" in measurement.quality_reasons
    assert "z_jump" in measurement.quality_reasons
    assert measurement.selected_source == "2d_shadow"


def test_angle_delta_and_angular_velocity_are_gated() -> None:
    tracker = ThreeDKinematicsTracker(
        quality_config=ThreeDQualityConfig(max_2d_3d_difference_deg=180.0)
    )
    tracker.update(_knee_result(timestamp_ms=0), capture_timestamp_ns=0)
    changed = tracker.update(
        _knee_result(
            timestamp_ms=100,
            ankle=(0.0, -1.0, 0.0),
        ),
        capture_timestamp_ns=100_000_000,
    )
    reasons = changed.measurements["left_knee_angle"].quality_reasons

    assert "angle_jump" in reasons
    assert "angular_velocity_exceeded" in reasons


def test_gap_and_pose_age_reject_current_3d_observation() -> None:
    tracker = ThreeDKinematicsTracker(max_pose_age_ms=150.0)
    tracker.update(_knee_result(timestamp_ms=0), capture_timestamp_ns=0)
    gap = tracker.update(
        _knee_result(timestamp_ms=300),
        capture_timestamp_ns=300_000_000,
    )
    old = tracker.update(
        _knee_result(timestamp_ms=333),
        capture_timestamp_ns=333_000_000,
        pose_age_ms=151.0,
    )

    assert "world_gap_exceeded" in gap.measurements["left_knee_angle"].quality_reasons
    assert "pose_too_old" in old.measurements["left_knee_angle"].quality_reasons


def test_left_right_identity_swap_is_rejected() -> None:
    def identity_result(timestamp_ms: int, swapped: bool) -> PoseResult:
        coordinates = {
            "left_shoulder": (-1.0, 1.0, 0.0),
            "right_shoulder": (1.0, 1.0, 0.0),
            "left_hip": (-1.0, 0.0, 0.0),
            "right_hip": (1.0, 0.0, 0.0),
        }
        image = [_point(name, xyz) for name, xyz in coordinates.items()]
        if swapped:
            coordinates = {
                name: coordinates[name.replace("left_", "right_")]
                if name.startswith("left_")
                else coordinates[name.replace("right_", "left_")]
                for name in coordinates
            }
        world = [_point(name, xyz, world=True) for name, xyz in coordinates.items()]
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

    tracker = ThreeDKinematicsTracker()
    tracker.update(identity_result(0, False), capture_timestamp_ns=0)
    swapped = tracker.update(identity_result(100, True), capture_timestamp_ns=100_000_000)

    assert "left_right_identity_swap" in swapped.quality_reasons
    assert not swapped.three_d_reliable


def test_pose_frame_preserves_world_landmarks_and_shadow_payload() -> None:
    attached, _ = ThreeDKinematicsTracker().attach(
        _knee_result(timestamp_ms=10),
        capture_timestamp_ns=10_000_000,
    )
    pose_frame = build_pose_frame_from_result(
        attached,
        frame_index=1,
        mirror=False,
        frame_shape=(480, 640, 3),
        fps=30.0,
    )

    knee = pose_frame.world_landmarks[POSE_NAME_TO_INDEX["left_knee"]]
    assert knee.x == pytest.approx(0.0)
    assert knee.visibility == pytest.approx(0.95)
    assert pose_frame.three_d_kinematics["decision_mode"] == "shadow"


def test_session_report_persists_3d_angles_differences_and_availability(
    tmp_path: Path,
) -> None:
    attached, _ = ThreeDKinematicsTracker().attach(
        _knee_result(timestamp_ms=10),
        capture_timestamp_ns=10_000_000,
    )
    pose_frame = build_pose_frame_from_result(
        attached,
        frame_index=1,
        mirror=False,
        frame_shape=(480, 640, 3),
        fps=30.0,
    )
    writer = SessionWriter(tmp_path)
    writer.start(
        SessionConfig(
            camera_index=0,
            width=640,
            height=480,
            mirror=False,
            smoothing=0.0,
            model_name="pose_landmarker_full.task",
            plot_on_save=False,
        ),
        session_id="round7_shadow",
    )
    writer.add_frame(pose_frame, KinematicsProcessor().process(pose_frame))
    session_dir = writer.stop()
    assert session_dir is not None

    with (session_dir / "kinematics_3d.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        row = next(csv.DictReader(file))
    summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))

    assert float(row["left_knee_angle_2d"]) == pytest.approx(90.0)
    assert float(row["left_knee_angle_3d"]) == pytest.approx(90.0)
    assert row["left_knee_angle_3d_reliable"] == "1"
    assert summary["three_d_kinematics"]["world_landmarks_availability_ratio"] == 1.0


def test_shadow_payload_does_not_change_hyrox_feature_stream() -> None:
    result = _knee_result()
    before = extract_basic_pose_features(result.keypoints, 640, 480)
    attached, _ = ThreeDKinematicsTracker().attach(
        result,
        capture_timestamp_ns=1_000_000,
    )
    after = extract_basic_pose_features(attached.keypoints, 640, 480)

    assert attached.keypoints == result.keypoints
    assert after == before


def test_summary_groups_availability_differences_and_failure_reasons() -> None:
    records = [
        {
            "action": "lunge",
            "camera_view": "side",
            "phase": "bottom",
            "three_d_kinematics": {
                "three_d_available": True,
                "three_d_reliable": False,
                "three_d_reliable_ratio": 0.5,
                "angle_differences_deg": {"left_knee_angle_2d_3d_difference_deg": 10.0},
                "quality_reasons": ["low_visibility"],
            },
        },
        {
            "action": "lunge",
            "camera_view": "side",
            "phase": "stand",
            "three_d_kinematics": {
                "three_d_available": True,
                "three_d_reliable": True,
                "three_d_reliable_ratio": 1.0,
                "angle_differences_deg": {"left_knee_angle_2d_3d_difference_deg": 20.0},
                "quality_reasons": [],
            },
        },
    ]

    summary = summarize_three_d_records(records)

    assert summary["world_landmarks_availability_ratio"] == 1.0
    assert summary["mean_reliable_angle_ratio"] == pytest.approx(0.75)
    assert summary["failure_reasons"]["low_visibility"] == 1
    assert summary["angle_difference_deg"]["left_knee_angle_2d_3d_difference_deg"]["p50"] == 15.0
    assert summary["by_action"]["lunge"]["frame_count"] == 2
    assert summary["by_camera_view"]["side"]["frame_count"] == 2


def test_round7_shadow_remains_supported_after_assist_promotion(tmp_path: Path) -> None:
    config = load_product_pose_config(Path("configs/product_pose.yaml"))
    assert config.three_d_kinematics.enabled
    assert config.three_d_kinematics.decision_mode == "assist"
    assert config.three_d_quality.min_visibility == pytest.approx(0.70)

    shadow = tmp_path / "shadow.yaml"
    shadow.write_text(
        "product_pose:\n"
        "  backend: mediapipe\n"
        "  allow_experimental_backends: false\n"
        "three_d_kinematics:\n"
        "  enabled: true\n"
        "  decision_mode: shadow\n",
        encoding="utf-8",
    )
    assert load_product_pose_config(shadow).three_d_kinematics.decision_mode == "shadow"

    invalid = tmp_path / "rule_specific.yaml"
    invalid.write_text(
        shadow.read_text(encoding="utf-8").replace("shadow", "rule-specific"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="rule-specific"):
        load_product_pose_config(invalid)


def test_joint_metric_exposes_raw_smooth_and_reliable_3d_selection() -> None:
    tracker = ThreeDKinematicsTracker(
        kinematics_config=load_product_pose_config(
            Path("configs/product_pose.yaml")
        ).three_d_kinematics
    )
    raw = _knee_result(
        hip=(0.0, 1.0, 0.0),
        knee=(0.0, 0.0, 0.0),
        ankle=(0.5, -0.8660254, 0.0),
    )
    smooth = _knee_result()

    metric = tracker.update(
        smooth,
        raw_result=raw,
        capture_timestamp_ns=1_000_000,
        camera_view="side",
    ).measurements["left_knee_angle"]

    assert metric.raw_2d == pytest.approx(150.0, abs=0.1)
    assert metric.smooth_2d == pytest.approx(90.0)
    assert metric.raw_3d == pytest.approx(150.0, abs=0.1)
    assert metric.smooth_3d == pytest.approx(90.0)
    assert metric.selected_value == pytest.approx(90.0)
    assert metric.source == "3D"
    assert metric.observable
    # Formal HYROX thresholds remain on their established 2D stream.
    assert metric.selected_angle == metric.smooth_2d
    assert metric.selected_source == "2d_assist"


def test_front_view_without_reliable_world_angle_is_unobservable() -> None:
    tracker = ThreeDKinematicsTracker(
        kinematics_config=load_product_pose_config(
            Path("configs/product_pose.yaml")
        ).three_d_kinematics
    )

    metric = tracker.update(
        _knee_result(include_world=False),
        camera_view="front",
    ).measurements["left_knee_angle"]

    assert metric.smooth_2d == pytest.approx(90.0)
    assert metric.selected_value is None
    assert metric.source == "UNAVAILABLE"
    assert not metric.observable
    assert "camera_view_limited" in metric.quality_reasons


def test_bone_length_uses_historical_median_and_does_not_accept_spike() -> None:
    tracker = ThreeDKinematicsTracker(
        quality_config=ThreeDQualityConfig(
            max_2d_3d_difference_deg=180.0,
            max_landmark_speed_m_s=100.0,
        )
    )
    for timestamp_ms in (0, 100, 200):
        tracker.update(
            _knee_result(timestamp_ms=timestamp_ms),
            capture_timestamp_ns=timestamp_ms * 1_000_000,
        )

    changed = tracker.update(
        _knee_result(timestamp_ms=300, ankle=(2.0, 0.0, 0.0)),
        capture_timestamp_ns=300_000_000,
    )
    metric = changed.measurements["left_knee_angle"]

    assert "bone_length_jump" in metric.quality_reasons
    medians = changed.reliability["bone_length_historical_median_m"]
    assert medians["left_ankle__left_knee"] == pytest.approx(1.0)
    assert changed.reliability["position_correction_applied"] is False


def _bilateral_leg_result(timestamp_ms: int = 0) -> PoseResult:
    coordinates = {
        "left_hip": (-0.2, 1.0, 0.0),
        "left_knee": (-0.2, 0.5, 0.0),
        "left_ankle": (-0.2, 0.0, 0.0),
        "left_heel": (-0.25, 0.02, 0.0),
        "left_foot_index": (-0.1, 0.0, 0.0),
        "right_hip": (0.2, 1.0, 0.0),
        "right_knee": (0.2, 0.5, 0.0),
        "right_ankle": (0.2, -0.5, 0.0),
        "right_heel": (0.15, -0.48, 0.0),
        "right_foot_index": (0.3, -0.5, 0.0),
    }
    return PoseResult(
        keypoints=[_point(name, xyz) for name, xyz in coordinates.items()],
        connections=(),
        model_name="mediapipe",
        num_keypoints=len(coordinates),
        success=True,
        inference_time_ms=1.0,
        timestamp_ms=timestamp_ms,
        extra={
            "world_keypoints": [
                _point(name, xyz, world=True) for name, xyz in coordinates.items()
            ]
        },
    )


def test_left_right_bone_mismatch_reduces_3d_reliability() -> None:
    result = ThreeDKinematicsTracker().update(_bilateral_leg_result())

    left = result.measurements["left_knee_angle"]
    right = result.measurements["right_knee_angle"]
    assert "left_right_bone_mismatch" in left.quality_reasons
    assert "left_right_bone_mismatch" in right.quality_reasons
    assert result.reliability["left_right_mismatch_segments"]


def test_isolated_landmark_velocity_is_rejected() -> None:
    tracker = ThreeDKinematicsTracker(
        quality_config=ThreeDQualityConfig(
            max_2d_3d_difference_deg=180.0,
            max_bone_length_change_ratio=10.0,
            max_z_change_body_scale=10.0,
        )
    )
    tracker.update(_knee_result(timestamp_ms=0), capture_timestamp_ns=0)
    changed = tracker.update(
        _knee_result(timestamp_ms=100, ankle=(1.0, 0.0, 1.0)),
        capture_timestamp_ns=100_000_000,
    )

    assert "left_ankle" in changed.reliability["isolated_velocity_joints"]
    assert (
        "isolated_landmark_velocity"
        in changed.measurements["left_knee_angle"].quality_reasons
    )


def test_stable_foot_builds_contact_confidence_without_promoting_rules() -> None:
    tracker = ThreeDKinematicsTracker()
    result = None
    for timestamp_ms in (0, 100, 200, 300):
        result = tracker.update(
            _bilateral_leg_result(timestamp_ms),
            capture_timestamp_ns=timestamp_ms * 1_000_000,
        )
    assert result is not None

    left = result.foot_contact_evidence["left"]
    assert left["stable_frames"] >= 3
    assert left["foot_contact_confidence"] >= 0.60
    assert left["likely_contact"] is True
    assert result.foot_contact_evidence["evidence_only"] is True
    assert (
        result.foot_contact_evidence["formal_rule_replacement_allowed"] is False
    )
