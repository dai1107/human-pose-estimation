from __future__ import annotations

import pytest

pytest.skip("WHAM/OpenCap integration is deferred", allow_module_level=True)

from src.offline3d.alignment import align_mediapipe_wham
from src.offline3d.base import Offline3DFrame, Offline3DResult
from src.offline3d.reports import build_advanced_report
from src.offline3d.wham.hyrox_fusion import apply_wham_hyrox_assist


def _joints(knee_x: float = -0.2) -> dict[str, tuple[float, float, float]]:
    return {
        "left_shoulder": (-0.2, -0.8, 5.0),
        "right_shoulder": (0.2, -0.8, 5.0),
        "left_elbow": (-0.35, -0.2, 5.0),
        "right_elbow": (0.35, -0.2, 5.0),
        "left_wrist": (-0.4, 0.4, 5.0),
        "right_wrist": (0.4, 0.4, 5.0),
        "left_hip": (-0.2, 0.0, 5.0),
        "right_hip": (0.2, 0.0, 5.0),
        "left_knee": (knee_x, 0.8, 5.0),
        "right_knee": (0.2, 0.8, 5.0),
        "left_ankle": (-0.2, 1.6, 5.0),
        "right_ankle": (0.2, 1.6, 5.0),
        "left_foot_index": (-0.2, 1.7, 5.2),
        "right_foot_index": (0.2, 1.7, 5.2),
    }


def _wham(joints: dict[str, tuple[float, float, float]] | None = None) -> Offline3DResult:
    return Offline3DResult(
        backend="wham",
        status="COMPLETED",
        reference_source="Official WHAM reconstructed 3D",
        frames=[
            Offline3DFrame(
                timestamp_ms=0.0,
                frame_index=0,
                joints_3d=joints or _joints(),
                confidence=0.92,
                global_trajectory=[0.0, 0.0, 0.0],
            )
        ],
        coordinate_system="wham_camera_joints_with_world_root_trajectory",
        metadata={
            "source_width": 1920,
            "source_height": 1080,
            "focal_length_px": 2202.9,
        },
    )


def _frame() -> dict[str, object]:
    return {
        "frame_index": 0,
        "timestamp_ms": 0.0,
        "reps": 3,
        "last_rep_decision": {"status": "VALID"},
        "three_d_kinematics": {
            "canonical_3d_angles": {
                "left_knee_angle": 178.0,
                "right_knee_angle": 178.0,
            },
            "body_coordinate_system": {"canonical_landmarks": []},
        },
        "keypoints": [],
    }


def test_wham_fusion_enriches_skeleton_without_replacing_hyrox_result() -> None:
    frame = _frame()
    wham = _wham()
    alignment = align_mediapipe_wham([frame], wham)

    fusion = apply_wham_hyrox_assist(
        [frame], alignment=alignment, wham_result=wham, action="lunge"
    )

    assist = frame["wham_assist"]
    assert fusion.status == "COMPLETED"
    assert assist["reliable"] is True
    assert assist["status"] == "CONFIRMED"
    assert assist["projected_keypoints"]
    assert assist["formal_rule_replacement_allowed"] is False
    assert frame["reps"] == 3
    assert frame["last_rep_decision"] == {"status": "VALID"}


def test_wham_fusion_marks_large_disagreement_as_conflict() -> None:
    frame = _frame()
    frame["three_d_kinematics"]["canonical_3d_angles"]["left_knee_angle"] = 60.0
    wham = _wham()
    alignment = align_mediapipe_wham([frame], wham)

    fusion = apply_wham_hyrox_assist(
        [frame], alignment=alignment, wham_result=wham, action="lunge"
    )

    assert frame["wham_assist"]["status"] == "CONFLICT"
    assert fusion.conflict_frame_count == 1


def test_advanced_report_exposes_wham_hyrox_assistance() -> None:
    frame = _frame()
    wham = _wham()
    alignment = align_mediapipe_wham([frame], wham)
    fusion = apply_wham_hyrox_assist(
        [frame], alignment=alignment, wham_result=wham, action="lunge"
    )

    report = build_advanced_report(
        [frame], wham=wham, alignment=alignment, opencap=None, hyrox_assist=fusion
    )

    assert report["hyrox_assist"]["status"] == "COMPLETED"
    assert report["hyrox_assist"]["skeleton_overlay_enabled"] is True
    assert report["hyrox_assist"]["formal_rule_replacement_allowed"] is False
    assert report["opencap"]["status"] == "NOT_REQUESTED"
