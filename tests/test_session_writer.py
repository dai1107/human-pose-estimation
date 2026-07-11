from __future__ import annotations

import csv
import json
from dataclasses import replace

from src.biomechanics.hand_landmarks import SUPPLEMENTAL_FINGER_JOINTS, empty_hand_landmarks
from src.biomechanics.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, empty_landmarks
from src.biomechanics.normalization import normalize_landmarks
from src.biomechanics.session_writer import SessionConfig, SessionWriter
from src.biomechanics.types import KinematicFrame, LandmarkPoint, PoseFrame
from src.biomechanics.velocity import KinematicsProcessor


def _landmarks(offset: float = 0.0) -> list[LandmarkPoint]:
    points = empty_landmarks()
    values = {
        "left_hip": (-0.5 + offset, 0.0, 0.0),
        "right_hip": (0.5 + offset, 0.0, 0.0),
        "left_shoulder": (-0.5 + offset, -1.0, 0.0),
        "right_shoulder": (0.5 + offset, -1.0, 0.0),
        "left_elbow": (-1.0 + offset, -1.0, 0.0),
        "right_elbow": (1.0 + offset, -1.0, 0.0),
        "left_wrist": (-1.5 + offset, -1.0, 0.0),
        "right_wrist": (1.5 + offset, -1.0, 0.0),
        "left_knee": (-0.5 + offset, 1.0, 0.0),
        "right_knee": (0.5 + offset, 1.0, 0.0),
        "left_ankle": (-0.5 + offset, 2.0, 0.0),
        "right_ankle": (0.5 + offset, 2.0, 0.0),
    }
    for name, coords in values.items():
        points[LANDMARK_INDEX[name]] = LandmarkPoint(*coords)
    return points


def _pose_frame(index: int, timestamp_ms: int, points: list[LandmarkPoint]) -> PoseFrame:
    normalization = normalize_landmarks(points)
    return PoseFrame(
        frame_index=index,
        timestamp_ms=timestamp_ms,
        pose_detected=True,
        image_landmarks=points,
        smoothed_landmarks=points,
        normalized_landmarks=normalization.landmarks,
        normalization_success=normalization.success,
        normalization_message=normalization.message,
        mirror=True,
        camera_width=1280,
        camera_height=720,
        fps=30.0,
    )


def test_simulated_session_writes_metadata_and_csv(tmp_path) -> None:
    writer = SessionWriter(tmp_path)
    writer.start(
        SessionConfig(
            camera_index=0,
            width=1280,
            height=720,
            mirror=True,
            smoothing=0.65,
            model_name="pose_landmarker_full.task",
            plot_on_save=True,
        ),
        session_id="unit_test_session",
    )

    processor = KinematicsProcessor()
    for index, offset in enumerate((0.0, 0.1), start=1):
        pose_frame = _pose_frame(index, 1000 + index * 100, _landmarks(offset))
        kinematic_frame: KinematicFrame = processor.process(pose_frame)
        writer.add_frame(pose_frame, kinematic_frame)

    session_dir = writer.stop()
    assert session_dir is not None
    assert (session_dir / "metadata.json").exists()
    assert (session_dir / "landmarks.csv").exists()
    assert (session_dir / "kinematics.csv").exists()
    assert (session_dir / "summary.json").exists()
    assert (session_dir / "sequence_summary.json").exists()
    assert (session_dir / "angle_curves.png").exists()
    assert (session_dir / "velocity_curves.png").exists()

    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["program_version"]
    assert metadata["camera_view"] == "unknown"
    assert metadata["landmark_frame_count"] == 2
    assert metadata["pose_detected_frame_count"] == 2

    with (session_dir / "kinematics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2

    with (session_dir / "landmarks.csv").open(newline="", encoding="utf-8") as file:
        landmark_rows = list(csv.DictReader(file))
    pose_rows = [row for row in landmark_rows if "_hand_" not in row["landmark_name"]]
    assert len(pose_rows) == 2 * 22
    assert not any(row["landmark_name"] == "nose" for row in pose_rows)
    assert any(row["landmark_name"] == "left_shoulder" for row in pose_rows)


def test_session_writer_can_export_full_pose_landmarks(tmp_path) -> None:
    writer = SessionWriter(tmp_path)
    writer.start(
        SessionConfig(
            camera_index=0,
            width=1280,
            height=720,
            mirror=True,
            smoothing=0.65,
            model_name="pose_landmarker_full.task",
            plot_on_save=False,
            landmark_profile="full",
        ),
        session_id="full_pose_unit_test_session",
    )
    pose_frame = _pose_frame(1, 1100, _landmarks())
    writer.add_frame(pose_frame, KinematicsProcessor().process(pose_frame))

    session_dir = writer.stop()
    assert session_dir is not None
    with (session_dir / "landmarks.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    pose_rows = [row for row in rows if "_hand_" not in row["landmark_name"]]
    assert len(pose_rows) == len(LANDMARK_NAMES)
    assert any(row["landmark_name"] == "nose" for row in pose_rows)


def test_session_writer_exports_hand_landmarks(tmp_path) -> None:
    writer = SessionWriter(tmp_path)
    writer.start(
        SessionConfig(
            camera_index=0,
            width=1280,
            height=720,
            mirror=True,
            smoothing=0.65,
            model_name="pose_landmarker_full.task",
            plot_on_save=False,
            landmark_profile="full",
            hands_enabled=True,
            hand_model_name="hand_landmarker.task",
        ),
        session_id="hand_unit_test_session",
    )

    hand_points = empty_hand_landmarks()
    hand_points[0] = LandmarkPoint(0.40, 0.50, 0.0, 1.0, 1.0)
    hand_points[7] = LandmarkPoint(0.45, 0.30, 0.0, 1.0, 1.0)
    hand_points[8] = LandmarkPoint(0.46, 0.24, 0.0, 1.0, 1.0)
    hand_points[9] = LandmarkPoint(0.50, 0.28, 0.0, 1.0, 1.0)
    hand_points[15] = LandmarkPoint(0.55, 0.26, 0.0, 1.0, 1.0)
    hand_points[16] = LandmarkPoint(0.56, 0.22, 0.0, 1.0, 1.0)
    pose_frame = replace(
        _pose_frame(1, 1100, _landmarks()),
        hands_detected=True,
        hand_landmarks={"left": hand_points},
        smoothed_hand_landmarks={"left": hand_points},
    )
    kinematic_frame = KinematicsProcessor().process(pose_frame)
    writer.add_frame(pose_frame, kinematic_frame)

    session_dir = writer.stop()
    assert session_dir is not None
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["hands_enabled"] is True
    assert metadata["hands_detected_frame_count"] == 1

    with (session_dir / "landmarks.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    hand_rows = [row for row in rows if row["landmark_name"].startswith("left_hand_")]
    assert len(hand_rows) == len(SUPPLEMENTAL_FINGER_JOINTS)
    assert any(row["landmark_name"] == "left_hand_index_finger_dip" and row["image_x"] == "0.45" for row in hand_rows)
    assert any(row["landmark_name"] == "left_hand_middle_finger_mcp" and row["image_x"] == "0.5" for row in hand_rows)
    assert any(row["landmark_name"] == "left_hand_ring_finger_dip" and row["image_x"] == "0.55" for row in hand_rows)
    assert any(row["landmark_name"] == "left_hand_index_finger_tip" and row["image_x"] == "0.46" for row in hand_rows)
    assert any(row["landmark_name"] == "left_hand_ring_finger_tip" and row["image_x"] == "0.56" for row in hand_rows)
    assert not any(row["landmark_name"] == "left_hand_wrist" for row in hand_rows)
