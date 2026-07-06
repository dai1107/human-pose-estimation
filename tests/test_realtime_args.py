from __future__ import annotations

from src.realtime_pose import parse_args


def test_realtime_defaults_keep_full_pose_model_and_hide_face() -> None:
    args = parse_args([])

    assert args.model == "models/pose_landmarker_full.task"
    assert args.landmark_profile == "no-face"
    assert args.detect_width == 640
    assert args.max_detect_fps == 24.0
    assert args.hand_detect_width == 416
    assert args.max_hand_detect_fps == 12.0
    assert args.max_pending_ms == 180
    assert args.max_result_lag_ms == 280


def test_realtime_can_start_with_face_landmarks_visible() -> None:
    args = parse_args(["--landmark-profile", "full"])

    assert args.landmark_profile == "full"
    assert args.model == "models/pose_landmarker_full.task"


def test_explicit_detection_options_override_defaults() -> None:
    args = parse_args(
        [
            "--model",
            "models/custom.task",
            "--detect-width",
            "960",
            "--max-detect-fps",
            "30",
            "--hand-detect-width",
            "480",
            "--max-hand-detect-fps",
            "15",
        ]
    )

    assert args.model == "models/custom.task"
    assert args.detect_width == 960
    assert args.max_detect_fps == 30.0
    assert args.hand_detect_width == 480
    assert args.max_hand_detect_fps == 15.0
