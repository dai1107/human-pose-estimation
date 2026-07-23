from __future__ import annotations

from src.realtime_pose import parse_args
from src.realtime.cli import parse_args as parse_consolidated_args


def test_realtime_defaults_keep_full_pose_model_and_hide_face() -> None:
    args = parse_args([])

    assert args.model == "models/pose_landmarker_full.task"
    assert args.landmark_profile == "no-face"
    assert args.width == 640
    assert args.height == 480
    assert args.camera_fps == 60.0
    assert args.camera_fourcc == "MJPG"
    assert args.detect_width == 480
    assert args.max_detect_fps == 30.0
    assert args.hand_detect_width == 416
    assert args.max_hand_detect_fps == 18.0
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
            "--camera-fps",
            "45",
            "--camera-fourcc",
            "YUY2",
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
    assert args.camera_fps == 45.0
    assert args.camera_fourcc == "YUY2"
    assert args.detect_width == 960
    assert args.max_detect_fps == 30.0
    assert args.hand_detect_width == 480
    assert args.max_hand_detect_fps == 15.0


def test_consolidated_camera_backend_defaults_to_device_cache_auto_selection() -> None:
    args = parse_consolidated_args([])

    assert args.camera_api == "auto"
    assert args.camera_backend_cache == "outputs/camera_backend_cache.json"


def test_consolidated_camera_backend_can_be_selected_explicitly() -> None:
    args = parse_consolidated_args(
        [
            "--camera-api",
            "msmf",
            "--camera-backend-cache",
            "outputs/test-camera-cache.json",
        ]
    )

    assert args.camera_api == "msmf"
    assert args.camera_backend_cache == "outputs/test-camera-cache.json"
