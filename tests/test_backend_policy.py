from __future__ import annotations

from pathlib import Path

from main import build_pose_frame_from_result, make_output_path, next_runtime_backend, parse_args, runtime_backend_switch_allowed
from src.backends.base import Keypoint, PoseResult
from src.biomechanics.types import LandmarkPoint
from src.runtime_hand import HandDetection
from src.utils.backend_policy import resolve_backend_choice


def test_backend_auto_defaults_to_mediapipe_for_unknown_actions() -> None:
    assert resolve_backend_choice("auto") == "mediapipe"
    assert resolve_backend_choice("auto", action_type="squat") == "mediapipe"


def test_backend_auto_uses_hyrox_policy_from_action_type() -> None:
    assert resolve_backend_choice("auto", action_type="rowing") == "yolo-pose"
    assert resolve_backend_choice("auto", action_type="ski_erg") == "yolo-pose"
    assert resolve_backend_choice("auto", action_type="burpee_broad_jump") == "mediapipe"


def test_backend_auto_can_infer_from_hyrox_video_stem() -> None:
    assert resolve_backend_choice("auto", input_video="HYROX视频/划船机.mp4") == "yolo-pose"
    assert resolve_backend_choice("auto", input_video="HYROX视频/波比跳远.mp4") == "mediapipe"


def test_explicit_backend_overrides_action_policy() -> None:
    assert resolve_backend_choice("mediapipe", action_type="rowing") == "mediapipe"
    assert resolve_backend_choice("yolo-pose", action_type="burpee_broad_jump") == "yolo-pose"


def test_main_defaults_to_auto_backend() -> None:
    args = parse_args([])

    assert args.backend == "auto"
    assert args.action_type == "auto"
    assert args.yolo_device == "auto"
    assert args.hyrox_debug is False
    assert args.hyrox_action == "none"
    assert args.hyrox_sensitivity == "medium"
    assert args.hyrox_config == ""
    assert args.normalized_pose_debug is False
    assert args.landmark_profile == "full"
    assert args.metrics_overlay is False
    assert args.session_autostart is False
    assert args.analysis_mode == "pose"
    assert args.camera_view == "unknown"
    assert args.shot_type == "set_shot"
    assert args.shooting_side == "right"
    assert args.save_dir == "outputs"
    assert args.show_hands is False
    assert args.hand_model == "models/hand_landmarker.task"
    assert args.hand_detect_width == 416
    assert args.max_hand_detect_fps == 18.0
    assert args.max_hands == 2


def test_main_can_enable_hyrox_debug_overlay() -> None:
    args = parse_args(["--hyrox-debug"])

    assert args.hyrox_debug is True


def test_main_can_enable_normalized_pose_debug() -> None:
    args = parse_args(["--normalized-pose-debug"])

    assert args.normalized_pose_debug is True


def test_main_can_enable_hyrox_lunge_action() -> None:
    args = parse_args(["--hyrox-action", "lunge"])

    assert args.hyrox_action == "lunge"


def test_main_can_enable_hyrox_wall_ball_action() -> None:
    args = parse_args(["--hyrox-action", "wall_ball"])

    assert args.hyrox_action == "wall_ball"


def test_main_can_override_hyrox_sensitivity() -> None:
    args = parse_args(["--hyrox-action", "lunge", "--hyrox-sensitivity", "high"])

    assert args.hyrox_action == "lunge"
    assert args.hyrox_sensitivity == "high"


def test_main_can_override_hyrox_config() -> None:
    args = parse_args(["--hyrox-action", "lunge", "--hyrox-config", "custom_lunge.yaml"])

    assert args.hyrox_action == "lunge"
    assert args.hyrox_config == "custom_lunge.yaml"


def test_runtime_backend_hotkey_toggles_supported_backends() -> None:
    assert next_runtime_backend("mediapipe") == "yolo-pose"
    assert next_runtime_backend("yolo-pose") == "mediapipe"


def test_runtime_backend_switch_allowed_for_plain_realtime_pipeline() -> None:
    args = parse_args(["--backend", "auto"])

    allowed, reason = runtime_backend_switch_allowed(args)

    assert allowed
    assert reason == ""


def test_runtime_backend_switch_disabled_for_fusion_pipeline() -> None:
    args = parse_args(["--backend", "mediapipe", "--person-detector", "yolo", "--fusion", "yolo-roi-mediapipe"])

    allowed, reason = runtime_backend_switch_allowed(args)

    assert not allowed
    assert "fusion" in reason


def test_runtime_backend_switch_disabled_for_person_detector() -> None:
    args = parse_args(["--backend", "mediapipe", "--person-detector", "yolo"])

    allowed, reason = runtime_backend_switch_allowed(args)

    assert not allowed
    assert "person_detector" in reason


def test_make_output_path_uses_expected_directory_and_suffix(tmp_path: Path) -> None:
    path = make_output_path("recordings", ".mp4", root=tmp_path)

    assert path.parent == tmp_path / "recordings"
    assert path.suffix == ".mp4"


def test_main_can_parse_realtime_overlay_and_analysis_options() -> None:
    args = parse_args(
        [
            "--landmark-profile",
            "shot",
            "--show-hands",
            "--hand-detect-width",
            "640",
            "--max-hand-detect-fps",
            "12",
            "--max-hands",
            "1",
            "--metrics-overlay",
            "--session-autostart",
            "--analysis-mode",
            "basketball",
            "--camera-view",
            "front_left",
            "--shot-type",
            "jump_shot",
            "--shooting-side",
            "left",
            "--save-dir",
            "custom_outputs",
        ]
    )

    assert args.landmark_profile == "shot"
    assert args.show_hands is True
    assert args.hand_detect_width == 640
    assert args.max_hand_detect_fps == 12.0
    assert args.max_hands == 1
    assert args.metrics_overlay is True
    assert args.session_autostart is True
    assert args.analysis_mode == "basketball"
    assert args.camera_view == "front_left"
    assert args.shot_type == "jump_shot"
    assert args.shooting_side == "left"
    assert args.save_dir == "custom_outputs"


def test_build_pose_frame_from_result_maps_named_keypoints_into_pose_slots() -> None:
    result = PoseResult(
        keypoints=[
            Keypoint("left_shoulder", 0.2, 0.3, confidence=0.9),
            Keypoint("right_shoulder", 0.8, 0.3, confidence=0.8),
            Keypoint("left_knee", 0.3, 0.7, confidence=0.7),
        ],
        connections=(),
        model_name="yolo-pose",
        num_keypoints=3,
        success=True,
        inference_time_ms=5.0,
        timestamp_ms=123,
    )

    pose_frame = build_pose_frame_from_result(
        result,
        frame_index=7,
        mirror=True,
        frame_shape=(480, 640, 3),
        fps=29.5,
    )

    assert pose_frame.frame_index == 7
    assert pose_frame.timestamp_ms == 123
    assert pose_frame.pose_detected is True
    assert pose_frame.image_landmarks[11].x == 0.2
    assert pose_frame.image_landmarks[12].x == 0.8
    assert pose_frame.image_landmarks[25].x == 0.3


def test_build_pose_frame_from_result_can_include_hand_landmarks() -> None:
    result = PoseResult(
        keypoints=[Keypoint("left_shoulder", 0.2, 0.3, confidence=0.9)],
        connections=(),
        model_name="mediapipe",
        num_keypoints=1,
        success=True,
        inference_time_ms=4.0,
        timestamp_ms=456,
    )
    hand_detection = HandDetection(
        side="left",
        score=0.95,
        landmarks=[LandmarkPoint(0.1, 0.2, visibility=0.9, presence=0.9)],
        world_landmarks=[LandmarkPoint(0.01, 0.02, 0.03, visibility=0.9, presence=0.9)],
    )

    pose_frame = build_pose_frame_from_result(
        result,
        frame_index=8,
        mirror=False,
        frame_shape=(720, 1280, 3),
        fps=30.0,
        hand_detections={"left": hand_detection},
    )

    assert pose_frame.hands_detected is True
    assert pose_frame.hand_landmarks["left"][0].x == 0.1
    assert pose_frame.hand_world_landmarks["left"][0].z == 0.03
    assert pose_frame.smoothed_hand_landmarks["left"][0].presence == 0.9
