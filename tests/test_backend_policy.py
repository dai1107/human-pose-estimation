from __future__ import annotations

from main import next_runtime_backend, parse_args, runtime_backend_switch_allowed
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
