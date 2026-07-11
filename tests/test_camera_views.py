from __future__ import annotations

import pytest
import numpy as np

from hyrox.feedback import FeedbackMessage
from hyrox.registry import create_action_analyzer
from hyrox.view_policy import filter_feedback_for_view, next_camera_view, view_profile
from src.camera.multiview import CameraSource, MultiCameraCapture, MultiCameraPlan, parse_camera_source
from src.utils.draw_utils import format_hyrox_action_lines
from tools.replay_hyrox_video import build_parser
from tools.check_multicamera import build_parser as build_multicamera_parser


def _message(code: str) -> FeedbackMessage:
    return FeedbackMessage("warn", code, code, 0.8)


def test_front_and_side_profiles_use_different_wall_ball_rules() -> None:
    messages = [_message("SQUAT_NOT_DEEP"), _message("KNEES_CAVE_IN"), _message("NOT_FULL_EXTENSION")]

    front, front_limited = filter_feedback_for_view("wall_ball", "front", messages)
    side, side_limited = filter_feedback_for_view("wall_ball", "side", messages)

    assert {message.code for message in front} == {"KNEES_CAVE_IN", "NOT_FULL_EXTENSION"}
    assert {message.code for message in side} == {"SQUAT_NOT_DEEP", "NOT_FULL_EXTENSION"}
    assert front_limited is False
    assert side_limited is False


def test_side_preferred_action_marks_front_view_as_limited() -> None:
    filtered, limited = filter_feedback_for_view(
        "rowing",
        "front",
        [_message("TOO_MUCH_BACK_LEAN"), _message("NOT_SEATED_OR_BAD_VIEW")],
    )

    assert limited is True
    assert [message.code for message in filtered] == ["NOT_SEATED_OR_BAD_VIEW", "CAMERA_VIEW_LIMITED"]


def test_factory_and_replay_parser_accept_camera_view() -> None:
    analyzer = create_action_analyzer("lunge", camera_view="front_left")
    args = build_parser().parse_args(
        ["--video", "sample.mp4", "--hyrox-action", "lunge", "--camera-view", "side"]
    )

    assert analyzer.camera_view == "front_left"
    assert analyzer.camera_view_profile == "front"
    assert args.camera_view == "side"


def test_camera_view_cycles_and_overlay_exposes_active_profile() -> None:
    assert next_camera_view("unknown") == "front"
    assert next_camera_view("front") == "side"
    assert next_camera_view("side") == "front"
    assert view_profile("front_right") == "front"
    lines = format_hyrox_action_lines(
        {
            "action": "rowing",
            "phase": "drive",
            "rep_count": 1,
            "feedback_messages": [],
            "debug": {"camera_view": "side", "view_profile": "side"},
        }
    )
    assert lines[2][0] == "view: side / side"


def test_unknown_view_context_prompts_for_explicit_selection() -> None:
    analyzer = create_action_analyzer("rowing")
    state = analyzer.attach_view_context(
        {"action": "rowing", "phase": "catch", "rep_count": 0, "feedback_messages": [], "debug": {}}
    )

    assert state["feedback_messages"][0].code == "CAMERA_VIEW_REQUIRED"
    assert state["debug"]["view_profile"] == "unknown"


def test_multicamera_plan_accepts_front_and_side_sources() -> None:
    front = parse_camera_source("0:front:mirror")
    side = parse_camera_source("1:side:no-mirror")
    plan = MultiCameraPlan.from_sources([front, side], primary_camera_index=0)

    assert front == CameraSource(0, "front", mirror=True, name="camera_0_front")
    assert side.mirror is False
    assert plan.is_multiview is True
    assert plan.has_front_and_side is True


def test_multicamera_plan_rejects_duplicate_indices_and_unknown_views() -> None:
    with pytest.raises(ValueError, match="explicit front or side"):
        CameraSource(0, "unknown")
    with pytest.raises(ValueError, match="unique"):
        MultiCameraPlan.from_sources([CameraSource(0, "front"), CameraSource(0, "side")])


def test_multicamera_capture_reads_and_releases_every_source(monkeypatch) -> None:
    import cv2

    captures = []

    class FakeCapture:
        def __init__(self, index: int) -> None:
            self.index = index
            self.released = False
            captures.append(self)

        def set(self, *_args) -> bool:
            return True

        def isOpened(self) -> bool:
            return True

        def read(self):
            return True, np.full((4, 6, 3), self.index, dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", lambda index, *_args: FakeCapture(index))
    plan = MultiCameraPlan.from_sources([CameraSource(0, "front"), CameraSource(1, "side")])

    with MultiCameraCapture(plan) as capture:
        bundle = capture.read()
        assert len(bundle.frames) == 2
        assert bundle.skew_ms >= 0

    assert all(capture.released for capture in captures)


def test_multicamera_check_tool_accepts_repeated_sources() -> None:
    args = build_multicamera_parser().parse_args(
        ["--camera", "0:front", "--camera", "1:side:no-mirror", "--frames", "10"]
    )
    assert args.camera == ["0:front", "1:side:no-mirror"]
    assert args.frames == 10
