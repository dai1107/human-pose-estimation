from __future__ import annotations

import pytest
import numpy as np

from hyrox.actions import LungeAnalyzer
from hyrox.actions.lunge import PHASE_CONFIRMATION_FRAMES
from hyrox.base import BaseActionAnalyzer
from hyrox.config import DEFAULT_LUNGE_CONFIG, load_lunge_config
from hyrox.features import extract_basic_pose_features
from hyrox.geometry import PosePoint, angle_3pts, midpoint, safe_distance
from hyrox.landmark_names import (
    LANDMARK_INDEX,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from src.backends.base import Keypoint
from src.utils.draw_utils import format_hyrox_action_lines, format_hyrox_debug_lines, put_text


def _make_landmarks() -> list[dict[str, float]]:
    landmarks = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0, "presence": 0.0}
        for _ in range(33)
    ]

    def set_point(index: int, x: float, y: float, visibility: float = 0.95) -> None:
        landmarks[index] = {
            "x": x,
            "y": y,
            "z": 0.0,
            "visibility": visibility,
            "presence": 1.0,
        }

    set_point(LEFT_SHOULDER, 0.40, 0.20)
    set_point(RIGHT_SHOULDER, 0.60, 0.20)
    set_point(LEFT_ELBOW, 0.36, 0.28)
    set_point(RIGHT_ELBOW, 0.64, 0.28)
    set_point(LEFT_WRIST, 0.34, 0.36)
    set_point(RIGHT_WRIST, 0.66, 0.36)
    set_point(LEFT_HIP, 0.40, 0.50)
    set_point(RIGHT_HIP, 0.60, 0.50)
    set_point(LEFT_KNEE, 0.40, 0.75)
    set_point(RIGHT_KNEE, 0.60, 0.75)
    set_point(LEFT_ANKLE, 0.40, 0.95)
    set_point(RIGHT_ANKLE, 0.60, 0.95)
    set_point(LEFT_HEEL, 0.39, 0.98)
    set_point(RIGHT_HEEL, 0.61, 0.98)
    set_point(LEFT_FOOT_INDEX, 0.42, 0.98)
    set_point(RIGHT_FOOT_INDEX, 0.58, 0.98)
    return landmarks


def test_landmark_index_exposes_required_names() -> None:
    assert LANDMARK_INDEX["left_shoulder"] == LEFT_SHOULDER
    assert LANDMARK_INDEX["right_hip"] == RIGHT_HIP
    assert LANDMARK_INDEX["left_foot_index"] == LEFT_FOOT_INDEX


def test_lunge_config_loader_uses_defaults_when_file_missing(tmp_path) -> None:
    config = load_lunge_config(tmp_path / "missing_lunge.yaml")

    assert config["config_name"] == DEFAULT_LUNGE_CONFIG["config_name"]
    assert config["visibility_min"] == DEFAULT_LUNGE_CONFIG["visibility_min"]


def test_lunge_analyzer_can_initialize_from_config_dict() -> None:
    analyzer = LungeAnalyzer.from_config(
        {
            "config_name": "unit_test_cfg",
            "visibility_min": 0.6,
            "stand_knee_angle_min": 162.0,
            "bottom_knee_angle_max": 109.0,
            "torso_lean_warn": 17.0,
            "stable_frames": 5,
            "rep_cooldown_ms": 520,
        },
        sensitivity="medium",
    )

    assert analyzer.config_name == "unit_test_cfg"
    assert analyzer.min_visible_score == 0.6
    assert analyzer.stand_knee_angle == 162.0
    assert analyzer.bottom_knee_angle == 109.0
    assert analyzer.torso_lean_warn_angle == 17.0
    assert analyzer.confirmation_frames == 5
    assert analyzer.rep_cooldown_ms == 520


def test_lunge_config_uses_file_stem_when_name_is_omitted(tmp_path) -> None:
    path = tmp_path / "camera_side.yaml"
    path.write_text("visibility_min: 0.52\n", encoding="utf-8")

    config = load_lunge_config(path)

    assert config["config_name"] == "camera_side"
    assert config["visibility_min"] == 0.52


def test_lunge_analyzer_invalid_config_values_fall_back_safely() -> None:
    analyzer = LungeAnalyzer.from_config(
        {
            "visibility_min": "invalid",
            "stable_frames": "invalid",
            "rep_cooldown_ms": None,
        }
    )

    assert analyzer.min_visible_score == 0.45
    assert analyzer.confirmation_frames == 3
    assert analyzer.rep_cooldown_ms == 400


def test_geometry_handles_visibility_and_none() -> None:
    hidden = {"x": 0.0, "y": 0.0, "visibility": 0.1, "presence": 1.0}
    visible_a = {"x": 0.0, "y": 0.0, "visibility": 1.0, "presence": 1.0}
    visible_b = {"x": 3.0, "y": 4.0, "visibility": 1.0, "presence": 1.0}
    visible_c = {"x": 1.0, "y": 0.0, "visibility": 1.0, "presence": 1.0}
    visible_d = {"x": 1.0, "y": 1.0, "visibility": 1.0, "presence": 1.0}

    assert safe_distance(None, visible_b) is None
    assert safe_distance(hidden, visible_b) is None
    assert safe_distance(visible_a, visible_b) == pytest.approx(5.0)
    assert angle_3pts(visible_a, visible_c, visible_d) == pytest.approx(90.0)

    center = midpoint(visible_a, visible_b)
    assert center == PosePoint(x=1.5, y=2.0, z=0.0, visibility=1.0, presence=1.0)


def test_extract_basic_pose_features_returns_expected_fields() -> None:
    features = extract_basic_pose_features(_make_landmarks(), image_width=640, image_height=480)

    assert features["left_knee_angle"] == pytest.approx(180.0)
    assert features["right_knee_angle"] == pytest.approx(180.0)
    assert features["left_hip_angle"] == pytest.approx(180.0)
    assert features["right_hip_angle"] == pytest.approx(180.0)
    assert features["left_elbow_angle"] is not None
    assert features["right_elbow_angle"] is not None
    assert features["left_shoulder_angle"] is not None
    assert features["right_shoulder_angle"] is not None
    assert features["torso_angle"] == pytest.approx(0.0)
    assert features["shoulder_tilt"] == pytest.approx(0.0)
    assert features["hip_tilt"] == pytest.approx(0.0)
    assert features["body_center_x"] == pytest.approx(0.5)
    assert features["body_center_y"] == pytest.approx(0.35)
    assert features["body_height_norm"] == pytest.approx(0.75)
    assert features["left_wrist_y"] == pytest.approx(0.36)
    assert features["right_wrist_y"] == pytest.approx(0.36)
    assert features["left_ankle_y"] == pytest.approx(0.95)
    assert features["right_ankle_y"] == pytest.approx(0.95)
    assert features["left_wrist_to_hip_y"] == pytest.approx(-0.14)
    assert features["right_wrist_to_hip_y"] == pytest.approx(-0.14)
    assert features["left_wrist_to_shoulder_y"] == pytest.approx(0.16)
    assert features["right_wrist_to_shoulder_y"] == pytest.approx(0.16)
    assert features["wrist_distance_norm"] == pytest.approx(0.32)
    assert features["ankle_distance_norm"] == pytest.approx(0.20)
    assert features["min_knee_angle"] == pytest.approx(180.0)
    assert features["min_hip_angle"] == pytest.approx(180.0)
    assert features["hip_center_y"] == pytest.approx(0.5)
    assert features["knee_center_y"] == pytest.approx(0.75)
    assert features["hip_knee_depth"] == pytest.approx(-0.25)
    assert features["wrist_above_shoulder"] == pytest.approx(-0.16)
    assert features["hip_width"] == pytest.approx(0.2)
    assert features["knee_width"] == pytest.approx(0.2)
    assert features["ankle_width"] == pytest.approx(0.2)
    assert features["visible_score"] == pytest.approx(0.95)
    assert features["upper_body_visible_score"] == pytest.approx(0.95)
    assert features["lower_body_visible_score"] == pytest.approx(0.95)
    assert features["left_side_visible_score"] == pytest.approx(0.95)
    assert features["right_side_visible_score"] == pytest.approx(0.95)


def test_extract_basic_pose_features_handles_missing_landmarks() -> None:
    features = extract_basic_pose_features(None, image_width=640, image_height=480)

    assert features["left_shoulder_angle"] is None
    assert features["body_center_x"] is None
    assert features["left_wrist_y"] is None
    assert features["wrist_distance_norm"] is None
    assert features["visible_score"] == 0.0
    assert features["upper_body_visible_score"] == 0.0


def test_base_action_analyzer_produces_minimal_state() -> None:
    analyzer = BaseActionAnalyzer(action="ski_erg", min_visible_score=0.5)

    ready_state = analyzer.update({"visible_score": 0.9}, timestamp_ms=1200)
    assert ready_state["action"] == "ski_erg"
    assert ready_state["phase"] == "ready"
    assert ready_state["rep_count"] == 0
    assert ready_state["feedback_messages"] == []
    assert ready_state["debug"]["timestamp_ms"] == 1200

    low_visibility_state = analyzer.update({"visible_score": 0.1}, timestamp_ms=1300)
    assert low_visibility_state["phase"] == "low_visibility"
    assert len(low_visibility_state["feedback_messages"]) == 1
    message = low_visibility_state["feedback_messages"][0]
    assert message.code == "low_visibility"
    assert message.level == "warn"

    analyzer.reset()
    assert analyzer.phase == "idle"
    assert analyzer.rep_count == 0
    assert analyzer.last_timestamp_ms is None


def test_extract_basic_pose_features_accepts_runtime_keypoints() -> None:
    keypoints = [
        Keypoint("left_shoulder", 0.40, 0.20, confidence=0.9),
        Keypoint("right_shoulder", 0.60, 0.20, confidence=0.9),
        Keypoint("left_hip", 0.40, 0.50, confidence=0.9),
        Keypoint("right_hip", 0.60, 0.50, confidence=0.9),
        Keypoint("left_knee", 0.40, 0.75, confidence=0.9),
        Keypoint("right_knee", 0.60, 0.75, confidence=0.9),
        Keypoint("left_ankle", 0.40, 0.95, confidence=0.9),
        Keypoint("right_ankle", 0.60, 0.95, confidence=0.9),
    ]

    features = extract_basic_pose_features(keypoints, image_width=640, image_height=480)

    assert features["left_knee_angle"] == pytest.approx(180.0)
    assert features["right_hip_angle"] == pytest.approx(180.0)
    assert features["visible_score"] == pytest.approx(0.45)


def test_hyrox_debug_lines_show_pose_values_or_no_pose() -> None:
    lines = format_hyrox_debug_lines(
        {
            "visible_score": 0.93,
            "left_knee_angle": 179.4,
            "right_knee_angle": 178.2,
            "left_hip_angle": 169.0,
            "right_hip_angle": 170.0,
            "torso_angle": 2.3,
        },
        has_pose=True,
    )

    assert lines[0] == "visible: 0.93"
    assert lines[1] == "lknee: 179.4"
    assert lines[-1] == "torso: 2.3"
    assert format_hyrox_debug_lines(None, has_pose=False) == ["No pose"]


def test_put_text_renders_chinese_feedback_without_crashing() -> None:
    frame = np.zeros((80, 420, 3), dtype=np.uint8)

    put_text(frame, "提示：保持核心稳定", (12, 42), (80, 230, 120))

    assert np.count_nonzero(frame) > 0


def test_lunge_analyzer_counts_reps_and_reports_extension_feedback() -> None:
    analyzer = LungeAnalyzer()

    stand = {
        "visible_score": 0.95,
        "left_knee_angle": 175.0,
        "right_knee_angle": 176.0,
        "left_hip_angle": 172.0,
        "right_hip_angle": 171.0,
        "torso_angle": 4.0,
    }
    shallow_stand = {
        "visible_score": 0.95,
        "left_knee_angle": 160.0,
        "right_knee_angle": 169.0,
        "left_hip_angle": 155.0,
        "right_hip_angle": 168.0,
        "torso_angle": 6.0,
    }

    pending = analyzer.update(stand, timestamp_ms=100)
    assert pending["phase"] == "unknown"
    assert pending["debug"]["raw_phase"] == "stand"
    assert pending["debug"]["stable_phase"] == "unknown"
    assert pending["debug"]["frames_in_phase"] == 1

    analyzer.update(stand, timestamp_ms=150)
    assert analyzer.update(stand, timestamp_ms=200)["phase"] == "stand"

    descent = {**stand, "left_knee_angle": 138.0, "left_hip_angle": 148.0}
    analyzer.update(descent, timestamp_ms=250)
    analyzer.update(descent, timestamp_ms=300)
    assert analyzer.update(descent, timestamp_ms=350)["phase"] == "descent"

    bottom = {**stand, "left_knee_angle": 92.0, "left_hip_angle": 132.0}
    analyzer.update(bottom, timestamp_ms=400)
    analyzer.update(bottom, timestamp_ms=450)
    assert analyzer.update(bottom, timestamp_ms=500)["phase"] == "bottom"

    ascent = {**stand, "left_knee_angle": 132.0, "left_hip_angle": 146.0}
    analyzer.update(ascent, timestamp_ms=550)
    analyzer.update(ascent, timestamp_ms=600)
    assert analyzer.update(ascent, timestamp_ms=650)["phase"] == "ascent"

    analyzer.update(shallow_stand, timestamp_ms=700)
    analyzer.update(shallow_stand, timestamp_ms=750)
    completed = analyzer.update(shallow_stand, timestamp_ms=800)

    assert completed["phase"] == "stand"
    assert completed["rep_count"] == 1
    assert completed["debug"]["last_rep_time_ms"] == 800
    assert {message.code for message in completed["feedback_messages"]} == {"STAND_EXTENSION"}


def test_lunge_analyzer_emits_depth_lean_and_visibility_feedback() -> None:
    analyzer = LungeAnalyzer()

    low_visibility = analyzer.update({"visible_score": 0.2}, timestamp_ms=100)
    assert low_visibility["phase"] == "unknown"
    assert low_visibility["feedback_messages"][0].code == "LOW_VISIBILITY"
    assert len(low_visibility["feedback_messages"]) == 1
    assert low_visibility["debug"]["raw_phase"] == "low_visibility"

    stand = {
        "visible_score": 0.95,
        "left_knee_angle": 170.0,
        "right_knee_angle": 172.0,
        "left_hip_angle": 170.0,
        "right_hip_angle": 171.0,
        "torso_angle": 4.0,
    }
    analyzer.update(stand, timestamp_ms=200)
    analyzer.update(stand, timestamp_ms=250)
    analyzer.update(stand, timestamp_ms=300)

    bottom = {
        "visible_score": 0.95,
        "left_knee_angle": 108.0,
        "right_knee_angle": 171.0,
        "left_hip_angle": 140.0,
        "right_hip_angle": 170.0,
        "torso_angle": 28.0,
    }
    analyzer.update(bottom, timestamp_ms=350)
    analyzer.update(bottom, timestamp_ms=400)
    state = analyzer.update(
        bottom,
        timestamp_ms=450,
    )

    assert state["phase"] == "bottom"
    assert [message.code for message in state["feedback_messages"]] == ["LEAN_TOO_MUCH", "NOT_DEEP_ENOUGH"]


def test_lunge_analyzer_debounces_phase_changes_before_confirming() -> None:
    analyzer = LungeAnalyzer()
    stand = {
        "visible_score": 0.95,
        "left_knee_angle": 174.0,
        "right_knee_angle": 175.0,
        "left_hip_angle": 170.0,
        "right_hip_angle": 171.0,
        "torso_angle": 3.0,
    }
    bottom = {
        "visible_score": 0.95,
        "left_knee_angle": 94.0,
        "right_knee_angle": 168.0,
        "left_hip_angle": 136.0,
        "right_hip_angle": 169.0,
        "torso_angle": 5.0,
    }

    analyzer.update(stand, timestamp_ms=100)
    analyzer.update(stand, timestamp_ms=150)
    analyzer.update(stand, timestamp_ms=200)

    first_bottom = analyzer.update(bottom, timestamp_ms=250)
    second_bottom = analyzer.update(bottom, timestamp_ms=300)
    third_bottom = analyzer.update(bottom, timestamp_ms=350)

    assert first_bottom["phase"] == "stand"
    assert first_bottom["debug"]["raw_phase"] == "bottom"
    assert first_bottom["debug"]["stable_phase"] == "stand"
    assert second_bottom["debug"]["frames_in_phase"] == 2
    assert third_bottom["phase"] == "bottom"
    assert third_bottom["debug"]["stable_phase"] == "bottom"
    assert analyzer.confirmation_frames == PHASE_CONFIRMATION_FRAMES


def test_lunge_analyzer_respects_rep_cooldown() -> None:
    analyzer = LungeAnalyzer()

    stand = {
        "visible_score": 0.95,
        "left_knee_angle": 175.0,
        "right_knee_angle": 176.0,
        "left_hip_angle": 171.0,
        "right_hip_angle": 172.0,
        "torso_angle": 4.0,
    }
    bottom = {
        "visible_score": 0.95,
        "left_knee_angle": 94.0,
        "right_knee_angle": 170.0,
        "left_hip_angle": 132.0,
        "right_hip_angle": 171.0,
        "torso_angle": 6.0,
    }

    for timestamp_ms in (100, 150, 200):
        analyzer.update(stand, timestamp_ms=timestamp_ms)
    for timestamp_ms in (250, 300, 350):
        analyzer.update(bottom, timestamp_ms=timestamp_ms)
    for timestamp_ms in (400, 450, 500):
        first_rep = analyzer.update(stand, timestamp_ms=timestamp_ms)

    assert first_rep["rep_count"] == 1
    assert first_rep["debug"]["last_rep_time_ms"] == 500

    for timestamp_ms in (560, 610, 660):
        analyzer.update(bottom, timestamp_ms=timestamp_ms)
    for timestamp_ms in (710, 760, 810):
        blocked_rep = analyzer.update(stand, timestamp_ms=timestamp_ms)

    assert blocked_rep["phase"] == "stand"
    assert blocked_rep["rep_count"] == 1
    assert blocked_rep["debug"]["last_rep_time_ms"] == 500

    for timestamp_ms in (920, 970, 1020):
        analyzer.update(bottom, timestamp_ms=timestamp_ms)
    for timestamp_ms in (1070, 1120, 1170):
        second_rep = analyzer.update(stand, timestamp_ms=timestamp_ms)

    assert second_rep["rep_count"] == 2
    assert second_rep["debug"]["last_rep_time_ms"] == 1170


def test_lunge_analyzer_sensitivity_profiles_change_confirmation_frames() -> None:
    assert LungeAnalyzer(sensitivity="low").confirmation_frames == 4
    assert LungeAnalyzer(sensitivity="medium").confirmation_frames == 3
    assert LungeAnalyzer(sensitivity="high").confirmation_frames == 2


def test_lunge_analyzer_sensitivity_adjusts_loaded_default_config() -> None:
    low = LungeAnalyzer.from_config_path("configs/hyrox/lunge.yaml", sensitivity="low")
    high = LungeAnalyzer.from_config_path("configs/hyrox/lunge.yaml", sensitivity="high")

    assert low.confirmation_frames == 4
    assert high.confirmation_frames == 2
    assert low.bottom_knee_angle == pytest.approx(105.0)
    assert high.bottom_knee_angle == pytest.approx(125.0)


def test_lunge_bottom_requires_hip_drop_when_stand_reference_is_available() -> None:
    analyzer = LungeAnalyzer(confirmation_frames=1, hip_drop_min=0.035)
    stand = {
        "visible_score": 0.95,
        "left_knee_angle": 175.0,
        "right_knee_angle": 176.0,
        "left_hip_angle": 171.0,
        "right_hip_angle": 172.0,
        "torso_angle": 4.0,
        "hip_center_y": 0.45,
    }
    not_low = {
        **stand,
        "left_knee_angle": 92.0,
        "left_hip_angle": 132.0,
        "hip_center_y": 0.46,
    }
    low = {**not_low, "hip_center_y": 0.50}

    assert analyzer.update(stand, timestamp_ms=100)["phase"] == "stand"
    rejected = analyzer.update(not_low, timestamp_ms=150)
    accepted = analyzer.update(low, timestamp_ms=200)

    assert rejected["phase"] == "descent"
    assert rejected["debug"]["hip_drop"] == pytest.approx(0.01)
    assert accepted["phase"] == "bottom"
    assert accepted["debug"]["hip_drop"] == pytest.approx(0.05)


def test_lunge_stand_phase_uses_both_knees_not_occluded_hip_angle() -> None:
    analyzer = LungeAnalyzer(confirmation_frames=1)

    state = analyzer.update(
        {
            "visible_score": 0.95,
            "left_knee_angle": 166.0,
            "right_knee_angle": 168.0,
            "left_hip_angle": 132.0,
            "right_hip_angle": 170.0,
            "torso_angle": 4.0,
        },
        timestamp_ms=100,
    )

    assert state["phase"] == "stand"


def test_hyrox_action_lines_show_state_and_feedback() -> None:
    state = {
        "action": "Lunge",
        "phase": "ascent",
        "rep_count": 2,
        "debug": {"config_name": "lunge_default"},
        "feedback_messages": [
            {"level": "warn", "text": "躯干前倾过多，保持核心稳定"},
        ],
    }

    lines = format_hyrox_action_lines(state)

    assert lines[0][0] == "action: Lunge"
    assert lines[1][0] == "cfg: lunge_default"
    assert lines[2][0] == "view: unknown / unknown"
    assert lines[3][0] == "phase: ascent"
    assert lines[4][0] == "reps: 2"
    assert lines[5][0] == "tip: 躯干前倾过多，保持核心稳定"


def test_hyrox_action_lines_limit_feedback_to_two_messages() -> None:
    lines = format_hyrox_action_lines(
        {
            "action": "Lunge",
            "phase": "stand",
            "rep_count": 1,
            "debug": {"config_name": "lunge_default"},
            "feedback_messages": [
                {"level": "error", "text": "A"},
                {"level": "warn", "text": "B"},
                {"level": "info", "text": "C"},
            ],
        }
    )

    assert [line for line, _ in lines[5:]] == ["tip: A", "tip: B"]
