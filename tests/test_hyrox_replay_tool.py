from __future__ import annotations

from pathlib import Path

from tools.replay_hyrox_video import (
    build_debug_row,
    build_parser,
    create_analyzer,
    frame_timestamp_ms,
    playback_delay_ms,
    serialize_feedback,
    write_debug_csv,
)


def test_replay_tool_parser_supports_expected_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--video",
            "sample.mp4",
            "--hyrox-action",
            "lunge",
            "--hyrox-sensitivity",
            "high",
            "--hyrox-config",
            "custom_lunge.yaml",
            "--speed",
            "2.0",
            "--save-debug-csv",
            "debug.csv",
        ]
    )

    assert args.video == "sample.mp4"
    assert args.hyrox_action == "lunge"
    assert args.hyrox_sensitivity == "high"
    assert args.hyrox_config == "custom_lunge.yaml"
    assert args.speed == 2.0
    assert args.save_debug_csv == "debug.csv"


def test_replay_tool_timing_helpers_use_speed_and_fps() -> None:
    assert frame_timestamp_ms(30, 30.0) == 1000
    assert playback_delay_ms(20.0, 2.0) == 25
    assert playback_delay_ms(20.0, 0.5) == 100


def test_replay_tool_serializes_feedback_and_debug_rows() -> None:
    state = {
        "action": "Lunge",
        "phase": "bottom",
        "rep_count": 2,
        "feedback_messages": [
            {"code": "LEAN_TOO_MUCH", "text": "躯干前倾过多，保持核心稳定"},
            {"code": "NOT_DEEP_ENOUGH", "text": "下蹲幅度不够，后侧膝盖应接近地面"},
        ],
        "debug": {
            "raw_phase": "bottom",
            "stable_phase": "bottom",
            "frames_in_phase": 3,
            "last_rep_time_ms": 800,
            "confirmation_frames": 3,
            "rep_cooldown_ms": 400,
            "sensitivity": "medium",
            "config_name": "lunge_default",
        },
    }
    features = {
        "visible_score": 0.92,
        "left_knee_angle": 101.0,
        "right_knee_angle": 170.0,
        "left_hip_angle": 138.0,
        "right_hip_angle": 172.0,
        "torso_angle": 24.0,
        "min_knee_angle": 101.0,
        "min_hip_angle": 138.0,
        "hip_center_y": 0.62,
        "knee_center_y": 0.78,
    }

    row = build_debug_row(frame_index=12, timestamp_ms=400, has_pose=True, features=features, state=state)

    assert serialize_feedback(state["feedback_messages"], "code") == "LEAN_TOO_MUCH | NOT_DEEP_ENOUGH"
    assert row["frame_index"] == 12
    assert row["pose_detected"] == 1
    assert row["phase"] == "bottom"
    assert row["feedback_codes"] == "LEAN_TOO_MUCH | NOT_DEEP_ENOUGH"
    assert row["raw_phase"] == "bottom"
    assert row["last_rep_time_ms"] == 800
    assert row["config_name"] == "lunge_default"


def test_replay_tool_can_write_debug_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "debug" / "replay.csv"
    write_debug_csv(
        output_path,
        [
            build_debug_row(
                frame_index=1,
                timestamp_ms=33,
                has_pose=False,
                features=None,
                state={"action": "Lunge", "phase": "unknown", "rep_count": 0, "feedback_messages": [], "debug": {}},
            )
        ],
    )

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8-sig")
    assert "frame_index,timestamp_ms,pose_detected" in text
    assert "Lunge,unknown,0" in text


def test_replay_tool_uses_shared_lunge_analyzer() -> None:
    analyzer = create_analyzer("lunge", "low", "configs/hyrox/lunge.yaml")

    assert analyzer.action == "Lunge"
    assert analyzer.sensitivity == "low"
    assert analyzer.config_name == "lunge_default"
