from __future__ import annotations

from pathlib import Path

from hyrox.action_names import HYROX_ACTION_NAMES


ROOT = Path(__file__).resolve().parents[1]


def _documents() -> tuple[str, str]:
    return (
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "使用说明.md").read_text(encoding="utf-8"),
    )


def test_main_capabilities_models_and_health_checks_are_documented() -> None:
    readme, guide = _documents()
    for document in (readme, guide):
        for required in (
            "MediaPipe Pose",
            "YOLO11n Pose",
            "One Euro",
            "DTW",
            "python -m src.doctor",
            "python -m pytest -q",
            "CAMERA_VIEW_LIMITED",
            "check_multicamera.py",
        ):
            assert required in document


def test_all_hyrox_actions_and_runtime_switch_keys_are_in_both_documents() -> None:
    readme, guide = _documents()
    for document in (readme, guide):
        for action_name in HYROX_ACTION_NAMES:
            assert action_name in document
        assert "按 `A`" in document
        assert "按 `N`" in document
        assert "按 `V`" in document


def test_every_documented_hyrox_replay_command_specifies_camera_view() -> None:
    for document in _documents():
        replay_lines = [
            line.strip()
            for line in document.splitlines()
            if "replay_hyrox_video.py" in line and "--hyrox-action" in line
        ]
        assert replay_lines
        assert all("--camera-view" in line for line in replay_lines)
