from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from main import main, read_capture_frame
from src.biomechanics.session_writer import (
    SessionConfig,
    SessionWriteError,
    SessionWriter,
)
from src.runtime_logging import (
    ExitCode,
    InputSourceError,
    configure_logging,
    safe_cleanup,
)
from tools.replay_hyrox_video import main as replay_main


class _BrokenCapture:
    def read(self) -> tuple[bool, None]:
        return False, None


def test_camera_read_failure_is_not_treated_as_normal_eof() -> None:
    with pytest.raises(InputSourceError, match="摄像头"):
        read_capture_frame(
            _BrokenCapture(),  # type: ignore[arg-type]
            input_mode="camera",
            processed_frames=3,
        )


def test_empty_or_damaged_video_is_not_treated_as_normal_eof() -> None:
    with pytest.raises(InputSourceError, match="损坏"):
        read_capture_frame(
            _BrokenCapture(),  # type: ignore[arg-type]
            input_mode="video",
            processed_frames=0,
        )


def test_video_eof_after_frames_is_a_normal_stop() -> None:
    ok, frame = read_capture_frame(
        _BrokenCapture(),  # type: ignore[arg-type]
        input_mode="video",
        processed_frames=3,
    )
    assert ok is False
    assert frame is None


def test_main_returns_config_exit_code_before_backend_start(
    tmp_path: Path,
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("visibility_mim: 0.5\n", encoding="utf-8")

    result = main(
        [
            "--hyrox-action",
            "lunge",
            "--hyrox-config",
            str(config),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    assert result == ExitCode.CONFIG_ERROR
    assert (tmp_path / "logs" / "desktop.log").exists()


def test_replay_returns_input_exit_code_for_missing_video(tmp_path: Path) -> None:
    result = replay_main(
        [
            "--video",
            str(tmp_path / "missing.mp4"),
            "--hyrox-action",
            "lunge",
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    assert result == ExitCode.INPUT_ERROR


def test_session_save_failure_keeps_partial_outputs_and_resets_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = SessionWriter(tmp_path)
    writer.start(
        SessionConfig(
            camera_index=0,
            width=640,
            height=480,
            mirror=True,
            smoothing=0.6,
            model_name="pose.task",
            plot_on_save=False,
        ),
        session_id="partial_session",
    )

    def fail_kinematics(path: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_kinematics_csv", fail_kinematics)
    with pytest.raises(SessionWriteError, match="disk full") as captured:
        writer.stop()

    session_dir = tmp_path / "sessions" / "partial_session"
    metadata = json.loads(
        (session_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert writer.is_active is False
    assert (session_dir / "landmarks.csv").exists()
    assert metadata["write_status"] == "partial"
    assert "disk full" in metadata["recovery_error"]
    assert "landmarks.csv" in captured.value.recovered_files


def test_cleanup_error_is_logged_and_returned(tmp_path: Path) -> None:
    logger = configure_logging(
        app_name="cleanup_test",
        log_dir=tmp_path,
        debug=False,
    )

    def fail_close() -> None:
        raise RuntimeError("close failed")

    error = safe_cleanup(logger, "resource", fail_close)
    for handler in logging.getLogger("pose").handlers:
        handler.flush()

    assert isinstance(error, RuntimeError)
    assert "REC001" in (tmp_path / "cleanup_test.log").read_text(encoding="utf-8")
