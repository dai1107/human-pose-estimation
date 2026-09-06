from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import tools.run_swim_wrist_tracking as swim_tool


def _points(frame: int) -> list[SimpleNamespace]:
    positions = {
        "left_shoulder": (0.30, 0.45),
        "left_elbow": (0.35, 0.50),
        "left_wrist": (0.40 + frame * 0.002, 0.55),
        "right_shoulder": (0.70, 0.45),
        "right_elbow": (0.65, 0.50),
        "right_wrist": (0.60 + frame * 0.002, 0.55),
        "left_hip": (0.40, 0.70),
        "right_hip": (0.60, 0.70),
    }
    return [
        SimpleNamespace(name=name, x=x, y=y, confidence=0.95)
        for name, (x, y) in positions.items()
    ]


class _FakeBackend:
    def __init__(self, *args, **kwargs) -> None:
        self.frame = 0

    def detect(self, frame: np.ndarray, timestamp_ms: int):
        result = SimpleNamespace(success=True, keypoints=_points(self.frame))
        self.frame += 1
        return result

    def close(self) -> None:
        return None


def _video(path: Path, frame_count: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (64, 64),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + index, 32), 4, (255, 255, 255), -1)
        cv2.circle(frame, (44 + index // 2, 32), 4, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_tracking_tool_runs_video_and_writes_auditable_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "swim.avi"
    model = tmp_path / "pose.task"
    output = tmp_path / "output"
    _video(video)
    model.write_bytes(b"fake")
    monkeypatch.setattr(swim_tool, "MediaPipeBackend", _FakeBackend)

    summary, rows, events = swim_tool.run_tracking(
        video,
        model,
        config_path=Path("configs/swim_wrist_tracking.yaml"),
        rotate_clockwise=False,
    )
    paths = swim_tool.write_artifacts(output, summary, rows, events)

    assert summary["rounds_completed"] == [4, 5]
    assert summary["hyrox_rules_changed"] is False
    assert summary["video"]["processed_frame_count"] == 8
    assert summary["round4_identity"]["persistent_track_identity_switch_count"] == 0
    assert summary["round5_trajectory"]["by_side"]["left"]["track_coverage"] == 1.0
    assert all(path.is_file() for path in paths)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["wristband_appearance_used"] is False
    assert "left_track_id" in rows[0]

