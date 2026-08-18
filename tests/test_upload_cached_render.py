from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from webui.app import _render_cached_annotated_video


def _create_test_video(path: Path, *, fps: float, frame_count: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder is unavailable")
    try:
        for index in range(frame_count):
            frame = np.full((120, 160, 3), 25 + index * 5, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_second_pass_renders_from_cache_without_pose_inference(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "annotated.mp4"
    fps = 20.0
    frame_count = 6
    _create_test_video(source_path, fps=fps, frame_count=frame_count)
    cached_frames = [
        {
            "frame_index": index,
            "keypoints": [
                {
                    "name": "left_hip",
                    "x": 0.35,
                    "y": 0.30,
                    "z": 0.0,
                    "confidence": 0.95,
                    "source_model": "cached",
                },
                {
                    "name": "left_knee",
                    "x": 0.50,
                    "y": 0.55,
                    "z": 0.0,
                    "confidence": 0.95,
                    "source_model": "cached",
                },
                {
                    "name": "left_ankle",
                    "x": 0.62,
                    "y": 0.82,
                    "z": 0.0,
                    "confidence": 0.95,
                    "source_model": "cached",
                },
            ],
            "connections": ((0, 1), (1, 2)),
            "model_name": "cached",
            "visible_names": frozenset(
                {"left_hip", "left_knee", "left_ankle"}
            ),
            "assessment": {"status": "good", "angles": []},
            "action_state": {
                "action": "lunge",
                "phase": "stand",
                "rep_count": 0,
                "candidate_count": 0,
                "feedback_messages": [],
                "last_rep_decision": {},
                "debug": {"contacts": {}, "foot_events": {}},
            },
        }
        for index in range(frame_count)
    ]

    summary = _render_cached_annotated_video(
        source_path=str(source_path),
        output_path=output_path,
        source_fps=fps,
        cached_frames=cached_frames,
    )

    assert output_path.is_file()
    assert summary["rendered_from_pose_cache"] is True
    assert summary["display_stabilized"] is True
    assert summary["pose_inference_count"] == 0
    assert summary["input_frame_count"] == frame_count
    assert summary["output_frame_count"] == frame_count
    assert summary["output_fps"] == pytest.approx(fps)
    assert summary["duration_difference_ms"] <= summary[
        "duration_tolerance_ms"
    ]
