from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.angle_validation import (
    append_annotation,
    build_manual_annotation,
    compare_manual_annotations,
    estimate_curve_lag,
    export_angle_curves,
    load_annotations,
    normalize_joint_name,
    write_comparison_artifacts,
)


def _report() -> dict[str, object]:
    raw = [170.0, 160.0, 140.0, 100.0, 80.0, 100.0, 140.0, 160.0, 170.0, 170.0]
    smooth: list[float | None] = [None, None, *raw[:-2]]
    frames = []
    for index, (raw_value, smooth_value) in enumerate(
        zip(raw, smooth, strict=True)
    ):
        frames.append(
            {
                "frame_index": index,
                "timestamp_ms": index * 40.0,
                "angle_observations": [
                    {
                        "frame_index": index,
                        "timestamp_ms": index * 40.0,
                        "joint_name": "knee",
                        "side": "left",
                        "angle_2d_raw_deg": raw_value,
                        "angle_2d_smoothed_deg": smooth_value,
                        "angle_3d_raw_deg": raw_value + 3.0,
                        "angle_3d_smoothed_deg": (
                            None
                            if smooth_value is None
                            else smooth_value + 2.0
                        ),
                        "rule_angle_deg": smooth_value,
                        "display_angle_deg": smooth_value,
                        "drawn_landmarks_angle_deg": smooth_value,
                        "landmark_visibility": 0.92,
                        "geometry_valid": smooth_value is not None,
                    }
                ],
            }
        )
    return {
        "performance": {"source_fps": 25.0},
        "frames": frames,
    }


def _create_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder is unavailable")
    try:
        for index in range(10):
            writer.write(
                np.full((120, 160, 3), 20 + index, dtype=np.uint8)
            )
    finally:
        writer.release()


def test_noninteractive_manual_annotation_uses_shared_angle_math_and_report(
    tmp_path: Path,
) -> None:
    video = tmp_path / "phone_lunge_001.mp4"
    _create_video(video)

    annotation = build_manual_annotation(
        video_path=video,
        frame_index=4,
        joint="left_knee",
        camera_view="side",
        points=((80, 20), (80, 60), (120, 60)),
        report=_report(),
        event="lowest_point",
        annotator="reviewer-a",
    )

    assert annotation["video_id"] == "phone_lunge_001"
    assert annotation["timestamp_ms"] == 160.0
    assert annotation["joint"] == "left_knee"
    assert annotation["manual_points"] == {
        "hip": [80.0, 20.0],
        "knee": [80.0, 60.0],
        "ankle": [120.0, 60.0],
    }
    assert annotation["manual_angle_deg"] == pytest.approx(90.0)
    assert annotation["model_2d_raw_deg"] == 80.0
    assert annotation["model_2d_smoothed_deg"] == 140.0
    assert annotation["model_3d_raw_deg"] == 83.0
    assert annotation["event"] == "lowest_point"

    output = tmp_path / "manual_angles.json"
    append_annotation(output, annotation)
    append_annotation(output, {**annotation, "manual_angle_deg": 91.0})
    loaded = load_annotations(output)
    assert len(loaded) == 1
    assert loaded[0]["manual_angle_deg"] == 91.0


def test_comparison_report_measures_errors_events_and_two_frame_smoothing_lag(
    tmp_path: Path,
) -> None:
    annotations = [
        {
            "video_id": "phone_lunge_001",
            "frame_index": 3,
            "joint": "left_knee",
            "camera_view": "side",
            "manual_angle_deg": 102.0,
            "landmark_visibility": 0.92,
            "event": "",
        },
        {
            "video_id": "phone_lunge_001",
            "frame_index": 4,
            "joint": "left_knee",
            "camera_view": "side",
            "manual_angle_deg": 82.0,
            "landmark_visibility": 0.92,
            "event": "lowest_point",
        },
    ]

    summary, rows = compare_manual_annotations(
        annotations,
        report=_report(),
        max_lag_frames=4,
    )

    raw_stats = summary["overall"]["2d_raw"]
    assert raw_stats["mae_deg"] == pytest.approx(2.0)
    assert raw_stats["median_absolute_error_deg"] == pytest.approx(2.0)
    assert summary["side_high_visibility"]["count"] == 2
    latency = summary["curve_latency"]["left_knee"]
    assert latency["raw_smoothed_lag_frames"] == 2
    assert latency["raw_smoothed_lag_ms"] == pytest.approx(80.0)
    assert latency["lowest_point_offset_frames"] == 2
    assert summary["event_offsets"] == [
        {
            "joint": "left_knee",
            "event": "lowest_point",
            "manual_event_frame": 4,
            "program_event_frame": 6,
            "offset_frames": 2,
        }
    ]
    assert rows[0]["error_2d_raw_deg"] == pytest.approx(2.0)

    summary_path, rows_path = write_comparison_artifacts(
        tmp_path / "comparison",
        summary,
        rows,
    )
    assert json.loads(summary_path.read_text(encoding="utf-8"))[
        "annotation_count"
    ] == 2
    assert rows_path.is_file()


def test_curve_export_contains_raw_smoothed_3d_and_rule_angles(
    tmp_path: Path,
) -> None:
    path = export_angle_curves(
        _report(),
        tmp_path / "left_knee.csv",
        joint="left_knee",
    )
    text = path.read_text(encoding="utf-8-sig")

    assert "angle_2d_raw_deg" in text
    assert "angle_2d_smoothed_deg" in text
    assert "angle_3d_raw_deg" in text
    assert "angle_3d_smoothed_deg" in text
    assert "rule_angle_deg" in text


def test_lag_estimator_rejects_mismatched_curve_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        estimate_curve_lag([1.0], [1.0, 2.0])


def test_manual_validation_supports_center_torso_projection() -> None:
    assert normalize_joint_name("torso_angle") == "torso"
