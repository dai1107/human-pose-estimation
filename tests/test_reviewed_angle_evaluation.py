from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tools.evaluate_manual_angle_report import (
    SOURCE_SPECS,
    build_model_matched_report,
    evaluate_report,
)


def _points(*, world: bool) -> list[dict[str, float | str]]:
    if world:
        coordinates = {
            "left_hip": (0.0, 1.0, 0.0),
            "left_knee": (0.0, 0.0, 0.0),
            "left_ankle": (0.8660254, 0.5, 0.0),
        }
    else:
        coordinates = {
            "left_hip": (0.5, 0.359375, 0.0),
            "left_knee": (0.5, 0.5, 0.0),
            "left_ankle": (0.7165064, 0.4296875, 0.0),
        }
    return [
        {
            "name": name,
            "x": xyz[0],
            "y": xyz[1],
            "z": xyz[2],
            "confidence": 0.95,
            "visibility": 0.95,
            "presence": 0.95,
        }
        for name, xyz in coordinates.items()
    ]


def _write_cache(path: Path, source_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if source_type == "raw":
        payload = {
            "frame_index": 7,
            "raw_native": {
                "image_landmarks": _points(world=False),
                "world_landmarks": _points(world=True),
            },
        }
    else:
        payload = {
            "frame_index": 7,
            "image_normalized_2d": _points(world=False),
            "mp_world_body_3d": _points(world=True),
        }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")


def test_reviewed_report_evaluates_all_cached_2d_and_3d_sources(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "pose_cache" / "record_1"
    for relative_path, source_type in SOURCE_SPECS.values():
        _write_cache(cache_root / relative_path, source_type)
    report = {
        "coordinate_system": "native_video_pixels",
        "angle_definition": "A-B-C projected 2D angle",
        "annotations": [
            {
                "annotation_id": "a1",
                "record_id": "record_1",
                "action": "lunge",
                "frame_index": 7,
                "joint": "left_knee",
                "camera_view": "side",
                "visibility": "high",
                "manual_angle_deg": 60.0,
                "native_frame_size": {"width": 720, "height": 1280},
            }
        ],
    }

    summary, rows = evaluate_report(report, dataset_root=tmp_path)

    assert summary["annotation_count"] == 1
    assert summary["missing_sources"] == []
    assert len(rows) == 1
    for source in SOURCE_SPECS:
        assert rows[0][f"{source}_2d_deg"] == pytest.approx(60.0, abs=0.1)
        assert rows[0][f"{source}_2d_error_deg"] == pytest.approx(0.0, abs=0.1)
        assert rows[0][f"{source}_3d_deg"] == pytest.approx(60.0, abs=0.1)
        assert abs(float(rows[0][f"{source}_uncorrected_2d_deg"]) - 60.0) > 5.0
    assert (
        summary["findings"]["aspect_ratio_correction"]["raw_full"][
            "corrected_mae_deg"
        ]
        == pytest.approx(0.0, abs=0.1)
    )
    round12 = summary["round12_validation"]
    assert round12["overall"]["raw_2d"]["mae_deg"] == pytest.approx(
        0.0, abs=0.1
    )
    assert round12["overall"]["filtered_2d"]["mae_deg"] == pytest.approx(
        0.0, abs=0.1
    )
    assert rows[0]["round12_canonical_3d_deg"] == pytest.approx(
        60.0, abs=0.1
    )
    assert (
        rows[0]["round12_selected_rule_source"]
        == "causal_full_image_landmarks_2d"
    )

    matched = build_model_matched_report(report, rows)
    annotation = matched["annotations"][0]
    assert annotation["model_angle_deg"] == pytest.approx(60.0, abs=0.1)
    assert annotation["comparison_status"] == "matched_raw_full_aspect_corrected"
    assert (
        annotation["three_d_comparison_status"]
        == "projection_consistency_only_no_3d_manual_truth"
    )


def test_reviewed_report_calls_3d_difference_a_projection_gap(
    tmp_path: Path,
) -> None:
    summary, rows = evaluate_report(
        {
            "annotations": [],
            "coordinate_system": "native_video_pixels",
        },
        dataset_root=tmp_path,
    )

    assert rows == []
    assert "not a direct 3D ground-truth error" in summary["three_d_interpretation"]
