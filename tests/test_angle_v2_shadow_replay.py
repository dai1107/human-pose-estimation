from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.angle_v2 import AngleV2Config
from tools.run_angle_v2_shadow import compare_annotations, replay_angle_v2


def test_replay_reports_quality_gates_and_never_applies_formal_rules() -> None:
    record_data = {
        "phone_lunge_001": {
            "action": "lunge",
            "camera_view": "side",
            "curves": {
                "left_knee": [
                    {
                        "frame_index": 0,
                        "timestamp_ms": 0.0,
                        "raw_2d_angle_deg": 120.0,
                        "raw_3d_angle_deg": 121.0,
                        "confidence": 0.9,
                        "quality_reasons": [],
                    },
                    {
                        "frame_index": 1,
                        "timestamp_ms": 40.0,
                        "raw_2d_angle_deg": 121.0,
                        "raw_3d_angle_deg": 122.0,
                        "confidence": 0.9,
                        "quality_reasons": ["bone_length_jump"],
                    },
                ]
            },
        }
    }

    summary, lookup, _endpoints = replay_angle_v2(
        record_data, config=AngleV2Config()
    )

    assert summary["record_count"] == 1
    assert summary["totals"]["sample_count"] == 2
    assert summary["totals"]["bone_length_rejection_count"] == 1
    assert lookup[("phone_lunge_001", "left_knee", 0)]["shadow_only"] is True
    assert lookup[("phone_lunge_001", "left_knee", 1)]["angle_valid"] is False


def test_annotation_comparison_filters_wrong_direction_and_pairs_same_rows(
    tmp_path: Path,
) -> None:
    annotations = [
        {
            "annotation_id": "a1",
            "record_id": "phone_wall_ball_001",
            "action": "wall_ball",
            "frame_index": 10,
            "joint": "left_knee",
            "event": "full_extension",
            "manual_angle_deg": 170.0,
        },
        {
            "annotation_id": "a2",
            "record_id": "phone_wall_ball_001",
            "action": "wall_ball",
            "frame_index": 20,
            "joint": "right_knee",
            "event": "lowest_point",
            "manual_angle_deg": 90.0,
        },
    ]
    lookup = {
        ("phone_wall_ball_001", "left_knee", 10): {
            "raw_2d_angle_deg": 158.0,
            "filtered_2d_angle_deg": 165.0,
            "angle_valid": True,
            "evidence_valid": True,
            "reason_codes": [],
        },
        ("phone_wall_ball_001", "right_knee", 20): {
            "raw_2d_angle_deg": 92.0,
            "filtered_2d_angle_deg": None,
            "angle_valid": False,
            "evidence_valid": False,
            "reason_codes": ["BONE_LENGTH_INCONSISTENT"],
        },
    }
    endpoints = [
        {
            "record_id": "phone_wall_ball_001",
            "joint": "left_knee",
            "kind": "maximum",
            "frame_index": 9,
            "raw_extremum_angle_deg": 88.0,
        },
        {
            "record_id": "phone_wall_ball_001",
            "joint": "left_knee",
            "kind": "maximum",
            "frame_index": 12,
            "raw_extremum_angle_deg": 169.0,
        },
    ]
    round12 = tmp_path / "reviewed_angle_rows.csv"
    with round12.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("annotation_id", "round12_filtered_2d_deg"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {"annotation_id": "a1", "round12_filtered_2d_deg": 166.0},
                {"annotation_id": "a2", "round12_filtered_2d_deg": 91.0},
            ]
        )

    summary, rows = compare_annotations(
        annotations, lookup, endpoints, round12_rows=round12
    )

    assert rows[0]["angle_v2_rule_event_angle_deg"] == pytest.approx(169.0)
    assert rows[0]["angle_v2_event_frame"] == 12
    assert summary["paired_available_comparison"]["count"] == 1
    assert summary["paired_available_comparison"]["raw_2d"]["mae_deg"] == pytest.approx(12.0)
    assert summary["angle_v2_filtered_rejections"] == {
        "count": 1,
        "reason_counts": {"BONE_LENGTH_INCONSISTENT": 1},
    }

