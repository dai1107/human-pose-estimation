from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.analyze_existing_angle_errors import (
    CHANNELS,
    analyze_rows,
    build_analysis_rows,
    load_round12_rows,
    main,
)


def _report() -> dict[str, object]:
    return {
        "action": "lunge",
        "camera_view": "side",
        "performance": {"source_fps": 25.0},
        "frames": [
            {
                "frame_index": 4,
                "timestamp_ms": 160.0,
                "action": "lunge",
                "camera_view": "side",
                "angle_observations": [
                    {
                        "joint_name": "knee",
                        "side": "left",
                        "angle_2d_raw_deg": 154.0,
                        "angle_2d_smoothed_deg": 158.0,
                        "angle_3d_raw_deg": 150.0,
                        "angle_3d_smoothed_deg": 145.0,
                        "angle_canonical_3d_deg": 146.0,
                        "rule_angle_deg": 158.0,
                        "landmark_visibility": 0.8,
                    }
                ],
            }
        ],
    }


def test_build_rows_recomputes_all_six_errors_and_dimensions() -> None:
    rows = build_analysis_rows(
        [
            {
                "annotation_id": "a1",
                "video_id": "phone_lunge_001",
                "frame_index": 4,
                "joint": "left_knee",
                "manual_angle_deg": 155.0,
                "event": "full_extension",
            }
        ],
        report=_report(),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "lunge"
    assert row["joint"] == "knee"
    assert row["side"] == "left"
    assert row["camera_view"] == "side"
    assert row["movement_phase"] == "full_extension"
    assert row["angle_range"] == "150–160°"
    assert row["landmark_confidence"] == "high"
    assert row["raw_2d_error_deg"] == pytest.approx(1.0)
    assert row["filtered_2d_error_deg"] == pytest.approx(3.0)
    assert row["raw_3d_error_deg"] == pytest.approx(5.0)
    assert row["filtered_3d_error_deg"] == pytest.approx(10.0)
    assert row["canonical_3d_error_deg"] == pytest.approx(9.0)
    assert row["selected_rule_error_deg"] == pytest.approx(3.0)
    assert row["raw_2d_3d_disagreement_deg"] == pytest.approx(4.0)
    assert row["filtered_2d_3d_disagreement_deg"] == pytest.approx(13.0)
    assert row["2d_3d_disagreement"] == "10–20°"


def test_round12_rows_override_embedded_values_and_preserve_unmatched() -> None:
    annotations = [
        {
            "annotation_id": "a1",
            "record_id": "phone_wall_ball_001",
            "frame_index": 9,
            "joint": "right_knee",
            "manual_angle_deg": 180.0,
            "model_2d_raw_deg": 100.0,
            "event": "standing",
        },
        {
            "annotation_id": "a2",
            "record_id": "phone_wall_ball_001",
            "frame_index": 10,
            "joint": "torso",
            "manual_angle_deg": 175.0,
            "model_2d_raw_deg": -170.0,
        },
    ]
    round12 = [
        {
            "annotation_id": "a1",
            "action": "wall_ball",
            "camera_view": "front_view",
            "visibility": "medium",
            "round12_raw_2d_deg": "179",
            "round12_filtered_2d_deg": "178",
            "round12_raw_3d_deg": "177",
            "round12_filtered_3d_deg": "176",
            "round12_canonical_3d_deg": "176",
            "round12_selected_rule_deg": "178",
        }
    ]

    rows = build_analysis_rows(annotations, round12_rows=round12)

    assert rows[0]["raw_2d_deg"] == pytest.approx(179.0)
    assert rows[0]["raw_2d_error_deg"] == pytest.approx(1.0)
    assert rows[0]["angle_range"] == "170–180°"
    assert rows[0]["camera_view"] == "front"
    assert rows[0]["landmark_confidence"] == "medium"
    assert rows[1]["side"] == "center"
    assert rows[1]["raw_2d_deg"] == pytest.approx(170.0)
    assert rows[1]["raw_2d_error_deg"] == pytest.approx(5.0)


def test_analysis_contains_all_required_dimensions_and_interactions() -> None:
    rows = []
    for index, (manual, raw, filtered) in enumerate(
        ((155.0, 150.0, 154.0), (165.0, 160.0, 170.0), (175.0, 170.0, 178.0))
    ):
        rows.extend(
            build_analysis_rows(
                [
                    {
                        "annotation_id": f"a{index}",
                        "record_id": "phone_lunge_001",
                        "frame_index": index,
                        "action": "lunge",
                        "joint": "left_knee",
                        "camera_view": "side",
                        "event": "full_extension",
                        "visibility": "high",
                        "manual_angle_deg": manual,
                        "round12_raw_2d_deg": raw,
                        "round12_filtered_2d_deg": filtered,
                        "round12_raw_3d_deg": manual + 10.0,
                        "round12_filtered_3d_deg": manual + 12.0,
                        "round12_canonical_3d_deg": manual + 12.0,
                        "round12_selected_rule_deg": filtered,
                    }
                ]
            )
        )

    analysis = analyze_rows(rows, minimum_finding_count=1)

    assert analysis["annotation_count"] == 3
    assert analysis["matched_annotation_count"] == 3
    assert set(analysis["by_dimension"]) == {
        "action",
        "joint",
        "side",
        "camera_view",
        "movement_phase",
        "angle_range",
        "landmark_confidence",
        "2d_3d_disagreement",
    }
    assert set(analysis["interactions"]) == {
        "action_joint_angle_range",
        "action_joint_movement_phase",
    }
    overall = analysis["overall"]
    assert set(overall["channels"]) == set(CHANNELS)
    assert overall["channels"]["raw_2d"]["mae_deg"] == pytest.approx(5.0)
    assert overall["paired_raw_filtered_2d"]["filtered_better_count"] == 2
    assert analysis["interpretation"]["formal_rules_changed"] is False


def test_cli_reads_round12_csv_and_writes_three_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    annotations_path = tmp_path / "manual_angles.json"
    annotations_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "annotation_id": "a1",
                        "record_id": "phone_rowing_001",
                        "frame_index": 1,
                        "action": "rowing",
                        "joint": "left_knee",
                        "camera_view": "side",
                        "event": "catch",
                        "visibility": "high",
                        "manual_angle_deg": 70.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    round12_path = tmp_path / "reviewed_angle_rows.csv"
    fields = [
        "annotation_id",
        "round12_raw_2d_deg",
        "round12_filtered_2d_deg",
        "round12_raw_3d_deg",
        "round12_filtered_3d_deg",
        "round12_canonical_3d_deg",
        "round12_selected_rule_deg",
    ]
    with round12_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "annotation_id": "a1",
                "round12_raw_2d_deg": 72,
                "round12_filtered_2d_deg": 71,
                "round12_raw_3d_deg": 75,
                "round12_filtered_3d_deg": 74,
                "round12_canonical_3d_deg": 74,
                "round12_selected_rule_deg": 71,
            }
        )
    output_dir = tmp_path / "analysis"

    assert (
        main(
            [
                "--annotations",
                str(annotations_path),
                "--round12",
                str(round12_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["annotation_count"] == 1
    assert payload["formal_rules_changed"] is False
    assert (output_dir / "angle_error_analysis.json").is_file()
    assert (output_dir / "angle_error_rows.csv").is_file()
    assert (output_dir / "ANGLE_ERROR_ANALYSIS.md").is_file()


def test_load_round12_rows_accepts_directory(tmp_path: Path) -> None:
    path = tmp_path / "reviewed_angle_rows.csv"
    path.write_text("annotation_id,round12_raw_2d_deg\na1,90\n", encoding="utf-8")

    assert load_round12_rows(tmp_path) == [
        {"annotation_id": "a1", "round12_raw_2d_deg": "90"}
    ]
