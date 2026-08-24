from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.skip("WHAM/OpenCap integration is deferred", allow_module_level=True)

from src.offline3d import Offline3DFrame, Offline3DManager, Offline3DResult
from src.offline3d.alignment import AlignmentConfig, align_mediapipe_wham
from src.offline3d.opencap.adapter import build_command, command_template_from_environment
from src.offline3d.opencap.backend import OpenCapBackend
from src.offline3d.opencap.parser import parse_opencap_payload
from src.offline3d.reports import build_advanced_report


def _wham(*timestamps: float) -> Offline3DResult:
    return Offline3DResult(
        backend="wham",
        status="COMPLETED",
        reference_source="WHAM reconstructed 3D",
        coordinate_system="wham_global",
        frames=[
            Offline3DFrame(
                timestamp_ms=value,
                frame_index=100 + index,
                joints_3d={"pelvis": (value / 100.0, 0.0, 0.0)},
                body_translation=[value / 100.0, 0.0, 0.0],
                confidence=0.8 + index * 0.1,
            )
            for index, value in enumerate(timestamps)
        ],
    )


def _mediapipe(timestamp: float, frame_index: int) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp,
        "analysis_timestamp_ms": timestamp + 33.0,
        "raw_keypoints": [{"name": "pelvis", "x": 0.4, "y": 0.5, "z": 0.0}],
        "keypoints": [{"name": "pelvis", "x": 0.41, "y": 0.5, "z": 0.0}],
        "raw_world_keypoints": [{"name": "pelvis", "x": 0.0, "y": 0.0, "z": 0.0}],
        "world_keypoints": [{"name": "pelvis", "x": 0.01, "y": 0.0, "z": 0.0}],
        "three_d_kinematics": {
            "canonical_3d_angles": {"left_knee": 88.0},
            "body_coordinate_system": {
                "canonical_landmarks": [{"name": "pelvis", "x": 0, "y": 0, "z": 0}]
            },
        },
        "angle_observations": [
            {"side": "left", "joint_name": "knee", "rule_angle_deg": 91.0}
        ],
    }


def test_alignment_uses_source_timestamp_not_frame_index() -> None:
    media = [_mediapipe(0.0, 999), _mediapipe(100.0, 1000)]
    aligned = align_mediapipe_wham(media, _wham(0.0, 100.0))

    assert aligned.status == "COMPLETED"
    assert [frame.alignment_method for frame in aligned.frames] == ["EXACT", "EXACT"]
    assert aligned.frames[0].wham["frame_index"] == 100
    assert aligned.frames[0].mediapipe["frame_index"] == 999
    assert aligned.as_dict()["frame_index_alignment_allowed"] is False


def test_alignment_interpolates_wham_only_and_labels_provenance() -> None:
    media = [_mediapipe(50.0, 7)]
    aligned = align_mediapipe_wham(media, _wham(0.0, 100.0))
    frame = aligned.frames[0]

    assert frame.alignment_method == "LINEAR_INTERPOLATION"
    assert frame.wham["joints_3d"]["pelvis"] == pytest.approx([0.5, 0.0, 0.0])
    assert frame.wham["sample_interpolated"] is True
    assert frame.mediapipe["observation_interpolated"] is False
    assert frame.mediapipe["raw_landmarks_2d"][0]["x"] == 0.4
    assert frame.mediapipe["selected_rule_angles"]["left_knee"] == 91.0
    assert "joints_3d" in frame.alignment["interpolated_fields"]


def test_alignment_rejects_large_gaps_and_never_extrapolates() -> None:
    media = [_mediapipe(-10.0, 1), _mediapipe(200.0, 2), _mediapipe(500.0, 3)]
    aligned = align_mediapipe_wham(
        media,
        _wham(0.0, 400.0),
        AlignmentConfig(maximum_interpolation_span_ms=120.0),
    )

    assert all(frame.alignment_method == "UNMATCHED" for frame in aligned.frames)
    assert aligned.statistics["unmatched_count"] == 3


def test_coordinate_and_joint_topology_are_not_silently_converted() -> None:
    aligned = align_mediapipe_wham([_mediapipe(0.0, 1)], _wham(0.0))
    payload = aligned.as_dict()

    assert payload["coordinate_relationship"]["coordinate_transform_applied"] is False
    assert payload["coordinate_relationship"]["direct_position_comparison_allowed"] is False
    compatibility = payload["motion_frames"][0]["alignment"]["joint_compatibility"]
    assert compatibility["topology_conversion_applied"] is False
    assert compatibility["shared_joint_names"] == ["pelvis"]


def test_opencap_parser_preserves_joint_kinematics_and_definition() -> None:
    result = parse_opencap_payload(
        {
            "schema_version": "opencap-export-1",
            "coordinate_system": "opensim_ground",
            "frames": [
                {
                    "source_timestamp_ms": 40.0,
                    "joint_kinematics": {"knee_flexion_l": 84.7},
                    "pelvis_motion": {"pelvis_tx": 0.12},
                    "opensim_ik": {"marker_error_rms": 0.02},
                    "confidence": 0.88,
                }
            ],
        }
    )

    assert result.status == "COMPLETED"
    assert result.frames[0].extra["joint_kinematics"]["knee_flexion_l"] == 84.7
    assert "not MediaPipe three-point angles" in result.angle_definition
    assert result.metadata["absolute_angle_mae_allowed"] is False


def test_unconfigured_opencap_requires_wham_and_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSE_OPENCAP_COMMAND", raising=False)
    monkeypatch.delenv("POSE_OPENCAP_COMMAND_JSON", raising=False)
    wham = _wham(0.0)
    alignment = align_mediapipe_wham([_mediapipe(0.0, 1)], wham)

    result = Offline3DManager.from_environment().refine_opencap(
        "unused.mp4", wham_result=wham, alignment=alignment
    )

    assert result.status == "UNAVAILABLE"
    assert "not configured" in result.warnings[0]


def test_opencap_command_uses_json_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "POSE_OPENCAP_COMMAND_JSON",
        json.dumps([sys.executable, "runner.py", "{input_json}", "{output_json}"]),
    )
    command = build_command(
        command_template_from_environment(),
        video_path=tmp_path / "clip.mp4",
        input_json=tmp_path / "input.json",
        output_json=tmp_path / "output.json",
        output_dir=tmp_path,
    )

    assert command[:2] == [sys.executable, "runner.py"]
    assert str(tmp_path / "input.json") in command
    assert str(tmp_path / "output.json") in command
    assert command[-2:] == ["--video", str(tmp_path / "clip.mp4")]


def test_external_opencap_receives_wham_and_alignment_bundle(tmp_path: Path) -> None:
    runner = tmp_path / "fake_opencap.py"
    runner.write_text(
        "import json, sys\n"
        "bundle=json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "assert bundle['wham']['backend']=='wham'\n"
        "assert bundle['motion_alignment']['alignment_key']=='source_timestamp_ms'\n"
        "json.dump({'frames':[{'timestamp_ms':0,'joint_kinematics':{'knee_flexion_l':80}}]}, open(sys.argv[2], 'w'))\n",
        encoding="utf-8",
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    wham = _wham(0.0)
    alignment = align_mediapipe_wham([_mediapipe(0.0, 1)], wham)
    backend = OpenCapBackend(
        [sys.executable, str(runner), "{input_json}", "{output_json}", "{video_path}"]
    )

    result = backend.analyze_with_reference(
        video,
        wham_result=wham,
        alignment=alignment,
        output_dir=tmp_path / "result",
    )

    assert result.status == "COMPLETED"
    assert result.metadata["initial_pose_source"] == "wham"
    assert result.metadata["alignment_source"] == "source_timestamp_ms"


def test_advanced_report_never_replaces_formal_rule_results() -> None:
    wham = _wham(0.0)
    alignment = align_mediapipe_wham([_mediapipe(0.0, 1)], wham)
    opencap = parse_opencap_payload({"frames": []})
    report = build_advanced_report(
        [_mediapipe(0.0, 1)], wham=wham, alignment=alignment, opencap=opencap
    )

    assert report["formal_rule_replacement_allowed"] is False
    assert report["absolute_joint_angle_mae_allowed"] is False
    assert "lowest_point_timing" in report["comparison_policy"]
    assert report["is_ground_truth"] is False
