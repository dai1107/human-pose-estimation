from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_swim_round6_experiment import (
    MODE_NAMES,
    aggregate_experiment,
    load_anchor_record,
    write_artifacts,
)


def _metrics(
    *,
    switches: int = 2,
    correct: int = 8,
    anchors: int = 10,
    coverage: float = 0.90,
    jitter: float | None = 0.05,
) -> dict[str, object]:
    return {
        "available": True,
        "identity_switch_proxy_count": switches,
        "anchor_count": anchors,
        "anchor_target_available_count": anchors,
        "anchor_identity_correct_count": correct,
        "anchor_identity_correct_rate": correct / max(1, anchors),
        "anchor_target_error_mae_px": 12.0,
        "mean_track_coverage": coverage,
        "mean_trajectory_jitter_body": jitter,
    }


def _record(*, cotracker: dict[str, object] | None = None) -> dict[str, object]:
    modes = {mode: _metrics() for mode in MODE_NAMES}
    if cotracker is None:
        unavailable = {"available": False, "reason": "checkpoint_missing"}
        modes["pose_cotracker"] = unavailable
        modes["pose_cotracker_wristband"] = unavailable
    else:
        modes["pose_cotracker"] = cotracker
        modes["pose_cotracker_wristband"] = cotracker
    return {
        "anchor_count": 10,
        "modes": modes,
        "supporting_modes": {"pose_lk_wristband": _metrics()},
    }


def test_load_anchor_record_uses_declared_side_and_points(tmp_path: Path) -> None:
    video = tmp_path / "marked.mp4"
    video.write_bytes(b"placeholder")
    anchors = tmp_path / "anchors.json"
    anchors.write_text(
        json.dumps({"side": "right", "anchors": [{"frame": 7, "x": 11, "y": 13}]}),
        encoding="utf-8",
    )

    record = load_anchor_record(video, anchors)

    assert record["target_side"] == "right"
    assert record["anchors"] == [{"frame": 7, "x": 11.0, "y": 13.0}]


def test_unavailable_cotracker_is_not_ranked_and_lk_remains_default() -> None:
    summary = aggregate_experiment(
        [_record(), _record()],
        cotracker_availability={"available": False, "reason": "checkpoint_missing"},
    )

    assert summary["modes"]["pose_cotracker"]["available"] is False
    assert summary["recommended_experimental_default"] == "pose_lk"
    assert summary["formal_default_changed"] is False
    assert summary["wristband_default_allowed"] is False


def test_cotracker_requires_all_gates_and_complete_video_coverage() -> None:
    better = _metrics(switches=1, correct=9, coverage=0.95, jitter=0.04)
    passing = aggregate_experiment(
        [_record(cotracker=better), _record(cotracker=better)],
        cotracker_availability={"available": True, "reason": "accepted"},
    )
    incomplete = aggregate_experiment(
        [_record(cotracker=better), _record()],
        cotracker_availability={"available": True, "reason": "accepted"},
    )

    assert passing["recommended_experimental_default"] == "pose_cotracker"
    assert passing["formal_default_changed"] is False
    assert incomplete["recommended_experimental_default"] == "pose_lk"
    assert incomplete["modes"]["pose_cotracker"]["complete_video_coverage"] is False


def test_empty_jitter_does_not_crash_aggregation_or_artifact_write(
    tmp_path: Path,
) -> None:
    record = _record()
    record["modes"]["pose_lk"] = _metrics(jitter=None)
    summary = aggregate_experiment(
        [record],
        cotracker_availability={"available": False, "reason": "checkpoint_missing"},
    )

    paths = write_artifacts(tmp_path, summary, [])

    assert summary["modes"]["pose_lk"]["mean_trajectory_jitter_body"] is None
    assert all(path.is_file() for path in paths)
    assert "pose_lk" in paths[2].read_text(encoding="utf-8")


def test_load_anchor_record_rejects_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_anchor_record(tmp_path / "missing.mp4", tmp_path / "missing.json")
