from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import webui.app as web_app
from src.backends.base import PoseResult
from webui.offline_fast import (
    AdaptiveOfflineFastScheduler,
    CandidateObservation,
    TimestampSampler,
    build_candidate_windows,
)
from webui.upload_pipeline import UploadInferenceAudit


def test_timestamp_sampler_uses_media_time_for_variable_frame_intervals() -> None:
    sampler = TimestampSampler(10.0)

    selected = [
        timestamp
        for timestamp in (0.0, 17.0, 51.0, 99.0, 101.0, 205.0, 207.0)
        if sampler.sample(timestamp)
    ]

    assert selected == [0.0, 101.0, 205.0]


def test_coarse_observation_opens_and_merges_dense_candidate_window() -> None:
    scheduler = AdaptiveOfflineFastScheduler(
        target_pose_fps=10.0,
        refinement_pose_fps=20.0,
        candidate_margin_ms=200.0,
    )
    selected: list[tuple[int, str]] = []
    for timestamp in range(0, 701, 50):
        decision = scheduler.select(float(timestamp))
        if decision.analyze:
            selected.append(
                (
                    timestamp,
                    "coarse" if decision.coarse else "refinement",
                )
            )
        if decision.coarse:
            phase = "descent" if 200 <= timestamp <= 400 else "stand"
            scheduler.observe_coarse(
                timestamp_ms=float(timestamp),
                phase=phase,
                candidate_count=1 if timestamp >= 500 else 0,
            )

    summary = scheduler.summary(source_frame_count=15)
    assert any(kind == "refinement" for _, kind in selected)
    assert selected == sorted(selected)
    assert summary["coarse_pose_frames"] == 8
    assert summary["refinement_pose_frames"] > 0
    assert summary["pose_frames"] < summary["source_frames"]
    assert summary["refinement_candidate_count"] == 1
    assert summary["discovery_decisions_are_formal"] is False


def test_two_pass_windows_include_margin_and_merge_phase_samples() -> None:
    windows = build_candidate_windows(
        [
            CandidateObservation(0.0, "stand", 0),
            CandidateObservation(200.0, "descent", 0),
            CandidateObservation(400.0, "bottom", 0),
            CandidateObservation(600.0, "stand", 1),
            CandidateObservation(1400.0, "descent", 1),
            CandidateObservation(1800.0, "stand", 2),
        ],
        margin_ms=300.0,
        duration_ms=2200.0,
    )

    assert [(window.start_ms, window.end_ms) for window in windows] == [
        (0.0, 900.0),
        (1100.0, 2100.0),
    ]


def test_sparse_upload_audit_accepts_timestamp_sampled_frames() -> None:
    audit = UploadInferenceAudit(sparse_analysis=True)
    audit.record_model_initialization()
    analyzed = {0: "coarse", 4: "coarse", 6: "refinement", 8: "coarse"}
    for frame_index in range(10):
        audit.record_decoded(frame_index)
        if frame_index in analyzed:
            audit.record_inference(frame_index)
            audit.record_analyzed(
                frame_index,
                pass_name=analyzed[frame_index],
            )

    audit.validate_complete()
    state = audit.as_dict()
    assert state["decoded_frame_count"] == 10
    assert state["pose_frames"] == 4
    assert state["coarse_pose_frames"] == 3
    assert state["refinement_pose_frames"] == 1
    assert state["pose_sampling_ratio"] == pytest.approx(0.4)
    assert state["model_initialization_count"] == 1


def _create_video(path: Path, *, frame_count: int = 60, fps: float = 60.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
    writer.release()


def _wait_for_engine(engine: web_app.PoseStreamEngine) -> dict[str, object]:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        state = engine.snapshot()
        if state["status"] in {"completed", "error"}:
            return state
        time.sleep(0.01)
    raise AssertionError("pose stream engine did not finish")


def test_fast_upload_samples_pose_without_playback_pacing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHandOverlay:
        def __init__(self, _model_path: Path) -> None:
            pass

        def update(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

        def close(self) -> None:
            pass

    class FakeMediaPipeBackend:
        initialization_count = 0
        inference_count = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            type(self).initialization_count += 1

        def detect(
            self,
            _frame: np.ndarray,
            timestamp_ms: int | None = None,
        ) -> PoseResult:
            type(self).inference_count += 1
            return PoseResult(
                keypoints=[],
                connections=(),
                model_name="mediapipe",
                num_keypoints=0,
                success=False,
                inference_time_ms=0.1,
                timestamp_ms=timestamp_ms,
                extra={
                    "performance": {
                        "resize_ms": 0.0,
                        "color_convert_ms": 0.0,
                        "pose_inference_ms": 0.1,
                    }
                },
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(web_app, "MediaPipeBackend", FakeMediaPipeBackend)
    monkeypatch.setattr(web_app, "WebHandOverlay", FakeHandOverlay)
    source = tmp_path / "fast.mp4"
    _create_video(source)
    engine = web_app.PoseStreamEngine(
        tmp_path / "output",
        pose_cache_root=tmp_path / "cache",
    )
    engine.start(
        {
            "source_mode": "upload",
            "source_name": source.name,
            "video_path": str(source),
            "backend": "mediapipe",
            "analysis_mode": "fast",
            "action": "none",
            "camera_view": "side",
            "sensitivity": "medium",
            "mirror": False,
            "landmark_profile": "full",
            "show_fingers": False,
            "manual_floor_points": [],
            "delete_source_after": False,
            "generate_annotated_video": False,
        }
    )

    state = _wait_for_engine(engine)
    assert state["status"] == "completed", state.get("error")
    assert state["offline_fast_enabled"] is True
    assert state["decoded_frame_count"] == 60
    assert 12 <= int(state["pose_frames"]) <= 18
    assert int(state["pose_inference_count"]) == int(state["pose_frames"])
    assert float(state["pose_sampling_ratio"]) < 0.35
    assert state["model_initialization_count"] == 1
    assert FakeMediaPipeBackend.initialization_count == 1
    assert FakeMediaPipeBackend.inference_count == state["pose_frames"]
    performance = state["performance_summary"]
    for field in (
        "decode_frames",
        "pose_frames",
        "pose_sampling_ratio",
        "decode_ms",
        "pose_inference_ms",
        "rule_engine_ms",
        "report_ms",
        "total_processing_ms",
        "processing_fps",
        "real_time_factor",
    ):
        assert field in performance
    assert performance["refinement_candidate_count"] == 0
    assert engine.report()["frames"][0]["angle_observations"] == []


def test_fast_upload_refines_only_candidate_windows_then_replays_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHandOverlay:
        def __init__(self, _model_path: Path) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeBackend:
        initialization_count = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            type(self).initialization_count += 1

        def detect(self, _frame: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
            return PoseResult([], (), "mediapipe", 0, False, 0.1, timestamp_ms=timestamp_ms)

        def close(self) -> None:
            pass

    analyzers: list[object] = []

    class FakeAnalyzer:
        def __init__(self) -> None:
            self.timestamps: list[int] = []

        def set_manual_floor_line(self, *_args: object) -> None:
            pass

        def update(self, _features: object, *, timestamp_ms: int) -> dict[str, object]:
            self.timestamps.append(timestamp_ms)
            active = 200 <= timestamp_ms < 600
            return {
                "phase": "descent" if active else "stand",
                "rep_count": 1 if timestamp_ms >= 600 else 0,
                "candidate_count": 1 if timestamp_ms >= 600 else 0,
                "pose_valid_rep_count": 1 if timestamp_ms >= 600 else 0,
                "no_rep_count": 0,
                "unsure_count": 0,
                "feedback": [],
                "debug": {"raw_phase": "descent" if active else "stand"},
            }

        def attach_view_context(self, state: dict[str, object]) -> dict[str, object]:
            return state

    def fake_create_analyzer(*_args: object, **_kwargs: object) -> FakeAnalyzer:
        analyzer = FakeAnalyzer()
        analyzers.append(analyzer)
        return analyzer

    monkeypatch.setattr(web_app, "MediaPipeBackend", FakeBackend)
    monkeypatch.setattr(web_app, "WebHandOverlay", FakeHandOverlay)
    monkeypatch.setattr(web_app, "create_action_analyzer", fake_create_analyzer)
    source = tmp_path / "candidate.mp4"
    _create_video(source)
    engine = web_app.PoseStreamEngine(tmp_path / "output", pose_cache_root=tmp_path / "cache")
    engine.start(
        {
            "source_mode": "upload",
            "source_name": source.name,
            "video_path": str(source),
            "backend": "mediapipe",
            "analysis_mode": "fast",
            "action": "lunge",
            "camera_view": "side",
            "sensitivity": "medium",
            "mirror": False,
            "landmark_profile": "full",
            "show_fingers": False,
            "manual_floor_points": [],
            "delete_source_after": False,
            "generate_annotated_video": False,
        }
    )

    state = _wait_for_engine(engine)
    assert state["status"] == "completed", state.get("error")
    assert state["refinement_candidate_count"] == 1
    assert int(state["refinement_pose_frames"]) > 0
    assert int(state["pose_frames"]) < int(state["source_frames"])
    assert FakeBackend.initialization_count == 2
    assert state["model_initialization_count"] == 2
    assert int(state["cross_pass_reinference_count"]) > 0
    assert state["single_inference_per_frame"] is False
    assert len(analyzers) == 2
    discovery, formal = analyzers
    assert formal.timestamps == sorted(formal.timestamps)
    assert len(formal.timestamps) > len(discovery.timestamps)
    assert [frame["timestamp_ms"] for frame in engine.report()["frames"]] == sorted(
        frame["timestamp_ms"] for frame in engine.report()["frames"]
    )
    assert state["candidate_count"] == 1
    assert state["pose_valid_rep_count"] == 1
