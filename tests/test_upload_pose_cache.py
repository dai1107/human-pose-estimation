from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.backends.base import Keypoint, PoseResult
from src.utils.keypoint_schema import MEDIAPIPE_CONNECTIONS
from webui import app as web_app
from webui.pose_cache import (
    CachedPoseBackend,
    PoseCacheIdentity,
    PoseCacheWriter,
    load_pose_cache,
)


def _identity(
    tmp_path: Path,
    *,
    segmentation_enabled: bool = False,
) -> PoseCacheIdentity:
    video = tmp_path / "input.mp4"
    model = tmp_path / "pose_landmarker_full.task"
    video.write_bytes(b"same uploaded video")
    model.write_bytes(b"same pose model")
    return PoseCacheIdentity.create(
        video_path=video,
        backend="mediapipe",
        model_type="pose_landmarker_full",
        model_path=model,
        inference_width=1920,
        inference_height=1080,
        segmentation_enabled=segmentation_enabled,
        pose_config={
            "running_mode": "video",
            "output_segmentation_masks": segmentation_enabled,
        },
        source_width=1920,
        source_height=1080,
        source_fps=30.0,
        source_frame_count=2,
    )


def _result(timestamp_ms: int, offset: float) -> PoseResult:
    image = [
        Keypoint(
            name="left_hip",
            x=0.4 + offset,
            y=0.3,
            z=-0.01,
            confidence=0.8,
            visibility=0.9,
            presence=0.8,
            source_model="mediapipe",
        ),
        Keypoint(
            name="left_knee",
            x=0.45 + offset,
            y=0.55,
            z=-0.02,
            confidence=0.75,
            visibility=0.85,
            presence=0.75,
            source_model="mediapipe",
        ),
    ]
    world = [
        Keypoint(
            name=point.name,
            x=point.x,
            y=point.y,
            z=point.z,
            confidence=point.confidence,
            visibility=point.visibility,
            presence=point.presence,
            source_model="mediapipe-world",
        )
        for point in image
    ]
    return PoseResult(
        keypoints=image,
        connections=MEDIAPIPE_CONNECTIONS,
        model_name="mediapipe",
        num_keypoints=len(image),
        success=True,
        inference_time_ms=12.0,
        timestamp_ms=timestamp_ms,
        extra={"world_keypoints": world},
    )


def test_pose_cache_persists_required_arrays_and_replays_without_inference(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path)
    writer = PoseCacheWriter(identity)
    writer.append(frame_index=0, timestamp_ms=0.0, result=_result(33, 0.0))
    writer.append(
        frame_index=1,
        timestamp_ms=1000.0 / 30.0,
        result=_result(67, 0.01),
    )

    landmarks_path, metadata_path = writer.write(tmp_path / "cache")
    loaded = load_pose_cache(tmp_path / "cache", identity)

    assert loaded is not None
    assert loaded.frame_count == 2
    assert landmarks_path.name == "pose_landmarks.npz"
    assert metadata_path.name == "pose_metadata.json"
    assert landmarks_path.parent.name == identity.video_sha256
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["video_sha256"] == identity.video_sha256
    assert metadata["model_type"] == "pose_landmarker_full"
    assert metadata["model_file_sha256"] == identity.model_file_sha256
    assert metadata["inference_resolution"] == {
        "width": 1920,
        "height": 1080,
    }
    assert metadata["segmentation_enabled"] is False
    assert metadata["pose_config_sha256"] == identity.pose_config_sha256
    with np.load(landmarks_path, allow_pickle=False) as archive:
        assert {
            "frame_index",
            "timestamp_ms",
            "image_landmarks",
            "world_landmarks",
            "visibility",
            "presence",
            "segmentation_available",
        }.issubset(archive.files)

    backend = CachedPoseBackend(loaded)
    first = backend.detect(np.empty((1, 1, 3)), timestamp_ms=33)
    second = backend.detect(np.empty((1, 1, 3)), timestamp_ms=67)
    assert first.extra["pose_cache_hit"] is True
    assert first.inference_time_ms == 0.0
    assert first.timestamp_ms == 33
    assert second.timestamp_ms == 67
    assert second.keypoints[23].name == "left_hip"
    assert second.keypoints[23].x == np.float32(0.41)


def test_pose_cache_key_invalidates_when_pose_configuration_changes(
    tmp_path: Path,
) -> None:
    identity = _identity(tmp_path, segmentation_enabled=False)
    writer = PoseCacheWriter(identity)
    writer.append(frame_index=0, timestamp_ms=0.0, result=_result(33, 0.0))
    writer.append(
        frame_index=1,
        timestamp_ms=1000.0 / 30.0,
        result=_result(67, 0.01),
    )
    writer.write(tmp_path / "cache")

    changed = _identity(tmp_path, segmentation_enabled=True)

    assert changed.video_sha256 == identity.video_sha256
    assert changed.cache_key != identity.cache_key
    assert load_pose_cache(tmp_path / "cache", changed) is None


def _create_video(path: Path, *, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder is unavailable")
    try:
        for index in range(frame_count):
            writer.write(
                np.full((48, 64, 3), 30 + index * 10, dtype=np.uint8)
            )
    finally:
        writer.release()


def _wait_for_engine(engine: web_app.PoseStreamEngine) -> dict[str, object]:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        state = engine.snapshot()
        if state["status"] in {"completed", "error"}:
            return state
        time.sleep(0.01)
    raise AssertionError("pose stream engine did not finish")


def test_second_upload_analysis_reuses_disk_landmarks_cache(
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
            offset = type(self).inference_count * 0.001
            return _result(int(timestamp_ms or 0), offset)

        def close(self) -> None:
            pass

    monkeypatch.setattr(web_app, "MediaPipeBackend", FakeMediaPipeBackend)
    monkeypatch.setattr(web_app, "WebHandOverlay", FakeHandOverlay)
    source = tmp_path / "source.mp4"
    _create_video(source)
    cache_root = tmp_path / "cache"
    config = {
        "source_mode": "upload",
        "source_name": "source.mp4",
        "video_path": str(source),
        "backend": "mediapipe",
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

    first = web_app.PoseStreamEngine(
        tmp_path / "first-output",
        pose_cache_root=cache_root,
    )
    first.start(config)
    first_state = _wait_for_engine(first)

    assert first_state["status"] == "completed", first_state.get("error")
    assert first_state["pose_cache_hit"] is False
    assert first_state["pose_inference_count"] == 3
    assert first_state["model_initialization_count"] == 1
    assert Path(str(first_state["pose_cache_path"])).is_file()
    assert Path(str(first_state["pose_cache_metadata_path"])).is_file()
    first_report = first.report()
    assert first_report["frames"][0]["angle_sources"][
        "rule_angle_source"
    ] == "image_landmarks_analysis_smoothed"

    second = web_app.PoseStreamEngine(
        tmp_path / "second-output",
        pose_cache_root=cache_root,
    )
    second.start({**config, "sensitivity": "high"})
    second_state = _wait_for_engine(second)

    assert second_state["status"] == "completed", second_state.get("error")
    assert second_state["pose_cache_hit"] is True
    assert second_state["backend"] == "mediapipe-pose-cache"
    assert second_state["pose_inference_count"] == 0
    assert second_state["model_initialization_count"] == 0
    assert second_state["analyzed_frame_count"] == 3
    assert FakeMediaPipeBackend.initialization_count == 1
    assert FakeMediaPipeBackend.inference_count == 3

    cached_landmarks = Path(str(second_state["pose_cache_path"]))
    cached_metadata = Path(str(second_state["pose_cache_metadata_path"]))
    second.delete_pose_cache()
    assert not cached_landmarks.exists()
    assert not cached_metadata.exists()
