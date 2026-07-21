from __future__ import annotations

from pathlib import Path

from src.backends.base import Keypoint, PoseResult
from src.utils.keypoint_schema import MEDIAPIPE_CONNECTIONS
from webui.sample_cache import (
    build_cache_payload,
    expected_source_backend,
    load_sample_pose_backend,
    serialize_hand_detections,
    serialize_pose_result,
    write_cache_payload,
)
from webui.hands import make_hand_detection
from src.biomechanics.types import LandmarkPoint


def pose() -> PoseResult:
    return PoseResult(
        keypoints=[
            Keypoint(
                "left_knee",
                0.4,
                0.6,
                z=-0.1,
                confidence=0.9,
                source_model="mediapipe",
                visibility=0.95,
                presence=0.9,
            )
        ],
        connections=MEDIAPIPE_CONNECTIONS,
        model_name="mediapipe",
        num_keypoints=1,
        success=True,
        inference_time_ms=14.5,
        bbox=(0.3, 0.2, 0.7, 0.9),
        timestamp_ms=33,
    )


def test_sample_pose_cache_round_trip_has_zero_runtime_inference(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fixed-sample-video")
    cache = tmp_path / "rowing.json.gz"
    payload = build_cache_payload(
        action="rowing",
        video_path=video,
        source_backend="mediapipe",
        fps=30.0,
        width=640,
        height=480,
        connections=MEDIAPIPE_CONNECTIONS,
        frames=[serialize_pose_result(pose())],
    )
    write_cache_payload(payload, cache)

    backend = load_sample_pose_backend(
        action="rowing",
        video_path=video,
        total_frames=1,
        path=cache,
    )

    assert backend is not None
    result = backend.detect(object(), timestamp_ms=100)
    assert result.inference_time_ms == 0.0
    assert result.timestamp_ms == 100
    assert result.keypoints[0].name == "left_knee"
    assert result.extra["cached_source_inference_ms"] == 14.5


def test_sample_pose_cache_invalidates_when_video_changes(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"first-version")
    cache = tmp_path / "rowing.json.gz"
    payload = build_cache_payload(
        action="rowing",
        video_path=video,
        source_backend="mediapipe",
        fps=30.0,
        width=640,
        height=480,
        connections=MEDIAPIPE_CONNECTIONS,
        frames=[serialize_pose_result(pose())],
    )
    write_cache_payload(payload, cache)
    video.write_bytes(b"changed-version")

    assert load_sample_pose_backend(
        action="rowing",
        video_path=video,
        total_frames=1,
        path=cache,
    ) is None


def test_lunge_cache_records_the_identity_tracking_source() -> None:
    assert expected_source_backend("lunge") == "yolo-guided-mediapipe"
    assert expected_source_backend("rowing") == "mediapipe"


def test_cached_hand_landmarks_are_reconstructed_without_runtime_inference(
    tmp_path: Path,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"sample-with-hands")
    cache = tmp_path / "rowing.json.gz"
    frame = serialize_pose_result(pose())
    frame["hands"] = serialize_hand_detections(
        {
            "left": make_hand_detection(
                "left",
                [LandmarkPoint(0.5, 0.5, 0.0, 0.9, 0.9) for _ in range(21)],
                score=0.9,
            )
        }
    )
    payload = build_cache_payload(
        action="rowing",
        video_path=video,
        source_backend="mediapipe",
        fps=30.0,
        width=640,
        height=480,
        connections=MEDIAPIPE_CONNECTIONS,
        frames=[frame],
    )
    write_cache_payload(payload, cache)
    backend = load_sample_pose_backend(
        action="rowing", video_path=video, total_frames=1, path=cache
    )

    assert backend is not None
    result = backend.detect(object(), timestamp_ms=33)
    cached_hands = result.extra["cached_hand_detections"]
    assert len(cached_hands["left"].landmarks) == 21
