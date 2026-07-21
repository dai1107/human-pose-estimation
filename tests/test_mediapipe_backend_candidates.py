from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

import src.backends.mediapipe_backend as mediapipe_module
from src.backends.mediapipe_backend import MediaPipeBackend


class FakeImage:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeLandmarker:
    def detect_for_video(self, image: object, timestamp_ms: int) -> object:
        del image, timestamp_ms

        def person(x: float) -> list[object]:
            return [
                SimpleNamespace(
                    x=x,
                    y=0.5,
                    z=0.0,
                    visibility=0.9,
                    presence=0.8,
                )
                for _ in range(33)
            ]

        return SimpleNamespace(
            pose_landmarks=[person(0.25), person(0.75)],
            pose_world_landmarks=[person(0.05), person(0.15)],
            segmentation_masks=[],
        )


def test_detect_exposes_all_pose_candidates_while_returning_first_for_compatibility(
    monkeypatch: Any,
) -> None:
    fake_mp = SimpleNamespace(
        Image=FakeImage,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    monkeypatch.setattr(mediapipe_module, "mp", fake_mp)
    backend = object.__new__(MediaPipeBackend)
    backend._landmarker = FakeLandmarker()
    backend._last_timestamp_ms = -1

    result = backend.detect(
        np.zeros((16, 16, 3), dtype=np.uint8),
        timestamp_ms=10,
    )
    candidates = result.extra["pose_candidates"]

    assert result.success
    assert result.keypoints[0].x == 0.25
    assert len(candidates) == 2
    assert candidates[1][0].x == 0.75
    assert result.extra["world_landmarks_available"] is True
    assert result.extra["world_keypoints"][0].x == 0.05
    assert result.extra["world_keypoints"][0].source_model == "mediapipe-world"
