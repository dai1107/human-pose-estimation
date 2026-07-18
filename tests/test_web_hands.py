from __future__ import annotations

import numpy as np

from src.backends.base import Keypoint
from src.biomechanics.types import LandmarkPoint
from webui.hands import (
    WebHandOverlay,
    hand_overlay_visible,
    make_hand_detection,
    rtmw_hand_detections,
    serialize_hand_overlay,
)


def _hand_points() -> list[LandmarkPoint]:
    return [
        LandmarkPoint(
            x=0.25 + index * 0.01,
            y=0.35 + index * 0.005,
            z=0.0,
            visibility=0.95,
            presence=0.90,
        )
        for index in range(21)
    ]


def test_web_hand_overlay_serializes_all_five_fingers_without_wrist() -> None:
    detection = make_hand_detection("left", _hand_points())

    points, connections = serialize_hand_overlay({"left": detection})

    names = {str(point["name"]) for point in points}
    assert len(points) == 20
    assert len(connections) == 15
    assert "left_hand_wrist" not in names
    assert "left_hand_thumb_tip" in names
    assert "left_hand_middle_finger_tip" in names
    assert "left_hand_ring_finger_tip" in names
    assert all(point["visibility"] == 0.9 for point in points)


def test_web_hand_overlay_is_rate_limited_and_clears_when_disabled(tmp_path) -> None:
    calls: list[int] = []
    detection = make_hand_detection("right", _hand_points())

    class FakeTracker:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False

        def detect(self, _frame: np.ndarray, *, timestamp_ms: int) -> dict[str, object]:
            calls.append(timestamp_ms)
            return {"right": detection}

        def close(self) -> None:
            self.closed = True

    overlay = WebHandOverlay(
        tmp_path / "unused.task",
        detect_fps=10,
        tracker_factory=FakeTracker,  # type: ignore[arg-type]
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    assert overlay.update(frame, timestamp_ms=1000, enabled=True)
    assert overlay.update(frame, timestamp_ms=1050, enabled=True)
    assert calls == [1000]
    assert overlay.update(frame, timestamp_ms=1100, enabled=True)
    assert calls == [1000, 1100]
    assert overlay.update(frame, timestamp_ms=1150, enabled=False) == {}
    overlay.close()


def test_fingers_are_hidden_for_lower_body_profile() -> None:
    assert hand_overlay_visible("full", True) is True
    assert hand_overlay_visible("upper-body", True) is True
    assert hand_overlay_visible("lower-body", True) is False
    assert hand_overlay_visible("full", False) is False


def test_rtmw_hand_points_use_the_existing_five_finger_overlay() -> None:
    points = tuple(
        Keypoint(
            f"left_hand_{index}",
            0.2 + index * 0.01,
            0.3,
            confidence=0.9,
            source_model="rtmw-wholebody",
        )
        for index in range(21)
    )

    detections = rtmw_hand_detections({"rtmw_hand_keypoints": {"left": points}})
    serialized, connections = serialize_hand_overlay(detections)

    assert set(detections) == {"left"}
    assert len(serialized) == 20
    assert len(connections) == 15
    assert {point["name"] for point in serialized} >= {
        "left_hand_thumb_tip",
        "left_hand_index_finger_tip",
        "left_hand_pinky_tip",
    }
