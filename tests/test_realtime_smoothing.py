from __future__ import annotations

from src.realtime_pose import DrawLandmark, LandmarkSmoother


def _landmarks(count: int = 33) -> list[DrawLandmark]:
    return [DrawLandmark(x=0.0, y=0.0, z=0.0, visibility=1.0, presence=1.0) for _ in range(count)]


def test_pose_smoother_suppresses_body_joint_jump_near_hand_occlusion() -> None:
    knee_index = 25
    smoother = LandmarkSmoother(alpha=0.65)
    first = _landmarks()
    first[knee_index] = DrawLandmark(x=0.50, y=0.50, z=0.0, visibility=1.0, presence=1.0)
    smoother.smooth(first, timestamp_ms=1000)

    second = _landmarks()
    second[knee_index] = DrawLandmark(x=0.80, y=0.50, z=0.0, visibility=1.0, presence=1.0)
    hand_point = DrawLandmark(x=0.80, y=0.50, z=0.0, visibility=1.0, presence=1.0)

    smoothed = smoother.smooth(
        second,
        timestamp_ms=1033,
        occlusion_points=[hand_point],
        occlusion_guard_indices=frozenset({knee_index}),
    )

    assert smoothed[knee_index].x < 0.53


def test_pose_smoother_allows_same_jump_without_occlusion_guard() -> None:
    knee_index = 25
    smoother = LandmarkSmoother(alpha=0.65)
    first = _landmarks()
    first[knee_index] = DrawLandmark(x=0.50, y=0.50, z=0.0, visibility=1.0, presence=1.0)
    smoother.smooth(first, timestamp_ms=1000)

    second = _landmarks()
    second[knee_index] = DrawLandmark(x=0.80, y=0.50, z=0.0, visibility=1.0, presence=1.0)

    smoothed = smoother.smooth(second, timestamp_ms=1033)

    assert smoothed[knee_index].x > 0.65
