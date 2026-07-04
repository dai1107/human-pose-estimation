from __future__ import annotations

from src.fitness.squat.view_metrics import summarize_view_metrics


def test_camera_views_only_expose_allowed_metrics() -> None:
    side = summarize_view_metrics("side")
    front = summarize_view_metrics("front")
    unknown = summarize_view_metrics("unknown")

    assert "pelvis_vertical_displacement_normalized" in side.available_metrics
    assert "knee_lateral_trajectory_proxy" in side.unavailable_or_low_reliability
    assert "pelvis_lateral_drift_proxy" in front.available_metrics
    assert "view_sensitive_metrics" in unknown.unavailable_or_low_reliability

