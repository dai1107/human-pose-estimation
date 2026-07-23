from __future__ import annotations

import pytest

from src.latency_audit import (
    derive_desktop_latencies,
    derive_web_latencies,
    external_sensor_to_photon_ms,
    summarize_latency_samples,
)


def test_web_latency_derivation_keeps_browser_clocks_separate() -> None:
    result = derive_web_latencies({
        "camera_frame_presented_ms": 100.0,
        "socket_send_ms": 106.0,
        "client_result_receive_ms": 126.0,
        "pose_render_start_ms": 128.0,
        "pose_render_end_ms": 130.0,
        "expected_display_time_ms": 133.0,
        "video_frame_presented_at_render_ms": 120.0,
        "render_loop_p95_ms": 4.5,
        "canvas_draw_p95_ms": 3.0,
        "dom_update_p95_ms": 1.5,
        "main_thread_long_task_count": 2,
        "main_thread_long_task_duration_ms": 110,
        "long_task_render_count": 1,
        "long_task_dom_update_count": 1,
    })
    assert result == {
        "capture_to_submit_ms": 6.0,
        "submit_to_result_ms": 20.0,
        "result_to_render_ms": 2.0,
        "render_to_expected_display_ms": 3.0,
        "pose_age_at_render_ms": 28.0,
        "video_frame_age_at_render_ms": 8.0,
        "pose_video_age_difference_ms": 20.0,
        "render_time_ms": 2.0,
        "render_loop_p95_ms": 4.5,
        "canvas_draw_p95_ms": 3.0,
        "dom_update_p95_ms": 1.5,
        "main_thread_long_task_count": 2.0,
        "main_thread_long_task_total_ms": 110.0,
        "long_task_render_count": 1.0,
        "long_task_dom_update_count": 1.0,
        "long_task_frame_copy_count": None,
        "long_task_encode_count": None,
        "long_task_pose_transfer_count": None,
        "long_task_other_count": None,
    }


def test_desktop_latency_derivation_uses_pose_and_current_frame_identity() -> None:
    result = derive_desktop_latencies({
        "capture_read_end_ns": 20_000_000,
        "pose_capture_timestamp_ns": 10_000_000,
        "inference_start_ns": 21_000_000,
        "inference_end_ns": 31_000_000,
        "draw_start_ns": 35_000_000,
        "draw_end_ns": 37_000_000,
        "imshow_return_ns": 38_000_000,
    })
    assert result["pose_age_at_render_ms"] == 25.0
    assert result["video_frame_age_at_render_ms"] == 15.0
    assert result["pose_video_age_difference_ms"] == 10.0


def test_summary_identifies_primary_bottleneck() -> None:
    summary = summarize_latency_samples([
        {"capture_to_submit_ms": 3, "submit_to_result_ms": 20, "result_to_render_ms": 2},
        {"capture_to_submit_ms": 5, "submit_to_result_ms": 30, "result_to_render_ms": 4},
    ])
    assert summary["primary_bottleneck"] == "submit_to_result_ms"
    assert summary["metrics"]["submit_to_result_ms"]["p50"] == 25.0


def test_external_high_speed_measurement() -> None:
    assert external_sensor_to_photon_ms(
        recording_fps=240, physical_motion_frame=100, display_motion_frame=112
    ) == pytest.approx(50.0)
