from __future__ import annotations

from src.realtime.budget import RealtimeBudgetController
from src.realtime.inference_resolution import InferenceResolutionController


def _feed_window(
    controller: RealtimeBudgetController,
    *,
    inference_ms: float,
    pose_age_ms: float,
    queue_depth: int,
    can_reduce_resolution: bool,
    samples: int = 3,
):
    decision = None
    for _ in range(samples):
        decision = controller.observe(
            inference_ms=inference_ms,
            pose_result_age_ms=pose_age_ms,
            queue_depth=queue_depth,
            resolution_can_downgrade=can_reduce_resolution,
        )
    return decision


def test_budget_controller_uses_fixed_degradation_order() -> None:
    controller = RealtimeBudgetController(
        target_pose_fps=20,
        max_pose_fps=20,
        min_pose_fps=12,
        sample_window=3,
        min_samples=3,
        recovery_windows=1,
    )

    first = _feed_window(
        controller,
        inference_ms=70,
        pose_age_ms=90,
        queue_depth=1,
        can_reduce_resolution=True,
    )
    second = _feed_window(
        controller,
        inference_ms=70,
        pose_age_ms=90,
        queue_depth=1,
        can_reduce_resolution=True,
    )
    third = _feed_window(
        controller,
        inference_ms=70,
        pose_age_ms=90,
        queue_depth=1,
        can_reduce_resolution=True,
    )
    fourth = _feed_window(
        controller,
        inference_ms=70,
        pose_age_ms=90,
        queue_depth=1,
        can_reduce_resolution=False,
    )

    assert (first.action, first.target_pose_fps) == ("reduce_pose_fps", 15)
    assert (second.action, second.target_pose_fps) == ("reduce_pose_fps", 12)
    assert third.action == "reduce_inference_resolution"
    assert fourth.action == "reduce_extra_analysis_frequency"
    assert fourth.extra_analysis_stride == 2


def test_budget_controller_can_raise_inference_fps_with_sustained_headroom() -> None:
    controller = RealtimeBudgetController(
        target_pose_fps=15,
        max_pose_fps=20,
        min_pose_fps=12,
        sample_window=3,
        min_samples=3,
        recovery_windows=2,
    )

    first = _feed_window(
        controller,
        inference_ms=12,
        pose_age_ms=20,
        queue_depth=0,
        can_reduce_resolution=True,
    )
    second = _feed_window(
        controller,
        inference_ms=12,
        pose_age_ms=20,
        queue_depth=0,
        can_reduce_resolution=True,
    )

    assert first.action == "none"
    assert second.action == "increase_pose_fps"
    assert second.target_pose_fps == 20


def test_resolution_controller_supports_budget_requested_single_steps() -> None:
    controller = InferenceResolutionController(
        640,
        adaptive=True,
        width_steps=(640, 512, 320),
    )

    assert controller.can_step_down is True
    assert controller.force_step_down() is True
    assert controller.current_width == 512
    assert controller.downgrade_count == 1

