from __future__ import annotations

import cv2
import numpy as np
import pytest
from pathlib import Path

import src.swimming.wrist_tracking as wrist_tracking
from src.swimming.wrist_tracking import (
    ArmChainObservation,
    LKOpticalFlowWristTracker,
    SwimWristIdentityTracker,
    SwimWristTrackerConfig,
    WristCandidate,
    hungarian_2x2,
    load_swim_wrist_tracker_config,
)
from src.configuration import ConfigValidationError


def _chains() -> dict[str, ArmChainObservation]:
    return {
        "left": ArmChainObservation((0.15, 0.5), (0.25, 0.5), 0.95),
        "right": ArmChainObservation((0.85, 0.5), (0.75, 0.5), 0.95),
    }


def _candidates(*, swapped_labels: bool = False) -> list[WristCandidate]:
    if swapped_labels:
        return [
            WristCandidate("left", (0.65, 0.5), 0.95),
            WristCandidate("right", (0.35, 0.5), 0.95),
        ]
    return [
        WristCandidate("left", (0.35, 0.5), 0.95),
        WristCandidate("right", (0.65, 0.5), 0.95),
    ]


def test_hungarian_2x2_selects_global_minimum() -> None:
    assert hungarian_2x2(((1.0, 0.1), (0.2, 2.0))) == (1, 0)
    assert hungarian_2x2(((0.1, 1.0), (2.0, 0.2))) == (0, 1)
    with pytest.raises(ValueError, match="2x2"):
        hungarian_2x2(((1.0,),))


def test_config_is_experimental_and_keeps_documented_weights(
    tmp_path: Path,
) -> None:
    config = load_swim_wrist_tracker_config()
    assert config.motion_weight == pytest.approx(0.40)
    assert config.chain_weight == pytest.approx(0.35)
    assert config.semantic_weight == pytest.approx(0.15)
    assert config.smoothness_weight == pytest.approx(0.10)

    invalid = tmp_path / "swim.yaml"
    invalid.write_text(
        "swim_wrist_tracking:\n  mode: formal\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="experimental mode"):
        load_swim_wrist_tracker_config(invalid)


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("appearance_ema_alpha: 1.1", "appearance_ema_alpha"),
        ("appearance_minimum_visibility: 1.1", "appearance_minimum_visibility"),
        ("cotracker_visibility_threshold: 1.1", "cotracker_visibility_threshold"),
    ],
)
def test_round6_probability_config_values_are_bounded(
    tmp_path: Path, line: str, key: str
) -> None:
    source = Path("configs/swim_wrist_tracking.yaml").read_text(encoding="utf-8")
    name = line.split(":", 1)[0]
    original = next(row for row in source.splitlines() if row.strip().startswith(name))
    invalid = tmp_path / f"{key}.yaml"
    invalid.write_text(source.replace(original, f"  {line}"), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="between 0 and 1"):
        load_swim_wrist_tracker_config(invalid)


def test_cotracker_overlap_must_be_smaller_than_window(tmp_path: Path) -> None:
    source = Path("configs/swim_wrist_tracking.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "overlap.yaml"
    invalid.write_text(
        source.replace("  cotracker_overlap_frames: 20", "  cotracker_overlap_frames: 120"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="smaller"):
        load_swim_wrist_tracker_config(invalid)


def test_persistent_tracks_use_chain_and_hysteresis_for_swapped_pose_labels() -> None:
    tracker = SwimWristIdentityTracker(
        SwimWristTrackerConfig(
            identity_confirmation_frames=3,
            identity_change_margin=0.05,
        )
    )
    first = tracker.update(
        _candidates(),
        _chains(),
        frame_index=0,
        timestamp_ms=0.0,
        body_scale=0.2,
    )
    assert first.tracks["left"].observed_semantic_side == "left"
    assert first.tracks["right"].observed_semantic_side == "right"

    pending = []
    for frame in (1, 2):
        pending.append(
            tracker.update(
                _candidates(swapped_labels=True),
                _chains(),
                frame_index=frame,
                timestamp_ms=frame * 40.0,
                body_scale=0.2,
            )
        )
    assert all(not item.mapping_changed for item in pending)
    assert all(
        "IDENTITY_HYSTERESIS_HOLD" in item.reason_codes for item in pending
    )

    confirmed = tracker.update(
        _candidates(swapped_labels=True),
        _chains(),
        frame_index=3,
        timestamp_ms=120.0,
        body_scale=0.2,
    )
    assert confirmed.mapping_changed is True
    assert confirmed.committed_semantic_mapping_swapped is True
    assert confirmed.tracks["left"].track_id == "swim_wrist_left"
    assert confirmed.tracks["left"].observed_semantic_side == "right"
    assert confirmed.tracks["left"].position[0] < confirmed.tracks["right"].position[0]
    assert confirmed.as_dict()["persistent_track_identity_switch_count"] == 0


def test_single_frame_label_swap_never_changes_committed_mapping() -> None:
    tracker = SwimWristIdentityTracker(
        SwimWristTrackerConfig(
            identity_confirmation_frames=3,
            identity_change_margin=0.05,
        )
    )
    tracker.update(
        _candidates(), _chains(), frame_index=0, timestamp_ms=0.0, body_scale=0.2
    )
    held = tracker.update(
        _candidates(swapped_labels=True),
        _chains(),
        frame_index=1,
        timestamp_ms=40.0,
        body_scale=0.2,
    )
    recovered = tracker.update(
        _candidates(),
        _chains(),
        frame_index=2,
        timestamp_ms=80.0,
        body_scale=0.2,
    )

    assert "IDENTITY_HYSTERESIS_HOLD" in held.reason_codes
    assert held.committed_semantic_mapping_swapped is False
    assert recovered.committed_semantic_mapping_swapped is False
    assert tracker.confirmed_mapping_change_count == 0


def test_constant_velocity_prediction_bridges_short_missing_gap() -> None:
    tracker = SwimWristIdentityTracker(
        SwimWristTrackerConfig(maximum_occlusion_frames=3)
    )
    tracker.update(
        _candidates(), _chains(), frame_index=0, timestamp_ms=0.0, body_scale=0.2
    )
    moved = [
        WristCandidate("left", (0.38, 0.5), 0.95),
        WristCandidate("right", (0.68, 0.5), 0.95),
    ]
    direct = tracker.update(
        moved, _chains(), frame_index=1, timestamp_ms=100.0, body_scale=0.2
    )
    missing = tracker.update(
        [], _chains(), frame_index=2, timestamp_ms=200.0, body_scale=0.2
    )

    assert missing.tracks["left"].state == "occluded"
    assert missing.tracks["left"].source == "prediction"
    assert missing.tracks["left"].position is not None
    assert missing.tracks["left"].position[0] >= direct.tracks["left"].position[0]


def test_track_transitions_occluded_lost_and_reacquired() -> None:
    tracker = SwimWristIdentityTracker(
        SwimWristTrackerConfig(
            maximum_occlusion_frames=1,
            maximum_prediction_frames=1,
            reacquire_confirmation_frames=2,
        )
    )
    tracker.update(
        _candidates(), _chains(), frame_index=0, timestamp_ms=0.0, body_scale=0.2
    )
    occluded = tracker.update(
        [], _chains(), frame_index=1, timestamp_ms=40.0, body_scale=0.2
    )
    lost = tracker.update(
        [], _chains(), frame_index=2, timestamp_ms=80.0, body_scale=0.2
    )
    reacquiring = tracker.update(
        _candidates(),
        _chains(),
        frame_index=3,
        timestamp_ms=120.0,
        body_scale=0.2,
    )
    reacquired = tracker.update(
        _candidates(),
        _chains(),
        frame_index=4,
        timestamp_ms=160.0,
        body_scale=0.2,
    )

    assert occluded.tracks["left"].state == "occluded"
    assert lost.tracks["left"].state == "lost"
    assert reacquiring.tracks["left"].state == "reacquiring"
    assert reacquired.tracks["left"].state == "reacquired"
    assert reacquired.tracks["left"].reacquisition_count == 1


def test_trajectory_outlier_is_rejected_without_moving_track() -> None:
    tracker = SwimWristIdentityTracker(
        SwimWristTrackerConfig(
            identity_change_margin=100.0,
            outlier_minimum_speed_body_s=0.5,
            outlier_maximum_speed_body_s=0.5,
        )
    )
    tracker.update(
        _candidates(), _chains(), frame_index=0, timestamp_ms=0.0, body_scale=0.2
    )
    jump = [
        WristCandidate("left", (0.95, 0.5), 0.95),
        WristCandidate("right", (0.65, 0.5), 0.95),
    ]
    result = tracker.update(
        jump,
        _chains(),
        frame_index=1,
        timestamp_ms=1000.0,
        body_scale=0.2,
    )

    assert "left_trajectory_outlier" in result.reason_codes
    assert result.tracks["left"].outlier_rejection_count == 1
    assert result.tracks["left"].source == "prediction"
    assert result.tracks["left"].position[0] == pytest.approx(0.35)


def test_lk_tracks_translation_with_forward_backward_consistency() -> None:
    rng = np.random.default_rng(7)
    first = rng.integers(0, 256, size=(80, 80), dtype=np.uint8)
    second = cv2.warpAffine(
        first,
        np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32),
        (80, 80),
    )
    first_bgr = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    second_bgr = cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)
    tracker = LKOpticalFlowWristTracker()
    tracker.advance(first_bgr, body_scale_px=30.0)
    tracker.reanchor("left", (0.4, 0.4), frame_shape=first_bgr.shape)

    result = tracker.advance(second_bgr, body_scale_px=30.0)["left"]

    assert result.reliable is True
    assert result.reason == "accepted"
    assert result.position is not None
    assert result.position[0] == pytest.approx(35.0 / 80.0, abs=0.02)
    assert result.position[1] == pytest.approx(34.0 / 80.0, abs=0.02)
    assert result.forward_backward_error_px < 1.5


def test_lk_rejects_forward_backward_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    tracker = LKOpticalFlowWristTracker()
    tracker.advance(frame, body_scale_px=20.0)
    tracker.reanchor("left", (0.5, 0.5), frame_shape=frame.shape)
    calls = 0

    def fake_flow(*args, **kwargs):
        nonlocal calls
        calls += 1
        points = np.asarray(args[2], dtype=np.float32)
        offset = 2.0 if calls == 1 else 10.0
        return points + offset, np.ones((1, 1), dtype=np.uint8), None

    monkeypatch.setattr(wrist_tracking.cv2, "calcOpticalFlowPyrLK", fake_flow)

    result = tracker.advance(frame, body_scale_px=20.0)["left"]

    assert result.reliable is False
    assert result.reason == "forward_backward_failure"
