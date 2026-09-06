from __future__ import annotations

from pathlib import Path

import pytest

from src.angle_v2 import (
    FAIL,
    PASS,
    UNSURE,
    AngleHysteresis,
    AngleV2Config,
    AngleV2ShadowProcessor,
    EndpointPreservingAngleFilter,
    JointAngleSmoothingConfig,
    TemporalRuleEvidence,
    load_angle_v2_config,
)
from src.configuration import ConfigValidationError


def test_config_rejects_non_shadow_mode(tmp_path: Path) -> None:
    path = tmp_path / "angle_v2.yaml"
    path.write_text(
        "angle_v2_shadow:\n  enabled: true\n  mode: formal\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="shadow mode"):
        load_angle_v2_config(path)


def test_config_parses_joint_specific_smoothing(tmp_path: Path) -> None:
    path = tmp_path / "angle_v2.yaml"
    path.write_text(
        """angle_v2_shadow:
  enabled: true
  mode: shadow
angle_v2_smoothing_knee:
  min_cutoff: 2.5
  beta: 0.2
  d_cutoff: 1.5
angle_v2_smoothing_default:
  min_cutoff: 1.0
  beta: 0.05
  d_cutoff: 1.0
""",
        encoding="utf-8",
    )

    config = load_angle_v2_config(path)

    assert config.smoothing_for("left_knee") == JointAngleSmoothingConfig(
        2.5, 0.2, 1.5
    )
    assert config.smoothing_for("unknown_joint") == JointAngleSmoothingConfig(
        1.0, 0.05, 1.0
    )


def test_bone_length_rejection_invalidates_angle() -> None:
    result = AngleV2ShadowProcessor().observe(
        joint="left_knee",
        frame_index=0,
        timestamp_ms=0.0,
        raw_2d_angle_deg=120.0,
        raw_3d_angle_deg=121.0,
        confidence=0.9,
        quality_reasons=("bone_length_jump",),
    )

    assert result.bone_length_valid is False
    assert result.angle_valid is False
    assert result.filtered_2d_angle_deg is None
    assert "BONE_LENGTH_INCONSISTENT" in result.reason_codes


def test_2d_3d_conflict_only_invalidates_shadow_evidence() -> None:
    config = AngleV2Config()
    result = AngleV2ShadowProcessor(config).observe(
        joint="left_knee",
        frame_index=0,
        timestamp_ms=0.0,
        raw_2d_angle_deg=100.0,
        raw_3d_angle_deg=150.0,
        confidence=0.9,
    )

    assert result.angle_valid is True
    assert result.filtered_2d_angle_deg is not None
    assert result.two_d_three_d_conflict is True
    assert result.evidence_valid is False
    assert result.as_dict()["shadow_only"] is True


def test_temporal_outlier_does_not_pollute_filter_state() -> None:
    config = AngleV2Config(
        temporal_minimum_velocity_limit_deg_s=100.0,
        temporal_maximum_velocity_deg_s=100.0,
    )
    processor = AngleV2ShadowProcessor(config)
    first = processor.observe(
        joint="left_knee",
        frame_index=0,
        timestamp_ms=0.0,
        raw_2d_angle_deg=100.0,
        raw_3d_angle_deg=None,
        confidence=0.9,
    )
    rejected = processor.observe(
        joint="left_knee",
        frame_index=1,
        timestamp_ms=100.0,
        raw_2d_angle_deg=170.0,
        raw_3d_angle_deg=None,
        confidence=0.9,
    )
    recovered = processor.observe(
        joint="left_knee",
        frame_index=2,
        timestamp_ms=200.0,
        raw_2d_angle_deg=102.0,
        raw_3d_angle_deg=None,
        confidence=0.9,
    )

    assert first.filtered_2d_angle_deg == pytest.approx(100.0)
    assert rejected.temporal_outlier is True
    assert rejected.filtered_2d_angle_deg is None
    assert recovered.angle_valid is True
    assert 100.0 <= recovered.filtered_2d_angle_deg <= 102.0


@pytest.mark.parametrize(
    ("values", "kind", "expected_angle"),
    [
        ([100.0, 80.0, 80.0, 80.0, 100.0], "minimum", 80.0),
        ([80.0, 100.0, 100.0, 100.0, 80.0], "maximum", 100.0),
    ],
)
def test_endpoint_filter_detects_short_plateaus(
    values: list[float], kind: str, expected_angle: float
) -> None:
    detector = EndpointPreservingAngleFilter(
        "left_knee",
        AngleV2Config(
            endpoint_radius_frames=2,
            endpoint_minimum_prominence_deg=1.0,
        ),
    )
    events = []
    for frame, value in enumerate(values):
        events.extend(
            detector.observe(
                frame_index=frame,
                timestamp_ms=frame * 40.0,
                raw_angle_deg=value,
                filtered_angle_deg=value,
                confidence=0.9,
                valid=True,
            )
        )

    assert len(events) == 1
    assert events[0].kind == kind
    assert events[0].raw_extremum_angle_deg == pytest.approx(expected_angle)
    assert events[0].confirmed_at_frame == 4


def test_endpoint_filter_rejects_monotonic_window() -> None:
    detector = EndpointPreservingAngleFilter("left_knee", AngleV2Config())
    events = []
    for frame, value in enumerate((80.0, 90.0, 100.0, 110.0, 120.0)):
        events.extend(
            detector.observe(
                frame_index=frame,
                timestamp_ms=frame * 40.0,
                raw_angle_deg=value,
                filtered_angle_deg=value,
                confidence=0.9,
                valid=True,
            )
        )
    assert events == []


def test_hysteresis_uses_distinct_enter_and_exit_thresholds() -> None:
    gate = AngleHysteresis(threshold=160.0, direction="above", width_deg=3.0)

    assert gate.observe(159.0) is False
    assert gate.observe(160.0) is True
    assert gate.observe(158.0) is True
    assert gate.observe(156.9) is False


def test_temporal_evidence_allows_unsure_gaps_but_not_alternation() -> None:
    evidence = TemporalRuleEvidence(
        window_size=5,
        pass_count=3,
        fail_count=3,
        minimum_hold_ms=100.0,
    )
    states = [PASS, PASS, UNSURE, PASS, PASS]
    output = [
        evidence.observe(state, timestamp_ms=index * 50.0)
        for index, state in enumerate(states)
    ]
    assert output[-1] == PASS

    evidence.reset()
    alternating = [PASS, FAIL, PASS, FAIL, PASS]
    output = [
        evidence.observe(state, timestamp_ms=index * 50.0)
        for index, state in enumerate(alternating)
    ]
    assert output[-1] == UNSURE

