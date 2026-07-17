from __future__ import annotations

from hyrox.actions import (
    FarmersCarryAnalyzer,
    RowingAnalyzer,
    SkiErgAnalyzer,
    SledPullAnalyzer,
    SledPushAnalyzer,
)
from hyrox.config import (
    DEFAULT_FARMERS_CARRY_CONFIG,
    DEFAULT_ROWING_CONFIG,
    DEFAULT_SLED_PULL_CONFIG,
)
from hyrox.violations import TemporalViolationTracker


def _rowing_pose(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "left_knee_angle": 90.0,
        "right_knee_angle": 92.0,
        "left_hip_angle": 95.0,
        "right_hip_angle": 98.0,
        "left_elbow_angle": 165.0,
        "right_elbow_angle": 167.0,
        "torso_angle": 20.0,
        "hip_center_y": 0.65,
        "body_height_norm": 0.70,
    }
    values.update(overrides)
    return values


def _sled_pose(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "left_elbow_angle": 160.0,
        "right_elbow_angle": 162.0,
        "left_knee_angle": 155.0,
        "right_knee_angle": 157.0,
        "left_hip_angle": 150.0,
        "right_hip_angle": 152.0,
        "torso_angle": 10.0,
        "body_center_y": 0.50,
        "hip_center_y": 0.45,
        "hip_center_x": 0.50,
        "body_height_norm": 0.70,
        "body_box_height_norm": 0.75,
        "lower_body_visible_score": 0.95,
        "left_knee_x": 0.45,
        "left_knee_y": 0.60,
        "left_knee_confidence": 0.95,
        "right_knee_x": 0.55,
        "right_knee_y": 0.60,
        "right_knee_confidence": 0.95,
        "knee_center_x": 0.50,
        "knee_center_y": 0.60,
        "left_ankle_x": 0.45,
        "left_ankle_y": 0.88,
        "left_ankle_confidence": 0.95,
        "right_ankle_x": 0.55,
        "right_ankle_y": 0.88,
        "right_ankle_confidence": 0.95,
        "left_heel_x": 0.43,
        "left_heel_y": 0.90,
        "left_heel_confidence": 0.95,
        "right_heel_x": 0.53,
        "right_heel_y": 0.90,
        "right_heel_confidence": 0.95,
        "left_foot_index_x": 0.47,
        "left_foot_index_y": 0.90,
        "left_foot_index_confidence": 0.95,
        "right_foot_index_x": 0.57,
        "right_foot_index_y": 0.90,
        "right_foot_index_confidence": 0.95,
        "left_wrist_y": 0.40,
        "right_wrist_y": 0.41,
    }
    values.update(overrides)
    return values


def _carry_pose(**overrides: float) -> dict[str, float]:
    values = {
        "visible_score": 0.95,
        "left_knee_angle": 174.0,
        "right_knee_angle": 175.0,
        "left_hip_angle": 168.0,
        "right_hip_angle": 169.0,
        "left_elbow_angle": 170.0,
        "right_elbow_angle": 171.0,
        "left_wrist_to_hip_y": 0.18,
        "right_wrist_to_hip_y": 0.19,
        "left_shoulder_x": 0.40,
        "right_shoulder_x": 0.60,
        "left_hip_x": 0.45,
        "right_hip_x": 0.55,
        "left_wrist_x": 0.46,
        "right_wrist_x": 0.54,
        "shoulder_tilt": 0.01,
        "hip_tilt": 0.01,
        "torso_angle": 3.0,
        "body_center_x": 0.50,
        "body_center_y": 0.42,
        "body_height_norm": 0.72,
        "ankle_distance_norm": 0.20,
    }
    values.update(overrides)
    return values


def test_temporal_violation_tracker_filters_short_anomaly() -> None:
    tracker = TemporalViolationTracker("TEST_VIOLATION", 300)

    assert tracker.update(True, 100, confidence=0.9).status == "CANDIDATE"
    assert tracker.update(True, 350, confidence=0.9).status == "CANDIDATE"
    assert tracker.update(False, 375, confidence=0.9).status == "CLEAR"
    assert tracker.update(True, 500, confidence=0.9).status == "CANDIDATE"
    assert tracker.update(True, 800, confidence=0.9).status == "ACTIVE"


def test_rowing_early_stand_requires_clear_300_ms_hold() -> None:
    analyzer = RowingAnalyzer.from_config(
        {**DEFAULT_ROWING_CONFIG, "stable_frames": 1}
    )
    analyzer.set_camera_view("side")
    analyzer.update(_rowing_pose(), 0)
    standing = _rowing_pose(
        left_knee_angle=170.0,
        right_knee_angle=172.0,
        left_hip_angle=168.0,
        right_hip_angle=170.0,
        torso_angle=2.0,
        hip_center_y=0.45,
    )

    candidate = analyzer.update(standing, 100)
    short = analyzer.update(standing, 350)
    active = analyzer.update(standing, 400)

    assert candidate["active_violation_codes"] == []
    assert short["active_violation_codes"] == []
    assert active["active_violation_codes"] == [
        "ROWING_EARLY_STAND_PROXY"
    ]
    assert active["debug"]["rowing_early_stand_proxy"]["duration_ms"] == 300
    assert active["rep_count"] == 0


def test_rowing_front_view_does_not_claim_early_stand() -> None:
    analyzer = RowingAnalyzer.from_config(
        {**DEFAULT_ROWING_CONFIG, "stable_frames": 1}
    )
    analyzer.set_camera_view("front")
    analyzer.update(_rowing_pose(), 0)
    standing = _rowing_pose(
        left_knee_angle=170.0,
        right_knee_angle=172.0,
        left_hip_angle=168.0,
        right_hip_angle=170.0,
        torso_angle=2.0,
        hip_center_y=0.45,
    )

    analyzer.update(standing, 100)
    state = analyzer.update(standing, 500)

    assert state["active_violation_codes"] == []
    assert state["violation_results"][0]["status"] == "UNSURE"


def _sled_analyzer(*, floor: bool = True) -> SledPullAnalyzer:
    analyzer = SledPullAnalyzer.from_config(
        {**DEFAULT_SLED_PULL_CONFIG, "stable_frames": 1}
    )
    analyzer.set_camera_view("side")
    if floor:
        analyzer.set_manual_floor_line((0.0, 0.90), (1.0, 0.90))
    return analyzer


def test_sled_pull_kneeling_requires_contact_and_150_ms_hold() -> None:
    analyzer = _sled_analyzer()
    analyzer.update(_sled_pose(), 0)
    kneeling_pull = _sled_pose(
        left_elbow_angle=125.0,
        right_elbow_angle=127.0,
        left_knee_angle=90.0,
        right_knee_angle=92.0,
        hip_center_y=0.65,
        left_knee_y=0.895,
        right_knee_y=0.895,
        knee_center_y=0.895,
    )
    states = [
        analyzer.update(kneeling_pull, timestamp)
        for timestamp in (100, 150, 200, 250, 300, 350, 400)
    ]

    assert states[-3]["active_violation_codes"] == []
    assert "SLED_PULL_KNEELING_VIOLATION" in states[-2][
        "active_violation_codes"
    ]
    assert (
        states[-1]["debug"]["knee_contact"]["status"] == "CONTACT"
    )


def test_sled_pull_seated_and_uncertain_seated_outputs_are_distinct() -> None:
    seated_pull = _sled_pose(
        left_elbow_angle=125.0,
        right_elbow_angle=127.0,
        left_knee_angle=110.0,
        right_knee_angle=112.0,
        hip_center_y=0.65,
        torso_angle=0.0,
    )

    clear = _sled_analyzer()
    clear.update(_sled_pose(), 0)
    clear_states = [
        clear.update(seated_pull, timestamp)
        for timestamp in (100, 150, 250, 300, 400)
    ]
    assert "SLED_PULL_SEATED_VIOLATION" in clear_states[-1][
        "active_violation_codes"
    ]
    assert clear_states[-1]["uncertain_violation_codes"] == []

    unsure = _sled_analyzer(floor=False)
    unsure.update(_sled_pose(), 0)
    unsure_states = [
        unsure.update(seated_pull, timestamp)
        for timestamp in (100, 150, 250, 300, 400)
    ]
    assert unsure_states[-1]["active_violation_codes"] == []
    assert unsure_states[-1]["uncertain_violation_codes"] == [
        "UNSURE_POSSIBLE_SEATED_PULL"
    ]


def test_sled_pull_standing_and_low_stance_are_not_pose_violations() -> None:
    standing = _sled_analyzer()
    standing.update(_sled_pose(), 0)
    standing_state: dict[str, object] = {}
    for timestamp in (100, 200, 300, 400):
        standing_state = standing.update(
            _sled_pose(left_elbow_angle=125.0, right_elbow_angle=127.0),
            timestamp,
        )

    low_stance = _sled_analyzer()
    low_stance.update(_sled_pose(), 0)
    low_state: dict[str, object] = {}
    for timestamp in (100, 200, 300, 400):
        low_state = low_stance.update(
            _sled_pose(
                left_elbow_angle=125.0,
                right_elbow_angle=127.0,
                left_knee_angle=110.0,
                right_knee_angle=112.0,
                hip_center_y=0.58,
                torso_angle=45.0,
            ),
            timestamp,
        )

    assert standing_state["active_violation_codes"] == []
    assert standing_state["uncertain_violation_codes"] == []
    assert low_state["active_violation_codes"] == []
    assert low_state["uncertain_violation_codes"] == []


def _run_carry_violation(
    *,
    overrides: dict[str, float],
) -> dict[str, object]:
    analyzer = FarmersCarryAnalyzer.from_config(
        {**DEFAULT_FARMERS_CARRY_CONFIG, "stable_frames": 1}
    )
    analyzer.set_camera_view("front")
    analyzer.update(_carry_pose(), 0)
    state: dict[str, object] = {}
    for timestamp, center_x in (
        (100, 0.51),
        (200, 0.52),
        (300, 0.53),
        (400, 0.54),
    ):
        state = analyzer.update(
            _carry_pose(body_center_x=center_x, **overrides),
            timestamp,
        )
    return state


def test_farmers_carry_arm_violations_require_300_ms() -> None:
    not_extended = _run_carry_violation(
        overrides={
            "left_elbow_angle": 140.0,
            "right_elbow_angle": 142.0,
        }
    )
    not_by_side = _run_carry_violation(
        overrides={
            "left_wrist_to_hip_y": 0.0,
            "right_wrist_to_hip_y": 0.0,
            "left_wrist_x": 0.20,
            "right_wrist_x": 0.80,
        }
    )

    assert not_extended["active_violation_codes"] == [
        "ARM_NOT_EXTENDED_VIOLATION"
    ]
    assert not_by_side["active_violation_codes"] == [
        "ARM_NOT_BY_SIDE_VIOLATION"
    ]
    assert not_extended["rep_count"] == 0
    assert not_by_side["rep_count"] == 0


def test_skierg_and_sled_push_do_not_add_pose_violation_codes() -> None:
    skierg = SkiErgAnalyzer.from_config({"stable_frames": 1})
    sled_push = SledPushAnalyzer.from_config({"stable_frames": 1})

    ski_state = skierg.update({"visible_score": 0.95}, 100)
    push_state = sled_push.update({"visible_score": 0.95}, 100)

    assert "active_violation_codes" not in ski_state
    assert "active_violation_codes" not in push_state
    assert ski_state["count_semantics"] == "analysis_cycle"
    assert push_state["count_semantics"] == "analysis_cycle"
    assert ski_state["official_rep_count_supported"] is False
    assert push_state["official_rep_count_supported"] is False
