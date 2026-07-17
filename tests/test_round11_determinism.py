from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.test_burpee_validity import _run_candidate as run_burpee_candidate
from tests.test_lunge_validity import (
    _analyzer as make_lunge_analyzer,
    _run_cycle as run_lunge_cycle,
)
from tests.test_wall_ball import _run_rule_candidate as run_wall_ball_candidate


CounterSnapshot = tuple[int, int, int, int, str]


def _counter_snapshot(state: dict[str, object]) -> CounterSnapshot:
    decision = state["last_rep_decision"]
    assert isinstance(decision, dict)
    return (
        int(state["candidate_count"]),
        int(state["pose_valid_rep_count"]),
        int(state["no_rep_count"]),
        int(state["unsure_count"]),
        str(decision["status"]),
    )


def _lunge_trace() -> dict[str, object]:
    return run_lunge_cycle(make_lunge_analyzer(), trailing="left")


def _burpee_trace() -> dict[str, object]:
    _, _, final = run_burpee_candidate()
    return final


def _wall_ball_trace() -> dict[str, object]:
    return run_wall_ball_candidate()


@pytest.mark.parametrize(
    "replay_trace",
    (_lunge_trace, _burpee_trace, _wall_ball_trace),
    ids=("lunge", "burpee_broad_jump", "wall_ball"),
)
def test_identical_feature_replay_has_deterministic_candidate_outcomes(
    replay_trace: Callable[[], dict[str, object]],
) -> None:
    first = _counter_snapshot(replay_trace())
    second = _counter_snapshot(replay_trace())

    assert first == second
    assert first[0] == first[1] + first[2] + first[3]
