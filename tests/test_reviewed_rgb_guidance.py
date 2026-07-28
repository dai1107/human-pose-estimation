from __future__ import annotations

from tools.evaluate_reviewed_rgb_guidance import (
    _candidate_alignment_frame,
    _match_statuses,
    _terminal_event_metrics,
)


def test_candidate_alignment_prefers_burpee_landing_event() -> None:
    candidate = {
        "source_frame": 140,
        "events": {"landing_frames": {"left": 82, "right": 84}},
    }

    assert _candidate_alignment_frame(candidate) == 84


def test_reviewed_rep_matching_is_monotonic_and_tolerance_bounded() -> None:
    human = [
        {"rep_id": "rep_1", "start_frame": 0, "end_frame": 90, "validity": "VALID"},
        {"rep_id": "rep_2", "start_frame": 91, "end_frame": 180, "validity": "NO_REP"},
    ]
    candidates = [
        {"source_frame": 125, "alignment_frame": 88, "status": "VALID"},
        {"source_frame": 181, "alignment_frame": 179, "status": "NO_REP"},
    ]

    matches = _match_statuses(human, candidates)

    assert [row["human_rep"]["rep_id"] for row in matches] == [
        "rep_1",
        "rep_2",
    ]
    assert all(row["status_match"] for row in matches)


def test_reviewed_rep_matching_does_not_force_a_distant_candidate() -> None:
    human = [
        {"rep_id": "rep_1", "start_frame": 0, "end_frame": 30, "validity": "VALID"},
    ]
    candidates = [
        {"source_frame": 200, "alignment_frame": 200, "status": "VALID"},
    ]

    matches = _match_statuses(human, candidates)

    assert len(matches) == 2
    assert matches[0]["candidate"] is None
    assert matches[1]["human_rep"] is None


def test_terminal_event_metrics_report_frames_and_time_tolerance() -> None:
    records = [
        {
            "fps": 30.0,
            "matches": [
                {
                    "candidate": {},
                    "human_rep": {},
                    "terminal_frame_error": 3,
                },
                {
                    "candidate": {},
                    "human_rep": {},
                    "terminal_frame_error": -9,
                },
            ],
        }
    ]

    metrics = _terminal_event_metrics(records)

    assert metrics["matched_terminal_event_count"] == 2
    assert metrics["mean_absolute_error_frames"] == 6
    assert metrics["within_5_frames_rate"] == 0.5
    assert metrics["within_200ms_rate"] == 0.5
