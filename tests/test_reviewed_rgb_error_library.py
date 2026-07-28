from __future__ import annotations

from tools.build_reviewed_rgb_error_library import _case_category


def test_error_library_case_categories_are_explicit() -> None:
    assert _case_category({"candidate": None, "human_rep": {}}) == "FN"
    assert _case_category({"candidate": {}, "human_rep": None}) == "FP"
    assert (
        _case_category(
            {
                "candidate": {"status": "UNSURE"},
                "human_rep": {"validity": "VALID"},
            }
        )
        == "UNSURE"
    )
    assert (
        _case_category(
            {
                "candidate": {"status": "VALID"},
                "human_rep": {"validity": "VALID"},
                "status_match": True,
            }
        )
        == "TP"
    )
    assert (
        _case_category(
            {
                "candidate": {"status": "NO_REP"},
                "human_rep": {"validity": "VALID"},
                "status_match": False,
            }
        )
        == "STATUS_MISMATCH"
    )
