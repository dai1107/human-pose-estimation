from pathlib import Path


def test_review_sidebar_record_queue_can_scroll_through_all_videos() -> None:
    css = Path("webui/static/review.css").read_text(encoding="utf-8")

    sidebar_rule = css.split(".quick-sidebar {", 1)[1].split("}", 1)[0]
    record_list_rule = css.split(".quick-record-list {", 1)[1].split("}", 1)[0]

    assert "min-height: 0;" in sidebar_rule
    assert "overflow: hidden;" in sidebar_rule
    assert "flex: 1 1 0;" in record_list_rule
    assert "min-height: 0;" in record_list_rule
    assert "overflow-y: auto;" in record_list_rule
    assert "scrollbar-gutter: stable;" in record_list_rule


def test_narrow_review_sidebar_scrolls_as_one_panel() -> None:
    css = Path("webui/static/review.css").read_text(encoding="utf-8")
    narrow_rules = css.split("@media (max-width: 820px)", 1)[1]

    assert "overflow-y: auto;" in narrow_rules.split(".quick-sidebar {", 1)[1].split(
        "}", 1
    )[0]
    assert "overflow: visible;" in narrow_rules.split(".quick-record-list {", 1)[
        1
    ].split("}", 1)[0]


def test_task_switcher_is_compact_so_the_video_queue_stays_tall() -> None:
    css = Path("webui/static/review.css").read_text(encoding="utf-8")

    switcher_rule = css.split(".task-switcher {", 1)[1].split("}", 1)[0]
    button_rule = css.split(".task-switcher button {", 1)[1].split("}", 1)[0]
    helper_rule = css.split(".task-switcher button small {", 1)[1].split("}", 1)[0]

    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in switcher_rule
    assert "padding: 8px;" in switcher_rule
    assert "min-height: 32px;" in button_rule
    assert "white-space: nowrap;" in button_rule
    assert "display: none;" in helper_rule
