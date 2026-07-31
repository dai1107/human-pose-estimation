from pathlib import Path

from webui.review import ANGLE_EVENTS_BY_ACTION, PHONE_ACTIONS


def test_angle_review_page_explains_the_task_and_exposes_save_actions() -> None:
    html = Path("webui/templates/review.html").read_text(encoding="utf-8")

    assert "每个视频至少保存 1 条代表性关节角度" in html
    assert "怎样算这个视频完成？" in html
    assert "未点完三个点也能保存进度" in html
    assert "不满三个有效点时保存为可恢复草稿" in html
    assert 'id="anglePointProgress"' in html
    assert 'id="angleSaveHelp"' in html
    assert 'id="angleSaveButton"' in html
    assert 'id="angleSaveNextButton"' in html
    assert "保存进度并进入下一个视频" in html
    save_button_tag = html.split('id="angleSaveButton"', 1)[1].split(">", 1)[0]
    assert " disabled" not in save_button_tag


def test_angle_save_buttons_explain_incomplete_points_and_support_next_video() -> None:
    source = Path("webui/static/review.js").read_text(encoding="utf-8")

    assert "const ANGLE_TASK_GUIDANCE" in source
    assert "save_as_draft: !complete" in source
    assert "当前进度已保存" in source
    assert "restoreAngleDraft" in source
    assert "已写入服务器" in source
    assert "不需要另外提交" in source
    assert "saveAngleAnnotation({next: true})" in source
    assert "if (next) selectNextRecord();" in source


def test_angle_events_are_specific_to_each_supported_action() -> None:
    assert set(ANGLE_EVENTS_BY_ACTION) == PHONE_ACTIONS
    assert dict(ANGLE_EVENTS_BY_ACTION["lunge"])["rear_knee_contact"] == "后膝接触"
    assert dict(ANGLE_EVENTS_BY_ACTION["rowing"])["finish"] == "蹬腿拉动结束"
    assert dict(ANGLE_EVENTS_BY_ACTION["burpee_broad_jump"])["landing"] == "落地"
    for events in ANGLE_EVENTS_BY_ACTION.values():
        values = [value for value, _ in events]
        assert values[0] == "unspecified"
        assert values[-1] == "other"
        assert len(values) == len(set(values))


def test_angle_event_select_is_populated_from_the_current_record() -> None:
    source = Path("webui/static/review.js").read_text(encoding="utf-8")
    html = Path("webui/templates/review.html").read_text(encoding="utf-8")

    assert "function angleEventsForRecord(payload)" in source
    assert "state.angleEvents = angleEventsForRecord(payload)" in source
    assert "if (fromServer.length) return fromServer" in source
    assert "ANGLE_EVENTS_BY_ACTION[payload.record?.action]" in source
    assert "fallback.filter(([value]) => legacyValues.has(value))" in source
    assert "angleEventsCompatibilityMode" in source
    assert '$("#angleEvent").innerHTML = state.angleEvents' in source
    assert 'id="angleEventHelp"' in html
    assert "当前后台服务尚未重启" in source
    assert "?v=20260801-angle-overlay-align" in html


def test_angle_save_actions_remain_visually_prominent() -> None:
    css = Path("webui/static/review.css").read_text(encoding="utf-8")

    guide_rule = css.split(".angle-task-guide ol {", 1)[1].split("}", 1)[0]
    actions_rule = css.split(".angle-form-actions {", 1)[1].split("}", 1)[0]

    assert "grid-template-columns: repeat(4, minmax(0, 1fr));" in guide_rule
    assert "position: sticky;" in actions_rule
    assert "bottom: 0;" in actions_rule
