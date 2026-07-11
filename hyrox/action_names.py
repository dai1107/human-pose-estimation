from __future__ import annotations


HYROX_ACTION_NAMES = (
    "lunge",
    "wall_ball",
    "farmers_carry",
    "rowing",
    "skierg",
    "burpee_broad_jump",
    "sled_push",
    "sled_pull",
)

HYROX_ACTION_OPTIONS = ("none", *HYROX_ACTION_NAMES)
HYROX_ACTION_LABELS = {
    "none": "关闭动作指导 / Off",
    "lunge": "负重箭步蹲 / Lunge",
    "wall_ball": "投掷药球 / Wall Ball",
    "farmers_carry": "农夫行走 / Farmers Carry",
    "rowing": "划船机 / Rowing",
    "skierg": "滑雪机 / SkiErg",
    "burpee_broad_jump": "波比跳远 / Burpee Broad Jump",
    "sled_push": "推雪橇 / Sled Push",
    "sled_pull": "拉雪橇 / Sled Pull",
}


def action_from_menu_key(key: int) -> str | None:
    if ord("0") <= key <= ord("8"):
        return HYROX_ACTION_OPTIONS[key - ord("0")]
    return None


def next_hyrox_action(current: str) -> str:
    try:
        index = HYROX_ACTION_OPTIONS.index(current)
    except ValueError:
        index = 0
    return HYROX_ACTION_OPTIONS[(index + 1) % len(HYROX_ACTION_OPTIONS)]


__all__ = [
    "HYROX_ACTION_LABELS",
    "HYROX_ACTION_NAMES",
    "HYROX_ACTION_OPTIONS",
    "action_from_menu_key",
    "next_hyrox_action",
]
