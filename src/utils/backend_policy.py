from __future__ import annotations

from pathlib import Path

from src.backends.catalog import PRODUCT_BACKEND


# Retained only for explicit offline comparisons and research ablations.
EXPERIMENTAL_HYROX_BACKEND_POLICY: dict[str, str] = {
    "农夫行走": "mediapipe",
    "farmers_carry": "mediapipe",
    "划船机": "yolo-pose",
    "rowing": "yolo-pose",
    "投掷药球": "mediapipe",
    "wall_ball": "mediapipe",
    "拉雪橇": "yolo-pose",
    "sled_pull": "yolo-pose",
    "推雪橇": "mediapipe",
    "sled_push": "mediapipe",
    "波比跳远": "mediapipe",
    "burpee_broad_jump": "mediapipe",
    "滑雪机": "yolo-pose",
    "ski_erg": "yolo-pose",
    "负重箭步蹲": "mediapipe",
    "weighted_lunge": "mediapipe",
}
HYROX_BACKEND_POLICY = EXPERIMENTAL_HYROX_BACKEND_POLICY


def resolve_backend_choice(
    requested_backend: str,
    action_type: str = "auto",
    input_video: str = "",
    *,
    product_mode: bool = True,
) -> str:
    if requested_backend != "auto":
        return requested_backend

    # In product mode auto is a compatibility alias, not model selection.
    if product_mode:
        return PRODUCT_BACKEND

    normalized_action = normalize_action_name(action_type)
    if normalized_action and normalized_action != "auto":
        return EXPERIMENTAL_HYROX_BACKEND_POLICY.get(normalized_action, PRODUCT_BACKEND)

    if input_video:
        stem = normalize_action_name(Path(input_video).stem)
        return EXPERIMENTAL_HYROX_BACKEND_POLICY.get(stem, PRODUCT_BACKEND)

    return PRODUCT_BACKEND


def normalize_action_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")
