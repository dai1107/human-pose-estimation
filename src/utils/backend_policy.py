from __future__ import annotations

from pathlib import Path


HYROX_BACKEND_POLICY: dict[str, str] = {
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


def resolve_backend_choice(requested_backend: str, action_type: str = "auto", input_video: str = "") -> str:
    if requested_backend != "auto":
        return requested_backend

    normalized_action = normalize_action_name(action_type)
    if normalized_action and normalized_action != "auto":
        return HYROX_BACKEND_POLICY.get(normalized_action, "mediapipe")

    if input_video:
        stem = normalize_action_name(Path(input_video).stem)
        return HYROX_BACKEND_POLICY.get(stem, "mediapipe")

    return "mediapipe"


def normalize_action_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")
