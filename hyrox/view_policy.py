from __future__ import annotations

from dataclasses import dataclass

from hyrox.feedback import FeedbackMessage


CAMERA_VIEWS = ("unknown", "front", "side", "front_left", "front_right")


def normalize_camera_view(value: str | None) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_")
    return normalized if normalized in CAMERA_VIEWS else "unknown"


def view_profile(value: str | None) -> str:
    normalized = normalize_camera_view(value)
    return "front" if normalized in {"front", "front_left", "front_right"} else normalized


def next_camera_view(value: str | None) -> str:
    profile = view_profile(value)
    return "front" if profile == "unknown" else ("side" if profile == "front" else "front")


@dataclass(frozen=True)
class ActionViewPolicy:
    preferred: frozenset[str]
    front_codes: frozenset[str]
    side_codes: frozenset[str]


_POLICIES: dict[str, ActionViewPolicy] = {
    "lunge": ActionViewPolicy(
        frozenset({"side"}),
        frozenset({"STAND_EXTENSION", "LOW_VISIBILITY"}),
        frozenset({"NOT_DEEP_ENOUGH", "LEAN_TOO_MUCH", "STAND_EXTENSION", "LOW_VISIBILITY"}),
    ),
    "wall_ball": ActionViewPolicy(
        frozenset({"front", "side"}),
        frozenset({"KNEES_CAVE_IN", "NOT_FULL_EXTENSION", "LOW_VISIBILITY"}),
        frozenset({"SQUAT_NOT_DEEP", "NOT_FULL_EXTENSION", "LOW_VISIBILITY"}),
    ),
    "farmers_carry": ActionViewPolicy(
        frozenset({"front"}),
        frozenset({"LEAN_LEFT_RIGHT", "SHOULDERS_UNEVEN", "ARMS_NOT_DOWN", "ARM_NOT_EXTENDED_VIOLATION", "ARM_NOT_BY_SIDE_VIOLATION", "UNSTABLE_CARRY", "LOW_VISIBILITY"}),
        frozenset({"ARMS_NOT_DOWN", "ARM_NOT_EXTENDED_VIOLATION", "ARM_NOT_BY_SIDE_VIOLATION", "TORSO_LEAN", "UNSTABLE_CARRY", "LOW_VISIBILITY"}),
    ),
    "rowing": ActionViewPolicy(
        frozenset({"side"}),
        frozenset({"NOT_SEATED_OR_BAD_VIEW", "LOW_VISIBILITY"}),
        frozenset({"ROWING_EARLY_STAND_PROXY", "TOO_MUCH_BACK_LEAN", "NO_FULL_LEG_DRIVE", "EARLY_ARM_PULL", "RUSHED_RECOVERY", "NOT_SEATED_OR_BAD_VIEW", "LOW_VISIBILITY"}),
    ),
    "skierg": ActionViewPolicy(
        frozenset({"front"}),
        frozenset({"ARMS_NOT_HIGH_ENOUGH", "TOO_MUCH_SQUAT", "ASYMMETRIC_PULL", "RUSHED_RETURN", "LOW_VISIBILITY"}),
        frozenset({"ARMS_NOT_HIGH_ENOUGH", "NO_HIP_HINGE", "TOO_MUCH_SQUAT", "RUSHED_RETURN", "LOW_VISIBILITY"}),
    ),
    "burpee_broad_jump": ActionViewPolicy(
        frozenset({"side"}),
        frozenset({"FEET_STAGGERED", "EXTRA_STEPS", "LOW_VISIBILITY"}),
        frozenset({"CHEST_NOT_LOW", "EXTRA_STEPS", "NO_BROAD_JUMP", "HIPS_TOO_HIGH_IN_BOTTOM", "LOW_VISIBILITY"}),
    ),
    "sled_push": ActionViewPolicy(
        frozenset({"side"}),
        frozenset({"SHORT_STEPS", "LOW_VISIBILITY"}),
        frozenset({"TORSO_TOO_UPRIGHT", "TORSO_TOO_LOW", "SHORT_STEPS", "NO_LEG_DRIVE", "HIP_TOO_HIGH_OR_BACK_ROUND", "LOW_VISIBILITY"}),
    ),
    "sled_pull": ActionViewPolicy(
        frozenset({"side"}),
        frozenset({"NOT_STANDING", "ASYMMETRIC_PULL", "LOW_VISIBILITY"}),
        frozenset({"SLED_PULL_KNEELING_VIOLATION", "SLED_PULL_SEATED_VIOLATION", "UNSURE_POSSIBLE_SEATED_PULL", "NOT_STANDING", "OVER_LEAN_BACK", "ARMS_ONLY_PULL", "NO_CLEAR_PULL", "ASYMMETRIC_PULL", "LOW_VISIBILITY"}),
    ),
}


def _action_key(action: str) -> str:
    return action.strip().lower().replace(" ", "_")


def filter_feedback_for_view(
    action: str,
    camera_view: str,
    messages: list[FeedbackMessage],
) -> tuple[list[FeedbackMessage], bool]:
    profile = view_profile(camera_view)
    policy = _POLICIES.get(_action_key(action))
    if policy is None or profile == "unknown":
        return messages, False
    allowed = policy.front_codes if profile == "front" else policy.side_codes
    filtered = [message for message in messages if message.code.upper() in allowed]
    limited = profile not in policy.preferred
    if limited:
        preferred_text = "正面" if "front" in policy.preferred else "侧面"
        filtered.append(
            FeedbackMessage(
                level="info",
                code="CAMERA_VIEW_LIMITED",
                text=f"当前视角仅提供有限评价；此动作建议使用{preferred_text}视角",
                confidence=1.0,
            )
        )
    return filtered, limited


def action_view_suitability(
    action: str,
    camera_view: str,
) -> bool | None:
    profile = view_profile(camera_view)
    policy = _POLICIES.get(_action_key(action))
    if policy is None or profile == "unknown":
        return None
    return profile in policy.preferred


__all__ = [
    "CAMERA_VIEWS",
    "ActionViewPolicy",
    "action_view_suitability",
    "filter_feedback_for_view",
    "normalize_camera_view",
    "next_camera_view",
    "view_profile",
]
