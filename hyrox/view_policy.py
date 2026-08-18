from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hyrox.feedback import FeedbackMessage


CAMERA_VIEWS = (
    "unknown",
    "front",
    "rear",
    "side",
    "front_left",
    "front_right",
    "oblique_front",
    "oblique_rear",
)


def normalize_camera_view(value: str | None) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_")
    return normalized if normalized in CAMERA_VIEWS else "unknown"


def view_profile(value: str | None) -> str:
    normalized = normalize_camera_view(value)
    return (
        "front"
        if normalized
        in {
            "front",
            "rear",
            "front_left",
            "front_right",
            "oblique_front",
            "oblique_rear",
        }
        else normalized
    )


def next_camera_view(value: str | None) -> str:
    profile = view_profile(value)
    return "front" if profile == "unknown" else ("side" if profile == "front" else "front")


@dataclass(frozen=True)
class ActionViewPolicy:
    preferred: frozenset[str]
    front_codes: frozenset[str]
    side_codes: frozenset[str]


ViewCapabilityLevel = Literal[
    "recommended", "usable", "not_recommended", "unjudgeable"
]


@dataclass(frozen=True)
class ActionViewCapability:
    level: ViewCapabilityLevel
    score_multiplier: float
    threshold_scale: float
    observable_feature_groups: tuple[str, ...]
    limited_feature_groups: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "score_multiplier": self.score_multiplier,
            "threshold_scale": self.threshold_scale,
            "observable_feature_groups": list(self.observable_feature_groups),
            "limited_feature_groups": list(self.limited_feature_groups),
            "decision_gate": False,
        }


_POLICIES: dict[str, ActionViewPolicy] = {
    "lunge": ActionViewPolicy(
        frozenset({"front", "side"}),
        frozenset({"NOT_DEEP_ENOUGH", "LEAN_TOO_MUCH", "STAND_EXTENSION", "LOW_VISIBILITY"}),
        frozenset({"NOT_DEEP_ENOUGH", "LEAN_TOO_MUCH", "STAND_EXTENSION", "LOW_VISIBILITY"}),
    ),
    "wall_ball": ActionViewPolicy(
        frozenset({"front", "side"}),
        frozenset({"KNEES_CAVE_IN", "HEEL_RISE", "NOT_FULL_EXTENSION", "LOW_VISIBILITY"}),
        frozenset({"SQUAT_NOT_DEEP", "HEEL_RISE", "NOT_FULL_EXTENSION", "LOW_VISIBILITY"}),
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
        frozenset({"front", "side"}),
        frozenset({"CHEST_NOT_LOW", "FEET_STAGGERED", "EXTRA_STEPS", "HIPS_TOO_HIGH_IN_BOTTOM", "LOW_VISIBILITY"}),
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


_FULL = ("joint_angles", "body_center", "normalized_rom", "phase_order")
_VIEW_CAPABILITIES: dict[str, dict[str, ActionViewCapability]] = {
    "lunge": {
        "side": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
        "front": ActionViewCapability("usable", 0.92, 1.08, _FULL, ("sagittal_depth",)),
    },
    "wall_ball": {
        "front": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
        "side": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
    },
    "farmers_carry": {
        "front": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
        "side": ActionViewCapability("usable", 0.90, 1.10, _FULL, ("bilateral_symmetry",)),
    },
    "rowing": {
        "side": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
        "front": ActionViewCapability("not_recommended", 0.72, 1.20, ("body_center", "phase_order"), ("sagittal_rom",)),
    },
    "skierg": {
        "front": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
        "side": ActionViewCapability("usable", 0.90, 1.10, _FULL, ("bilateral_symmetry",)),
    },
    "burpee_broad_jump": {
        "side": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
        "front": ActionViewCapability("usable", 0.88, 1.12, _FULL, ("forward_displacement",)),
    },
    "sled_push": {
        "side": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
        "front": ActionViewCapability("not_recommended", 0.74, 1.20, ("phase_order", "bilateral_symmetry"), ("torso_lean", "forward_displacement")),
    },
    "sled_pull": {
        "side": ActionViewCapability("recommended", 1.00, 1.00, _FULL),
        "front": ActionViewCapability("not_recommended", 0.76, 1.20, ("phase_order", "bilateral_symmetry"), ("torso_lean", "pull_rom")),
    },
}


_UNKNOWN_CAPABILITY = ActionViewCapability(
    "usable", 1.0, 1.0, ("joint_angles", "body_center", "normalized_rom", "phase_order")
)


def action_view_capability(action: str, camera_view: str) -> ActionViewCapability:
    """Return conservative capability metadata; labels alone never reject a rep."""
    profile = view_profile(camera_view)
    if profile == "unknown":
        return _UNKNOWN_CAPABILITY
    action_matrix = _VIEW_CAPABILITIES.get(_action_key(action))
    if action_matrix is None:
        return _UNKNOWN_CAPABILITY
    return action_matrix.get(
        profile,
        ActionViewCapability("unjudgeable", 0.55, 1.30, ("phase_order",), ("joint_angles", "normalized_rom")),
    )


def action_view_capability_matrix() -> dict[str, dict[str, dict[str, object]]]:
    return {
        action: {view: capability.as_dict() for view, capability in views.items()}
        for action, views in _VIEW_CAPABILITIES.items()
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
    not_recommended = profile not in policy.preferred
    if not_recommended:
        preferred_text = "正面" if "front" in policy.preferred else "侧面"
        filtered.append(
            FeedbackMessage(
                level="info",
                code="CAMERA_VIEW_NOT_RECOMMENDED",
                text=(
                    f"当前视角不是推荐视角；建议使用{preferred_text}视角以提高稳定性，"
                    "系统仍将按可观测人体证据继续分析"
                ),
                confidence=1.0,
            )
        )
    return filtered, not_recommended


def action_view_suitability(
    action: str,
    camera_view: str,
) -> bool | None:
    profile = view_profile(camera_view)
    policy = _POLICIES.get(_action_key(action))
    if policy is None or profile == "unknown":
        return None
    return profile in policy.preferred


def recommended_camera_views(action: str) -> tuple[str, ...]:
    """Return recommendation metadata without implying a decision gate."""
    policy = _POLICIES.get(_action_key(action))
    if policy is None:
        return ()
    return tuple(
        profile for profile in ("front", "side") if profile in policy.preferred
    )


__all__ = [
    "CAMERA_VIEWS",
    "ActionViewPolicy",
    "ActionViewCapability",
    "ViewCapabilityLevel",
    "action_view_capability",
    "action_view_capability_matrix",
    "action_view_suitability",
    "filter_feedback_for_view",
    "normalize_camera_view",
    "next_camera_view",
    "recommended_camera_views",
    "view_profile",
]
