"""Display profiles and transient UI state for the desktop runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.biomechanics.landmarks import LANDMARK_NAMES

FACE_KEYPOINT_NAMES = frozenset(LANDMARK_NAMES[:11])
NO_FACE_KEYPOINT_NAMES = frozenset(LANDMARK_NAMES[11:])
UPPER_BODY_KEYPOINT_NAMES = frozenset(
    {
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip",
    }
)
LOWER_BODY_KEYPOINT_NAMES = frozenset(
    {
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    }
)
DRAW_MODE_KEYPOINT_NAMES: dict[str, frozenset[str] | None] = {
    "full": None,
    "no-face": NO_FACE_KEYPOINT_NAMES,
    "upper-body": UPPER_BODY_KEYPOINT_NAMES,
    "lower-body": LOWER_BODY_KEYPOINT_NAMES,
}


def visible_keypoint_names_for_mode(mode: str) -> set[str] | None:
    allowed = DRAW_MODE_KEYPOINT_NAMES.get(mode)
    return None if allowed is None else set(allowed)


def highlight_keypoint_names_for_mode(mode: str) -> set[str]:
    del mode
    return set()


@dataclass
class DesktopViewState:
    """Mutable display-only state kept separate from analysis state."""

    mirror_enabled: bool
    display_mode: str
    metrics_overlay_enabled: bool
    status_message: str = ""
    status_until: float = 0.0
    action_selector_open: bool = False

    def set_status(self, message: str, seconds: float = 2.5) -> None:
        self.status_message = message
        self.status_until = time.perf_counter() + seconds

    def visible_status(self) -> str:
        return self.status_message if time.perf_counter() < self.status_until else ""
