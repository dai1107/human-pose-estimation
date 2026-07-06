from __future__ import annotations

from math import nan
from typing import Sequence

from .landmarks import coerce_landmark
from .types import LandmarkPoint


HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

HAND_LANDMARK_INDEX: dict[str, int] = {name: index for index, name in enumerate(HAND_LANDMARK_NAMES)}

FINGER_LANDMARK_GROUPS: tuple[tuple[str, ...], ...] = (
    ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"),
    ("index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip"),
    ("middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip"),
    ("ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip"),
    ("pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
)

SUPPLEMENTAL_FINGER_JOINTS: tuple[tuple[str, int], ...] = tuple(
    (name, HAND_LANDMARK_INDEX[name])
    for finger in FINGER_LANDMARK_GROUPS
    for name in finger
)

SUPPLEMENTAL_FINGER_DISPLAY_INDICES: frozenset[int] = frozenset(
    HAND_LANDMARK_INDEX[name]
    for finger in FINGER_LANDMARK_GROUPS
    for name in finger
)

SUPPLEMENTAL_FINGER_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (HAND_LANDMARK_INDEX[start], HAND_LANDMARK_INDEX[end])
    for finger in FINGER_LANDMARK_GROUPS
    for start, end in zip(finger, finger[1:])
)


def empty_hand_landmarks(count: int = 21) -> list[LandmarkPoint]:
    return [LandmarkPoint(nan, nan, nan, 0.0, 0.0) for _ in range(count)]


def coerce_hand_landmarks(points: Sequence[object] | None, expected_count: int = 21) -> list[LandmarkPoint]:
    if points is None:
        return []
    landmarks = [coerce_landmark(point) for point in points]
    if len(landmarks) < expected_count:
        landmarks.extend(empty_hand_landmarks(expected_count - len(landmarks)))
    return landmarks[:expected_count]


def hand_landmark_name(side: str, index: int) -> str:
    side_key = side.strip().lower().replace(" ", "_") or "unknown"
    if 0 <= index < len(HAND_LANDMARK_NAMES):
        return f"{side_key}_hand_{HAND_LANDMARK_NAMES[index]}"
    return f"{side_key}_hand_landmark_{index}"
