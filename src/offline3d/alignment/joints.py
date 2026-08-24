from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def joint_overlap(
    mediapipe: Mapping[str, Any],
    reconstructed: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe name overlap without coercing WHAM into MediaPipe's topology."""

    left = set(mediapipe)
    right = set(reconstructed)
    return {
        "shared_joint_names": sorted(left & right),
        "mediapipe_only_joint_names": sorted(left - right),
        "reconstructed_only_joint_names": sorted(right - left),
        "topology_conversion_applied": False,
    }


def landmark_list_to_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in value:
        if isinstance(item, Mapping) and item.get("name"):
            result[str(item["name"])] = dict(item)
    return result
