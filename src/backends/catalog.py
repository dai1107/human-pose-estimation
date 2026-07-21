"""Pose backend support tiers shared by product entry points."""

from __future__ import annotations


PRODUCT_BACKEND = "mediapipe"
PRODUCT_BACKENDS = frozenset({PRODUCT_BACKEND})
EXPERIMENTAL_BACKENDS = frozenset(
    {
        "yolo-pose",
        "yolo-mediapipe",
        "yolo-guided-mediapipe",
        "rtmw-wholebody",
        "yolo-rtmw-wholebody",
    }
)


def backend_tier(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in PRODUCT_BACKENDS or normalized == "auto":
        return "product"
    if normalized in EXPERIMENTAL_BACKENDS or normalized.startswith("yolo-"):
        return "experimental"
    return "unknown"


def is_experimental_backend(name: str) -> bool:
    return backend_tier(name) == "experimental"


__all__ = [
    "EXPERIMENTAL_BACKENDS",
    "PRODUCT_BACKEND",
    "PRODUCT_BACKENDS",
    "backend_tier",
    "is_experimental_backend",
]
