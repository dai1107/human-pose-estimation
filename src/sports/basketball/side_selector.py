from __future__ import annotations

from .schema import SHOOTING_SIDES


def validate_shooting_side(side: str) -> str:
    normalized = side.strip().lower()
    if normalized not in SHOOTING_SIDES:
        raise ValueError("shooting_side must be 'right' or 'left'")
    return normalized


def opposite_side(side: str) -> str:
    side = validate_shooting_side(side)
    return "left" if side == "right" else "right"


def side_field(side: str, field: str) -> str:
    return f"{validate_shooting_side(side)}_{field}"

