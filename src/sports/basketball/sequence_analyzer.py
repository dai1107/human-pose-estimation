from __future__ import annotations

from math import isfinite
from typing import Any

from .schema import ShotEvent


def normalize_event_name(event_name: str, shooting_side: str) -> str:
    if event_name.startswith(f"{shooting_side}_"):
        return "shooting_side_" + event_name.removeprefix(f"{shooting_side}_")
    return event_name


def analyze_event_sequence(events: list[ShotEvent], config: dict[str, Any], shooting_side: str) -> dict[str, Any]:
    template = config.get("sequence_template", {})
    expected = list(template.get("events", []))
    rules = dict(template.get("rules", {}))
    overlap_ms = float(rules.get("allow_event_overlap_ms", 80))
    event_map = {normalize_event_name(event.event, shooting_side): event for event in events}
    missing = [name for name in expected if event_map.get(name) is None or event_map[name].timestamp_ms is None]
    comparisons: list[dict[str, Any]] = []
    order_ok = True
    previous_name: str | None = None
    previous_ts: int | None = None
    for name in expected:
        event = event_map.get(name)
        if event is None or event.timestamp_ms is None:
            continue
        if previous_ts is not None and previous_name is not None:
            delta = event.timestamp_ms - previous_ts
            status = "overlap" if abs(delta) <= overlap_ms else ("in_order" if delta > 0 else "early")
            if status == "early":
                order_ok = False
            comparisons.append({"from": previous_name, "to": name, "delta_ms": delta, "status": status})
        previous_name = name
        previous_ts = event.timestamp_ms
    return {
        "template_name": template.get("template_name", config.get("template_name", "set_shot_chain_v1")),
        "expected_order": expected,
        "events": {event.event: event.to_dict() for event in events},
        "missing_events": missing,
        "pairwise_timing": comparisons,
        "order_consistent_with_template": order_ok and (not missing or bool(rules.get("allow_missing_events", True))),
        "interpretation_note": "Events describe keypoint speed and angular-velocity timing only; they do not measure real force or joint torque.",
    }

