from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyrox.action_names import HYROX_ACTION_NAMES
from hyrox.actions.burpee_broad_jump import BurpeeBroadJumpAnalyzer
from hyrox.actions.farmers_carry import FarmersCarryAnalyzer
from hyrox.actions.lunge import LungeAnalyzer
from hyrox.actions.rowing import RowingAnalyzer
from hyrox.actions.skierg import SkiErgAnalyzer
from hyrox.actions.sled_pull import SledPullAnalyzer
from hyrox.actions.sled_push import SledPushAnalyzer
from hyrox.actions.wall_ball import WallBallAnalyzer
from hyrox.base import BaseActionAnalyzer
from hyrox.config import (
    load_burpee_broad_jump_config,
    load_farmers_carry_config,
    load_lunge_config,
    load_rowing_config,
    load_skierg_config,
    load_sled_pull_config,
    load_sled_push_config,
    load_wall_ball_config,
    validate_action_config,
)


_ACTION_ANALYZERS: dict[str, type[BaseActionAnalyzer]] = {
    "lunge": LungeAnalyzer,
    "wall_ball": WallBallAnalyzer,
    "farmers_carry": FarmersCarryAnalyzer,
    "rowing": RowingAnalyzer,
    "skierg": SkiErgAnalyzer,
    "burpee_broad_jump": BurpeeBroadJumpAnalyzer,
    "sled_push": SledPushAnalyzer,
    "sled_pull": SledPullAnalyzer,
}
_ACTION_CONFIG_LOADERS = {
    "lunge": load_lunge_config,
    "wall_ball": load_wall_ball_config,
    "farmers_carry": load_farmers_carry_config,
    "rowing": load_rowing_config,
    "skierg": load_skierg_config,
    "burpee_broad_jump": load_burpee_broad_jump_config,
    "sled_push": load_sled_push_config,
    "sled_pull": load_sled_pull_config,
}

if tuple(_ACTION_ANALYZERS) != HYROX_ACTION_NAMES:
    raise RuntimeError("HYROX registry and action-name list are out of sync")


def create_action_analyzer(
    action_name: str,
    config: str | Path | Mapping[str, Any] | None = None,
    *,
    sensitivity: str = "medium",
    camera_view: str = "unknown",
    live_mode: bool = False,
) -> BaseActionAnalyzer:
    """Create an action analyzer using its default, file, or mapping config.

    ``config=None`` selects ``configs/hyrox/<action_name>.yaml``. A path uses
    that file instead, while a mapping is passed directly to the analyzer.
    """
    normalized_name = str(action_name).strip().lower()
    try:
        analyzer_class = _ACTION_ANALYZERS[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unknown HYROX action: {action_name}") from exc

    configured_path = None if config is None or isinstance(config, Mapping) else str(config)
    if configured_path and not Path(configured_path).is_file():
        raise FileNotFoundError(f"HYROX config not found: {configured_path}")
    config_data = (
        validate_action_config(
            normalized_name,
            config,
            path="<in-memory HYROX config>",
        )
        if isinstance(config, Mapping)
        else _ACTION_CONFIG_LOADERS[normalized_name](configured_path)
    )
    if live_mode and sensitivity != "low":
        # The realtime queue intentionally keeps only the newest frame. During
        # inference pressure a short terminal pose may therefore be observed
        # once even when the camera itself runs at 30/60 FPS. The ordered
        # endpoint tracker provides the anti-jitter guard, so a single confirmed
        # realtime frame is sufficient for medium/high sensitivity.
        config_data["stable_frames"] = 1
    analyzer = analyzer_class.from_config(config_data, sensitivity=sensitivity)
    analyzer.configure_feedback_limits(config_data)
    analyzer.set_camera_view(camera_view)
    return analyzer


__all__ = ["create_action_analyzer"]
