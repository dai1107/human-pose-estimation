"""HYROX analyzer selection and per-frame fault isolation."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer

LOGGER = logging.getLogger("pose.desktop")


def runtime_hyrox_config_path(
    action_name: str,
    *,
    startup_action: str,
    startup_config: str | None,
) -> str | None:
    """Only reuse an explicit config for the action it was supplied for."""
    return startup_config if startup_config and action_name == startup_action else None


class HyroxAnalysisController:
    """Own the active analyzer without coupling it to the UI event loop."""

    def __init__(
        self,
        *,
        action: str,
        sensitivity: str,
        camera_view: str,
        startup_config: str | None,
        live_mode: bool,
    ) -> None:
        self.startup_action = action
        self.startup_config = startup_config
        self.sensitivity = sensitivity
        self.camera_view = camera_view
        self.live_mode = live_mode
        self.action = "none"
        self.analyzer = None
        self.extraction_error_reported = False
        self.analysis_error_reported = False
        self.switch(action)

    @property
    def enabled(self) -> bool:
        return self.analyzer is not None

    def switch(self, action: str) -> None:
        self.analyzer = (
            create_action_analyzer(
                action,
                runtime_hyrox_config_path(
                    action,
                    startup_action=self.startup_action,
                    startup_config=self.startup_config,
                ),
                sensitivity=self.sensitivity,
                camera_view=self.camera_view,
                live_mode=self.live_mode,
            )
            if action != "none"
            else None
        )
        self.action = action
        self.analysis_error_reported = False

    def set_camera_view(self, camera_view: str) -> None:
        self.camera_view = camera_view
        if self.analyzer is not None:
            self.analyzer.set_camera_view(camera_view)
            self.analyzer.reset()

    def update(
        self,
        keypoints: Sequence[object],
        *,
        has_pose: bool,
        timestamp_ms: int,
        image_width: int,
        image_height: int,
        segmentation_mask: object | None,
        three_d_kinematics: Mapping[str, object] | None = None,
        extract_when_disabled: bool = False,
    ) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
        features = None
        state = None
        if (extract_when_disabled or self.enabled) and has_pose:
            try:
                features = extract_basic_pose_features(
                    keypoints,
                    image_width=image_width,
                    image_height=image_height,
                    segmentation_mask=segmentation_mask,
                )
                if isinstance(three_d_kinematics, Mapping):
                    features["three_d_kinematics"] = dict(three_d_kinematics)
            except Exception as exc:
                if not self.extraction_error_reported:
                    LOGGER.warning("HYROX debug extraction failed: %s", exc)
                    self.extraction_error_reported = True
        if self.analyzer is not None:
            try:
                state = self.analyzer.attach_view_context(
                    self.analyzer.update(features if has_pose else None, timestamp_ms=timestamp_ms)
                )
            except Exception as exc:
                if not self.analysis_error_reported:
                    LOGGER.warning("HYROX action analysis failed: %s", exc)
                    self.analysis_error_reported = True
        return features, state
