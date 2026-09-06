"""Swimming-specific tracking components kept separate from HYROX rules."""

from src.swimming.wrist_tracking import (
    ArmChainObservation,
    LKOpticalFlowWristTracker,
    OpticalFlowObservation,
    SwimWristFrame,
    SwimWristIdentityTracker,
    SwimWristTrackerConfig,
    WristCandidate,
    WristTrackSnapshot,
    hungarian_2x2,
    load_swim_wrist_tracker_config,
)
from src.swimming.wrist_appearance import (
    AppearanceUpdate,
    WristAppearanceBank,
    WristAppearanceDescriptor,
    WristAppearanceModel,
    extract_wrist_appearance,
)
from src.swimming.cotracker_backend import (
    CoTrackerAvailability,
    CoTrackerOfflineBackend,
    CoTrackerWindowResult,
)

__all__ = [
    "ArmChainObservation",
    "AppearanceUpdate",
    "CoTrackerAvailability",
    "CoTrackerOfflineBackend",
    "CoTrackerWindowResult",
    "LKOpticalFlowWristTracker",
    "OpticalFlowObservation",
    "SwimWristFrame",
    "SwimWristIdentityTracker",
    "SwimWristTrackerConfig",
    "WristCandidate",
    "WristAppearanceBank",
    "WristAppearanceDescriptor",
    "WristAppearanceModel",
    "WristTrackSnapshot",
    "hungarian_2x2",
    "extract_wrist_appearance",
    "load_swim_wrist_tracker_config",
]
