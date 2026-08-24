from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..alignment import MotionAlignmentResult
from ..base import Offline3DResult
from ..wham.hyrox_fusion import WhamHyroxFusionResult


def build_advanced_report(
    mediapipe_frames: Sequence[Mapping[str, Any]],
    *,
    wham: Offline3DResult | None,
    alignment: MotionAlignmentResult | None,
    opencap: Offline3DResult | None,
    hyrox_assist: WhamHyroxFusionResult | None = None,
) -> dict[str, Any]:
    """Compose references while keeping the formal rule result independent."""

    return {
        "schema_version": "advanced_report_v1",
        "result_role": "advanced_3d_reference",
        "formal_hyrox_result_source": "current MediaPipe 2D HYROX rule engine",
        "formal_rule_replacement_allowed": False,
        "absolute_joint_angle_mae_allowed": False,
        "comparison_policy": [
            "trend",
            "lowest_point_timing",
            "full_extension_timing",
            "range_of_motion",
            "cycle_duration",
        ],
        "mediapipe": {
            "status": "COMPLETED" if mediapipe_frames else "UNAVAILABLE",
            "frame_count": len(mediapipe_frames),
            "selected_angle_definition": "current view-appropriate HYROX rule angle",
            "canonical_3d_definition": "MediaPipe world landmarks in body canonical frame",
        },
        "wham": (
            wham.as_dict()
            if wham is not None
            else Offline3DResult.unavailable("wham", "not requested").as_dict()
        ),
        "motion_alignment": (
            alignment.as_dict()
            if alignment is not None
            else MotionAlignmentResult.unavailable("not requested").as_dict()
        ),
        "hyrox_assist": (
            hyrox_assist.as_dict()
            if hyrox_assist is not None
            else {
                "schema_version": "wham_hyrox_fusion_v1",
                "status": "NOT_REQUESTED",
                "formal_rule_replacement_allowed": False,
            }
        ),
        "opencap": (
            opencap.as_dict()
            if opencap is not None
            else {
                "backend": "opencap",
                "status": "NOT_REQUESTED",
                "reference_source": "not used in WHAM-only advanced mode",
            }
        ),
        "is_ground_truth": False,
    }
