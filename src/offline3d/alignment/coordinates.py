from __future__ import annotations


def coordinate_relationship(
    mediapipe_coordinate_system: str,
    reconstructed_coordinate_system: str,
) -> dict[str, object]:
    """Make coordinate non-equivalence explicit until extrinsics are solved."""

    same_label = bool(
        mediapipe_coordinate_system
        and reconstructed_coordinate_system
        and mediapipe_coordinate_system == reconstructed_coordinate_system
    )
    return {
        "mediapipe_coordinate_system": mediapipe_coordinate_system,
        "reconstructed_coordinate_system": reconstructed_coordinate_system,
        "coordinate_transform_applied": False,
        "direct_position_comparison_allowed": False,
        "labels_match": same_label,
        "reason": (
            "camera extrinsics/world trajectory transform has not been solved"
        ),
    }
