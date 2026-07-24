from __future__ import annotations

import cv2
import numpy as np

from tools.dataset.round7_objects import ACTION_OBJECT_CLASSES, candidate_region
from tools.dataset.round7_roi import affine_matrices, affine_roundtrip_error
from tools.dataset.round7_tracking import (
    DetectionCandidate,
    MultiPersonTrackManager,
    _intervals_from_locked_flags,
    _normalize_source_track_segments,
    _source_track_id_for_frame,
    _write_ignore_masks,
    bbox_iou,
    propose_target_track,
    relative_center_distance,
)
from tools.run_round7_tracking import _parse_segment_overrides, build_parser


def _candidate(
    bbox: tuple[float, float, float, float], confidence: float = 0.9
) -> DetectionCandidate:
    xy = np.zeros((17, 2), dtype=np.float32)
    keypoint_confidence = np.ones(17, dtype=np.float32)
    return DetectionCandidate(
        bbox=bbox,
        confidence=confidence,
        keypoints_xy=xy,
        keypoints_confidence=keypoint_confidence,
        appearance=np.ones(64, dtype=np.float32),
        skeleton=np.ones(12, dtype=np.float32) * 0.1,
    )


def test_track_manager_keeps_identity_by_motion_appearance_and_skeleton() -> None:
    manager = MultiPersonTrackManager(max_gap_frames=10)
    first = manager.update(
        [_candidate((10, 10, 110, 210)), _candidate((300, 20, 380, 180))],
        frame_index=0,
        width=640,
        height=480,
    )
    second = manager.update(
        [_candidate((18, 12, 118, 212)), _candidate((294, 22, 374, 182))],
        frame_index=5,
        width=640,
        height=480,
    )

    assert [item["track_id"] for item in first] == [
        "person_candidate_001",
        "person_candidate_002",
    ]
    assert [item["track_id"] for item in second] == [
        "person_candidate_001",
        "person_candidate_002",
    ]
    assert all(
        set(item["association_components"])
        == {
            "bbox_iou",
            "center_velocity",
            "appearance_embedding",
            "bone_proportion",
            "action_continuity",
        }
        for item in second
    )


def test_target_proposal_balances_coverage_area_motion_and_confidence() -> None:
    selected = propose_target_track(
        [
            {
                "track_id": "background",
                "detection_count": 50,
                "mean_bbox_area": 1000,
                "center_travel_pixels": 20,
                "mean_confidence": 0.8,
            },
            {
                "track_id": "athlete",
                "detection_count": 50,
                "mean_bbox_area": 10000,
                "center_travel_pixels": 300,
                "mean_confidence": 0.9,
            },
        ]
    )

    assert selected == "athlete"


def test_bbox_metrics_are_bounded_and_geometric() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert relative_center_distance((0, 0, 10, 10), (0, 0, 10, 10)) == 0.0


def test_every_round7_object_class_has_bounded_candidate_region() -> None:
    required = {
        "wall_ball",
        "wall_target",
        "sled",
        "rope",
        "erg_handle",
        "erg_display_roi",
        "farmers_carry_weight",
        "lunge_load",
        "lane_or_finish_line",
        "floor_region",
    }
    assert {
        item for values in ACTION_OBJECT_CLASSES.values() for item in values
    } == required
    for object_class in required:
        bbox, method, confidence = candidate_region(
            object_class,
            (100.0, 100.0, 300.0, 450.0),
            width=720,
            height=1280,
        )
        assert 0 <= bbox[0] < bbox[2] <= 719
        assert 0 <= bbox[1] < bbox[3] <= 1279
        assert method
        assert 0 < confidence < 0.5


def test_roi_affine_is_reversible() -> None:
    bbox = (50.0, 100.0, 450.0, 900.0)
    forward, inverse = affine_matrices(bbox)

    assert np.allclose(np.asarray(inverse) @ np.asarray(forward), np.eye(3))
    assert affine_roundtrip_error(bbox) <= 1e-9


def test_round7_cli_requires_explicit_review_approval_flag() -> None:
    args = build_parser().parse_args(
        [
            "--approve-reviewed-proposals",
            "--reviewer-id",
            "reviewer-1",
            "--target-override",
            "phone_lunge_001=person_candidate_002",
            "--target-segment",
            "phone_sled_push_005=person_candidate_001:0-549",
            "--target-segment",
            "phone_sled_push_005=person_candidate_002:550-607",
        ]
    )

    assert args.approve_reviewed_proposals is True
    assert args.reviewer_id == "reviewer-1"
    assert args.target_override == [
        "phone_lunge_001=person_candidate_002"
    ]
    assert len(args.target_segment) == 2


def test_split_source_tracks_map_to_one_canonical_target() -> None:
    parsed = _parse_segment_overrides(
        [
            "phone_sled_push_005=person_candidate_001:0-549",
            "phone_sled_push_005=person_candidate_002:550-607",
        ]
    )
    segments = _normalize_source_track_segments(
        parsed["phone_sled_push_005"], 608
    )

    assert _source_track_id_for_frame(segments, 549) == "person_candidate_001"
    assert _source_track_id_for_frame(segments, 550) == "person_candidate_002"


def test_formal_intervals_bind_canonical_not_source_track() -> None:
    frames = [
        {
            "frame_index": 0,
            "target_track_id": "target_athlete_001",
            "source_candidate_track_id": "person_candidate_001",
            "candidates": [{"track_id": "person_candidate_001"}],
            "target_locked": True,
            "events": [],
        },
        {
            "frame_index": 1,
            "target_track_id": "target_athlete_001",
            "source_candidate_track_id": "person_candidate_002",
            "candidates": [{"track_id": "person_candidate_002"}],
            "target_locked": True,
            "events": [],
        },
    ]
    locked, excluded = _intervals_from_locked_flags(
        frames, "target_athlete_001"
    )

    assert excluded == []
    assert locked == [
        {
            "start_frame": 0,
            "end_frame": 1,
            "target_locked": True,
            "target_track_id": "target_athlete_001",
            "reasons": [],
        }
    ]


def test_ignore_mask_excludes_all_source_tracks_of_same_target(tmp_path) -> None:
    references = _write_ignore_masks(
        [
            {
                "frame_index": 7,
                "candidates": [
                    {
                        "track_id": "person_candidate_001",
                        "bbox_xyxy": [0, 0, 5, 5],
                    },
                    {
                        "track_id": "person_candidate_002",
                        "bbox_xyxy": [5, 5, 10, 10],
                    },
                    {
                        "track_id": "person_candidate_003",
                        "bbox_xyxy": [12, 12, 18, 18],
                    },
                ],
            }
        ],
        target_source_track_ids=[
            "person_candidate_001",
            "person_candidate_002",
        ],
        width=20,
        height=20,
        output_dir=tmp_path,
    )
    mask = cv2.imread(
        str(tmp_path / references[7]), cv2.IMREAD_GRAYSCALE
    )

    assert mask[2, 2] == 0
    assert mask[7, 7] == 0
    assert mask[15, 15] == 255
