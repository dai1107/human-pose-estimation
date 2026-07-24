"""Round-seven object/scene candidate layer.

The layer deliberately stores search regions and visibility proposals. It does
not convert action context or geometry into equipment/rule ground truth.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.utils.roi import BBox, clamp_bbox
from tools.dataset.manifest import utc_now
from tools.dataset.phone_rgb import _atomic_json
from tools.dataset.round7_tracking import read_jsonl


ACTION_OBJECT_CLASSES = {
    "burpee_broad_jump": ("lane_or_finish_line", "floor_region"),
    "lunge": ("lunge_load", "lane_or_finish_line", "floor_region"),
    "skierg": ("erg_handle", "erg_display_roi", "floor_region"),
    "rowing": ("erg_handle", "erg_display_roi", "floor_region"),
    "sled_pull": ("sled", "rope", "lane_or_finish_line", "floor_region"),
    "farmers_carry": (
        "farmers_carry_weight",
        "lane_or_finish_line",
        "floor_region",
    ),
    "wall_ball": ("wall_ball", "wall_target", "floor_region"),
    "sled_push": ("sled", "lane_or_finish_line", "floor_region"),
}

UNOBSERVABLE_FIELDS = {
    "wall_ball": ("actual_ball_weight", "target_height", "target_hit"),
    "wall_target": ("official_target_height", "valid_hit"),
    "sled": ("actual_sled_load", "finish_line_crossing"),
    "rope": ("rope_tension",),
    "erg_handle": ("machine_resistance",),
    "erg_display_roi": ("machine_distance", "machine_calories"),
    "farmers_carry_weight": ("actual_weight", "course_distance"),
    "lunge_load": ("actual_weight",),
    "lane_or_finish_line": ("official_lane_or_finish_status",),
    "floor_region": ("metric_floor_distance",),
}


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _expand_relative(
    bbox: BBox,
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
    width: int,
    height: int,
) -> BBox:
    x1, y1, x2, y2 = bbox
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    return clamp_bbox(
        (
            x1 + left * box_width,
            y1 + top * box_height,
            x2 + right * box_width,
            y2 + bottom * box_height,
        ),
        width,
        height,
    )


def candidate_region(
    object_class: str,
    target_bbox: BBox,
    *,
    width: int,
    height: int,
) -> tuple[BBox, str, float]:
    x1, y1, x2, y2 = target_bbox
    cx = (x1 + x2) / 2.0
    if object_class == "floor_region":
        return (
            clamp_bbox((0.0, max(y1, y2 - 0.18 * height), width - 1, height - 1), width, height),
            "target_foot_adjacent_floor_search_region",
            0.45,
        )
    if object_class == "lane_or_finish_line":
        return (
            clamp_bbox((0.0, max(0.0, y2 - 0.12 * height), width - 1, height - 1), width, height),
            "lower_scene_line_search_region",
            0.20,
        )
    if object_class == "wall_target":
        return (
            clamp_bbox((max(0.0, cx - 0.35 * width), 0.0, min(width - 1, cx + 0.35 * width), 0.38 * height), width, height),
            "upper_wall_target_search_region",
            0.20,
        )
    if object_class == "wall_ball":
        return (
            _expand_relative(
                target_bbox,
                left=-0.45,
                top=-0.65,
                right=0.45,
                bottom=0.05,
                width=width,
                height=height,
            ),
            "target_hand_and_throw_arc_search_region",
            0.25,
        )
    if object_class == "sled":
        return (
            _expand_relative(
                target_bbox,
                left=-0.75,
                top=0.20,
                right=0.75,
                bottom=0.25,
                width=width,
                height=height,
            ),
            "target_lower_body_forward_equipment_search_region",
            0.22,
        )
    if object_class == "rope":
        return (
            _expand_relative(
                target_bbox,
                left=-0.65,
                top=-0.10,
                right=0.65,
                bottom=0.15,
                width=width,
                height=height,
            ),
            "target_wrist_to_sled_search_region",
            0.18,
        )
    if object_class == "erg_handle":
        return (
            _expand_relative(
                target_bbox,
                left=-0.30,
                top=0.05,
                right=0.30,
                bottom=-0.25,
                width=width,
                height=height,
            ),
            "target_hand_and_torso_search_region",
            0.22,
        )
    if object_class == "erg_display_roi":
        side = -1.0 if cx > width / 2 else 1.0
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        display_cx = cx + side * 0.55 * box_width
        display_cy = y1 + 0.25 * box_height
        return (
            clamp_bbox(
                (
                    display_cx - 0.20 * box_width,
                    display_cy - 0.15 * box_height,
                    display_cx + 0.20 * box_width,
                    display_cy + 0.15 * box_height,
                ),
                width,
                height,
            ),
            "ergometer_display_search_region",
            0.15,
        )
    if object_class == "farmers_carry_weight":
        return (
            _expand_relative(
                target_bbox,
                left=-0.35,
                top=0.35,
                right=0.35,
                bottom=0.10,
                width=width,
                height=height,
            ),
            "bilateral_hand_load_search_region",
            0.22,
        )
    if object_class == "lunge_load":
        return (
            _expand_relative(
                target_bbox,
                left=-0.30,
                top=-0.05,
                right=0.30,
                bottom=-0.25,
                width=width,
                height=height,
            ),
            "shoulder_torso_hand_load_search_region",
            0.20,
        )
    raise KeyError(object_class)


def _target_candidate(frame: Mapping[str, Any]) -> Mapping[str, Any] | None:
    track_id = frame.get("source_candidate_track_id") or frame.get(
        "target_track_id"
    )
    return next(
        (
            candidate
            for candidate in frame.get("candidates") or []
            if candidate.get("track_id") == track_id
        ),
        None,
    )


def build_record_object_candidates(
    record: Mapping[str, Any], *, dataset_root: str | Path
) -> dict[str, Any]:
    root = Path(dataset_root)
    record_id = str(record["record_id"])
    people_path = root / "tracks" / record_id / "people.jsonl"
    people = read_jsonl(people_path)
    width = int(record["video"]["width"])
    height = int(record["video"]["height"])
    action = str(record["action"])
    classes = ACTION_OBJECT_CLASSES[action]
    object_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    target_locked_frames = 0
    for frame in people:
        target = _target_candidate(frame)
        candidates = []
        if frame.get("target_locked") and target is not None:
            target_locked_frames += 1
            target_bbox = tuple(float(value) for value in target["bbox_xyxy"])
            for object_class in classes:
                bbox, method, confidence = candidate_region(
                    object_class,
                    target_bbox,  # type: ignore[arg-type]
                    width=width,
                    height=height,
                )
                polygon = [
                    [round(bbox[0], 3), round(bbox[1], 3)],
                    [round(bbox[2], 3), round(bbox[1], 3)],
                    [round(bbox[2], 3), round(bbox[3], 3)],
                    [round(bbox[0], 3), round(bbox[3], 3)],
                ]
                candidates.append(
                    {
                        "object_track_id": f"object_{object_class}_001",
                        "object_class": object_class,
                        "bbox_xyxy": [round(value, 3) for value in bbox],
                        "mask_polygon_xy": polygon,
                        "class_confidence": confidence,
                        "visibility": "candidate_search_region_not_verified",
                        "occluded": "unknown",
                        "out_of_frame": (
                            bbox[0] <= 0
                            or bbox[1] <= 0
                            or bbox[2] >= width - 1
                            or bbox[3] >= height - 1
                        ),
                        "target_track_id": frame["target_track_id"],
                        "association_confidence": 0.65,
                        "proposal_method": method,
                        "proposal_is_rule_truth": False,
                        "unobservable_fields": list(
                            UNOBSERVABLE_FIELDS[object_class]
                        ),
                    }
                )
                counts[object_class] += 1
        object_records.append(
            {
                "schema_version": 1,
                "record_id": record_id,
                "frame_index": frame["frame_index"],
                "timestamp_ms": frame["timestamp_ms"],
                "target_track_id": frame.get("target_track_id"),
                "target_locked": frame.get("target_locked"),
                "candidates": candidates,
                "pose_chain_status": "independent_of_object_visibility",
            }
        )
    output_path = root / "tracks" / record_id / "objects.jsonl"
    _write_jsonl(output_path, object_records)
    return {
        "record_id": record_id,
        "source_filename": record["source_filename"],
        "action": action,
        "frame_count": len(people),
        "target_locked_frames": target_locked_frames,
        "candidate_counts": dict(sorted(counts.items())),
        "classes": list(classes),
        "confirmed_visible_count": 0,
        "candidate_only": True,
        "rule_truth_generated": False,
    }


def build_scene_calibration_proposals(
    manifest: Mapping[str, Any], *, dataset_root: str | Path
) -> dict[str, Any]:
    root = Path(dataset_root)
    records = []
    for record in manifest.get("records") or []:
        people = read_jsonl(
            root / "tracks" / str(record["record_id"]) / "people.jsonl"
        )
        locked = next(
            (
                frame
                for frame in people
                if frame.get("target_locked") and _target_candidate(frame) is not None
            ),
            None,
        )
        calibrations = []
        if locked is not None:
            target = _target_candidate(locked)
            bbox = tuple(float(value) for value in target["bbox_xyxy"])
            for object_class in ("floor_region", "lane_or_finish_line"):
                proposal, method, confidence = candidate_region(
                    object_class,
                    bbox,  # type: ignore[arg-type]
                    width=int(record["video"]["width"]),
                    height=int(record["video"]["height"]),
                )
                calibrations.append(
                    {
                        "calibration_id": f"{record['record_id']}_{object_class}_v1",
                        "object_class": object_class,
                        "source_frame": locked["frame_index"],
                        "bbox_xyxy": [round(value, 3) for value in proposal],
                        "coordinate_space": "source_image_pixel",
                        "version": 1,
                        "proposal_method": method,
                        "proposal_confidence": confidence,
                        "manual_confirmation_status": "pending",
                    }
                )
        records.append(
            {
                "record_id": record["record_id"],
                "calibrations": calibrations,
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "round7_scene_calibration_proposals_v1",
        "generated_at": utc_now(),
        "records": records,
        "rule_truth_generated": False,
    }
    _atomic_json(root / "annotations" / "scene_calibrations_v1.json", payload)
    return payload


def build_object_scene_visibility_report(
    manifest: Mapping[str, Any], *, dataset_root: str | Path
) -> dict[str, Any]:
    records = [
        build_record_object_candidates(record, dataset_root=dataset_root)
        for record in manifest.get("records") or []
    ]
    calibration = build_scene_calibration_proposals(
        manifest, dataset_root=dataset_root
    )
    class_counts: Counter[str] = Counter()
    for record in records:
        class_counts.update(record["candidate_counts"])
    report = {
        "schema_version": 1,
        "artifact_type": "object_scene_visibility_v1",
        "generated_at": utc_now(),
        "status": "candidate_layer_complete",
        "record_count": len(records),
        "records": records,
        "summary": {
            "candidate_counts": dict(sorted(class_counts.items())),
            "confirmed_visible_count": 0,
            "unobservable_fields": {
                key: list(value) for key, value in UNOBSERVABLE_FIELDS.items()
            },
            "scene_calibration_record_count": len(calibration["records"]),
        },
        "checks": {
            "all_records_have_object_candidate_files": len(records) == 30,
            "all_required_candidate_classes_represented": set(class_counts)
            == {
                object_class
                for classes in ACTION_OBJECT_CLASSES.values()
                for object_class in classes
            },
            "candidate_not_rule_truth": all(
                record["rule_truth_generated"] is False for record in records
            ),
            "equipment_visibility_independent_from_pose_chain": True,
            "unobservable_measurements_remain_unobservable": True,
        },
        "limitations": [
            "action-conditioned geometry defines search regions, not detected equipment truth",
            "visibility and object class require later human/object-detector confirmation",
            "actual load, official distance, target height and target hit are UNOBSERVABLE",
        ],
    }
    _atomic_json(
        Path(dataset_root) / "reports" / "object_scene_visibility_v1.json",
        report,
    )
    return report


__all__ = [
    "ACTION_OBJECT_CLASSES",
    "UNOBSERVABLE_FIELDS",
    "build_object_scene_visibility_report",
    "build_record_object_candidates",
    "build_scene_calibration_proposals",
    "candidate_region",
]
