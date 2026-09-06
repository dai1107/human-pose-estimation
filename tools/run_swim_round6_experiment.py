"""Compare five swimming wrist tracking modes on existing marked videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends.mediapipe_backend import MediaPipeBackend  # noqa: E402
from src.swimming.cotracker_backend import (  # noqa: E402
    CoTrackerOfflineBackend,
)
from src.swimming.wrist_appearance import (  # noqa: E402
    WristAppearanceBank,
    extract_wrist_appearance,
)
from src.swimming.wrist_tracking import (  # noqa: E402
    SIDES,
    LKOpticalFlowWristTracker,
    SwimWristIdentityTracker,
    load_swim_wrist_tracker_config,
)
from tools.run_swim_wrist_tracking import (  # noqa: E402
    _arm_chains,
    _body_scale,
    _point,
    _position,
    _trajectory_jitter,
    _wrist_candidates,
)


MODE_NAMES = (
    "mediapipe_only",
    "pose_identity",
    "pose_lk",
    "pose_cotracker",
    "pose_cotracker_wristband",
)
DEFAULT_RECORDS = (
    (
        Path("游泳视频（标记左手）.mp4"),
        Path("outputs/swim_joint_inspect/left_anchors_clear.json"),
    ),
    (
        Path("游泳视频（标记右手）.mp4"),
        Path("outputs/swim_joint_inspect/right_anchors_dense.json"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Round-6 wristband + CoTracker swimming experiment."
    )
    parser.add_argument(
        "--record",
        nargs=2,
        metavar=("VIDEO", "ANCHORS"),
        action="append",
        default=[],
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_full.task"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/swim_wrist_tracking.yaml"),
    )
    parser.add_argument("--cotracker-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--allow-cotracker-download",
        action="store_true",
        help="Explicitly allow torch.hub to download official CoTracker code/weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/swim_wrist_round6"),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def load_anchor_record(
    video_path: str | Path, anchors_path: str | Path
) -> dict[str, Any]:
    video = Path(video_path)
    source = Path(anchors_path)
    if not video.is_file():
        raise FileNotFoundError(video)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    side = str(payload.get("side", "")).lower()
    if side not in SIDES:
        name = video.name.lower()
        side = "right" if "右" in name or "right" in name else "left"
    anchors = [
        {
            "frame": int(item["frame"]),
            "x": float(item["x"]),
            "y": float(item["y"]),
        }
        for item in payload.get("anchors", ())
        if isinstance(item, Mapping)
    ]
    return {
        "video": video,
        "anchors_path": source,
        "target_side": side,
        "anchors": anchors,
    }


def evaluate_record(
    record: Mapping[str, Any],
    *,
    model_path: str | Path,
    config_path: str | Path,
    cotracker: CoTrackerOfflineBackend,
    max_frames: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    video = Path(record["video"])
    config = load_swim_wrist_tracker_config(config_path)
    target_side = str(record["target_side"])
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open swimming video: {video}")
    source_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if max_frames is not None and int(max_frames) <= 0:
        capture.release()
        raise ValueError("max_frames must be a positive integer")
    frame_limit = (
        min(source_frames, int(max_frames))
        if max_frames is not None
        else source_frames
    )
    backend = MediaPipeBackend(
        model_path,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )
    identity_tracker = SwimWristIdentityTracker(config)
    lk_tracker = SwimWristIdentityTracker(config)
    appearance_tracker = SwimWristIdentityTracker(config)
    lk_flow = LKOpticalFlowWristTracker(config)
    appearance_flow = LKOpticalFlowWristTracker(config)
    appearance_bank = WristAppearanceBank(config)
    positions = {
        mode: {side: [] for side in SIDES}
        for mode in ("mediapipe_only", "pose_identity", "pose_lk", "pose_lk_wristband")
    }
    elbows: dict[str, list[tuple[float, float] | None]] = {
        side: [] for side in SIDES
    }
    scales: list[float] = []
    appearance_updates = 0
    appearance_rejections: dict[str, int] = {}
    pose_detected = 0
    processed = 0
    try:
        while processed < frame_limit:
            ok, source_frame = capture.read()
            if not ok:
                break
            frame = cv2.rotate(source_frame, cv2.ROTATE_90_CLOCKWISE)
            timestamp_ms = processed * 1000.0 / max(fps, 1e-6)
            result = backend.detect(frame, int(round(timestamp_ms)))
            pose_detected += int(bool(result.success))
            candidates = _wrist_candidates(result.keypoints)
            candidate_by_side = {
                candidate.semantic_side: candidate for candidate in candidates
            }
            chains = _arm_chains(result.keypoints)
            scale = _body_scale(result.keypoints)
            scales.append(scale)
            descriptors = {}
            for side in SIDES:
                wrist = candidate_by_side.get(side)
                elbow = chains[side].elbow
                elbows[side].append(elbow)
                descriptors[side] = (
                    extract_wrist_appearance(
                        frame,
                        wrist=wrist.position,
                        elbow=elbow,
                        roi_forearm_ratio=config.appearance_roi_forearm_ratio,
                    )
                    if wrist is not None and elbow is not None
                    else None
                )
                positions["mediapipe_only"][side].append(
                    wrist.position if wrist is not None else None
                )
            identity_frame = identity_tracker.update(
                candidates,
                chains,
                frame_index=processed,
                timestamp_ms=timestamp_ms,
                body_scale=scale,
            )
            flow = lk_flow.advance(
                frame, body_scale_px=scale * max(frame.shape[:2])
            )
            lk_frame = lk_tracker.update(
                candidates,
                chains,
                frame_index=processed,
                timestamp_ms=timestamp_ms,
                body_scale=scale,
                optical_flow=flow,
            )
            appearance_costs = appearance_bank.costs(descriptors)
            appearance_flow_values = appearance_flow.advance(
                frame, body_scale_px=scale * max(frame.shape[:2])
            )
            appearance_frame = appearance_tracker.update(
                candidates,
                chains,
                frame_index=processed,
                timestamp_ms=timestamp_ms,
                body_scale=scale,
                optical_flow=appearance_flow_values,
                appearance_costs=appearance_costs,
            )
            for mode, tracked in (
                ("pose_identity", identity_frame),
                ("pose_lk", lk_frame),
                ("pose_lk_wristband", appearance_frame),
            ):
                for side in SIDES:
                    positions[mode][side].append(tracked.tracks[side].position)
            _reanchor_flow(lk_flow, lk_frame, candidate_by_side, frame.shape)
            _reanchor_flow(
                appearance_flow,
                appearance_frame,
                candidate_by_side,
                frame.shape,
            )
            mapping_confidence = _mapping_confidence(appearance_frame)
            for side in SIDES:
                snapshot = appearance_frame.tracks[side]
                semantic = snapshot.observed_semantic_side
                descriptor = descriptors.get(semantic) if semantic else None
                update = appearance_bank.update(
                    side,
                    descriptor,
                    identity_confidence=snapshot.confidence * mapping_confidence,
                    visibility=snapshot.confidence,
                )
                if update.updated:
                    appearance_updates += 1
                else:
                    appearance_rejections[update.reason] = (
                        appearance_rejections.get(update.reason, 0) + 1
                    )
            processed += 1
    finally:
        capture.release()
        backend.close()

    mode_results: dict[str, dict[str, Any]] = {}
    anchor_rows: list[dict[str, Any]] = []
    switch_counts = {
        "mediapipe_only": identity_tracker.instantaneous_mapping_flip_count,
        "pose_identity": identity_tracker.confirmed_mapping_change_count,
        "pose_lk": lk_tracker.confirmed_mapping_change_count,
        "pose_lk_wristband": appearance_tracker.confirmed_mapping_change_count,
    }
    for mode in positions:
        metrics, rows = _mode_metrics(
            mode,
            positions[mode],
            record["anchors"],
            target_side=target_side,
            source_width=source_width,
            source_height=source_height,
            body_scales=scales,
            identity_switch_proxy_count=switch_counts[mode],
        )
        mode_results[mode] = metrics
        for row in rows:
            row["video"] = video.name
        anchor_rows.extend(rows)

    cotracker_positions: dict[str, list[tuple[float, float] | None]] | None = None
    cotracker_visibility: dict[str, list[float]] | None = None
    cotracker_reason = cotracker.availability.reason
    if cotracker.availability.available:
        cotracker_positions, cotracker_visibility, cotracker_reason = (
            _run_cotracker_windows(
                video,
                positions["pose_lk"],
                cotracker,
                frame_count=processed,
                config=config,
            )
        )
    if cotracker_positions is not None and cotracker_visibility is not None:
        metrics, rows = _mode_metrics(
            "pose_cotracker",
            cotracker_positions,
            record["anchors"],
            target_side=target_side,
            source_width=source_width,
            source_height=source_height,
            body_scales=scales,
            identity_switch_proxy_count=0,
        )
        metrics["mean_visibility"] = fmean(
            value
            for side in SIDES
            for value in cotracker_visibility[side]
        )
        mode_results["pose_cotracker"] = metrics
        anchor_rows.extend({**row, "video": video.name} for row in rows)
        fused, fusion_stats = _appearance_validate_cotracker(
            video,
            cotracker_positions,
            cotracker_visibility,
            elbows,
            appearance_bank,
            config=config,
        )
        metrics, rows = _mode_metrics(
            "pose_cotracker_wristband",
            fused,
            record["anchors"],
            target_side=target_side,
            source_width=source_width,
            source_height=source_height,
            body_scales=scales,
            identity_switch_proxy_count=int(fusion_stats["mapping_change_count"]),
        )
        metrics["appearance_fusion"] = fusion_stats
        mode_results["pose_cotracker_wristband"] = metrics
        anchor_rows.extend({**row, "video": video.name} for row in rows)
    else:
        for mode in ("pose_cotracker", "pose_cotracker_wristband"):
            mode_results[mode] = {
                "available": False,
                "reason": cotracker_reason,
                "anchor_count": 0,
            }
    summary = {
        "video": str(video.resolve()),
        "target_side": target_side,
        "processed_frame_count": processed,
        "pose_detection_rate": pose_detected / max(1, processed),
        "anchor_count": sum(
            int(item["frame"]) < processed for item in record["anchors"]
        ),
        "appearance": {
            "model": appearance_bank.as_dict(),
            "accepted_update_count": appearance_updates,
            "rejection_reason_counts": dict(sorted(appearance_rejections.items())),
            "roi_forearm_ratio": config.appearance_roi_forearm_ratio,
            "ema_alpha": config.appearance_ema_alpha,
            "offline_supporting_ablation": mode_results["pose_lk_wristband"],
        },
        "modes": {
            mode: mode_results[mode]
            for mode in MODE_NAMES
        },
        "supporting_modes": {
            "pose_lk_wristband": mode_results["pose_lk_wristband"]
        },
    }
    return summary, anchor_rows


def _run_cotracker_windows(
    video: Path,
    anchors: Mapping[str, Sequence[tuple[float, float] | None]],
    backend: CoTrackerOfflineBackend,
    *,
    frame_count: int,
    config: object,
) -> tuple[
    dict[str, list[tuple[float, float] | None]] | None,
    dict[str, list[float]] | None,
    str,
]:
    output = {side: [None] * frame_count for side in SIDES}
    visibility = {side: [0.0] * frame_count for side in SIDES}
    window = int(config.cotracker_window_frames)
    overlap = min(window - 1, int(config.cotracker_overlap_frames))
    step = max(1, window - overlap)
    capture = cv2.VideoCapture(str(video))
    try:
        for start in range(0, frame_count, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(min(window, frame_count - start)):
                ok, source_frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.rotate(source_frame, cv2.ROTATE_90_CLOCKWISE))
            if not frames:
                break
            queries = []
            query_sides = []
            height, width = frames[0].shape[:2]
            for side in SIDES:
                local = next(
                    (
                        index
                        for index in range(min(len(frames), overlap + 1))
                        if anchors[side][start + index] is not None
                    ),
                    None,
                )
                if local is None:
                    continue
                point = anchors[side][start + local]
                queries.append([local, point[0] * width, point[1] * height])
                query_sides.append(side)
            if not queries:
                continue
            result = backend.track_window(
                frames, np.asarray(queries, dtype=np.float32)
            )
            if not result.available:
                return None, None, result.reason
            for query_index, side in enumerate(query_sides):
                for local in range(len(frames)):
                    score = float(result.visibility[local, query_index])
                    global_index = start + local
                    if score < config.cotracker_visibility_threshold:
                        continue
                    if score >= visibility[side][global_index]:
                        point = result.tracks_normalized[local, query_index]
                        output[side][global_index] = (
                            float(point[0]),
                            float(point[1]),
                        )
                        visibility[side][global_index] = score
    finally:
        capture.release()
    return output, visibility, "accepted"


def _appearance_validate_cotracker(
    video: Path,
    positions: Mapping[str, Sequence[tuple[float, float] | None]],
    visibility: Mapping[str, Sequence[float]],
    elbows: Mapping[str, Sequence[tuple[float, float] | None]],
    bank: WristAppearanceBank,
    *,
    config: object,
) -> tuple[dict[str, list[tuple[float, float] | None]], dict[str, int]]:
    output = {side: list(positions[side]) for side in SIDES}
    capture = cv2.VideoCapture(str(video))
    pending_swap = 0
    swapped = False
    changes = 0
    conflict_rejections = 0
    try:
        for index in range(len(output["left"])):
            ok, source_frame = capture.read()
            if not ok:
                break
            frame = cv2.rotate(source_frame, cv2.ROTATE_90_CLOCKWISE)
            descriptors = {}
            for side in SIDES:
                point = output[side][index]
                elbow = elbows[side][index]
                descriptors[side] = (
                    extract_wrist_appearance(
                        frame,
                        wrist=point,
                        elbow=elbow,
                        roi_forearm_ratio=config.appearance_roi_forearm_ratio,
                    )
                    if point is not None and elbow is not None
                    else None
                )
            costs = bank.costs(descriptors)
            required = {
                (candidate, track) for candidate in SIDES for track in SIDES
            }
            if not required.issubset(costs):
                continue
            direct = costs[("left", "left")] + costs[("right", "right")]
            reverse = costs[("left", "right")] + costs[("right", "left")]
            proposed = reverse + 0.10 < direct
            if proposed != swapped:
                pending_swap += 1
                if pending_swap >= config.identity_confirmation_frames:
                    swapped = proposed
                    changes += 1
                    pending_swap = 0
            else:
                pending_swap = 0
            if swapped:
                output["left"][index], output["right"][index] = (
                    output["right"][index],
                    output["left"][index],
                )
            for side in SIDES:
                descriptor_side = _opposite(side) if swapped else side
                descriptor = descriptors[descriptor_side]
                cost = bank.models[side].cost(descriptor)
                descriptor_visibility = visibility[descriptor_side][index]
                if cost is not None and cost > 0.65 and descriptor_visibility < 0.75:
                    output[side][index] = None
                    conflict_rejections += 1
    finally:
        capture.release()
    return output, {
        "mapping_change_count": changes,
        "appearance_conflict_rejection_count": conflict_rejections,
    }


def _mode_metrics(
    mode: str,
    positions: Mapping[str, Sequence[tuple[float, float] | None]],
    anchors: Sequence[Mapping[str, Any]],
    *,
    target_side: str,
    source_width: int,
    source_height: int,
    body_scales: Sequence[float],
    identity_switch_proxy_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opposite = "right" if target_side == "left" else "left"
    rows = []
    for anchor in anchors:
        frame = int(anchor["frame"])
        if frame >= len(positions[target_side]):
            continue
        target = _source_pixels(
            positions[target_side][frame], source_width, source_height
        )
        other = _source_pixels(
            positions[opposite][frame], source_width, source_height
        )
        target_error = _pixel_error(target, anchor)
        other_error = _pixel_error(other, anchor)
        correct = (
            target_error is not None
            and (other_error is None or target_error < other_error)
        )
        rows.append(
            {
                "mode": mode,
                "frame_index": frame,
                "target_side": target_side,
                "anchor_x": anchor["x"],
                "anchor_y": anchor["y"],
                "target_error_px": target_error,
                "opposite_error_px": other_error,
                "target_identity_correct": correct,
                "target_available": target is not None,
            }
        )
    errors = [float(row["target_error_px"]) for row in rows if row["target_error_px"] is not None]
    correct_count = sum(bool(row["target_identity_correct"]) for row in rows)
    available_count = sum(bool(row["target_available"]) for row in rows)
    coverage_by_side = {
        side: sum(point is not None for point in positions[side])
        / max(1, len(positions[side]))
        for side in SIDES
    }
    jitters = {
        side: _trajectory_jitter(positions[side], body_scales) for side in SIDES
    }
    jitter_values = [value for value in jitters.values() if value is not None]
    return (
        {
            "available": True,
            "identity_switch_proxy_count": int(identity_switch_proxy_count),
            "anchor_count": len(rows),
            "anchor_target_available_count": available_count,
            "anchor_identity_correct_count": correct_count,
            "anchor_identity_correct_rate": correct_count / max(1, len(rows)),
            "anchor_target_error_mae_px": fmean(errors) if errors else None,
            "track_coverage_by_side": coverage_by_side,
            "mean_track_coverage": fmean(coverage_by_side.values()),
            "trajectory_jitter_body_by_side": jitters,
            "mean_trajectory_jitter_body": (
                fmean(jitter_values) if jitter_values else None
            ),
        },
        rows,
    )


def aggregate_experiment(
    records: Sequence[Mapping[str, Any]],
    *,
    cotracker_availability: Mapping[str, Any],
) -> dict[str, Any]:
    modes = {}
    for mode in MODE_NAMES:
        available_rows = [
            record["modes"][mode]
            for record in records
            if bool(record["modes"][mode].get("available"))
        ]
        if not available_rows:
            reasons = sorted(
                {
                    str(record["modes"][mode].get("reason", "unavailable"))
                    for record in records
                }
            )
            modes[mode] = {"available": False, "reasons": reasons}
            continue
        modes[mode] = _aggregate_available_rows(
            available_rows, expected_video_count=len(records)
        )
    wristband_rows = [
        record["supporting_modes"]["pose_lk_wristband"] for record in records
    ]
    supporting_modes = {
        "pose_lk_wristband": _aggregate_available_rows(
            wristband_rows, expected_video_count=len(records)
        )
    }
    default = "pose_lk"
    reason = (
        "Pose+LK 保持实验默认：它可用、因果执行且不需要额外模型"
    )
    if modes["pose_cotracker"].get("available"):
        candidate = modes["pose_cotracker"]
        baseline = modes["pose_lk"]
        if (
            candidate["complete_video_coverage"]
            and candidate["anchor_count"] > 0
            and candidate["anchor_count"] == baseline["anchor_count"]
            and candidate["identity_switch_proxy_count"]
            <= baseline["identity_switch_proxy_count"]
            and candidate["anchor_identity_correct_rate"]
            >= baseline["anchor_identity_correct_rate"]
            and candidate["mean_track_coverage"] >= baseline["mean_track_coverage"]
            and candidate["mean_trajectory_jitter_body"] is not None
            and baseline["mean_trajectory_jitter_body"] is not None
            and candidate["mean_trajectory_jitter_body"]
            <= baseline["mean_trajectory_jitter_body"]
        ):
            default = "pose_cotracker"
            reason = "CoTracker 通过了完整视频上的身份、锚点、覆盖率和抖动门槛"
        else:
            reason = (
                "Pose+LK 保持实验默认：CoTracker 未通过完整视频上的全部身份、"
                "锚点、覆盖率和抖动门槛"
            )
    else:
        reason += "；本机缺少 CoTracker 包/权重或有效推理结果"
    return {
        "schema_version": 1,
        "artifact_type": "swimming_round6_wristband_cotracker_ablation_v1",
        "round_completed": 6,
        "record_count": len(records),
        "anchor_count": sum(int(record["anchor_count"]) for record in records),
        "only_existing_videos_used": True,
        "new_training_performed": False,
        "cotracker": dict(cotracker_availability),
        "modes": modes,
        "supporting_modes": supporting_modes,
        "recommended_experimental_default": default,
        "formal_default_changed": False,
        "default_decision_reason": reason,
        "wristband_default_allowed": False,
        "wristband_default_reason": (
            "appearance evidence is evaluated only on the same two marked clips; "
            "there is no independent unmarked identity holdout"
        ),
        "hyrox_rules_changed": False,
        "reliable_side_selector_changed": False,
        "limitations": [
            "reviewed wristband centers evaluate marked clips but are not dense frame-level identity truth",
            "identity switch count remains a self-consistency proxy outside reviewed anchor frames",
            "unavailable modes are never imputed or ranked",
        ],
        "records": list(records),
    }


def _reanchor_flow(flow, frame, candidates, shape) -> None:
    for side in SIDES:
        snapshot = frame.tracks[side]
        semantic = snapshot.observed_semantic_side
        if snapshot.source == "pose" and semantic in candidates:
            flow.reanchor(side, candidates[semantic].position, frame_shape=shape)


def _mapping_confidence(frame: object) -> float:
    direct = frame.direct_assignment_cost
    swapped = frame.swapped_assignment_cost
    if direct is None or swapped is None:
        return 0.0
    return min(1.0, abs(float(direct) - float(swapped)) / 0.35)


def _opposite(side: str) -> str:
    return "right" if side == "left" else "left"


def _source_pixels(
    rotated_normalized: tuple[float, float] | None,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    if rotated_normalized is None:
        return None
    return rotated_normalized[1] * width, (1.0 - rotated_normalized[0]) * height


def _pixel_error(
    point: tuple[float, float] | None, anchor: Mapping[str, Any]
) -> float | None:
    if point is None:
        return None
    return math.hypot(point[0] - float(anchor["x"]), point[1] - float(anchor["y"]))


def _weighted_mean(rows: Sequence[Mapping[str, Any]], *, value: str, weight: str):
    pairs = [
        (float(row[value]), int(row[weight]))
        for row in rows
        if row.get(value) is not None and int(row[weight]) > 0
    ]
    total = sum(item[1] for item in pairs)
    return sum(item[0] * item[1] for item in pairs) / total if total else None


def _aggregate_available_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_video_count: int
) -> dict[str, Any]:
    anchor_count = sum(int(row["anchor_count"]) for row in rows)
    correct = sum(int(row["anchor_identity_correct_count"]) for row in rows)
    available_anchors = sum(int(row["anchor_target_available_count"]) for row in rows)
    return {
        "available": True,
        "video_count": len(rows),
        "complete_video_coverage": len(rows) == expected_video_count,
        "identity_switch_proxy_count": sum(
            int(row["identity_switch_proxy_count"]) for row in rows
        ),
        "anchor_count": anchor_count,
        "anchor_target_available_count": available_anchors,
        "anchor_identity_correct_count": correct,
        "anchor_identity_correct_rate": correct / max(1, anchor_count),
        "mean_track_coverage": fmean(
            float(row["mean_track_coverage"]) for row in rows
        ),
        "mean_trajectory_jitter_body": _optional_mean(
            row.get("mean_trajectory_jitter_body") for row in rows
        ),
        "anchor_target_error_mae_px": _weighted_mean(
            rows,
            value="anchor_target_error_mae_px",
            weight="anchor_target_available_count",
        ),
    }


def _optional_mean(values: Sequence[object] | Any) -> float | None:
    resolved = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            resolved.append(number)
    return fmean(resolved) if resolved else None


def write_artifacts(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    anchor_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "swim_round6_ablation.json"
    rows_path = target / "swim_round6_anchor_rows.csv"
    report_path = target / "SWIM_ROUND6_REPORT.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = list(anchor_rows[0]) if anchor_rows else []
    with rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(anchor_rows)
    report_path.write_text(_markdown(summary), encoding="utf-8")
    return summary_path, rows_path, report_path


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# 第 6 轮：游泳腕带 + CoTracker 实验",
        "",
        f"- 现有标记视频：{summary['record_count']}",
        f"- 人工腕带中心锚点：{summary['anchor_count']}",
        "- 新训练：无",
        f"- CoTracker available：{str(summary['cotracker']['available']).lower()}",
        "",
        "| 模式 | 可用 | Identity switch proxy | Anchor correct | Coverage | Jitter | Anchor MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODE_NAMES:
        values = summary["modes"][mode]
        if not values.get("available"):
            lines.append(f"| {mode} | false | — | — | — | — | — |")
            continue
        lines.append(
            f"| {mode} | true | {values['identity_switch_proxy_count']} | "
            f"{values['anchor_identity_correct_rate']:.4f} | "
            f"{values['mean_track_coverage']:.4f} | "
            f"{_number(values['mean_trajectory_jitter_body'], digits=6)} | "
            f"{_number(values['anchor_target_error_mae_px'])} |"
        )
    lines.extend(
        [
            "",
            "## 腕带辅助消融（非五模式正式比较项）",
            "",
        ]
    )
    wristband = summary["supporting_modes"]["pose_lk_wristband"]
    lines.extend(
        [
            "| 模式 | Identity switch proxy | Anchor correct | Coverage | Jitter | Anchor MAE |",
            "|---|---:|---:|---:|---:|---:|",
            f"| pose_lk_wristband | {wristband['identity_switch_proxy_count']} | "
            f"{wristband['anchor_identity_correct_rate']:.4f} | "
            f"{wristband['mean_track_coverage']:.4f} | "
            f"{_number(wristband['mean_trajectory_jitter_body'], digits=6)} | "
            f"{_number(wristband['anchor_target_error_mae_px'])} |",
            "",
            "该分支仅用于确认 ROI appearance 与 EMA 在线更新确实工作；因与原型更新"
            "使用同一批视频，不作为默认方案选择依据。",
            "",
            "## 默认策略",
            "",
            f"- 推荐实验默认：{summary['recommended_experimental_default']}",
            "- 正式默认已修改：false",
            "- 腕带外观允许成为默认：false",
            f"- 原因：{summary['default_decision_reason']}",
            "",
            "缺少的 CoTracker 结果不会被估算或用 LK 冒充。腕带模型仅在身份与可见度"
            "均为高置信度时以 EMA 更新；当前两段标记视频不是独立 holdout。",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: object, *, digits: int = 3) -> str:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return "—"
    return f"{resolved:.{digits}f}" if math.isfinite(resolved) else "—"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be a positive integer")
    raw_records = args.record or [
        (str(video), str(anchors)) for video, anchors in DEFAULT_RECORDS
    ]
    records = [load_anchor_record(video, anchors) for video, anchors in raw_records]
    config = load_swim_wrist_tracker_config(args.config)
    cotracker = CoTrackerOfflineBackend(
        config,
        checkpoint=args.cotracker_checkpoint,
        allow_torch_hub_download=args.allow_cotracker_download,
    )
    record_summaries = []
    all_rows = []
    for record in records:
        summary, rows = evaluate_record(
            record,
            model_path=args.model,
            config_path=args.config,
            cotracker=cotracker,
            max_frames=args.max_frames,
        )
        record_summaries.append(summary)
        all_rows.extend(rows)
    aggregate = aggregate_experiment(
        record_summaries,
        cotracker_availability=cotracker.availability.as_dict(),
    )
    paths = write_artifacts(args.output_dir, aggregate, all_rows)
    print(
        json.dumps(
            {
                "summary": str(paths[0]),
                "anchor_rows": str(paths[1]),
                "report": str(paths[2]),
                "recommended_experimental_default": aggregate[
                    "recommended_experimental_default"
                ],
                "formal_default_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODE_NAMES",
    "aggregate_experiment",
    "build_parser",
    "evaluate_record",
    "load_anchor_record",
    "main",
    "write_artifacts",
]
