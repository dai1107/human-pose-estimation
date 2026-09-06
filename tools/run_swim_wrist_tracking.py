"""Run swimming wrist identity + LK trajectory rounds 4-5 on an existing video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
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
from src.swimming.wrist_tracking import (  # noqa: E402
    SIDES,
    ArmChainObservation,
    LKOpticalFlowWristTracker,
    SwimWristIdentityTracker,
    WristCandidate,
    load_swim_wrist_tracker_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a swimming video through persistent anatomical wrist "
            "identity and short-gap LK optical-flow tracking."
        )
    )
    parser.add_argument("input_video", type=Path)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/swim_wrist_tracking_round4_5"),
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Do not rotate the prone-swimmer source clockwise for pose inference.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional development limit; omitted means the complete video.",
    )
    return parser


def run_tracking(
    input_video: str | Path,
    model_path: str | Path,
    *,
    config_path: str | Path,
    rotate_clockwise: bool = True,
    max_frames: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source = Path(input_video)
    model = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not model.is_file():
        raise FileNotFoundError(model)
    config = load_swim_wrist_tracker_config(config_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open swimming video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if fps <= 0.0:
        capture.release()
        raise RuntimeError(f"invalid video FPS: {source}")
    tracker = SwimWristIdentityTracker(config)
    optical_flow = LKOpticalFlowWristTracker(config)
    backend = MediaPipeBackend(
        model,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    source_counts: dict[str, Counter[str]] = {
        side: Counter() for side in SIDES
    }
    state_counts: dict[str, Counter[str]] = {
        side: Counter() for side in SIDES
    }
    flow_reasons: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    pose_detected_count = 0
    baseline_positions: dict[str, list[tuple[float, float] | None]] = {
        side: [] for side in SIDES
    }
    tracked_positions: dict[str, list[tuple[float, float] | None]] = {
        side: [] for side in SIDES
    }
    body_scales: list[float] = []
    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, source_frame = capture.read()
            if not ok:
                break
            frame = (
                cv2.rotate(source_frame, cv2.ROTATE_90_CLOCKWISE)
                if rotate_clockwise
                else source_frame
            )
            timestamp_ms = frame_index * 1000.0 / fps
            result = backend.detect(frame, int(round(timestamp_ms)))
            pose_detected_count += int(bool(result.success))
            candidates = _wrist_candidates(result.keypoints)
            chains = _arm_chains(result.keypoints)
            body_scale = _body_scale(result.keypoints)
            body_scale_px = body_scale * max(frame.shape[:2])
            flow = optical_flow.advance(frame, body_scale_px=body_scale_px)
            for item in flow.values():
                if item.reason not in {"accepted", "warming_up", "not_initialized"}:
                    flow_reasons[item.reason] += 1
            tracked = tracker.update(
                candidates,
                chains,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                body_scale=body_scale,
                optical_flow=flow,
            )
            candidate_by_side = {
                candidate.semantic_side: candidate for candidate in candidates
            }
            for side in SIDES:
                snapshot = tracked.tracks[side]
                source_counts[side][snapshot.source] += 1
                state_counts[side][snapshot.state] += 1
                tracked_positions[side].append(snapshot.position)
                baseline = candidate_by_side.get(side)
                baseline_positions[side].append(
                    baseline.position if baseline is not None else None
                )
                if (
                    snapshot.source == "pose"
                    and snapshot.observed_semantic_side in candidate_by_side
                ):
                    anchor = candidate_by_side[snapshot.observed_semantic_side]
                    optical_flow.reanchor(
                        side,
                        anchor.position,
                        frame_shape=frame.shape,
                    )
            body_scales.append(body_scale)
            reason_counts.update(tracked.reason_codes)
            row = _frame_row(tracked, flow)
            rows.append(row)
            event_reasons = set(tracked.reason_codes)
            event_reasons.update(
                f"{side}_{item.reason}"
                for side, item in flow.items()
                if item.reason
                not in {"accepted", "warming_up", "not_initialized"}
            )
            if (
                tracked.mapping_changed
                or event_reasons
                or any(
                    snapshot.state in {"lost", "reacquiring", "reacquired"}
                    for snapshot in tracked.tracks.values()
                )
            ):
                events.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "event_reasons": ",".join(sorted(event_reasons)),
                        "left_state": tracked.tracks["left"].state,
                        "right_state": tracked.tracks["right"].state,
                        "mapping_changed": tracked.mapping_changed,
                    }
                )
            frame_index += 1
    finally:
        capture.release()
        backend.close()

    metrics = {
        side: {
            "track_coverage": _coverage(tracked_positions[side]),
            "raw_mediapipe_coverage": _coverage(baseline_positions[side]),
            "source_counts": dict(sorted(source_counts[side].items())),
            "state_counts": dict(sorted(state_counts[side].items())),
            "reacquisition_count": (
                rows[-1][f"{side}_reacquisition_count"] if rows else 0
            ),
            "trajectory_outlier_rejection_count": (
                rows[-1][f"{side}_outlier_rejection_count"] if rows else 0
            ),
            "raw_mediapipe_trajectory_jitter_body": _trajectory_jitter(
                baseline_positions[side], body_scales
            ),
            "persistent_trajectory_jitter_body": _trajectory_jitter(
                tracked_positions[side], body_scales
            ),
        }
        for side in SIDES
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "swimming_wrist_tracking_round4_5_v1",
        "input_video": str(source.resolve()),
        "model": str(model.resolve()),
        "config": str(Path(config_path).resolve()),
        "rounds_completed": [4, 5],
        "experimental_only": True,
        "new_model_added": False,
        "wristband_appearance_used": False,
        "cotracker_used": False,
        "hyrox_rules_changed": False,
        "video": {
            "source_frame_count": source_frame_count,
            "processed_frame_count": frame_index,
            "fps": fps,
            "rotation_clockwise_for_inference": rotate_clockwise,
        },
        "pose_detected_count": pose_detected_count,
        "pose_detection_rate": pose_detected_count / max(1, frame_index),
        "round4_identity": {
            "persistent_left_track_id": "swim_wrist_left",
            "persistent_right_track_id": "swim_wrist_right",
            "persistent_track_identity_switch_count": 0,
            "instantaneous_assignment_flip_count": (
                tracker.instantaneous_mapping_flip_count
            ),
            "confirmed_semantic_mapping_change_count": (
                tracker.confirmed_mapping_change_count
            ),
            "identity_hysteresis_hold_frame_count": reason_counts[
                "IDENTITY_HYSTERESIS_HOLD"
            ],
            "assignment": "fixed 2x2 Hungarian-equivalent global minimum",
            "association_weights": {
                "motion": config.motion_weight,
                "shoulder_elbow_wrist_chain": config.chain_weight,
                "mediapipe_semantic_label": config.semantic_weight,
                "trajectory_smoothness": config.smoothness_weight,
            },
        },
        "round5_trajectory": {
            "by_side": metrics,
            "optical_flow_reason_counts": dict(sorted(flow_reasons.items())),
            "forward_backward_failure_count": flow_reasons[
                "forward_backward_failure"
            ],
            "trajectory_outlier_rejection_count": sum(
                int(metrics[side]["trajectory_outlier_rejection_count"])
                for side in SIDES
            ),
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "limitations": [
            "persistent track IDs are invariant by construction; zero is not "
            "human-labelled identity-switch truth",
            "instantaneous assignment flips are a self-consistency proxy until "
            "frame-level identity labels exist",
            "LK is limited to short gaps and is rejected by forward/backward or "
            "robust displacement gates",
            "no wristband appearance or CoTracker evidence is used in rounds 4-5",
        ],
    }
    return summary, rows, events


def _wrist_candidates(points: Sequence[object]) -> list[WristCandidate]:
    output = []
    for side in SIDES:
        point = _point(points, f"{side}_wrist")
        position = _position(point)
        if position is None:
            continue
        output.append(
            WristCandidate(
                semantic_side=side,
                position=position,
                confidence=_confidence(point),
            )
        )
    return output


def _arm_chains(points: Sequence[object]) -> dict[str, ArmChainObservation]:
    output = {}
    for side in SIDES:
        shoulder = _point(points, f"{side}_shoulder")
        elbow = _point(points, f"{side}_elbow")
        output[side] = ArmChainObservation(
            shoulder=_position(shoulder),
            elbow=_position(elbow),
            confidence=min(_confidence(shoulder), _confidence(elbow)),
        )
    return output


def _body_scale(points: Sequence[object]) -> float:
    left_shoulder = _position(_point(points, "left_shoulder"))
    right_shoulder = _position(_point(points, "right_shoulder"))
    left_hip = _position(_point(points, "left_hip"))
    right_hip = _position(_point(points, "right_hip"))
    scales = []
    if left_shoulder is not None and right_shoulder is not None:
        scales.append(_distance(left_shoulder, right_shoulder))
    if all(
        point is not None
        for point in (left_shoulder, right_shoulder, left_hip, right_hip)
    ):
        shoulder_center = _midpoint(left_shoulder, right_shoulder)
        hip_center = _midpoint(left_hip, right_hip)
        scales.append(_distance(shoulder_center, hip_center))
    finite = [value for value in scales if math.isfinite(value) and value >= 0.03]
    return max(0.08, fmean(finite)) if finite else 0.18


def _frame_row(frame: object, flow: Mapping[str, object]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame_index": frame.frame_index,
        "timestamp_ms": frame.timestamp_ms,
        "proposed_semantic_mapping_swapped": frame.proposed_semantic_mapping_swapped,
        "committed_semantic_mapping_swapped": frame.committed_semantic_mapping_swapped,
        "mapping_change_pending_frames": frame.mapping_change_pending_frames,
        "mapping_changed": frame.mapping_changed,
        "direct_assignment_cost": frame.direct_assignment_cost,
        "swapped_assignment_cost": frame.swapped_assignment_cost,
        "reason_codes": ",".join(frame.reason_codes),
    }
    for side in SIDES:
        snapshot = frame.tracks[side]
        row.update(
            {
                f"{side}_track_id": snapshot.track_id,
                f"{side}_state": snapshot.state,
                f"{side}_x": snapshot.position[0] if snapshot.position else None,
                f"{side}_y": snapshot.position[1] if snapshot.position else None,
                f"{side}_vx": snapshot.velocity[0],
                f"{side}_vy": snapshot.velocity[1],
                f"{side}_confidence": snapshot.confidence,
                f"{side}_source": snapshot.source,
                f"{side}_observed_semantic_side": snapshot.observed_semantic_side,
                f"{side}_missing_pose_frames": snapshot.missing_pose_frames,
                f"{side}_reacquisition_count": snapshot.reacquisition_count,
                f"{side}_outlier_rejection_count": snapshot.outlier_rejection_count,
                f"{side}_lk_reliable": flow[side].reliable,
                f"{side}_lk_forward_backward_error_px": (
                    flow[side].forward_backward_error_px
                ),
                f"{side}_lk_reason": flow[side].reason,
            }
        )
    return row


def _trajectory_jitter(
    positions: Sequence[tuple[float, float] | None],
    body_scales: Sequence[float],
) -> float | None:
    values = []
    for index in range(2, len(positions)):
        first, middle, last = positions[index - 2 : index + 1]
        if first is None or middle is None or last is None:
            continue
        second_difference = np.asarray(last) - 2.0 * np.asarray(middle) + np.asarray(first)
        values.append(
            float(np.linalg.norm(second_difference))
            / max(float(body_scales[index]), 1e-4)
        )
    return float(np.median(values)) if values else None


def _coverage(values: Sequence[tuple[float, float] | None]) -> float:
    return sum(value is not None for value in values) / max(1, len(values))


def _point(points: Sequence[object], name: str) -> object | None:
    return next(
        (point for point in points if str(getattr(point, "name", "")) == name),
        None,
    )


def _position(point: object | None) -> tuple[float, float] | None:
    if point is None:
        return None
    try:
        x = float(getattr(point, "x"))
        y = float(getattr(point, "y"))
    except (TypeError, ValueError, AttributeError, OverflowError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _confidence(point: object | None) -> float:
    if point is None:
        return 0.0
    raw = getattr(point, "confidence", getattr(point, "visibility", 0.0))
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


def _midpoint(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float]:
    return (0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1]))


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def write_artifacts(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "swim_wrist_tracking_round4_5.json"
    rows_path = target / "swim_wrist_tracks.csv"
    events_path = target / "swim_wrist_hard_frames.csv"
    report_path = target / "SWIM_WRIST_ROUND4_5_REPORT.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(rows_path, rows)
    _write_csv(events_path, events)
    report_path.write_text(_markdown(summary), encoding="utf-8")
    return summary_path, rows_path, events_path, report_path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _markdown(summary: Mapping[str, Any]) -> str:
    identity = summary["round4_identity"]
    trajectory = summary["round5_trajectory"]
    lines = [
        "# 第 4、5 轮：游泳腕部身份与轨迹稳定",
        "",
        f"- 完整处理帧：{summary['video']['processed_frame_count']}",
        f"- Pose 检出率：{summary['pose_detection_rate']:.4f}",
        "- 新模型：无",
        "- 腕带外观 / CoTracker：未使用",
        "- HYROX 正式规则变化：无",
        "",
        "## 第 4 轮：身份稳定",
        "",
        f"- 瞬时分配翻转：{identity['instantaneous_assignment_flip_count']}",
        f"- 迟滞确认后的语义映射变化：{identity['confirmed_semantic_mapping_change_count']}",
        f"- 迟滞 hold 帧：{identity['identity_hysteresis_hold_frame_count']}",
        "- persistent track ID：固定 left/right，不随 MediaPipe 标签直接改变",
        "",
        "## 第 5 轮：轨迹稳定",
        "",
        "| 侧别 | Raw coverage | Track coverage | Raw jitter | Track jitter | "
        "LK bridge | Lost | Reacquired |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for side in SIDES:
        values = trajectory["by_side"][side]
        lines.append(
            f"| {side} | {values['raw_mediapipe_coverage']:.4f} | "
            f"{values['track_coverage']:.4f} | "
            f"{_number(values['raw_mediapipe_trajectory_jitter_body'])} | "
            f"{_number(values['persistent_trajectory_jitter_body'])} | "
            f"{values['source_counts'].get('optical_flow', 0)} | "
            f"{values['state_counts'].get('lost', 0)} | "
            f"{values['reacquisition_count']} |"
        )
    lines.extend(
        [
            "",
            f"- Forward/backward failure：{trajectory['forward_backward_failure_count']}",
            f"- 轨迹异常拒绝：{trajectory['trajectory_outlier_rejection_count']}",
            "",
            "## 解释边界",
            "",
            "persistent track ID 为代码不变量，因此其 identity switch count 为 0 不能替代人工真值。"
            "当前瞬时分配翻转、覆盖率、重捕获和 jitter 属于现有视频 self-consistency 指标；"
            "第 6 轮加入腕带/CoTracker 后才能做更强的 A/B 身份验证。",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: object) -> str:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return "—"
    return f"{resolved:.6f}" if math.isfinite(resolved) else "—"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, rows, events = run_tracking(
        args.input_video,
        args.model,
        config_path=args.config,
        rotate_clockwise=not args.no_rotate,
        max_frames=args.max_frames,
    )
    paths = write_artifacts(args.output_dir, summary, rows, events)
    print(
        json.dumps(
            {
                "summary": str(paths[0]),
                "tracks": str(paths[1]),
                "hard_frames": str(paths[2]),
                "report": str(paths[3]),
                "processed_frame_count": summary["video"]["processed_frame_count"],
                "hyrox_rules_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "main",
    "run_tracking",
    "write_artifacts",
]
