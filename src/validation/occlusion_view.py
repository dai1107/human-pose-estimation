"""Diagnostics for occlusion stability and cross-view phase failures.

This module observes the existing formal MediaPipe/HYROX path.  It exports
raw and formally filtered landmarks side by side, but never feeds diagnostic
or display-only values back into the analyzer.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from math import hypot, isfinite
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import cv2

from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from src.backends.base import Keypoint, PoseResult
from src.backends.mediapipe_backend import MediaPipeBackend
from src.paths import resolve_asset
from src.utils.smoothing import KeypointSmoother
from src.validation.baseline import utc_now, write_json
from src.validation.golden_videos import (
    GoldenCase,
    GoldenObservation,
    build_report,
    load_manifest,
)


SCHEMA_VERSION = 1
PARENT_BY_NAME: Mapping[str, str] = {
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    "left_pinky": "left_wrist",
    "right_pinky": "right_wrist",
    "left_index": "left_wrist",
    "right_index": "right_wrist",
    "left_thumb": "left_wrist",
    "right_thumb": "right_wrist",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "left_heel": "left_ankle",
    "right_heel": "right_ankle",
    "left_foot_index": "left_ankle",
    "right_foot_index": "right_ankle",
}
FORMAL_COUNT_FIELDS = (
    "candidate_count",
    "pose_valid_rep_count",
    "no_rep_count",
    "unsure_count",
    "cycle_count",
    "rep_count",
)


def _confidence(point: Keypoint | None) -> float:
    if point is None:
        return 0.0
    return min(
        float(point.confidence),
        float(point.visibility if point.visibility is not None else point.confidence),
        float(point.presence if point.presence is not None else point.confidence),
    )


def _point_map(points: Sequence[Keypoint]) -> dict[str, Keypoint]:
    return {point.name: point for point in points}


def _distance(first: Keypoint | None, second: Keypoint | None) -> float | None:
    if first is None or second is None:
        return None
    values = (first.x, first.y, first.z, second.x, second.y, second.z)
    if not all(isfinite(value) for value in values):
        return None
    return hypot(first.x - second.x, first.y - second.y, first.z - second.z)


def _body_scale(points: Mapping[str, Keypoint], fallback: float) -> float:
    pairs = (
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
    )
    values = [
        value
        for first, second in pairs
        if (value := _distance(points.get(first), points.get(second))) is not None
        and value > 0.01
    ]
    return max(0.06, min(1.5, median(values))) if values else fallback


def _percentile(values: Iterable[float | None], ratio: float) -> float | None:
    ordered = sorted(float(value) for value in values if value is not None and isfinite(value))
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(len(ordered) * ratio + 0.999999) - 1))
    return ordered[index]


def _debug_phases(state: Mapping[str, object]) -> tuple[str, str]:
    debug = state.get("debug")
    debug_map = debug if isinstance(debug, Mapping) else {}
    stable = str(
        debug_map.get(
            "stable_phase", state.get("stable_phase", state.get("phase", "unknown"))
        )
    )
    raw = str(debug_map.get("raw_phase", state.get("raw_phase", stable)))
    return raw, stable


def _keypoint_payload(point: Keypoint) -> dict[str, object]:
    return {
        "name": point.name,
        "x": point.x,
        "y": point.y,
        "z": point.z,
        "confidence": point.confidence,
        "visibility": point.visibility,
        "presence": point.presence,
    }


class DiagnosticTrace:
    """Collect per-frame evidence without influencing formal decisions."""

    def __init__(self, case: GoldenCase) -> None:
        self.case = case
        self.landmark_rows: list[dict[str, object]] = []
        self.phase_rows: list[dict[str, object]] = []
        self.timeline_frames: list[dict[str, object]] = []
        self._history: dict[tuple[str, str], tuple[Keypoint, float, tuple[float, float, float] | None]] = {}
        self._bone_history: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=31))
        self._last_body_scale = 0.25

    def observe(
        self,
        frame_index: int,
        timestamp_ms: float,
        raw_result: PoseResult,
        filtered_result: PoseResult,
        state: Mapping[str, object],
    ) -> None:
        raw_points = _point_map(raw_result.keypoints)
        filtered_points = _point_map(filtered_result.keypoints)
        self._last_body_scale = _body_scale(filtered_points or raw_points, self._last_body_scale)
        names = sorted(set(raw_points) | set(filtered_points))
        for name in names:
            raw = raw_points.get(name)
            filtered = filtered_points.get(name)
            row: dict[str, object] = {
                "case_id": self.case.case_id,
                "action": self.case.action,
                "camera_view": self.case.camera_view,
                "occlusion_level": "unlabelled",
                "frame_index": frame_index,
                "timestamp_ms": round(timestamp_ms, 3),
                "landmark": name,
                "body_scale": self._last_body_scale,
                "confidence": _confidence(raw),
                "visibility": raw.visibility if raw is not None else None,
                "presence": raw.presence if raw is not None else None,
            }
            for stream, point, points in (
                ("raw", raw, raw_points),
                ("filtered", filtered, filtered_points),
            ):
                row[f"{stream}_x"] = point.x if point is not None else None
                row[f"{stream}_y"] = point.y if point is not None else None
                row[f"{stream}_z"] = point.z if point is not None else None
                speed, acceleration, velocity = self._motion(
                    stream, name, point, timestamp_ms, self._last_body_scale
                )
                row[f"{stream}_speed_body_s"] = speed
                row[f"{stream}_acceleration_body_s2"] = acceleration
                parent = points.get(PARENT_BY_NAME.get(name, ""))
                bone_length = _distance(point, parent)
                history = self._bone_history[(stream, name)]
                reference = median(history) if len(history) >= 3 else None
                row[f"{stream}_bone_length"] = bone_length
                row[f"{stream}_bone_length_change_ratio"] = (
                    abs(bone_length - reference) / reference
                    if bone_length is not None and reference is not None and reference > 1e-9
                    else None
                )
                if bone_length is not None and _confidence(point) >= 0.5:
                    history.append(bone_length)
                if point is not None:
                    self._history[(stream, name)] = (point, timestamp_ms, velocity)
            self.landmark_rows.append(row)

        raw_phase, stable_phase = _debug_phases(state)
        phase_row: dict[str, object] = {
            "case_id": self.case.case_id,
            "action": self.case.action,
            "camera_view": self.case.camera_view,
            "occlusion_level": "unlabelled",
            "frame_index": frame_index,
            "timestamp_ms": round(timestamp_ms, 3),
            "pose_detected": bool(filtered_result.success and filtered_result.keypoints),
            "raw_phase": raw_phase,
            "stable_phase": stable_phase,
            "phase_disagreement": raw_phase != stable_phase,
        }
        phase_row.update(
            {field: int(state.get(field, 0) or 0) for field in FORMAL_COUNT_FIELDS}
        )
        self.phase_rows.append(phase_row)
        self.timeline_frames.append(
            {
                **phase_row,
                "phase": stable_phase,
                "keypoints": [
                    _keypoint_payload(point) for point in filtered_result.keypoints
                ],
            }
        )

    def _motion(
        self,
        stream: str,
        name: str,
        point: Keypoint | None,
        timestamp_ms: float,
        body_scale: float,
    ) -> tuple[float | None, float | None, tuple[float, float, float] | None]:
        previous = self._history.get((stream, name))
        if point is None or previous is None:
            return None, None, None
        previous_point, previous_ms, previous_velocity = previous
        elapsed_seconds = (timestamp_ms - previous_ms) / 1000.0
        if elapsed_seconds <= 0 or elapsed_seconds >= 0.5:
            return None, None, None
        velocity = (
            (point.x - previous_point.x) / elapsed_seconds / body_scale,
            (point.y - previous_point.y) / elapsed_seconds / body_scale,
            (point.z - previous_point.z) / elapsed_seconds / body_scale,
        )
        speed = hypot(*velocity)
        acceleration = (
            hypot(*(velocity[index] - previous_velocity[index] for index in range(3)))
            / elapsed_seconds
            if previous_velocity is not None
            else None
        )
        return speed, acceleration, velocity

    def summary(self) -> dict[str, object]:
        phase_disagreement_frames = sum(
            bool(row["phase_disagreement"]) for row in self.phase_rows
        )
        total_frames = len(self.phase_rows)
        final = self.phase_rows[-1] if self.phase_rows else {}
        return {
            "case_id": self.case.case_id,
            "action": self.case.action,
            "camera_view": self.case.camera_view,
            "occlusion_level": "unlabelled",
            "frame_count": total_frames,
            "pose_detection_rate": (
                sum(bool(row["pose_detected"]) for row in self.phase_rows) / total_frames
                if total_frames
                else 0.0
            ),
            "phase_disagreement_rate": (
                phase_disagreement_frames / total_frames if total_frames else 0.0
            ),
            "raw_acceleration_p95": _percentile(
                (row.get("raw_acceleration_body_s2") for row in self.landmark_rows), 0.95
            ),
            "filtered_acceleration_p95": _percentile(
                (row.get("filtered_acceleration_body_s2") for row in self.landmark_rows), 0.95
            ),
            "raw_bone_change_p95": _percentile(
                (row.get("raw_bone_length_change_ratio") for row in self.landmark_rows), 0.95
            ),
            "filtered_bone_change_p95": _percentile(
                (row.get("filtered_bone_length_change_ratio") for row in self.landmark_rows), 0.95
            ),
            "low_confidence_landmark_rate": (
                sum(float(row["confidence"]) < 0.5 for row in self.landmark_rows)
                / len(self.landmark_rows)
                if self.landmark_rows
                else 0.0
            ),
            "final_counts": {field: int(final.get(field, 0) or 0) for field in FORMAL_COUNT_FIELDS},
        }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_golden_diagnostics(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    model_override: str = "",
    selected_cases: set[str] | None = None,
) -> dict[str, object]:
    model, cases = load_manifest(manifest_path)
    if selected_cases:
        cases = [case for case in cases if case.case_id in selected_cases]
    selected_model = model_override or model
    output_root = Path(output_dir)
    all_landmarks: list[dict[str, object]] = []
    all_phases: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    observations: list[GoldenObservation] = []
    timeline_dir = output_root / "timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        video_path = resolve_asset(case.video)
        backend = MediaPipeBackend(
            resolve_asset(selected_model), output_segmentation_masks=False
        )
        smoother = KeypointSmoother(mode="one-euro", max_missing_frames=5)
        analyzer = create_action_analyzer(
            case.action, camera_view=case.camera_view, live_mode=False
        )
        trace = DiagnosticTrace(case)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            backend.close()
            raise RuntimeError(f"could not open diagnostic video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0
        frame_index = 0
        detected_frames = 0
        final_state: Mapping[str, object] = {}
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                timestamp_ms = frame_index * 1000.0 / fps
                raw_result = backend.detect(frame, timestamp_ms=int(round(timestamp_ms)))
                filtered_result = smoother.smooth_result(raw_result)
                has_pose = bool(filtered_result.success and filtered_result.keypoints)
                detected_frames += int(has_pose)
                features = None
                if has_pose:
                    height, width = frame.shape[:2]
                    features = extract_basic_pose_features(
                        filtered_result.keypoints,
                        image_width=width,
                        image_height=height,
                        segmentation_mask=filtered_result.extra.get("segmentation_mask"),
                    )
                final_state = analyzer.attach_view_context(
                    analyzer.update(features if has_pose else None, timestamp_ms=int(round(timestamp_ms)))
                )
                trace.observe(
                    frame_index, timestamp_ms, raw_result, filtered_result, final_state
                )
        finally:
            capture.release()
            backend.close()
        integer = lambda name: int(final_state.get(name, 0) or 0)
        observations.append(
            GoldenObservation(
                case_id=case.case_id,
                video=case.video,
                action=case.action,
                total_frames=frame_index,
                pose_detected_frames=detected_frames,
                pose_detected_rate=detected_frames / max(1, frame_index),
                candidate_count=integer("candidate_count"),
                pose_valid_rep_count=integer("pose_valid_rep_count"),
                no_rep_count=integer("no_rep_count"),
                unsure_count=integer("unsure_count"),
                cycle_count=integer("cycle_count"),
                rep_count=integer("rep_count"),
                final_phase=str(final_state.get("phase", "unknown")),
            )
        )
        summary = trace.summary()
        summaries.append(summary)
        all_landmarks.extend(trace.landmark_rows)
        all_phases.extend(trace.phase_rows)
        write_json(
            timeline_dir / f"{case.case_id}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "occlusion_view_display_timeline",
                "case": summary,
                "frames": trace.timeline_frames,
            },
        )

    _write_csv(output_root / "landmark_trace.csv", all_landmarks)
    _write_csv(output_root / "phase_trace.csv", all_phases)
    golden = build_report(cases, observations)
    write_json(output_root / "golden_regression.json", golden)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "occlusion_cross_view_diagnostic_baseline",
        "generated_at": utc_now(),
        "model": selected_model,
        "case_count": len(cases),
        "grouping_dimensions": ["action", "camera_view", "occlusion_level"],
        "occlusion_label_status": "unlabelled_is_not_equivalent_to_clear",
        "cases": summaries,
        "golden_regression_status": golden["status"],
        "formal_display_isolation": True,
        "files": {
            "landmark_csv": "landmark_trace.csv",
            "phase_csv": "phase_trace.csv",
            "golden_regression": "golden_regression.json",
            "timelines": "timelines/*.json",
        },
    }
    write_json(output_root / "baseline.json", report)
    return report


def build_hard_case_inventory(
    error_library_path: str | Path,
    phone_manifest_path: str | Path,
    review_root: str | Path,
) -> list[dict[str, object]]:
    error_payload = json.loads(Path(error_library_path).read_text(encoding="utf-8"))
    manifest_payload = json.loads(Path(phone_manifest_path).read_text(encoding="utf-8"))
    records = {
        str(item["record_id"]): item
        for item in manifest_payload.get("records", [])
        if isinstance(item, Mapping) and item.get("record_id")
    }
    review_meta: dict[str, tuple[str, Counter[str]]] = {}
    for path in sorted(Path(review_root).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        record_id = str(payload.get("record_id", path.stem))
        quick = payload.get("review", {}).get("quick_review", {})
        if not isinstance(quick, Mapping):
            quick = {}
        observability = str(quick.get("observability", "UNKNOWN")).upper()
        event_counts: Counter[str] = Counter()
        for event in quick.get("events", []) or []:
            if isinstance(event, Mapping):
                event_counts[str(event.get("observability", "UNKNOWN")).upper()] += 1
        review_meta[record_id] = (observability, event_counts)

    category_to_failure = {
        "FN": "missed_count",
        "FP": "over_count",
        "STATUS_MISMATCH": "status_mismatch",
        "UNSURE": "unsure",
        "TP": "matched_control",
    }
    rows: list[dict[str, object]] = []
    for case in error_payload.get("cases", []) or []:
        if not isinstance(case, Mapping):
            continue
        record_id = str(case.get("record_id", ""))
        source = records.get(record_id, {})
        overall_observability, event_counts = review_meta.get(
            record_id, ("UNKNOWN", Counter())
        )
        if overall_observability in {"PARTIAL", "OCCLUDED", "OUT_OF_FRAME"}:
            occlusion_level = overall_observability.lower()
        elif any(event_counts[name] for name in ("PARTIAL", "OCCLUDED", "OUT_OF_FRAME")):
            occlusion_level = "partial_event"
        else:
            occlusion_level = "unlabelled"
        category = str(case.get("category", "UNKNOWN")).upper()
        media = case.get("media") if isinstance(case.get("media"), Mapping) else {}
        rows.append(
            {
                "case_id": str(case.get("case_id", "")),
                "record_id": record_id,
                "action": str(case.get("action", source.get("action", "unknown"))),
                "camera_view": str(case.get("camera_view", source.get("camera_view", "unknown"))),
                "occlusion_level": occlusion_level,
                "failure_class": category_to_failure.get(category, category.lower()),
                "category": category,
                "anchor_frame": case.get("anchor_frame"),
                "clip_start_frame": case.get("clip_start_frame"),
                "clip_end_frame": case.get("clip_end_frame"),
                "source_video": str(case.get("source_video", source.get("source_file", ""))),
                "clip_path": str(media.get("path", "")),
                "dataset_role": str(case.get("dataset_role", "unknown")),
                "human_validity": (
                    str(case.get("human_rep", {}).get("validity", ""))
                    if isinstance(case.get("human_rep"), Mapping)
                    else ""
                ),
                "runtime_status": (
                    str(case.get("runtime_candidate", {}).get("status", ""))
                    if isinstance(case.get("runtime_candidate"), Mapping)
                    else ""
                ),
                "review_observability": overall_observability,
                "partial_event_count": sum(
                    event_counts[name]
                    for name in ("PARTIAL", "OCCLUDED", "OUT_OF_FRAME")
                ),
            }
        )
    return rows


def write_failure_inventory(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output_root = Path(output_dir)
    _write_csv(output_root / "hard_cases.csv", rows)
    counts = {
        "by_failure_class": dict(sorted(Counter(str(row["failure_class"]) for row in rows).items())),
        "by_action": dict(sorted(Counter(str(row["action"]) for row in rows).items())),
        "by_camera_view": dict(sorted(Counter(str(row["camera_view"]) for row in rows).items())),
        "by_occlusion_level": dict(sorted(Counter(str(row["occlusion_level"]) for row in rows).items())),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "occlusion_cross_view_hard_case_inventory",
        "generated_at": utc_now(),
        "case_count": len(rows),
        "counts": counts,
        "cases": list(rows),
        "label_caveat": (
            "unlabelled means occlusion severity was not explicitly reviewed; it must not be treated as clear"
        ),
    }
    write_json(output_root / "hard_cases.json", payload)
    lines = [
        "# 遮挡与跨视角失败分类基线",
        "",
        f"- 难例总数：`{len(rows)}`",
        "- `unlabelled` 表示尚无明确遮挡等级，不能解释为无遮挡。",
        "- `development` 难例用于定位问题，不作为独立测试集成绩。",
        "",
        "## 按失败类型",
        "",
        "| 类型 | 数量 |",
        "|---|---:|",
        *[f"| `{name}` | {count} |" for name, count in counts["by_failure_class"].items()],
        "",
        "## 按视角",
        "",
        "| 视角 | 数量 |",
        "|---|---:|",
        *[f"| `{name}` | {count} |" for name, count in counts["by_camera_view"].items()],
        "",
        "## 后续标注缺口",
        "",
        "现有人工记录的多数事件可观测性仍为 `UNKNOWN`。阶段 A 已保留该缺口，未把未知值静默当作无遮挡；后续应优先复核 FN、FP、UNSURE 片段的决定性关节遮挡等级。",
        "",
    ]
    (output_root / "failure_taxonomy.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


__all__ = [
    "DiagnosticTrace",
    "build_hard_case_inventory",
    "run_golden_diagnostics",
    "write_failure_inventory",
]
