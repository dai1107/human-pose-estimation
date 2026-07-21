"""Golden-video regression validation for the eight bundled HYROX samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from hyrox.features import extract_basic_pose_features
from hyrox.registry import create_action_analyzer
from src.backends.mediapipe_backend import MediaPipeBackend
from src.output_schema import versioned_payload
from src.paths import resolve_asset
from src.utils.smoothing import KeypointSmoother

SUPPORTED_METRICS = frozenset(
    {
        "total_frames",
        "pose_detected_rate",
        "candidate_count",
        "pose_valid_rep_count",
        "no_rep_count",
        "unsure_count",
        "cycle_count",
        "rep_count",
    }
)


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    video: str
    action: str
    camera_view: str
    expectations: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class GoldenObservation:
    case_id: str
    video: str
    action: str
    total_frames: int
    pose_detected_frames: int
    pose_detected_rate: float
    candidate_count: int
    pose_valid_rep_count: int
    no_rep_count: int
    unsure_count: int
    cycle_count: int
    rep_count: int
    final_phase: str


def load_manifest(path: str | Path) -> tuple[str, list[GoldenCase]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("golden manifest schema_version must be 1")
    model = str(payload.get("model", "models/pose_landmarker_full.task"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("golden manifest cases must be a non-empty list")
    cases: list[GoldenCase] = []
    seen_ids: set[str] = set()
    seen_videos: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise ValueError(f"golden case {index} must be an object")
        case_id = str(raw.get("id", "")).strip()
        video = str(raw.get("video", "")).strip()
        action = str(raw.get("action", "")).strip()
        camera_view = str(raw.get("camera_view", "unknown")).strip()
        if not case_id or not video or not action:
            raise ValueError(f"golden case {index} requires id, video, and action")
        if case_id in seen_ids or video in seen_videos:
            raise ValueError(f"duplicate golden id or video: {case_id} / {video}")
        expectations = _parse_expectations(raw.get("expectations"), case_id)
        cases.append(GoldenCase(case_id, video, action, camera_view, expectations))
        seen_ids.add(case_id)
        seen_videos.add(video)
    return model, cases


def _parse_expectations(raw: object, case_id: str) -> dict[str, tuple[float, float]]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"golden case {case_id} requires expectations")
    parsed: dict[str, tuple[float, float]] = {}
    for metric, interval in raw.items():
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"golden case {case_id} has unknown metric: {metric}")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"golden case {case_id} metric {metric} must be [min, max]")
        lower, upper = float(interval[0]), float(interval[1])
        if lower > upper:
            raise ValueError(f"golden case {case_id} metric {metric} has min > max")
        parsed[str(metric)] = (lower, upper)
    return parsed


def evaluate_case(
    case: GoldenCase,
    model_path: str | Path,
    *,
    backend: MediaPipeBackend | None = None,
) -> GoldenObservation:
    video_path = resolve_asset(case.video)
    if not video_path.exists():
        raise FileNotFoundError(f"golden video not found: {video_path}")
    owns_backend = backend is None
    backend = backend or MediaPipeBackend(
        resolve_asset(model_path),
        output_segmentation_masks=False,
    )
    analyzer = create_action_analyzer(case.action, camera_view=case.camera_view, live_mode=False)
    smoother = KeypointSmoother(mode="one-euro", max_missing_frames=5)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        backend.close()
        raise RuntimeError(f"could not open golden video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps > 0 else 30.0
    frame_index = 0
    detected_frames = 0
    final_state: Mapping[str, object] = {}
    backend_timestamp_offset = max(0, int(getattr(backend, "_last_timestamp_ms", -1)) + 1)
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            timestamp_ms = max(0, int(round(frame_index * 1000.0 / source_fps)))
            result = smoother.smooth_result(
                backend.detect(frame, timestamp_ms=backend_timestamp_offset + timestamp_ms)
            )
            has_pose = bool(result.success and result.keypoints)
            detected_frames += int(has_pose)
            features = None
            if has_pose:
                height, width = frame.shape[:2]
                features = extract_basic_pose_features(
                    result.keypoints,
                    image_width=width,
                    image_height=height,
                    segmentation_mask=result.extra.get("segmentation_mask"),
                )
            final_state = analyzer.attach_view_context(
                analyzer.update(features if has_pose else None, timestamp_ms=timestamp_ms)
            )
    finally:
        capture.release()
        if owns_backend:
            backend.close()
    if frame_index == 0:
        raise RuntimeError(f"golden video has no decodable frames: {video_path}")
    integer = lambda name: int(final_state.get(name, 0) or 0)
    return GoldenObservation(
        case_id=case.case_id,
        video=case.video,
        action=case.action,
        total_frames=frame_index,
        pose_detected_frames=detected_frames,
        pose_detected_rate=detected_frames / frame_index,
        candidate_count=integer("candidate_count"),
        pose_valid_rep_count=integer("pose_valid_rep_count"),
        no_rep_count=integer("no_rep_count"),
        unsure_count=integer("unsure_count"),
        cycle_count=integer("cycle_count"),
        rep_count=integer("rep_count"),
        final_phase=str(final_state.get("phase", "unknown")),
    )


def compare_observation(case: GoldenCase, observation: GoldenObservation) -> list[str]:
    failures: list[str] = []
    values = asdict(observation)
    for metric, (lower, upper) in case.expectations.items():
        value = float(values[metric])
        if not lower <= value <= upper:
            failures.append(f"{metric}={value:g} outside [{lower:g}, {upper:g}]")
    return failures


def build_report(
    cases: Sequence[GoldenCase],
    observations: Sequence[GoldenObservation],
) -> dict[str, Any]:
    by_id = {item.case_id: item for item in observations}
    records: list[dict[str, Any]] = []
    for case in cases:
        observation = by_id[case.case_id]
        failures = compare_observation(case, observation)
        records.append(
            {
                "id": case.case_id,
                "status": "passed" if not failures else "failed",
                "failures": failures,
                "expectations": {name: list(bounds) for name, bounds in case.expectations.items()},
                "observation": asdict(observation),
            }
        )
    return versioned_payload(
        "hyrox_golden_video_report",
        {
            "status": "passed" if all(record["status"] == "passed" for record in records) else "failed",
            "case_count": len(records),
            "passed_count": sum(record["status"] == "passed" for record in records),
            "cases": records,
        },
    )
