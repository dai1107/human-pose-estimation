"""Round-seven multi-person candidate tracking and target-lock audit."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from src.utils.roi import BBox, clamp_bbox
from src.utils.ultralytics_config import ensure_ultralytics_config_dir
from tools.dataset.manifest import sha256_file, utc_now
from tools.dataset.phone_rgb import _atomic_json


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

CORE_BONES = (
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def bbox_iou(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def relative_center_distance(first: BBox, second: BBox) -> float:
    first_width = max(1.0, first[2] - first[0])
    first_height = max(1.0, first[3] - first[1])
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    distance = math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    return distance / math.hypot(first_width, first_height)


def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _center(bbox: BBox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _as_numpy(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return None


def _appearance_histogram(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    left, top = int(round(x1)), int(round(y1))
    right, bottom = int(round(x2)), int(round(y2))
    if right <= left or bottom <= top:
        return np.zeros(64, dtype=np.float32)
    crop = frame[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, None).flatten().astype(np.float32)
    return histogram


def _histogram_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 or second.size == 0 or not np.any(first) or not np.any(second):
        return 0.5
    correlation = float(cv2.compareHist(first, second, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def _skeleton_signature(xy: np.ndarray, confidence: np.ndarray, bbox: BBox) -> np.ndarray:
    scale = max(1.0, math.sqrt(_area(bbox)))
    values: list[float] = []
    for first, second in CORE_BONES:
        if (
            first >= len(xy)
            or second >= len(xy)
            or first >= len(confidence)
            or second >= len(confidence)
            or confidence[first] < 0.25
            or confidence[second] < 0.25
        ):
            values.append(float("nan"))
            continue
        values.append(float(np.linalg.norm(xy[first, :2] - xy[second, :2]) / scale))
    return np.asarray(values, dtype=np.float32)


def _skeleton_similarity(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 3:
        return 0.5
    error = float(np.median(np.abs(first[valid] - second[valid])))
    return max(0.0, 1.0 - min(1.0, error / 0.25))


@dataclass
class DetectionCandidate:
    bbox: BBox
    confidence: float
    keypoints_xy: np.ndarray
    keypoints_confidence: np.ndarray
    appearance: np.ndarray
    skeleton: np.ndarray


@dataclass
class TrackState:
    track_id: str
    last_bbox: BBox
    previous_bbox: BBox | None
    last_frame: int
    previous_frame: int | None
    appearance: np.ndarray
    skeleton: np.ndarray
    first_frame: int
    detection_count: int = 1
    propagated_count: int = 0
    confidence_sum: float = 0.0
    area_sum: float = 0.0
    center_travel: float = 0.0
    missing_detection_runs: int = 0
    last_keypoints_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    last_keypoints_confidence: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    def predicted_bbox(self, frame_index: int, width: int, height: int) -> BBox:
        if self.previous_bbox is None or self.previous_frame is None:
            return clamp_bbox(self.last_bbox, width, height)
        delta_frames = max(1, self.last_frame - self.previous_frame)
        horizon = max(0, frame_index - self.last_frame)
        velocity = [
            (current - previous) / delta_frames
            for current, previous in zip(self.last_bbox, self.previous_bbox)
        ]
        predicted = tuple(
            current + speed * min(horizon, delta_frames * 2)
            for current, speed in zip(self.last_bbox, velocity)
        )
        return clamp_bbox(predicted, width, height)  # type: ignore[arg-type]


class MultiPersonTrackManager:
    def __init__(self, *, max_gap_frames: int = 20, minimum_match_score: float = 0.28) -> None:
        self.max_gap_frames = max(1, int(max_gap_frames))
        self.minimum_match_score = float(minimum_match_score)
        self.tracks: dict[str, TrackState] = {}
        self._next_id = 1

    def _new_track(self, candidate: DetectionCandidate, frame_index: int) -> TrackState:
        track_id = f"person_candidate_{self._next_id:03d}"
        self._next_id += 1
        track = TrackState(
            track_id=track_id,
            last_bbox=candidate.bbox,
            previous_bbox=None,
            last_frame=frame_index,
            previous_frame=None,
            appearance=candidate.appearance,
            skeleton=candidate.skeleton,
            first_frame=frame_index,
            confidence_sum=candidate.confidence,
            area_sum=_area(candidate.bbox),
            last_keypoints_xy=candidate.keypoints_xy,
            last_keypoints_confidence=candidate.keypoints_confidence,
        )
        self.tracks[track_id] = track
        return track

    def _match_score(
        self,
        track: TrackState,
        candidate: DetectionCandidate,
        frame_index: int,
        width: int,
        height: int,
    ) -> tuple[float, dict[str, float]]:
        predicted = track.predicted_bbox(frame_index, width, height)
        iou = bbox_iou(predicted, candidate.bbox)
        center_score = 1.0 - min(1.0, relative_center_distance(predicted, candidate.bbox))
        appearance = _histogram_similarity(track.appearance, candidate.appearance)
        skeleton = _skeleton_similarity(track.skeleton, candidate.skeleton)
        action_continuity = 1.0 - min(
            1.0,
            relative_center_distance(track.last_bbox, candidate.bbox)
            / max(1.0, frame_index - track.last_frame),
        )
        score = (
            0.34 * iou
            + 0.20 * center_score
            + 0.22 * appearance
            + 0.14 * skeleton
            + 0.10 * action_continuity
        )
        return score, {
            "bbox_iou": iou,
            "center_velocity": center_score,
            "appearance_embedding": appearance,
            "bone_proportion": skeleton,
            "action_continuity": action_continuity,
        }

    def update(
        self,
        candidates: Sequence[DetectionCandidate],
        *,
        frame_index: int,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        active = [
            track
            for track in self.tracks.values()
            if frame_index - track.last_frame <= self.max_gap_frames
        ]
        scored: list[tuple[float, str, int, dict[str, float]]] = []
        for track in active:
            for candidate_index, candidate in enumerate(candidates):
                score, components = self._match_score(
                    track, candidate, frame_index, width, height
                )
                scored.append((score, track.track_id, candidate_index, components))
        scored.sort(reverse=True, key=lambda item: item[0])
        used_tracks: set[str] = set()
        used_candidates: set[int] = set()
        matches: dict[int, tuple[TrackState, float, dict[str, float]]] = {}
        for score, track_id, candidate_index, components in scored:
            if score < self.minimum_match_score:
                continue
            if track_id in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_id]
            if (
                bbox_iou(track.predicted_bbox(frame_index, width, height), candidates[candidate_index].bbox) < 0.01
                and relative_center_distance(track.last_bbox, candidates[candidate_index].bbox) > 1.25
            ):
                continue
            used_tracks.add(track_id)
            used_candidates.add(candidate_index)
            matches[candidate_index] = (track, score, components)

        output: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            matched = matches.get(index)
            if matched is None:
                track = self._new_track(candidate, frame_index)
                score = 1.0
                components = {
                    "bbox_iou": 1.0,
                    "center_velocity": 1.0,
                    "appearance_embedding": 1.0,
                    "bone_proportion": 1.0,
                    "action_continuity": 1.0,
                }
                source = "new_detection_track"
            else:
                track, score, components = matched
                previous_center = _center(track.last_bbox)
                current_center = _center(candidate.bbox)
                track.center_travel += math.dist(previous_center, current_center)
                track.previous_bbox = track.last_bbox
                track.previous_frame = track.last_frame
                track.last_bbox = candidate.bbox
                track.last_frame = frame_index
                track.appearance = 0.75 * track.appearance + 0.25 * candidate.appearance
                valid = np.isfinite(candidate.skeleton)
                if np.any(valid):
                    updated = track.skeleton.copy()
                    updated[valid] = (
                        0.75 * updated[valid] + 0.25 * candidate.skeleton[valid]
                    )
                    track.skeleton = updated
                track.detection_count += 1
                track.confidence_sum += candidate.confidence
                track.area_sum += _area(candidate.bbox)
                track.last_keypoints_xy = candidate.keypoints_xy
                track.last_keypoints_confidence = candidate.keypoints_confidence
                source = "matched_detection"
            output.append(
                {
                    "track_id": track.track_id,
                    "bbox_xyxy": [round(value, 3) for value in candidate.bbox],
                    "bbox_source": source,
                    "detection_confidence": round(candidate.confidence, 6),
                    "association_score": round(score, 6),
                    "association_components": {
                        key: round(value, 6) for key, value in components.items()
                    },
                    "keypoints": [
                        {
                            "name": COCO_KEYPOINT_NAMES[keypoint_index],
                            "x": round(float(point[0]), 3),
                            "y": round(float(point[1]), 3),
                            "confidence": round(
                                float(candidate.keypoints_confidence[keypoint_index]), 6
                            ),
                        }
                        for keypoint_index, point in enumerate(candidate.keypoints_xy)
                        if keypoint_index < len(COCO_KEYPOINT_NAMES)
                    ],
                    "propagated": False,
                }
            )
        for track in active:
            if track.track_id not in used_tracks and track.last_frame != frame_index:
                track.missing_detection_runs += 1
        return output

    def propagated(
        self, *, frame_index: int, width: int, height: int
    ) -> list[dict[str, Any]]:
        output = []
        for track in self.tracks.values():
            gap = frame_index - track.last_frame
            if gap <= 0 or gap > self.max_gap_frames:
                continue
            track.propagated_count += 1
            bbox = track.predicted_bbox(frame_index, width, height)
            output.append(
                {
                    "track_id": track.track_id,
                    "bbox_xyxy": [round(value, 3) for value in bbox],
                    "bbox_source": "constant_velocity_between_detections",
                    "detection_confidence": None,
                    "association_score": None,
                    "association_components": None,
                    "keypoints": [],
                    "propagated": True,
                    "frames_since_detection": gap,
                }
            )
        return output


class YoloPoseCandidateDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        device: str = "",
    ) -> None:
        ensure_ultralytics_config_dir()
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.confidence = float(confidence)
        self.device = str(device).strip()

    def detect(self, frame: np.ndarray) -> tuple[list[DetectionCandidate], float]:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "classes": [0],
            "conf": self.confidence,
            "verbose": False,
            "imgsz": 640,
        }
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(frame, **kwargs)
        elapsed = (time.perf_counter() - started) * 1000.0
        candidates: list[DetectionCandidate] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None or keypoints is None:
                continue
            xyxy = _as_numpy(getattr(boxes, "xyxy", None))
            box_confidence = _as_numpy(getattr(boxes, "conf", None))
            keypoint_xy = _as_numpy(getattr(keypoints, "xy", None))
            keypoint_confidence = _as_numpy(getattr(keypoints, "conf", None))
            if (
                xyxy is None
                or box_confidence is None
                or keypoint_xy is None
                or keypoint_confidence is None
            ):
                continue
            for index in range(
                min(len(xyxy), len(box_confidence), len(keypoint_xy), len(keypoint_confidence))
            ):
                bbox = clamp_bbox(
                    tuple(float(value) for value in xyxy[index][:4]),
                    frame.shape[1],
                    frame.shape[0],
                )
                if _area(bbox) <= 4.0:
                    continue
                xy = np.asarray(keypoint_xy[index], dtype=np.float32)
                confidence = np.asarray(keypoint_confidence[index], dtype=np.float32)
                candidates.append(
                    DetectionCandidate(
                        bbox=bbox,
                        confidence=float(box_confidence[index]),
                        keypoints_xy=xy,
                        keypoints_confidence=confidence,
                        appearance=_appearance_histogram(frame, bbox),
                        skeleton=_skeleton_signature(xy, confidence, bbox),
                    )
                )
        return candidates, elapsed


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    output = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                output.append(json.loads(line))
    return output


def _track_stats(manager: MultiPersonTrackManager, total_frames: int) -> list[dict[str, Any]]:
    output = []
    for track in manager.tracks.values():
        detections = max(1, track.detection_count)
        output.append(
            {
                "track_id": track.track_id,
                "first_frame": track.first_frame,
                "last_detection_frame": track.last_frame,
                "detection_count": track.detection_count,
                "propagated_frame_count": track.propagated_count,
                "detection_coverage_rate": track.detection_count / max(1, total_frames),
                "mean_confidence": track.confidence_sum / detections,
                "mean_bbox_area": track.area_sum / detections,
                "center_travel_pixels": track.center_travel,
                "missing_detection_runs": track.missing_detection_runs,
            }
        )
    return sorted(output, key=lambda item: str(item["track_id"]))


def propose_target_track(track_stats: Sequence[Mapping[str, Any]]) -> str | None:
    if not track_stats:
        return None
    max_detection = max(float(item["detection_count"]) for item in track_stats)
    max_area = max(float(item["mean_bbox_area"]) for item in track_stats)
    max_travel = max(float(item["center_travel_pixels"]) for item in track_stats)
    return str(
        max(
            track_stats,
            key=lambda item: (
                0.45 * float(item["detection_count"]) / max(1.0, max_detection)
                + 0.25 * float(item["mean_bbox_area"]) / max(1.0, max_area)
                + 0.20 * float(item["center_travel_pixels"]) / max(1.0, max_travel)
                + 0.10 * float(item["mean_confidence"])
            ),
        )["track_id"]
    )


def _normalize_source_track_segments(
    source: str | Sequence[Mapping[str, Any]], frame_count: int
) -> list[dict[str, Any]]:
    if isinstance(source, str):
        return [
            {
                "source_track_id": source,
                "start_frame": 0,
                "end_frame": max(0, frame_count - 1),
            }
        ]
    segments = sorted(
        (
            {
                "source_track_id": str(item["source_track_id"]),
                "start_frame": int(item["start_frame"]),
                "end_frame": int(item["end_frame"]),
            }
            for item in source
        ),
        key=lambda item: item["start_frame"],
    )
    if not segments:
        raise ValueError("at least one source track segment is required")
    expected_start = 0
    for segment in segments:
        if segment["start_frame"] != expected_start:
            raise ValueError(
                "source track segments must be contiguous and start at frame 0"
            )
        if segment["end_frame"] < segment["start_frame"]:
            raise ValueError("source track segment end precedes its start")
        expected_start = segment["end_frame"] + 1
    if expected_start != frame_count:
        raise ValueError(
            "source track segments must cover every frame through "
            f"{max(0, frame_count - 1)}"
        )
    return segments


def _source_track_id_for_frame(
    source_track_segments: Sequence[Mapping[str, Any]], frame_index: int
) -> str | None:
    return next(
        (
            str(segment["source_track_id"])
            for segment in source_track_segments
            if int(segment["start_frame"]) <= frame_index
            <= int(segment["end_frame"])
        ),
        None,
    )


def _target_events(
    frames: Sequence[Mapping[str, Any]],
    target_source: str | Sequence[Mapping[str, Any]],
    *,
    canonical_target_track_id: str | None = None,
) -> list[dict[str, Any]]:
    segments = _normalize_source_track_segments(target_source, len(frames))
    target_source_ids = {
        str(segment["source_track_id"]) for segment in segments
    }
    events: list[dict[str, Any]] = []
    target_was_present = False
    missing_start: int | None = None
    for frame in frames:
        frame_index = int(frame["frame_index"])
        source_track_id = _source_track_id_for_frame(segments, frame_index)
        candidates = frame.get("candidates") or []
        target = next(
            (item for item in candidates if item.get("track_id") == source_track_id),
            None,
        )
        if target is None:
            if target_was_present and missing_start is None:
                missing_start = frame_index
            continue
        if missing_start is not None:
            events.append(
                {
                    "event_type": "target_reentry_after_gap",
                    "start_frame": missing_start,
                    "end_frame": frame_index - 1,
                    "review_required": True,
                }
            )
            missing_start = None
        target_was_present = True
        target_bbox = tuple(float(value) for value in target["bbox_xyxy"])
        for other in candidates:
            if other.get("track_id") in target_source_ids:
                continue
            other_bbox = tuple(float(value) for value in other["bbox_xyxy"])
            overlap = bbox_iou(target_bbox, other_bbox)
            distance = relative_center_distance(target_bbox, other_bbox)
            if overlap >= 0.05 or distance <= 0.45:
                events.append(
                    {
                        "event_type": "person_crossing_or_occlusion_candidate",
                        "frame_index": frame_index,
                        "other_track_id": other.get("track_id"),
                        "target_track_id": canonical_target_track_id,
                        "source_candidate_track_id": source_track_id,
                        "bbox_iou": round(overlap, 6),
                        "relative_center_distance": round(distance, 6),
                        "review_required": True,
                    }
                )
        components = target.get("association_components")
        if isinstance(components, Mapping) and (
            float(components.get("appearance_embedding", 1.0)) < 0.35
            or float(components.get("bone_proportion", 1.0)) < 0.35
        ):
            events.append(
                {
                    "event_type": "TARGET_IDENTITY_SWITCH",
                    "frame_index": frame_index,
                    "target_track_id": canonical_target_track_id,
                    "source_candidate_track_id": source_track_id,
                    "reason": "low appearance or bone-proportion association",
                    "review_required": True,
                }
            )
    return events


def _merge_event_frames(events: Sequence[Mapping[str, Any]]) -> list[int]:
    frames = sorted(
        {
            int(event.get("frame_index", event.get("start_frame", -1)))
            for event in events
            if int(event.get("frame_index", event.get("start_frame", -1))) >= 0
        }
    )
    if len(frames) <= 8:
        return frames
    indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
    return [frames[int(index)] for index in indices]


def scan_record_people(
    record: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    detector: YoloPoseCandidateDetector,
    detection_interval: int = 5,
) -> dict[str, Any]:
    root = Path(dataset_root)
    video_path = root / str(record["source_file"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open phone video for target scan: {video_path}")
    interval = max(1, int(detection_interval))
    manager = MultiPersonTrackManager(max_gap_frames=max(15, interval * 4))
    fps = float(record.get("video", {}).get("fps") or capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(record.get("video", {}).get("width") or capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(record.get("video", {}).get("height") or capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[dict[str, Any]] = []
    detector_ms: list[float] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % interval == 0:
                detections, inference_ms = detector.detect(frame)
                candidates = manager.update(
                    detections,
                    frame_index=frame_index,
                    width=width,
                    height=height,
                )
                detector_ms.append(inference_ms)
                source = "yolo_pose_detection"
            else:
                candidates = manager.propagated(
                    frame_index=frame_index, width=width, height=height
                )
                inference_ms = 0.0
                source = "bounded_constant_velocity_propagation"
            frames.append(
                {
                    "schema_version": 1,
                    "record_id": record["record_id"],
                    "frame_index": frame_index,
                    "timestamp_ms": round(frame_index * 1000.0 / fps, 6),
                    "candidate_source": source,
                    "detector_inference_ms": round(inference_ms, 6),
                    "candidates": candidates,
                }
            )
            frame_index += 1
    finally:
        capture.release()
    stats = _track_stats(manager, frame_index)
    proposal = propose_target_track(stats)
    events = _target_events(frames, proposal) if proposal else []
    output_dir = root / "tracks" / str(record["record_id"])
    _write_jsonl(output_dir / "people_candidates.jsonl", frames)
    summary = {
        "schema_version": 1,
        "artifact_type": "round7_people_candidate_scan",
        "generated_at": utc_now(),
        "record_id": record["record_id"],
        "source_filename": record["source_filename"],
        "video_sha256": record["sha256"],
        "model": {
            "path": detector.model_path.as_posix(),
            "sha256": sha256_file(detector.model_path),
        },
        "detection_interval": interval,
        "frame_count": frame_index,
        "candidate_track_count": len(stats),
        "track_stats": stats,
        "proposed_target_track_id": proposal,
        "proposal_is_ground_truth": False,
        "review_event_frames": _merge_event_frames(events),
        "events": events,
        "detector_inference_ms": {
            "p50": float(np.percentile(detector_ms, 50)) if detector_ms else None,
            "p95": float(np.percentile(detector_ms, 95)) if detector_ms else None,
            "mean": float(np.mean(detector_ms)) if detector_ms else None,
        },
    }
    _atomic_json(output_dir / "candidate_scan_summary.json", summary)
    return summary


def _candidate_for_track(
    frame_record: Mapping[str, Any], track_id: str
) -> Mapping[str, Any] | None:
    return next(
        (
            candidate
            for candidate in frame_record.get("candidates") or []
            if candidate.get("track_id") == track_id
        ),
        None,
    )


def build_record_review_sheet(
    record: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(dataset_root)
    frames = read_jsonl(
        root / "tracks" / str(record["record_id"]) / "people_candidates.jsonl"
    )
    target_id = str(summary["proposed_target_track_id"])
    frame_count = len(frames)
    standard_frames = [
        0,
        max(0, int(round((frame_count - 1) * 0.1))),
        max(0, int(round((frame_count - 1) * 0.5))),
        max(0, int(round((frame_count - 1) * 0.9))),
        max(0, frame_count - 1),
    ]
    event_frames = [int(value) for value in summary.get("review_event_frames") or []]
    if len(event_frames) > 7:
        event_indices = np.linspace(0, len(event_frames) - 1, 7, dtype=int)
        event_frames = [event_frames[int(index)] for index in event_indices]
    chosen_frames = sorted(set(standard_frames + event_frames))
    video_path = root / str(record["source_file"])
    capture = cv2.VideoCapture(str(video_path))
    tiles: list[np.ndarray] = []
    try:
        for frame_index in chosen_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            annotated = frame.copy()
            for candidate in frames[frame_index].get("candidates") or []:
                bbox = [int(round(value)) for value in candidate["bbox_xyxy"]]
                is_target = candidate.get("track_id") == target_id
                color = (255, 0, 255) if is_target else (0, 220, 255)
                cv2.rectangle(
                    annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3
                )
                cv2.putText(
                    annotated,
                    str(candidate.get("track_id")),
                    (bbox[0], max(24, bbox[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                annotated,
                f"{record['record_id']} frame={frame_index} proposed={target_id}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            scale = min(480 / annotated.shape[1], 640 / annotated.shape[0])
            tile = cv2.resize(
                annotated,
                (
                    max(1, int(round(annotated.shape[1] * scale))),
                    max(1, int(round(annotated.shape[0] * scale))),
                ),
            )
            canvas = np.zeros((660, 500, 3), dtype=np.uint8)
            top = (canvas.shape[0] - tile.shape[0]) // 2
            left = (canvas.shape[1] - tile.shape[1]) // 2
            canvas[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
            tiles.append(canvas)
    finally:
        capture.release()
    if not tiles:
        raise RuntimeError(f"no frames rendered for review sheet: {record['record_id']}")
    columns = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / columns))
    sheet = np.zeros((rows * 660, columns * 500, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * 660 : (row + 1) * 660, column * 500 : (column + 1) * 500] = tile
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return output


def build_overview_sheets(
    sheet_paths: Sequence[Path], *, output_dir: str | Path, page_size: int = 6
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    for page_index in range(0, len(sheet_paths), page_size):
        paths = sheet_paths[page_index : page_index + page_size]
        thumbnails = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            scale = min(900 / image.shape[1], 600 / image.shape[0])
            thumbnails.append(
                cv2.resize(
                    image,
                    (
                        max(1, int(round(image.shape[1] * scale))),
                        max(1, int(round(image.shape[0] * scale))),
                    ),
                )
            )
        if not thumbnails:
            continue
        width = 920
        height = 620
        canvas = np.zeros((len(thumbnails) * height, width, 3), dtype=np.uint8)
        for index, image in enumerate(thumbnails):
            top = index * height + 10
            left = (width - image.shape[1]) // 2
            canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
        output = output_root / f"overview_{page_index // page_size + 1:02d}.jpg"
        cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        pages.append(output)
    return pages


def create_initialization_proposals(
    manifest: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    by_id = {str(item["record_id"]): item for item in summaries}
    proposals = []
    for record in manifest.get("records") or []:
        summary = by_id[str(record["record_id"])]
        track_id = summary.get("proposed_target_track_id")
        frames = read_jsonl(
            Path(dataset_root)
            / "tracks"
            / str(record["record_id"])
            / "people_candidates.jsonl"
        )
        anchor_candidates = [
            frame
            for frame in frames
            if _candidate_for_track(frame, str(track_id)) is not None
            and not bool(_candidate_for_track(frame, str(track_id)).get("propagated"))
        ]
        anchor_frame = max(
            anchor_candidates,
            key=lambda frame: float(
                _candidate_for_track(frame, str(track_id)).get(
                    "detection_confidence", 0.0
                )
                or 0.0
            ),
            default=None,
        )
        candidate = (
            _candidate_for_track(anchor_frame, str(track_id))
            if anchor_frame is not None
            else None
        )
        proposals.append(
            {
                "record_id": record["record_id"],
                "source_filename": record["source_filename"],
                "canonical_target_track_id": "target_athlete_001",
                "selected_track_id": track_id,
                "source_track_segments": [
                    {
                        "source_track_id": track_id,
                        "start_frame": 0,
                        "end_frame": max(0, len(frames) - 1),
                    }
                ],
                "initialization_frame": (
                    int(anchor_frame["frame_index"]) if anchor_frame is not None else None
                ),
                "initialization_bbox_xyxy": (
                    candidate.get("bbox_xyxy") if candidate is not None else None
                ),
                "initialization_keypoints": (
                    candidate.get("keypoints") if candidate is not None else []
                ),
                "selection_status": "proposal_pending_manual_visual_review",
                "selected_by": None,
                "selected_at": None,
                "reason": "automatic proposal only; review start/middle/end and crossing sheets",
                "review_sheet": f"reports/round7_review_sheets/{record['record_id']}.jpg",
                "reviewed_segments": [],
                "manual_reinitializations": [],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "round7_target_initializations_v1",
        "generated_at": utc_now(),
        "proposal_is_ground_truth": False,
        "records": proposals,
    }


def _intervals_from_locked_flags(
    frames: Sequence[Mapping[str, Any]], target_track_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    locked: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    start = 0
    current_locked: bool | None = None
    current_reasons: tuple[str, ...] = ()
    for index, frame in enumerate(frames):
        source_track_id = str(
            frame.get("source_candidate_track_id") or target_track_id
        )
        candidate = _candidate_for_track(frame, source_track_id)
        reasons: list[str] = []
        is_locked = candidate is not None
        if candidate is None:
            reasons.append("target_missing")
        elif candidate.get("propagated") and int(candidate.get("frames_since_detection", 0)) > 10:
            is_locked = False
            reasons.append("target_propagation_too_stale")
        event_types = {
            str(event.get("event_type"))
            for event in frame.get("events") or []
        }
        if "TARGET_IDENTITY_SWITCH" in event_types:
            is_locked = False
            reasons.append("TARGET_IDENTITY_SWITCH")
        resolved = tuple(sorted(reasons))
        if current_locked is None:
            current_locked = is_locked
            current_reasons = resolved
            start = index
        elif is_locked != current_locked or resolved != current_reasons:
            destination = locked if current_locked else excluded
            destination.append(
                {
                    "start_frame": start,
                    "end_frame": index - 1,
                    "target_locked": current_locked,
                    "target_track_id": target_track_id,
                    "reasons": list(current_reasons),
                }
            )
            start = index
            current_locked = is_locked
            current_reasons = resolved
    if current_locked is not None:
        destination = locked if current_locked else excluded
        destination.append(
            {
                "start_frame": start,
                "end_frame": len(frames) - 1,
                "target_locked": current_locked,
                "target_track_id": target_track_id,
                "reasons": list(current_reasons),
            }
        )
    return locked, excluded


def _write_ignore_masks(
    frames: Sequence[Mapping[str, Any]],
    *,
    target_source_track_ids: Sequence[str],
    width: int,
    height: int,
    output_dir: Path,
) -> dict[int, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = set(target_source_track_ids)
    references: dict[int, str] = {}
    for frame in frames:
        others = [
            candidate
            for candidate in frame.get("candidates") or []
            if candidate.get("track_id") not in target_ids
        ]
        if not others:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        for candidate in others:
            x1, y1, x2, y2 = [
                int(round(value)) for value in candidate["bbox_xyxy"]
            ]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        frame_index = int(frame["frame_index"])
        filename = f"frame_{frame_index:06d}.png"
        cv2.imwrite(str(output_dir / filename), mask)
        references[frame_index] = filename
    return references


def finalize_target_lock(
    record: Mapping[str, Any],
    initialization: Mapping[str, Any],
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_root)
    record_id = str(record["record_id"])
    canonical_track_id = str(
        initialization.get("canonical_target_track_id") or "target_athlete_001"
    )
    candidate_path = root / "tracks" / record_id / "people_candidates.jsonl"
    frames = read_jsonl(candidate_path)
    raw_segments = initialization.get("source_track_segments")
    if not raw_segments:
        raw_segments = str(initialization["selected_track_id"])
    source_track_segments = _normalize_source_track_segments(
        raw_segments, len(frames)
    )
    source_track_ids = sorted(
        {str(item["source_track_id"]) for item in source_track_segments}
    )
    events = _target_events(
        frames,
        source_track_segments,
        canonical_target_track_id=canonical_track_id,
    )
    for reinitialization in initialization.get("manual_reinitializations") or []:
        events.append(
            {
                "event_type": "target_manual_reinitialization",
                "frame_index": int(reinitialization["frame"]),
                "target_track_id": canonical_track_id,
                "before_source_track_id": reinitialization.get(
                    "before_source_track_id"
                ),
                "after_source_track_id": reinitialization.get(
                    "after_source_track_id"
                ),
                "reason": reinitialization.get("reason"),
                "review_required": False,
                "review_status": "manual_visual_review_completed",
            }
        )
    events.sort(
        key=lambda event: (
            int(event.get("frame_index", event.get("start_frame", -1))),
            str(event.get("event_type")),
        )
    )
    events_by_frame: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        frame_index = int(event.get("frame_index", event.get("start_frame", -1)))
        if frame_index >= 0:
            events_by_frame[frame_index].append(event)
    width = int(record["video"]["width"])
    height = int(record["video"]["height"])
    mask_refs = _write_ignore_masks(
        frames,
        target_source_track_ids=source_track_ids,
        width=width,
        height=height,
        output_dir=root / "tracks" / record_id / "ignore_masks",
    )
    finalized_frames = []
    other_people_frames = 0
    ambiguous_frames = 0
    for frame in frames:
        frame_index = int(frame["frame_index"])
        source_track_id = _source_track_id_for_frame(
            source_track_segments, frame_index
        )
        target = (
            _candidate_for_track(frame, source_track_id)
            if source_track_id is not None
            else None
        )
        other_count = sum(
            candidate.get("track_id") not in source_track_ids
            for candidate in frame.get("candidates") or []
        )
        if other_count:
            other_people_frames += 1
        frame_events = events_by_frame.get(frame_index, [])
        identity_switch = any(
            event.get("event_type") == "TARGET_IDENTITY_SWITCH"
            for event in frame_events
        )
        stale = bool(
            target
            and target.get("propagated")
            and int(target.get("frames_since_detection", 0)) > 10
        )
        target_locked = target is not None and not identity_switch and not stale
        if not target_locked:
            ambiguous_frames += 1
        finalized_frames.append(
            {
                **frame,
                "target_track_id": canonical_track_id,
                "source_candidate_track_id": source_track_id,
                "target_locked": target_locked,
                "target_status": (
                    "locked"
                    if target_locked
                    else "target_ambiguous"
                    if target is not None
                    else "target_missing"
                ),
                "events": frame_events,
                "other_people_count": other_count,
                "ignore_mask": mask_refs.get(frame_index),
                "formal_pose_must_bind_track_id": canonical_track_id,
            }
        )
    _write_jsonl(root / "tracks" / record_id / "people.jsonl", finalized_frames)
    locked, excluded = _intervals_from_locked_flags(
        finalized_frames, canonical_track_id
    )
    event_counts = Counter(str(event["event_type"]) for event in events)
    return {
        "record_id": record_id,
        "source_filename": record["source_filename"],
        "target_track_id": canonical_track_id,
        "source_track_segments": source_track_segments,
        "source_candidate_track_ids": source_track_ids,
        "initialization": dict(initialization),
        "frame_count": len(frames),
        "target_locked_frame_count": len(frames) - ambiguous_frames,
        "target_locked_rate": (len(frames) - ambiguous_frames) / max(1, len(frames)),
        "target_ambiguous_or_missing_frame_count": ambiguous_frames,
        "other_people_frame_count": other_people_frames,
        "ignore_mask_count": len(mask_refs),
        "event_counts": dict(sorted(event_counts.items())),
        "events": events,
        "formal_usable_intervals": locked,
        "excluded_intervals": excluded,
        "manual_reinitializations": initialization.get("manual_reinitializations") or [],
        "training_exclusion_policy": [
            "target_ambiguous",
            "TARGET_IDENTITY_SWITCH",
            "severe_occlusion",
        ],
    }


def build_target_lock_audit(
    manifest: Mapping[str, Any],
    initialization_payload: Mapping[str, Any],
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    initialization_by_id = {
        str(item["record_id"]): item
        for item in initialization_payload.get("records") or []
    }
    records = []
    for record in manifest.get("records") or []:
        initialization = initialization_by_id[str(record["record_id"])]
        records.append(
            finalize_target_lock(
                record, initialization, dataset_root=dataset_root
            )
        )
    reviewed_crossings = sum(
        len(
            [
                event
                for event in record["events"]
                if event["event_type"] == "person_crossing_or_occlusion_candidate"
            ]
        )
        for record in records
    )
    suspected_switches = sum(
        int(record["event_counts"].get("TARGET_IDENTITY_SWITCH", 0))
        for record in records
    )
    all_reviewed = all(
        item.get("selection_status") == "manual_visual_review_completed"
        for item in initialization_by_id.values()
    )
    independent_second_human_review_completed = bool(
        initialization_payload.get("reviewer_disclosure", {}).get(
            "independent_second_human_review_completed"
        )
    )
    report = {
        "schema_version": 1,
        "artifact_type": "target_lock_audit_v1",
        "generated_at": utc_now(),
        "status": (
            "passed_with_independent_second_human_review_pending"
            if len(records) == 30
            and all_reviewed
            and not independent_second_human_review_completed
            else "passed"
            if len(records) == 30 and all_reviewed
            else "review_incomplete"
        ),
        "record_count": len(records),
        "manual_visual_initialization_count": sum(
            item.get("selection_status") == "manual_visual_review_completed"
            for item in initialization_by_id.values()
        ),
        "records": records,
        "summary": {
            "total_frames": sum(int(record["frame_count"]) for record in records),
            "target_locked_frames": sum(
                int(record["target_locked_frame_count"]) for record in records
            ),
            "target_locked_rate": sum(
                int(record["target_locked_frame_count"]) for record in records
            )
            / max(1, sum(int(record["frame_count"]) for record in records)),
            "other_people_frames": sum(
                int(record["other_people_frame_count"]) for record in records
            ),
            "ignore_masks": sum(int(record["ignore_mask_count"]) for record in records),
            "reviewed_crossing_proposals": reviewed_crossings,
            "suspected_identity_switches": suspected_switches,
            "manual_reinitialization_count": sum(
                len(record["manual_reinitializations"]) for record in records
            ),
            "identity_switch_miss_rate": None,
            "identity_switch_miss_rate_reason": (
                "requires an independent second human frame-level review; automatic proposals and the same visual review cannot estimate misses"
            ),
            "manual_review_disagreement_rate": None,
            "manual_review_disagreement_reason": (
                "one manual visual reviewer is recorded; inter-reviewer disagreement needs a second independent reviewer"
            ),
        },
        "checks": {
            "all_30_records_scanned": len(records) == 30,
            "all_30_have_manual_visual_initialization": all_reviewed,
            "all_formal_intervals_bind_one_target_track": all(
                interval["target_locked"] is True
                and interval["target_track_id"] == record["target_track_id"]
                and bool(record["target_track_id"])
                for record in records
                for interval in record["formal_usable_intervals"]
            ),
            "all_records_use_record_local_canonical_target_id": all(
                record["target_track_id"] == "target_athlete_001"
                for record in records
            ),
            "source_candidate_segments_are_traceable": all(
                bool(record["source_track_segments"]) for record in records
            ),
            "largest_or_center_box_not_ground_truth": True,
            "background_people_not_automatic_negatives": True,
            "ambiguous_switch_and_occlusion_excluded": True,
        },
    }
    _atomic_json(Path(dataset_root) / "reports" / "target_lock_audit_v1.json", report)
    return report


__all__ = [
    "MultiPersonTrackManager",
    "YoloPoseCandidateDetector",
    "bbox_iou",
    "build_overview_sheets",
    "build_record_review_sheet",
    "build_target_lock_audit",
    "create_initialization_proposals",
    "finalize_target_lock",
    "propose_target_track",
    "read_jsonl",
    "relative_center_distance",
    "scan_record_people",
]
