"""Round 11 offline ONI Depth/IR subject audit and research reports.

The current recordings contain independent Depth and IR streams but no Color.
All target tracks produced here are review proposals, never identity truth.
Depth and IR are processed independently and are not spatially registered or
frame-paired by this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.configuration import ConfigValidationError, load_simple_yaml, reject_unknown_fields


ARTIFACT_VERSION = "round11_oni_research_v1"
STREAMS = ("depth", "ir")
FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "phone_frame_id",
        "phone_timestamp_ms",
        "phone_joint",
        "oni_depth_pixel",
        "rgb_depth_transform",
        "rgb_depth_pairs",
        "phone_oni_offset",
    }
)


@dataclass(frozen=True, slots=True)
class OniResearchContract:
    version: str
    mode: str
    sample_count_per_stream: int
    minimum_component_area_px: int
    maximum_component_area_ratio: float
    depth_foreground_delta_mm: int
    ir_mad_multiplier: float
    proposal_confidence_threshold: float
    require_independent_depth_ir_tracking: bool
    require_human_identity_confirmation: bool
    allow_rgb_depth_registration: bool
    allow_phone_oni_pairing: bool
    allow_phone_frame_labels: bool
    allow_ir_as_rgb: bool
    allow_uncalibrated_ground_plane: bool
    allow_contact_truth_from_silhouette: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boolean(value: object, *, path: Path, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError("must be true or false", path=path, key=key)
    return value


def _positive_int(value: object, *, path: Path, key: str) -> int:
    if isinstance(value, bool):
        raise ConfigValidationError("must be a positive integer", path=path, key=key)
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigValidationError("must be a positive integer", path=path, key=key) from exc
    if resolved <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ConfigValidationError("must be a positive integer", path=path, key=key)
    return resolved


def _positive_float(
    value: object,
    *,
    path: Path,
    key: str,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ConfigValidationError("must be a positive number", path=path, key=key)
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigValidationError("must be a positive number", path=path, key=key) from exc
    if not math.isfinite(resolved) or resolved <= 0 or (
        upper is not None and resolved > upper
    ):
        raise ConfigValidationError("must be in the allowed positive range", path=path, key=key)
    return resolved


def load_oni_research_contract(path: str | Path) -> OniResearchContract:
    resolved = Path(path)
    values = load_simple_yaml(resolved)
    allowed = {
        "schema_version",
        "contract_version",
        "mode",
        "sample_count_per_stream",
        "minimum_component_area_px",
        "maximum_component_area_ratio",
        "depth_foreground_delta_mm",
        "ir_mad_multiplier",
        "proposal_confidence_threshold",
        "require_independent_depth_ir_tracking",
        "require_human_identity_confirmation",
        "allow_rgb_depth_registration",
        "allow_phone_oni_pairing",
        "allow_phone_frame_labels",
        "allow_ir_as_rgb",
        "allow_uncalibrated_ground_plane",
        "allow_contact_truth_from_silhouette",
    }
    reject_unknown_fields(values, allowed, path=resolved)
    if values.get("schema_version") != 1:
        raise ConfigValidationError("schema_version must be 1", path=resolved)
    contract = OniResearchContract(
        version=str(values.get("contract_version", "")),
        mode=str(values.get("mode", "")),
        sample_count_per_stream=_positive_int(
            values.get("sample_count_per_stream"),
            path=resolved,
            key="sample_count_per_stream",
        ),
        minimum_component_area_px=_positive_int(
            values.get("minimum_component_area_px"),
            path=resolved,
            key="minimum_component_area_px",
        ),
        maximum_component_area_ratio=_positive_float(
            values.get("maximum_component_area_ratio"),
            path=resolved,
            key="maximum_component_area_ratio",
            upper=1.0,
        ),
        depth_foreground_delta_mm=_positive_int(
            values.get("depth_foreground_delta_mm"),
            path=resolved,
            key="depth_foreground_delta_mm",
        ),
        ir_mad_multiplier=_positive_float(
            values.get("ir_mad_multiplier"),
            path=resolved,
            key="ir_mad_multiplier",
        ),
        proposal_confidence_threshold=_positive_float(
            values.get("proposal_confidence_threshold"),
            path=resolved,
            key="proposal_confidence_threshold",
            upper=1.0,
        ),
        require_independent_depth_ir_tracking=_boolean(
            values.get("require_independent_depth_ir_tracking"),
            path=resolved,
            key="require_independent_depth_ir_tracking",
        ),
        require_human_identity_confirmation=_boolean(
            values.get("require_human_identity_confirmation"),
            path=resolved,
            key="require_human_identity_confirmation",
        ),
        allow_rgb_depth_registration=_boolean(
            values.get("allow_rgb_depth_registration"),
            path=resolved,
            key="allow_rgb_depth_registration",
        ),
        allow_phone_oni_pairing=_boolean(
            values.get("allow_phone_oni_pairing"),
            path=resolved,
            key="allow_phone_oni_pairing",
        ),
        allow_phone_frame_labels=_boolean(
            values.get("allow_phone_frame_labels"),
            path=resolved,
            key="allow_phone_frame_labels",
        ),
        allow_ir_as_rgb=_boolean(
            values.get("allow_ir_as_rgb"),
            path=resolved,
            key="allow_ir_as_rgb",
        ),
        allow_uncalibrated_ground_plane=_boolean(
            values.get("allow_uncalibrated_ground_plane"),
            path=resolved,
            key="allow_uncalibrated_ground_plane",
        ),
        allow_contact_truth_from_silhouette=_boolean(
            values.get("allow_contact_truth_from_silhouette"),
            path=resolved,
            key="allow_contact_truth_from_silhouette",
        ),
    )
    if contract.version != "oni_research_v1" or contract.mode != "offline_oni":
        raise ConfigValidationError("must declare oni_research_v1/offline_oni", path=resolved)
    if not contract.require_independent_depth_ir_tracking:
        raise ConfigValidationError(
            "Depth and IR tracking must remain independent",
            path=resolved,
            key="require_independent_depth_ir_tracking",
        )
    if not contract.require_human_identity_confirmation:
        raise ConfigValidationError(
            "human identity confirmation must be required",
            path=resolved,
            key="require_human_identity_confirmation",
        )
    prohibited = {
        "allow_rgb_depth_registration": contract.allow_rgb_depth_registration,
        "allow_phone_oni_pairing": contract.allow_phone_oni_pairing,
        "allow_phone_frame_labels": contract.allow_phone_frame_labels,
        "allow_ir_as_rgb": contract.allow_ir_as_rgb,
        "allow_uncalibrated_ground_plane": contract.allow_uncalibrated_ground_plane,
        "allow_contact_truth_from_silhouette": contract.allow_contact_truth_from_silhouette,
    }
    enabled = [key for key, value in prohibited.items() if value]
    if enabled:
        raise ConfigValidationError(
            f"prohibited capability enabled: {', '.join(enabled)}",
            path=resolved,
        )
    return contract


def sample_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(frame_count, sample_count)
    return sorted(
        {
            int(round(value))
            for value in np.linspace(0, frame_count - 1, count, dtype=np.float64)
        }
    )


def _read_timeline(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "output_frame": int(row["output_frame"]),
                "source_frame_index": int(row["source_frame_index"]),
                "timestamp_us": int(row["timestamp_us"]),
            }
        )
    return result


def _iou(first: Sequence[int] | None, second: Sequence[int]) -> float:
    if first is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    union += max(0, bx2 - bx1) * max(0, by2 - by1) - intersection
    return intersection / union if union else 0.0


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    output = mask.astype(np.uint8) * 255
    output = cv2.morphologyEx(
        output,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return cv2.morphologyEx(
        output,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=2,
    )


def _component_candidates(
    mask: np.ndarray,
    *,
    minimum_area: int,
    maximum_area_ratio: float,
) -> list[dict[str, object]]:
    height, width = mask.shape
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    maximum_area = int(height * width * maximum_area_ratio)
    candidates: list[dict[str, object]] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if area < minimum_area or area > maximum_area:
            continue
        bbox = [x, y, x + box_width, y + box_height]
        touches = sum(
            (
                x <= 1,
                y <= 1,
                x + box_width >= width - 1,
                y + box_height >= height - 1,
            )
        )
        candidates.append(
            {
                "bbox": bbox,
                "area_px": area,
                "centroid": [float(centroids[label][0]), float(centroids[label][1])],
                "edge_touch_count": touches,
            }
        )
    return candidates


def _select_candidate(
    candidates: Sequence[Mapping[str, object]],
    *,
    previous_bbox: Sequence[int] | None,
    width: int,
    height: int,
) -> dict[str, object] | None:
    if not candidates:
        return None
    if previous_bbox is None:
        measured = [
            item
            for item in candidates
            if item.get("surface_depth_raw") is not None
        ]
        if measured:
            # The current ONI target is the foreground athlete.  Initialize the
            # independent Depth proposal from the nearest measured component,
            # then let temporal continuity carry the same candidate.  This is
            # still a review proposal, not identity truth.
            return dict(
                min(
                    measured,
                    key=lambda item: float(item["surface_depth_raw"]),
                )
            )
    diagonal = math.hypot(width, height)

    def score(item: Mapping[str, object]) -> float:
        bbox = item["bbox"]
        centroid = item["centroid"]
        assert isinstance(bbox, list) and isinstance(centroid, list)
        area_ratio = float(item["area_px"]) / float(width * height)
        edge_penalty = 0.18 * int(item["edge_touch_count"])
        continuity = 1.8 * _iou(previous_bbox, bbox)
        if previous_bbox is None:
            center_distance = math.hypot(
                float(centroid[0]) - width / 2.0,
                float(centroid[1]) - height / 2.0,
            ) / diagonal
        else:
            previous_center = (
                (previous_bbox[0] + previous_bbox[2]) / 2.0,
                (previous_bbox[1] + previous_bbox[3]) / 2.0,
            )
            center_distance = math.hypot(
                float(centroid[0]) - previous_center[0],
                float(centroid[1]) - previous_center[1],
            ) / diagonal
        return continuity + min(0.8, area_ratio * 12.0) - center_distance - edge_penalty

    return dict(max(candidates, key=score))


def _depth_masks(
    frames: np.ndarray,
    contract: OniResearchContract,
) -> tuple[list[np.ndarray], list[str]]:
    background = np.max(frames, axis=0)
    masks: list[np.ndarray] = []
    methods: list[str] = []
    for frame in frames:
        valid = frame > 0
        foreground = valid & (background > 0) & (
            frame.astype(np.int32) + contract.depth_foreground_delta_mm
            < background.astype(np.int32)
        )
        cleaned = _clean_mask(foreground)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned,
            8,
        )
        # Depth quantization/jitter can turn a ceiling or upper wall into one
        # large temporal component.  It is not a plausible independently
        # tracked athlete; remove only large regions attached to the top edge.
        for label in range(1, component_count):
            top = int(stats[label, cv2.CC_STAT_TOP])
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            if top <= int(frame.shape[0] * 0.12) and (
                area >= int(frame.size * 0.03)
                or width >= int(frame.shape[1] * 0.35)
            ):
                cleaned[labels == label] = 0
        if np.count_nonzero(cleaned) < contract.minimum_component_area_px:
            values = frame[valid]
            if values.size:
                near = float(np.percentile(values, 30.0))
                cleaned = _clean_mask(valid & (frame <= near))
                component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    cleaned,
                    8,
                )
                for label in range(1, component_count):
                    top = int(stats[label, cv2.CC_STAT_TOP])
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    width = int(stats[label, cv2.CC_STAT_WIDTH])
                    if top <= int(frame.shape[0] * 0.12) and (
                        area >= int(frame.size * 0.03)
                        or width >= int(frame.shape[1] * 0.35)
                    ):
                        cleaned[labels == label] = 0
                methods.append("near_surface_fallback")
            else:
                methods.append("no_valid_depth")
        else:
            methods.append("temporal_far_surface_background")
        masks.append(cleaned)
    return masks, methods


def _ir_masks(
    frames: np.ndarray,
    contract: OniResearchContract,
) -> tuple[list[np.ndarray], list[str]]:
    background = np.median(frames.astype(np.float32), axis=0)
    deviation = np.abs(frames.astype(np.float32) - background)
    mad = np.median(deviation, axis=0)
    # These GRAY16 files use only about 10 effective bits.  A fixed 16-bit
    # threshold would erase the athlete.  Keep independently measured temporal
    # IR deviations, remove isolated sensor speckles, then connect the sparse
    # body texture without consulting Depth.
    global_floor = max(8.0, float(np.median(mad)) * contract.ir_mad_multiplier)
    masks: list[np.ndarray] = []
    for index in range(frames.shape[0]):
        raw = (
            deviation[index]
            > np.maximum(global_floor, mad * contract.ir_mad_multiplier)
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        retained = np.zeros(raw.shape, dtype=np.uint8)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= 16:
                retained[labels == label] = 255
        retained = cv2.dilate(
            retained,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        masks.append(
            cv2.morphologyEx(
                retained,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            )
        )
    return masks, ["temporal_ir_median_deviation"] * len(masks)


def propose_subject_track(
    frames: np.ndarray,
    *,
    modality: str,
    sampled_output_frames: Sequence[int],
    timeline: Sequence[Mapping[str, object]],
    contract: OniResearchContract,
) -> list[dict[str, object]]:
    """Build independent automatic review proposals for one ONI stream."""
    if modality not in STREAMS:
        raise ValueError(f"unsupported modality: {modality}")
    if frames.ndim != 3:
        raise ValueError("frames must have shape [frame, height, width]")
    masks, methods = (
        _depth_masks(frames, contract)
        if modality == "depth"
        else _ir_masks(frames, contract)
    )
    height, width = frames.shape[1:]
    timeline_by_output = {
        int(row["output_frame"]): row for row in timeline
    }
    previous_bbox: list[int] | None = None
    proposals: list[dict[str, object]] = []
    for index, (frame, mask, method) in enumerate(zip(frames, masks, methods)):
        candidates = _component_candidates(
            mask,
            minimum_area=contract.minimum_component_area_px,
            maximum_area_ratio=contract.maximum_component_area_ratio,
        )
        if modality == "depth":
            plausible_candidates: list[dict[str, object]] = []
            for candidate in candidates:
                candidate_bbox = candidate["bbox"]
                assert isinstance(candidate_bbox, list)
                cx1, cy1, cx2, cy2 = (int(value) for value in candidate_bbox)
                box_width = cx2 - cx1
                box_height = cy2 - cy1
                box_area = max(1, box_width * box_height)
                fill_ratio = float(candidate["area_px"]) / box_area
                # Reject sensor-boundary structures and sparse wall/ceiling
                # regions before target initialization.  A true athlete that
                # is only partly in frame remains no-candidate until enough of
                # the body is visible for an auditable proposal.
                if int(candidate["edge_touch_count"]) > 0:
                    continue
                if fill_ratio < 0.32 and (
                    box_width >= int(width * 0.25)
                    or box_height >= int(height * 0.40)
                ):
                    continue
                if box_width >= int(width * 0.55) or box_height >= int(height * 0.85):
                    continue
                component_values = frame[cy1:cy2, cx1:cx2]
                component_values = component_values[component_values > 0]
                candidate["surface_depth_raw"] = (
                    float(np.median(component_values))
                    if component_values.size
                    else None
                )
                candidate["component_fill_ratio"] = round(fill_ratio, 6)
                plausible_candidates.append(candidate)
            candidates = plausible_candidates
        selected = _select_candidate(
            candidates,
            previous_bbox=previous_bbox,
            width=width,
            height=height,
        )
        output_frame = int(sampled_output_frames[index])
        time_row = timeline_by_output.get(output_frame, {})
        if selected is None:
            proposals.append(
                {
                    "modality": modality,
                    "output_frame": output_frame,
                    "source_frame_index": time_row.get("source_frame_index"),
                    "timestamp_us": time_row.get("timestamp_us"),
                    "target_track_id": f"oni_{modality}_target_candidate_001",
                    "target_lock_status": "no_candidate",
                    "human_confirmed": False,
                    "bbox_px": None,
                    "bbox_normalized": None,
                    "confidence": 0.0,
                    "segmentation_method": method,
                    "metric_surface_distance_m": None,
                    "metric_surface_distance_scope": (
                        "not_applicable_ir"
                        if modality == "ir"
                        else "not_available"
                    ),
                    "review_required": True,
                }
            )
            previous_bbox = None
            continue
        bbox = [int(value) for value in selected["bbox"]]
        x1, y1, x2, y2 = bbox
        area_ratio = float(selected["area_px"]) / float(width * height)
        continuity = _iou(previous_bbox, bbox)
        edge_factor = max(0.0, 1.0 - 0.2 * int(selected["edge_touch_count"]))
        method_factor = 0.65 if "fallback" in method else 1.0
        shape_factor = 1.0
        quality_reasons: list[str] = []
        if modality == "depth":
            box_width_ratio = (x2 - x1) / width
            box_height_ratio = (y2 - y1) / height
            fill_ratio = float(selected.get("component_fill_ratio", 0.0) or 0.0)
            if box_width_ratio >= 0.35 or box_height_ratio >= 0.60:
                shape_factor *= 0.35
                quality_reasons.append("depth_bbox_too_large_for_reliable_subject_lock")
            if fill_ratio < 0.40 and (
                box_width_ratio >= 0.20 or box_height_ratio >= 0.30
            ):
                shape_factor *= 0.65
                quality_reasons.append("sparse_depth_component")
        confidence = max(
            0.0,
            min(
                0.95,
                method_factor
                * edge_factor
                * shape_factor
                * (0.22 + min(0.43, area_ratio * 8.0) + 0.30 * continuity),
            ),
        )
        selected_mask = mask[y1:y2, x1:x2] > 0
        selected_values = frame[y1:y2, x1:x2][selected_mask]
        if modality == "depth":
            selected_values = selected_values[selected_values > 0]
            metric_distance = (
                float(np.median(selected_values)) / 1000.0
                if selected_values.size
                else None
            )
            distance_scope = (
                "oni_surface_line_of_sight_not_body_joint_or_ground_distance"
                if metric_distance is not None
                else "not_available"
            )
        else:
            metric_distance = None
            distance_scope = "not_applicable_ir"
        status = (
            "automated_candidate"
            if confidence >= contract.proposal_confidence_threshold
            else "low_confidence_candidate"
        )
        proposals.append(
            {
                "modality": modality,
                "output_frame": output_frame,
                "source_frame_index": time_row.get("source_frame_index"),
                "timestamp_us": time_row.get("timestamp_us"),
                "target_track_id": f"oni_{modality}_target_candidate_001",
                "target_lock_status": status,
                "human_confirmed": False,
                "bbox_px": bbox,
                "bbox_normalized": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
                "confidence": round(confidence, 6),
                "segmentation_method": method,
                "component_area_px": int(selected["area_px"]),
                "proposal_quality_reasons": quality_reasons,
                "metric_surface_distance_m": (
                    round(metric_distance, 6)
                    if metric_distance is not None
                    else None
                ),
                "metric_surface_distance_scope": distance_scope,
                "review_required": True,
            }
        )
        previous_bbox = bbox
    return proposals


def _preview_image(frame: np.ndarray, modality: str) -> np.ndarray:
    if modality == "depth":
        valid = frame > 0
        display = np.zeros(frame.shape, dtype=np.uint8)
        if np.any(valid):
            values = np.clip(frame.astype(np.float32), 500.0, 8000.0)
            display[valid] = np.clip(
                255.0 - (values[valid] - 500.0) * 255.0 / 7500.0,
                0,
                255,
            ).astype(np.uint8)
        return cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
    lower, upper = np.percentile(frame, (1.0, 99.0))
    if upper <= lower:
        gray = np.zeros(frame.shape, dtype=np.uint8)
    else:
        gray = np.clip(
            (frame.astype(np.float32) - lower) * 255.0 / (upper - lower),
            0,
            255,
        ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def create_contact_sheet(
    path: Path,
    frames: np.ndarray,
    proposals: Sequence[Mapping[str, object]],
    *,
    modality: str,
    record_id: str,
) -> Path:
    thumbnails: list[np.ndarray] = []
    for frame, proposal in zip(frames, proposals):
        image = _preview_image(frame, modality)
        bbox = proposal.get("bbox_px")
        confidence = float(proposal.get("confidence", 0.0) or 0.0)
        color = (40, 220, 40) if confidence >= 0.30 else (0, 180, 255)
        if isinstance(bbox, list):
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
        cv2.putText(
            image,
            f"f{proposal['output_frame']} p={confidence:.2f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
    columns = 4
    rows = max(1, math.ceil(len(thumbnails) / columns))
    canvas = np.zeros((rows * 240 + 54, columns * 320, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{record_id} | {modality.upper()} ONLY | AUTO PROPOSAL - HUMAN REVIEW REQUIRED",
        (10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for index, image in enumerate(thumbnails):
        row, column = divmod(index, columns)
        canvas[54 + row * 240 : 54 + (row + 1) * 240, column * 320 : (column + 1) * 320] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write preview: {path}")
    return path


def _load_sampled_frames(
    stream_root: Path,
    indices: Sequence[int],
) -> np.ndarray:
    frame_paths = sorted((stream_root / "frames").glob("*.npy"))
    if not frame_paths:
        return np.empty((0, 0, 0), dtype=np.uint16)
    return np.stack(
        [np.load(frame_paths[index], allow_pickle=False) for index in indices],
        axis=0,
    )


ERROR_OBSERVABILITY = {
    "FOOT_DESYNCHRONIZED": ("research_candidate_only", "feet are not identified in raw silhouette"),
    "HANDS_FEET_TOO_FAR": ("research_candidate_only", "body-part correspondence is unavailable"),
    "NO_CHEST_CONTACT": ("not_reliably_observable", "floor plane and chest contact are uncalibrated"),
    "EXTRA_STEP": ("research_candidate_only", "fast foot events need body-part tracking and denser review"),
    "KETTLEBELL_SWING": ("research_candidate_only", "equipment is not independently identified"),
    "SAME_LEG_CONSECUTIVE": ("not_reliably_observable", "left/right leg identity is unavailable"),
    "NO_KNEE_CONTACT": ("not_reliably_observable", "floor plane and knee contact are uncalibrated"),
    "HIP_NOT_EXTENDED": ("research_candidate_only", "joint angles are unavailable"),
    "HANDLE_AROUND_KNEES": ("not_reliably_observable", "handle and knee identities are unavailable"),
    "LEAN_TOO_MUCH": ("research_candidate_only", "coarse contour is available but trunk angle is uncalibrated"),
    "NOT_DEEP_ENOUGH": ("research_candidate_only", "hip/knee identities and ground frame are unavailable"),
    "HEEL_RISE": ("not_reliably_observable", "heel contact cannot be separated from silhouette"),
}


def _modality_evidence_contract() -> list[dict[str, object]]:
    return [
        {
            "evidence": "target_surface_line_of_sight_distance",
            "modality": "depth",
            "observability": "DIRECT_SENSOR_MEASUREMENT_WITH_SCOPE_LIMIT",
            "reliable_when": [
                "valid_depth_pixel",
                "human_confirmed_target_bbox",
                "known DEPTH_1_MM scale",
            ],
            "limitations": [
                "not camera/world XYZ without calibration",
                "not body-joint depth",
                "not traveled competition distance",
            ],
        },
        {
            "evidence": "coarse_target_contour",
            "modality": "depth_or_ir_independent",
            "observability": "CONDITIONAL_AFTER_HUMAN_TARGET_REVIEW",
            "reliable_when": ["single visible athlete", "stable independent target track"],
            "limitations": ["no body-part identity", "occlusion merges contours"],
        },
        {
            "evidence": "coarse_spatial_motion",
            "modality": "depth_or_ir_independent",
            "observability": "CONDITIONAL_AFTER_HUMAN_TARGET_REVIEW",
            "reliable_when": ["stable track", "sufficient checkpoint sampling"],
            "limitations": ["not a calibrated trajectory", "sampled checkpoints miss fast events"],
        },
        {
            "evidence": "ground_plane",
            "modality": "depth",
            "observability": "UNOBSERVABLE_RELIABLY_CURRENT_DATA",
            "reliable_when": [],
            "limitations": ["camera intrinsics/extrinsics and ground calibration unavailable"],
        },
        {
            "evidence": "body_or_equipment_contact",
            "modality": "depth_or_ir_independent",
            "observability": "UNOBSERVABLE_RELIABLY_CURRENT_DATA",
            "reliable_when": [],
            "limitations": [
                "silhouette overlap is not physical contact",
                "body parts and equipment are not identified",
            ],
        },
    ]


def _record_summary(
    record: Mapping[str, object],
    modality_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    errors = [str(value) for value in record.get("expected_errors_unverified", [])]
    error_status = [
        {
            "error_code": code,
            "criterion_id": f"{record['action']}.{code.lower()}.oni_round11_v1",
            "status": ERROR_OBSERVABILITY.get(
                code,
                ("not_assessed", "no modality assessment declared"),
            )[0],
            "reason": ERROR_OBSERVABILITY.get(
                code,
                ("not_assessed", "no modality assessment declared"),
            )[1],
            "label_confirmed": False,
        }
        for code in errors
    ]
    return {
        "record_id": record["record_id"],
        "action": record["action"],
        "recording_intent_code": record["recording_intent_code"],
        "recording_intent_verified": bool(record.get("recording_intent_verified")),
        "subject_id": record.get("subject_id"),
        "subject_identity_confirmed": False,
        "modalities": dict(modality_results),
        "expected_error_observability": error_status,
        "training_eligible": False,
        "review_gate": "independent_human_depth_and_ir_subject_review_pending",
    }


def _recapture_plan(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    action_counts: dict[str, int] = {}
    errors: list[dict[str, object]] = []
    for record in records:
        action = str(record["action"])
        action_counts[action] = action_counts.get(action, 0) + 1
        for code in record.get("expected_errors_unverified", []):
            status, reason = ERROR_OBSERVABILITY.get(
                str(code),
                ("not_assessed", "no modality assessment declared"),
            )
            errors.append(
                {
                    "action": action,
                    "error_code": str(code),
                    "oni_observability": status,
                    "reason": reason,
                    "recommended_phone_capture": [
                        "full body and relevant equipment visible",
                        "60 fps or higher for contact/step timing when supported",
                        "clear floor/lane context",
                        "repeat compliant, error, and boundary severity takes",
                        "capture authorization and stable subject_id",
                    ],
                }
            )
    priority_actions = ["skierg", "sled_pull", "sled_push"]
    return {
        "schema_version": 1,
        "artifact_type": "round11_phone_recapture_plan",
        "generated_at": utc_now(),
        "source_scope": "current_oni_video_level_findings_only",
        "does_not_create_phone_labels": True,
        "priority_order": [
            {
                "priority": index + 1,
                "action": action,
                "reason": (
                    "Round 12 priority and current ONI has no verified RGB error/boundary truth"
                ),
                "current_oni_record_count": action_counts.get(action, 0),
            }
            for index, action in enumerate(priority_actions)
        ],
        "error_recaptures": errors,
        "cross_cutting_captures": [
            "continuous mixed actions with transitions and idle",
            "unknown/OOD motions",
            "unseen subjects and devices",
            "occlusion and other people",
            "fast endpoints and contact events",
        ],
    }


def _future_rgbd_value_report(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "round11_future_synchronized_rgbd_value",
        "generated_at": utc_now(),
        "current_dataset": {
            "record_count": len(records),
            "color_depth_record_count": 0,
            "depth_ir_only_record_count": len(records),
            "rgb_depth_calibration_available": False,
            "capture_synchronization_confirmed": False,
            "phone_oni_pair_count": 0,
            "direct_rgbd_training_eligible_record_count": 0,
        },
        "current_conclusion": (
            "Current ONI Depth+IR is useful for modality-specific research but cannot "
            "validate RGB pose against registered depth."
        ),
        "future_value": [
            {
                "capability": "registered RGB keypoints with metric depth",
                "value": "high",
                "requires": [
                    "same-device synchronized Color and Depth",
                    "stored intrinsics and Color-to-Depth extrinsics",
                    "registration residual audit",
                    "human-confirmed target identity",
                ],
            },
            {
                "capability": "ground/contact research",
                "value": "high_but_not_automatic_truth",
                "requires": [
                    "ground-plane calibration",
                    "body/equipment annotations",
                    "independent human event anchors",
                ],
            },
            {
                "capability": "phone RGB production accuracy",
                "value": "indirect",
                "requires": [
                    "separate phone-domain validation",
                    "no unpaired distillation",
                    "subject-disjoint evaluation",
                ],
            },
        ],
        "minimum_capture_protocol": [
            "hardware-synchronized Color and Depth from the same RGB-D device",
            "IR retained as a separate sensor stream",
            "ChArUco/checkerboard calibration at every resolution and device setup",
            "shared timestamps, frame indices, dropped-frame accounting and hashes",
            "identity slate, authorization and target selection before action",
            "full body, floor, lane and action-specific equipment visibility",
            "compliant/error/boundary repeats plus idle/transition/OOD",
        ],
        "explicitly_forbidden": [
            "retrofitting current phone frames to current ONI frames",
            "mapping current phone joints to ONI depth pixels",
            "treating IR as RGB",
            "unpaired teacher-student distillation",
        ],
    }


def _validate_no_forbidden_fields(value: object, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_OUTPUT_FIELDS:
                errors.append(f"{path}.{key}: forbidden field")
            errors.extend(_validate_no_forbidden_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_validate_no_forbidden_fields(child, f"{path}[{index}]"))
    return errors


def build_round11_reports(
    dataset_root: str | Path,
    *,
    contract_path: str | Path,
    create_previews: bool = True,
) -> dict[str, Path]:
    root = Path(dataset_root)
    contract = load_oni_research_contract(contract_path)
    manifest = load_json(root / "manifests" / "oni_records.json")
    records = [
        dict(item)
        for item in manifest.get("records", [])
        if isinstance(item, Mapping)
    ]
    report_root = root / "reports"
    track_root = root / "oni_tracks"
    preview_root = report_root / "round11_subject_previews"
    record_reports: list[dict[str, object]] = []
    preview_artifacts: list[dict[str, object]] = []
    total_proposals = 0
    total_candidates = 0
    for record in records:
        record_id = str(record["record_id"])
        if record.get("paired_group_id") is not None:
            raise ValueError(f"{record_id}: current ONI paired_group_id must remain null")
        metadata_path = root / "extracted" / record_id / "metadata.json"
        metadata = load_json(metadata_path)
        if metadata.get("streams", {}).get("color", {}).get("exists") is True:
            raise ValueError(f"{record_id}: Round 11 current-data path expects no Color")
        modality_results: dict[str, dict[str, object]] = {}
        for modality in STREAMS:
            stream_root = root / "extracted" / record_id / modality
            timeline = _read_timeline(stream_root / "index.csv")
            indices = sample_frame_indices(
                len(timeline),
                contract.sample_count_per_stream,
            )
            frames = _load_sampled_frames(stream_root, indices)
            if not indices or frames.size == 0:
                proposals: list[dict[str, object]] = []
            else:
                output_frames = [int(timeline[index]["output_frame"]) for index in indices]
                proposals = propose_subject_track(
                    frames,
                    modality=modality,
                    sampled_output_frames=output_frames,
                    timeline=timeline,
                    contract=contract,
                )
            track_path = write_jsonl(
                track_root / record_id / f"{modality}_target_proposals.jsonl",
                proposals,
            )
            candidate_count = sum(
                item["target_lock_status"] == "automated_candidate"
                for item in proposals
            )
            total_proposals += len(proposals)
            total_candidates += candidate_count
            confidences = [float(item["confidence"]) for item in proposals]
            modality_results[modality] = {
                "stream_present": bool(proposals),
                "processed_independently": True,
                "sampled_checkpoint_count": len(proposals),
                "automated_candidate_count": candidate_count,
                "candidate_rate": (
                    round(candidate_count / len(proposals), 6)
                    if proposals
                    else 0.0
                ),
                "confidence_p50": (
                    round(float(np.percentile(confidences, 50)), 6)
                    if confidences
                    else None
                ),
                "track_path": track_path.relative_to(root).as_posix(),
                "track_sha256": file_sha256(track_path),
                "human_confirmed_checkpoint_count": 0,
                "identity_status": "pending_independent_human_review",
            }
            if create_previews and proposals:
                preview_path = create_contact_sheet(
                    preview_root / record_id / f"{modality}_subject_proposals.jpg",
                    frames,
                    proposals,
                    modality=modality,
                    record_id=record_id,
                )
                preview_artifacts.append(
                    {
                        "record_id": record_id,
                        "modality": modality,
                        "path": preview_path.relative_to(root).as_posix(),
                        "sha256": file_sha256(preview_path),
                        "independent_stream_only": True,
                        "human_review_required": True,
                    }
                )
        record_reports.append(_record_summary(record, modality_results))

    subject_report = {
        "schema_version": 1,
        "artifact_type": "round11_oni_subject_audit",
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": utc_now(),
        "contract_version": contract.version,
        "mode": contract.mode,
        "status": "engineering_complete_independent_human_subject_review_pending",
        "record_count": len(records),
        "depth_ir_independent": True,
        "target_proposal_count": total_proposals,
        "accepted_automatic_candidate_count": total_candidates,
        "human_confirmed_target_count": 0,
        "identity_switch_rate": None,
        "release_or_training_eligible_record_count": 0,
        "preview_artifacts": preview_artifacts,
        "records": record_reports,
    }
    observability_report = {
        "schema_version": 1,
        "artifact_type": "round11_oni_modality_observability",
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": utc_now(),
        "contract_version": contract.version,
        "modality_evidence": _modality_evidence_contract(),
        "error_observability": [
            {
                "error_code": code,
                "status": status,
                "reason": reason,
                "verified_error_labels_in_current_oni": 0,
            }
            for code, (status, reason) in sorted(ERROR_OBSERVABILITY.items())
        ],
        "conclusion": (
            "Depth supplies scoped metric surface distance; current uncalibrated "
            "Depth/IR silhouettes do not supply reliable ground, contact, body-part "
            "or action-error truth."
        ),
    }
    recapture = _recapture_plan(records)
    future_rgbd = _future_rgbd_value_report(records)
    pairing_guard = {
        "rgb_depth_registration_generated": False,
        "phone_oni_pairs_generated": 0,
        "phone_frame_labels_generated": 0,
        "ir_treated_as_rgb": False,
        "unpaired_distillation_generated": False,
    }
    summary = {
        "schema_version": 1,
        "artifact_type": "round11_implementation_summary",
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": utc_now(),
        "status": "engineering_complete_human_review_and_calibrated_rgbd_pending",
        "record_count": len(records),
        "stream_count": len(records) * 2,
        "sampled_subject_proposal_count": total_proposals,
        "automatic_candidate_count": total_candidates,
        "human_confirmed_subject_count": 0,
        "verified_action_error_count": 0,
        "pairing_guard": pairing_guard,
        "acceptance": {
            "independent_depth_ir_subject_proposals_generated": total_proposals > 0,
            "modality_observability_declared": True,
            "phone_recapture_plan_generated": True,
            "future_synchronized_rgbd_value_report_generated": True,
            "no_forbidden_pairing_or_labels": True,
            "human_identity_gate_passed": False,
            "reliable_error_truth_gate_passed": False,
        },
        "blockers": [
            "independent_human_depth_and_ir_subject_review_pending",
            "current_oni_has_no_color",
            "rgb_depth_calibration_and_capture_sync_unavailable",
            "ground_plane_and_body_part_contact_annotations_unavailable",
            "recording_intent_labels_unverified",
        ],
    }
    payloads = {
        "subject_audit": subject_report,
        "observability": observability_report,
        "phone_recapture": recapture,
        "future_rgbd": future_rgbd,
        "summary": summary,
    }
    forbidden_errors = _validate_no_forbidden_fields(payloads)
    if forbidden_errors:
        raise ValueError("; ".join(forbidden_errors))
    outputs = {
        "subject_audit": write_json(
            report_root / "oni_subject_audit_v1.json",
            subject_report,
        ),
        "observability": write_json(
            report_root / "oni_modality_observability_v1.json",
            observability_report,
        ),
        "phone_recapture": write_json(
            report_root / "oni_phone_recapture_plan_v1.json",
            recapture,
        ),
        "future_rgbd": write_json(
            report_root / "oni_future_rgbd_value_v1.json",
            future_rgbd,
        ),
        "summary": write_json(
            report_root / "round11_implementation_summary.json",
            summary,
        ),
    }
    return outputs


__all__ = [
    "ARTIFACT_VERSION",
    "ERROR_OBSERVABILITY",
    "FORBIDDEN_OUTPUT_FIELDS",
    "OniResearchContract",
    "build_round11_reports",
    "create_contact_sheet",
    "load_oni_research_contract",
    "propose_subject_track",
    "sample_frame_indices",
]
