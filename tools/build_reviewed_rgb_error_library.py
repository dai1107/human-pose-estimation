"""Build a traceable TP/FP/FN/UNSURE clip library from reviewed RGB evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _case_category(match: dict[str, Any]) -> str:
    candidate = match.get("candidate")
    human = match.get("human_rep")
    if candidate is None:
        return "FN"
    if human is None:
        return "FP"
    if str(candidate.get("status", "")) == "UNSURE":
        return "UNSURE"
    if bool(match.get("status_match")):
        return "TP"
    return "STATUS_MISMATCH"


def _stable_case_id(record_id: str, index: int, category: str) -> str:
    suffix = hashlib.sha256(
        f"{record_id}|{index}|{category}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{record_id}_{category.lower()}_{index:03d}_{suffix}"


def _export_clip(
    source: Path,
    destination: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolved_fps = fps or float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        resolved_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create clip: {destination}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _frame_index in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    capture.release()
    if not destination.is_file() or written == 0:
        raise RuntimeError(f"clip export produced no frames: {destination}")
    return {
        "written_frame_count": written,
        "codec": "mp4v",
        "audio_included": False,
        "sha256": _sha256(destination),
    }


def build_error_library(
    dataset_root: str | Path,
    *,
    export_media: bool = False,
) -> Path:
    root = Path(dataset_root)
    project_root = root.parents[1]
    report_path = (
        root / "reports" / "reviewed_phone_rgb_guidance_evaluation_optimized_v1.json"
    )
    fine_path = root / "reviews" / "human_rgb_fine_annotations_v1.json"
    roles_path = root / "manifests" / "data_roles_v1.json"
    manifest_path = root / "manifests" / "phone_records.json"
    report = _load(report_path)
    fine = _load(fine_path)
    roles = _load(roles_path)
    manifest = _load(manifest_path)
    fine_by_id = {
        str(item.get("record_id", "")): item
        for item in fine.get("records", [])
        if isinstance(item, dict)
    }
    roles_by_id = {
        str(item.get("record_id", "")): item
        for item in roles.get("assignments", [])
        if isinstance(item, dict)
    }
    manifest_by_id = {
        str(item.get("record_id", "")): item
        for item in manifest.get("records", [])
        if isinstance(item, dict)
    }
    output_root = root / "reviews" / "error_library_v1"
    clips_root = output_root / "clips"
    cases: list[dict[str, Any]] = []
    for record in report.get("records", []):
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("record_id", ""))
        manifest_record = manifest_by_id[record_id]
        fine_record = fine_by_id[record_id]
        role_record = roles_by_id.get(record_id, {})
        video = manifest_record.get("video") or {}
        fps = float(video.get("fps", 0.0) or 0.0)
        maximum_frame = int(video.get("decoded_frame_count", 0) or 0) - 1
        source_relative = str(manifest_record.get("source_file", ""))
        source_path = root / source_relative
        for index, match in enumerate(record.get("matches", []), start=1):
            if not isinstance(match, dict):
                continue
            category = _case_category(match)
            candidate = match.get("candidate")
            human = match.get("human_rep")
            if isinstance(human, dict):
                anchor = int(human.get("end_frame", human.get("start_frame", 0)))
            elif isinstance(candidate, dict):
                anchor = int(
                    candidate.get(
                        "alignment_frame",
                        candidate.get("source_frame", 0),
                    )
                )
            else:
                continue
            desired_frame_count = max(1, round(fps))
            start_frame = max(0, anchor - desired_frame_count // 2)
            end_frame = min(
                maximum_frame,
                start_frame + desired_frame_count - 1,
            )
            start_frame = max(0, end_frame - desired_frame_count + 1)
            case_id = _stable_case_id(record_id, index, category)
            clip_relative = Path("clips") / record_id / f"{case_id}.mp4"
            clip_path = output_root / clip_relative
            media: dict[str, Any] = {
                "status": "not_exported",
                "path": str(clip_relative),
            }
            if export_media:
                media.update(
                    {
                        "status": "exported",
                        **_export_clip(
                            source_path,
                            clip_path,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            fps=fps,
                        ),
                    }
                )
            cases.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "record_id": record_id,
                    "action": record.get("action"),
                    "camera_view": record.get("camera_view"),
                    "subject_group": record.get("subject_group"),
                    "dataset_role": record.get("dataset_role"),
                    "training_eligible": bool(
                        role_record.get("training_eligible")
                    ),
                    "evaluation_eligible": bool(
                        role_record.get("evaluation_eligible")
                    ),
                    "anchor_frame": anchor,
                    "clip_start_frame": start_frame,
                    "clip_end_frame": end_frame,
                    "clip_duration_seconds": (
                        (end_frame - start_frame + 1) / fps if fps else None
                    ),
                    "human_rep": human,
                    "runtime_candidate": candidate,
                    "terminal_frame_error": match.get(
                        "terminal_frame_error"
                    ),
                    "status_match": bool(match.get("status_match")),
                    "human_record_errors": record.get("expected_errors", []),
                    "runtime_record_errors": record.get(
                        "predicted_error_classes", []
                    ),
                    "source_video": source_relative,
                    "source_video_sha256": manifest_record.get("sha256"),
                    "fine_review_record_sha256": fine_record.get(
                        "review_record_sha256"
                    ),
                    "media": media,
                }
            )

    category_counts = dict(sorted(Counter(case["category"] for case in cases).items()))
    implementation_paths = [
        project_root / "tools" / "evaluate_reviewed_rgb_guidance.py",
        project_root / "hyrox" / "actions" / "burpee_broad_jump.py",
        project_root / "hyrox" / "actions" / "lunge.py",
        project_root / "hyrox" / "actions" / "wall_ball.py",
        project_root / "hyrox" / "contact.py",
        project_root / "hyrox" / "foot_events.py",
        project_root / "hyrox" / "validity.py",
        project_root / "configs" / "hyrox" / "burpee_broad_jump.yaml",
        project_root / "configs" / "hyrox" / "lunge.yaml",
        project_root / "configs" / "hyrox" / "wall_ball.yaml",
        project_root / "configs" / "hyrox" / "contact.yaml",
        project_root / "configs" / "hyrox" / "foot_events.yaml",
        project_root / "configs" / "hyrox" / "observability.yaml",
    ]
    implementation_artifacts = [
        {
            "path": str(path.relative_to(project_root)),
            "sha256": _sha256(path),
        }
        for path in implementation_paths
        if path.is_file()
    ]
    snapshot_digest = hashlib.sha256(
        "\n".join(
            f"{item['path']}:{item['sha256']}"
            for item in implementation_artifacts
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "artifact_type": "reviewed_rgb_error_library_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_profile": "optimized",
        "source_evaluation": str(report_path.relative_to(root)),
        "source_evaluation_sha256": _sha256(report_path),
        "source_fine_annotations": str(fine_path.relative_to(root)),
        "source_fine_annotations_sha256": _sha256(fine_path),
        "source_data_roles": str(roles_path.relative_to(root)),
        "source_data_roles_sha256": _sha256(roles_path),
        "source_manifest": str(manifest_path.relative_to(root)),
        "source_manifest_sha256": _sha256(manifest_path),
        "case_count": len(cases),
        "category_counts": category_counts,
        "implementation_snapshot_sha256": snapshot_digest,
        "implementation_artifacts": implementation_artifacts,
        "media_exported": export_media,
        "clip_policy": {
            "window": "anchor +/- 0.5 seconds",
            "maximum_nominal_duration_seconds": 1.0,
            "audio_included": False,
            "source_video_unchanged": True,
        },
        "scope": (
            "Internal reviewed phone-RGB regression cases only; temporary "
            "subject groups are not real identity claims."
        ),
        "cases": cases,
    }
    return _write(output_root / "index.json", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a traceable reviewed-RGB TP/FP/FN/UNSURE short-clip library."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/hyrox"),
    )
    parser.add_argument(
        "--export-media",
        action="store_true",
        help="Decode source videos and write the 0.5-1.0 second MP4 clips.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    output = build_error_library(
        dataset_root,
        export_media=args.export_media,
    )
    print(
        json.dumps(
            {"error_library": str(output)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
