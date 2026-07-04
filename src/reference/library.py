from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from src.utils.time_utils import make_session_id, now_iso

from .aggregate import build_reference_template, write_template_csv
from .clipper import clip_session, write_clip
from .features import extract_feature_matrix, load_feature_config
from .quality import evaluate_quality, load_quality_rules
from .schema import ReferenceAction
from .session_loader import read_csv_rows, write_csv_rows


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_").lower()
    return slug[:40] or "reference"


def make_reference_id(name: str) -> str:
    return f"{make_session_id()}_{_slug(name)}"


def _unique_reference_dir(root: Path, reference_id: str) -> Path:
    candidate = root / reference_id
    if not candidate.exists():
        return candidate
    suffix = 1
    while (root / f"{reference_id}_{suffix}").exists():
        suffix += 1
    return root / f"{reference_id}_{suffix}"


def save_reference_action(reference_dir: Path, action: ReferenceAction) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "reference.json").write_text(
        json.dumps(action.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_reference(reference_dir: str | Path) -> ReferenceAction:
    path = Path(reference_dir)
    payload = json.loads((path / "reference.json").read_text(encoding="utf-8"))
    return ReferenceAction.from_dict(payload)


def list_references(root: str | Path = "outputs/references") -> list[ReferenceAction]:
    path = Path(root)
    if not path.exists():
        return []
    references: list[ReferenceAction] = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "reference.json").exists():
            references.append(load_reference(child))
    return references


def create_reference_from_session(
    session_dir: str | Path,
    output_root: str | Path = "outputs/references",
    start_ms: int | None = None,
    end_ms: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    name: str = "reference action",
    description: str = "",
    action_type: str = "generic_motion",
    camera_view: str = "unknown",
    movement_side: str = "unknown",
    tags: list[str] | None = None,
    notes: str = "",
    mirror_canonicalization_enabled: bool = False,
    reference_id: str | None = None,
) -> Path:
    clip = clip_session(session_dir, start_ms=start_ms, end_ms=end_ms, start_frame=start_frame, end_frame=end_frame)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    base_reference_id = reference_id or make_reference_id(name)
    reference_dir = _unique_reference_dir(root, base_reference_id)
    reference_dir.mkdir(parents=True, exist_ok=False)

    rules = load_quality_rules()
    quality = evaluate_quality(
        clip.kinematics,
        clip.landmarks,
        metadata=clip.session.metadata,
        rules=rules,
        camera_view=camera_view,
    )
    feature_config = load_feature_config()
    extracted = extract_feature_matrix(clip.kinematics, feature_config)

    action = ReferenceAction(
        reference_id=reference_dir.name,
        name=name,
        description=description,
        action_type=action_type,
        camera_view=camera_view,
        movement_side=movement_side,
        created_at=now_iso(),
        source_session_ids=[clip.clip_range.session_id],
        source_clip_ranges=[clip.clip_range.to_dict()],
        feature_set_name=feature_config.name,
        normalization_method="body_relative",
        mirror_canonicalization_enabled=mirror_canonicalization_enabled,
        quality_summary=quality.to_dict(),
        tags=tags or [],
        notes=notes,
        clip_count=1,
    )
    save_reference_action(reference_dir, action)
    write_clip(reference_dir, clip)
    (reference_dir / "source_metadata.json").write_text(
        json.dumps(clip.session.metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (reference_dir / "feature_processing.json").write_text(
        json.dumps(extracted.processing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    feature_rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(extracted.timestamps):
        row: dict[str, Any] = {"timestamp_ms": f"{timestamp:.8g}"}
        for feature_index, feature_name in enumerate(extracted.feature_names):
            row[feature_name] = f"{extracted.matrix[index, feature_index]:.8g}"
        feature_rows.append(row)
    write_csv_rows(reference_dir / "features.csv", feature_rows)
    return reference_dir


def create_reference_from_reference_clips(
    clip_dirs: list[str | Path],
    output_root: str | Path = "outputs/references",
    name: str = "reference template",
    action_type: str = "generic_motion",
    camera_view: str = "unknown",
    movement_side: str = "unknown",
    reference_id: str | None = None,
) -> Path:
    if not clip_dirs:
        raise ValueError("at least one clip directory is required")
    feature_config = load_feature_config()
    clips = [read_csv_rows(Path(path) / "clip_kinematics.csv") for path in clip_dirs]
    template = build_reference_template(clips, feature_config=feature_config)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    reference_dir = _unique_reference_dir(root, reference_id or make_reference_id(name))
    reference_dir.mkdir(parents=True, exist_ok=False)
    write_template_csv(reference_dir, template)
    source_ids: list[str] = []
    source_ranges: list[dict[str, Any]] = []
    for path in clip_dirs:
        ref_path = Path(path)
        if (ref_path / "reference.json").exists():
            reference = load_reference(ref_path)
            source_ids.extend(reference.source_session_ids)
            source_ranges.extend(reference.source_clip_ranges)
    action = ReferenceAction(
        reference_id=reference_dir.name,
        name=name,
        action_type=action_type,
        camera_view=camera_view,
        movement_side=movement_side,
        source_session_ids=source_ids,
        source_clip_ranges=source_ranges,
        feature_set_name=feature_config.name,
        normalization_method="body_relative",
        quality_summary={"template": template["template_stability_status"], "clip_count": template["clip_count"]},
        clip_count=int(template["clip_count"]),
    )
    save_reference_action(reference_dir, action)
    (reference_dir / "template_summary.json").write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    return reference_dir


def export_reference(reference_dir: str | Path, output_path: str | Path | None = None) -> Path:
    path = Path(reference_dir)
    if not path.exists():
        raise FileNotFoundError(f"reference directory not found: {path}")
    target = Path(output_path) if output_path is not None else path.with_suffix(".zip")
    if target.suffix.lower() != ".zip":
        target = target.with_suffix(".zip")
    if target.exists():
        target.unlink()
    shutil.make_archive(str(target.with_suffix("")), "zip", path)
    return target

