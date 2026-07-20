from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.paths import runtime_output_root
from src.output_schema import versioned_payload


RETENTION_DAYS: dict[str, int] = {
    "web_sessions": 2,
    "web_uploads": 7,
    "screenshots": 30,
    "recordings": 30,
    "logs": 30,
    "web": 30,
    "sessions": 90,
    "comparisons": 90,
    "references": 90,
    "squat_reports": 90,
}


@dataclass(frozen=True)
class CleanupCandidate:
    category: str
    path: str
    modified_unix: float
    size_bytes: int
    reason: str


@dataclass(frozen=True)
class CleanupResult:
    root: str
    applied: bool
    candidate_count: int
    bytes_selected: int
    deleted_count: int
    errors: list[str]
    candidates: list[CleanupCandidate]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidates"] = [asdict(item) for item in self.candidates]
        return versioned_payload("output_cleanup_report", payload)


def _safe_root(root: str | Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    anchor = Path(resolved.anchor).resolve()
    if resolved == anchor:
        raise ValueError("output root cannot be a filesystem root")
    return resolved


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _item_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        descendants = path.rglob("*")
    except OSError:
        return 0
    for descendant in descendants:
        try:
            if descendant.is_file() and not descendant.is_symlink():
                total += descendant.stat().st_size
        except OSError:
            continue
    return total


def _top_level_items(root: Path, categories: set[str]) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for category in sorted(categories):
        category_dir = root / category
        if not _inside_root(category_dir, root) or not category_dir.is_dir():
            continue
        try:
            children = list(category_dir.iterdir())
        except OSError:
            continue
        items.extend((category, child) for child in children)
    return items


def plan_cleanup(
    root: str | Path,
    *,
    categories: Sequence[str] | None = None,
    older_than_days: int | None = None,
    max_total_gb: float | None = None,
    now: float | None = None,
) -> list[CleanupCandidate]:
    output_root = _safe_root(root)
    selected_categories = set(categories or RETENTION_DAYS)
    unknown = sorted(selected_categories - set(RETENTION_DAYS))
    if unknown:
        raise ValueError(f"unknown output categories: {', '.join(unknown)}")
    if older_than_days is not None and older_than_days < 0:
        raise ValueError("--older-than-days must be >= 0")
    if max_total_gb is not None and max_total_gb < 0:
        raise ValueError("--max-total-gb must be >= 0")

    current_time = time.time() if now is None else float(now)
    items: list[tuple[str, Path, float, int]] = []
    for category, path in _top_level_items(output_root, selected_categories):
        try:
            modified = path.lstat().st_mtime
        except OSError:
            continue
        items.append((category, path, modified, _item_size(path)))

    selected: dict[Path, CleanupCandidate] = {}
    for category, path, modified, size in items:
        days = older_than_days if older_than_days is not None else RETENTION_DAYS[category]
        if modified <= current_time - days * 86400:
            selected[path] = CleanupCandidate(
                category,
                str(path),
                modified,
                size,
                f"older_than_{days}_days",
            )

    if max_total_gb is not None:
        quota_bytes = int(max_total_gb * 1024**3)
        remaining_bytes = sum(size for _, path, _, size in items if path not in selected)
        for category, path, modified, size in sorted(items, key=lambda item: item[2]):
            if remaining_bytes <= quota_bytes:
                break
            if path in selected:
                continue
            selected[path] = CleanupCandidate(
                category,
                str(path),
                modified,
                size,
                f"quota_{max_total_gb:g}_gb",
            )
            remaining_bytes -= size
    return sorted(selected.values(), key=lambda item: (item.modified_unix, item.path))


def execute_cleanup(
    root: str | Path,
    candidates: Sequence[CleanupCandidate],
    *,
    apply: bool = False,
) -> CleanupResult:
    output_root = _safe_root(root)
    deleted = 0
    errors: list[str] = []
    if apply:
        for candidate in candidates:
            path = Path(candidate.path)
            if not _inside_root(path, output_root):
                errors.append(f"outside output root: {path}")
                continue
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path)
                deleted += 1
            except OSError as exc:
                errors.append(f"{path}: {exc}")
    return CleanupResult(
        root=str(output_root),
        applied=apply,
        candidate_count=len(candidates),
        bytes_selected=sum(candidate.size_bytes for candidate in candidates),
        deleted_count=deleted,
        errors=errors,
        candidates=list(candidates),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or safely clean generated pose-estimation outputs."
    )
    parser.add_argument("--root", default=str(runtime_output_root()))
    parser.add_argument(
        "--category",
        action="append",
        choices=tuple(RETENTION_DAYS),
        default=[],
        help="Limit cleanup to this generated-output category; repeat as needed.",
    )
    parser.add_argument("--older-than-days", type=int, default=None)
    parser.add_argument("--max-total-gb", type=float, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete selected items. Without this flag the command is preview-only.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidates = plan_cleanup(
            args.root,
            categories=args.category or None,
            older_than_days=args.older_than_days,
            max_total_gb=args.max_total_gb,
        )
        result = execute_cleanup(args.root, candidates, apply=args.apply)
    except ValueError as exc:
        print(f"ERROR: [OUT004] {exc}")
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        action = "deleted" if args.apply else "would delete"
        print(
            f"{action}: {result.candidate_count} item(s), "
            f"{result.bytes_selected / 1024 / 1024:.1f} MiB under {result.root}"
        )
        for candidate in result.candidates:
            print(f"- {candidate.path} ({candidate.reason})")
        for error in result.errors:
            print(f"ERROR: {error}")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
