from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".yaml", ".yml", ".toml"}
SKIP_PARTS = {
    ".git",
    ".cache",
    ".venv",
    "venv",
    "outputs",
    "build",
    "dist",
    ".eggs",
    "__pycache__",
}


def main() -> int:
    errors: list[str] = []
    for path in _source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            errors.append(f"{path.relative_to(ROOT)}: not UTF-8 ({exc})")
            continue
        if text and not text.endswith("\n"):
            errors.append(f"{path.relative_to(ROOT)}: missing final newline")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number}: trailing whitespace"
                )
            if path.suffix.lower() in {".yaml", ".yml"} and "\t" in line:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number}: tab in YAML"
                )
    if errors:
        print("Text format check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Text format check passed.")
    return 0


def _source_files() -> list[Path]:
    """Return files that can actually be included in the next push."""

    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        candidates = ROOT.rglob("*")
    else:
        candidates = (
            ROOT / item.decode("utf-8", errors="surrogateescape")
            for item in completed.stdout.split(b"\0")
            if item
        )
    return sorted(
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in SUFFIXES
        and not any(
            part in SKIP_PARTS for part in path.relative_to(ROOT).parts
        )
    )


if __name__ == "__main__":
    sys.exit(main())
