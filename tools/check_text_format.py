from __future__ import annotations

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
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
            continue
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


if __name__ == "__main__":
    sys.exit(main())
