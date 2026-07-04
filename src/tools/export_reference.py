from __future__ import annotations

import argparse
from pathlib import Path

from src.reference.library import export_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a personal reference action as a zip archive.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = export_reference(Path(args.reference), Path(args.output) if args.output else None)
    print(f"Reference exported: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

