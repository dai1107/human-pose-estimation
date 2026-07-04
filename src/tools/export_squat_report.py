from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a squat report directory as a zip archive.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = Path(args.report)
    if not report_dir.exists():
        raise FileNotFoundError(f"squat report directory not found: {report_dir}")
    output = Path(args.output) if args.output else report_dir.with_suffix(".zip")
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    if output.exists():
        output.unlink()
    shutil.make_archive(str(output.with_suffix("")), "zip", report_dir)
    print(f"Squat report exported: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
