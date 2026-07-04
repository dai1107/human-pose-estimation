from __future__ import annotations

import argparse

from src.reference.library import list_references


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List personal reference actions.")
    parser.add_argument("--root", default="outputs/references")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    references = list_references(args.root)
    if not references:
        print("No references found.")
        return 0
    for reference in references:
        print(
            f"{reference.reference_id}\t{reference.name}\t"
            f"{reference.action_type}\t{reference.camera_view}\t{reference.movement_side}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

