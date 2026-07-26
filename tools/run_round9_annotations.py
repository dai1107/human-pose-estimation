from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.round9_annotations import run_round9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build truthful Round 9 annotation proposals and human-review gates."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/hyrox"),
        help="HYROX dataset root (default: datasets/hyrox)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_round9(args.dataset_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
