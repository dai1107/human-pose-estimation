"""Build Round 11 independent ONI Depth/IR research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.dataset.round11_oni_research import build_round11_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build independent ONI Depth/IR target proposals, observability, "
            "phone recapture and future synchronized RGB-D value reports."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/hyrox"))
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/contracts/oni_research_v1.yaml"),
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip derived subject-proposal contact sheets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    contract = args.contract
    if not contract.is_absolute():
        contract = project_root / contract
    outputs = build_round11_reports(
        dataset_root,
        contract_path=contract,
        create_previews=not args.no_previews,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

