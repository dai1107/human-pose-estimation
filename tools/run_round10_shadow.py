"""Build Round 10 contracts, shadow-ablation readiness and failure reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.dataset.round10_shadow import build_round10_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build truthful Round 10 shadow-ablation and failure-pool reports."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/hyrox"))
    parser.add_argument("--contract-dir", type=Path, default=Path("configs/contracts"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    contract_dir = args.contract_dir
    if not contract_dir.is_absolute():
        contract_dir = project_root / contract_dir
    outputs = build_round10_reports(dataset_root, contract_dir=contract_dir)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

