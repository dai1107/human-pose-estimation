from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset.round9_review import (
    apply_ai_review_decisions,
    write_multimethod_review,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-review Round 9 proposals.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/hyrox"))
    parser.add_argument("--no-sheets", action="store_true")
    parser.add_argument(
        "--apply-decisions",
        type=Path,
        help="Apply a declared AI review-decision file after rebuilding the cross-check.",
    )
    args = parser.parse_args()
    report = write_multimethod_review(
        args.project_root.resolve(),
        (args.project_root / args.dataset_root).resolve()
        if not args.dataset_root.is_absolute()
        else args.dataset_root,
        write_sheets=not args.no_sheets,
    )
    result: dict[str, object] = {
        key: report[key] for key in ("status", "record_count", "core_record_count", "metrics")
    }
    if args.apply_decisions:
        decision_path = (
            args.apply_decisions
            if args.apply_decisions.is_absolute()
            else args.project_root / args.apply_decisions
        )
        result["implementation_summary"] = apply_ai_review_decisions(
            args.project_root.resolve(),
            (args.project_root / args.dataset_root).resolve()
            if not args.dataset_root.is_absolute()
            else args.dataset_root,
            decision_path.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
