"""Apply a validated single-reviewer quick-review bundle and authorization scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.dataset.human_review_application import apply_quick_review_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and apply reviewer A quick-review results without "
            "promoting them to double-reviewed ground truth."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/hyrox"))
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument(
        "--phone-only-authorization",
        action="store_true",
        help="Do not apply the user's all-use authorization confirmation to ONI records.",
    )
    parser.add_argument(
        "--keep-subject-identity-gate",
        action="store_true",
        help="Do not record the temporary record_id-grouped internal-experiment waiver.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    review_bundle = args.review_bundle
    if not review_bundle.is_absolute():
        review_bundle = project_root / review_bundle
    outputs = apply_quick_review_bundle(
        dataset_root,
        review_bundle,
        authorize_oni=not args.phone_only_authorization,
        waive_subject_identity_for_internal_record_grouping=(
            not args.keep_subject_identity_gate
        ),
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
