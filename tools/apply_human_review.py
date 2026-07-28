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
            "Validate and apply reviewer A quick-review and fine RGB results "
            "with explicit, auditable promotion gates."
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
    parser.add_argument(
        "--accept-fine-rgb-calibration",
        action="store_true",
        help=(
            "Accept validated single-reviewer rep/phase/error/event labels as "
            "internal phone-RGB rule-calibration truth; this does not enable "
            "supervised model training."
        ),
    )
    parser.add_argument(
        "--force-phone-observable",
        action="store_true",
        help=(
            "Apply the user's explicit all-observable decision to phone-RGB "
            "record, event and phase/error observability with an audit trail."
        ),
    )
    parser.add_argument(
        "--confirm-fine-rgb-human-reviewed",
        action="store_true",
        help=(
            "Record the user's confirmation that all exported fine RGB results and "
            "retained AI proposals were manually reviewed; this enables the "
            "validated fine labels for supervised internal experiments."
        ),
    )
    parser.add_argument(
        "--derive-missing-noncore-events",
        action="store_true",
        help=(
            "Derive canonical non-core events from saved, human-reviewed phase "
            "boundaries when event arrays are empty; record the derivation."
        ),
    )
    parser.add_argument(
        "--confirm-valid-proposal-record",
        action="append",
        default=[],
        metavar="RECORD_ID",
        help=(
            "For an explicitly named record with empty fine arrays, apply every "
            "current non-core AI candidate as VALID under the user's confirmation."
        ),
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
        accept_fine_annotations_for_internal_rgb_calibration=(
            args.accept_fine_rgb_calibration
        ),
        force_phone_observable=args.force_phone_observable,
        confirm_fine_rgb_human_reviewed=(
            args.confirm_fine_rgb_human_reviewed
        ),
        derive_missing_noncore_events=args.derive_missing_noncore_events,
        confirmed_valid_proposal_record_ids=(
            args.confirm_valid_proposal_record
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
