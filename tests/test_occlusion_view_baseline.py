from __future__ import annotations

import json

from src.validation.occlusion_view import (
    build_hard_case_inventory,
    write_failure_inventory,
)


def test_hard_case_inventory_groups_count_failures_and_preserves_unknown_occlusion(tmp_path):
    error_path = tmp_path / "errors.json"
    manifest_path = tmp_path / "manifest.json"
    review_root = tmp_path / "reviews"
    review_root.mkdir()
    error_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case_fn",
                        "record_id": "record_1",
                        "category": "FN",
                        "action": "lunge",
                        "camera_view": "oblique",
                        "media": {"path": "fn.mp4"},
                    },
                    {
                        "case_id": "case_fp",
                        "record_id": "record_2",
                        "category": "FP",
                        "action": "wall_ball",
                        "camera_view": "front",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {"record_id": "record_1", "source_file": "one.mp4"},
                    {"record_id": "record_2", "source_file": "two.mp4"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (review_root / "record_1.json").write_text(
        json.dumps(
            {
                "record_id": "record_1",
                "review": {
                    "quick_review": {
                        "observability": "OBSERVABLE",
                        "events": [{"observability": "PARTIAL"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rows = build_hard_case_inventory(error_path, manifest_path, review_root)

    assert rows[0]["failure_class"] == "missed_count"
    assert rows[0]["occlusion_level"] == "partial_event"
    assert rows[1]["failure_class"] == "over_count"
    assert rows[1]["occlusion_level"] == "unlabelled"

    report = write_failure_inventory(tmp_path / "output", rows)
    assert report["counts"]["by_failure_class"] == {
        "missed_count": 1,
        "over_count": 1,
    }
    assert (tmp_path / "output" / "hard_cases.csv").is_file()
    assert (tmp_path / "output" / "failure_taxonomy.md").is_file()
