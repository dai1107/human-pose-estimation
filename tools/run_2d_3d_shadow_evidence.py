"""Run leakage-reduced 2D versus constrained 2D+3D shadow evidence.

Each held-out video is evaluated with a threshold profile selected only from
the other reviewed videos.  This is still an internal experiment: the small
single-reviewer corpus is not an independent product test.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.biomechanics.shadow_evidence_3d import ShadowEvidence3DConfig
from tools.evaluate_reviewed_rgb_guidance import (
    PROJECT_ROOT,
    TARGET_ACTIONS,
    _evaluate_record,
    _group_metrics,
    _load,
    _write,
)


CONFIG_CANDIDATES: tuple[tuple[str, ShadowEvidence3DConfig], ...] = (
    (
        "quality_gated_2d_fallback",
        ShadowEvidence3DConfig(
            angle_assist_enabled=False,
            body_assist_enabled=False,
        ),
    ),
    (
        "body_temporal_only",
        ShadowEvidence3DConfig(
            angle_assist_enabled=False,
            body_assist_enabled=True,
            depth_order_gap_body_ratio=0.16,
            stationary_speed_body_per_s=0.14,
            stationary_dwell_frames=4,
            synchronous_event_ms=100.0,
            conflict_event_ms=260.0,
            prone_horizontal_score_min=0.74,
            confidence_boost=0.03,
            conflict_confidence_cap=0.45,
        ),
    ),
    (
        "angle_temporal_only",
        ShadowEvidence3DConfig(
            angle_assist_enabled=True,
            body_assist_enabled=False,
            max_2d_3d_difference_deg=60.0,
            angle_conflict_min_frames=5,
            angle_conflict_min_ratio=0.50,
            angle_support_min_frames=5,
            angle_support_min_ratio=0.60,
            confidence_boost=0.03,
            conflict_confidence_cap=0.45,
        ),
    ),
    (
        "combined_temporal_conservative",
        ShadowEvidence3DConfig(
            angle_assist_enabled=True,
            body_assist_enabled=True,
            depth_order_gap_body_ratio=0.16,
            stationary_speed_body_per_s=0.14,
            stationary_dwell_frames=4,
            synchronous_event_ms=100.0,
            conflict_event_ms=260.0,
            prone_horizontal_score_min=0.74,
            max_2d_3d_difference_deg=60.0,
            angle_conflict_min_frames=5,
            angle_conflict_min_ratio=0.50,
            angle_support_min_frames=5,
            angle_support_min_ratio=0.60,
            confidence_boost=0.03,
            conflict_confidence_cap=0.45,
        ),
    ),
)


def leave_one_video_out_folds(record_ids: list[str]) -> list[dict[str, Any]]:
    ordered = sorted(dict.fromkeys(record_ids))
    return [
        {
            "fold_index": index,
            "held_out_video_id": held_out,
            "training_video_ids": [
                record_id for record_id in ordered if record_id != held_out
            ],
        }
        for index, held_out in enumerate(ordered, start=1)
    ]


def _calibration_score(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    accuracy = float(metrics.get("matched_rep_status_accuracy") or 0.0)
    unsure = float(metrics.get("unsure_rate") or 0.0)
    confusion = metrics.get("error_confusion_by_code")
    error_count = 0
    if isinstance(confusion, Mapping):
        error_count = sum(
            int(row.get("false_positive_records", 0) or 0)
            + int(row.get("false_negative_records", 0) or 0)
            for row in confusion.values()
            if isinstance(row, Mapping)
        )
    records = max(1, int(metrics.get("record_count", 0) or 0))
    # Accuracy is primary; abstention and per-rule errors break ties.  The
    # final component prefers the conservative candidate on an exact tie.
    return (
        accuracy,
        -unsure,
        -(error_count / records),
    )


def _candidate_refs(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for match in record.get("matches") or []
        if isinstance(match, Mapping)
        and isinstance((candidate := match.get("candidate")), dict)
    ]


def _refresh_record_status_fields(record: dict[str, Any]) -> None:
    candidates = _candidate_refs(record)
    matches = [
        row
        for row in record.get("matches") or []
        if isinstance(row, dict)
    ]
    for row in matches:
        candidate = row.get("candidate")
        human = row.get("human_rep")
        row["status_match"] = bool(
            isinstance(candidate, Mapping)
            and isinstance(human, Mapping)
            and str(candidate.get("status")) == str(human.get("validity"))
        )
    counts = Counter(str(item.get("status")) for item in candidates)
    record["predicted_status_counts"] = dict(sorted(counts.items()))
    record["matched_status_count"] = sum(
        bool(row.get("status_match"))
        for row in matches
        if row.get("candidate") is not None and row.get("human_rep") is not None
    )
    record["exact_count_and_status_match"] = bool(
        record.get("exact_count_match")
        and all(
            bool(row.get("status_match"))
            for row in matches
            if row.get("candidate") is not None
            and row.get("human_rep") is not None
        )
    )


def enforce_no_three_d_valid_promotion(
    baseline: Mapping[str, Any],
    assisted: Mapping[str, Any],
) -> dict[str, Any]:
    """Block any VALID promotion while preserving confidence/UNSURE changes."""

    result = copy.deepcopy(dict(assisted))
    baseline_candidates = _candidate_refs(baseline)
    assisted_candidates = _candidate_refs(result)
    invariant = len(baseline_candidates) == len(assisted_candidates)
    blocked = []
    if invariant:
        for index, (before, after) in enumerate(
            zip(baseline_candidates, assisted_candidates),
            start=1,
        ):
            if (
                str(after.get("status")) == "VALID"
                and str(before.get("status")) != "VALID"
            ):
                blocked.append(
                    {
                        "candidate_index": index,
                        "baseline_status": before.get("status"),
                        "attempted_status": "VALID",
                        "final_status": before.get("status"),
                    }
                )
                after["status"] = before.get("status")
                after["reason_codes"] = list(
                    dict.fromkeys(
                        (
                            "THREE_D_VALID_PROMOTION_BLOCKED",
                            *(after.get("reason_codes") or []),
                        )
                    )
                )
    result["shadow_safety_guard"] = {
        "candidate_count_invariant": invariant,
        "baseline_candidate_count": len(baseline_candidates),
        "assisted_candidate_count": len(assisted_candidates),
        "blocked_valid_promotion_count": len(blocked),
        "blocked_valid_promotions": blocked,
        "three_d_contact_inference_allowed": False,
        "three_d_validity_promotion_allowed": False,
        "floor_reference_source": "unchanged_2d_local_floor",
    }
    _refresh_record_status_fields(result)
    return result


def _load_review_inputs(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    manifest_payload = _load(root / "manifests" / "phone_records.json")
    fine_payload = _load(
        root / "reviews" / "human_rgb_fine_annotations_v1.json"
    )
    manifest = {
        str(item.get("record_id")): dict(item)
        for item in manifest_payload.get("records") or []
        if isinstance(item, dict)
    }
    reviews = [
        dict(item)
        for item in fine_payload.get("records") or []
        if isinstance(item, dict)
        and item.get("action") in TARGET_ACTIONS
        and bool(item.get("internal_rgb_rule_calibration_eligible"))
    ]
    return manifest, reviews


def _metric_comparison(
    baseline: Mapping[str, Any],
    assisted: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "candidate_recall",
        "candidate_precision",
        "matched_rep_status_accuracy",
        "unsure_rate",
        "definite_decision_coverage",
        "count_mae",
        "exact_count_rate",
    )
    result = {}
    for key in keys:
        before = baseline.get(key)
        after = assisted.get(key)
        result[key] = {
            "two_d": before,
            "two_d_plus_three_d": after,
            "delta": (
                None
                if before is None or after is None
                else float(after) - float(before)
            ),
        }
    result["terminal_event_metrics"] = {
        "two_d": baseline.get("terminal_event_metrics"),
        "two_d_plus_three_d": assisted.get("terminal_event_metrics"),
    }
    result["error_confusion_by_code"] = {
        code: {
            "two_d": row,
            "two_d_plus_three_d": (
                assisted.get("error_confusion_by_code", {}).get(code)
            ),
            "false_positive_delta": (
                int(
                    assisted.get("error_confusion_by_code", {})
                    .get(code, {})
                    .get("false_positive_records", 0)
                    or 0
                )
                - int(row.get("false_positive_records", 0) or 0)
            ),
            "false_negative_delta": (
                int(
                    assisted.get("error_confusion_by_code", {})
                    .get(code, {})
                    .get("false_negative_records", 0)
                    or 0
                )
                - int(row.get("false_negative_records", 0) or 0)
            ),
        }
        for code, row in baseline.get("error_confusion_by_code", {}).items()
    }
    return result


def _markdown_report(payload: Mapping[str, Any]) -> str:
    comparison = payload["comparison"]
    selected = Counter(
        str(fold["selected_config_name"])
        for fold in payload["leakage_audit"]["folds"]
    )
    lines = [
        "# 2D + 3D 影子证据内部实验",
        "",
        "本报告使用逐视频留一验证。3D 只参与置信度、腿侧/主侧选择与冲突降级；"
        "二维局部地板和触地证据保持不变，3D 不得单独把结果改成 VALID 或已触地。",
        "",
        "| 指标 | 2D | 2D+3D | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "candidate_recall",
        "matched_rep_status_accuracy",
        "unsure_rate",
        "definite_decision_coverage",
    ):
        row = comparison[key]
        lines.append(
            f"| {key} | {_fmt(row['two_d'])} | "
            f"{_fmt(row['two_d_plus_three_d'])} | {_fmt(row['delta'])} |"
        )
    event_before = comparison["terminal_event_metrics"]["two_d"]
    event_after = comparison["terminal_event_metrics"]["two_d_plus_three_d"]
    lines.extend(
        [
            "",
            "## 事件帧误差",
            "",
            "| 指标 | 2D | 2D+3D |",
            "|---|---:|---:|",
            f"| mean_absolute_error_frames | {_fmt(event_before.get('mean_absolute_error_frames'))} | {_fmt(event_after.get('mean_absolute_error_frames'))} |",
            f"| median_absolute_error_frames | {_fmt(event_before.get('median_absolute_error_frames'))} | {_fmt(event_after.get('median_absolute_error_frames'))} |",
            f"| within_5_frames_rate | {_fmt(event_before.get('within_5_frames_rate'))} | {_fmt(event_after.get('within_5_frames_rate'))} |",
            "",
            "## 消融与逐折选择",
            "",
            "| 候选策略 | 状态准确率 | UNSURE rate | HIP_NOT_EXTENDED FP |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, metrics in payload[
        "ablation_metrics_pooled_internal_diagnostic_only"
    ].items():
        hip = metrics["error_confusion_by_code"]["HIP_NOT_EXTENDED"]
        lines.append(
            f"| {name} | {_fmt(metrics['matched_rep_status_accuracy'])} | "
            f"{_fmt(metrics['unsure_rate'])} | "
            f"{hip['false_positive_records']} |"
        )
    lines.extend(
        [
            "",
            "逐视频留一选中次数："
            + "，".join(
                f"`{name}` {count}/15"
                for name, count in sorted(selected.items())
            )
            + "。当所有 3D 消融候选都未在其余视频上稳定胜过 2D 时，"
            "质量门选择精确回退 2D。",
            "",
            "## 各错误规则 FP / FN",
            "",
            "| 错误规则 | 2D FP | 2D FN | 2D+3D FP | 2D+3D FN |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for code, row in comparison["error_confusion_by_code"].items():
        before = row["two_d"]
        after = row["two_d_plus_three_d"]
        lines.append(
            f"| {code} | {before['false_positive_records']} | "
            f"{before['false_negative_records']} | "
            f"{after['false_positive_records']} | "
            f"{after['false_negative_records']} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 这是小样本、单人工复核数据上的内部实验，不是独立测试或生产泛化结论。",
            "- 每折阈值只由其余视频选择；报告保留完整训练/测试视频 ID 供泄漏审计。",
            "- candidate recall 和事件帧若不变，是受限融合不创建候选、不改二维事件锚点的预期结果。",
            "- MediaPipe world landmarks 是身体相对坐标，不是相机坐标或真实米制深度。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run_experiment(dataset_root: str | Path) -> tuple[Path, Path]:
    root = Path(dataset_root)
    manifest, reviews = _load_review_inputs(root)
    record_ids = [str(review["record_id"]) for review in reviews]
    baseline: dict[str, dict[str, Any]] = {}
    assisted_cache: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name, _config in CONFIG_CANDIDATES
    }
    for review in reviews:
        record_id = str(review["record_id"])
        manifest_record = manifest[record_id]
        baseline[record_id] = _evaluate_record(
            root,
            manifest_record,
            review,
            profile="optimized",
        )
        for name, config in CONFIG_CANDIDATES:
            assisted = _evaluate_record(
                root,
                manifest_record,
                review,
                profile="optimized",
                shadow_evidence_config=config,
            )
            assisted_cache[name][record_id] = enforce_no_three_d_valid_promotion(
                baseline[record_id],
                assisted,
            )

    folds = leave_one_video_out_folds(record_ids)
    selected_records: list[dict[str, Any]] = []
    fold_reports = []
    for fold in folds:
        training_ids = fold["training_video_ids"]
        candidate_rows = []
        for preference, (name, config) in enumerate(CONFIG_CANDIDATES):
            metrics = _group_metrics(
                [assisted_cache[name][record_id] for record_id in training_ids]
            )
            candidate_rows.append(
                {
                    "config_name": name,
                    "config": config.as_dict(),
                    "training_metrics": metrics,
                    "selection_score": list(_calibration_score(metrics)),
                    "preference": -preference,
                }
            )
        selected = max(
            candidate_rows,
            key=lambda row: (
                *row["selection_score"],
                row["preference"],
            ),
        )
        held_out = str(fold["held_out_video_id"])
        held_out_record = copy.deepcopy(
            assisted_cache[str(selected["config_name"])][held_out]
        )
        held_out_record["lovo_fold"] = fold["fold_index"]
        held_out_record["selected_config_name"] = selected["config_name"]
        selected_records.append(held_out_record)
        fold_reports.append(
            {
                **fold,
                "selected_config_name": selected["config_name"],
                "selected_config": selected["config"],
                "calibration_candidates": candidate_rows,
                "training_test_overlap": sorted(
                    set(training_ids).intersection({held_out})
                ),
                "held_out_metrics": _group_metrics([held_out_record]),
            }
        )

    baseline_records = [baseline[record_id] for record_id in sorted(record_ids)]
    selected_records.sort(key=lambda item: str(item["record_id"]))
    baseline_metrics = _group_metrics(baseline_records)
    assisted_metrics = _group_metrics(selected_records)
    invariant_failures = [
        record["record_id"]
        for record in selected_records
        if not record["shadow_safety_guard"]["candidate_count_invariant"]
    ]
    promotion_blocks = sum(
        int(record["shadow_safety_guard"]["blocked_valid_promotion_count"])
        for record in selected_records
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "internal_2d_plus_3d_shadow_evidence_lovo_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": (
            "internal_single_reviewer_leave_one_video_out_experiment"
        ),
        "source_type": "reviewed_phone_rgb",
        "oni_used": False,
        "record_count": len(record_ids),
        "video_level_fold_count": len(folds),
        "feature_families": [
            "hip_compensated_knee_and_foot_height_change",
            "front_back_leg_depth_order",
            "three_d_knee_and_hip_angles",
            "foot_speed_stationary_dwell_and_left_right_timing",
            "torso_prone_and_shoulder_hip_spatial_relation",
        ],
        "safety_contract": {
            "two_d_floor_reference_retained": True,
            "three_d_may_adjust_confidence": True,
            "three_d_may_adjust_leg_or_primary_side": True,
            "three_d_conflict_may_downgrade_to_unsure": True,
            "three_d_may_infer_contact_alone": False,
            "three_d_may_promote_to_valid": False,
            "blocked_valid_promotion_count": promotion_blocks,
            "candidate_count_invariant_failures": invariant_failures,
        },
        "leakage_audit": {
            "split_unit": "video_record_id",
            "same_video_tuning_and_evaluation_overlap": False,
            "all_fold_training_test_overlaps_empty": all(
                not fold["training_test_overlap"] for fold in fold_reports
            ),
            "folds": fold_reports,
        },
        "two_d_metrics": baseline_metrics,
        "two_d_plus_three_d_metrics": assisted_metrics,
        "ablation_metrics_pooled_internal_diagnostic_only": {
            name: _group_metrics(
                [
                    assisted_cache[name][record_id]
                    for record_id in sorted(record_ids)
                ]
            )
            for name, _config in CONFIG_CANDIDATES
        },
        "comparison": _metric_comparison(
            baseline_metrics,
            assisted_metrics,
        ),
        "three_d_availability": {
            "eligible_frame_count": sum(
                int(record["formal_pose_eligible_frame_count"])
                for record in selected_records
            ),
            "world_available_frame_count": sum(
                int(record["three_d_shadow"]["world_available_frame_count"])
                for record in selected_records
            ),
            "body_relative_reliable_frame_count": sum(
                int(
                    record["three_d_shadow"][
                        "body_relative_reliable_frame_count"
                    ]
                )
                for record in selected_records
            ),
        },
        "records": {
            "two_d": baseline_records,
            "two_d_plus_three_d_lovo": selected_records,
        },
        "limitations": [
            "This is an internal experiment, not an independent test result.",
            "The reviewed corpus is small and comes from a single human review workflow.",
            "Leave-one-video-out reduces direct same-video tuning leakage but does not establish cross-subject generalization.",
            "World landmarks are body-relative monocular estimates, not calibrated metric depth.",
            "The constrained fusion cannot create candidates, infer floor contact, or promote a non-VALID 2D result to VALID.",
        ],
    }
    json_path = _write(
        root / "reports" / "internal_2d_3d_shadow_evidence_lovo_v2.json",
        payload,
    )
    markdown_path = (
        root / "reports" / "internal_2d_3d_shadow_evidence_lovo_v2.md"
    )
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run internal 2D versus constrained 2D+3D LOVO experiment."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/hyrox"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.dataset_root
    if not root.is_absolute():
        root = args.project_root.resolve() / root
    json_path, markdown_path = run_experiment(root)
    payload = _load(json_path)
    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
                "comparison": payload["comparison"],
                "safety_contract": payload["safety_contract"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
