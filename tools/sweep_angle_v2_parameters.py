from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from types import MappingProxyType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyrox.config import load_lunge_config, load_wall_ball_config  # noqa: E402
from src.angle_v2 import (  # noqa: E402
    FAIL,
    PASS,
    UNSURE,
    AngleHysteresis,
    AngleV2Config,
    JointAngleSmoothingConfig,
    TemporalRuleEvidence,
    load_angle_v2_config,
)
from tools.run_angle_v2_shadow import (  # noqa: E402
    load_phone_manifest,
    load_record_angle_curves,
    replay_angle_v2,
)


TARGET_ACTIONS = frozenset({"lunge", "wall_ball"})
ANGLE_ERROR_CODES = {
    "lunge": "HIP_NOT_EXTENDED",
    "wall_ball": "NOT_DEEP_ENOUGH",
}
TARGET_PHASES = {"lunge": "stand", "wall_ball": "bottom"}
TARGET_EVENTS = {"lunge": "full_extension", "wall_ball": "bottom_reached"}


@dataclass(frozen=True, slots=True)
class SweepCandidate:
    name: str
    threshold_offset_deg: float = 0.0
    minimum_hold_ms: float = 100.0
    hysteresis_width_deg: float = 3.0
    minimum_confidence: float = 0.50
    disagreement_limit_deg: float = 25.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold_offset_deg": self.threshold_offset_deg,
            "minimum_hold_ms": self.minimum_hold_ms,
            "hysteresis_width_deg": self.hysteresis_width_deg,
            "minimum_confidence": self.minimum_confidence,
            "2d_3d_disagreement_limit_deg": self.disagreement_limit_deg,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded Angle V2 shadow parameter sweep against reviewed "
            "phone-RGB lunge and wall-ball angle-rule labels."
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/hyrox")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/angle_v2_shadow.yaml")
    )
    parser.add_argument(
        "--round2-summary",
        type=Path,
        default=Path(
            "outputs/angle_validation/angle_v2_round2/angle_v2_shadow_replay.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/angle_validation/angle_v2_round3_sweep"),
    )
    return parser


def bounded_candidates() -> list[SweepCandidate]:
    baseline = SweepCandidate("baseline_shadow")
    rows = [baseline]
    rows.extend(
        replace(
            baseline,
            name=f"threshold_{offset:+g}",
            threshold_offset_deg=float(offset),
        )
        for offset in (-5, -3, -2, 2, 3, 5)
    )
    rows.extend(
        replace(baseline, name=f"hold_{value:g}ms", minimum_hold_ms=float(value))
        for value in (0, 66, 150)
    )
    rows.extend(
        replace(
            baseline,
            name=f"hysteresis_{value:g}",
            hysteresis_width_deg=float(value),
        )
        for value in (0, 2, 5)
    )
    rows.extend(
        replace(
            baseline,
            name=f"confidence_{value:.2f}",
            minimum_confidence=float(value),
        )
        for value in (0.35, 0.45, 0.60, 0.70)
    )
    rows.extend(
        replace(
            baseline,
            name=f"conflict_{value:g}",
            disagreement_limit_deg=float(value),
        )
        for value in (15, 20, 30, 35, 45)
    )
    rows.extend(
        (
            SweepCandidate("lenient_bundle", -3, 66, 2, 0.45, 35),
            SweepCandidate("false_no_rep_guard", -5, 100, 3, 0.45, 35),
            SweepCandidate("stable_bundle", 2, 150, 5, 0.60, 25),
            SweepCandidate("quality_guard_bundle", 0, 100, 3, 0.60, 20),
        )
    )
    return rows


def apply_round2_smoothing(
    config: AngleV2Config, summary_path: str | Path
) -> AngleV2Config:
    source = Path(summary_path)
    if not source.is_file():
        return config
    payload = json.loads(source.read_text(encoding="utf-8"))
    tuned = payload.get("tuned_shadow_config")
    smoothing = tuned.get("joint_smoothing") if isinstance(tuned, Mapping) else None
    if not isinstance(smoothing, Mapping):
        return config
    profiles = dict(config.joint_smoothing)
    for name, raw in smoothing.items():
        if not isinstance(raw, Mapping):
            continue
        profiles[str(name)] = JointAngleSmoothingConfig(
            min_cutoff=float(raw["min_cutoff"]),
            beta=float(raw["beta"]),
            d_cutoff=float(raw.get("d_cutoff", 1.0)),
        )
    return replace(config, joint_smoothing=MappingProxyType(profiles))


def load_reviewed_angle_rules(
    dataset_root: str | Path,
) -> list[dict[str, Any]]:
    path = Path(dataset_root) / "reviews" / "human_rgb_fine_annotations_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("oni_records_included") is not False:
        raise ValueError("Angle V2 sweep must not include ONI records")
    return [
        dict(item)
        for item in payload.get("records", ())
        if isinstance(item, Mapping)
        and str(item.get("action")) in TARGET_ACTIONS
        and bool(item.get("internal_rgb_rule_calibration_eligible"))
    ]


def evaluate_candidate(
    candidate: SweepCandidate,
    reviews: Sequence[Mapping[str, Any]],
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for review in reviews:
        action = str(review["action"])
        record_id = str(review["record_id"])
        for rep in review.get("reps", ()):
            if not isinstance(rep, Mapping):
                continue
            segment = _target_segment(review, rep, action=action)
            expected = _expected_angle_status(review, rep, action=action)
            predicted, settled_frame = _predict_rep(
                candidate,
                action=action,
                record_id=record_id,
                start_frame=segment[0],
                end_frame=segment[1],
                lookup=lookup,
            )
            event_frame = _human_event_frame(review, rep, action=action)
            rows.append(
                {
                    "candidate": candidate.name,
                    "record_id": record_id,
                    "action": action,
                    "rep_id": rep.get("rep_id"),
                    "rep_start_frame": int(rep.get("start_frame", 0)),
                    "rep_end_frame": int(rep.get("end_frame", 0)),
                    "evaluation_start_frame": segment[0],
                    "evaluation_end_frame": segment[1],
                    "expected_angle_status": expected,
                    "predicted_angle_status": predicted,
                    "human_event_frame": event_frame,
                    "settled_event_frame": settled_frame,
                    "event_frame_error": (
                        settled_frame - event_frame
                        if settled_frame is not None and event_frame is not None
                        else None
                    ),
                }
            )
    metrics = _loss_metrics(rows)
    metrics["candidate"] = candidate.as_dict()
    metrics["by_action"] = {
        action: _loss_metrics([row for row in rows if row["action"] == action])
        for action in sorted(TARGET_ACTIONS)
    }
    metrics["by_record"] = {
        record_id: _loss_metrics(
            [row for row in rows if row["record_id"] == record_id]
        )
        for record_id in sorted({str(row["record_id"]) for row in rows})
    }
    return metrics, rows


def run_sweep(
    reviews: Sequence[Mapping[str, Any]],
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    candidates: Sequence[SweepCandidate] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = list(candidates or bounded_candidates())
    results = []
    all_rows = []
    for candidate in grid:
        metrics, rows = evaluate_candidate(candidate, reviews, lookup)
        results.append(metrics)
        all_rows.extend(rows)
    baseline = next(
        item for item in results if item["candidate"]["name"] == "baseline_shadow"
    )
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["weighted_loss"]),
            int(item["false_no_rep_count"]),
            int(item["false_valid_count"]),
            int(item["excess_unsure_count"]),
        ),
    )
    best = ranked[0]
    improvement = float(baseline["weighted_loss"]) - float(best["weighted_loss"])
    improved_records = sum(
        float(best["by_record"][record]["weighted_loss"])
        < float(baseline["by_record"][record]["weighted_loss"])
        for record in best["by_record"]
    )
    worsened_records = sum(
        float(best["by_record"][record]["weighted_loss"])
        > float(baseline["by_record"][record]["weighted_loss"])
        for record in best["by_record"]
    )
    independent_holdout_available = any(
        str(review.get("dataset_role")) in {"validation", "test"}
        for review in reviews
    )
    clear_improvement = (
        improvement > 0.0
        and int(best["false_no_rep_count"])
        <= int(baseline["false_no_rep_count"])
        and improved_records > worsened_records
    )
    default_replacement_allowed = clear_improvement and independent_holdout_available
    summary = {
        "schema_version": 1,
        "artifact_type": "hyrox_angle_v2_round3_parameter_sweep_v1",
        "shadow_only": True,
        "candidate_count": len(grid),
        "reviewed_record_count": len(reviews),
        "reviewed_rep_count": sum(len(review.get("reps", ())) for review in reviews),
        "objective": (
            "3*false_NO_REP + 2*false_VALID + 1*excess_UNSURE + "
            "0.5*mean_absolute_event_frame_error"
        ),
        "baseline": baseline,
        "best": best,
        "weighted_loss_improvement": improvement,
        "best_improved_record_count": improved_records,
        "best_worsened_record_count": worsened_records,
        "clear_shadow_improvement": clear_improvement,
        "independent_holdout_available": independent_holdout_available,
        "default_replacement_allowed": default_replacement_allowed,
        "default_replaced": False,
        "default_replacement_reason": (
            "正式默认值保持不变；目标动作的人工复核记录没有独立的 "
            "validation/test holdout"
            if not independent_holdout_available
            else "正式默认值保持不变，等待显式集成复核"
        ),
        "ranked_candidates": ranked,
        "limitations": [
            "sweep evaluates angle-rule evidence, not complete HYROX validity",
            "lunge evaluates knee+hip full extension",
            "wall ball evaluates knee-angle depth proxy; hip-below-knee remains a separate rule",
            "human phase intervals are used only to isolate calibration windows",
            "no 3D channel is promoted to formal truth",
        ],
    }
    return summary, all_rows


def _predict_rep(
    candidate: SweepCandidate,
    *,
    action: str,
    record_id: str,
    start_frame: int,
    end_frame: int,
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[str, int | None]:
    lunge = load_lunge_config()
    wall = load_wall_ball_config()
    if action == "lunge":
        knee_threshold = (
            float(lunge["full_extension_knee_angle_min"])
            + candidate.threshold_offset_deg
        )
        hip_threshold = (
            float(lunge["full_extension_hip_angle_min"])
            + candidate.threshold_offset_deg
        )
        hysteresis = AngleHysteresis(
            threshold=0.0,
            direction="above",
            width_deg=candidate.hysteresis_width_deg,
        )
    else:
        knee_threshold = float(wall["bottom_knee_angle_max"]) + candidate.threshold_offset_deg
        hip_threshold = 0.0
        hysteresis = AngleHysteresis(
            threshold=knee_threshold,
            direction="below",
            width_deg=candidate.hysteresis_width_deg,
        )
    evidence = TemporalRuleEvidence(
        window_size=5,
        pass_count=3,
        fail_count=3,
        minimum_hold_ms=candidate.minimum_hold_ms,
    )
    saw_pass = False
    saw_fail = False
    first_pass_frame: int | None = None
    timestamps = []
    for frame in range(start_frame, end_frame + 1):
        if action == "lunge":
            values = []
            for side in ("left", "right"):
                knee = lookup.get((record_id, f"{side}_knee", frame))
                hip = lookup.get((record_id, f"{side}_hip", frame))
                if _usable(knee, candidate) and _usable(hip, candidate):
                    values.append(
                        min(
                            float(knee["filtered_2d_angle_deg"]) - knee_threshold,
                            float(hip["filtered_2d_angle_deg"]) - hip_threshold,
                        )
                    )
            value = max(values) if values else None
            valid = value is not None
        else:
            knees = [
                row
                for side in ("left", "right")
                if (row := lookup.get((record_id, f"{side}_knee", frame))) is not None
                and _usable(row, candidate)
            ]
            value = (
                min(float(row["filtered_2d_angle_deg"]) for row in knees)
                if knees
                else None
            )
            valid = value is not None
        timestamp = _frame_timestamp(
            lookup, record_id=record_id, frame=frame, action=action
        )
        timestamps.append(timestamp)
        condition = hysteresis.observe(value, valid=valid)
        settled = evidence.observe(condition, timestamp_ms=timestamp)
        if settled == PASS:
            saw_pass = True
            if first_pass_frame is None:
                first_pass_frame = frame
        elif settled == FAIL:
            saw_fail = True
    if saw_pass:
        return PASS, first_pass_frame
    if saw_fail:
        return FAIL, None
    return UNSURE, None


def _usable(row: Mapping[str, Any] | None, candidate: SweepCandidate) -> bool:
    if not isinstance(row, Mapping):
        return False
    value = _finite(row.get("filtered_2d_angle_deg"))
    confidence = _finite(row.get("confidence"))
    disagreement = _finite(row.get("disagreement_deg"))
    return (
        value is not None
        and confidence is not None
        and confidence >= candidate.minimum_confidence
        and bool(row.get("bone_length_valid"))
        and not bool(row.get("temporal_outlier"))
        and (
            disagreement is None
            or disagreement <= candidate.disagreement_limit_deg
        )
    )


def _target_segment(
    review: Mapping[str, Any], rep: Mapping[str, Any], *, action: str
) -> tuple[int, int]:
    rep_start = int(rep.get("start_frame", 0))
    rep_end = int(rep.get("end_frame", rep_start))
    phase = TARGET_PHASES[action]
    intervals = [
        item
        for item in review.get("phase_error_intervals", ())
        if isinstance(item, Mapping)
        and str(item.get("phase")) == phase
        and int(item.get("end_frame", -1)) >= rep_start
        and int(item.get("start_frame", rep_end + 1)) <= rep_end
    ]
    if intervals:
        selected = max(intervals, key=lambda item: int(item.get("end_frame", 0)))
        return (
            max(rep_start, int(selected["start_frame"])),
            min(rep_end, int(selected["end_frame"])),
        )
    if action == "lunge":
        return max(rep_start, rep_end - max(5, (rep_end - rep_start) // 4)), rep_end
    middle = (rep_start + rep_end) // 2
    return max(rep_start, middle - 5), min(rep_end, middle + 5)


def _expected_angle_status(
    review: Mapping[str, Any], rep: Mapping[str, Any], *, action: str
) -> str:
    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", start))
    target_error = ANGLE_ERROR_CODES[action]
    has_error = any(
        isinstance(item, Mapping)
        and str(item.get("error_code")) == target_error
        and int(item.get("end_frame", -1)) >= start
        and int(item.get("start_frame", end + 1)) <= end
        for item in review.get("phase_error_intervals", ())
    )
    return FAIL if has_error else PASS


def _human_event_frame(
    review: Mapping[str, Any], rep: Mapping[str, Any], *, action: str
) -> int | None:
    start = int(rep.get("start_frame", 0))
    end = int(rep.get("end_frame", start))
    event_type = TARGET_EVENTS[action]
    frames = [
        int(item["frame_index"])
        for item in review.get("events", ())
        if isinstance(item, Mapping)
        and str(item.get("event_type")) == event_type
        and start <= int(item.get("frame_index", -1)) <= end
    ]
    return frames[-1] if frames else None


def _loss_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    false_no_rep = sum(
        row["expected_angle_status"] == PASS
        and row["predicted_angle_status"] == FAIL
        for row in rows
    )
    false_valid = sum(
        row["expected_angle_status"] == FAIL
        and row["predicted_angle_status"] == PASS
        for row in rows
    )
    unsure = sum(row["predicted_angle_status"] == UNSURE for row in rows)
    errors = [
        abs(int(row["event_frame_error"]))
        for row in rows
        if row.get("event_frame_error") is not None
        and row["expected_angle_status"] == PASS
        and row["predicted_angle_status"] == PASS
    ]
    event_error = fmean(errors) if errors else 0.0
    loss = 3.0 * false_no_rep + 2.0 * false_valid + unsure + 0.5 * event_error
    confusion = {
        expected: {
            predicted: sum(
                row["expected_angle_status"] == expected
                and row["predicted_angle_status"] == predicted
                for row in rows
            )
            for predicted in (PASS, FAIL, UNSURE)
        }
        for expected in (PASS, FAIL)
    }
    return {
        "rep_count": len(rows),
        "false_no_rep_count": false_no_rep,
        "false_valid_count": false_valid,
        "excess_unsure_count": unsure,
        "mean_absolute_event_frame_error": event_error,
        "event_error_count": len(errors),
        "weighted_loss": loss,
        "status_confusion": confusion,
    }


def _frame_timestamp(
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    record_id: str,
    frame: int,
    action: str,
) -> float:
    names = (
        ("left_knee", "right_knee", "left_hip", "right_hip")
        if action == "lunge"
        else ("left_knee", "right_knee")
    )
    for name in names:
        row = lookup.get((record_id, name, frame))
        value = _finite(row.get("timestamp_ms")) if isinstance(row, Mapping) else None
        if value is not None:
            return value
    return frame * 1000.0 / 30.0


def write_artifacts(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "angle_v2_parameter_sweep.json"
    rows_path = target / "angle_v2_sweep_rep_rows.csv"
    report_path = target / "ANGLE_V2_ROUND3_REPORT.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0]) if rows else []
    with rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    report_path.write_text(_markdown(summary), encoding="utf-8")
    return summary_path, rows_path, report_path


def _markdown(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline"]
    best = summary["best"]
    return "\n".join(
        [
            "# 第 3 轮：HYROX Angle V2 参数 Sweep",
            "",
            f"- 受评记录：{summary['reviewed_record_count']}",
            f"- 受评动作：Lunge、Wall Ball",
            f"- 受评 repetition：{summary['reviewed_rep_count']}",
            f"- 参数候选：{summary['candidate_count']}",
            "- ONI 数据：未使用",
            "",
            "## 结果",
            "",
            f"- 基线 weighted loss：{baseline['weighted_loss']:.4f}",
            f"- 最佳 weighted loss：{best['weighted_loss']:.4f}",
            f"- 最佳候选：{best['candidate']['name']}",
            f"- false NO_REP：{baseline['false_no_rep_count']} → {best['false_no_rep_count']}",
            f"- false VALID：{baseline['false_valid_count']} → {best['false_valid_count']}",
            f"- excess UNSURE：{baseline['excess_unsure_count']} → {best['excess_unsure_count']}",
            "- 平均事件帧误差："
            f"{baseline['mean_absolute_event_frame_error']:.4f} → "
            f"{best['mean_absolute_event_frame_error']:.4f}",
            "",
            "## 默认值决策",
            "",
            f"- clear shadow improvement：{str(summary['clear_shadow_improvement']).lower()}",
            "- independent holdout available："
            f"{str(summary['independent_holdout_available']).lower()}",
            f"- default replacement allowed：{str(summary['default_replacement_allowed']).lower()}",
            "- 正式默认值已替换：false",
            "",
            summary["default_replacement_reason"] + "。",
            "",
            "该 sweep 只比较角度规则证据，不替代膝触地、脚跟稳定、球体或其他正式动作规则。",
            "",
        ]
    )


def _finite(value: object) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if math.isfinite(resolved) else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_round2_smoothing(
        load_angle_v2_config(args.config), args.round2_summary
    )
    reviews = load_reviewed_angle_rules(args.dataset_root)
    manifest = {
        str(item["record_id"]): item for item in load_phone_manifest(args.dataset_root)
    }
    record_ids = {str(review["record_id"]) for review in reviews}
    joints = [
        f"{side}_{joint}"
        for side in ("left", "right")
        for joint in ("knee", "hip")
    ]
    record_data = {
        record_id: load_record_angle_curves(
            args.dataset_root,
            manifest[record_id],
            joints=joints,
            config=config,
        )
        for record_id in sorted(record_ids)
    }
    _replay, lookup, _endpoints = replay_angle_v2(record_data, config=config)
    summary, rows = run_sweep(reviews, lookup)
    summary["round2_config_source"] = str(args.round2_summary)
    summary["angle_v2_config"] = config.as_dict()
    paths = write_artifacts(args.output_dir, summary, rows)
    print(
        json.dumps(
            {
                "summary": str(paths[0]),
                "rows": str(paths[1]),
                "report": str(paths[2]),
                "best_candidate": summary["best"]["candidate"]["name"],
                "default_replaced": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SweepCandidate",
    "apply_round2_smoothing",
    "bounded_candidates",
    "evaluate_candidate",
    "load_reviewed_angle_rules",
    "main",
    "run_sweep",
    "write_artifacts",
]
