"""Build the Stage C/D formal-quality and cross-view diagnostic report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyrox.phase_decoder import TemporalPhaseDecoder
from hyrox.view_policy import action_view_capability, action_view_capability_matrix
from src.paths import installation_root


DEFAULT_OUTPUT = installation_root() / "outputs" / "occlusion_view_phase_cd"
DEFAULT_BASELINE = installation_root() / "outputs" / "occlusion_view_phase_a" / "baseline.json"


def build_report(baseline: dict[str, Any]) -> dict[str, Any]:
    cases = list(baseline.get("cases") or ())
    levels: Counter[str] = Counter()
    enriched: list[dict[str, Any]] = []
    for case in cases:
        capability = action_view_capability(
            str(case.get("action", "unknown")), str(case.get("camera_view", "unknown"))
        )
        levels[capability.level] += 1
        enriched.append(
            {
                "case_id": case.get("case_id"),
                "action": case.get("action"),
                "camera_view": case.get("camera_view"),
                "capability": capability.as_dict(),
                "pose_detection_rate": case.get("pose_detection_rate"),
                "phase_disagreement_rate": case.get("phase_disagreement_rate"),
                "low_confidence_landmark_rate": case.get("low_confidence_landmark_rate"),
                "final_counts": case.get("final_counts"),
            }
        )

    decoder = TemporalPhaseDecoder()
    decoder_trace = []
    stable = "start"
    for proposed in ("down", "down", "bottom", "bottom", "up", "up", "start", "start"):
        decoded = decoder.update(
            proposed,
            current_stable_phase=stable,
            minimum_duration_frames=2,
            legal_sequence=("start", "down", "bottom", "up", "start"),
        )
        stable = decoded.stable_phase
        decoder_trace.append(decoded.as_dict())
    return {
        "schema_version": 1,
        "artifact_type": "formal_quality_cross_view_phase_cd",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "formal_quality_contract": {
            "prediction_allowed": False,
            "forbidden_origins": ["held", "predicted", "rejected", "synthetic"],
            "unobservable_metric_value": None,
            "unobservable_metric_reason_codes_required": True,
            "identity_discontinuity_resets_in_progress_state": True,
            "identity_discontinuity_preserves_lifetime_counts": True,
            "low_quality_completed_candidate_semantics": "UNSURE",
            "interrupted_non_completed_cycle_semantics": "incomplete_not_a_candidate",
        },
        "view_capability_matrix": action_view_capability_matrix(),
        "baseline_case_capability_counts": dict(sorted(levels.items())),
        "cases": enriched,
        "decoder_reference_trace": decoder_trace,
        "calibration": {
            "status": "DATA_LIMITED",
            "independent_person_test_set_available": False,
            "thresholds_human_calibrated": False,
            "claim": "Infrastructure evaluation only; no independent-person calibration claim.",
        },
    }


def markdown(report: dict[str, Any]) -> str:
    rows = []
    for case in report["cases"]:
        rows.append(
            f"| {case['action']} | {case['camera_view']} | {case['capability']['level']} | "
            f"{float(case.get('pose_detection_rate') or 0):.3f} | "
            f"{float(case.get('phase_disagreement_rate') or 0):.3f} |"
        )
    return "\n".join(
        [
            "# 阶段 C/D：正式证据与跨视角评测",
            "",
            "## 结论",
            "",
            "- 正式流不允许 held/predicted/synthetic/rejected 点确认指标、端点或接触。",
            "- 不可观测指标返回 `null`，并输出依赖关节与原因码。",
            "- 身份不连续时清空进行中的阶段、候选、接触和地面状态，历史计数保留。",
            "- 相位输出包含确定性证据分数、质量、迟滞、持续时间与合法转移诊断。",
            "- 视角标签仅调节证据分数/建议，不会单独把正式结果降级。",
            f"- 冻结黄金视频回归：{report.get('golden_regression', {}).get('passed_count', '未运行')}/{report.get('golden_regression', {}).get('case_count', '未运行')}。",
            "",
            "## 基线样例映射",
            "",
            "| 动作 | 视角 | 能力级别 | 姿态检出率 | 原始/稳定阶段分歧率 |",
            "|---|---|---:|---:|---:|",
            *rows,
            "",
            "## 校准边界",
            "",
            "当前 8 个黄金视频没有可验证的独立人员划分和逐阶段人工真值，因此本阶段只完成能力矩阵、评分/解码基础设施和非回归验证；不声明阈值已经完成人员外校准。独立人员标注集补齐后，应按人员分组冻结训练/校准/测试集，再报告分动作、视角、遮挡等级的阶段 F1、端点召回和计数误差。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    report = build_report(baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    golden_path = args.output_dir / "golden_regression.json"
    if golden_path.exists():
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        report["golden_regression"] = {
            "status": golden.get("status"),
            "passed_count": golden.get("passed_count"),
            "case_count": golden.get("case_count"),
            "report": str(golden_path),
        }
    (args.output_dir / "phase_cd_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "phase_cd_report.md").write_text(markdown(report), encoding="utf-8")
    print(args.output_dir / "phase_cd_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
