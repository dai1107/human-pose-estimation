from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sports.basketball.consistency import analyze_shot_consistency, load_shot_summary
from src.utils.time_utils import make_session_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple basketball shot reports for repeatability.")
    parser.add_argument("--shots", nargs="+", required=True, help="Shot report directories or shot_summary.json files.")
    parser.add_argument("--shooting-side", required=True, choices=["right", "left"])
    parser.add_argument("--shot-type", required=True, choices=["set_shot", "jump_shot"])
    parser.add_argument("--reference", default=None, help="Reserved for reference-aware multi-shot comparison.")
    parser.add_argument("--output-dir", default="outputs/basketball/reports")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = [load_shot_summary(path) for path in args.shots]
    consistency = analyze_shot_consistency(summaries)
    output_dir = Path(args.output_dir) / f"{make_session_id()}_shot_consistency"
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "shot_type": args.shot_type,
        "shooting_side": args.shooting_side,
        "reference": args.reference,
        "consistency_metrics": consistency,
        "note": "Consistency describes repeat-to-repeat kinematic similarity, not shooting percentage or skill level.",
    }
    (output_dir / "shot_consistency.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown(payload), encoding="utf-8")
    print(f"Shot consistency report written: {output_dir}")
    return 0


def _markdown(payload: dict) -> str:
    metrics = payload["consistency_metrics"]
    return (
        "# 多次投篮一致性报告\n\n"
        f"共分析 {metrics['shot_count']} 次投篮。\n\n"
        f"出手代理时刻标准差：{metrics.get('release_proxy_time_std_ms')} ms\n\n"
        f"动力链事件顺序一致性：{metrics['sequence_order_consistency']['consistent_count']} / {metrics['sequence_order_consistency']['total_count']}。\n\n"
        "这里的一致性描述重复动作之间的相似程度，不等同于投篮命中率或技术水平。\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())

