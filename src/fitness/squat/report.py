from __future__ import annotations

import json
import os
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".cache" / "matplotlib"))

from src.reference.session_loader import load_session, write_csv_rows
from src.utils.time_utils import make_session_id, now_iso

from .phase_metrics import build_squat_frames_from_session, compute_all_rep_metrics
from .reference_compare import compare_reps_to_reference
from .rep_detector import detect_squat_reps
from .schema import SquatFrameMeasurement, SquatRepMetrics, load_squat_config
from .view_metrics import summarize_view_metrics


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.8g}" if isfinite(number) else ""


def _metrics_row(metrics: SquatRepMetrics) -> dict[str, Any]:
    return {key: _number(value) if isinstance(value, (int, float)) or value is None else value for key, value in metrics.to_dict().items()}


def _summary(metrics: list[SquatRepMetrics], frame_count: int) -> dict[str, Any]:
    def avg(name: str) -> float | None:
        values = [getattr(item, name) for item in metrics]
        finite = [float(value) for value in values if value is not None and isfinite(float(value))]
        return mean(finite) if finite else None

    quality_warnings = sum(1 for item in metrics if item.data_quality_level != "GOOD")
    return {
        "complete_rep_count": len(metrics),
        "valid_rep_count": sum(1 for item in metrics if item.data_quality_level in {"GOOD", "WARNING"}),
        "data_quality_warning_count": quality_warnings,
        "frame_count": frame_count,
        "average_total_duration_ms": avg("total_duration_ms"),
        "average_descent_duration_ms": avg("descent_duration_ms"),
        "average_ascent_duration_ms": avg("ascent_duration_ms"),
        "average_pelvis_vertical_displacement_normalized": avg("pelvis_vertical_displacement_normalized"),
        "average_left_right_knee_difference_mean": avg("left_right_knee_difference_mean"),
    }


def _pelvis_displacements(frames: list[SquatFrameMeasurement]) -> list[float]:
    usable = [frame for frame in frames if frame.pelvis_y is not None and frame.body_scale is not None and frame.body_scale > 1e-9]
    if not usable:
        return [0.0 for _ in frames]
    baseline_y = usable[0].pelvis_y
    scale = usable[0].body_scale or 1.0
    return [
        ((frame.pelvis_y - baseline_y) / scale) if frame.pelvis_y is not None and scale > 1e-9 else float("nan")
        for frame in frames
    ]


def _plot_rep_timeline(path: Path, frames: list[SquatFrameMeasurement], frame_states: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frames:
        return
    first_ts = frames[0].timestamp_ms
    times = [(frame.timestamp_ms - first_ts) / 1000.0 for frame in frames]
    displacement = _pelvis_displacements(frames)
    plt.figure(figsize=(10, 4))
    plt.plot(times, displacement, label="pelvis displacement proxy")
    state_to_y = {"READY": 0.0, "DESCENT": 0.2, "BOTTOM": 0.4, "ASCENT": 0.6, "PAUSED": 0.8}
    state_values = [state_to_y.get(row.get("state"), 0.0) for row in frame_states[: len(times)]]
    plt.plot(times[: len(state_values)], state_values, label="state track", alpha=0.6)
    plt.title("Squat repetition timeline")
    plt.xlabel("time (s)")
    plt.ylabel("normalized proxy")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_angle_curves(path: Path, frames: list[SquatFrameMeasurement]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frames:
        return
    first_ts = frames[0].timestamp_ms
    times = [(frame.timestamp_ms - first_ts) / 1000.0 for frame in frames]
    series = {
        "left_knee": [frame.left_knee_angle for frame in frames],
        "right_knee": [frame.right_knee_angle for frame in frames],
        "left_hip": [frame.left_hip_angle for frame in frames],
        "right_hip": [frame.right_hip_angle for frame in frames],
    }
    plt.figure(figsize=(10, 5))
    for name, values in series.items():
        plt.plot(times, values, label=name)
    plt.title("Squat angle curves")
    plt.xlabel("time (s)")
    plt.ylabel("angle (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_symmetry(path: Path, frames: list[SquatFrameMeasurement]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frames:
        return
    first_ts = frames[0].timestamp_ms
    times = [(frame.timestamp_ms - first_ts) / 1000.0 for frame in frames]
    knee_diff = [
        abs(frame.left_knee_angle - frame.right_knee_angle)
        if frame.left_knee_angle is not None and frame.right_knee_angle is not None
        else float("nan")
        for frame in frames
    ]
    hip_diff = [
        abs(frame.left_hip_angle - frame.right_hip_angle)
        if frame.left_hip_angle is not None and frame.right_hip_angle is not None
        else float("nan")
        for frame in frames
    ]
    plt.figure(figsize=(10, 4))
    plt.plot(times, knee_diff, label="knee angle difference")
    plt.plot(times, hip_diff, label="hip angle difference")
    plt.title("Left-right symmetry proxy")
    plt.xlabel("time (s)")
    plt.ylabel("absolute difference (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _write_keyframe(path: Path, title: str, frames: list[SquatFrameMeasurement], timestamp_ms: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    times = [frame.timestamp_ms for frame in frames]
    displacement = _pelvis_displacements(frames)
    plt.figure(figsize=(5, 4))
    if times:
        relative = [(value - times[0]) / 1000.0 for value in times]
        plt.plot(relative, displacement)
        marker_x = (timestamp_ms - times[0]) / 1000.0
        plt.axvline(marker_x, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("pelvis displacement proxy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _write_keyframes(report_dir: Path, metrics: list[SquatRepMetrics], frames_by_rep: dict[int, list[SquatFrameMeasurement]]) -> None:
    key_dir = report_dir / "annotated_keyframes"
    for item in metrics:
        frames = frames_by_rep.get(item.rep_index, [])
        prefix = f"rep_{item.rep_index:03d}"
        _write_keyframe(key_dir / f"{prefix}_start.png", f"Rep {item.rep_index} start", frames, item.start_timestamp_ms)
        _write_keyframe(key_dir / f"{prefix}_bottom.png", f"Rep {item.rep_index} bottom", frames, item.bottom_timestamp_ms)
        _write_keyframe(key_dir / f"{prefix}_end.png", f"Rep {item.rep_index} end", frames, item.end_timestamp_ms)


def _write_markdown(path: Path, summary: dict[str, Any], metrics: list[SquatRepMetrics], view: dict[str, Any], reference_summary: dict[str, Any] | None) -> None:
    metric_labels = {
        "rep_count": "重复次数",
        "basic_joint_angles": "基础关节角",
        "left_knee_min_angle": "左膝最小角度",
        "right_knee_min_angle": "右膝最小角度",
        "left_hip_min_angle": "左髋最小角度",
        "right_hip_min_angle": "右髋最小角度",
        "pelvis_vertical_displacement_normalized": "骨盆垂直位移",
        "left_knee_angle_range": "左膝角度范围",
        "right_knee_angle_range": "右膝角度范围",
        "left_hip_angle_range": "左髋角度范围",
        "right_hip_angle_range": "右髋角度范围",
        "trunk_tilt_range": "躯干倾斜范围",
        "descent_duration_ms": "下降时长",
        "ascent_duration_ms": "起身时长",
        "bottom_duration_ms": "底部停留时长",
        "left_right_knee_difference_mean": "左右膝平均差异",
        "left_right_knee_difference_peak": "左右膝峰值差异",
        "left_right_hip_difference_mean": "左右髋平均差异",
        "left_right_hip_difference_peak": "左右髋峰值差异",
        "pelvis_lateral_drift_proxy": "骨盆横向偏移代理",
        "trunk_lateral_drift_proxy": "躯干横向偏移代理",
        "knee_lateral_trajectory_proxy": "膝盖横向轨迹代理",
        "precise_squat_depth": "精确下蹲深度",
        "precise_hip_knee_flexion_depth": "精确髋膝屈曲深度",
        "precise_depth_or_lateral_tracking": "精确深度或横向轨迹",
        "view_sensitive_metrics": "视角敏感指标",
    }
    note_labels = {
        "SIDE view: lateral knee trajectory is not reported or is low reliability.": "侧面视角：膝盖横向轨迹不输出，或可靠性较低。",
        "FRONT view: hip and knee flexion depth is only a 2D visual proxy, not precise depth measurement.": "正面视角：髋膝屈曲深度只作为二维视觉代理，不代表精确深度测量。",
        "Angled front view provides mixed visual proxy metrics with lower reliability.": "斜前方视角：可输出混合视觉代理指标，但可靠性低于标准侧面或正面视角。",
        "UNKNOWN view: only basic joint angles and repetition count are reported.": "未知视角：仅输出基础关节角和重复次数。",
    }

    def label(name: str) -> str:
        return metric_labels.get(name, name)

    def label_list(names: list[str]) -> str:
        return "、".join(label(name) for name in names) if names else "无"

    def seconds(value_ms: int | float | None) -> str:
        if value_ms is None:
            return ""
        return f"{float(value_ms) / 1000.0:.2f} s"

    lines = [
        "# 深蹲专项分析报告",
        "",
        "## 概览",
        "",
        f"- 检测到完整深蹲次数：{summary['complete_rep_count']}",
        f"- 有效重复次数：{summary['valid_rep_count']}",
        f"- 数据质量警告次数：{summary['data_quality_warning_count']}",
        f"- 分析帧数：{summary['frame_count']}",
        "",
        "## 视角与指标",
        "",
        f"- 当前分析视角：{view['camera_view']}",
        f"- 当前可用指标：{label_list(view['available_metrics'])}",
        f"- 当前不可靠或不可用指标：{label_list(view['unavailable_or_low_reliability'])}",
        "",
    ]
    for note in view["notes"]:
        lines.append(f"- {note_labels.get(note, note)}")
    lines.append("")

    if metrics:
        lines.extend(["## 单次深蹲明细", ""])
        for item in metrics[:5]:
            lines.extend(
                [
                    f"### 第 {item.rep_index} 次深蹲",
                    "",
                    f"- 总时长：{seconds(item.total_duration_ms)}",
                    f"- 下降时长：{seconds(item.descent_duration_ms)}",
                    f"- 底部停留：{seconds(item.bottom_duration_ms)}",
                    f"- 起身时长：{seconds(item.ascent_duration_ms)}",
                    f"- 左膝最小角度：{_number(item.left_knee_min_angle)} deg",
                    f"- 右膝最小角度：{_number(item.right_knee_min_angle)} deg",
                    f"- 左髋最小角度：{_number(item.left_hip_min_angle)} deg",
                    f"- 右髋最小角度：{_number(item.right_hip_min_angle)} deg",
                    f"- 左右膝峰值差异：{_number(item.left_right_knee_difference_peak)} deg",
                    f"- 左右髋峰值差异：{_number(item.left_right_hip_difference_peak)} deg",
                    f"- 躯干倾斜变化范围：{_number(item.trunk_tilt_range)} deg",
                    f"- 骨盆相对垂直位移：{_number(item.pelvis_vertical_displacement_normalized)} body-scale",
                    f"- 姿态有效帧比例：{_number(item.pose_valid_ratio)}",
                    f"- 数据质量等级：{item.data_quality_level}",
                    "",
                ]
            )
    else:
        lines.extend(["## 单次深蹲明细", "", "未检测到完整深蹲重复。", ""])

    if reference_summary:
        lines.extend(["## 与个人参考动作对比", ""])
        if reference_summary.get("rep_comparisons"):
            for comparison in reference_summary["rep_comparisons"][:5]:
                top = comparison.get("top_difference_features", [])
                feature = label(top[0]["feature"]) if top else "可用特征"
                lines.append(f"- 第 {comparison['rep_index']} 次深蹲在 {feature} 上与参考动作差异较大。")
        else:
            lines.append("- 参考比较未生成稳定结果。")
        lines.extend(["", "以上内容是相对个人参考动作的差异描述，不代表绝对正确或错误。", ""])

    lines.extend(
        [
            "## 说明",
            "",
            "以上结果来自单摄像头视觉运动学测量和相对变化分析，不构成医疗诊断、动作安全性判断或训练处方。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_squat_session(
    session_dir: str | Path,
    camera_view: str = "unknown",
    output_dir: str | Path = "outputs/squat_reports",
    config_path: str | Path | None = None,
    reference_dir: str | Path | None = None,
) -> Path:
    session = load_session(session_dir)
    config = load_squat_config(config_path)
    frames = build_squat_frames_from_session(session)
    detection = detect_squat_reps(frames, camera_view=camera_view, config=config)
    metrics = compute_all_rep_metrics(detection.reps, detection.calibration)
    view_summary = summarize_view_metrics(camera_view, metrics)

    report_id = make_session_id()
    root = Path(output_dir)
    report_dir = root / report_id
    suffix = 1
    while report_dir.exists():
        report_dir = root / f"{report_id}_{suffix}"
        suffix += 1
    report_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "report_id": report_dir.name,
        "created_at": now_iso(),
        "session_id": session.session_id,
        "session_dir": str(Path(session_dir)),
        "camera_view": camera_view,
        "analysis_name": config.get("analysis_name", "squat_basic_v1"),
        "calibration": detection.calibration.to_dict(),
        "view_metrics": view_summary.to_dict(),
    }
    (report_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv_rows(report_dir / "squat_reps.csv", [_metrics_row(item) for item in metrics])
    write_csv_rows(report_dir / "squat_frames.csv", detection.frame_states)
    summary = _summary(metrics, len(frames))
    reference_summary = None
    if reference_dir is not None and metrics:
        reference_summary = compare_reps_to_reference(detection.reps, reference_dir, report_dir)
        summary["reference_comparison"] = reference_summary
    (report_dir / "squat_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_rep_timeline(report_dir / "rep_timeline.png", frames, detection.frame_states)
    _plot_angle_curves(report_dir / "angle_curves_by_rep.png", frames)
    _plot_symmetry(report_dir / "symmetry_curves.png", frames)
    frames_by_rep = {rep.rep_index: rep.frames for rep in detection.reps}
    _write_keyframes(report_dir, metrics, frames_by_rep)
    _write_markdown(report_dir / "report.md", summary, metrics, view_summary.to_dict(), reference_summary)
    return report_dir
