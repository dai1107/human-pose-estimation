from __future__ import annotations

import os
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"))


ANGLE_FEATURES = (
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
)

VELOCITY_FEATURES = (
    "pelvis_speed",
    "left_wrist_speed",
    "right_wrist_speed",
    "left_ankle_speed",
    "right_ankle_speed",
)


def _plot_feature_group(
    output_path: Path,
    reference: np.ndarray,
    candidate: np.ndarray,
    feature_names: list[str],
    selected_features: tuple[str, ...],
    title: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if reference.size == 0 or candidate.size == 0:
        return
    axis = np.linspace(0.0, 100.0, reference.shape[0])
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    plt.figure(figsize=(10, 5))
    plotted = False
    for name in selected_features:
        index = name_to_index.get(name)
        if index is None:
            continue
        plt.plot(axis, reference[:, index], label=f"ref {name}", linewidth=1.8)
        plt.plot(axis, candidate[:, index], label=f"candidate {name}", linestyle="--", linewidth=1.4)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(title)
    plt.xlabel("normalized aligned timeline (%)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _plot_phase_difference(
    output_path: Path,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if reference.size == 0 or candidate.size == 0:
        return
    diff = np.abs(reference - candidate)
    local = np.nanmean(diff, axis=1)
    axis = np.linspace(0.0, 100.0, len(local))
    plt.figure(figsize=(10, 4))
    plt.plot(axis, local, color="#2f6f9f", linewidth=2.0)
    plt.title("Phase difference over aligned timeline")
    plt.xlabel("normalized aligned timeline (%)")
    plt.ylabel("mean feature error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def write_comparison_plots(
    output_dir: str | Path,
    aligned_reference: np.ndarray,
    aligned_candidate: np.ndarray,
    feature_names: list[str],
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    _plot_feature_group(path / "angle_comparison.png", aligned_reference, aligned_candidate, feature_names, ANGLE_FEATURES, "Angle comparison", "angle (deg)")
    _plot_feature_group(path / "velocity_comparison.png", aligned_reference, aligned_candidate, feature_names, VELOCITY_FEATURES, "Velocity proxy comparison", "normalized units / s")
    _plot_phase_difference(path / "phase_difference.png", aligned_reference, aligned_candidate)


def similarity_label(normalized_distance: float) -> str:
    if not isfinite(normalized_distance):
        return "无法评估"
    if normalized_distance < 5.0:
        return "较高"
    if normalized_distance < 20.0:
        return "中等"
    return "较低"


def write_markdown_report(output_path: str | Path, summary: dict[str, Any]) -> None:
    top_features = summary.get("top_difference_features", [])
    top_ranges = summary.get("top_difference_time_ranges", [])
    quality = summary.get("data_quality_summary", {})
    label = similarity_label(float(summary.get("normalized_dtw_distance", float("nan"))))
    lines = [
        "# 个人参考动作比较报告",
        "",
        f"本次动作与参考动作的总体运动学相似度：{label}。",
        "",
        "差异较大的特征：",
    ]
    if top_features:
        for index, item in enumerate(top_features, start=1):
            lines.append(f"{index}. {item['feature']}，平均偏离 {item['mean_absolute_error']:.3g}。")
    else:
        lines.append("1. 可用特征不足，未生成稳定排序。")
    lines.extend(["", "差异较大的归一化时间区间："])
    if top_ranges:
        for item in top_ranges:
            lines.append(f"- {item['start_percent']:.0f}% 至 {item['end_percent']:.0f}%，平均偏离 {item['mean_error']:.3g}。")
    else:
        lines.append("- 未检测到可排序区间。")
    lines.extend(
        [
            "",
            f"数据质量状态：{quality.get('status', 'UNKNOWN')}。",
            f"镜像规范化：{'已应用' if summary.get('mirror_canonicalization_applied') else '未应用'}。",
            "",
            "以上为相对参考动作的运动学差异，不代表动作是否正确、是否安全或是否适合个人。",
        ]
    )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

