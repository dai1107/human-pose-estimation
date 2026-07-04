from __future__ import annotations

import json
import os
from math import isfinite
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".cache" / "matplotlib"))

import numpy as np

from src.reference.session_loader import write_csv_rows
from src.utils.time_utils import make_session_id, now_iso

from .arm_alignment import compute_arm_alignment_proxy
from .chain_features import detect_chain_events, extract_chain_feature_rows
from .config import load_basketball_config
from .phase_detector import detect_shot_phases
from .reference_compare import compare_shot_to_reference
from .release_proxy import estimate_release_proxy, release_event
from .schema import ShotClip, ShotEvent, unique_output_dir
from .sequence_analyzer import analyze_event_sequence
from .shot_clipper import clip_shot_session, detect_shot_candidates


LIMITATIONS = [
    "本报告基于单目视频人体关键点与运动学代理指标生成。",
    "它不能直接测量地面反作用力、真实关节力矩、肌肉发力、球离手的精确时刻、投篮命中率、医学风险或投篮技术是否绝对标准。",
    "release_proxy_time 是人体关键点代理估计；只有未来接入并验证篮球检测后，才可标记 BALL_RELEASE_CANDIDATE。",
]


def analyze_shot_session(
    session_dir: str | Path,
    shot_type: str,
    shooting_side: str,
    camera_view: str,
    output_dir: str | Path = "outputs/basketball/reports",
    start_ms: int | None = None,
    end_ms: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    release_ms: int | None = None,
    reference_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    config = load_basketball_config(config_path)
    if start_ms is None and end_ms is None and start_frame is None and end_frame is None:
        candidates = detect_shot_candidates(session_dir, shooting_side)
        if candidates:
            start_ms, end_ms = candidates[0].start_ms, candidates[0].end_ms
    clip = clip_shot_session(session_dir, shooting_side, start_ms=start_ms, end_ms=end_ms, start_frame=start_frame, end_frame=end_frame)
    feature_rows = extract_chain_feature_rows(clip.frames)
    release = estimate_release_proxy(clip.frames, feature_rows, manual_release_ms=release_ms, config=config.get("release_proxy", {}))
    phases = detect_shot_phases(clip.frames, release, config.get("phase_detection", {}))
    events = detect_chain_events(clip.frames, feature_rows, shooting_side)
    events.append(release_event(release, clip.frames))
    sequence = analyze_event_sequence(events, config, shooting_side)
    arm_alignment = compute_arm_alignment_proxy(clip.frames, camera_view)

    report_dir = unique_output_dir(output_dir, make_session_id())
    report_dir.mkdir(parents=True, exist_ok=False)
    reference_comparison = None
    if reference_dir:
        reference_comparison = compare_shot_to_reference(feature_rows, reference_dir, report_dir)

    metadata = {
        "report_id": report_dir.name,
        "created_at": now_iso(),
        "session_id": clip.session_id,
        "shot_type": shot_type,
        "shooting_side": shooting_side,
        "camera_view": camera_view,
        "analysis_name": config.get("analysis_name", "basketball_shot_v1"),
    }
    summary = {
        "shot_id": report_dir.name,
        "shot_type": shot_type,
        "shooting_side": shooting_side,
        "camera_view": camera_view,
        "clip_range": clip.to_range_dict(),
        "phase_timestamps": phases.phase_timestamps,
        "release_proxy": release.to_dict(),
        "data_quality": _data_quality(clip.frames),
        "event_sequence": sequence,
        "kinematic_peaks": {event.event: event.to_dict() for event in events},
        "reference_comparison": reference_comparison,
        "consistency_metrics": None,
        "arm_alignment": arm_alignment,
        "limitations": LIMITATIONS,
    }
    (report_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "shot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "chain_sequence.json").write_text(json.dumps(sequence, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv_rows(report_dir / "shot_events.csv", [_event_row(event) for event in events])
    write_csv_rows(report_dir / "shot_features.csv", [_format_feature_row(row) for row in feature_rows])
    _plot_phase_timeline(report_dir / "phase_timeline.png", clip.frames, phases.phase_by_frame)
    _plot_angle_curves(report_dir / "angle_curves.png", clip.frames)
    _plot_velocity_curves(report_dir / "velocity_curves.png", feature_rows)
    _plot_event_sequence(report_dir / "event_sequence.png", events)
    _plot_arm_path(report_dir / "arm_path.png", clip.frames, camera_view)
    _write_keyframes(report_dir, clip, phases.phase_timestamps, release.release_proxy_time)
    _write_markdown(report_dir / "report.md", summary)
    return report_dir


def _data_quality(frames) -> dict[str, Any]:
    total = len(frames)
    valid = sum(1 for frame in frames if frame.pose_detected)
    visibility = [frame.visibility_mean for frame in frames if frame.visibility_mean is not None]
    ratio = valid / total if total else 0.0
    mean_visibility = sum(visibility) / len(visibility) if visibility else 0.0
    return {
        "pose_valid_ratio": ratio,
        "landmark_visibility_mean": mean_visibility,
        "level": "GOOD" if ratio >= 0.8 and mean_visibility >= 0.65 else ("WARNING" if ratio >= 0.5 else "LOW"),
    }


def _event_row(event: ShotEvent) -> dict[str, Any]:
    return {
        "event": event.event,
        "timestamp_ms": event.timestamp_ms if event.timestamp_ms is not None else "",
        "normalized_time_percent": _num(event.normalized_time_percent),
        "confidence": _num(event.confidence),
        "source_signal": event.source_signal,
        "data_quality": event.data_quality,
    }


def _format_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _num(value) if isinstance(value, (int, float)) or value is None else value for key, value in row.items()}


def _num(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.8g}" if isfinite(number) else ""


def _times(frames) -> list[float]:
    if not frames:
        return []
    first = frames[0].timestamp_ms
    return [(frame.timestamp_ms - first) / 1000.0 for frame in frames]


def _plot_phase_timeline(path: Path, frames, phase_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = _times(frames)
    phase_order = {phase: index for index, phase in enumerate(("IDLE", "SETUP", "DIP", "RISE", "ARM_EXTENSION", "RELEASE_PROXY", "FOLLOW_THROUGH", "RECOVERY"))}
    values = [phase_order.get(row["phase"], 0) for row in phase_rows]
    plt.figure(figsize=(10, 4))
    plt.step(times[: len(values)], values, where="post")
    plt.yticks(list(phase_order.values()), list(phase_order.keys()), fontsize=8)
    plt.xlabel("time (s)")
    plt.title("Shot phase timeline")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_angle_curves(path: Path, frames) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = _times(frames)
    plt.figure(figsize=(10, 5))
    for name, values in {
        "shooting knee": [frame.shooting_knee_angle for frame in frames],
        "shooting hip": [frame.shooting_hip_angle for frame in frames],
        "shooting shoulder": [frame.shooting_shoulder_angle for frame in frames],
        "shooting elbow": [frame.shooting_elbow_angle for frame in frames],
        "trunk tilt": [frame.trunk_tilt_proxy for frame in frames],
    }.items():
        plt.plot(times, values, label=name)
    plt.title("Shot angle curves")
    plt.xlabel("time (s)")
    plt.ylabel("angle / proxy")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_velocity_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    first = float(rows[0]["timestamp_ms"])
    times = [(float(row["timestamp_ms"]) - first) / 1000.0 for row in rows]
    plt.figure(figsize=(10, 5))
    for name in ("shooting_knee_angular_velocity", "shooting_hip_angular_velocity", "shooting_elbow_angular_velocity", "shooting_wrist_speed", "pelvis_vertical_velocity_proxy"):
        plt.plot(times, [row.get(name) for row in rows], label=name)
    plt.title("Shot velocity proxy curves")
    plt.xlabel("time (s)")
    plt.ylabel("proxy units / s")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_event_sequence(path: Path, events: list[ShotEvent]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [event for event in events if event.timestamp_ms is not None]
    if not available:
        return
    first = min(event.timestamp_ms for event in available if event.timestamp_ms is not None)
    plt.figure(figsize=(10, 4))
    for index, event in enumerate(available):
        plt.scatter((event.timestamp_ms - first) / 1000.0, index)
        plt.text((event.timestamp_ms - first) / 1000.0, index, event.event, fontsize=8)
    plt.title("Kinematic event sequence")
    plt.xlabel("time from first event (s)")
    plt.yticks([])
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_arm_path(path: Path, frames, camera_view: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [frame.shooting_wrist_x for frame in frames if frame.shooting_wrist_x is not None and frame.shooting_wrist_y is not None]
    ys = [frame.shooting_wrist_y for frame in frames if frame.shooting_wrist_x is not None and frame.shooting_wrist_y is not None]
    plt.figure(figsize=(5, 5))
    if xs and ys:
        plt.plot(xs, ys, marker="o", markersize=3)
        plt.gca().invert_yaxis()
    plt.title(f"Arm path proxy ({camera_view})")
    plt.xlabel("x proxy")
    plt.ylabel("y proxy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _write_keyframes(report_dir: Path, clip: ShotClip, phase_timestamps: dict[str, int | None], release_ms: int | None) -> None:
    key_dir = report_dir / "keyframes"
    key_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "setup.png": phase_timestamps.get("SETUP"),
        "dip.png": phase_timestamps.get("DIP"),
        "rise.png": phase_timestamps.get("RISE"),
        "arm_extension.png": phase_timestamps.get("ARM_EXTENSION"),
        "release_proxy.png": release_ms or phase_timestamps.get("RELEASE_PROXY"),
        "follow_through.png": phase_timestamps.get("FOLLOW_THROUGH"),
    }
    for filename, timestamp in mapping.items():
        _write_keyframe_plot(key_dir / filename, clip.frames, timestamp, filename.removesuffix(".png"))


def _write_keyframe_plot(path: Path, frames, timestamp_ms: int | None, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = _times(frames)
    wrist = [frame.shooting_wrist_speed for frame in frames]
    plt.figure(figsize=(5, 4))
    if times:
        plt.plot(times, wrist, label="wrist speed")
        if timestamp_ms is not None:
            plt.axvline((timestamp_ms - frames[0].timestamp_ms) / 1000.0, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("proxy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    release = summary["release_proxy"]
    sequence = summary["event_sequence"]
    lines = [
        "# 篮球投篮专项分析报告",
        "",
        f"投篮类型：{summary['shot_type']}",
        f"投篮侧：{summary['shooting_side']}",
        f"机位：{summary['camera_view']}",
        "",
        f"出手代理时刻：{release.get('release_proxy_time') if release.get('release_proxy_time') is not None else 'N/A'} ms",
        f"出手代理置信度：{release.get('release_proxy_confidence'):.2f}",
        f"依据：{release.get('release_proxy_reason')}",
        "",
        "EVENT SEQUENCE",
        "",
    ]
    for item in sequence.get("pairwise_timing", []):
        lines.append(f"- {item['from']} -> {item['to']}: {item['delta_ms']} ms ({item['status']})")
    if sequence.get("missing_events"):
        lines.append(f"- 缺失事件：{', '.join(sequence['missing_events'])}")
    lines.extend(
        [
            "",
            "与个人参考投篮相比：" if summary.get("reference_comparison") else "未提供个人参考投篮动作。",
        ]
    )
    ref = summary.get("reference_comparison")
    if ref and ref.get("top_difference_features"):
        for item in ref["top_difference_features"][:4]:
            lines.append(f"- {item['feature']} 平均差异 {item['mean_absolute_error']:.3g}。")
        lines.append("该报告描述的是与指定参考动作之间的运动学差异，不代表投篮动作正确性、命中概率或真实发力质量。")
    lines.extend(["", "固定限制说明：", "本报告基于单目视频人体关键点与运动学代理指标生成。", ""])
    lines.extend(
        [
            "它不能直接测量：",
            "- 地面反作用力；",
            "- 真实关节力矩；",
            "- 肌肉发力；",
            "- 球离手的精确时刻（除非未来接入并验证篮球检测）；",
            "- 投篮命中率；",
            "- 医学风险；",
            "- 投篮技术是否绝对标准。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
