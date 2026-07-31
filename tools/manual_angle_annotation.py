from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.angle_validation import (
    append_annotation,
    build_manual_annotation,
    find_observation,
    joint_point_names,
    load_report,
    normalize_joint_name,
    read_video_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually click three joint points on one video frame."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--joint", required=True, help="Example: left_knee")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--camera-view", default="side")
    parser.add_argument(
        "--event",
        choices=("lowest_point", "full_extension"),
        default="",
    )
    parser.add_argument("--annotator", default="")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/angle_validation/manual_angles.json"),
    )
    parser.add_argument(
        "--points",
        help=(
            "Non-interactive pixel points: 'x1,y1;x2,y2;x3,y3'. "
            "Order follows the selected joint definition."
        ),
    )
    return parser


def parse_points(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in value.split(";"):
        fields = item.split(",")
        if len(fields) != 2:
            raise ValueError("each point must use x,y")
        points.append((float(fields[0]), float(fields[1])))
    if len(points) != 3:
        raise ValueError("--points requires exactly three x,y pairs")
    return points


def collect_points(
    video: Path,
    *,
    frame_index: int,
    joint: str,
    observation: Mapping[str, object] | None = None,
) -> list[tuple[float, float]]:
    frame = read_video_frame(video, frame_index)
    labels = [
        name.removeprefix("left_").removeprefix("right_")
        for name in joint_point_names(joint)
    ]
    points: list[tuple[float, float]] = []
    window = "Manual joint angle: click A, B(vertex), C"

    def mouse_callback(
        event: int,
        x: int,
        y: int,
        _flags: int,
        _data: object,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((float(x), float(y)))

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse_callback)
        while True:
            display = frame.copy()
            for index, (x, y) in enumerate(points):
                point = (int(round(x)), int(round(y)))
                cv2.circle(display, point, 5, (70, 220, 110), -1)
                cv2.putText(
                    display,
                    labels[index],
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (70, 220, 110),
                    2,
                    cv2.LINE_AA,
                )
            if len(points) >= 2:
                for start, end in zip(points, points[1:]):
                    cv2.line(
                        display,
                        tuple(int(round(value)) for value in start),
                        tuple(int(round(value)) for value in end),
                        (70, 220, 110),
                        2,
                        cv2.LINE_AA,
                    )
            prompt = (
                f"Frame {frame_index} | click {labels[len(points)]}"
                if len(points) < 3
                else "S/Enter: save | R: reset | Q/Esc: cancel"
            )
            cv2.putText(
                display,
                prompt,
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if observation is not None:
                values = (
                    ("2D RAW", observation.get("angle_2d_raw_deg")),
                    ("2D SMOOTH", observation.get("angle_2d_smoothed_deg")),
                    ("3D RAW", observation.get("angle_3d_raw_deg")),
                    ("RULE", observation.get("rule_angle_deg")),
                )
                for index, (label, value) in enumerate(values):
                    rendered = (
                        f"{float(value):.1f} deg"
                        if isinstance(value, (int, float))
                        else "--"
                    )
                    cv2.putText(
                        display,
                        f"{label}: {rendered}",
                        (16, 58 + index * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (245, 210, 90),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("r"), ord("R")):
                points.clear()
            elif key in (ord("q"), ord("Q"), 27):
                raise RuntimeError("manual annotation cancelled")
            elif key in (ord("s"), ord("S"), 13) and len(points) == 3:
                return points
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI is unavailable; use --points for non-interactive input"
        ) from exc
    finally:
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    joint = normalize_joint_name(args.joint)
    report = load_report(args.report) if args.report else None
    observation = find_observation(
        report,
        frame_index=args.frame,
        joint=joint,
    )
    points = (
        parse_points(args.points)
        if args.points
        else collect_points(
            args.video,
            frame_index=args.frame,
            joint=joint,
            observation=observation,
        )
    )
    annotation = build_manual_annotation(
        video_path=args.video,
        frame_index=args.frame,
        joint=joint,
        camera_view=args.camera_view,
        points=points,
        report=report,
        event=args.event,
        annotator=args.annotator,
    )
    output = append_annotation(args.output, annotation)
    print(
        json.dumps(
            {
                "saved_to": str(output),
                "annotation": annotation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
