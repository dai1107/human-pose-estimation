from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UploadInferenceAudit:
    """Runtime guard for one-model, one-inference-per-analyzed-frame uploads."""

    sparse_analysis: bool = False
    max_model_initializations: int = 1
    allow_cross_pass_inference: bool = False
    decoded_frame_count: int = 0
    analyzed_frame_count: int = 0
    pose_inference_count: int = 0
    model_initialization_count: int = 0
    _inferred_frames: set[int] = field(default_factory=set, repr=False)
    _inferred_pass_frames: set[tuple[str, int]] = field(default_factory=set, repr=False)
    _analyzed_frames: set[int] = field(default_factory=set, repr=False)
    coarse_pose_frames: int = 0
    refinement_pose_frames: int = 0

    def record_model_initialization(self) -> None:
        self.model_initialization_count += 1
        if self.model_initialization_count > self.max_model_initializations:
            raise RuntimeError("上传视频的一次分析中姿态模型被重复初始化")

    def record_decoded(self, frame_index: int) -> None:
        expected = self.decoded_frame_count
        if int(frame_index) != expected:
            raise RuntimeError(
                f"上传视频解码帧索引不连续：expected={expected}, actual={frame_index}"
            )
        self.decoded_frame_count += 1

    def record_inference(self, frame_index: int, *, pass_name: str = "full") -> None:
        frame_index = int(frame_index)
        pass_key = (str(pass_name), frame_index)
        if pass_key in self._inferred_pass_frames:
            raise RuntimeError(f"同一上传视频帧重复运行姿态推理：frame={frame_index}")
        if frame_index in self._inferred_frames and not self.allow_cross_pass_inference:
            raise RuntimeError(f"same upload frame was inferred in multiple passes: frame={frame_index}")
        if frame_index >= self.decoded_frame_count:
            raise RuntimeError(
                f"姿态推理发生在解码之前：frame={frame_index}, decoded={self.decoded_frame_count}"
            )
        self._inferred_frames.add(frame_index)
        self._inferred_pass_frames.add(pass_key)
        self.pose_inference_count += 1
        if not self.allow_cross_pass_inference and self.pose_inference_count > self.decoded_frame_count:
            raise RuntimeError("姿态推理次数超过已解码帧数")

    def record_analyzed(self, frame_index: int, *, pass_name: str = "full") -> None:
        frame_index = int(frame_index)
        if frame_index in self._analyzed_frames:
            raise RuntimeError(
                f"same upload frame was analyzed twice: frame={frame_index}"
            )
        if not self.sparse_analysis and frame_index != self.analyzed_frame_count:
            raise RuntimeError(
                "上传视频分析帧索引不连续："
                f"expected={self.analyzed_frame_count}, actual={frame_index}"
            )
        if frame_index >= self.decoded_frame_count:
            raise RuntimeError(
                f"视频帧在解码前被标记为已分析：frame={frame_index}"
            )
        self._analyzed_frames.add(frame_index)
        self.analyzed_frame_count += 1
        if pass_name == "coarse":
            self.coarse_pose_frames += 1
        elif pass_name == "refinement":
            self.refinement_pose_frames += 1

    def validate_complete(self, *, inference_expected: bool = True) -> None:
        if not self.sparse_analysis and self.analyzed_frame_count != self.decoded_frame_count:
            raise RuntimeError(
                "上传视频分析未覆盖全部解码帧："
                f"decoded={self.decoded_frame_count}, analyzed={self.analyzed_frame_count}"
            )
        if not self.allow_cross_pass_inference and self.pose_inference_count > self.decoded_frame_count:
            raise RuntimeError("姿态推理次数超过解码帧数")
        if (
            inference_expected
            and not self.allow_cross_pass_inference
            and self.pose_inference_count != self.analyzed_frame_count
        ):
            raise RuntimeError(
                "逐帧上传分析的姿态推理次数与分析帧数不一致："
                f"inference={self.pose_inference_count}, analyzed={self.analyzed_frame_count}"
            )
        if inference_expected and self.model_initialization_count not in {
            1,
            self.max_model_initializations,
        }:
            raise RuntimeError(
                "上传视频分析必须且只能初始化一次姿态模型："
                f"actual={self.model_initialization_count}"
            )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "decoded_frame_count": self.decoded_frame_count,
            "analyzed_frame_count": self.analyzed_frame_count,
            "pose_inference_count": self.pose_inference_count,
            "model_initialization_count": self.model_initialization_count,
            "single_inference_per_frame": (
                self.pose_inference_count == len(self._inferred_frames)
            ),
        }
        if self.sparse_analysis:
            payload.update(
                {
                    "coarse_pose_frames": self.coarse_pose_frames,
                    "refinement_pose_frames": self.refinement_pose_frames,
                    "pose_frames": self.analyzed_frame_count,
                    "pose_sampling_ratio": (
                        round(
                            self.analyzed_frame_count / self.decoded_frame_count,
                            6,
                        )
                        if self.decoded_frame_count > 0
                        else 0.0
                    ),
                    "unique_pose_inference_frames": len(self._inferred_frames),
                    "cross_pass_reinference_count": max(
                        0,
                        self.pose_inference_count - len(self._inferred_frames),
                    ),
                }
            )
        return payload
