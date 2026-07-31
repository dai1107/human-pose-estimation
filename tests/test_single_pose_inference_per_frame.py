from __future__ import annotations

import pytest

from webui.upload_pipeline import UploadInferenceAudit


def test_upload_inference_audit_accepts_exactly_one_inference_per_frame() -> None:
    audit = UploadInferenceAudit()
    audit.record_model_initialization()
    for frame_index in range(4):
        audit.record_decoded(frame_index)
        audit.record_inference(frame_index)
        audit.record_analyzed(frame_index)

    audit.validate_complete()

    assert audit.as_dict() == {
        "decoded_frame_count": 4,
        "analyzed_frame_count": 4,
        "pose_inference_count": 4,
        "model_initialization_count": 1,
        "single_inference_per_frame": True,
    }


def test_upload_inference_audit_rejects_duplicate_frame_inference() -> None:
    audit = UploadInferenceAudit()
    audit.record_model_initialization()
    audit.record_decoded(0)
    audit.record_inference(0)

    with pytest.raises(RuntimeError, match="重复运行姿态推理"):
        audit.record_inference(0)


def test_upload_inference_audit_rejects_reinitializing_model() -> None:
    audit = UploadInferenceAudit()
    audit.record_model_initialization()

    with pytest.raises(RuntimeError, match="重复初始化"):
        audit.record_model_initialization()


def test_upload_inference_audit_requires_inference_for_each_analyzed_frame() -> None:
    audit = UploadInferenceAudit()
    audit.record_model_initialization()
    audit.record_decoded(0)
    audit.record_analyzed(0)

    with pytest.raises(RuntimeError, match="推理次数与分析帧数不一致"):
        audit.validate_complete()
