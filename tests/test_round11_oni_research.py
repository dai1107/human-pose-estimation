from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.configuration import ConfigValidationError
from src.doctor import run_checks
from tools.dataset.round11_oni_research import (
    load_oni_research_contract,
    propose_subject_track,
    sample_frame_indices,
)
from tools.run_round11_oni_research import build_parser


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "contracts" / "oni_research_v1.yaml"


def test_round11_contract_keeps_modalities_independent_and_pairing_forbidden() -> None:
    contract = load_oni_research_contract(CONTRACT)

    assert contract.mode == "offline_oni"
    assert contract.require_independent_depth_ir_tracking is True
    assert contract.require_human_identity_confirmation is True
    assert contract.allow_rgb_depth_registration is False
    assert contract.allow_phone_oni_pairing is False
    assert contract.allow_phone_frame_labels is False
    assert contract.allow_ir_as_rgb is False


def test_round11_contract_rejects_enabling_phone_pairing(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        CONTRACT.read_text(encoding="utf-8").replace(
            "allow_phone_oni_pairing: false",
            "allow_phone_oni_pairing: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="prohibited capability"):
        load_oni_research_contract(path)


def test_sample_indices_cover_first_and_last_without_duplicates() -> None:
    assert sample_frame_indices(5, 24) == [0, 1, 2, 3, 4]
    indices = sample_frame_indices(100, 24)
    assert indices[0] == 0
    assert indices[-1] == 99
    assert len(indices) == len(set(indices)) == 24


def _timeline(count: int) -> list[dict[str, int]]:
    return [
        {
            "output_frame": index,
            "source_frame_index": index + 1,
            "timestamp_us": index * 33333,
        }
        for index in range(count)
    ]


def test_depth_track_is_a_review_proposal_and_metric_scope_is_limited() -> None:
    contract = load_oni_research_contract(CONTRACT)
    frames = np.full((4, 96, 128), 4000, dtype=np.uint16)
    for index in range(4):
        x = 20 + index * 8
        frames[index, 20:80, x : x + 28] = 1800

    proposals = propose_subject_track(
        frames,
        modality="depth",
        sampled_output_frames=[0, 1, 2, 3],
        timeline=_timeline(4),
        contract=contract,
    )

    assert len(proposals) == 4
    assert all(item["human_confirmed"] is False for item in proposals)
    assert any(item["bbox_px"] is not None for item in proposals)
    measured = [
        item for item in proposals if item["metric_surface_distance_m"] is not None
    ]
    assert measured
    assert measured[0]["metric_surface_distance_m"] == pytest.approx(1.8)
    assert (
        measured[0]["metric_surface_distance_scope"]
        == "oni_surface_line_of_sight_not_body_joint_or_ground_distance"
    )


def test_ir_track_never_emits_metric_depth_or_implies_registration() -> None:
    contract = load_oni_research_contract(CONTRACT)
    frames = np.full((4, 96, 128), 1000, dtype=np.uint16)
    for index in range(4):
        x = 20 + index * 8
        frames[index, 20:80, x : x + 28] = 8000

    proposals = propose_subject_track(
        frames,
        modality="ir",
        sampled_output_frames=[0, 1, 2, 3],
        timeline=_timeline(4),
        contract=contract,
    )

    assert all(item["modality"] == "ir" for item in proposals)
    assert all(item["metric_surface_distance_m"] is None for item in proposals)
    assert all(item["metric_surface_distance_scope"] == "not_applicable_ir" for item in proposals)
    assert all("depth" not in item["target_track_id"] for item in proposals)


def test_round11_cli_defaults_to_real_contract_and_previews() -> None:
    args = build_parser().parse_args([])

    assert args.contract == Path("configs/contracts/oni_research_v1.yaml")
    assert args.no_previews is False


def test_doctor_validates_round11_safety_contract() -> None:
    checks = {item.name: item for item in run_checks(project_root=ROOT)}

    check = checks["config:round11-oni-research"]
    assert check.passed is True
    assert "phone-ONI" in check.message
