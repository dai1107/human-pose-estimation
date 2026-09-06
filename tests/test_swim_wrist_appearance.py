from __future__ import annotations

import numpy as np
import pytest

from src.swimming.wrist_appearance import (
    WristAppearanceBank,
    WristAppearanceModel,
    extract_wrist_appearance,
)
from src.swimming.wrist_tracking import SwimWristTrackerConfig


def _frame(color: tuple[int, int, int]) -> np.ndarray:
    frame = np.full((120, 160, 3), (80, 90, 100), dtype=np.uint8)
    yy, xx = np.ogrid[:120, :160]
    mask = (xx - 80) ** 2 + (yy - 60) ** 2 <= 14**2
    frame[mask] = color
    return frame


def _descriptor(color: tuple[int, int, int], *, elbow_x: float = 0.25):
    return extract_wrist_appearance(
        _frame(color),
        wrist=(0.5, 0.5),
        elbow=(elbow_x, 0.5),
        roi_forearm_ratio=0.25,
    )


def test_descriptor_uses_forearm_scaled_roi_and_hsv_lab_histograms() -> None:
    short = _descriptor((10, 10, 180), elbow_x=0.35)
    long = _descriptor((10, 10, 180), elbow_x=0.15)

    assert short is not None and long is not None
    assert short.vector.shape == (56,)
    assert np.sum(short.vector) == pytest.approx(1.0)
    assert long.radius_px > short.radius_px
    assert short.sample_count > 20


def test_appearance_cost_prefers_matching_color_distribution() -> None:
    config = SwimWristTrackerConfig()
    model = WristAppearanceModel("left", config)
    red = _descriptor((10, 10, 190))
    blue = _descriptor((190, 10, 10))
    assert red is not None and blue is not None
    model.update(red, identity_confidence=0.95, visibility=0.95)

    assert model.cost(red) == pytest.approx(0.0, abs=1e-9)
    assert model.cost(red) < model.cost(blue)


def test_low_identity_or_visibility_never_contaminates_ema() -> None:
    model = WristAppearanceModel("left", SwimWristTrackerConfig())
    red = _descriptor((10, 10, 190))
    blue = _descriptor((190, 10, 10))
    assert red is not None and blue is not None
    initialized = model.update(red, identity_confidence=0.95, visibility=0.95)
    prototype = model.prototype.copy()

    low_identity = model.update(
        blue, identity_confidence=0.40, visibility=0.95
    )
    low_visibility = model.update(
        blue, identity_confidence=0.95, visibility=0.40
    )

    assert initialized.updated is True
    assert low_identity.reason == "identity_confidence_low"
    assert low_visibility.reason == "wrist_visibility_low"
    assert model.update_count == 1
    assert np.array_equal(model.prototype, prototype)


def test_bank_produces_candidate_to_anatomical_track_costs() -> None:
    bank = WristAppearanceBank()
    red = _descriptor((10, 10, 190))
    blue = _descriptor((190, 10, 10))
    assert red is not None and blue is not None
    bank.update("left", red, identity_confidence=0.95, visibility=0.95)
    bank.update("right", blue, identity_confidence=0.95, visibility=0.95)

    costs = bank.costs({"left": blue, "right": red})

    assert costs[("right", "left")] < costs[("left", "left")]
    assert costs[("left", "right")] < costs[("right", "right")]

