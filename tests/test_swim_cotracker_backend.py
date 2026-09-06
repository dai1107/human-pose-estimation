from __future__ import annotations

import numpy as np
import pytest

from src.swimming.cotracker_backend import CoTrackerOfflineBackend
from src.swimming.wrist_tracking import SwimWristTrackerConfig


class _FakeCoTracker:
    def __init__(self) -> None:
        self.video_shape = None
        self.queries = None

    def to(self, device: str):
        return self

    def eval(self):
        return self

    def __call__(self, video, *, queries):
        import torch

        self.video_shape = tuple(video.shape)
        self.queries = queries.detach().cpu().numpy()
        frames = video.shape[1]
        points = queries.shape[1]
        xy = queries[:, :, 1:].unsqueeze(1).repeat(1, frames, 1, 1)
        visibility = torch.ones(
            (1, frames, points), dtype=torch.float32, device=video.device
        )
        return xy, visibility


def test_backend_reports_missing_package_or_checkpoint_without_download() -> None:
    backend = CoTrackerOfflineBackend(allow_torch_hub_download=False)

    assert backend.availability.available is False
    assert (
        backend.availability.reason.startswith("torch_unavailable:")
        or backend.availability.reason
        in {
            "cotracker_package_and_weights_unavailable",
            "cotracker_checkpoint_missing",
        }
    )
    result = backend.track_window(
        [np.zeros((32, 32, 3), dtype=np.uint8)],
        np.asarray([[0.0, 10.0, 10.0]], dtype=np.float32),
    )
    assert result.available is False


def test_injected_model_tracks_scaled_queries_and_returns_normalized_points() -> None:
    pytest.importorskip("torch", reason="injected CoTracker model requires torch")
    model = _FakeCoTracker()
    config = SwimWristTrackerConfig(cotracker_maximum_width=100)
    backend = CoTrackerOfflineBackend(config, model=model, device="cpu")
    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(4)]
    queries = np.asarray([[0.0, 100.0, 50.0]], dtype=np.float32)

    result = backend.track_window(frames, queries)

    assert result.available is True
    assert result.reason == "accepted"
    assert model.video_shape == (1, 4, 3, 50, 100)
    assert model.queries[0, 0] == pytest.approx([0.0, 50.0, 25.0])
    assert result.tracks_normalized.shape == (4, 1, 2)
    assert result.tracks_normalized[:, 0] == pytest.approx(
        np.asarray([[0.5, 0.5]] * 4)
    )
    assert np.all(result.visibility == 1.0)


def test_backend_rejects_inconsistent_frame_shapes() -> None:
    backend = CoTrackerOfflineBackend(model=_FakeCoTracker(), device="cpu")

    with pytest.raises(ValueError, match="same shape"):
        backend.track_window(
            [
                np.zeros((32, 32, 3), dtype=np.uint8),
                np.zeros((40, 32, 3), dtype=np.uint8),
            ],
            np.asarray([[0.0, 10.0, 10.0]], dtype=np.float32),
        )
