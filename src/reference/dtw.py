from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DtwResult:
    total_distance: float
    normalized_distance: float
    path: list[tuple[int, int]]
    per_feature_error: dict[str, float]
    aligned_reference: np.ndarray
    aligned_candidate: np.ndarray


def _weights_array(
    feature_count: int,
    feature_weights: Mapping[str, float] | list[float] | np.ndarray | None,
    feature_names: list[str] | None,
) -> np.ndarray:
    if feature_weights is None:
        return np.ones(feature_count, dtype=float)
    if isinstance(feature_weights, Mapping):
        return np.array([float(feature_weights.get(name, 1.0)) for name in (feature_names or [])], dtype=float)
    weights = np.asarray(feature_weights, dtype=float)
    if weights.size != feature_count:
        return np.ones(feature_count, dtype=float)
    return weights


def _pair_cost(reference_row: np.ndarray, candidate_row: np.ndarray, weights: np.ndarray) -> float:
    finite_mask = np.isfinite(reference_row) & np.isfinite(candidate_row) & np.isfinite(weights) & (weights > 0)
    if not finite_mask.any():
        return 0.0
    diff = reference_row[finite_mask] - candidate_row[finite_mask]
    active_weights = weights[finite_mask]
    return float(np.sqrt(np.sum(active_weights * diff * diff) / np.sum(active_weights)))


def constrained_dtw(
    reference_features: np.ndarray,
    candidate_features: np.ndarray,
    window_ratio: float = 0.15,
    feature_weights: Mapping[str, float] | list[float] | np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> DtwResult:
    reference = np.asarray(reference_features, dtype=float)
    candidate = np.asarray(candidate_features, dtype=float)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    if candidate.ndim == 1:
        candidate = candidate.reshape(-1, 1)
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("reference and candidate feature dimensions must match")

    n, feature_count = reference.shape
    m = candidate.shape[0]
    names = feature_names or [f"feature_{index}" for index in range(feature_count)]
    if n == 0 or m == 0:
        empty = np.empty((0, feature_count), dtype=float)
        return DtwResult(float("inf"), float("inf"), [], {name: float("nan") for name in names}, empty, empty)

    weights = _weights_array(feature_count, feature_weights, names)
    window = max(abs(n - m), int(ceil(max(n, m) * max(0.0, window_ratio))), 1)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = _pair_cost(reference[i - 1], candidate[j - 1], weights)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    if not isfinite(float(dp[n, m])):
        return constrained_dtw(reference, candidate, window_ratio=1.0, feature_weights=feature_weights, feature_names=names)

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = (dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
        step = int(np.argmin(choices))
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    while i > 0:
        path.append((i - 1, 0))
        i -= 1
    while j > 0:
        path.append((0, j - 1))
        j -= 1
    path.reverse()

    aligned_reference = np.array([reference[i] for i, _ in path], dtype=float)
    aligned_candidate = np.array([candidate[j] for _, j in path], dtype=float)
    per_feature_error: dict[str, float] = {}
    for index, name in enumerate(names):
        ref_values = aligned_reference[:, index]
        cand_values = aligned_candidate[:, index]
        finite_mask = np.isfinite(ref_values) & np.isfinite(cand_values)
        if finite_mask.any():
            per_feature_error[name] = float(np.mean(np.abs(ref_values[finite_mask] - cand_values[finite_mask])))
        else:
            per_feature_error[name] = float("nan")

    total = float(dp[n, m])
    return DtwResult(
        total_distance=total,
        normalized_distance=total / max(len(path), 1),
        path=path,
        per_feature_error=per_feature_error,
        aligned_reference=aligned_reference,
        aligned_candidate=aligned_candidate,
    )

