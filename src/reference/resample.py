from __future__ import annotations

from math import isfinite

import numpy as np


def resample_sequence(
    features: np.ndarray,
    timestamps: np.ndarray | list[float],
    target_length: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    target_length = max(2, int(target_length))
    target_axis = np.linspace(0.0, 100.0, target_length, dtype=float)
    if matrix.shape[0] == 0:
        return np.zeros((target_length, matrix.shape[1] if matrix.ndim == 2 else 0), dtype=float), target_axis

    times = np.asarray(timestamps, dtype=float)
    if times.size != matrix.shape[0] or not np.isfinite(times).any():
        times = np.arange(matrix.shape[0], dtype=float)
    finite_time_mask = np.isfinite(times)
    times = times[finite_time_mask]
    matrix = matrix[finite_time_mask]
    if times.size == 0:
        return np.zeros((target_length, matrix.shape[1]), dtype=float), target_axis

    order = np.argsort(times)
    times = times[order]
    matrix = matrix[order]
    unique_times, unique_indices = np.unique(times, return_index=True)
    times = unique_times
    matrix = matrix[unique_indices]
    if times.size == 1 or not isfinite(float(times[-1] - times[0])) or times[-1] == times[0]:
        return np.repeat(matrix[:1], target_length, axis=0), target_axis

    source_axis = (times - times[0]) / (times[-1] - times[0]) * 100.0
    result = np.empty((target_length, matrix.shape[1]), dtype=float)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            result[:, column] = 0.0
        elif finite_mask.sum() == 1:
            result[:, column] = values[finite_mask][0]
        else:
            result[:, column] = np.interp(target_axis, source_axis[finite_mask], values[finite_mask])
    return result, target_axis

