"""Custom median filter using local sliding windows."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .common import to_uint8, validate_kernel_size


def _median_channel(channel: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    mode = "reflect" if channel.shape[0] > 1 and channel.shape[1] > 1 else "edge"
    padded = np.pad(channel, ((radius, radius), (radius, radius)), mode=mode)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return np.median(windows, axis=(-2, -1))


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    validate_kernel_size(kernel_size)
    source = np.asarray(image)
    if source.ndim == 2:
        return to_uint8(_median_channel(source, kernel_size))
    if source.ndim == 3 and source.shape[2] == 3:
        return to_uint8(np.stack([_median_channel(source[:, :, ch], kernel_size) for ch in range(3)], axis=2))
    raise ValueError("Image must be grayscale or RGB.")
