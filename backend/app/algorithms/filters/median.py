"""Custom median filter using explicitly sorted local windows."""

from __future__ import annotations

import numpy as np

from .common import to_uint8, validate_kernel_size


def _median(values: list[int]) -> int:
    for index in range(1, len(values)):
        value = values[index]
        position = index - 1
        while position >= 0 and values[position] > value:
            values[position + 1] = values[position]
            position -= 1
        values[position + 1] = value
    return values[len(values) // 2]


def _median_channel(channel: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    mode = "reflect" if channel.shape[0] > 1 and channel.shape[1] > 1 else "edge"
    padded = np.pad(channel, ((radius, radius), (radius, radius)), mode=mode)
    result = np.empty_like(channel)
    for row in range(channel.shape[0]):
        for column in range(channel.shape[1]):
            window = padded[row : row + kernel_size, column : column + kernel_size]
            result[row, column] = _median(window.reshape(-1).tolist())
    return result


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    validate_kernel_size(kernel_size)
    source = np.asarray(image)
    if source.ndim == 2:
        return to_uint8(_median_channel(source, kernel_size))
    if source.ndim == 3 and source.shape[2] == 3:
        return to_uint8(np.stack([_median_channel(source[:, :, ch], kernel_size) for ch in range(3)], axis=2))
    raise ValueError("Image must be grayscale or RGB.")
