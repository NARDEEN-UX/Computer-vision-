"""Custom Gaussian-kernel generation and filtering."""

from __future__ import annotations

import numpy as np

from ..convolution.conv2d import conv2d
from .common import to_uint8, validate_kernel_size


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    validate_kernel_size(kernel_size)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    x, y = np.meshgrid(coordinates, coordinates)
    kernel = np.exp(-((x * x) + (y * y)) / (2 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    return to_uint8(conv2d(image, gaussian_kernel(kernel_size, sigma)))
