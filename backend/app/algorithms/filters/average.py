"""Custom normalized box/average filter."""

from __future__ import annotations

import numpy as np

from ..convolution.conv2d import conv2d
from .common import to_uint8, validate_kernel_size


def average_kernel(kernel_size: int) -> np.ndarray:
    validate_kernel_size(kernel_size)
    return np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)


def average_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return to_uint8(conv2d(image, average_kernel(kernel_size)))
