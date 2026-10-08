"""Hybrid image synthesis from a low-frequency image and a high-frequency image."""

from __future__ import annotations

import numpy as np

from ..color.conversion import rgb_to_grayscale
from ..convolution.conv2d import conv2d
from .gaussian import gaussian_kernel


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] == 3:
        return rgb_to_grayscale(array).astype(np.float32)
    raise ValueError("Image must be grayscale or RGB with three channels.")


def hybrid_image(image_a: np.ndarray, image_b: np.ndarray, sigma: float = 8.0, kernel_size: int = 15) -> np.ndarray:
    """Combine the low frequencies of image A with the high frequencies of image B."""
    a = _as_grayscale(image_a)
    b = _as_grayscale(image_b)

    if a.shape != b.shape:
        height = min(a.shape[0], b.shape[0])
        width = min(a.shape[1], b.shape[1])
        a = a[:height, :width]
        b = b[:height, :width]

    kernel = gaussian_kernel(kernel_size, sigma)
    low_a = conv2d(a, kernel)
    blurred_b = conv2d(b, kernel)
    high_b = b - blurred_b
    hybrid = low_a + high_b
    return np.clip(np.rint(hybrid), 0, 255).astype(np.uint8)
