"""Transparent 2-D mathematical convolution for grayscale and RGB images."""

from __future__ import annotations

import numpy as np


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("Image must be grayscale (H, W) or RGB (H, W, 3).")
    if array.ndim == 3 and array.shape[2] != 3:
        raise ValueError("Only grayscale and RGB images are supported.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Image dimensions must be greater than zero.")
    return array.astype(np.float32, copy=False)


def _validate_kernel(kernel: np.ndarray) -> np.ndarray:
    array = np.asarray(kernel, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Kernel must be a non-empty 2-D array.")
    if array.shape[0] % 2 == 0 or array.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be positive odd numbers.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Kernel values must be finite.")
    return array


def _pad(image: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Use reflection at boundaries; edge padding is a safe 1-pixel fallback."""
    mode = "reflect" if image.shape[0] > 1 and image.shape[1] > 1 else "edge"
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply mathematical convolution and return float32 values.

    The kernel is flipped vertically and horizontally, so this is mathematical
    convolution rather than cross-correlation. The output keeps the input size.
    Reflection padding avoids introducing artificial black borders. Filter
    kernels in this project are odd-sized, which gives an unambiguous centre.
    No clipping or uint8 conversion is performed here; callers decide how to
    post-process the numerical result.
    """
    image_array = _validate_image(image)
    kernel_array = _validate_kernel(kernel)
    flipped_kernel = np.flip(kernel_array, axis=(0, 1))
    kh, kw = flipped_kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    def convolve_channel(channel: np.ndarray) -> np.ndarray:
        padded = _pad(channel, pad_h, pad_w)
        output = np.zeros(channel.shape, dtype=np.float32)
        for row in range(kh):
            for col in range(kw):
                output += padded[row : row + channel.shape[0], col : col + channel.shape[1]] * flipped_kernel[row, col]
        return output

    if image_array.ndim == 2:
        return convolve_channel(image_array)

    return np.stack([convolve_channel(image_array[:, :, channel]) for channel in range(3)], axis=2)
