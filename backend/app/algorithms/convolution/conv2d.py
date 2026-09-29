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
    if not np.all(np.isfinite(array)):
        raise ValueError("Kernel values must be finite.")
    return array


def _pad(image: np.ndarray, pad_before: tuple[int, int], pad_after: tuple[int, int]) -> np.ndarray:
    """Use reflection at boundaries; edge padding is a safe tiny-image fallback."""
    mode = "reflect" if image.shape[0] > 1 and image.shape[1] > 1 else "edge"
    return np.pad(
        image,
        ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1])),
        mode=mode,
    )


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply mathematical convolution and return float32 values.

    The kernel is flipped vertically and horizontally, so this is mathematical
    convolution rather than cross-correlation. The output keeps the input size.
    Reflection padding avoids introducing artificial black borders. Odd kernels
    are centred symmetrically. Even kernels use a top-left anchor with the
    remaining padding placed on the bottom and right; this supports the 2x2
    Roberts operator while preserving the input dimensions.
    No clipping or uint8 conversion is performed here; callers decide how to
    post-process the numerical result.
    """
    image_array = _validate_image(image)
    kernel_array = _validate_kernel(kernel)
    flipped_kernel = np.flip(kernel_array, axis=(0, 1))
    kh, kw = flipped_kernel.shape
    pad_top = kh // 2 if kh % 2 else kh // 2 - 1
    pad_left = kw // 2 if kw % 2 else kw // 2 - 1
    pad_bottom = kh - 1 - pad_top
    pad_right = kw - 1 - pad_left

    def convolve_channel(channel: np.ndarray) -> np.ndarray:
        padded = _pad(channel, (pad_top, pad_left), (pad_bottom, pad_right))
        output = np.zeros(channel.shape, dtype=np.float32)
        for row in range(kh):
            for col in range(kw):
                output += padded[row : row + channel.shape[0], col : col + channel.shape[1]] * flipped_kernel[row, col]
        return output

    if image_array.ndim == 2:
        return convolve_channel(image_array)

    return np.stack([convolve_channel(image_array[:, :, channel]) for channel in range(3)], axis=2)
