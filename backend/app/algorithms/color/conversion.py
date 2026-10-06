"""RGB and grayscale conversion algorithms."""

from __future__ import annotations

import numpy as np


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to uint8 grayscale using luminance weights."""
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("RGB input must have shape (H, W, 3).")
    rgb = values.astype(np.float64, copy=False)
    grayscale = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.rint(np.clip(grayscale, 0, 255)).astype(np.uint8)


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to RGB by repeating its channel three times."""
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("Grayscale input must have shape (H, W).")
    return np.repeat(values[:, :, np.newaxis], 3, axis=2).astype(np.uint8, copy=False)
