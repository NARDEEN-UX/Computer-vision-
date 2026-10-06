"""Histogram equalization for grayscale and RGB images."""

from __future__ import annotations

import numpy as np

from ..histogram.histogram import compute_cdf, compute_distribution, compute_histogram

# INTENSITY_LEVELS since we are using uint8
INTENSITY_LEVELS = 256

def equalize_channel(channel: np.ndarray) -> np.ndarray:
    """Equalize one 2-D image channel using its cumulative distribution."""
    values = np.asarray(channel)
    if values.ndim != 2:
        raise ValueError("An image channel must be a 2-D array.")
    if values.size == 0:
        raise ValueError("The image channel cannot be empty.")

    values = values.astype(np.int64, copy=False)
    if int(values.min()) < 0 or int(values.max()) > INTENSITY_LEVELS - 1:
        raise ValueError("Image values must be between 0 and {}.".format(INTENSITY_LEVELS - 1))

    counts = compute_histogram(values)
    pdf = compute_distribution(counts)
    cdf = compute_cdf(pdf)
    nonzero_cdf = cdf[cdf > 0]

    if nonzero_cdf.size == 0 or nonzero_cdf[-1] == nonzero_cdf[0]:
        return values.astype(np.uint8, copy=True)

    lookup = np.rint(cdf * (INTENSITY_LEVELS - 1))
    lookup = np.clip(lookup, 0, INTENSITY_LEVELS-1).astype(np.uint8)

    return lookup[values]


def equalize_image(image: np.ndarray) -> np.ndarray:
    """Equalize a grayscale image or each channel of an RGB image."""
    values = np.asarray(image)
    if values.ndim == 2:
        return equalize_channel(values)
    if values.ndim == 3 and values.shape[2] == 3:
        channels = [equalize_channel(values[:, :, index]) for index in range(3)]
        return np.stack(channels, axis=2)
    raise ValueError("Image must be grayscale (H, W) or RGB (H, W, 3).")