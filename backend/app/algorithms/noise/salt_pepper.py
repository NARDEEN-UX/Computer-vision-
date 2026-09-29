"""Salt-and-pepper noise with exact pixel-count density control."""

from __future__ import annotations

import numpy as np


def salt_pepper_noise(
    image: np.ndarray,
    density: float = 0.05,
    salt_probability: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Replace a requested fraction of unique pixels with 0 or 255."""
    if not np.isfinite(density) or not 0 <= density <= 1:
        raise ValueError("Density must be between 0 and 1.")
    if not np.isfinite(salt_probability) or not 0 <= salt_probability <= 1:
        raise ValueError("Salt probability must be between 0 and 1.")

    source = np.asarray(image)
    if source.ndim not in (2, 3) or (source.ndim == 3 and source.shape[2] != 3):
        raise ValueError("Image must be grayscale or RGB.")

    result = source.copy().astype(np.uint8)
    height, width = source.shape[:2]
    number_of_pixels = int(round(density * height * width))
    if number_of_pixels == 0:
        return result

    rng = np.random.default_rng(seed)
    selected = rng.choice(height * width, size=number_of_pixels, replace=False)
    rows, cols = np.unravel_index(selected, (height, width))
    salt_mask = rng.random(number_of_pixels) < salt_probability
    result[rows[salt_mask], cols[salt_mask]] = 255
    result[rows[~salt_mask], cols[~salt_mask]] = 0
    return result
