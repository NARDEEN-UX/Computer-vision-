"""Gaussian noise implemented with NumPy random sampling."""

from __future__ import annotations

import numpy as np


def gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: int | None = None,
) -> np.ndarray:
    """Add independent Gaussian noise to every channel without mutating image."""
    if not np.isfinite(mean):
        raise ValueError("Mean must be finite.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    source = np.asarray(image)
    if source.ndim not in (2, 3) or (source.ndim == 3 and source.shape[2] != 3):
        raise ValueError("Image must be grayscale or RGB.")
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=sigma, size=source.shape)
    result = source.astype(np.float32) + noise.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)
