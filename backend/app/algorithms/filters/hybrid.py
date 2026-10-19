"""Hybrid image synthesis from ideal Fourier-domain low and high passes."""

from __future__ import annotations

import numpy as np

from .frequency import frequency_filter_values


def hybrid_image(
    image_a: np.ndarray,
    image_b: np.ndarray,
    low_cutoff: float = 30.0,
    high_cutoff: float = 30.0,
) -> np.ndarray:
    """Combine ideal low frequencies from image A with ideal high frequencies from B."""
    a = np.asarray(image_a)
    b = np.asarray(image_b)

    if a.shape != b.shape:
        height = min(a.shape[0], b.shape[0])
        width = min(a.shape[1], b.shape[1])
        a = a[:height, :width]
        b = b[:height, :width]

    low_a = frequency_filter_values(a, low_cutoff, "low", "ideal")
    high_b = frequency_filter_values(b, high_cutoff, "high", "ideal")
    hybrid = low_a + high_b
    return np.clip(np.rint(hybrid), 0, 255).astype(np.uint8)
