"""Image histogram algorithms implemented without histogram helper APIs."""

from __future__ import annotations

import numpy as np


INTENSITIES = list(range(256))


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    """Count the occurrences of every 8-bit intensity using explicit counting."""
    values = np.asarray(channel)
    if values.ndim != 2:
        raise ValueError("A histogram channel must be a 2-D array.")
    values = values.astype(np.int64, copy=False)
    if values.size == 0:
        raise ValueError("The image channel cannot be empty.")
    if int(values.min()) < 0 or int(values.max()) > 255:
        raise ValueError("Histogram values must be between 0 and 255.")

    counts = np.zeros(256, dtype=np.int64)
    for value in values.reshape(-1):
        counts[int(value)] += 1
    return counts


def compute_distribution(histogram: np.ndarray) -> np.ndarray:
    """Normalize histogram counts into probabilities; this is not a CDF."""
    counts = np.asarray(histogram, dtype=np.int64)
    if counts.shape != (256,):
        raise ValueError("A histogram must contain exactly 256 bins.")
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("A histogram must contain at least one pixel.")
    return counts.astype(np.float64) / total


def _channel_payload(channel: np.ndarray) -> dict[str, list]:
    counts = compute_histogram(channel)
    distribution = compute_distribution(counts)
    return {
        "intensity": INTENSITIES.copy(),
        "histogram": counts.tolist(),
        "distribution": distribution.tolist(),
    }


def compute_grayscale_histogram(image: np.ndarray) -> dict[str, object]:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("A grayscale histogram requires a single-channel (H, W) image.")
    return {"image_type": "grayscale", "channels": {"grayscale": _channel_payload(array)}}


def compute_rgb_histogram(image: np.ndarray) -> dict[str, object]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("An RGB histogram requires an (H, W, 3) image.")
    return {
        "image_type": "rgb",
        "channels": {
            "red": _channel_payload(array[:, :, 0]),
            "green": _channel_payload(array[:, :, 1]),
            "blue": _channel_payload(array[:, :, 2]),
        },
    }


def compute_image_histogram(image: np.ndarray) -> dict[str, object]:
    array = np.asarray(image)
    if array.ndim == 2:
        return compute_grayscale_histogram(array)
    if array.ndim == 3 and array.shape[2] == 3:
        return compute_rgb_histogram(array)
    raise ValueError("Image must be grayscale (H, W) or RGB (H, W, 3).")
