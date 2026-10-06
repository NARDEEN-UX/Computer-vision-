"""Min-max image normalization for the 8-bit image pipeline."""

from __future__ import annotations

import numpy as np


def normalize_image(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Map values from [min_val, max_val] to the display range [0, 255]."""
    if max_val <= min_val:
        raise ValueError("max_val must be greater than min_val.")

    values = np.asarray(image, dtype=np.float64)
    normalized = (values - min_val) / (max_val - min_val)
    return np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
