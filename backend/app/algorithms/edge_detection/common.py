"""Shared grayscale, gradient, and display-normalization utilities."""

from __future__ import annotations

import numpy as np


VALID_DIRECTIONS = ("magnitude", "x", "y")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB to luminance using the standard 0.299/0.587/0.114 weights."""
    source = np.asarray(image)
    if source.ndim == 2:
        return source.astype(np.float32, copy=False)
    if source.ndim == 3 and source.shape[2] == 3:
        red, green, blue = source[:, :, 0], source[:, :, 1], source[:, :, 2]
        return (0.299 * red + 0.587 * green + 0.114 * blue).astype(np.float32)
    raise ValueError("Image must be grayscale or RGB.")


def validate_direction(direction: str) -> str:
    value = direction.lower()
    if value not in VALID_DIRECTIONS:
        raise ValueError("Direction must be one of: magnitude, x, y.")
    return value


def gradient_to_uint8(gradient_x: np.ndarray, gradient_y: np.ndarray, direction: str) -> np.ndarray:
    """Select a gradient representation and normalize it for display.

    Directional responses can be negative, so X/Y outputs use absolute values.
    Every selected response is linearly scaled from its own maximum to [0, 255].
    A constant image therefore produces an all-zero edge image.
    """
    direction = validate_direction(direction)
    if direction == "x":
        selected = np.abs(gradient_x)
    elif direction == "y":
        selected = np.abs(gradient_y)
    else:
        selected = np.hypot(gradient_x, gradient_y)

    maximum = float(np.max(selected)) if selected.size else 0.0
    if maximum <= 0 or not np.isfinite(maximum):
        return np.zeros(selected.shape, dtype=np.uint8)
    return np.clip(np.rint(selected * (255.0 / maximum)), 0, 255).astype(np.uint8)
