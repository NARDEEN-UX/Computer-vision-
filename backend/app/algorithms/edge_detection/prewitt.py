"""Prewitt edge detection implemented with custom kernels and conv2d."""

from __future__ import annotations

import numpy as np

from ..convolution.conv2d import conv2d
from .common import gradient_to_uint8, to_grayscale, validate_direction

PREWITT_X = np.array(
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32
)
PREWITT_Y = np.array(
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32
)


def prewitt(image: np.ndarray, direction: str = "magnitude") -> np.ndarray:
    """Calculate Prewitt X/Y responses and return a normalized display image."""
    direction = validate_direction(direction)
    grayscale = to_grayscale(image)
    gradient_x = conv2d(grayscale, PREWITT_X)
    gradient_y = conv2d(grayscale, PREWITT_Y)
    return gradient_to_uint8(gradient_x, gradient_y, direction)
