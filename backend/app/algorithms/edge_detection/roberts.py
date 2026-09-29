"""Roberts Cross edge detection implemented with 2x2 custom kernels."""

from __future__ import annotations

import numpy as np

from ..convolution.conv2d import conv2d
from .common import gradient_to_uint8, to_grayscale, validate_direction

ROBERTS_X = np.array([[1, 0], [0, -1]], dtype=np.float32)
ROBERTS_Y = np.array([[0, 1], [-1, 0]], dtype=np.float32)


def roberts(image: np.ndarray, direction: str = "magnitude") -> np.ndarray:
    """Calculate diagonal Roberts responses with output dimensions preserved."""
    direction = validate_direction(direction)
    grayscale = to_grayscale(image)
    gradient_x = conv2d(grayscale, ROBERTS_X)
    gradient_y = conv2d(grayscale, ROBERTS_Y)
    return gradient_to_uint8(gradient_x, gradient_y, direction)
