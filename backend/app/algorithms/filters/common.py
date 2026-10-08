"""Shared validation and output conversion for custom filters."""

from __future__ import annotations

import numpy as np


ALLOWED_KERNEL_SIZES = (3, 5, 7, 9, 15)


def validate_kernel_size(kernel_size: int) -> int:
    if kernel_size not in ALLOWED_KERNEL_SIZES:
        choices = ", ".join(str(value) for value in ALLOWED_KERNEL_SIZES)
        raise ValueError(f"Kernel size must be one of: {choices}.")
    return kernel_size


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)
