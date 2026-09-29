"""Canny edge detection using the permitted OpenCV implementation."""

from __future__ import annotations

import cv2
import numpy as np

from .common import to_grayscale


def canny(image: np.ndarray, lower_threshold: int = 50, upper_threshold: int = 150) -> np.ndarray:
    """Run OpenCV Canny after the shared RGB-to-luminance conversion."""
    if not 0 <= lower_threshold < upper_threshold:
        raise ValueError("Lower threshold must be non-negative and less than upper threshold.")
    grayscale = np.clip(np.rint(to_grayscale(image)), 0, 255).astype(np.uint8)
    return cv2.Canny(grayscale, lower_threshold, upper_threshold)
