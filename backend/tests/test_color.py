from __future__ import annotations

import numpy as np

from app.algorithms.color.conversion import grayscale_to_rgb, rgb_to_grayscale


def test_rgb_to_grayscale_uses_luminance_weights() -> None:
    image = np.array([[[100, 150, 200]]], dtype=np.uint8)

    result = rgb_to_grayscale(image)

    assert result.shape == (1, 1)
    assert result.dtype == np.uint8
    assert result[0, 0] == 141


def test_grayscale_to_rgb_repeats_channel() -> None:
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    result = grayscale_to_rgb(image)

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 0], image)
    assert np.array_equal(result[:, :, 1], image)
    assert np.array_equal(result[:, :, 2], image)
