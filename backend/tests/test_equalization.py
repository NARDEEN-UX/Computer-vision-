from __future__ import annotations

import numpy as np

from app.algorithms.equalization.equalization import equalize_channel, equalize_image


def test_constant_channel_is_unchanged() -> None:
    image = np.full((4, 5), 100, dtype=np.uint8)

    result = equalize_channel(image)

    assert np.array_equal(result, image)
    assert result.dtype == np.uint8


def test_equalize_channel_stretches_two_level_input() -> None:
    image = np.array([[50, 50], [200, 200]], dtype=np.uint8)

    result = equalize_channel(image)

    assert np.array_equal(result, [[128, 128], [255, 255]])


def test_equalize_channel_preserves_shape_and_dtype() -> None:
    image = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

    result = equalize_channel(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_equalize_image_equalizes_rgb_channels_independently() -> None:
    image = np.array(
        [
            [[50, 10, 80], [50, 10, 80]],
            [[200, 100, 160], [200, 100, 160]],
        ],
        dtype=np.uint8,
    )

    result = equalize_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 0], [[128, 128], [255, 255]])
    assert np.array_equal(result[:, :, 1], [[128, 128], [255, 255]])
    assert np.array_equal(result[:, :, 2], [[128, 128], [255, 255]])
