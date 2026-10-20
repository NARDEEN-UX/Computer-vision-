from __future__ import annotations

import numpy as np
import pytest

from app.algorithms.convolution.conv2d import conv2d
from app.algorithms.filters.average import average_filter, average_kernel
from app.algorithms.filters.gaussian import gaussian_filter, gaussian_kernel
from app.algorithms.filters.hybrid import hybrid_image
from app.algorithms.filters.median import median_filter
from app.algorithms.noise.gaussian import gaussian_noise
from app.algorithms.noise.salt_pepper import salt_pepper_noise


def test_convolution_flips_kernel_and_preserves_shape() -> None:
    image = np.arange(9, dtype=np.float32).reshape(3, 3)
    kernel = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    result = conv2d(image, kernel)
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert result[1, 1] == 8  # flipped kernel samples the lower-right neighbor


def test_average_kernel_is_normalized_and_supports_sizes() -> None:
    for size in (3, 5, 7, 9):
        kernel = average_kernel(size)
        assert kernel.shape == (size, size)
        assert np.isclose(kernel.sum(), 1.0)


def test_filters_support_grayscale_and_rgb() -> None:
    gray = np.arange(49, dtype=np.uint8).reshape(7, 7)
    rgb = np.stack([gray, gray + 1, gray + 2], axis=2)
    for operation in (
        lambda value: average_filter(value, 3),
        lambda value: gaussian_filter(value, 5, 1.2),
        lambda value: median_filter(value, 3),
    ):
        assert operation(gray).shape == gray.shape
        assert operation(rgb).shape == rgb.shape
        assert operation(rgb).dtype == np.uint8


def test_median_filter_orders_window_values_without_numpy_median() -> None:
    image = np.array([[0, 255, 0], [255, 128, 255], [0, 255, 0]], dtype=np.uint8)
    result = median_filter(image, 3)
    assert result[1, 1] == 128


def test_gaussian_kernel_is_symmetric_and_normalized() -> None:
    kernel = gaussian_kernel(5, 1.5)
    assert np.isclose(kernel.sum(), 1.0)
    assert np.allclose(kernel, np.flip(kernel))


def test_hybrid_uses_frequency_cutoffs_and_preserves_output_shape() -> None:
    image_a = np.zeros((32, 32), dtype=np.uint8)
    image_a[8:24, 8:24] = 220
    image_b = np.zeros((32, 32), dtype=np.uint8)
    image_b[::2, :] = 255

    result = hybrid_image(image_a, image_b, low_cutoff=1, high_cutoff=2)

    assert result.shape == image_a.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255


def test_gaussian_noise_is_reproducible_and_does_not_mutate_source() -> None:
    source = np.full((10, 10, 3), 128, dtype=np.uint8)
    original = source.copy()
    first = gaussian_noise(source, mean=0, sigma=10, seed=7)
    second = gaussian_noise(source, mean=0, sigma=10, seed=7)
    assert np.array_equal(first, second)
    assert np.array_equal(source, original)
    assert first.shape == source.shape
    assert first.min() >= 0 and first.max() <= 255


def test_salt_pepper_density_and_seed() -> None:
    source = np.full((10, 10), 128, dtype=np.uint8)
    first = salt_pepper_noise(source, density=0.2, salt_probability=0.5, seed=4)
    second = salt_pepper_noise(source, density=0.2, salt_probability=0.5, seed=4)
    assert np.array_equal(first, second)
    assert np.count_nonzero(first != source) == 20
    assert set(np.unique(first)).issubset({0, 128, 255})
    assert np.array_equal(salt_pepper_noise(source, density=0, seed=1), source)


def test_invalid_parameters_are_rejected() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="Sigma"):
        gaussian_noise(image, sigma=0)
    with pytest.raises(ValueError, match="Density"):
        salt_pepper_noise(image, density=1.1)
    with pytest.raises(ValueError, match="one of"):
        average_filter(image, kernel_size=4)
    even_result = conv2d(image, np.ones((2, 2), dtype=np.float32))
    assert even_result.shape == image.shape
