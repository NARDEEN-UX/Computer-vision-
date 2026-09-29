from __future__ import annotations

import numpy as np
import pytest

from app.algorithms.edge_detection.canny import canny
from app.algorithms.edge_detection.common import to_grayscale
from app.algorithms.edge_detection.prewitt import prewitt
from app.algorithms.edge_detection.roberts import roberts
from app.algorithms.edge_detection.sobel import sobel


@pytest.fixture
def step_image() -> np.ndarray:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[:, 16:] = 255
    return image


@pytest.fixture
def rgb_step_image(step_image: np.ndarray) -> np.ndarray:
    return np.stack([step_image, step_image, step_image], axis=2)


@pytest.mark.parametrize("detector", [sobel, prewitt, roberts, canny])
def test_edge_detectors_preserve_shape_and_return_display_image(detector, rgb_step_image) -> None:
    result = detector(rgb_step_image)
    assert result.shape == rgb_step_image.shape[:2]
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255


@pytest.mark.parametrize("detector", [sobel, prewitt, roberts, canny])
def test_constant_image_has_no_edges(detector) -> None:
    image = np.full((20, 20, 3), 128, dtype=np.uint8)
    result = detector(image)
    assert np.max(result) == 0


def test_luminance_conversion_uses_all_rgb_channels() -> None:
    image = np.array([[[100, 200, 50]]], dtype=np.uint8)
    gray = to_grayscale(image)
    assert gray.shape == (1, 1)
    assert np.isclose(gray[0, 0], 0.299 * 100 + 0.587 * 200 + 0.114 * 50)


def test_direction_outputs_are_supported(step_image) -> None:
    for detector in (sobel, prewitt, roberts):
        magnitude = detector(step_image, direction="magnitude")
        x_response = detector(step_image, direction="x")
        y_response = detector(step_image, direction="y")
        assert magnitude.shape == x_response.shape == y_response.shape
        assert np.max(x_response) > 0
        assert np.max(y_response) >= 0


def test_canny_thresholds_are_validated(step_image) -> None:
    with pytest.raises(ValueError, match="less than upper"):
        canny(step_image, lower_threshold=150, upper_threshold=50)
    with pytest.raises(ValueError, match="less than upper"):
        canny(step_image, lower_threshold=-1, upper_threshold=50)
