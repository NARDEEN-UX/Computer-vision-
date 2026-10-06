from __future__ import annotations

import numpy as np
import pytest

from app.algorithms.normalization.normalization import normalize_image


def test_normalize_maps_bounds_to_display_range() -> None:
    image = np.array([[50, 125, 200]], dtype=np.uint8)

    result = normalize_image(image, 50, 200)

    assert np.array_equal(result, [[0, 128, 255]])
    assert result.dtype == np.uint8


def test_normalize_clips_values_outside_input_range() -> None:
    image = np.array([[0, 50, 200, 255]], dtype=np.uint8)

    result = normalize_image(image, 50, 200)

    assert np.array_equal(result, [[0, 0, 255, 255]])


def test_normalize_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="max_val"):
        normalize_image(np.zeros((2, 2), dtype=np.uint8), 10, 10)
