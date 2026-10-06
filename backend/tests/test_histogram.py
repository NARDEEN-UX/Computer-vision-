from __future__ import annotations

import numpy as np

from app.algorithms.histogram.histogram import (
    compute_cdf,
    compute_distribution,
    compute_grayscale_histogram,
    compute_histogram,
    compute_image_histogram,
    compute_rgb_histogram,
)


def test_constant_grayscale_histogram_and_distribution() -> None:
    image = np.full((10, 10), 100, dtype=np.uint8)
    result = compute_grayscale_histogram(image)["channels"]["grayscale"]
    assert result["intensity"] == list(range(256))
    assert result["histogram"][100] == 100
    assert sum(result["histogram"]) == 100
    assert result["distribution"][100] == 1.0
    assert sum(result["distribution"]) == 1.0
    assert result["cdf"][99] == 0.0
    assert result["cdf"][100] == 1.0
    assert result["cdf"][-1] == 1.0
    assert sum(result["histogram"][:100]) == 0
    assert sum(result["histogram"][101:]) == 0


def test_known_small_grayscale_histogram() -> None:
    image = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)
    counts = compute_histogram(image)
    assert counts[0] == 1
    assert counts[1] == 2
    assert counts[2] == 3
    assert counts.sum() == image.size
    assert np.isclose(compute_distribution(counts).sum(), 1.0)
    cdf = compute_cdf(compute_distribution(counts))
    assert np.all(np.diff(cdf) >= 0)
    assert np.allclose(cdf[:3], [1 / 6, 3 / 6, 1.0])


def test_known_small_rgb_histogram_keeps_channel_order() -> None:
    image = np.array(
        [
            [[10, 20, 30], [10, 21, 31]],
            [[11, 20, 30], [12, 21, 32]],
        ],
        dtype=np.uint8,
    )
    result = compute_rgb_histogram(image)["channels"]
    assert result["red"]["histogram"][10] == 2
    assert result["red"]["histogram"][11] == 1
    assert result["red"]["histogram"][12] == 1
    assert result["green"]["histogram"][20] == 2
    assert result["green"]["histogram"][21] == 2
    assert result["blue"]["histogram"][30] == 2
    assert result["blue"]["histogram"][31] == 1
    assert result["blue"]["histogram"][32] == 1
    for channel in result.values():
        assert len(channel["intensity"]) == 256
        assert len(channel["histogram"]) == 256
        assert len(channel["distribution"]) == 256
        assert len(channel["cdf"]) == 256
        assert sum(channel["histogram"]) == 4
        assert np.isclose(sum(channel["distribution"]), 1.0)
        assert np.all(np.diff(channel["cdf"]) >= 0)
        assert np.isclose(channel["cdf"][-1], 1.0)


def test_grayscale_image_has_one_channel_only() -> None:
    result = compute_image_histogram(np.zeros((4, 5), dtype=np.uint8))
    assert result["image_type"] == "grayscale"
    assert list(result["channels"]) == ["grayscale"]
