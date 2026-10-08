"""Fourier-domain low-pass and high-pass filtering."""

from __future__ import annotations

import numpy as np

from ..color.conversion import rgb_to_grayscale


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] == 3:
        return rgb_to_grayscale(array).astype(np.float32)
    raise ValueError("Image must be grayscale or RGB with three channels.")


def _filter_mask(
    shape: tuple[int, int],
    cutoff: float,
    filter_type: str,
    filter_family: str,
    mean: float,
    sigma: float,
) -> np.ndarray:
    rows, cols = shape
    center_y, center_x = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.hypot(x - center_x, y - center_y)

    if filter_family == "ideal":
        low_pass = (distance <= cutoff).astype(np.float32)
    elif filter_family == "gaussian":
        low_pass = np.exp(-((distance - mean) ** 2) / (2.0 * sigma**2)).astype(np.float32)
    else:
        raise ValueError("filter_family must be 'ideal' or 'gaussian'.")

    if filter_type == "low":
        return low_pass
    if filter_type == "high":
        return 1.0 - low_pass
    raise ValueError("filter_type must be 'low' or 'high'.")


def _spectrum_image(transform: np.ndarray, already_shifted: bool = False) -> np.ndarray:
    spectrum = transform if already_shifted else np.fft.fftshift(transform)
    magnitude = np.log1p(np.abs(spectrum))
    minimum = float(magnitude.min())
    maximum = float(magnitude.max())
    if maximum <= minimum:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    scaled = (magnitude - minimum) * (255.0 / (maximum - minimum))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _normalize_image(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - minimum) * (255.0 / (maximum - minimum))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _signed_image(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    span = max(abs(minimum), abs(maximum))
    if span == 0:
        return np.full(values.shape, 128, dtype=np.uint8)
    scaled = (values + span) * (255.0 / (2.0 * span))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def frequency_filter_details(
    image: np.ndarray,
    cutoff: float = 30.0,
    filter_type: str = "low",
    filter_family: str = "gaussian",
    mean: float = 0.0,
    sigma: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the filtered image and visualizations of its frequency-domain steps."""
    grayscale = _as_grayscale(image)
    transform = np.fft.fft2(grayscale)
    shifted = np.fft.fftshift(transform)
    mask = _filter_mask(grayscale.shape, cutoff, filter_type, filter_family, mean, sigma)
    filtered = shifted * mask
    inverse_shifted = np.fft.ifftshift(filtered)
    restored = np.fft.ifft2(inverse_shifted)
    normalized = np.clip(np.rint(restored.real), 0, 255).astype(np.uint8)
    spatial_filter = np.fft.ifft2(np.fft.ifftshift(mask)).real
    return (
        normalized,
        _spectrum_image(transform),
        _spectrum_image(filtered, already_shifted=True),
        _signed_image(spatial_filter),
        np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8),
    )


def frequency_filter(
    image: np.ndarray,
    cutoff: float = 30.0,
    filter_type: str = "low",
    filter_family: str = "gaussian",
    mean: float = 0.0,
    sigma: float = 30.0,
) -> np.ndarray:
    """Apply an ideal or Gaussian Fourier-domain filter."""
    return frequency_filter_details(image, cutoff, filter_type, filter_family, mean, sigma)[0]
