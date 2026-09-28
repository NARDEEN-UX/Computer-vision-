import numpy as np
from functions.conv2d import conv2d

def average_filter(img, k_size=3):
    # create box kernel where all values are equal and sum to 1
    kernel = np.ones((k_size, k_size), dtype=np.float32) / (k_size * k_size)

    # apply convolution
    out = conv2d(img, kernel)

    # round and clip to valid uint8 range
    out = np.clip(np.round(out), 0, 255)
    return out.astype(np.uint8)


def gaussian_filter(img, k_size=3, sigma=1.0):
    pad = k_size // 2
    kernel = np.zeros((k_size, k_size), dtype=np.float32)

    # calculate gaussian weights
    for r in range(k_size):
        for c in range(k_size):
            y = r - pad
            x = c - pad
            kernel[r, c] = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    # normalize kernel so the sum is 1
    kernel = kernel / np.sum(kernel)

    # apply convolution
    out = conv2d(img, kernel)

    # round and clip to valid uint8 range
    out = np.clip(np.round(out), 0, 255)
    return out.astype(np.uint8)


def median_filter(img, k_size=3):
    h, w = img.shape[:2]
    pad = k_size // 2

    # if color image, apply to each channel
    if len(img.shape) == 3:
        out = np.zeros_like(img)
        for ch in range(3):
            out[:, :, ch] = median_filter(img[:, :, ch], k_size)
        return out

    # pad image edges using reflection
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')

    # get neighbor windows
    neighbors = []
    for r in range(k_size):
        for c in range(k_size):
            neighbors.append(padded[r: r + h, c: c + w])

    # find median value
    out = np.median(neighbors, axis=0)
    return out.astype(np.uint8)