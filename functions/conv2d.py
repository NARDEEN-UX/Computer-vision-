import numpy as np

def conv2d(img, kernel):
    # flip kernel horizontally and vertically for convolution
    kernel = np.flip(kernel)

    # get dimensions of image and kernel
    h, w = img.shape[:2]
    kh, kw = kernel.shape

    # calculate padding size
    pad_h = kh // 2
    pad_w = kw // 2

    # check if image is RGB, run convolution on each channel
    if len(img.shape) == 3:
        out = np.zeros_like(img, dtype=np.float32)
        for ch in range(3):
            out[:, :, ch] = conv2d(img[:, :, ch], kernel)
        return out

    # apply zero padding around the image
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # create output array initialized with zeros
    out = np.zeros((h, w), dtype=np.float32)

    # compute convolution sum
    for r in range(kh):
        for c in range(kw):
            out += padded[r: r + h, c: c + w] * kernel[r, c]

    return out