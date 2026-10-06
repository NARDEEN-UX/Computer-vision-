"""Image decoding, algorithm dispatch, encoding, and timing."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, UploadFile


def _decode_preserving_channels(data: bytes) -> np.ndarray:
    """Decode an image as (H, W) grayscale or (H, W, 3) RGB.

    Pillow modes L/LA/1/I/F and grayscale palettes are kept as one channel.
    Coloured modes, including RGBA, are converted to RGB after removing alpha.
    """
    with Image.open(BytesIO(data)) as image:
        if image.mode in {"1", "L", "LA", "I", "F", "I;16", "I;16B", "I;16L"}:
            return np.asarray(image.convert("L"), dtype=np.uint8)

        if image.mode == "P":
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2]):
                return rgb[:, :, 0].copy()
            return rgb

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        # Some black-and-white photographs are stored in an RGB JPEG/PNG
        # container. Treat nearly identical channels as one logical intensity
        # channel so histogram analysis does not show three fake curves.
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        if np.max(np.abs(red - green)) <= 2 and np.max(np.abs(green - blue)) <= 2:
            return np.rint((red + green + blue) / 3).astype(np.uint8)
        return rgb


async def read_image(upload: UploadFile) -> np.ndarray:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        return _decode_preserving_channels(data)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc


def encode_png(image: np.ndarray) -> str:
    buffer = BytesIO()
    array = np.asarray(image)
    if array.ndim == 2:
        pil_image = Image.fromarray(array.astype(np.uint8), mode="L")
    else:
        pil_image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


async def run_image_operation(
    upload: UploadFile,
    operation: Callable[..., np.ndarray],
    parameters: dict[str, Any],
    implementation: str = "Custom NumPy",
) -> dict[str, Any]:
    image = await read_image(upload)
    started = time.perf_counter()
    try:
        processed = operation(image, **parameters)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    processed_array = np.asarray(processed)
    input_is_grayscale = image.ndim == 2
    is_grayscale = processed_array.ndim == 2
    return {
        "image": encode_png(processed_array),
        "processing_time_ms": round(elapsed_ms, 3),
        "parameters": parameters,
        "metadata": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "input_image_type": "Grayscale" if input_is_grayscale else "RGB",
            "input_channels": 1 if input_is_grayscale else 3,
            "output_image_type": "Grayscale" if is_grayscale else "RGB",
            "channels": 1 if is_grayscale else 3,
            "color_convention": "grayscale" if is_grayscale else "RGB",
            "implementation": implementation,
        },
    }
