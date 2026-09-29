"""Image decoding, algorithm dispatch, encoding, and timing."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, UploadFile


async def read_rgb_image(upload: UploadFile) -> np.ndarray:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        with Image.open(BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc


def encode_png(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


async def run_image_operation(
    upload: UploadFile,
    operation: Callable[..., np.ndarray],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    image = await read_rgb_image(upload)
    started = time.perf_counter()
    try:
        processed = operation(image, **parameters)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "image": encode_png(processed),
        "processing_time_ms": round(elapsed_ms, 3),
        "parameters": parameters,
        "metadata": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "channels": 3,
            "color_convention": "RGB",
        },
    }
