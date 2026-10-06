"""Service boundary for histogram analysis."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from fastapi import HTTPException, UploadFile

from ..algorithms.histogram.histogram import compute_image_histogram
from .image_service import read_image


async def analyze_histogram(upload: UploadFile) -> dict[str, Any]:
    image = await read_image(upload)
    started = time.perf_counter()
    try:
        result = compute_image_histogram(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result["metadata"] = {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "input_image_type": "Grayscale" if image.ndim == 2 else "RGB",
        "input_channels": 1 if image.ndim == 2 else 3,
        "implementation": "Custom NumPy",
    }
    return result
