"""Spatial and frequency-domain filter endpoints for the assignment."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...algorithms.filters.average import average_filter
from ...algorithms.filters.frequency import frequency_filter_details
from ...algorithms.filters.gaussian import gaussian_filter
from ...algorithms.filters.hybrid import hybrid_image
from ...algorithms.filters.median import median_filter
from ...schemas.validation import validated_kernel_size, validated_positive
from ...services.image_service import encode_png, read_image, run_image_operation

router = APIRouter(prefix="/api/filters", tags=["Filters"])


@router.post("/average")
async def apply_average_filter(
    image: Annotated[UploadFile, File(...)],
    kernel_size: Annotated[int, Form()] = 3,
):
    kernel_size = validated_kernel_size(kernel_size)
    return await run_image_operation(image, average_filter, {"kernel_size": kernel_size})


@router.post("/gaussian")
async def apply_gaussian_filter(
    image: Annotated[UploadFile, File(...)],
    kernel_size: Annotated[int, Form()] = 3,
    sigma: Annotated[float, Form()] = 1.0,
):
    kernel_size = validated_kernel_size(kernel_size)
    validated_positive(sigma, "Sigma")
    return await run_image_operation(
        image,
        gaussian_filter,
        {"kernel_size": kernel_size, "sigma": sigma},
    )


@router.post("/median")
async def apply_median_filter(
    image: Annotated[UploadFile, File(...)],
    kernel_size: Annotated[int, Form()] = 3,
):
    kernel_size = validated_kernel_size(kernel_size)
    return await run_image_operation(image, median_filter, {"kernel_size": kernel_size})


@router.post("/frequency")
async def apply_frequency_filter(
    image: Annotated[UploadFile, File(...)],
    cutoff: Annotated[float, Form()] = 30.0,
    filter_type: Annotated[str, Form()] = "low",
    filter_family: Annotated[str, Form()] = "gaussian",
    mean: Annotated[float, Form()] = 0.0,
    sigma: Annotated[float, Form()] = 30.0,
):
    if cutoff <= 0:
        raise HTTPException(status_code=422, detail="Cutoff must be greater than zero.")
    if sigma <= 0:
        raise HTTPException(status_code=422, detail="Gaussian sigma must be greater than zero.")
    if mean < 0:
        raise HTTPException(status_code=422, detail="Gaussian mean must be zero or greater.")
    normalized_type = filter_type.lower()
    normalized_family = filter_family.lower()
    if normalized_type not in {"low", "high"}:
        raise HTTPException(status_code=422, detail="filter_type must be 'low' or 'high'.")
    if normalized_family not in {"ideal", "gaussian"}:
        raise HTTPException(status_code=422, detail="filter_family must be 'ideal' or 'gaussian'.")
    source = await read_image(image)
    started = time.perf_counter()
    (
        processed,
        input_fourier,
        result_fourier,
        filter_image,
        filter_fourier,
    ) = frequency_filter_details(
        source,
        cutoff,
        normalized_type,
        normalized_family,
        mean,
        sigma,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "image": encode_png(processed),
        "fourier_input": encode_png(input_fourier),
        "fourier_result": encode_png(result_fourier),
        "filter_image": encode_png(filter_image),
        "fourier_filter": encode_png(filter_fourier),
        "processing_time_ms": round(elapsed_ms, 3),
        "parameters": {
            "cutoff": cutoff,
            "filter_type": normalized_type,
            "filter_family": normalized_family,
            "mean": mean,
            "sigma": sigma,
        },
        "metadata": {
            "width": int(source.shape[1]),
            "height": int(source.shape[0]),
            "input_image_type": "Grayscale" if source.ndim == 2 else "RGB",
            "input_channels": 1 if source.ndim == 2 else 3,
            "output_image_type": "Grayscale",
            "channels": 1,
            "color_convention": "grayscale",
            "implementation": f"Custom NumPy {normalized_family.capitalize()} Fourier filter",
        },
    }


@router.post("/hybrid")
async def apply_hybrid_filter(
    image_a: Annotated[UploadFile, File(...)],
    image_b: Annotated[UploadFile, File(...)],
    sigma: Annotated[float, Form()] = 8.0,
    kernel_size: Annotated[int, Form()] = 15,
):
    validated_positive(sigma, "Sigma")
    kernel_size = validated_kernel_size(kernel_size)

    source_a = await read_image(image_a)
    source_b = await read_image(image_b)
    started = time.perf_counter()
    processed = hybrid_image(source_a, source_b, sigma=sigma, kernel_size=kernel_size)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "image": encode_png(processed),
        "processing_time_ms": round(elapsed_ms, 3),
        "parameters": {"sigma": sigma, "kernel_size": kernel_size},
        "metadata": {
            "width": int(processed.shape[1]),
            "height": int(processed.shape[0]),
            "input_image_type": "RGB" if source_a.ndim == 3 else "Grayscale",
            "input_channels": 1 if source_a.ndim == 2 else 3,
            "output_image_type": "Grayscale" if processed.ndim == 2 else "RGB",
            "channels": 1 if processed.ndim == 2 else 3,
            "color_convention": "grayscale" if processed.ndim == 2 else "RGB",
            "implementation": "Custom NumPy",
        },
    }
