"""Low-pass filter endpoints for the current assignment stage."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from ...algorithms.filters.average import average_filter
from ...algorithms.filters.gaussian import gaussian_filter
from ...algorithms.filters.median import median_filter
from ...schemas.validation import validated_kernel_size, validated_positive
from ...services.image_service import run_image_operation

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
