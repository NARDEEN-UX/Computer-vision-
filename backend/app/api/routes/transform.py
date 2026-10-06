"""Image transformation endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, File, UploadFile

from ...algorithms.color.conversion import grayscale_to_rgb, rgb_to_grayscale
from ...algorithms.equalization.equalization import equalize_image
from ...algorithms.histogram.histogram import compute_image_histogram
from ...services.image_service import run_image_operation

router = APIRouter(prefix="/api/transform", tags=["Transform"])


def _comparison_histograms(original: Any, processed: Any) -> dict[str, Any]:
    return {
        "histograms": {
            "before": compute_image_histogram(original),
            "after": compute_image_histogram(processed),
        }
    }


@router.post("/equalize")
async def equalize(image: Annotated[UploadFile, File(...)]):
    return await run_image_operation(
        image,
        equalize_image,
        {},
        extras=_comparison_histograms,
    )


@router.post("/to-gray")
async def to_gray(image: Annotated[UploadFile, File(...)]):
    return await run_image_operation(image, rgb_to_grayscale, {})


@router.post("/to-rgb")
async def to_rgb(image: Annotated[UploadFile, File(...)]):
    return await run_image_operation(image, grayscale_to_rgb, {})