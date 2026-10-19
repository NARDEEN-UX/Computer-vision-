"""Image transformation endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile

from ...algorithms.color.conversion import grayscale_to_rgb, rgb_to_grayscale
from ...algorithms.equalization.equalization import equalize_image
from ...algorithms.histogram.histogram import compute_image_histogram
from ...algorithms.normalization.normalization import normalize_image
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
async def equalize(
    image: Annotated[UploadFile | None, File()] = None,
    image_id: Annotated[str | None, Form()] = None,
):
    return await run_image_operation(
        image,
        equalize_image,
        {},
        extras=_comparison_histograms,
        image_id=image_id,
    )


@router.post("/to-gray")
async def to_gray(
    image: Annotated[UploadFile | None, File()] = None,
    image_id: Annotated[str | None, Form()] = None,
):
    return await run_image_operation(
        image,
        rgb_to_grayscale,
        {},
        extras=_comparison_histograms,
        image_id=image_id,
    )


@router.post("/to-rgb")
async def to_rgb(
    image: Annotated[UploadFile | None, File()] = None,
    image_id: Annotated[str | None, Form()] = None,
):
    return await run_image_operation(
        image,
        grayscale_to_rgb,
        {},
        extras=_comparison_histograms,
        image_id=image_id,
    )


@router.post("/normalize")
async def normalize(
    image: Annotated[UploadFile | None, File()] = None,
    min_val: Annotated[float, Form()] = 0.0,
    max_val: Annotated[float, Form()] = 255.0,
    image_id: Annotated[str | None, Form()] = None,
):
    return await run_image_operation(
        image,
        normalize_image,
        {"min_val": min_val, "max_val": max_val},
        extras=_comparison_histograms,
        image_id=image_id,
    )