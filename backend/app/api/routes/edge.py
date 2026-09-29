"""Edge-detection endpoints for the current assignment stage."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from ...algorithms.edge_detection.canny import canny
from ...algorithms.edge_detection.prewitt import prewitt
from ...algorithms.edge_detection.roberts import roberts
from ...algorithms.edge_detection.sobel import sobel
from ...schemas.edge_validation import validated_canny_thresholds, validated_direction
from ...services.image_service import run_image_operation

router = APIRouter(prefix="/api/edge", tags=["Edge Detection"])


@router.post("/sobel")
async def apply_sobel(
    image: Annotated[UploadFile, File(...)],
    direction: Annotated[str, Form()] = "magnitude",
):
    direction = validated_direction(direction)
    return await run_image_operation(image, sobel, {"direction": direction}, "Custom NumPy")


@router.post("/roberts")
async def apply_roberts(
    image: Annotated[UploadFile, File(...)],
    direction: Annotated[str, Form()] = "magnitude",
):
    direction = validated_direction(direction)
    return await run_image_operation(image, roberts, {"direction": direction}, "Custom NumPy")


@router.post("/prewitt")
async def apply_prewitt(
    image: Annotated[UploadFile, File(...)],
    direction: Annotated[str, Form()] = "magnitude",
):
    direction = validated_direction(direction)
    return await run_image_operation(image, prewitt, {"direction": direction}, "Custom NumPy")


@router.post("/canny")
async def apply_canny(
    image: Annotated[UploadFile, File(...)],
    lower_threshold: Annotated[int, Form()] = 50,
    upper_threshold: Annotated[int, Form()] = 150,
):
    lower_threshold, upper_threshold = validated_canny_thresholds(lower_threshold, upper_threshold)
    return await run_image_operation(
        image,
        canny,
        {"lower_threshold": lower_threshold, "upper_threshold": upper_threshold},
        "OpenCV",
    )
