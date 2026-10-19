"""Histogram and normalized distribution endpoint."""

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from ...services.histogram_service import analyze_histogram

router = APIRouter(prefix="/api/histogram", tags=["Histogram"])


@router.post("")
@router.post("/analyze")
async def apply_histogram_analysis(
    image: Annotated[UploadFile | None, File()] = None,
    image_id: Annotated[str | None, Form()] = None,
):
    return await analyze_histogram(image, image_id)
