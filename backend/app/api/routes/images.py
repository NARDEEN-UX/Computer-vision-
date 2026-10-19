"""Endpoints for storing the active source image."""

from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from ...services.image_service import store_image

router = APIRouter(prefix="/api/images", tags=["Images"])


@router.post("/upload")
async def upload_image(image: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    return await store_image(image)
