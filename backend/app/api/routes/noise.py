"""Noise endpoints for the current assignment stage."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from ...algorithms.noise.gaussian import gaussian_noise
from ...algorithms.noise.salt_pepper import salt_pepper_noise
from ...schemas.validation import validated_positive, validated_range
from ...services.image_service import run_image_operation

router = APIRouter(prefix="/api/noise", tags=["Noise"])


@router.post("/gaussian")
async def apply_gaussian_noise(
    image: Annotated[UploadFile, File(...)],
    mean: Annotated[float, Form()] = 0.0,
    sigma: Annotated[float, Form()] = 20.0,
    seed: Annotated[int | None, Form()] = None,
):
    validated_positive(sigma, "Sigma")
    return await run_image_operation(
        image,
        gaussian_noise,
        {"mean": mean, "sigma": sigma, "seed": seed},
    )


@router.post("/salt-pepper")
async def apply_salt_pepper_noise(
    image: Annotated[UploadFile, File(...)],
    density: Annotated[float, Form()] = 0.05,
    salt_probability: Annotated[float, Form()] = 0.5,
    seed: Annotated[int | None, Form()] = None,
):
    validated_range(density, "Density")
    validated_range(salt_probability, "Salt probability")
    return await run_image_operation(
        image,
        salt_pepper_noise,
        {"density": density, "salt_probability": salt_probability, "seed": seed},
    )
