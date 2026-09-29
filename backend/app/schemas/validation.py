"""Backend validation shared by route handlers."""

from __future__ import annotations

import math

from fastapi import HTTPException

from ..algorithms.filters.common import validate_kernel_size


def validated_kernel_size(value: int) -> int:
    try:
        return validate_kernel_size(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def validated_positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise HTTPException(status_code=422, detail=f"{name} must be greater than zero.")
    return value


def validated_range(value: float, name: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise HTTPException(status_code=422, detail=f"{name} must be between {minimum} and {maximum}.")
    return value
