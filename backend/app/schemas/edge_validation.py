"""Validation helpers specific to edge-detection API inputs."""

from __future__ import annotations

from fastapi import HTTPException

from ..algorithms.edge_detection.common import validate_direction


def validated_direction(value: str) -> str:
    try:
        return validate_direction(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def validated_canny_thresholds(lower: int, upper: int) -> tuple[int, int]:
    if not 0 <= lower < upper:
        raise HTTPException(
            status_code=422,
            detail="Lower threshold must be non-negative and less than upper threshold.",
        )
    return lower, upper
