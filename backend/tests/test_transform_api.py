from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _image_file(mode: str = "RGB") -> tuple[str, BytesIO, str]:
    if mode == "L":
        pixels = np.array([[50, 50], [200, 200]], dtype=np.uint8)
    else:
        pixels = np.array(
            [[[50, 10, 80], [50, 10, 80]], [[200, 100, 160], [200, 100, 160]]],
            dtype=np.uint8,
        )
    data = BytesIO()
    Image.fromarray(pixels, mode=mode).save(data, format="PNG")
    data.seek(0)
    return ("image.png", data, "image/png")


def test_equalization_api_returns_before_and_after_histograms() -> None:
    response = TestClient(app).post(
        "/api/transform/equalize",
        files={"image": _image_file("L")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["image"].startswith("data:image/png;base64,")
    assert body["histograms"]["before"]["image_type"] == "grayscale"
    assert body["histograms"]["after"]["image_type"] == "grayscale"
    assert body["histograms"]["before"]["channels"]["grayscale"]["cdf"][-1] == 1.0
    assert body["histograms"]["after"]["channels"]["grayscale"]["cdf"][-1] == 1.0


def test_rgb_to_gray_api_returns_grayscale_metadata() -> None:
    response = TestClient(app).post(
        "/api/transform/to-gray",
        files={"image": _image_file()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["output_image_type"] == "Grayscale"
    assert body["histograms"]["before"]["image_type"] == "rgb"
    assert body["histograms"]["after"]["image_type"] == "grayscale"
    assert body["histograms"]["after"]["channels"]["grayscale"]["cdf"][-1] == 1.0


def test_gray_to_rgb_api_returns_rgb_metadata() -> None:
    response = TestClient(app).post(
        "/api/transform/to-rgb",
        files={"image": _image_file("L")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["output_image_type"] == "RGB"
    assert body["histograms"]["before"]["image_type"] == "grayscale"
    assert body["histograms"]["after"]["image_type"] == "rgb"
    assert body["histograms"]["after"]["channels"]["red"]["cdf"][-1] == 1.0


def test_normalize_api_returns_before_and_after_histograms() -> None:
    response = TestClient(app).post(
        "/api/transform/normalize",
        data={"min_val": "50", "max_val": "200"},
        files={"image": _image_file("L")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["output_image_type"] == "Grayscale"
    assert body["parameters"] == {"min_val": 50.0, "max_val": 200.0}
    assert body["histograms"]["after"]["channels"]["grayscale"]["histogram"][0] == 2
    assert body["histograms"]["after"]["channels"]["grayscale"]["histogram"][255] == 2
