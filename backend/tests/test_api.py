from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _image_file() -> tuple[str, BytesIO, str]:
    image = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8), mode="RGB")
    data = BytesIO()
    image.save(data, format="PNG")
    data.seek(0)
    return ("image.png", data, "image/png")


def test_health_and_filter_endpoint() -> None:
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.post(
        "/api/filters/gaussian",
        files={"image": _image_file()},
        data={"kernel_size": "5", "sigma": "1.5"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["image"].startswith("data:image/png;base64,")
    assert body["parameters"] == {"kernel_size": 5, "sigma": 1.5}
    assert body["metadata"]["color_convention"] == "RGB"


def test_api_rejects_invalid_parameters() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/filters/average",
        files={"image": _image_file()},
        data={"kernel_size": "4"},
    )
    assert response.status_code == 422
    assert "Kernel size" in response.json()["detail"]


def test_edge_api_returns_grayscale_metadata_and_implementation() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/edge/sobel",
        files={"image": _image_file()},
        data={"direction": "magnitude"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["channels"] == 1
    assert body["metadata"]["color_convention"] == "grayscale"
    assert body["metadata"]["implementation"] == "Custom NumPy"


def test_canny_api_rejects_invalid_threshold_order() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/edge/canny",
        files={"image": _image_file()},
        data={"lower_threshold": "200", "upper_threshold": "100"},
    )
    assert response.status_code == 422
    assert "less than upper" in response.json()["detail"]
