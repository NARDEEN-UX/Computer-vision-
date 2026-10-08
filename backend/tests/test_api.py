from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _image_file() -> tuple[str, BytesIO, str]:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :3] = [220, 40, 30]
    pixels[:, 3:6] = [30, 200, 60]
    pixels[:, 6:] = [40, 80, 220]
    image = Image.fromarray(pixels, mode="RGB")
    data = BytesIO()
    image.save(data, format="PNG")
    data.seek(0)
    return ("image.png", data, "image/png")


def _grayscale_file() -> tuple[str, BytesIO, str]:
    pixels = np.tile(np.arange(32, dtype=np.uint8) * 8, (32, 1))
    image = Image.fromarray(pixels, mode="L")
    data = BytesIO()
    image.save(data, format="PNG")
    data.seek(0)
    return ("grayscale.png", data, "image/png")


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


def test_frequency_api_returns_gaussian_visualizations_for_both_modes() -> None:
    client = TestClient(app)
    for filter_type in ("low", "high"):
        response = client.post(
            "/api/filters/frequency",
            files={"image": _grayscale_file()},
            data={"cutoff": "12", "filter_type": filter_type, "filter_family": "gaussian", "mean": "6", "sigma": "18"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["parameters"] == {
            "cutoff": 12.0,
            "filter_type": filter_type,
            "filter_family": "gaussian",
            "mean": 6.0,
            "sigma": 18.0,
        }
        for key in ("image", "fourier_input", "fourier_result", "filter_image", "fourier_filter"):
            assert body[key].startswith("data:image/png;base64,")


def test_frequency_api_accepts_ideal_filter_family() -> None:
    response = TestClient(app).post(
        "/api/filters/frequency",
        files={"image": _grayscale_file()},
        data={"cutoff": "12", "filter_type": "high", "filter_family": "ideal", "mean": "0"},
    )
    assert response.status_code == 200
    assert response.json()["parameters"]["filter_family"] == "ideal"


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


def test_all_current_operations_accept_true_grayscale_png() -> None:
    operations = [
        ("/api/noise/gaussian", {"mean": "0", "sigma": "10", "seed": "7"}),
        ("/api/noise/salt-pepper", {"density": "0.1", "salt_probability": "0.5", "seed": "7"}),
        ("/api/filters/average", {"kernel_size": "3"}),
        ("/api/filters/gaussian", {"kernel_size": "3", "sigma": "1"}),
        ("/api/filters/median", {"kernel_size": "3"}),
        ("/api/edge/sobel", {"direction": "magnitude"}),
        ("/api/edge/roberts", {"direction": "magnitude"}),
        ("/api/edge/prewitt", {"direction": "magnitude"}),
        ("/api/edge/canny", {"lower_threshold": "50", "upper_threshold": "150"}),
    ]
    client = TestClient(app)
    for endpoint, form_data in operations:
        response = client.post(endpoint, files={"image": _grayscale_file()}, data=form_data)
        assert response.status_code == 200, (endpoint, response.text)
        metadata = response.json()["metadata"]
        assert metadata["input_image_type"] == "Grayscale"
        assert metadata["input_channels"] == 1
        assert metadata["output_image_type"] == "Grayscale"
        assert metadata["channels"] == 1
        encoded = response.json()["image"].split(",", 1)[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as output_image:
            assert output_image.mode == "L"


def test_rgb_input_still_reports_rgb_metadata() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/noise/gaussian",
        files={"image": _image_file()},
        data={"mean": "0", "sigma": "10", "seed": "7"},
    )
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["input_image_type"] == "RGB"
    assert metadata["input_channels"] == 3
    assert metadata["output_image_type"] == "RGB"
    assert metadata["channels"] == 3


def test_rgb_container_with_monochrome_content_is_detected_as_grayscale() -> None:
    pixels = np.full((8, 8, 3), 128, dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    data = BytesIO()
    image.save(data, format="JPEG")
    data.seek(0)
    client = TestClient(app)
    response = client.post(
        "/api/histogram/analyze",
        files={"image": ("monochrome-rgb.jpg", data, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["image_type"] == "grayscale"
    assert list(body["channels"]) == ["grayscale"]
    assert body["metadata"]["input_image_type"] == "Grayscale"


def test_histogram_api_returns_rgb_channels() -> None:
    client = TestClient(app)
    response = client.post("/api/histogram/analyze", files={"image": _image_file()})
    assert response.status_code == 200
    body = response.json()
    assert body["image_type"] == "rgb"
    assert list(body["channels"]) == ["red", "green", "blue"]
    assert body["metadata"]["implementation"] == "Custom NumPy"
    for channel in body["channels"].values():
        assert len(channel["intensity"]) == 256
        assert len(channel["histogram"]) == 256
        assert len(channel["distribution"]) == 256
        assert sum(channel["histogram"]) == 64


def test_histogram_api_returns_one_grayscale_channel() -> None:
    client = TestClient(app)
    response = client.post("/api/histogram", files={"image": _grayscale_file()})
    assert response.status_code == 200
    body = response.json()
    assert body["image_type"] == "grayscale"
    assert list(body["channels"]) == ["grayscale"]
    assert sum(body["channels"]["grayscale"]["histogram"]) == 32 * 32
