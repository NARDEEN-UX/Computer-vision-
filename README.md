# Computer Vision Task 1 — Current Stage

This project currently implements only the first verified stage of the assignment: image upload, Gaussian noise, salt-and-pepper noise, average filtering, Gaussian filtering, median filtering, processing-time measurement, and a small experiment history UI.

## Architecture

```text
React/Vite frontend
        │ Axios multipart request
        ▼
FastAPI backend
        │
        ▼
Custom NumPy algorithms
```

The backend uses RGB as its internal convention. Pillow decodes uploaded files into RGB arrays, every filter processes channels independently, and results are returned as PNG data URLs. Reflection padding is used for filter boundaries. The algorithms do not call OpenCV filtering functions.

## Folder structure

```text
backend/app/
  api/routes/              FastAPI endpoints only
  algorithms/
    convolution/conv2d.py custom mathematical convolution
    noise/                 Gaussian and salt-pepper noise
    filters/               Average, Gaussian, and median filters
  services/image_service.py image I/O, timing, encoding
  schemas/validation.py   backend validation
backend/tests/             algorithm and API tests
frontend/src/              React UI and Axios client
outputs/                   legacy sample outputs
```

## Run it

From the repository root:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

Run backend tests from the repository root with:

```powershell
pytest -q
```

## Current API

All processing endpoints accept multipart form data with an `image` file and return `image`, `processing_time_ms`, `parameters`, and RGB metadata.

| Endpoint | Parameters |
|---|---|
| `POST /api/noise/gaussian` | `mean`, `sigma`, optional `seed` |
| `POST /api/noise/salt-pepper` | `density`, `salt_probability`, optional `seed` |
| `POST /api/filters/average` | `kernel_size` (3, 5, 7, or 9) |
| `POST /api/filters/gaussian` | `kernel_size`, `sigma` |
| `POST /api/filters/median` | `kernel_size` (3, 5, 7, or 9) |
| `GET /api/health` | none |

## Custom implementation policy

| Algorithm | Implementation |
|---|---|
| Gaussian noise | Custom NumPy random sampling, clipping, and conversion |
| Salt & pepper | Custom unique-pixel selection and replacement |
| 2-D convolution | Custom padded sliding-sum implementation |
| Average filter | Custom normalized kernel + custom convolution |
| Gaussian filter | Custom dynamic Gaussian kernel + custom convolution |
| Median filter | Custom local-window median selection |

NumPy is used for arrays, arithmetic, random numbers, padding, and numerical operations. Pillow is used only for image decoding/encoding. FastAPI, React, Vite, and Axios provide the application layer. No `cv2.blur`, `cv2.GaussianBlur`, `cv2.medianBlur`, or `cv2.filter2D` is used.

## Deliberately not implemented yet

Uniform noise; Sobel, Roberts, Prewitt, and Canny edges; histograms and CDF; equalization; normalization; thresholding; FFT/frequency-domain filters; hybrid images; and final report generation are intentionally excluded until this stage is manually verified.
