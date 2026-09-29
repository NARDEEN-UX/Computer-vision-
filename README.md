# Computer Vision Task 1 — Current Stage

This project currently implements image upload, Gaussian noise, salt-and-pepper noise, average filtering, Gaussian filtering, median filtering, and edge detection with Sobel, Roberts, Prewitt, and Canny. The interface supports applying later operations to the latest generated result.

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
    edge_detection/        Sobel, Roberts, Prewitt, and Canny
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
python -m uvicorn backend.app.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
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

All processing endpoints accept multipart form data with an `image` file and return `image`, `processing_time_ms`, `parameters`, and output metadata. Edge detectors return normalized grayscale images.

| Endpoint | Parameters |
|---|---|
| `POST /api/noise/gaussian` | `mean`, `sigma`, optional `seed` |
| `POST /api/noise/salt-pepper` | `density`, `salt_probability`, optional `seed` |
| `POST /api/filters/average` | `kernel_size` (3, 5, 7, or 9) |
| `POST /api/filters/gaussian` | `kernel_size`, `sigma` |
| `POST /api/filters/median` | `kernel_size` (3, 5, 7, or 9) |
| `POST /api/edge/sobel` | `direction` (`magnitude`, `x`, or `y`) |
| `POST /api/edge/roberts` | `direction` (`magnitude`, `x`, or `y`) |
| `POST /api/edge/prewitt` | `direction` (`magnitude`, `x`, or `y`) |
| `POST /api/edge/canny` | `lower_threshold`, `upper_threshold` |
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
| Sobel | Custom NumPy kernels, custom convolution, and gradient magnitude |
| Roberts | Custom 2×2 NumPy kernels, custom convolution, and gradient magnitude |
| Prewitt | Custom NumPy kernels, custom convolution, and gradient magnitude |
| Canny | OpenCV `cv2.Canny` (allowed built-in implementation) |

NumPy is used for arrays, arithmetic, random numbers, padding, and numerical operations. Pillow is used only for image decoding/encoding. FastAPI, React, Vite, and Axios provide the application layer. RGB edge inputs are converted to luminance grayscale using `0.299R + 0.587G + 0.114B`. Sobel, Roberts, and Prewitt responses are normalized to an 8-bit display image; X/Y responses use absolute values so negative gradients are represented correctly. Reflection padding preserves dimensions, including the 2×2 Roberts kernels. No `cv2.Sobel`, `cv2.filter2D`, `skimage.filters.sobel`, `skimage.filters.prewitt`, `cv2.blur`, `cv2.GaussianBlur`, or `cv2.medianBlur` is used.

## Deliberately not implemented yet

Uniform noise; histograms and CDF; equalization; normalization; thresholding; FFT/frequency-domain filters; hybrid images; and final report generation remain intentionally excluded until this stage is manually verified.
