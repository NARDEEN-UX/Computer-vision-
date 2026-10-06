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

The backend preserves the logical image representation: RGB images are `(H, W, 3)` and grayscale images are `(H, W)`. Pillow decodes grayscale files as one-channel arrays. If a black-and-white photograph is stored in an RGB container with nearly identical channels, it is detected as logical grayscale for analysis so histogram results do not create three fake curves. Noise and spatial filters preserve that representation. Edge detectors use the existing luminance conversion only for RGB inputs; grayscale edge inputs are processed directly. Results are returned as PNG data URLs with input/output type metadata. Reflection padding is used for filter boundaries. The algorithms do not call OpenCV filtering functions except the permitted Canny implementation.

## Folder structure

```text
backend/app/
  api/routes/              FastAPI endpoints only
  algorithms/
    convolution/conv2d.py custom mathematical convolution
    noise/                 Gaussian and salt-pepper noise
    filters/               Average, Gaussian, and median filters
    edge_detection/        Sobel, Roberts, Prewitt, and Canny
    histogram/              Custom histogram and distribution calculations
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
| `POST /api/histogram` | no parameters; analyzes the uploaded image |
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
| Histogram | Custom 256-bin counting with explicit pixel iteration |
| Distribution | Custom histogram-count normalization |

NumPy is used for arrays, arithmetic, random numbers, padding, and numerical operations. Pillow is used only for image decoding/encoding. FastAPI, React, Vite, and Axios provide the application layer. RGB edge inputs are converted to luminance grayscale using `0.299R + 0.587G + 0.114B`; grayscale edge inputs skip conversion and remain `(H, W)`. Sobel, Roberts, and Prewitt responses are normalized to an 8-bit display image; X/Y responses use absolute values so negative gradients are represented correctly. Reflection padding preserves dimensions, including the 2×2 Roberts kernels. No `cv2.Sobel`, `cv2.filter2D`, `skimage.filters.sobel`, `skimage.filters.prewitt`, `cv2.blur`, `cv2.GaussianBlur`, or `cv2.medianBlur` is used.

## Deliberately not implemented yet

## Histogram & Distribution

The histogram operation is implemented from scratch. For each 8-bit channel, the backend creates 256 bins and explicitly counts how many pixels have each intensity from 0 through 255. It does not use `np.histogram`, `cv2.calcHist`, or a ready-made histogram utility.

The normalized distribution is calculated separately as:

```text
distribution[i] = histogram[i] / total_pixel_count
```

This is a probability distribution, not a cumulative distribution function (CDF). Histogram Y-values represent pixel counts; distribution Y-values represent probability. RGB images produce red, green, and blue channels. Grayscale images produce one grayscale channel only. React renders the backend arrays as charts; it does not calculate histogram values.

Uniform noise; CDF; equalization; normalization; thresholding; FFT/frequency-domain filters; hybrid images; and final report generation remain intentionally excluded until these stages are requested.
