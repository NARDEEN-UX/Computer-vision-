# Computer Vision Task 1 — Current Stage

This project currently implements image upload, Gaussian noise, salt-and-pepper noise, average filtering, Gaussian filtering, median filtering, edge detection with Sobel, Roberts, Prewitt, and Canny, RGB/grayscale conversion, and histogram equalization. The interface supports applying later operations to the latest generated result and comparing histograms before and after equalization.

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
    color/                  RGB and grayscale conversion
    histogram/              Custom histogram, PDF, and CDF calculations
    equalization/           CDF-based histogram equalization
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
| `POST /api/histogram/analyze` | no parameters; analyzes the uploaded image |
| `POST /api/transform/equalize` | no parameters; returns equalized image and before/after histograms |
| `POST /api/transform/to-gray` | no parameters; converts RGB to grayscale |
| `POST /api/transform/to-rgb` | no parameters; converts grayscale to RGB |
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
| Distribution/PDF | Custom histogram-count normalization |
| CDF | Custom cumulative sum of the normalized distribution |
| Histogram equalization | Custom CDF lookup table applied per grayscale/RGB channel |
| RGB to grayscale | Custom luminance conversion using `0.299R + 0.587G + 0.114B` |
| Grayscale to RGB | Custom channel replication |

NumPy is used for arrays, arithmetic, random numbers, padding, and numerical operations. Pillow is used only for image decoding/encoding. FastAPI, React, Vite, and Axios provide the application layer. RGB edge inputs are converted to luminance grayscale using `0.299R + 0.587G + 0.114B`; grayscale edge inputs skip conversion and remain `(H, W)`. Sobel, Roberts, and Prewitt responses are normalized to an 8-bit display image; X/Y responses use absolute values so negative gradients are represented correctly. Reflection padding preserves dimensions, including the 2×2 Roberts kernels. No `cv2.Sobel`, `cv2.filter2D`, `skimage.filters.sobel`, `skimage.filters.prewitt`, `cv2.blur`, `cv2.GaussianBlur`, or `cv2.medianBlur` is used.

## Histogram & Distribution

The histogram operation is implemented from scratch. For each 8-bit channel, the backend creates 256 bins and explicitly counts how many pixels have each intensity from 0 through 255. It does not use `np.histogram`, `cv2.calcHist`, or a ready-made histogram utility.

The normalized distribution is calculated separately as:

```text
distribution[i] = histogram[i] / total_pixel_count
```

The cumulative distribution is calculated from the PDF as:

```text
cdf[i] = pdf[0] + pdf[1] + ... + pdf[i]
```

Histogram Y-values represent pixel counts; PDF and CDF Y-values represent probabilities. RGB images produce red, green, and blue channels. Grayscale images produce one grayscale channel only. React renders the backend arrays as charts; it does not calculate histogram values.

## Histogram equalization and transformations

Equalization calculates a lookup table for each channel using the 8-bit mapping:

```text
new_intensity = round(CDF[intensity] * 255)
```

The lookup table is applied to every pixel. Constant channels are preserved, and RGB images are equalized independently per channel. The equalization response includes the processed image plus histogram, PDF, and CDF data for both the original and equalized images. The frontend displays these results as before/after comparisons.

Uniform noise; normalization; thresholding; FFT/frequency-domain filters; hybrid images; and final report generation remain intentionally excluded until these stages are requested.
