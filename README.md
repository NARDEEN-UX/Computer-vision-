# Computer Vision Task 1 — Current Stage

This project currently implements image upload, Gaussian noise, salt-and-pepper noise, average filtering, Gaussian filtering, median filtering, edge detection with Sobel, Roberts, Prewitt, and Canny, RGB/grayscale conversion, histogram equalization, and min-max image normalization. The interface supports applying later operations to the latest generated result and comparing histograms before and after every Transform & Enhancement operation.

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
| `POST /api/transform/to-gray` | no parameters; converts RGB to grayscale and returns before/after histograms |
| `POST /api/transform/to-rgb` | no parameters; converts grayscale to RGB and returns before/after histograms |
| `POST /api/transform/normalize` | `min_val`, `max_val`; maps the selected input range to 0–255 and returns before/after histograms |
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

Equalization currently calculates a lookup table for each channel using the conventional CDF-min 8-bit mapping:

```text
new_intensity = round(((CDF[intensity] - CDF_min) / (1 - CDF_min)) * 255)
```

`CDF_min` is the first non-zero CDF value. The lookup table is applied to every pixel. Constant channels are preserved, and RGB images are equalized independently per channel. Every Transform & Enhancement response includes the processed image plus histogram, PDF, and CDF data for both the original and processed images. The frontend displays these results as before/after comparisons with shared channel toggles and histogram scale controls.

Normalization uses min-max scaling:

```text
normalized = (image - min_val) / (max_val - min_val)
```

The normalized values are clipped to `[0, 1]` and scaled to `[0, 255]` for PNG display and histogram analysis.

### Why the CDF-min mapping was selected

The first working version used the direct mapping `round(CDF[intensity] * 255)`, which is a valid monotonic mapping but does not remove the probability mass below the first occupied intensity. As a result, the darkest occupied input value can map to a value above `0`, so the output may not use the available display range effectively. The selected mapping subtracts the first non-zero CDF value and normalizes by the remaining CDF range. It maps the first occupied intensity toward `0` and the last occupied intensity toward `255`, making contrast expansion easier to observe while preserving the same histogram → PDF → CDF → lookup-table workflow. Constant channels are handled separately because they contain no contrast to redistribute.

## Contribution summary

- Added custom 256-bin histogram, PDF, and CDF analysis for grayscale and RGB images.
- Added histogram equalization for grayscale images and independent RGB channels.
- Added RGB-to-grayscale and grayscale-to-RGB conversion operations.
- Added min-max image normalization with configurable input bounds.
- Added before/after histogram, PDF, and CDF data to all Transform & Enhancement operations.
- Added latest-result processing so operations can be chained from the generated image.
- Added interactive chart channel toggles, linear/log histogram scaling, and responsive before/after layouts.
- Added backend algorithm/API tests and frontend production-build validation.

Uniform noise; thresholding; FFT/frequency-domain filters; hybrid images; and final report generation remain intentionally excluded until these stages are requested.
