"""Small compatibility benchmark for the new backend algorithms."""

import time

import cv2

from backend.app.algorithms.filters.average import average_filter
from backend.app.algorithms.filters.gaussian import gaussian_filter
from backend.app.algorithms.filters.median import median_filter
from backend.app.algorithms.noise.gaussian import gaussian_noise
from backend.app.algorithms.noise.salt_pepper import salt_pepper_noise

# create outputs folder if it doesn't exist
import os

os.makedirs('outputs', exist_ok=True)

# 1. load test image
loaded = cv2.imread('test.jpg')
img = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB) if loaded is not None else None

if img is None:
    print("Error: test.jpg not found! Make sure it is in the root folder.")
else:
    print(f"Image loaded: {img.shape[1]}x{img.shape[0]} | Channels: {img.shape[2] if len(img.shape) == 3 else 1}")
    print("-" * 55)

    # --------------------------------------------------
    # 1. Noise Generation Benchmarks
    # --------------------------------------------------
    print("1. Generating Noisy Images:")

    t0 = time.perf_counter()
    noisy_gauss = gaussian_noise(img, sigma=25)
    t_gauss_noise = time.perf_counter() - t0
    print(f"   [+] Gaussian Noise:        {t_gauss_noise:.4f} sec")

    t0 = time.perf_counter()
    noisy_sp = salt_pepper_noise(img, density=0.05, salt_probability=0.5)
    t_sp_noise = time.perf_counter() - t0
    print(f"   [+] Salt & Pepper Noise:   {t_sp_noise:.4f} sec")

    cv2.imwrite('outputs/noisy_gaussian.jpg', cv2.cvtColor(noisy_gauss, cv2.COLOR_RGB2BGR))
    cv2.imwrite('outputs/noisy_sp.jpg', cv2.cvtColor(noisy_sp, cv2.COLOR_RGB2BGR))

    print("-" * 55)

    # --------------------------------------------------
    # 2. Filters on Gaussian Noise Benchmarks
    # --------------------------------------------------
    print("2. Filtering Gaussian Noise:")

    t0 = time.perf_counter()
    gauss_avg = average_filter(noisy_gauss, kernel_size=5)
    t_avg_5 = time.perf_counter() - t0
    print(f"   [+] Average Filter (k=5):  {t_avg_5:.4f} sec")

    t0 = time.perf_counter()
    gauss_filtered = gaussian_filter(noisy_gauss, kernel_size=5, sigma=1.2)
    t_gauss_filt = time.perf_counter() - t0
    print(f"   [+] Gaussian Filter (k=5): {t_gauss_filt:.4f} sec")

    cv2.imwrite('outputs/gauss_filtered_by_avg.jpg', cv2.cvtColor(gauss_avg, cv2.COLOR_RGB2BGR))
    cv2.imwrite('outputs/gauss_filtered_by_gaussian.jpg', cv2.cvtColor(gauss_filtered, cv2.COLOR_RGB2BGR))

    print("-" * 55)

    # --------------------------------------------------
    # 3. Filters on Salt & Pepper Benchmarks
    # --------------------------------------------------
    print("3. Filtering Salt & Pepper Noise:")

    t0 = time.perf_counter()
    sp_avg = average_filter(noisy_sp, kernel_size=3)
    t_avg_3 = time.perf_counter() - t0
    print(f"   [+] Average Filter (k=3):  {t_avg_3:.4f} sec")

    t0 = time.perf_counter()
    sp_median = median_filter(noisy_sp, kernel_size=3)
    t_median_3 = time.perf_counter() - t0
    print(f"   [+] Median Filter (k=3):   {t_median_3:.4f} sec")

    cv2.imwrite('outputs/sp_filtered_by_avg.jpg', cv2.cvtColor(sp_avg, cv2.COLOR_RGB2BGR))
    cv2.imwrite('outputs/sp_filtered_by_median.jpg', cv2.cvtColor(sp_median, cv2.COLOR_RGB2BGR))

    print("-" * 55)
    print("Done! Check your console output to identify the slowest function.")
