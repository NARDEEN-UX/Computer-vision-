import os
import cv2
import time
from functions.noise import gaussian_noise, snp_noise
from functions.filters import average_filter, gaussian_filter, median_filter

# create outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# 1. load test image
img = cv2.imread('test.jpg')

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
    noisy_sp = snp_noise(img, density=0.05, ratio=0.5)
    t_sp_noise = time.perf_counter() - t0
    print(f"   [+] Salt & Pepper Noise:   {t_sp_noise:.4f} sec")

    cv2.imwrite('outputs/noisy_gaussian.jpg', noisy_gauss)
    cv2.imwrite('outputs/noisy_sp.jpg', noisy_sp)

    print("-" * 55)

    # --------------------------------------------------
    # 2. Filters on Gaussian Noise Benchmarks
    # --------------------------------------------------
    print("2. Filtering Gaussian Noise:")

    t0 = time.perf_counter()
    gauss_avg = average_filter(noisy_gauss, k_size=5)
    t_avg_5 = time.perf_counter() - t0
    print(f"   [+] Average Filter (k=5):  {t_avg_5:.4f} sec")

    t0 = time.perf_counter()
    gauss_filtered = gaussian_filter(noisy_gauss, k_size=5, sigma=1.2)
    t_gauss_filt = time.perf_counter() - t0
    print(f"   [+] Gaussian Filter (k=5): {t_gauss_filt:.4f} sec")

    cv2.imwrite('outputs/gauss_filtered_by_avg.jpg', gauss_avg)
    cv2.imwrite('outputs/gauss_filtered_by_gaussian.jpg', gauss_filtered)

    print("-" * 55)

    # --------------------------------------------------
    # 3. Filters on Salt & Pepper Benchmarks
    # --------------------------------------------------
    print("3. Filtering Salt & Pepper Noise:")

    t0 = time.perf_counter()
    sp_avg = average_filter(noisy_sp, k_size=3)
    t_avg_3 = time.perf_counter() - t0
    print(f"   [+] Average Filter (k=3):  {t_avg_3:.4f} sec")

    t0 = time.perf_counter()
    sp_median = median_filter(noisy_sp, k_size=3)
    t_median_3 = time.perf_counter() - t0
    print(f"   [+] Median Filter (k=3):   {t_median_3:.4f} sec")

    cv2.imwrite('outputs/sp_filtered_by_avg.jpg', sp_avg)
    cv2.imwrite('outputs/sp_filtered_by_median.jpg', sp_median)

    print("-" * 55)
    print("Done! Check your console output to identify the slowest function.")