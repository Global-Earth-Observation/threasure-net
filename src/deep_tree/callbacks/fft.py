#!/usr/bin/env python3
# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
""" Functions for FFT callbacks"""

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt


def fft_magnitude(img: np.array) -> np.array:
    """Compute log-scaled FFT magnitude."""
    f = fft2(img.astype(float))
    fshift = fftshift(f)
    mag = np.abs(fshift)
    return np.log10(mag)


def radial_profile(magnitude: np.array, mask: np.array = None) -> np.array:
    """Compute radial profile of FFT magnitude, optionally using a mask."""
    h, w = magnitude.shape
    y, x = np.indices((h, w))
    center = np.array([w // 2, h // 2])
    r = np.hypot(x - center[0], y - center[1])
    r_int = r.astype(int)

    if mask is not None:
        values = magnitude * mask
        counts = mask
    else:
        values = magnitude
        counts = np.ones_like(magnitude)

    radial_sum = np.bincount(r_int.ravel(), values.ravel())
    radial_count = np.bincount(r_int.ravel(), counts.ravel())
    radial_mean = radial_sum / np.maximum(radial_count, 1e-12)
    return radial_mean


def masked_fft_profile(sr_img: np.array, hr_img: np.array, mask: np.array = None) -> np.array:
    """Compute FFT-based radial profiles with mask correction."""
    # If mask exists, we fill pixels with imgage mean value
    if mask is not None:
        sr_fill = sr_img.copy()
        hr_fill = hr_img.copy()
        fill_value = sr_img[mask].mean() if mask.any() else sr_img.mean()
        sr_fill[~mask] = fill_value
        hr_fill[~mask] = fill_value
    else:
        sr_fill = sr_img
        hr_fill = hr_img

    # FFT magnitude
    sr_mag = fft_magnitude(sr_fill)
    hr_mag = fft_magnitude(hr_fill)

    # Mask for radial profiles
    mask_for_profile = mask.astype(float) if mask is not None else None

    # Radial profiles
    sr_profile = radial_profile(sr_mag, mask=mask_for_profile)
    hr_profile = radial_profile(hr_mag, mask=mask_for_profile)

    # L1 / L2 norms
    normalize = "l2"
    if normalize == "l1":
        sr_profile /= sr_profile.sum()
        hr_profile /= hr_profile.sum()
    elif normalize == "l2":
        sr_profile /= np.linalg.norm(sr_profile)
        hr_profile /= np.linalg.norm(hr_profile)

    # Metrics
    mse = np.mean((sr_profile - hr_profile) ** 2)
    cos_sim = np.dot(sr_profile, hr_profile) / (
            np.linalg.norm(sr_profile) * np.linalg.norm(hr_profile)
    )

    return sr_profile, hr_profile, mse, cos_sim


def plot_radial_profiles(
        sr_profile: np.array, hr_profile: np.array,
        mse: float, cos_sim: float
) -> None:
    """Plot radial profiles"""
    plt.figure(figsize=(7, 5))
    plt.plot(sr_profile, label="SR")
    plt.plot(hr_profile, label="HR")
    plt.xlabel("Radius")
    plt.ylabel("Normalized magnitude")
    plt.title(f"Radial FFT Profiles\nMSE={mse:.4e}, CosSim={cos_sim:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_profile_dif(sr_profile: np.array, hr_profile: np.array) -> None:
    """Plot radial profiles difference"""
    plt.plot(hr_profile - sr_profile)
    plt.title("Difference: HR - SR")
    plt.xlabel("Radius")
    plt.ylabel("Magnitude difference")
    plt.show()


def plot_high_freq(sr_profile: np.array, hr_profile: np.array) -> None:
    """Compute the index array for high frequencies"""
    high_freq_idx = np.arange(len(sr_profile) // 2, len(sr_profile))

    # Plot
    plt.plot(high_freq_idx, hr_profile[high_freq_idx], label="HR")
    plt.plot(high_freq_idx, sr_profile[high_freq_idx], label="SR")
    plt.xlabel("Radius")
    plt.ylabel("Magnitude")
    plt.title("High-frequency radial profiles")
    plt.legend()
    plt.show()

#
# if __name__ == "__main__":
#
#     path = "/your/path/dev/deep-tree/results_images/"
#     path_sr = (
#         path
#         + "Shuffle_pe_tq_PredSITS_int_percentiles_0384_6430_2023-07-21_0.95.tif_r2_0.67_mae_30.41"
#     )
#     path_hr = path + "Target_percentiles_0384_6430_2023-07-21_0.95.tif"
#
#     with rasterio.open(path_sr) as src:
#         sr_img = src.read(1)
#         h, w = sr_img.shape
#         sr_img = sr_img[int(h / 2) :, int(w / 2) :]
#
#     with rasterio.open(path_hr) as src:
#         hr_img = src.read(1)
#         hr_img = hr_img[int(h / 2) :, int(w / 2) :]
#
#     mask = sr_img != 0  # маска нулевых пикселей
#
#     sr_profile, hr_profile, mse, cos_sim = masked_fft_profile(
#         sr_img / sr_img.max(), hr_img / sr_img.max(), mask=mask
#     )
#
#     plot_radial_profiles(sr_profile, hr_profile, mse, cos_sim)
#
#     plot_profile_dif(sr_profile, hr_profile)
#
#     plot_high_freq(sr_profile, hr_profile)
