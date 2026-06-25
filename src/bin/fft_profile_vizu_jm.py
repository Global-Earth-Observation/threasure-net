import os

import torch
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from matplotlib import pyplot as plt

from torchsisr import fda


import matplotlib
matplotlib.use("Agg")

plt.rcParams["font.family"] = "Nimbus Roman"
plt.rcParams.update({'font.size': 22})



def read_image_as_tensor(path: str, sr_ref_path: str, add_noise: bool=False) -> torch.Tensor:
    """
    Read an image aligned to SR image using SR bounding box.
    Output is float32 Torch tensor [1, H, W] in range [0,1].
    """
    with rasterio.open(sr_ref_path) as src_sr:
        bbox = src_sr.bounds
        ref_transform = src_sr.transform

    with rasterio.open(path) as src:
        # Crop other image to SR extent
        win = from_bounds(*bbox, transform=src.transform)
        img = src.read(1, window=win).astype(np.float32)

    if add_noise:
        eps = 0.05
        img += np.random.normal(scale=eps, size=img.shape)


    # Normalization
    img = img/300.0

    # Convert to tensor
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return img.cuda()


def compute_fft_profile(img: torch.Tensor):
    """
    Wrapper for fda.compute_fft_profile.
    Ensures that freqs and profile are 1D arrays with same length.
    """
    _, freqs, profile = fda.compute_fft_profile(img, s=2*img.shape[-1])

    # freqs: shape could be (N,) or (1,N)
    freqs = freqs.squeeze().cpu().numpy()

    # profile: [B, N, C] → assume single image, single channel
    profile = profile.squeeze().cpu().numpy()  # gives shape (N,)

    return freqs, profile

def main(sr_path, hr_path, up_path):
    folder = "/work/scratch/data/kalinie/THREASURE/qgis_files_cl/"
    hrh_path=os.path.join(folder, "Target_percentiles_0384_6430_2023-07-21_0.95.tif")

    # --- Read images ---
    sr = read_image_as_tensor(sr_path, sr_path)
    hr = read_image_as_tensor(hr_path, sr_path)
    up = read_image_as_tensor(up_path, sr_path)
    hrh = read_image_as_tensor(hrh_path, sr_path)

    # --- Compute FFT radial profiles ---
    freqs_sr, prof_sr = compute_fft_profile(sr)
    freqs_hr, prof_hr = compute_fft_profile(hr)
    freqs_up, prof_up = compute_fft_profile(up)
    freqs_hrh, prof_hrh = compute_fft_profile(hrh)

    # Convert to log-profile (exact same logic as original script)
    def log_profile(p):
        return 10 * np.log10(p) - 10 * np.log10(p[1])

    log_sr = log_profile(prof_sr)
    log_hr = log_profile(prof_hr)
    log_up = log_profile(prof_up)
    log_hrh = log_profile(prof_hrh)
    # log_sr = log_sr/0.5
    # log_hr = log_hr / 0.5
    # log_up =log_up / 0.5

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    # plt.plot(freqs_hr, log_hr, label="HR reference", linewidth=2, color="blue")
    # plt.plot(freqs_sr, log_sr, label=f"SR", linestyle="--", color="red")
    # plt.plot(freqs_up, log_up, label="Bicubic", linestyle=":", color="green")

    plt.plot(freqs_hrh, log_hrh, label="HR reference", linewidth=2, color="blue")
    plt.plot(freqs_sr, log_sr, label="Our", linewidth=2, color="red")
    plt.plot(freqs_up, log_up, label="MAE", linewidth=2, color="green")
    plt.plot(freqs_hr, log_hr, label="pwMAE", linewidth=2, color="brown")

    plt.xlabel("Spatial frequency, 1/px")
    plt.ylabel("Attenuation, dB")
    plt.grid(True)
    plt.legend()
    plt.ylim(-30, 0)
    # plt.title("FFT Radial Profiles")
    plt.tight_layout()
    plt.savefig("./fda_5m_ablation.pdf", dpi=500, format="pdf")
    plt.show()

# if __name__ == "__main__":
#     folder = "/work/scratch/data/kalinie/THREASURE/qgis_files_cl/"
#     main(
#         sr_path=os.path.join(folder, "Height_f2_percentiles_0384_6430_2023-07-21_0.95_r2_0.85_mae_18.06_mape_0.17_rmse_25.58.tif"),
#         hr_path=os.path.join(folder, "Target_percentiles_0384_6430_2023-07-21_0.95.tif"),
#         up_path=os.path.join(folder, "10m_upsampled_to_5m.tif"),
#     )


if __name__ == "__main__":
    folder = "/work/scratch/data/kalinie/THREASURE/qgis_files_cl/"
    main(
        sr_path=os.path.join(folder, "Height_f2_percentiles_0384_6430_2023-07-21_0.95_r2_0.85_mae_18.06_mape_0.17_rmse_25.58.tif"),
        hr_path=os.path.join(folder, "Height_f2_pmae_percentiles_0384_6430_2023-07-21_0.95_r2_0.85_mae_18.19_mape_0.18_rmse_26.01.tif"),
        up_path=os.path.join(folder, "Height_f2_mmae_percentiles_0384_6430_2023-07-21_0.95_r2_0.85_mae_18.29_mape_0.17_rmse_25.66.tif"),
    )

# if __name__ == "__main__":
#     folder = "/work/scratch/data/kalinie/THREASURE/qgis_files_cl/"
#     main(
#         sr_path=os.path.join(folder, "Height_f4_percentiles_0384_6430_2023-07-21_0.95_r2_0.8_mae_19.9_mape_0.23_rmse_29.23.tif"),
#         hr_path=os.path.join(folder, "percentiles_0384_6430_2023-07-21_0.95_2.5m.tif"),
#         up_path=os.path.join(folder, "5m_upsampled_to_2_5m.tif"),
#     )
    # "Height_f2__percentiles_0384_6430_2023-07-21_0.95_r2_0.86_mae_17.58_mape_0.17_rmse_25.18"