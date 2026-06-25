import os
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from torch import nn


def remove_small_components(binary_mask, min_size):
    """
    binary_mask: torch.BoolTensor of shape (H, W)
    min_size: int, minimum number of pixels to keep a component

    Returns a torch.BoolTensor of the same shape
    """

    mask_np = binary_mask.cpu().numpy().astype(np.uint8)
    labeled, num = label(mask_np)

    # Count sizes
    counts = np.bincount(labeled.ravel())
    remove = counts < min_size
    remove_mask = remove[labeled]
    mask_np[remove_mask] = 0
    return torch.from_numpy(mask_np).to(binary_mask.device).bool()


class FancyConv(nn.Module):
    def __init__(
        self,
        k: int,
    ):
        super().__init__()

        self.padding = k // 2
        self.filter = torch.ones(1, 1, k, k)

    def forward(self, image: torch.Tensor):
        image = F.pad(
            image,
            pad=(self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        return F.conv2d(image, self.filter, padding=0)


mask_path = "/your/path/input_data/fields_mask"
lidar_path = "/your/path/input_data/lidarHD/lidarHD_processed"
mask_path_output = "/your/path/input_data/lidarHD/lidarHD_mask_5m"

height_threshold_field = 50
height_threshold = 15

resolution_lidar = 1
resolution_s2 = 10
small_object_side_m = 5

TILES = os.listdir(lidar_path)
TILES = [
    "32TMN",
    "31TDG",
    "31TGJ",
    "31TEK",
    "30TYP",
    "32TNN",
    "31UDR",
    "30TXQ",
    "30UXV",
    "31TDN",
    "31TEJ",
    "31UCR",
    "32TLS",
    "31TCL",
    "31TEN",
    "31TDK",
    "31TFM",
    "31TFN",
    "30TXR",
    "32TML",
    "31TDH",
    "32ULU",
    "31TFH",
    "31TEM",
    "31UDQ",
    "31UFQ",
    "30TXN",
    "32TLP",
    "32UMU",
    "32ULV",
    "30TYN",
    "30TXS",
    "31UCP",
    "32TLN",
    "31TGK",
    "31UFP",
    "31UCQ",
    "31UEQ",
    "30TWP",
    "30TXT",
    "31UGQ",
    "31UDS",
    "31TGM",
    "31UGP",
    "31TFK",
    "31TFL",
    "30TXP",
    "32TLT",
    "31TEG",
    "30TWN",
    "32TLR",
    "31TCK",
    "30UXU",
    "32TNM",
    "31TDL",
    "30TYR",
    "31TGL",
    "32UMV",
    "32TLQ",
    "32TMM",
    "32TNL",
    "30UYU",
    "31UDP",
    "30TYT",
    "31UCS",
    "31TCJ",
    "30UVV",
    "31TGN",
    "31TFJ",
    "30TYQ",
    "31TCH",
    "30UYV",
    "30UVU",
    "31TEH",
    "31UEP",
    "31TDM",
    "31TGH",
    "31TEL",
    "30TYS",
    "31TDJ",
]


def compute_target_mask(tile):
    tile_folder = os.path.join(lidar_path, tile, f"res_{resolution_lidar}")
    patches = [
        patch
        for patch in os.listdir(tile_folder)
        if (patch.startswith("percentiles") and patch.__contains__("0.95"))
    ]

    mask_folder = os.path.join(mask_path_output, tile)
    Path(mask_folder).mkdir(parents=True, exist_ok=True)

    for patch in patches:
        lidar_id = re.search(r"percentiles_((\d{4})_(\d{4}))", patch).group(1)
        if not os.path.exists(
            os.path.join(mask_folder, f"lidar_mask_{lidar_id}_{tile}.tif")
        ):
            with rasterio.open(os.path.join(tile_folder, patch)) as src:
                image = torch.from_numpy(src.read()).to(torch.float32)
                kwds = src.profile
                kwds["dtype"] = "uint8"

                mask = torch.zeros(image.shape)
                mask[image == 0] = 1
                mask[image < 0] = 1

                small_mask = remove_small_components(
                    (~mask.bool()).to(torch.int8),
                    int(small_object_side_m / resolution_lidar) ** 2,
                ).to(torch.bool)

                mask[~small_mask] = 1

            k = 7
            conv5 = FancyConv(k)
            image_count = conv5(small_mask.to(torch.float32))

            height = conv5(image) / torch.clamp(image_count, min=1)

            mask_name = f"crop_mask_{lidar_id}_{tile}_10.tif"
            # print(os.path.join(mask_path, tile, mask_name))
            if os.path.exists(os.path.join(mask_path, tile, mask_name)):
                with rasterio.open(
                    os.path.join(mask_path, tile, mask_name)
                ) as src_mask:
                    mask_crop = torch.from_numpy(src_mask.read()).to(torch.int16)
                    print(mask_crop.shape)
                    h, w = mask_crop.shape[1:]
                    if not h == w == 1000:
                        mask_crop = mask_crop[:, :1000, :1000]
                new_mask = (image < height_threshold_field) & mask_crop
                mask[new_mask.bool()] = 1

            else:
                print("Not exists: ", os.path.join(mask_path, tile, mask_name))
            new_mask1 = (
                (height < height_threshold) & (image < height_threshold)
            )

            mask[new_mask1.bool()] = 1

            small_mask = remove_small_components(
                (~mask.bool()).to(torch.int8),
                int(small_object_side_m / resolution_lidar) ** 2,
            ).to(torch.bool)

            mask[~small_mask] = 1

            print(os.path.join(mask_folder, f"lidar_mask_{lidar_id}_{tile}.tif"))
            with rasterio.open(
                os.path.join(mask_folder, f"lidar_mask_{lidar_id}_{tile}.tif"),
                "w",
                **kwds,
            ) as dst:
                dst.write(mask)


compute_target_mask(TILES[int(sys.argv[1])])
