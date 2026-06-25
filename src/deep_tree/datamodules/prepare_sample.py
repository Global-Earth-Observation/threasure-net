"""Open Lidar and S2 data, randomly crop them and prepare batch"""

import os

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from deep_tree.datamodules.datatypes import BatchData


def read_tiff_to_tensor(path: str, window: Window | None = None) -> torch.Tensor:
    """Read geotiff as tensor"""
    with rasterio.open(path) as src:
        data = src.read(window=window)
    return torch.from_numpy(data)


class RandomPatchCrop:
    def __init__(
        self,
        lidar_folder: str,
        lidar_mask_folder: str,
        s2_folder: str,
        cropped_s2_patch_size: int = 64,  # without margin
        lidar_patch_size: int = 1000,
        margin_s2: int = 16,
        exclude_values_lhd: list = [-1, 0],
        nodata_s2: int = -10000,
        s2_resolution: float = 10.0,
        lidar_resolution: float = 1.0,
        training: bool = True,
    ):
        """Class to open Lidar and S2 data, randomly crop them and prepare batch"""
        self.lidar_folder = os.path.join(lidar_folder, f"res_{str(lidar_resolution)}")
        self.lidar_mask_folder = lidar_mask_folder
        self.s2_folder = s2_folder

        self.cropped_s2_size = cropped_s2_patch_size
        self.scale = s2_resolution / lidar_resolution
        self.cropped_lidar_size = int(cropped_s2_patch_size * self.scale)

        self.lidar_size = lidar_patch_size
        self.margin_s2 = margin_s2
        self.exclude_values_lhd = exclude_values_lhd
        self.nodata_s2 = nodata_s2

        self.training = training

    def get_random_coords(
        self,
    ) -> tuple[int, int]:
        """Get random x,y offsets for patch cropping. If validation, it is always central patch."""
        if not self.training:
            return int((self.lidar_size - self.cropped_lidar_size) / 2), int(
                (self.lidar_size - self.cropped_lidar_size) / 2
            )

        # S2 patches are prepared with a 25-pixel margin in height and width compared to LiDAR patches.
        # This margin accounts for edge effects caused by 2D convolutions (model margin).
        # If the model margin exceeds 25, we adjust random cropping to prevent shape mismatches.
        if self.margin_s2 > 25:
            diff = self.margin_s2 - 25
            diff = diff * self.scale
            possible = np.arange(
                diff, self.lidar_size - self.cropped_lidar_size + 1 - diff, self.scale
            )  # we sample on S2 10m grid
        else:
            possible = np.arange(
                0, self.lidar_size - self.cropped_lidar_size + 1, self.scale
            )  # we sample on S2 10m grid
        x_c, y_c = np.random.choice(possible, size=2, replace=True)
        return x_c, y_c

    def get_lidar(
        self, path: str, x_c: int, y_c: int, path_mask: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cropped lidar patch"""
        window = Window(
            col_off=x_c,
            row_off=y_c,
            width=self.cropped_lidar_size,
            height=self.cropped_lidar_size,
        )
        data = read_tiff_to_tensor(path, window).to(torch.float32)
        # data = read_tiff_to_tensor(path)[:, y_c: y_c + cropped_patch_size, x_c: x_c + cropped_patch_size]
        if path_mask is not None:
            mask = read_tiff_to_tensor(path_mask, window)
            assert torch.all(
                mask[torch.isin(data, torch.Tensor(self.exclude_values_lhd))] == 1
            )
        else:
            mask = torch.isin(data, torch.Tensor(self.exclude_values_lhd))

        return data, mask

    def get_s2(
        self,
        path: str,
        x_c: int,
        y_c: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cropped S2 patch"""
        # TODO decide what to do with S2 margins
        window = Window(
            col_off=x_c + 25 - self.margin_s2,
            row_off=y_c + 25 - self.margin_s2,
            width=self.cropped_s2_size + self.margin_s2 * 2,
            height=self.cropped_s2_size + self.margin_s2 * 2,
        )
        data = read_tiff_to_tensor(path, window)
        # data = read_tiff_to_tensor(path)[
        #        :,
        #        y_c - margin: y_c + cropped_patch_size + margin,
        #        x_c - margin: x_c + cropped_patch_size + margin
        #        ]
        mask = data == self.nodata_s2
        mask2 = (data == 0).sum(0) == data.shape[0]

        mask = (mask + mask2) > 0

        return data, mask

    def get_sample(
        self, name_lidar: str, names_s2: list[str], lidar_id: str, name_lidar_mask: str
    ) -> BatchData:
        """Get sample with cropped and aligned S2 and Lidar"""
        x_c, y_c = self.get_random_coords()
        x_c_s2, y_c_s2 = int(x_c / self.scale), int(y_c / self.scale)

        if self.lidar_mask_folder is not None and name_lidar_mask is not None:
            path_mask = os.path.join(self.lidar_mask_folder, name_lidar_mask)
        else:
            path_mask = None
        target_tensor, target_tensor_mask = self.get_lidar(
            os.path.join(self.lidar_folder, name_lidar), x_c, y_c, path_mask=path_mask
        )
        target_tensor_mask[target_tensor<0] = 1
        inputs = [
            self.get_s2(os.path.join(self.s2_folder, lidar_id, name_s2), x_c_s2, y_c_s2)
            for name_s2 in names_s2
        ]
        input_tensor, input_tensor_mask = zip(*inputs)
        input_tensor, input_tensor_mask = torch.stack(input_tensor, 0), torch.stack(
            input_tensor_mask, 0
        )
        # s2_mask =  input_tensor_mask.any(dim=1).sum(dim=0) >= len(input_tensor_mask) * 0.75
        # TODO integrate input tensor mask properly
        s2_mask = (input_tensor_mask.sum(0) >= len(input_tensor_mask) * 0.75).unsqueeze(0)
        return BatchData(
            input_tensor=input_tensor,
            input_tensor_mask=s2_mask,
            target_tensor=target_tensor,
            target_tensor_mask=target_tensor_mask,
            name_s2=names_s2,
            name_lidar=name_lidar,
        )
