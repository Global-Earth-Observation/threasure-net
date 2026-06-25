"""Dataset datatypes"""
import dataclasses
from copy import deepcopy

from dataclasses import dataclass
from typing import Self

import torch
from torch._C import Size
from torch.nn import functional as F


@dataclass(frozen=True)
class DeepTreeSingleTileConfig:
    """
    Dataset paramters.
    """

    bands: list[str] | None = (
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
    )
    input_resolution: int = 10
    target_resolution: float = 1
    min_valid_pixels: int = 100
    max_masked_pixels: int = 75
    min_target_valid: int = 1
    min_len_sits: int = 5
    max_len_sits: int | None = None
    cropped_s2_patch_size: int = 64  # without margin
    lidar_patch_size: int = 1000  # TODO: move to constants
    margin_s2: int | None = 16  # TODO: get it from model
    lidarhd_exclude_values: list[float] | None = (-1, 0)
    nodata_s2: int | None = -10000


@dataclass(frozen=True)
class DeepTreeDataModuleConfig:
    """
    Data module config.

    Attributes
    ----------
    db_folder: str
    lidar_folder: str
    s2_folder: str
    tiles: List[str] | None = None
    lidarhd_metrics: str
    testing_tiles: List[str] | None = None
    max_patches_per_site: int | None = None
    single_tile_dataset_config: DeepTreeSingleTileConfig
    prefetch_factor: int
    cache_validation_dataset : bool = False
    cache_testing_dataset : bool = False
    batch_size: int = 32
    testing_validation_batch_size: int = 128
    num_workers: int = 10
    """

    db_folder: str
    lidar_folder: str
    s2_folder: str
    lidarhd_metrics: str
    single_tile_dataset_config: DeepTreeSingleTileConfig
    prefetch_factor: int
    angles_folder: str | None = None
    tiles: list[str] | None = None
    batch_size: int = 32
    testing_validation_batch_size: int = 128
    num_workers: int = 4
    lidar_mask_folder: str | None = None
    testing_tiles: list[str] | None = None
    max_patches_per_site: int | None = None
    cache_validation_dataset: bool = False
    cache_testing_dataset: bool = False
    stats_computation: bool = False
    ref_date: str = "2014-03-03"


@dataclass
class Prediction:
    """Prediction"""
    height: torch.Tensor
    logits: torch.Tensor | None = None
    classification: torch.Tensor | None = None

    def __post_init__(self) -> Self:
        """Computes classes from logits"""
        if self.logits is not None:
            pred_label = (F.sigmoid(self.logits) > 0.5).float()
            setattr(self, "classification", pred_label)
        return self

    def __getitem__(self, item: int | list[int]) -> Self:
        """Get the same slice (batch elements) for each field (tensor)
        of the data class"""
        for key, value in dataclasses.asdict(self).items():
            setattr(self, key, value[item] if value is not None else None)
        return self

    def detach(self) -> Self:
        """Detach gradient"""
        for key, value in dataclasses.asdict(self).items():
            setattr(self, key, value.detach() if value is not None else None)
        return self

    @property
    def shape(self) -> Size:
        """Get shape"""
        return self.height.shape


@dataclass
class BatchData:
    """Single element in batch.

    Attributes
    ----------
    input_tensor: torch.Tensor
    target_tensor: torch.Tensor
    target_tensor_mask: torch.Tensor | None
    """

    input_tensor: torch.Tensor
    target_tensor: torch.Tensor | None = None
    input_tensor_mask: torch.Tensor | None = None
    target_tensor_mask: torch.Tensor | None = None
    mask_no_veg: torch.Tensor | None = None
    full_mask: torch.Tensor | None = None
    name_s2: list | str | None = None
    name_lidar: list | str | None = None
    doy_s2: torch.Tensor | None = None
    doy_lidar: torch.Tensor | int | None = None
    pad_mask: torch.Tensor | None = None
    angles: torch.Tensor | None = None

    def to(self, device: torch.device) -> Self:
        """Transport to device.
        Parameters
        ----------
        device=torch.device

        Return
        ------
        Self
        """
        for field in dataclasses.fields(self.__class__):
            value = getattr(self, field.name)
            if (value is not None) and (type(value) is not list):
                setattr(self, field.name, value.to(device))
        return self

    def pin_memory(self) -> Self:
        """Pin batch memory."""
        for field in dataclasses.fields(self.__class__):
            value = getattr(self, field.name)
            if (value is not None) and (type(value) is not list):
                setattr(self, field.name, value.pin_memory())
        return self

    def __getitem__(self, item: int | list[int]) -> Self:
        """Get the same slice (batch elements) for each field (tensor)
        of the data class"""
        new = deepcopy(self)
        for key, value in dataclasses.asdict(self).items():
            if value is not None:
                if (type(value) is not list) and (type(item) is not int):
                    setattr(new, key, value[item])
                else:
                    setattr(new, key, [value[i] for i in item])
            else:
                setattr(new, key, None)
        return new
