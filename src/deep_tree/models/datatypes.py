import dataclasses
import typing
from copy import deepcopy
from dataclasses import dataclass

import torch

from deep_tree.models.torch.deeptree_sits import DeepTreeSITSModel
from torchsisr.loss_helper import PixelLossWrapper


@dataclass
class OptimizationParameters:
    """
    Optimization parameters.

    Attributes
    ----------
    learning_rate: float
    t_0: int
    t_mul: float
    decay_factor: float
    type: str
    """

    learning_rate: float
    t_0: int
    t_mult: float
    decay_factor: float | None = None
    type: str = "cos_warm_restart"


@dataclass
class StandardizationParameters:
    """
    Standardization parameters.

    Attributes
    ----------
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    bands: Tuple[str, ...] | None = None
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]
    bands: tuple[str, ...] | None = None

    def __post_init__(self):
        assert len(self.mean) == len(self.std)
        if self.bands is not None:
            if self.bands == ["STACK_10"]:
                assert len(self.mean) == 10
                assert len(self.std) == 10
            else:
                assert len(self.bands) == len(self.mean)
                assert len(self.bands) == len(self.std)


@dataclass
class DeepTreeTrainingModuleConfig:
    """
    Training module config.

    Attributes
    ----------
    optimization: OptimizationParameters
    standardization: StandardizationParameters
    model: DeepTreeSITSModel
    real_losses: tuple[PixelLossWrapper, ...]
    validation_metrics: tuple[PixelLossWrapper, ...]
    test_metrics: tuple[PixelLossWrapper, ...]
    target_standardization: StandardizationParameters | None = None
    """

    optimization: OptimizationParameters
    standardization: StandardizationParameters
    model: DeepTreeSITSModel
    real_losses: tuple[PixelLossWrapper, ...]
    validation_metrics: tuple[PixelLossWrapper, ...]
    test_metrics: tuple[PixelLossWrapper, ...]
    target_standardization: StandardizationParameters | None = None
    target_resolution: float = 1


@dataclass
class PredVisu:
    input_s2: list[torch.Tensor]
    pred: torch.Tensor
    target: torch.Tensor
    input_sr: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    dates_s2: list | None = None
    dates_lidar: list | None = None
    name_s2: list | None = None
    name_lidar: list | None = None
    classif: torch.Tensor | None = None

    def __getitem__(self, item: int | list[int]) -> typing.Self:
        """Get the same slice (batch elements) for each field (tensor)
        of the data class"""
        new = deepcopy(self)
        for key, value in dataclasses.asdict(self).items():
            if value is not None:
                setattr(new, key, value[item])
        return new
