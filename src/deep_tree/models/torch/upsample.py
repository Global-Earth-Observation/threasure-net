"""Upsample module"""

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


def icnr_init(
    tensor: torch.Tensor,
        scale: int = 2,
        initializer: nn.init = nn.init.kaiming_normal_,
        noise_std: float = 1e-3
) -> None:
    """ICNR initialization for sub-pixel convolution weights."""
    out_channels, in_channels, h, w = tensor.shape
    num_subchannels = out_channels // (scale**2)
    subkernel = torch.zeros((num_subchannels, in_channels, h, w))
    initializer(subkernel)
    subkernel = subkernel.repeat_interleave(scale**2, dim=0)
    if noise_std > 0:
        subkernel += torch.randn_like(subkernel) * noise_std
    tensor.data.copy_(subkernel).to(torch.float32)


def apply_icnr(model: nn.Module, scale: int = 2) -> None:
    """Recursively apply ICNR init to convs before PixelShuffle layers."""
    prev_module = None
    for m in model.modules():
        if isinstance(m, nn.PixelShuffle) and isinstance(prev_module, nn.Conv2d):
            icnr_init(prev_module.weight, scale=scale)
        prev_module = m


class Interpolate(torch.nn.Module):
    """
    Interpolation class.
    Created so we can integrate it in nn.Sequential
    """
    def __init__(self, scale_factor: int = 2, mode: str = "nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
        )


class UpsamleFeatures(nn.Module):
    """Class for feature upsampling."""

    def __init__(
        self,
        conv_in: int,
        conv_out: int | None,
        upsampling_factor: int,
        type: Literal["shuffle", "interpolation", "transposed"] = "shuffle",
        init_shuffle: bool = False,
        activate_shuffle: bool = True
    ) -> None:
        super().__init__()

        self.upsampling_factor = upsampling_factor
        assert self.upsampling_factor % 2 == 0

        if conv_out is None:
            conv_out = conv_in

        if type == "shuffle":
            layers = []
            for _ in range(0, self.upsampling_factor, 2):
                layers.append(nn.Conv2d(conv_in, conv_in * 4, 3, padding=1))
                layers.append(nn.PixelShuffle(upscale_factor=2))
                if activate_shuffle:
                    layers.append(nn.ReLU())
            layers.append(nn.Conv2d(conv_in, conv_out, 3, padding=1))
            layers.append(nn.ReLU())
            self.upsample = nn.Sequential(*layers)
            if init_shuffle:
                apply_icnr(self.upsample)

        elif type == "interpolation":
            layers = []
            for _ in range(0, self.upsampling_factor, 2):
                layers.append(nn.Conv2d(conv_in, conv_in, 3, padding=1))
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                layers.append(Interpolate(scale_factor=2))
            layers.append(nn.Conv2d(conv_in, conv_in, 3, padding=1))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.upsample = nn.Sequential(*layers)
        elif type == "transposed":
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(conv_in, conv_in, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(conv_in, conv_out, 3, padding=1),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward upsample"""
        return self.upsample(data)
