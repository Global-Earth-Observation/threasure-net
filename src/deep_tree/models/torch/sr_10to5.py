#!/usr/bin/env python

# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
SuperResolution Inference Model
This is a legacy module.
It is used only in case we do not use SR block of our module,
but use SISR code of Julien Michel to use SR images as model input.
"""
import os.path
from pathlib import Path
from typing import Tuple

import hydra
import torch
import torch.nn.functional as F
from einops import repeat
from omegaconf import OmegaConf
from torch import nn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CHECKPOINT = os.path.join(Path(__file__).parent.parent,
                          "carn_model/carn_3x3x64g4sw_bootstrap_small.ckpt")
CONFIG = os.path.join(Path(__file__).parent.parent,
                      "carn_model/carn_3x3x64g4sw_bootstrap_small.yaml")


class PreprocessingModule(torch.nn.Module):
    """
    Preprocessing module in charge of data normalization for export
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Handles standardization and conversion from millirefl
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward
        """
        data = data / 10000
        data = (data - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return data


class PostProcessingModule(torch.nn.Module):
    """
    Post-processing module in charge of data de-normalization for export
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Handles unstardardization and conversion to millirefl
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward
        """
        data = self.mean[None, :, None, None] + (data * self.std[None, :, None, None])
        return 10000 * data


class InferenceSR(torch.nn.Module):
    """CARN SR Inference"""

    def __init__(self,
                 checkpoint: str = CHECKPOINT, config: str = CONFIG,
                 margin: int = 32, nodata: int = -10000):
        """Init from path to checkpoint and hydra config"""
        super().__init__()
        carn_sr, (mean, std) = self.init_model(checkpoint, config)

        preprocessing = PreprocessingModule(
            mean,
            std,
        )
        postprocessing = PostProcessingModule(
            mean,
            std,
        )
        self.net = torch.nn.Sequential(preprocessing, carn_sr.sisr_model, postprocessing)
        self.net.eval()

        self.margin = margin
        self.nodata = nodata

    @staticmethod
    def init_model(
            ckpt_path: str,
            cfg_path: str
    ) -> (torch.nn.Module, Tuple[torch.Tensor, torch.Tensor]):
        """Initialize SR model"""
        checkpoint = torch.load(
            ckpt_path, weights_only=False
        )
        model_checkpoint = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not (
                    k.startswith("discriminator")
                    or k.startswith("mean")
                    or k.startswith("std")
            )
        }


        config = OmegaConf.load(cfg_path)

        # We instantiate the checkpoint configuration
        srnet = hydra.utils.instantiate(config.model.model)
        srnet.load_state_dict(model_checkpoint, strict=False)

        mean = torch.tensor(config.training_module.standardization_parameters.mean)
        std = torch.tensor(config.training_module.standardization_parameters.std)

        return srnet, (mean, std)

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        data = data[:, [0, 1, 2, 6, 3, 4, 5, 7, 8, 9], ...] # TODO: get from config?
        nodata_mask, nodata_mask_sr = self.get_nodata(data)
        data[nodata_mask] = 0
        # with highest_matmul_precision():
        with torch.no_grad():
            sr = self.net(data)
            # TODO get rid of mask bad values one day
            mask_bad_values = repeat((sr < -100).sum(1) > 0, 'b h w -> b c h w', c=sr.shape[1])
            nodata_mask_sr = (nodata_mask_sr + mask_bad_values) > 0
            sr[nodata_mask_sr] = self.nodata
            sr = sr[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9], ...]
            if self.margin > 0:
                return sr[:, :, self.margin:-self.margin, self.margin:-self.margin]
            return sr

    def get_nodata(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mask = ((data == self.nodata).sum(1) + (data < 0).sum(1)) > 0
        mask = repeat(mask, 'b h w -> b c h w', c=data.shape[1])
        mask_sr = F.interpolate(
            mask.to(torch.float32),
            scale_factor=2,
            mode='nearest'
        ) > 0
        mask_sr = mask_sr + F.max_pool2d(
            mask_sr.to(torch.float16), kernel_size=3, stride=1, padding=1
        ).to(torch.bool)
        return mask, mask_sr


class SRSITSWrapper(nn.Module):
    """Wrap Single Image SR for (padded) SITS"""
    def __init__(self,
                 model: InferenceSR = InferenceSR()):
        super().__init__()

        self.margin = model.margin if model.margin is not None else 0
        self.model: InferenceSR = model

    # def forward(self, data: torch.Tensor,
    # pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    #     b, n, _, h, w = data.shape
    #     data = rearrange(data, "b n c h w -> (b n ) c h w")
    #     if pad_mask is not None:
    #         # We ignore the padded values
    #         mask = rearrange(pad_mask, "b n -> (b n )")
    #         data = self.model(data[~mask])
    #         data_res = torch.zeros(
    #             b * n,
    #             data.shape[1],
    #             h - self.margin * 2,
    #             w - self.margin * 2
    #         ).to(data.device)
    #         data_res[~mask] = data
    #         data = data_res
    #     else:
    #         data = self.model(data)
    #     return rearrange(data, "( b n ) c h w -> b n c h w ", b=b)

    def forward(
            self, data: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass"""
        b, n, c, h, w = data.shape
        data_ = torch.zeros(
            b,
            n,
            c,
            int(h*2 - self.margin*2),
            int(w*2 - self.margin*2)
        ).to(data.device)
        for b_ in range(b):
            data_res_b = torch.zeros(
                n,
                c,
                int(h*2 - self.margin*2),
                int(h*2 - self.margin * 2)
            ).to(data.device)
            if pad_mask is not None:
                # We ignore the padded values
                data_ = self.model(data[b_][~pad_mask[b_]])
                data_res_b[~pad_mask[b_]] = data_
                data_[b_] = data_res_b
            else:
                data_[b_] = self.model(data[b_])
        return data_
