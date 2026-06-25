import os

import hydra
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig

from deep_tree.datamodules.datatypes import BatchData

torch.set_float32_matmul_precision('high')


def get_model(config_path: str) -> tuple[DictConfig, nn.Module]:
    """Super-regression model."""
    mean_s2 = torch.Tensor([319.0, 607.0, 479.0, 3098.0, 1020.0, 2450.0, 2877.0, 3205.0, 1980.0, 1029.0])
    std_s2 = torch.Tensor([521.0, 791.0, 877.5, 3025.5, 1197.0, 2369.0, 2830.5, 3115.0, 2074.0, 1354.0])

    model_cfg = OmegaConf.load(config_path)

    # recreate the real Hydra tree
    config = OmegaConf.create({
        "model": model_cfg
    })

    if config.model.model.sr_10to5 is not None:
        sr_10to5 = hydra.utils.instantiate(config.model.model.sr_10to5)
    else:
        sr_10to5 = None

    spatio_spectral_encoder = hydra.utils.instantiate(
        config.model.model.spatio_spectral_encoder
    )

    temporal_encoder = hydra.utils.instantiate(
        config.model.model.temporal_encoder
    )


    if config.model.model.upsample_module is not None:
        upsample_module = hydra.utils.instantiate(
            config.model.model.upsample_module
        )
    else:
        upsample_module = None

    assert (
        (upsample_module is None and sr_10to5 is not None)
        or
        (upsample_module is not None and sr_10to5 is None)
    )

    regression = hydra.utils.instantiate(
        config.model.model.regression
    )

    superregression = hydra.utils.instantiate(
        config.model.model,
    )

    superregression.mean = mean_s2
    superregression.std = std_s2

    return config, superregression


def dummy_batch(dummy_padding: bool = False) -> BatchData | tuple[BatchData, BatchData]:
    """Get dummy batch with temporal padding or not"""
    b, t, c, h, w = [2, 2, 10, 64, 64]
    input_tensor = torch.randint(low=0, high=10000, size=[b, t, c, h, w])
    target_tensor = torch.randint(low=0, high=30, size=[b, 1, h * 2, w * 2])
    angles = torch.rand(size=[b, t, 6])
    # input_tensor = torch.randn(b, t, c, h, w)
    # target_tensor = torch.randn(b, t, c, h * 2, w * 2)
    doy_s2 = torch.randint(low=0, high=366, size=[b, t]).sort(dim=1).values
    doy_lidar = torch.randint(low=0, high=366, size=[b])

    if dummy_padding:
        p = 1
        pad_mask = torch.zeros(size=[b, t])
        return BatchData(
            input_tensor=input_tensor,
            target_tensor=target_tensor,
            target_tensor_mask=None,
            doy_lidar=doy_lidar,
            doy_s2=doy_s2,
            pad_mask=None,
            angles=angles
        ), BatchData(
            input_tensor=F.pad(input_tensor, (0, 0, 0, 0, 0, 0, 0, p)),
            target_tensor=target_tensor,
            target_tensor_mask=None,
            doy_lidar=doy_lidar,
            doy_s2=F.pad(doy_s2, (0, p)),
            pad_mask=F.pad(pad_mask, (0, p), value=1).to(torch.bool),
            angles=F.pad(angles, (0, 0, 0, p))
        )

    return BatchData(
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        target_tensor_mask=None,
        doy_lidar=doy_lidar,
        doy_s2=doy_s2,
        angles=angles
    )


def get_configs() -> list[str]:
    """Get configs for SITS superregression models"""
    path = os.path.abspath("./hydra/model/")
    return [os.path.join(path, p) for p in os.listdir(path) if p.__contains__("sits_rdb_")]


@pytest.mark.parametrize("path_config", get_configs())
def test_model(path_config: str) -> None:
    """Test super-regression model"""
    config, superregression = get_model(path_config)
    superregression.eval()
    margin = superregression.get_prediction_margin()
    assert margin
    batch, batch_padded = dummy_batch(dummy_padding=True)
    if config.model.model.spatio_spectral_encoder.spectral_encoding.model.in_channels == 10:
        batch.angles = None
        batch_padded.angles = None
    result_padded = superregression(batch_padded).height[:, :, margin:-margin, margin:-margin]
    result = superregression(batch).height[:, :, margin:-margin, margin:-margin]
    assert(~torch.any(torch.isnan(result)))
    assert (torch.allclose(result, result_padded, atol=1e-7))
