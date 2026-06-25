# -*- coding: utf-8 -*-
"""Compute metrics for std from patches."""

import json
import logging
import os
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm


MAX_SAMPLES_QUANTILE = 16_000_000


def clip_and_divide_s2(buffer: torch.Tensor,
                       nodata: list = [-10000]) -> torch.Tensor:
    """Clip S2 to 0:15000 and divide by 10000 """
    buffer[~torch.isin(buffer, torch.Tensor(nodata))] = buffer[~torch.isin(buffer, torch.Tensor(nodata))].clip(0, 15000)
    return buffer


def compute_stats(
        buffer: torch.Tensor,
        nodata: int | float | list | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std on given buffer.
    """
    buffer = buffer.to(torch.float32)
    if nodata is None:
        mean = buffer.mean(dim=(0, 2, 3))
        std = buffer.std(dim=(0, 2, 3))
    else:
        b = torch.flatten(buffer.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
        valid_mask = (torch.isin(buffer, torch.Tensor(nodata))).sum(1) == 0
        mean = b[valid_mask].mean(dim=0)
        std = b[valid_mask].std(dim=0)
    return mean, std



def compute_stats_quantiles(
        buffer: torch.Tensor,
        q: torch.Tensor = torch.Tensor([0.05, 0.5, 0.95]),
        nodata: int | float | None | list = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std on given buffer using quantiles.
    """
    buffer = buffer.to(torch.float32)
    if buffer.dim() == 4:
        b = torch.flatten(buffer.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
    else:
        b = torch.flatten(buffer.permute(0, 1, 3, 4, 2), start_dim=0, end_dim=3)
    if nodata is not None:
        valid_mask = (torch.isin(b, torch.Tensor(nodata))).sum(1) == 0
        b = b[valid_mask]
    if b.shape[0] > MAX_SAMPLES_QUANTILE:
        b = b[torch.randperm(MAX_SAMPLES_QUANTILE)]

    min, median, max = torch.quantile(b, q, dim=0)

    # import matplotlib.pyplot as plt
    # plt.hist(b[:, 0], density=True, bins=30)
    # plt.show()
    regul = torch.nn.Threshold(1e-10, 10)
    std = regul((min + max) / 2.0)
    return median, std


def write_metrics(
        filename: str,
        mean_src: torch.Tensor,
        std_src: torch.Tensor,
        mean_target: torch.Tensor,
        std_target: torch.Tensor,
) -> None:
    """Export metrics to filename.

    Parameters
    ----------
    filename: str
    mean_src: torch.Tensor
    std_src: torch.Tensor
    mean_target: torch.Tensor
    std_target: torch.Tensor
    """
    out = {
        "source": {"mean": mean_src.tolist(), "std": std_src.tolist()},
        "target": {"mean": mean_target.tolist(), "std": std_target.tolist()},
    }
    with open(filename, "w", encoding="utf-8") as h:
        json.dump(out, h)


@hydra.main(config_path="../../hydra/", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    """Processor."""
    dataloader = hydra.utils.instantiate(
        config.datamodule.data_module
    ).train_dataloader()
    std_metrics = hydra.utils.instantiate(config.standardization_metrics.config)
    nb_batches = std_metrics["nb_batches"]
    iterator = iter(dataloader)
    sources = []
    targets = []

    nb_iterations = min(int(nb_batches/100), len(dataloader))
    for _ in tqdm(range(nb_iterations), total=nb_iterations, desc="Collecting samples"):
        batch = next(iterator)
        clipped_input_tensor = batch.input_tensor[:, :, :, 25:33, 25:33][~batch.pad_mask]
        sources.append(clipped_input_tensor)
        targets.append(batch.target_tensor[:, :, range(0, 80, 10), :][:, :, :, range(0, 80, 10)])

    mean_src, std_src = compute_stats_quantiles(clip_and_divide_s2(torch.cat(sources)), nodata=[-10000])
    mean_target, std_target = compute_stats_quantiles(torch.cat(targets), nodata=[0, -1])
    print(mean_src, std_src)
    # print(mean_target, std_target)
    f = os.path.join(config.work_dir, std_metrics["filename"])
    logging.info("Export metrics to: %s", f)
    write_metrics(
        f,
        mean_src=mean_src,
        std_src=std_src,
        mean_target=mean_target,
        std_target=std_target,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
