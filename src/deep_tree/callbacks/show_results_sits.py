#!/usr/bin/env python3
# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
""" Lightning image callbacks """
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.axes import Axes
from pytorch_lightning.callbacks import Callback
from scipy.interpolate import interpn

from deep_tree.datamodules.datatypes import BatchData
from deep_tree.models.datatypes import PredVisu
from deep_tree.models.lightning.training_sits import DeepTreeSITSTrainingModule

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class SampleInfo:
    """Information about the sample to be displayed"""

    batch_idx: int
    batch_size: int
    current_epoch: int


class DeepTreeSITSCallback(Callback):
    """
    Callback to inspect the reconstruction image
    """

    def __init__(self, save_dir: str, n_samples: int = 5, n_sits_img: int = 8, show_no_veg: bool = True):
        self.save_dir = save_dir
        self.n_samples = n_samples
        self.n_sits_img = n_sits_img
        self.show_no_veg = show_no_veg

        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True, exist_ok=True)

    def save_image_grid(
        self,
        visu: PredVisu,
        sample: SampleInfo,
    ) -> None:
        """Generate the matplotlib figure and save it to a file"""

        to_show = [
            (
                self.prepare_patches_row(visu[samp_idx]),
                self.prepare_patches_to_scatter(visu[samp_idx]),
                (visu[samp_idx].name_lidar, visu[samp_idx].name_s2),
                (visu[samp_idx].dates_lidar, visu[samp_idx].dates_s2),
            )
            for samp_idx in range(len(visu.target))
        ]
        to_show_img = [i[0] for i in to_show]
        to_show_scatter = [i[1] for i in to_show]
        to_show_names = [i[2] for i in to_show]
        to_show_date = [i[3][1] for i in to_show]

        # export name
        image_basename = (
            f"DeepTree_val_ep_{sample.current_epoch:03}_batch_{sample.batch_idx}"
        )
        image_name = Path(f"{self.save_dir}/{image_basename}.png")
        if not image_name.is_file() or sample.current_epoch == 0:
            self.save_image_for_sample(
                to_show_img, to_show_scatter, to_show_date, to_show_names, image_name
            )

    @staticmethod
    def scatterplot_pred(axes: Axes, visu: PredVisu, row: int, col: int = -1) -> Axes:
        """Scatterplot for GT and prediction"""
        axes[row, col].scatter(visu.pred[~visu.mask], visu.target[~visu.mask])
        axes[row, col].plot(
            visu.target[~visu.mask].min(), visu.target[~visu.mask].max()
        )
        axes[row, col].set_xlabel("pred, 10^-1 m")
        axes[row, col].set_ylabel("GT, 10^-1 m")
        return axes

    @staticmethod
    def fancy_scatter(
        preds: np.array, gt: np.array
    ) -> tuple[np.array, np.array, np.array]:
        """Scatterplot with point density"""
        # We compute point density to show it on the plot
        # We use simplified bin method to overcome memory problem
        data, x_e, y_e = np.histogram2d(
            preds, gt, bins=np.arange(gt.min(), gt.max(), 0.05), density=True
        )
        z = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([preds, gt]).T,
            method="splinef2d",
            bounds_error=False,
        )

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        preds, gt, z = preds[idx], gt[idx], z[idx]

        # plt.close()
        # plt.scatter(preds[preds>0], gt[preds>0], c=z[preds>0], s=0.1)
        return preds, gt, z

    @staticmethod
    def compute_position(
        ncols: int, nrows: int, i: int
    ) -> tuple[float, float, float, float]:
        """Compute position for SITS plotting"""
        h_gap = 0.02  # horizontal gap between images (in relative coords)
        v_gap = 0.1  # vertical gap between rows

        r = i // ncols
        c = i % ncols

        # Compute position and size for each inset axes
        width = 1 / ncols
        height = 1 / nrows
        left = c * (width + h_gap)
        bottom = 1 - (r + 1) * height - r * v_gap

        return left, bottom, width, height

    def save_image_for_sample(
        self,
        images: list[
            tuple[
                np.array,
                np.array,
                np.array,
            ]
        ],
        scatter: list[
            tuple[
                np.array,
                np.array,
            ]
        ],
        dates_s2: list[list[str]],
        names: list[tuple[str, str]],
        image_name: Path,
        **kwargs,
    ) -> None:
        """Save the image grid for a sample"""
        plt.close()
        fig = plt.figure(
            # nrows=len(images),
            # ncols=5,
            constrained_layout=True,
            figsize=(
                (len(images) + 1 + max([len(row[0]) for row in images]) / 2) * 2,
                20,
            ),
        )
        fig.suptitle("Tree height prediction", fontsize=20)
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
        )

        labels = ["S2 10m", "GT", "Pred"]

        # h_gap = 0.02  # horizontal gap between images (in relative coords)
        # v_gap = 0.1  # vertical gap between rows

        subfigs = fig.subfigures(nrows=len(images), ncols=1)
        for row, subfig in enumerate(subfigs):
            # for row in range(len(images)):
            # subfig.suptitle(f'{names[row][0]} \n {names[row][1]}')
            subfig.suptitle(names[row][0])

            # create 1 x cols subplots per subfig
            axs = subfig.subplots(nrows=1, ncols=len(images[row]) + 1)

            for col, ax in enumerate(axs):
                ax.axis("off")

                # for col in range(len(images[row])):
                if col == 0:
                    ncols = math.ceil(len(images[row][col]) / 2)
                    nrows = 2

                    for i, (img, date) in enumerate(
                        zip(images[row][col], dates_s2[row])
                    ):

                        ax_ts_inset = ax.inset_axes(
                            [self.compute_position(ncols, nrows, i)]
                        )

                        ax_ts_inset.imshow(img)
                        ax_ts_inset.set_title(date, fontsize=6)
                        ax_ts_inset.axis("off")

                elif 0 < col < 3:
                    ax.imshow(images[row][col], interpolation="bicubic")
                    ax.set_title(labels[col])
                # add scatter plot of predicted values
                else:
                    if len(scatter[row][0]) > 0:
                        preds, gt, z = self.fancy_scatter(
                            scatter[row][0], scatter[row][1]
                        )
                        ax.scatter(preds, gt, c=z, s=0.5)
                        ax.plot(scatter[row][1], scatter[row][1], c="red")
                        ax.set_xlabel("pred, 10^-1 m")
                        ax.set_ylabel("GT, 10^-1 m")
                        ax.axis("on")
        fig.savefig(image_name, dpi=100)

    @staticmethod
    def norm_image(
        img,
        nodata: int | None = None,
        norm_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        series: bool = False,
    ):
        """Normalize image for rendering"""
        img = img.to(torch.float32)
        if nodata is not None:
            img[img == nodata] = torch.nan
            # mask = (img == nodata).sum(0) > 0
            # mask = repeat(mask, "h w -> c h w", c=img.shape[0])
            # img[mask] = torch.nan
        if norm_values is None:
            if series:
                dmin, dmax = torch.nanquantile(
                    img.contiguous().permute(1, 0, 2, 3).reshape(img.shape[1], -1),
                    q=torch.Tensor([0.05, 0.95]).to(img.device),
                    dim=1,
                )
            else:
                dmin, dmax = torch.nanquantile(
                    img.contiguous().view(img.shape[0], -1),
                    q=torch.Tensor([0.05, 0.95]).to(img.device),
                    dim=1,
                )

        else:
            dmin, dmax = norm_values

        if series:
            return torch.clip(
                (
                    (img - dmin[None, :, None, None])
                    / (dmax - dmin)[None, :, None, None]
                ).nan_to_num(),
                0,
                1,
            ), (dmin, dmax)

        return torch.clip(
            ((img - dmin[:, None, None]) / (dmax - dmin)[:, None, None]).nan_to_num(),
            0,
            1,
        ), (dmin, dmax)

    def prepare_patches_row(
        self,
        visu: PredVisu,
    ) -> tuple[np.array, np.array, np.array]:
        """Generate the patches for the visualization grid
        Select the appropriate channels and crop with the patch margin.
        """
        input_s2, pred, target, mask = visu.input_s2, visu.pred, visu.target, visu.mask
        # bands = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]
        input_s2, _ = self.norm_image(
            input_s2[:, [6, 2, 1], ...], nodata=-10000, series=True
        )
        target, norm_values = self.norm_image(target, nodata=-1)
        target = target.squeeze(0)
        pred = self.norm_image(pred, nodata=-1, norm_values=norm_values)[0].squeeze(0)
        mask = mask.squeeze(0)
        if not self.show_no_veg:
            pred[mask] = 0
            target[mask] = 0
        else:
            mask_no_veg = target == 0
            mask_uncertain_pix = (mask.to(int) - mask_no_veg.to(int)).to(bool)
            pred[mask_uncertain_pix] = 0
            target[mask_uncertain_pix] = 0

        return (
            input_s2.cpu().permute(0, 2, 3, 1).numpy(),
            target.cpu().numpy(),
            pred.cpu().numpy(),
        )

    @staticmethod
    def prepare_patches_to_scatter(
        visu: PredVisu,
    ) -> tuple[np.array, np.array]:
        """Generate the patches for the scatterplot of predicted values"""
        return (
            visu.pred[~visu.mask].cpu().numpy(),
            visu.target[~visu.mask].cpu().numpy(),
        )

    def sample_equally_spaced_frames(
        self, sits: torch.Tensor, dates: list
    ) -> tuple[torch.Tensor, list]:
        """
        Sample evenly spaced from SITS
        """
        t = sits.shape[0]
        if t == self.n_sits_img:
            return sits, dates
        idx = torch.linspace(0, t - 1, steps=self.n_sits_img).round().long()
        return sits[idx], [dates[i] for i in idx.cpu().numpy()]

    def on_validation_batch_end(  # pylint: disable=too-many-arguments
        self,
        trainer: pl.trainer.Trainer,
        pl_module: DeepTreeSITSTrainingModule,
        outputs: Any,
        batch: BatchData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Method called from the validation loop"""
        if batch_idx in [0, 5, 10]:
            sample = SampleInfo(
                batch_idx,
                batch.target_tensor.shape[0],
                trainer.current_epoch,
            )
            idx = list(
                np.arange(
                    0,
                    sample.batch_size,
                    sample.batch_size // min(sample.batch_size, self.n_samples),
                )
            )
            visu = pl_module.predict_for_visu(batch[idx])

            if self.n_sits_img is not None:
                visu.input_s2, visu.dates_s2 = map(
                    list,
                    zip(
                        *[
                            self.sample_equally_spaced_frames(sits, dates)
                            for sits, dates in zip(visu.input_s2, visu.dates_s2)
                        ]
                    ),
                )

            self.save_image_grid(visu, sample)
