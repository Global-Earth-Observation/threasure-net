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
from matplotlib.colors import ListedColormap

from deep_tree.callbacks.show_results_sits import DeepTreeSITSCallback
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


class DeepTreeSITSCallbackClassif(DeepTreeSITSCallback):
    """
    Callback to inspect the reconstruction image
    """

    def __init__(self, save_dir: str, n_samples: int = 5, n_sits_img: int = 8, show_no_veg: bool = True):
        super().__init__(save_dir, n_samples, n_sits_img, show_no_veg)
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
                visu.classif[samp_idx]
            )
            for samp_idx in range(len(visu.target))
        ]
        to_show_img = [i[0] for i in to_show]
        to_show_scatter = [i[1] for i in to_show]
        to_show_names = [i[2] for i in to_show]
        to_show_date = [i[3][1] for i in to_show]
        to_show_classif = [i[4].cpu().numpy() for i in to_show]

        # export name
        image_basename = (
            f"DeepTree_val_ep_{sample.current_epoch:03}_batch_{sample.batch_idx}"
        )
        image_name = Path(f"{self.save_dir}/{image_basename}.png")
        if not image_name.is_file() or sample.current_epoch == 0:
            self.save_image_for_sample(
                to_show_img, to_show_scatter, to_show_date, to_show_names, image_name, to_show_classif
            )


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
        classif:np.array,
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

        labels = ["S2 10m", "GT", "Pred", "Classif"]

        # h_gap = 0.02  # horizontal gap between images (in relative coords)
        # v_gap = 0.1  # vertical gap between rows

        subfigs = fig.subfigures(nrows=len(images), ncols=1)
        for row, subfig in enumerate(subfigs):
            # for row in range(len(images)):
            # subfig.suptitle(f'{names[row][0]} \n {names[row][1]}')
            subfig.suptitle(names[row][0])

            # create 1 x cols subplots per subfig
            axs = subfig.subplots(nrows=1, ncols=len(images[row]) + 1 + 1)

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
                            self.compute_position(ncols, nrows, i)
                        )

                        ax_ts_inset.imshow(img)
                        ax_ts_inset.set_title(date, fontsize=6)
                        ax_ts_inset.axis("off")

                elif 0 < col < 3:
                    ax.imshow(images[row][col], interpolation="bicubic")
                    ax.set_title(labels[col])
                # add scatter plot of predicted values
                elif 0 < col == 3:
                    cmap_cl = ListedColormap(["yellow", "green"])
                    ax.imshow(classif[row][0], cmap=cmap_cl, vmin=0, vmax=1)
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
