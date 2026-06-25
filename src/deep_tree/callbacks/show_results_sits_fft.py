#!/usr/bin/env python3
# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
""" Lightning image callbacks with FFT"""

import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from deep_tree.callbacks.fft import masked_fft_profile
from deep_tree.callbacks.show_results_sits import DeepTreeSITSCallback, SampleInfo
from deep_tree.models.datatypes import PredVisu


class DeepTreeSITSCallbackFFT(DeepTreeSITSCallback):
    """
    Callback to inspect the reconstruction image
    """

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

        hr_img = visu.target.cpu().numpy()
        sr_img = visu.pred.cpu().detach().numpy()
        mask = visu.mask.cpu().numpy()
        sr_profile, hr_profile, mse, cos_sim = map(
            list,
            zip(
                *[
                    (
                        masked_fft_profile(
                            sr_img[i, 0] / np.max(hr_img[i, 0]),
                            hr_img[i, 0] / np.max(hr_img[i, 0]),
                            mask=mask[i, 0],
                        )
                    )
                    for i in range(len(visu.target))
                ]
            ),
        )

        fft_profile = sr_profile, hr_profile, mse, cos_sim

        # export name
        image_basename = (
            f"DeepTree_val_ep_{sample.current_epoch:03}_batch_{sample.batch_idx}"
        )
        image_name = Path(f"{self.save_dir}/{image_basename}.png")
        if not image_name.is_file() or sample.current_epoch == 0:
            self.save_image_for_sample(
                to_show_img,
                to_show_scatter,
                to_show_date,
                to_show_names,
                image_name,
                fft_profile,
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
        fft_profile: tuple[np.array, np.array, float, float],
    ) -> None:
        """Save the image grid for a sample"""
        plt.close()
        n_cols = len(images[0]) + 2
        width_ratios = [1] * (n_cols - 2) + [3] + [3]
        fig = plt.figure(
            # nrows=len(images),
            # ncols=5,
            constrained_layout=True,
            figsize=(
                (len(images) + 1 + max([len(row[0]) for row in images])) * 3,
                25,
            ),
        )
        fig.suptitle("Tree height prediction", fontsize=20)
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
        )

        labels = ["S2 10m", "GT", "Pred"]

        sr_profile, hr_profile, mse, cos_sim = fft_profile

        subfigs = fig.subfigures(nrows=len(images), ncols=1)
        for row, subfig in enumerate(subfigs):
            # for row in range(len(images)):
            # subfig.suptitle(f'{names[row][0]} \n {names[row][1]}')
            subfig.suptitle(names[row][0])

            # create 1 x cols subplots per subfig
            axs = subfig.subplots(
                nrows=1,
                ncols=len(images[row]) + 2,
                gridspec_kw={"width_ratios": width_ratios},
            )

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
                elif col == 3:
                    if len(scatter[row][0]) > 0:
                        preds, gt, z = self.fancy_scatter(
                            scatter[row][0], scatter[row][1]
                        )
                        ax.scatter(preds, gt, c=z, s=0.5)
                        ax.plot(scatter[row][1], scatter[row][1], c="red")
                        ax.set_xlabel("pred, 10^-1 m")
                        ax.set_ylabel("GT, 10^-1 m")
                        ax.axis("on")
                else:  # col == 4
                    ax.plot(sr_profile[row], label="SR")
                    ax.plot(hr_profile[row], label="HR")
                    ax.set_xlabel("Radius")
                    ax.set_ylabel("Normalized magnitude")
                    ax.set_title(
                        f"Radial FFT Profiles\nMSE={mse[row]:.4e}, CosSim={cos_sim[row]:.4f}"
                    )
                    ax.legend()
                    ax.axis("on")
                    ax.grid(True)

        fig.savefig(image_name, dpi=100)
