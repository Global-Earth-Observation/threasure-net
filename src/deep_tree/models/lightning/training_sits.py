"""Training module"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

# from deep_tree.models.components.binned_mae import BinnedErrorStats
from torchsisr import patches

from deep_tree.models.components.decay_scheduler import DecayCosineAnnealingWarmRestarts
from deep_tree.models.components.utils import change_target_resolution
from deep_tree.models.datatypes import DeepTreeTrainingModuleConfig, PredVisu
from deep_tree.datamodules.datatypes import Prediction, BatchData
from deep_tree.models.torch.deeptree_sits import DeepTreeSITSModel


class DeepTreeSITSTrainingModule(pl.LightningModule):
    # pylint: disable=too-many-ancestors
    """
    Lightning module wrapper for the training of DeepTreeSITSModel.
    """

    def __init__(self, config: DeepTreeTrainingModuleConfig):
        """
        Init.

        Parameters
        ----------

        config: DeepTreeDataModuleConfig
        """

        super().__init__()

        # # In order to handle optional GAN loss
        # self.automatic_optimization = False
        self.save_hyperparameters(logger=False)

        self.config = config

        # Store models as class members so that their parameters
        # are visible to pytorch lightning
        self.model: DeepTreeSITSModel = self.config.model

        if self.model.sr_10to5 is not None:
            for param in self.model.sr_10to5.parameters():
                param.requires_grad = False

        self.register_buffer(
            "input_mean", torch.tensor(self.config.standardization.mean)  # type:ignore
        )
        self.register_buffer(
            "input_std", torch.tensor(self.config.standardization.std)  # type:ignore
        )
        if self.config.target_standardization is not None:  # type:ignore
            self.register_buffer(
                "target_mean",
                torch.tensor(self.config.target_standardization.mean),  # type:ignore
            )
            self.register_buffer(
                "target_std",
                torch.tensor(self.config.target_standardization.std),  # type:ignore
            )

        self.real_losses = torch.nn.ModuleList(self.config.real_losses)  # type:ignore
        self.validation_metrics = torch.nn.ModuleList(
            self.config.validation_metrics  # type:ignore
        )
        self.test_metrics = torch.nn.ModuleList(self.config.test_metrics)  # type:ignore

        self.factor = 1
        if self.model.upsample_module is not None:
            self.factor = self.model.upsample_module.upsampling_factor

        self.margin = self.get_margin()

        self.ref_date = None

        logging.info(f"Margin {self.margin}")

        self._batch_counter = 0  # keeps track of saved batches

    def write_data(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
            name_lidar: str,
            classif: torch.Tensor | None=None
    ):
        """
        Save prediction, target, mask, and lidar names per batch.
        Each call writes one compressed .npz file like batch_00001.npz
        """
        out_path = os.path.join(self.trainer.checkpoint_callback.dirpath, "predictions")
        os.makedirs(out_path, exist_ok=True)

        # Detach + move to CPU for serialization
        prediction = prediction.detach().cpu().float().numpy()
        target = target.detach().cpu().float().numpy()
        mask = mask.detach().cpu().bool().numpy()
        if classif is not None:
            classif = classif.detach().cpu().bool().numpy()

        # Convert name_lidar to numpy array of strings
        if isinstance(name_lidar, (list, tuple)):
            name_lidar = np.array(name_lidar, dtype=object)
        elif torch.is_tensor(name_lidar):
            name_lidar = np.array([str(n) for n in name_lidar])
        else:
            name_lidar = np.array([name_lidar], dtype=object)

        # Build filename with incrementing batch counter
        fname = os.path.join(out_path, f"batch_{self._batch_counter:05d}.npz")
        self._batch_counter += 1

        np.savez_compressed(
            fname,
            prediction=prediction,
            target=target,
            mask=mask,
            name_lidar=name_lidar,
            classification=classif
        )

    def get_margin(self) -> int:
        """
        Get prediction margin in TARGET resolution.
        We add it to S2 data in dataloader, while keeping the target patches of smaller size
        """
        margin = (
            self.model.get_prediction_margin()
        )  # get margin from model (final resolution)
        # We want that input patch size is dividable by 16 (so input margin at each side dividable by 8)
        margin = np.ceil(margin / (8 * self.factor)) * (8 * self.factor)
        assert margin.is_integer()
        return int(margin)

    def compute_losses(
            self,
            target: torch.Tensor,
            prediction: Prediction,
            losses: torch.nn.ModuleList,
            context: str = "traning",
            mask: torch.Tensor | None = None,
            mask_no_veg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss from a sequence of loss functions and argregate them.

        Parameters
        ----------
        target: torch.Tensor
        prediction: torch.Tensor
        losses: torch.nn.ModuleList
        context: str = "training"
        mask: torch.Tensor | None = None
        Return
        ------
        torch.Tensor
        """
        total_loss = torch.zeros((1,), device=self.device)
        mask = mask.to(int)
        mask[mask_no_veg] = -1
        if self.margin is not None and self.margin > 0:
            height = prediction.height[:, :, self.margin:-self.margin, self.margin:-self.margin]
        for element in losses:
            loss = element(target, height, mask)
            self.log(
                context + "/" + element.name,
                loss,
                batch_size=prediction.shape[0],
            )
            if "metric" not in context:
                total_loss += loss

        if "metric" not in context:
            self.log(
                context + "/total_loss",
                total_loss,
                batch_size=prediction.shape[0],
            )
        return total_loss

    def step(self, batch: BatchData) -> tuple[BatchData, Prediction]:
        """One common step for all stages"""
        # Call full res forward pass
        prediction: Prediction = self.model(batch)
        # TODO change resolution in dataloder
        # The initial target resolution is at 1m, change if needed
        if (
                self.config.target_resolution
                != self.trainer.datamodule.config.single_tile_dataset_config.target_resolution
        ):
            # TODO: bicubic produces negative values
            # TODO: parametrize initial resolution
            batch.target_tensor, batch.target_tensor_mask = change_target_resolution(
                batch.target_tensor,
                batch.target_tensor_mask,
                self.config.target_resolution,
                remove_small=False,
            )

        batch.mask_no_veg = batch.target_tensor == 0

        # if batch.input_tensor_mask.dim() == 5:
        #     batch.full_mask = batch.target_tensor_mask
        # else:
        # input_tensor_mask = (
        #         F.interpolate(
        #             batch.input_tensor_mask.to(torch.float16),
        #             scale_factor=10 / self.config.target_resolution,
        #             mode="bilinear",
        #         )[:, :, self.margin: -self.margin, self.margin: -self.margin]
        #         > 0
        # )
        #
        # batch.input_tensor_mask = input_tensor_mask + F.max_pool2d(
        #     input_tensor_mask.to(torch.float16), kernel_size=5, stride=1, padding=2
        # ).to(torch.bool)
        #
        # # We create a common mask
        # batch.full_mask = (batch.input_tensor_mask + batch.target_tensor_mask) > 0

        batch.full_mask = batch.target_tensor_mask

        assert prediction.shape[:2] == batch.target_tensor.shape[:2]
        assert prediction.shape[2] == batch.target_tensor.shape[2] + self.margin * 2
        assert prediction.shape[3] == batch.target_tensor.shape[3] + self.margin * 2

        batch.target_tensor = batch.target_tensor.masked_fill(batch.full_mask != 0, 0)

        return batch, prediction

    # pylint: disable=arguments-differ
    def training_step(self, batch: BatchData, _batch_idx: int):
        """Overloaded train iteration.

        Parameters
        -----------
        batch: BatchData
        _batch_idx: int

        Return
        ------
        BatchData
        """
        batch, prediction = self.step(batch)

        target_std = patches.standardize(
            data=batch.target_tensor, mean=self.target_mean, std=self.target_std
        )

        # Evaluate pixel losses
        assert self.real_losses is not None
        total_loss = self.compute_losses(
            target=target_std,
            prediction=prediction,
            losses=self.real_losses,
            context="training",
            mask=batch.full_mask,
            mask_no_veg=batch.mask_no_veg
        )

        return {"loss": total_loss}  # type: ignore

    def unstandardize_pred(self, predicted: torch.Tensor) -> torch.Tensor:
        """Apply unstandardization.

        Parameters
        ----------
        predicted: torch.Tensor

        Return
        ------
        torch.Tensor
        """
        return patches.unstandardize(
            predicted, self.target_mean, self.target_std
        ).clamp(min=0)


    def validation_test_step(
            self,
            batch: BatchData,
            metrics: torch.nn.ModuleList,
            losses: torch.nn.ModuleList | None,
            context: str = "validation",
    ) -> dict[torch.Tensor | None]:
        """Perform validation or test step.

        Parameters
        ----------
        batch: BatchData
        metrics: torch.nn.ModuleList
        losses: torch.nn.ModuleList | None
        context: str = "validation"

        Return
        ------
        torch.Tensor | None
        """
        batch, prediction = self.step(batch)
        prediction = prediction.detach()
        target_std = patches.standardize(
            data=batch.target_tensor, mean=self.target_mean, std=self.target_std
        )
        prediction_unstd = Prediction(self.unstandardize_pred(prediction.height),
                                      prediction.logits,
                                      )
        total_loss: torch.Tensor | None = None
        # Eval losses
        if losses is not None:
            # Evaluate pixel losses
            total_loss = self.compute_losses(
                target=target_std,
                prediction=prediction,
                losses=losses,
                context=context + "_losses",
                mask=batch.full_mask,
                mask_no_veg=batch.mask_no_veg,
            )
        _ = self.compute_losses(
            target=batch.target_tensor,
            prediction=prediction_unstd,
            losses=metrics,
            context=context + "_metrics",
            mask=batch.full_mask,
            mask_no_veg=batch.mask_no_veg,
        )
        if context == "test":
            # self.compute_bins(prediction_unstd, batch.target_tensor, batch.full_mask)
            self.write_data(prediction_unstd.height, batch.target_tensor, batch.full_mask, batch.name_lidar)
        return {"loss": total_loss}

    def validation_step(self, batch: BatchData, _batch_idx: int) -> None:
        """Perform validation step.

        Parameters
        ----------
        batch: BatchData
        _batch_idx: int
        """
        self.validation_test_step(
            batch,
            metrics=self.validation_metrics,
            losses=self.real_losses,
            context="validation",
        )

    def test_step(self, batch: BatchData, _batch_idx: int) -> None:
        """Perform tes step.

        Parameters
        ----------
        batch: BatchData
        _batch_idx: int
        """
        self.validation_test_step(
            batch, metrics=self.test_metrics, losses=self.real_losses, context="test"
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """A single optimizer with a LR scheduler"""
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.optimization.learning_rate,
        )

        # training_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=self.lr_sched_factor,
        #     patience=self.lr_sched_patience,
        #     min_lr=self.lr_sched_min_lr,
        # )
        #
        steps_per_epoch = (
                len(self.trainer.datamodule.train_dataloader())
                // self.trainer.accumulate_grad_batches
        )
        assert steps_per_epoch > 0

        if self.config.optimization.type == "cos_warm_restart":
            training_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.optimization.t_0 * steps_per_epoch,
                T_mult=self.config.optimization.t_mult,
                eta_min=1e-6,
                last_epoch=-1,
            )
        elif self.config.optimization.type == "cos_warm_restart_decay":
            training_scheduler = DecayCosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.optimization.t_0 * steps_per_epoch,
                T_mult=self.config.optimization.t_mult,
                eta_min=1e-6,
                last_epoch=-1,
                decay_factor=self.config.optimization.decay_factor,
            )
        elif self.config.optimization.type == "step":

            training_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=100, gamma=0.9
            )
        else:
            raise NotImplementedError

        scheduler = {
            "scheduler": training_scheduler,
            "interval": "step",
            "monitor": "validation_losses/total_loss",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def predict_for_visu(self, batch: BatchData) -> PredVisu:
        """Get predicted values for visualization"""
        with torch.no_grad():
            self.model.eval()
            pred = self.model(batch)[
                   :, :, self.margin: -self.margin, self.margin: -self.margin
                   ]
            pred = self.unstandardize_pred(pred.height)

        target = (
            batch.target_tensor
        )  # [:, :, self.margin:-self.margin, self.margin:-self.margin]
        mask = (
            batch.full_mask
        )  # [:, :, self.margin:-self.margin, self.margin:-self.margin]
        # Target does not have the same size as S2 and S2 SR images,
        # We need to re-adjust the margin
        # We add +16 because we clip some margins after SR 10->5m
        input_s2 = batch.input_tensor
        if self.model.sr_10to5 is not None:
            s2_margin = int(self.margin / 2) + int(self.model.sr_10to5.margin / 2)
        else:
            s2_margin = int(self.margin / self.factor)
        input_s2 = input_s2[:, :, :, s2_margin:-s2_margin, s2_margin:-s2_margin]
        input_s2 = [input_s2[b][~batch.pad_mask[b]] for b in range(input_s2.shape[0])]
        dates_lidar = [self.get_date(d) for d in batch.doy_lidar]
        dates_s2 = [
            [self.get_date(days) for days in sublist if days != 0]
            for sublist in batch.doy_s2
        ]

        return PredVisu(
            input_s2=input_s2,
            input_sr=None,
            pred=pred,
            target=target,
            mask=mask,
            name_lidar=batch.name_lidar,
            name_s2=batch.name_s2,
            dates_lidar=dates_lidar,
            dates_s2=dates_s2,
        )

    def get_date(self, days: int) -> str:
        """Get calendar date from DOY"""
        if self.ref_date != "year":
            return (self.ref_date + timedelta(days=int(days))).strftime("%Y-%m-%d")
        dummy_year = datetime(2000, 1, 1)
        return (timedelta(days=int(days)) + dummy_year).strftime("%m-%d")

    def set_input_stats(self):
        """Set stats to torch model"""
        self.model.mean = self.input_mean
        self.model.std = self.input_std

    def set_margin_into_dataloader(self):
        """
        We compute margin added to S2 data, so it is removed from prediction
        """
        if self.model.sr_10to5 is not None:
            margin_to_set = int(self.margin / 2) + int(self.model.sr_10to5.margin / 2)
        else:
            margin_to_set = int(self.margin / self.factor)
        # Check that our margin does not surpass 25 pixels (250m) from each size
        # That is the size difference between lidar patch (1000x1000m) and S2 patch (1500x1500m)
        # assert margin_to_set <= 25

        assert hasattr(self.trainer, "datamodule")
        for ds in self.trainer.datamodule.training_dataset.dataset.datasets:
            ds.sample_getter.margin_s2 = margin_to_set
        for ds in self.trainer.datamodule.validation_dataset.dataset.datasets:
            ds.sample_getter.margin_s2 = margin_to_set
        for ds in self.trainer.datamodule.testing_dataset.dataset.datasets:
            ds.sample_getter.margin_s2 = margin_to_set

    def setup(self, stage: str):
        if stage in ("fit", "test"):
            logging.info("On fit start")

            self.set_input_stats()
            self.set_margin_into_dataloader()
            if self.trainer.datamodule.config.ref_date != "year":
                self.ref_date = datetime.strptime(
                    self.trainer.datamodule.config.ref_date, "%Y-%m-%d"
                )
            else:
                self.ref_date = "year"
