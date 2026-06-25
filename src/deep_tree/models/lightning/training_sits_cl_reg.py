"""Training module"""

import torch
import torchmetrics

from deep_tree.datamodules.datatypes import Prediction, BatchData
from deep_tree.models.components.loss import PixelLossWrapper
from deep_tree.models.lightning.training_sits import DeepTreeSITSTrainingModule
from deep_tree.models.datatypes import DeepTreeTrainingModuleConfig, PredVisu
from torchsisr import patches


class DeepTreeRegSITSTrainingModule(DeepTreeSITSTrainingModule):
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

        super().__init__(config)

        def make_metrics(prefix):
            # use the same prefix as your context
            metrics = {
                f"{prefix}/precision": torchmetrics.classification.BinaryPrecision(),
                f"{prefix}/recall": torchmetrics.classification.BinaryRecall(),
                f"{prefix}/f1": torchmetrics.classification.BinaryF1Score(),
                f"{prefix}/accuracy": torchmetrics.classification.BinaryAccuracy(),
                f"{prefix}/mIoU": torchmetrics.classification.BinaryJaccardIndex(),
            }
            # register them so Lightning can find them
            for name, metric in metrics.items():
                self.add_module(name.replace("/", "_"), metric)
            return metrics

        self.validation_torchmetrics = make_metrics("validation_metrics")
        self.test_torchmetrics = make_metrics("test_metrics")

    def compute_torchmetrics(self, prediction: Prediction, mask: torch.Tensor):
        for name, metric in self.validation_torchmetrics.items():
            metric.update(
                prediction.classification[:, :, self.margin: -self.margin, self.margin: -self.margin],
                mask == 0
            )
            self.log(name, metric)

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
        context: str = "traning"
        mask: torch.Tensor | None = None
        Return
        ------
        torch.Tensor
        """
        total_loss = torch.zeros((1,), device=self.device)
        # mask = torch.Tensor((mask_no_veg==1) & (mask.to(int) == 0))
        height, logits = prediction.height, prediction.logits
        if self.margin is not None and self.margin>0:
            height = height[:, :, self.margin:-self.margin, self.margin:-self.margin]
            logits = logits[:, :, self.margin:-self.margin, self.margin:-self.margin]


        for element in losses:
            if type(element) is PixelLossWrapper:
                loss = element(target, height, mask.int())
                self.log(
                    context + "/" + element.name,
                    loss,
                    batch_size=prediction.shape[0],
                )

            else:
                loss = element((mask==0).to(torch.int), logits)
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
                                      prediction.logits
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
            )
        _ = self.compute_losses(
            target=batch.target_tensor,
            prediction=prediction_unstd,
            losses=metrics,
            context=context + "_metrics",
            mask=batch.full_mask,
        )

        self.compute_torchmetrics(prediction, batch.full_mask)

        if context == "test":
            self.write_data(
                prediction_unstd.height,
                batch.target_tensor,
                batch.full_mask,
                batch.name_lidar,
                prediction_unstd.classification
            )

        return {"loss": total_loss}

    def predict_for_visu(self, batch: BatchData) -> PredVisu:
        """Get predicted values for visualization"""
        with torch.no_grad():
            self.model.eval()
            pred = self.model(batch)[
                :, :, self.margin: -self.margin, self.margin: -self.margin
            ]
            pred = Prediction(self.unstandardize_pred(pred.height),
                                              pred.logits,
                                              )
        target = (
            batch.target_tensor
        )

        mask = (
            batch.full_mask
        )
        target[mask.expand_as(target) != 0] = 0
        # Target does not have the same size as S2 and S2 SR images,
        # We need to re-adjust the margin
        # We add +16 because we clip some margins after SR 10->5m
        input_s2 = batch.input_tensor

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
            pred=pred.height,
            target=target,
            classif=pred.classification,
            mask=mask,
            name_lidar=batch.name_lidar,
            name_s2=batch.name_s2,
            dates_lidar=dates_lidar,
            dates_s2=dates_s2,
        )
