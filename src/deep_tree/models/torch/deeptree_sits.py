"""Torch Deep Tree SITS model"""

import torch

from esrgan.models import ESREncoder
from torch import nn
from torchsisr import patches


from deep_tree.datamodules.datatypes import Prediction, BatchData

from deep_tree.models.torch.RDB import RDBNet
from deep_tree.models.torch.spatio_spectral_encoder import SSEncoder
from deep_tree.models.torch.sr_10to5 import InferenceSR
from deep_tree.models.torch.temporal_encoder import TemporalEncoder
from deep_tree.models.torch.temporal_encoder_tq import TemporalEncoderTQ
from deep_tree.models.torch.mlp import MLP, MLPDouble
from deep_tree.models.torch.unet_baseline import Unet
from deep_tree.models.torch.upsample import UpsamleFeatures



class DeepTreeSITSModel(nn.Module):
    """
    Input encoding class:
    spectro-spatial encoding + positional encoding
    """

    def __init__(
            self,
            spatio_spectral_encoder: SSEncoder,
            temporal_encoder: TemporalEncoder | TemporalEncoderTQ,
            regression: MLP,
            upsample_module: UpsamleFeatures | None = None,
            sr_10to5: InferenceSR | None = None,
            classification: MLP | None = None,
    ) -> None:
        super().__init__()

        self.sr_10to5: InferenceSR = sr_10to5
        self.spatio_spectral_encoder: SSEncoder = spatio_spectral_encoder
        self.temporal_encoder: TemporalEncoder | TemporalEncoderTQ = temporal_encoder
        self.upsample_module: UpsamleFeatures = upsample_module
        self.regression: MLP = regression
        self.classification: MLP = classification

        self.mean = None
        self.std = None

        # self.regression.layers[-1].bias.data.fill_(120./300.)

    def encoder(self, batch: BatchData) -> torch.Tensor:
        sits = batch.input_tensor.to(torch.float32)

        if self.sr_10to5 is not None:
            sits = self.sr_10to5(sits)
        sits[sits == -10000] = torch.nan  # TODO do smth with this
        sits = patches.standardize(
            data=sits.clip(0, 15000),
            mean=self.mean.to(sits.device),
            std=self.std.to(sits.device),
        ).nan_to_num()

        if batch.angles is not None:
            angles = torch.round(
                batch.angles[:, :, :, None, None].expand(-1, -1, -1, *sits.shape[-2:])
            )
            sits = torch.concatenate([sits, angles], 2)

        encoded_input, encoded_doy = self.spatio_spectral_encoder(
            sits, batch.doy_s2, batch.doy_lidar, pad_mask=batch.pad_mask
        )
        if isinstance(self.temporal_encoder, TemporalEncoder):
            temporal_embeddings = self.temporal_encoder(
                encoded_input, encoded_doy, pad_mask=batch.pad_mask
            )
        elif isinstance(self.temporal_encoder, TemporalEncoderTQ):
            temporal_embeddings = self.temporal_encoder(
                encoded_input,
                encoded_doy,
                doy_lidar=torch.Tensor(batch.doy_lidar).to(sits.device),
                pad_mask=batch.pad_mask,
            )
        else:
            raise NotImplementedError

        if self.upsample_module is not None:
            temporal_embeddings = self.upsample_module(temporal_embeddings)

        return temporal_embeddings

    def forward(self, batch: BatchData) -> Prediction:
        """Forward.

        Parameters
        ----------
        batch: torch.Tensor

        Return
        ------
        torch.Tensor
        """
        temporal_embeddings = self.encoder(batch)
        height = self.regression(temporal_embeddings)

        if self.classification is not None:
            classif = self.classification(temporal_embeddings)
            return Prediction(height, classif)
        return Prediction(height)

    def predict(self, batch: BatchData) -> Prediction:
        """Predict.

        Parameters
        ----------
        batch: torch.Tensor

        Return
        ------
        torch.Tensor
        """
        self.eval()
        with torch.no_grad():
            return self.forward(batch)

    def get_prediction_margin(self) -> int:
        """
        Compute the margin added to the image patches during
        the prediction step. Predicted margin pixels will have
        the "border effect". So when we reconstruct the image from predicted
        patches, the margins are not taken into account.

        We count the number of Conv in Feature Extraction module
        If we do not use SR10to5, but use upsampling module, then
        + one Conv before upsampling,
        Then we multiply by upsampling factor + one Conv after upsampling
        """
        if self.upsample_module is not None:
            factor = self.upsample_module.upsampling_factor
            nb_up_conv = 1
        else:
            factor = 1
            nb_up_conv = 0

        if isinstance(self.spatio_spectral_encoder.spectral_encoding.model, Unet):
            spectral_margin = (
                self.spatio_spectral_encoder.spectral_encoding.model.get_prediction_margin()
            )
        elif isinstance(
                self.spatio_spectral_encoder.spectral_encoding.model, ESREncoder
        ):
            spectral_margin = (
                    5
                    * (len(self.spatio_spectral_encoder.spectral_encoding.model.blocks) - 2)
                    + 2
            )
        elif isinstance(self.spatio_spectral_encoder.spectral_encoding.model, RDBNet):
            spectral_margin = (
                    2
                    + self.spatio_spectral_encoder.spectral_encoding.model.num_rdb
                    * self.spatio_spectral_encoder.spectral_encoding.model.num_dense_layers
                    + 1
            )
        else:
            raise NotImplementedError
        return int((spectral_margin + nb_up_conv) * factor + nb_up_conv)



class DeepTreeSITSDoubleModel(DeepTreeSITSModel):
    """
    Input encoding class:
    spectro-spatial encoding + positional encoding
    """

    def __init__(
            self,
            spatio_spectral_encoder: SSEncoder,
            temporal_encoder: TemporalEncoder | TemporalEncoderTQ,
            regression: MLP | MLPDouble,
            upsample_module: UpsamleFeatures | None = None,
            sr_10to5: InferenceSR | None = None,
            classification: MLP | None = None,
    ) -> None:
        super().__init__(spatio_spectral_encoder, temporal_encoder,
                         regression, upsample_module, sr_10to5, classification)
        assert self.classification is None

    def forward(self, batch: BatchData) -> Prediction:
        """
        Forward with double branched MLP for height and classif
        """
        temporal_embeddings = self.encoder(batch)
        height, veg_logit = self.regression(temporal_embeddings)
        return Prediction(height, veg_logit)
