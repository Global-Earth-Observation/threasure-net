"""Spatio-spectral encoding block"""
# pylint: disable=invalid-name
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from deep_tree.models.torch.unet_baseline import Unet


class PositionalEncoder(nn.Module):
    """
    Positional encoder class
    """

    def __init__(self, d: int, T: int = 1000, repeat: int | None = None, offset: int = 0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to encode DOYs
        """
        self.denom = self.denom.to(batch_positions.device)
        sinusoid_table = (
                batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(
            sinusoid_table[:, :, 0::2]
        )  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(
            sinusoid_table[:, :, 1::2]
        )  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )
        return sinusoid_table


# class LearnablePositionalEncoder(nn.Module):
#     """
#     Learnable positional encoder with log-spaced frequencies.
#     """
#     def __init__(self,
#                  d: int,
#                  n_freqs: int = 8,
#                  init_T: float = 365.0,
#                  normalize_dt: bool = True):
#         """
#         d: embedding dimension
#         n_freqs: number of distinct frequency scales
#         init_T: maximum period in days (e.g., 365)
#         normalize_dt: whether to scale input Δt / DOY to [0,1] or [-1,1]
#         """
#         super().__init__()
#         self.d = d
#         self.n_freqs = n_freqs
#         self.init_T = init_T
#         self.normalize_dt = normalize_dt
#
#         # Log-spaced frequencies from init_T down to 1 day
#         freqs = 2 * math.pi / torch.logspace(math.log10(init_T), math.log10(1.0), n_freqs)
#         self.freqs = nn.Parameter(freqs)  # learnable frequencies
#
#         # Optional learnable phase offsets
#         self.phase = nn.Parameter(torch.zeros(n_freqs))
#
#         # Project sin+cos features to final embedding dimension
#         self.proj = nn.Linear(n_freqs * 2, d)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, T) tensor of positions (DOY or Δt)
#         returns: (B, T, d) encodings
#         """
#         if self.normalize_dt:
#             # Example: scale ±6 months (~180 days) to [-1,1]
#             x_norm = x / self.init_T
#         else:
#             x_norm = x
#
#         # Compute angles: (B, T, n_freqs)
#         angles = x_norm[:, :, None] * self.freqs[None, None, :] + self.phase[None, None, :]
#
#         # Sin and cos encoding
#         enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, T, 2*n_freqs)
#
#         # Project to final embedding
#         return self.proj(enc)


class SpectralEncoder(nn.Module):
    """Spectral Encoder wrapper."""
    def __init__(
            self,
            model: Unet,
    ) -> None:
        super().__init__()
        self.model: Unet = model

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """Forward pass"""
        b, n, _, h, w = x.shape
        x = rearrange(x, "b n c h w -> (b n ) c h w")
        if mask is not None:
            # We ignore the padded values
            mask = rearrange(mask, "b n -> (b n )")
            x = self.model(x[~mask])
            x_res = torch.zeros(b * n, x.shape[1], h, w).to(x.device)
            x_res[~mask] = x.to(torch.float32)
            x = x_res
        else:
            x = self.model(x)
        x = rearrange(x, "( b n ) c h w -> b n c h w ", b=b)
        return x


class SSEncoder(nn.Module):
    """Spatio-spectral encoder class"""
    def __init__(
            self,
            spectral_encoding: SpectralEncoder,
            pe_encoding: PositionalEncoder,
            pe_encoding2: PositionalEncoder | None = None,
            pe_type: Literal["abs", "relative", "both"] = "abs"
    ):
        super().__init__()
        self.spectral_encoding: SpectralEncoder = spectral_encoding
        self.pe_encoding: PositionalEncoder = pe_encoding
        self.pe_encoding2: PositionalEncoder = pe_encoding2
        self.pe_type = pe_type
        # assert self.spectral_encoding.model.planes == self.pe_encoding.d

    def get_encoded_doy(
            self,
            doy_s2: torch.Tensor,
            doy_lidar: torch.Tensor | None = None,
            pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get encoded doy depending on pe_type"""

        # If inference and doy_lidar is not available
        if self.pe_type in ["relative", "both"] and doy_lidar is None:
            if self.training:
                raise AttributeError("doy_lidar should not be None for model training")
            if pad_mask is not None:
                doy_s2[pad_mask] = 1e10
            doy_lidar = doy_s2.min(-1).values

        if self.pe_type == "abs":
            return self.pe_encoding(doy_s2)
        if self.pe_type == "relative":
            return self.pe_encoding(doy_s2 - doy_lidar.view(-1, 1))
        if self.pe_type == "both":
            return self.pe_encoding(doy_s2) + self.pe_encoding2(doy_s2 - doy_lidar.view(-1, 1))
        raise NotImplementedError('pe_type should be "abs" | "relative" | "both"')

    def forward(
            self,
            data: torch.Tensor,
            doy_s2: torch.Tensor,
            doy_lidar: torch.Tensor | None = None,
            pad_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Encode input images individually and encode DOY"""
        input_encoding = self.spectral_encoding(data, pad_mask)
        doy_encoding = self.get_encoded_doy(doy_s2=doy_s2,
                                            doy_lidar=torch.Tensor(doy_lidar).to(doy_s2.device),
                                            pad_mask=pad_mask)
        doy_encoding = rearrange(doy_encoding, " b n c-> b n c 1 1")
        return input_encoding, doy_encoding.to(input_encoding)
