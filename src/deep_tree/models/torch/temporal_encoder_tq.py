# pylint: disable=W1203

"""
Temporal encoder with temporal query that condition predictions
on a specific LiDAR acquisition date
"""

import logging

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn

from deep_tree.models.torch.temporal_encoder import LearnedMultiHeadAttention

my_logger = logging.getLogger(__name__)


class PositionalTQEncoding(nn.Module):
    """
    PositionalEncoder to encode LiDAR DOY.
    """

    def __init__(self,
                 d_k: int,
                 T: int = 365,
                 offset: int = 0,
                 normalize_dt: bool = False,
                 center: bool = True):
        super().__init__()

        self.d_k = d_k
        self.T = T  # pylint: disable=C0103
        self.offset = offset
        self.normalize_dt = normalize_dt
        self.center = center

        # Denominator like in PositionalEncoder
        self.denom = torch.pow(
            torch.tensor(T, dtype=torch.float32),
            2 * (torch.arange(offset, offset + d_k).float() // 2) / d_k,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] or [B,1] tensor of time values
        returns: [B, d_k] positional-like encoding
        """
        if isinstance(t, (list, np.ndarray)):
            t = torch.tensor(t, dtype=torch.float32)
        elif isinstance(t, torch.Tensor):
            t = t.float()
        if t.ndim == 1:
            t = t[:, None]

        if self.center:
            t = t - 183

        if self.normalize_dt:
            t = t / self.T

        denom = self.denom.to(t.device)
        angles = t / denom[None, :]  # [B, d_k]

        # Apply sin/cos to even/odd indices
        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])

        return angles



class LearnedTemporalMultiHeadAttention(LearnedMultiHeadAttention):
    """Temporal Multi Head Attention with Learnable Option (Fourier)"""
    def __init__(self, n_head: int, d_k: int, d_in: int,
                 attn_dropout: float = 0, init_T: int = 365):
        super().__init__(n_head, d_k, d_in, attn_dropout)
        self.temporal_encoder = PositionalTQEncoding(d_k=d_k, T=init_T)

    def forward(self,
                v: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                doy_lidar: torch.Tensor | None = None
                ) -> torch.Tensor:
        """

        Args:
            v (): b,t,c
            pad_mask (): b,t, true means the value should take part in attention

        Returns:
        :param doy_lidar:

        """
        d_k, n_head = self.d_k, self.n_head
        sz_b, seq_len, _ = v.size()
        q = self.temporal_encoder(doy_lidar)
        q = repeat(q, "b dk -> (b pix) dk", pix=sz_b // q.shape[0])
        q = repeat(
            q, "b dk -> (head b) dk", head=n_head
        )

        my_logger.debug(f"query{q.shape}")
        # q = rearrange(q, "head b c -> (head b) c")
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k).to(torch.float32)
        my_logger.debug(f"key {k.shape}")
        k = rearrange(k, "b t head c -> (head b) t c")
        my_logger.debug(f"key {k.shape}")
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk
            my_logger.debug(f"Pad mask shape {pad_mask.shape}")
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1))
        v = rearrange(v, "head b t c -> (head b) t c")
        my_logger.debug(f"value {v.shape}")
        my_logger.debug(f"query{q.shape}")
        output = self.attention(
            q=q, k=k, v=v, pad_mask=pad_mask
        ).squeeze(1)  # head*b,d_in
        my_logger.debug(f"output {output.shape}")
        return rearrange(output, "(h b) c -> b (h c)", b=sz_b)


class TemporalEncoderTQ(nn.Module):
    """Spatio-spectral encoder class"""

    def __init__(
            self,
            n_head: int = 4,
            d_k: int = 16,
            d_in: int = 64,
            attn_dropout: float = 0.,
    ):
        super().__init__()
        self.attention = LearnedTemporalMultiHeadAttention(n_head, d_k, d_in,
                                                           attn_dropout)

    def forward(self, input_encoding: torch.Tensor, doy_encoding: torch.Tensor,
                doy_lidar: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass"""
        x = input_encoding + doy_encoding
        b, _, _, h, w = x.shape
        if pad_mask is not None:
            pad_mask = repeat(
                pad_mask, "b t -> b t h w ", h=h, w=w
            )
            pad_mask = rearrange(pad_mask, " b t h w -> (b h w ) t")
        x = rearrange(x, "b t c h w -> (b h w ) t c")
        x = self.attention(x, doy_lidar=doy_lidar, pad_mask=pad_mask)
        x = rearrange(x, "(b h w ) c -> b c h w ", b=b, h=h, w=w)
        return x
