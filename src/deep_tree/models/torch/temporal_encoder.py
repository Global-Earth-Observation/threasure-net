"""Multi-head attention"""
#pylint: disable=W1203

import logging

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn

my_logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, scale: float, attn_dropout: float = 0):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward"""
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.scale
        if pad_mask is not None:
            _MASKING_VALUE = -1e+30 if attn.dtype == torch.float32 else -1e+4   #pylint: disable=C0103
            attn = attn.masked_fill(pad_mask.unsqueeze(1), _MASKING_VALUE)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output


class LearnedMultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head: int, d_k: int, d_in: int, attn_dropout: float = 0):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.Q = nn.Parameter(  #pylint: disable=C0103
            torch.zeros((n_head, d_k))
        ).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))
        self.fc1_k = nn.Linear(d_in, n_head * d_k, bias=False)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))
        self.attention = ScaledDotProductAttention(scale=d_k ** 0.5, attn_dropout=attn_dropout)

    def forward(self, v: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """

        Args:
            v (): b,t,c
            pad_mask (): b,t, true means the value should take part in attention

        Returns:

        """
        d_k, n_head = self.d_k, self.n_head
        sz_b, seq_len, _ = v.size()
        q = repeat(
            self.Q, "head dk -> (head b) dk", b=sz_b
        )  # torch.stack([self.Q for _ in range(sz_b)], dim=1)
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


class TemporalEncoder(nn.Module):
    """
    Spatio-spectral encoder class
    Takes encoded SITS + their encoded DOY
    and passes them to attention block
    """
    def __init__(
            self,
            n_head: int = 4,
            d_k: int = 16,
            d_in: int = 64,
            attn_dropout: float = 0.
    ):
        super().__init__()
        self.attention = LearnedMultiHeadAttention(n_head, d_k, d_in, attn_dropout)

    def forward(self, input_encoding: torch.Tensor, doy_encoding: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward"""
        x = input_encoding + doy_encoding
        b, _, _, h, w = x.shape
        if pad_mask is not None:
            pad_mask = repeat(
                pad_mask, "b t -> b t h w ", h=h, w=w
            )
            pad_mask = rearrange(pad_mask, " b t h w -> (b h w ) t")
        x = rearrange(x, "b t c h w -> (b h w ) t c")
        x = self.attention(x, pad_mask=pad_mask)
        x = rearrange(x, "(b h w ) c -> b c h w ", b=b, h=h, w=w)
        return x
