"""Taken from https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py"""

# pylint: skip-file


import logging
from typing import Literal

import torch
import torch.nn as nn

my_logger = logging.getLogger(__name__)


class Unet(nn.Module):
    def __init__(
        self,
        inplanes: int = 10,
        planes: int = 32,
        encoder_widths: list = [64, 64, 64, 128],
        decoder_widths: list = [32, 32, 64, 128],
        pad_value: int = 0,
        encoder_norm: Literal["group", "batch"] = "batch",
        padding_mode: str = "reflect",
        decoding_norm: Literal["group", "batch"] = "batch",
        return_maps: bool = False,
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        skip_conv_norm: Literal["group", "batch"] = "batch",
    ):
        super().__init__()
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths

        self.decoder_widths = decoder_widths
        self.in_conv = ConvBlock(
            nkernels=[inplanes] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0], planes],
            pad_value=pad_value,
            norm=decoding_norm,
            padding_mode=padding_mode,
        )
        self.n_stages = len(encoder_widths)
        my_logger.debug(self.n_stages)
        my_logger.info(f"Last conf is {planes}")
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    d_in=decoder_widths[i],
                    d_out=decoder_widths[i - 1],
                    d_skip=encoder_widths[i - 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    norm=decoding_norm,
                    padding_mode=padding_mode,
                    skip_conv_norm=skip_conv_norm,
                )
                for i in range(self.n_stages - 1, 0, -1)
            ]
        )

    def forward(self, input):
        my_logger.debug(f"Unet in {input.shape}")
        dtype = input.dtype
        # input = input.float()  #TODO fix that required for torch.compile
        out = self.in_conv(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i](feature_maps[-1])
            feature_maps.append(out)
            # print(out.shape)
        if self.return_maps:
            maps = [out]
        # print([out.shape for out in feature_maps])
        for i in range(self.n_stages - 1):
            skip = feature_maps[-(i + 2)]
            #  print(skip.shape, out.shape)
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)
            #            out = rearrange(out, "b c h w -> b h w c")
        out = self.out_conv(out)
        my_logger.debug(f"out shape UTAE {out.shape}")
        if self.return_maps:
            return out.to(dtype), maps.to(dtype)

        return out.to(dtype)

    def get_prediction_margin(self):
        def compute_unet_prediction_margin(n_down, convs_per_block=2, kernel_size=3):
            base_margin = (kernel_size - 1) // 2 * convs_per_block
            margin = 0
            for i in range(n_down):
                margin += base_margin * (2 ** i)
            return margin
        return compute_unet_prediction_margin(n_down=len(self.down_blocks), convs_per_block=2, kernel_size=3)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        final_out=None,
        norm="batch",
        d_skip=None,
        padding_mode="reflect",
        skip_conv_norm="batch",
    ):
        super().__init__()
        d = d_out if d_skip is None else d_skip
        if skip_conv_norm == "batch":
            skip_norm_begin = nn.BatchNorm2d(d)
            skip_norm_end = nn.BatchNorm2d(d_out)
        else:
            skip_norm_begin = nn.GroupNorm(num_groups=4, num_channels=d)
            skip_norm_end = nn.GroupNorm(num_groups=4, num_channels=d_out)
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            skip_norm_begin,
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in,
                out_channels=d_out,
                kernel_size=k,
                stride=s,
                padding=p,
            ),
            skip_norm_end,
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        if final_out is None:
            final_out = d_out

        self.conv2 = ConvLayer(
            nkernels=[d_out, final_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class DownConvBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super().__init__()
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super().__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":

            def group_norm(num_feats):
                return nn.GroupNorm(
                    num_channels=num_feats,
                    num_groups=n_groups,
                )

            nl = group_norm
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super().__init__()
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)
