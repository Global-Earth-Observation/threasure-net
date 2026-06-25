"""
Lightweight Residual Dense Network (RDN) / Residual Dense Blocks (RDB) implementation
for super-resolution regression or feature extraction on multi-spectral imagery (PyTorch).

- Configurable: out_channels, growth_channels, num_rdb, num_dense_layers
- Supports multiband inputs (e.g., Sentinel-2)
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv block"""
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = True,
                 activation: nn.Module = nn.ReLU(True)
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.act = activation

    def forward(self, x):
        """Forward"""
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResidualDenseBlock(nn.Module):
    """A single Residual Dense Block (RDB)
    - num_layers: number of conv layers inside RDB ("dense" style: concatenation)
    - growth_channels: channels added per dense layer
    - local feature fusion: 1x1 conv to fuse concatenated features back to out_channels
    """

    def __init__(self,
                 in_channels: int,
                 growth_channels: int = 24,
                 num_layers: int = 4,
                 kernel: int = 3,
                 activation: nn.Module = nn.ReLU(True)
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.growth = growth_channels
        self.in_channels = in_channels
        self.activation = activation()

        # create dense conv layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels + i * growth_channels
            self.layers.append(
                nn.Conv2d(in_ch, growth_channels, kernel, padding=kernel // 2, bias=True)
            )
        # local feature fusion: compress concatenated features back to in_channels
        self.lff = nn.Conv2d(
            in_channels + num_layers * growth_channels,
            in_channels,
            kernel_size=1,
            bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        features = [x]
        for conv in self.layers:
            inp = torch.cat(features, dim=1)
            out = conv(inp)
            if self.activation is not None:
                out = self.activation(out)
            features.append(out)
        concat = torch.cat(features, dim=1)
        fused = self.lff(concat)
        # residual scaling (small value helps stability); you can adjust externally
        return fused + x


class RDBNet(nn.Module):
    """Residual Dense Network composed of stacked ResidualDenseBlock modules.

    Arguments:
        in_channels: number of input spectral bands (e.g., 10 or 12 for Sentinel-2)
        out_channels: number of feature channels in trunk (e.g., 64)
        growth_channels: growth channels inside each RDB (e.g., 24)
        num_rdb: number of RDB blocks to stack
        num_dense_layers: number of layers inside each RDB
        activation: activation function
    """

    def __init__(self,
                 in_channels: int = 10,
                 out_channels: int = 64,
                 growth_channels: int = 24,
                 num_rdb: int = 3,
                 num_dense_layers: int = 4,
                 activation: nn.Module = nn.ReLU(True)):
        super().__init__()
        self.activation = activation
        self.num_rdb = num_rdb
        self.num_dense_layers = num_dense_layers

        # shallow feature extraction
        self.conv_in = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)])

        # trunk of RDBs
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(out_channels, growth_channels, num_dense_layers, activation=self.activation)
            for _ in range(num_rdb)
        ])

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[ # pylint: disable=C0103
            nn.Conv2d(out_channels * num_rdb, out_channels, 1, padding=0, stride=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)
        ])

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        shallow = self.conv_in(x)
        out = shallow

        RDBs_out = []   # pylint: disable=C0103
        # Residual blocks
        for rdb in self.rdbs:
            out = rdb(out)
            RDBs_out.append(out)

        out = self.GFF(torch.cat(RDBs_out, 1))

        out += shallow  # global residual

        return out

    def _initialize_weights(self):
        # simple init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
