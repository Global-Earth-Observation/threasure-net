"""MLP for tree regression blocks"""

import torch


class MLP(torch.nn.Module):
    """Muti layers perceptron."""

    def __init__(
            self,
            in_channels: int,
            hidden_dims: tuple[int, ...],
            out_channels: int,
            bias: bool = True,
    ) -> None:
        """
        Init.

        Parameters
        ----------
        in_channels: int
        hidden_dims: Tuple[int, ...]
        out_channels: int
        bias: bool = True
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        dim_in = in_channels
        for dim_out in hidden_dims:
            self.layers.append(
                torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=bias)
            )
            self.layers.append(torch.nn.ReLU())
            dim_in = dim_out

        self.layers.append(
            torch.nn.Linear(in_features=dim_in, out_features=out_channels, bias=bias)
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        batch: torch.Tensor

        Return
        ------
        torch.Tensor
        """
        batch_size, channels, width, height = batch.shape
        batch = batch.permute(0, 2, 3, 1).reshape(-1, channels)
        for layer in self.layers:
            batch = layer(batch)
        batch = batch.view(batch_size, width, height, -1).permute(0, 3, 1, 2)
        return batch


class MLPDouble(torch.nn.Module):
    """
    Shared MLP trunk + regression + classification heads.
    """

    def __init__(
            self,
            in_channels: int,
            hidden_dims: tuple[int, ...],
            out_channels: int,
            bias: bool = True,
    ):
        super().__init__()

        # ---- shared trunk ----
        layers = []
        dim = in_channels
        for h in hidden_dims:
            layers.append(torch.nn.Linear(dim, h, bias=bias))
            layers.append(torch.nn.ReLU())
            dim = h

        self.trunk = torch.nn.Sequential(*layers)
        self.trunk_dim = dim

        # ---- heads ----
        self.reg_head = torch.nn.Linear(self.trunk_dim, out_channels, bias=bias)
        self.cls_head = torch.nn.Linear(self.trunk_dim, 1, bias=bias)

    def forward(self, batch: torch.Tensor):
        """
        Returns:
            height: (B, out_channels, H, W)
            veg_logit: (B, 1, H, W)
        """
        B, C, H, W = batch.shape    # pylint: disable=C0103

        x = batch.permute(0, 2, 3, 1).reshape(-1, C)
        feat = self.trunk(x)

        height = self.reg_head(feat)

        veg_logit = self.cls_head(feat.detach())

        height = height.view(B, H, W, -1).permute(0, 3, 1, 2)
        veg_logit = veg_logit.view(B, H, W, 1).permute(0, 3, 1, 2)

        return height, veg_logit
