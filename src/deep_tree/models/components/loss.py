"""
Loss module.
"""

import inspect
import logging

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchmetrics import Metric

logger = logging.getLogger(__name__)

EPS = 1e-5


class PixelLossWrapper(torch.nn.Module):
    """
    Wrapper class from pixel losses.
    """

    def __init__(
            self, loss_fn: torch.nn.Module, name: str, weight: float = 1.0
    ) -> None:
        """
        Constructor
        """
        super().__init__()
        self.name = name
        self.loss_fn = loss_fn
        self.weight = weight

    def has_mask_arg(self) -> bool:
        """Check if loss takes mask as parameter"""
        sig = inspect.signature(self.loss_fn.forward)
        return "mask" in sig.parameters or "mask_no_veg" in sig.parameters

    def forward(
            self,
            target: torch.Tensor,
            predicted: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Forward and compute loss against prediction.
        Mask values are:
        1 - no vegetation
        0 - vegetation
        """

        if mask is not None:
            assert predicted.shape[2:] == mask.shape[2:]  # type: ignore
            assert target.shape[2:] == mask.shape[2:]  # type: ignore

            # If loss function has mask parameter
            if self.has_mask_arg():
                return self.weight * self.loss_fn(predicted, target, mask)

            predicted = predicted.permute(1, 0, 2, 3)[:, (mask == 0).squeeze(1)]
            target = target.permute(1, 0, 2, 3)[:, (mask == 0).squeeze(1)]
            if predicted.shape[0] == 1:
                predicted, target = predicted.squeeze(0), target.squeeze(0)
        else:
            if isinstance(self.loss_fn, Metric):
                predicted = predicted.flatten()
                target = target.flatten()
        return self.weight * self.loss_fn(predicted, target)


class ClassifMetricWrapper(PixelLossWrapper):
    """
    Wrapper class from pixel losses.
    """

    def forward(
            self,
            target: torch.Tensor,
            predicted: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Forward and compute classification loss against prediction.
        Mask currently is not used
        """
        if mask is not None:
            assert predicted.shape == mask.shape  # type: ignore
        if isinstance(self.loss_fn, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)):
            return self.weight * self.loss_fn(predicted, target.float())
        predicted = F.sigmoid(predicted)
        pred_label = (predicted > 0.5).float()
        return self.weight * self.loss_fn(pred_label, target)


class MAELoss(torch.nn.Module):
    """
    Compute custom weighted MAE loss
    with weight=1 for tree pixels and weight=weight_nv
    for non vegetation pixels.
    """

    def __init__(self, reduction: str | None = "mean", weight_nv: float | None = None):
        super().__init__()
        self.reduction = reduction
        self.loss = torch.nn.L1Loss(reduction="none")  # type: ignore
        self.weight_nv = weight_nv

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # type:ignore
        """Forward pass."""
        loss = self.loss(prediction, target)

        if mask is not None:
            mask = (mask == 0).to(torch.int)
            if self.weight_nv is not None:
                mask = mask.to(torch.float32)
                mask[mask == 0] = self.weight_nv

            valid_pix = mask.sum()
            loss_sum_patch = (loss * mask).sum()
        else:
            valid_pix = loss.shape.numel()
            loss_sum_patch = loss.sum()

        if self.reduction == "mean":
            return loss_sum_patch / valid_pix
        if self.reduction == "sum":
            return loss_sum_patch
        return loss if mask is None else loss*mask



class RMSELoss(torch.nn.Module):
    """Compute RMSE loss."""

    def __init__(self, reduction: str | None = "mean"):
        """
        Constructor.
        """
        super().__init__()
        self.reduction = reduction
        self.loss = torch.nn.MSELoss(reduction="none")  # type: ignore

    def forward(
            self,
            predicted: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # type:ignore
        """
        Forward with mask or not.
        """
        if mask is None:
            loss = self.loss(predicted, target)
        else:
            loss = self.loss(
                predicted.permute(1, 0, 2, 3)[:, (mask == 0).squeeze(1)],
                target.permute(1, 0, 2, 3)[:, (mask == 0).squeeze(1)]
            )

        if self.reduction == "mean":
            return torch.sqrt(loss.mean())
        if self.reduction == "sum":
            return torch.sqrt(loss).sum()
        return torch.sqrt(loss)


class PatchAvgLoss(torch.nn.Module):
    """Compute PatchAverage loss."""

    def __init__(
            self,
            loss: torch.nn.Module = torch.nn.L1Loss(reduction="none"),
            weight_nv: float | None = None
    ):
        """
        Constructor.
        """
        super().__init__()
        self.loss = loss
        self.weight_nv = weight_nv

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # type:ignore
        """
        Forward with mask or not.

        Parameters
        ----------
        prediction: torch.Tensor [Bx1xWxH]
        target: torch.Tensor [Bx1xWxH]
        mask: torch.Tensor | None = None [Bx1xWxH]

        Return
        ------
        List[torch.Tensor]
        """

        loss = self.loss(prediction, target)
        if mask is not None:
            mask = (mask == 0).to(torch.int)
            if self.weight_nv is not None:
                mask = mask.to(torch.float32)
                mask[mask == 0] = self.weight_nv
            valid_pix = mask.sum((1, 2, 3))
            loss_sum_patch = (loss * mask).sum((1, 2, 3))
            loss_avg_patch = loss_sum_patch / (valid_pix + EPS)
        else:
            loss_sum_patch = loss.sum((1, 2, 3))
            loss_avg_patch = loss_sum_patch / loss.shape[-3:].numel()
        return loss_avg_patch.mean()


class GradientDifferenceLoss(torch.nn.Module):
    """
    Gradient difference Loss.
    Should give sharper edges.
    """

    def __init__(self, reduction: str | None = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
            self,
            predicted: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward GDL"""

        predicted = predicted.float()
        target = target.float()

        grad_x = (
                predicted.diff(dim=-1).clamp(-1e3, 1e3)
                - target.diff(dim=-1).clamp(-1e3, 1e3)
        ).pow(2)
        grad_y = (
                predicted.diff(dim=-2).clamp(-1e3, 1e3)
                - target.diff(dim=-2).clamp(-1e3, 1e3)
        ).pow(2)

        if mask is not None:
            mask = mask != 1

            mask_x = mask[..., :, 1:]  # last dim reduced by diff(dim=-1)
            mask_y = mask[..., 1:, :]  # second-last dim reduced by diff(dim=-2)
            grad_x = grad_x * mask_x.float()
            grad_y = grad_y * mask_y.float()

        if self.reduction == "mean":
            if mask is not None:
                return (
                        (grad_x * mask_x.float()).sum() / mask_x.sum()
                        + (grad_y * mask_y.float()).sum() / mask_y.sum()
                ) / 2
            return (grad_x.mean() + grad_y.mean()) / 2
        if self.reduction == "sum":
            return grad_x.sum() + grad_y.sum()
        raise NotImplementedError  # no reduction


class WeightedGradientDifferenceLoss(torch.nn.Module):
    """
    Gradient Difference Loss with texture-aware weighting.
    Downweights homogeneous regions based on target gradient magnitude.
    Masked pixels are excluded from normalization.
    """

    def __init__(self, reduction: str = "mean", weight_nv: float | None = None
                 ):
        super().__init__()
        self.reduction = reduction
        self.weight_nv = weight_nv

    def forward(
            self,
            predicted: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """WGDL forward"""
        predicted = predicted.float()
        target = target.float()

        # Gradients
        pred_dx = predicted.diff(dim=-1)
        pred_dy = predicted.diff(dim=-2)
        tgt_dx = target.diff(dim=-1)
        tgt_dy = target.diff(dim=-2)

        # Texture strength (gradient magnitude)
        weight_x = torch.abs(tgt_dx)
        weight_y = torch.abs(tgt_dy)

        # Mask alignment for gradient maps
        mask_x = None
        mask_y = None
        if mask is not None:
            mask = mask != 1  # 0 = valid, 1 = invalid
            if self.weight_nv is not None:
                mask = mask.to(torch.float32)
                mask[mask == 0] = self.weight_nv
            mask_x = mask[..., :, 1:]
            mask_y = mask[..., 1:, :]

            # exclude invalid pixels from normalization
            weight_x = weight_x * mask_x
            weight_y = weight_y * mask_y

        # --- Normalize weights per-sample over *valid* pixels only ---
        def safe_percentile(tensor, mask=None, q: float = 0.95):
            """
            Computes the q-th percentile of tensor values (default 95th) within valid pixels.
            Falls back to 1.0 if no valid pixels exist.
            """
            if mask is None:
                # percentile over spatial dimensions
                tensor_flat = tensor.flatten(start_dim=-2)  # [..., H*W]
                return torch.quantile(tensor_flat, q, dim=-1, keepdim=True).clamp(min=EPS)

            # avoid all-zero mask
            valid_vals = tensor[mask.bool()]
            if valid_vals.numel() == 0:
                return torch.tensor(1.0, device=tensor.device)

            return torch.quantile(valid_vals, q).clamp(min=EPS)

        wmax_x = safe_percentile(weight_x, mask_x)
        wmax_y = safe_percentile(weight_y, mask_y)

        weight_x = weight_x / (wmax_x + EPS)
        weight_y = weight_y / (wmax_y + EPS)

        # Floor to avoid zero weights (0.1–1 range)
        weight_x = 0.1 + 0.9 * weight_x
        weight_y = 0.1 + 0.9 * weight_y

        # --- Gradient difference weighted by texture ---
        grad_diff_x = (pred_dx - tgt_dx).pow(2) * weight_x
        grad_diff_y = (pred_dy - tgt_dy).pow(2) * weight_y

        if mask is not None:
            grad_diff_x *= mask_x
            grad_diff_y *= mask_y

        # --- Reduction ---
        if self.reduction == "mean":
            denom_x = mask_x.sum() if mask is not None else grad_diff_x.numel()
            denom_y = mask_y.sum() if mask is not None else grad_diff_y.numel()
            loss = (
                           grad_diff_x.sum() / (denom_x + EPS)
                           + grad_diff_y.sum() / (denom_y + EPS)
                   ) / 2
        elif self.reduction == "sum":
            loss = grad_diff_x.sum() + grad_diff_y.sum()
        else:
            loss = (grad_diff_x + grad_diff_y) / 2

        return loss


class FFTMagnitudeLoss(torch.nn.Module):
    """
    FFT Magnitude Loss with Radial Profile.
    Compares frequency magnitude spectra of prediction vs target,
    focusing on radial frequency distribution (more robust to local noise).
    """

    def __init__(
            self, reduction: str = "mean", use_log: bool = True, emphasize_high: bool = True
    ):
        super().__init__()
        self.reduction = reduction
        self.use_log = use_log
        self.emphasize_high = emphasize_high

    def fft_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """Compute shifted FFT magnitude."""
        f = torch.fft.fft2(img.float())
        fshift = torch.fft.fftshift(f, dim=(-2, -1))
        mag = torch.abs(fshift)
        if self.use_log:
            mag = torch.log1p(mag)  # stabilizes small values
        return mag

    @staticmethod
    def radial_profile(
            mag: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute radial mean profile."""
        H, W = mag.shape[-2:]  # pylint: disable=C0103
        y, x = torch.meshgrid(
            torch.arange(H, device=mag.device),
            torch.arange(W, device=mag.device),
            indexing="ij",
        )
        center = torch.tensor([W // 2, H // 2], device=mag.device)
        r = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        r_int = r.to(torch.int64)

        if mask is None:
            mask = torch.ones_like(mag)

        max_r = r_int.max() + 1
        radial_sum = torch.zeros(max_r, device=mag.device)
        radial_count = torch.zeros(max_r, device=mag.device)

        radial_sum = radial_sum.scatter_add(0, r_int.flatten(), (mag * mask).flatten())
        radial_count = radial_count.scatter_add(0, r_int.flatten(), mask.flatten())

        radial_mean = radial_sum / (radial_count + 1e-12)
        return radial_mean

    def forward(
            self,
            predicted: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        predicted, target: (B, H, W) or (B, 1, H, W)
        mask: optional (B, H, W) binary mask (1=valid, 0=ignore)
        """
        if predicted.ndim == 4:
            predicted = predicted[:, 0]  # assume grayscale channel
        if target.ndim == 4:
            target = target[:, 0]

        B = predicted.shape[0]  # pylint: disable=C0103
        losses = []

        for b in range(B):
            pred_mag = self.fft_magnitude(predicted[b])
            tgt_mag = self.fft_magnitude(target[b])
            mask_b = None if mask is None else mask[b]

            pred_prof = self.radial_profile(pred_mag, mask_b)
            tgt_prof = self.radial_profile(tgt_mag, mask_b)

            # Normalize
            pred_prof = pred_prof / (pred_prof.sum() + 1e-12)
            tgt_prof = tgt_prof / (tgt_prof.sum() + 1e-12)

            # Emphasize high frequencies
            if self.emphasize_high:
                weights = torch.linspace(
                    0.1, 1.0, steps=pred_prof.shape[0], device=pred_prof.device
                )
            else:
                weights = torch.ones_like(pred_prof)

            diff = (pred_prof - tgt_prof).pow(2) * weights

            if self.reduction == "mean":
                losses.append(diff.mean())
            elif self.reduction == "sum":
                losses.append(diff.sum())
            else:
                losses.append(diff)

        return torch.stack(losses).mean() if self.reduction else torch.stack(losses)


class FFTHFPowerVariation(torch.nn.Module):
    """
    Compute FFTPowerVariation.
    Adapted to our task from https://src.koda.cnrs.fr/julien.michel.14/torchsisr.git
    """

    def __init__(
            self,
            support: float = 0.5,
            is_metric: bool = True,
            min_valid_ratio: float = 0.7,
            nb_subpatches: int = 4,
    ):
        """
        Constructor

        :param loss: Loss instance to use to compute HR fidelity
        """
        super().__init__()
        self.support = support
        self.min_valid_ratio = min_valid_ratio
        self.nb_subpatches = nb_subpatches
        self.is_metric = is_metric

    @staticmethod
    def fft(data: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Custom FFT with optional mask normalization.
        If mask is provided, magnitude is normalized by the number of valid pixels
        instead of full H*W.
        """
        B, _, H, W = data.shape  # pylint: disable=C0103

        out_fft = torch.cat(
            [
                torch.abs(
                    torch.fft.fftshift(
                        torch.fft.fft2(
                            data[None, i, ...].to(dtype=torch.float32), norm="backward"
                        )
                    )
                )
                for i in range(data.shape[0])
            ],
            dim=0,
        )

        if mask is None:
            norm = H * W
        else:
            # mask: (B, 1, H, W) → count valid pixels per batch
            norm = (
                mask.view(B, -1).sum(dim=1).clamp_min(1).view(B, 1, 1, 1)
            )  # broadcast to (B, C, H, W)

        return out_fft / norm

    def reshape_and_filter(
            self, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        We divide input patches into chunks and filter out
        the ones with a lot of non-forest pixels
        """
        H, W = pred.shape[-2:]  # pylint: disable=C0103
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even"

        sp_nbr_axis = int(np.sqrt(self.nb_subpatches))

        # We divide each patch into 4 sub-patches
        pred_chunk = rearrange(
            pred, "b c (h ph) (w pw) -> (b h w) c ph pw", h=sp_nbr_axis, w=sp_nbr_axis
        )

        ref_chunk = rearrange(
            ref, "b c (h ph) (w pw) -> (b h w) c ph pw", h=sp_nbr_axis, w=sp_nbr_axis
        )

        mask_chunk = rearrange(
            mask, "b c (h ph) (w pw) -> (b h w) c ph pw", h=sp_nbr_axis, w=sp_nbr_axis
        )

        # We compute the ratio of valid (forest) pixels
        non_masked_ratio = mask_chunk.to(torch.float16).mean((-1, -2)).squeeze(1)
        pred_chunk = pred_chunk[(non_masked_ratio >= self.min_valid_ratio).bool()]
        ref_chunk = ref_chunk[(non_masked_ratio >= self.min_valid_ratio).bool()]

        mask_chunk = mask_chunk[(non_masked_ratio >= self.min_valid_ratio).bool()]

        return pred_chunk, ref_chunk, mask_chunk

    @staticmethod
    def fill_masked_with_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Replace non-valid pixels in x with the per-sample per-channel mean of valid pixels.

        x: (B, C, H, W)
        mask: (B, 1, H, W), 1=valid, 0=invalid
        """
        # broadcast mask to channels
        mask = mask.expand(-1, x.size(1), -1, -1)

        # mean of valid pixels (avoid div0)
        mean_val = (x * mask).sum(dim=(-1, -2), keepdim=True) / mask.sum(
            dim=(-1, -2), keepdim=True
        )

        # fill invalid pixels with mean
        out = torch.where(mask.bool(), x, mean_val)
        return out

    def forward(
            self, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Module forward method
        """
        mask = mask == 0

        # Split into subpatches and filter invalid ones
        pred, ref, mask = self.reshape_and_filter(pred, ref, mask)

        # Fill masked regions with mean to reduce artifacts
        pred = self.fill_masked_with_mean(pred, mask)
        ref = self.fill_masked_with_mean(ref, mask)

        # FFT magnitudes
        pred_fft = self.fft(pred, mask)
        ref_fft = self.fft(ref, mask)

        # Define HF support region
        hf_support_mult = int(np.ceil(2 / self.support))
        cx, cy = ref_fft.shape[-1] // 2, ref_fft.shape[-2] // 2
        rx, ry = (
            ref_fft.shape[-1] // hf_support_mult,
            ref_fft.shape[-2] // hf_support_mult,
        )

        # Index ranges
        x_low, x_high = cx - rx, cx + rx
        y_low, y_high = cy - ry, cy + ry

        # valid pixels per sample (B, 1) — we use this per-sample for normalization
        valid_counts = mask.sum(dim=(-1, -2))  # (B, 1, H, W) → (B, 1)

        def hf_power(fft: torch.Tensor, valid_counts: torch.Tensor) -> torch.Tensor:
            """
            Compute per-sample HF power normalized by valid pixel count.
            Returns: (B, C)
            """
            parts = []
            # left
            parts.append(fft[:, :, :x_low, :].sum(dim=(-1, -2)))
            # right
            parts.append(fft[:, :, x_high:, :].sum(dim=(-1, -2)))
            # top
            parts.append(fft[:, :, x_low:x_high, :y_low].sum(dim=(-1, -2)))
            # bottom
            parts.append(fft[:, :, x_low:x_high, y_high:].sum(dim=(-1, -2)))

            total = sum(parts)  # (B, C)
            return total / valid_counts  # per sample normalization

        ref_hf_power = hf_power(ref_fft, valid_counts)
        pred_hf_power = hf_power(pred_fft, valid_counts)

        # Now average across batch
        ref_hf_power = ref_hf_power.mean(dim=0)  # (C,)
        pred_hf_power = pred_hf_power.mean(dim=0)  # (C,)

        # Percent variation
        eps = 1e-6
        variation = (pred_hf_power - ref_hf_power) / (ref_hf_power + eps)

        if self.is_metric:
            return variation * 100

        return variation.pow(2)


class ShiftedLoss(torch.nn.Module):
    """
    Shifted loss for super-resolution.
    Computed only for test
    """

    def __init__(
            self,
            pix_shift: int,
            loss_fn: torch.nn.Module = torch.nn.L1Loss(reduction="none")
    ):
        super().__init__()
        self.pix_shift = pix_shift
        self.loss_fn = loss_fn
        self.fill_value = 10000

    def forward(self, prediction, target, mask=None):
        """Patch-wise average for shift selection, but pixel-wise loss output"""
        B, _, H, W = target.shape

        # Pad target
        target = F.pad(
            target,
            (self.pix_shift, self.pix_shift, self.pix_shift, self.pix_shift),
            value=self.fill_value
        )

        # Pad mask if exists
        if mask is not None:
            mask = F.pad(
                mask,
                (self.pix_shift, self.pix_shift, self.pix_shift, self.pix_shift),
                value=0
            )

        losses_pixel = []  # per-shift pixel-wise loss
        losses_patch = []  # per-shift patch-wise average (scalar per batch item)

        # loop over shifts
        for dx in range(-self.pix_shift, self.pix_shift + 1):
            for dy in range(-self.pix_shift, self.pix_shift + 1):
                # shifted target
                target_shifted = target[
                    ...,
                    self.pix_shift + dy:self.pix_shift + dy + H,
                    self.pix_shift + dx:self.pix_shift + dx + W
                ]

                # mask
                if mask is not None:
                    mask_shifted = mask[
                        ...,
                        self.pix_shift + dy:self.pix_shift + dy + H,
                        self.pix_shift + dx:self.pix_shift + dx + W
                    ]
                    valid_pix = mask_shifted
                else:
                    valid_pix = (target_shifted != self.fill_value).float()

                # pixel-wise loss
                loss_per_pixel = self.loss_fn(prediction, target_shifted)  # (B, C, H, W)
                loss_per_pixel = loss_per_pixel * valid_pix
                losses_pixel.append(loss_per_pixel)

                # patch-wise average per batch
                patch_loss = (loss_per_pixel.sum(dim=(1, 2, 3)) /
                              valid_pix.sum(dim=(1, 2, 3)).clamp(min=1))  # (B,)
                losses_patch.append(patch_loss)

        # stack over shifts
        losses_pixel = torch.stack(losses_pixel, dim=0)  # (num_shifts, B, C, H, W)
        losses_patch = torch.stack(losses_patch, dim=0)  # (num_shifts, B)

        # find best shift per batch (per patch)
        best_shift_idx = torch.argmin(losses_patch, dim=0)  # shape: (B,)

        # select pixel-wise loss corresponding to best shift
        # gather along shift dimension
        B_range = torch.arange(B, device=prediction.device)
        pixel_losses_best_shift = losses_pixel[best_shift_idx, B_range, :, :, :]  # (B, C, H, W)

        valid_pixels = pixel_losses_best_shift > 0
        loss_scalar = pixel_losses_best_shift.sum() / valid_pixels.sum().clamp(min=1)
        return loss_scalar
