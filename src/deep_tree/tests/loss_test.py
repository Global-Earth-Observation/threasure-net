"""Pytest loss test module"""

import pytest
import torch
from torch import Tensor

from deep_tree.models.components.loss import (
    PixelLossWrapper,
    RMSELoss,
    GradientDifferenceLoss,
    FFTMagnitudeLoss, FFTHFPowerVariation,
)


class DummyLoss(torch.nn.Module):
    """A simple dummy loss to simulate custom behaviors."""

    def __init__(self, use_mask: bool = False) -> None:
        super().__init__()
        self.use_mask = use_mask

    def forward(
        self, pred: Tensor, target: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """Forward loss"""
        if self.use_mask and mask is not None:
            # Apply mask-aware absolute difference
            return ((pred - target) * mask).abs().mean()
        return (pred - target).abs().mean()


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_rmse_loss_basic(reduction: str) -> None:
    """Check RMSE loss produces positive scalar output without mask."""
    pred = torch.ones((2, 1, 4, 4))
    target = torch.zeros((2, 1, 4, 4))
    loss = RMSELoss(reduction=reduction)

    out: Tensor = loss(pred, target)
    assert out.ndim == 0  # should return scalar
    assert out > 0


def test_rmse_loss_with_mask() -> None:
    """RMSE should handle masks by applying them correctly."""
    pred = torch.ones((1, 1, 4, 4))
    target = torch.zeros((1, 1, 4, 4))
    mask = torch.tensor(
        [
            [
                [
                    [
                        1,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        1,
                        1,
                        0,
                        1,
                        0,
                        0,
                        1,
                        0,
                        1,
                    ]
                ]
            ]
        ],
        dtype=torch.float,
    ).reshape(
        1, 1, 4, 4
    ).to(torch.bool)  # alternating 0/1 mask

    loss = RMSELoss(reduction="mean")
    out: Tensor = loss(pred, target, mask)
    assert out > 0


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_gradient_difference_loss(reduction: str) -> None:
    """Check GradientDifferenceLoss basic functionality."""
    pred = torch.rand((1, 1, 8, 8))
    target = torch.rand((1, 1, 8, 8))
    loss = GradientDifferenceLoss(reduction=reduction)

    out: Tensor = loss(pred, target)
    assert out >= 0


def test_gradient_difference_loss_with_mask() -> None:
    """GradientDifferenceLoss should apply mask correctly."""
    pred = torch.rand((1, 1, 8, 8))
    target = torch.rand((1, 1, 8, 8))
    mask = torch.zeros((1, 1, 8, 8))
    mask[:, :, ::2, ::2] = 1  # checkerboard mask (half 1s, half 0s)
    mask = mask.to(torch.bool)

    loss = GradientDifferenceLoss(reduction="mean")
    out: Tensor = loss(pred, target, mask)
    assert out >= 0


@pytest.mark.parametrize("reduction", ["mean", "sum", None])
def test_fft_magnitude_loss(reduction: str | None) -> None:
    """FFT Magnitude Loss should compute frequency-domain error."""
    pred = torch.rand((1, 1, 16, 16))
    target = torch.rand((1, 1, 16, 16))
    loss = FFTMagnitudeLoss(reduction=reduction)

    out: Tensor = loss(pred, target)
    if reduction is not None:
        assert out >= 0
    else:
        assert torch.all(out >= 0)


def test_pixel_loss_wrapper_without_mask() -> None:
    """PixelLossWrapper should call loss correctly without masks."""
    pred = torch.ones((1, 1, 6, 6))
    target = torch.zeros((1, 1, 6, 6))
    wrapper = PixelLossWrapper(DummyLoss(), "dummy")

    out: Tensor = wrapper(target, pred)
    assert out > 0


def test_pixel_loss_wrapper_with_loss_that_accepts_mask() -> None:
    """PixelLossWrapper should pass mask if loss supports it."""
    pred = torch.ones((1, 1, 4, 4))
    target = torch.zeros((1, 1, 4, 4))
    mask = torch.zeros((1, 1, 4, 4))
    mask[:, :, :2, :2] = 1  # only top-left quadrant is active
    mask = mask.to(torch.bool)

    wrapper = PixelLossWrapper(DummyLoss(use_mask=True), "dummy")
    out: Tensor = wrapper(target, pred, mask=mask)
    assert out >= 0

@pytest.fixture
def dummy_data():
    """Create dummy data"""
    B, C, H, W = 2, 3, 8, 8
    pred = torch.rand(B, C, H, W)
    ref = torch.rand(B, C, H, W)
    mask = torch.ones(B, 1, H, W)  # all valid
    return pred, ref, mask


def test_fft_normalization(dummy_data):
    pred, _, mask = dummy_data
    B, C, H, W = pred.shape
    module = FFTHFPowerVariation()

    # --- Case 1: no mask ---
    out_fft = module.fft(pred, mask=None)
    assert torch.all(out_fft >= 0)

    expected_norm = H * W

    # Reproduce the module’s per-image FFT for batch=0
    raw_fft = torch.abs(
        torch.fft.fftshift(
            torch.fft.fft2(pred[None, 0].float(), norm="backward")
        )
    ) / expected_norm

    # Compare one channel
    assert torch.allclose(out_fft[0], raw_fft[0], atol=1e-6)

    # --- Case 2: with mask (normalize by valid pixels) ---
    out_fft = module.fft(pred, mask=mask)
    assert torch.all(out_fft >= 0)

    # Compute expected normalization: valid pixels per image
    valid_pixels = mask.view(B, -1).sum(dim=1)

    # Reproduce the module’s per-image FFT for batch=0
    raw_fft = torch.abs(
        torch.fft.fftshift(
            torch.fft.fft2(pred[None, 0].float(), norm="backward")
        )
    ) / valid_pixels[0].item()

    # Compare one channel
    assert torch.allclose(out_fft[0], raw_fft[0], atol=1e-6)


def test_fft_fill_masked_with_mean(dummy_data):
    pred, _, mask = dummy_data
    module = FFTHFPowerVariation()

    # Make half invalid
    mask[:, :, :4, :] = 0
    out = module.fill_masked_with_mean(pred, mask)

    # Check shape and type
    assert out.shape == pred.shape
    assert torch.is_tensor(out)

    # All invalid entries replaced with mean
    for b in range(pred.size(0)):
        for c in range(pred.size(1)):
            valid_vals = pred[b, c][mask[b, 0].bool()]
            mean_val = valid_vals.mean()
            invalid_vals = out[b, c][~mask[b, 0].bool()]
            assert torch.allclose(invalid_vals, mean_val, atol=1e-6)


def test_fft_reshape_and_filter_keeps_valid(dummy_data):
    pred, ref, mask = dummy_data
    module = FFTHFPowerVariation(nb_subpatches=4, min_valid_ratio=0.7)

    # First, all valid
    pred_chunks, ref_chunks, mask_chunks = module.reshape_and_filter(pred, ref, mask)
    assert pred_chunks.shape[0] == 4 * pred.shape[0]  # 4 subpatches per batch

    # Now make mask invalid (all zeros → should filter everything)
    mask[:] = 0
    pred_chunks, ref_chunks, mask_chunks = module.reshape_and_filter(pred, ref, mask)
    assert pred_chunks.shape[0] == 0
    assert ref_chunks.shape[0] == 0
