import numpy as np
import torch
from scipy.ndimage import label
import torch.nn.functional as F


def filter_small_objects(mask, min_size):
    """
    mask: binary mask (1 for object, 0 for background), shape (1, 1, H, W)
    min_size: minimum number of pixels to keep
    """
    kernel_size = min_size - 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)

    # Convolve to count local object pixels
    local_sum = F.conv2d(mask.float(), kernel, padding=kernel_size // 2)

    # Keep only regions where local sum ≥ min_size
    mask_filtered = (local_sum >= min_size).float() * mask

    return mask_filtered


def remove_small_components(binary_mask, min_size):
    """
    binary_mask: torch.BoolTensor of shape (H, W)
    min_size: int, minimum number of pixels to keep a component

    Returns a torch.BoolTensor of the same shape
    """

    mask_np = binary_mask.cpu().numpy().astype(np.uint8)
    labeled, num = label(mask_np)

    # Count sizes
    counts = np.bincount(labeled.ravel())
    remove = counts < min_size
    remove_mask = remove[labeled]
    mask_np[remove_mask] = 0
    return torch.from_numpy(mask_np).to(binary_mask.device).bool()


def remove_isolated(mask: torch.Tensor, min_neighbors: int = 2) -> torch.Tensor:
    """
    Remove isolated '1' pixels from a binary mask (B,1,H,W) or (H,W).
    A pixel must have at least `min_neighbors` active neighbors to survive.
    """
    mask = ~mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)

    # Count neighbors with a 3x3 kernel
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)
    neighbor_count = F.conv2d(mask.float(), kernel, padding=1)

    # Keep pixel if it has at least `min_neighbors` neighbors (including itself)
    cleaned = (neighbor_count >= min_neighbors).float() * mask
    return ~(cleaned.bool())


def change_target_resolution(target_tensor: torch.Tensor,
                             target_tensor_mask: torch.Tensor,
                             target_resolution: float,
                             remove_small: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Change resolution of lidar target and the corresponding mask"""
    if target_resolution == 10:
        maxpool = torch.nn.MaxPool2d(10)
    else:
        maxpool = torch.nn.MaxPool2d(5)
    # target_tensor = target_tensor.unsqueeze(0)
    if target_resolution == 2.5:
        target_tensor = target_tensor.to(torch.float32).repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        target_tensor = maxpool(target_tensor)

        target_tensor_mask = (~target_tensor_mask.to(torch.bool)).to(torch.float32).repeat_interleave(2,
                                                                                                      dim=-2).repeat_interleave(
            2, dim=-1)
        target_tensor_mask = ~(maxpool(target_tensor_mask).to(torch.bool))
        # target_tensor = F.interpolate(target_tensor, scale_factor=0.2, mode="bicubic")
    elif target_resolution in [5, 10]:
        target_tensor = maxpool(target_tensor.to(torch.float32))
        target_tensor_mask = ~(maxpool((~target_tensor_mask.to(torch.bool)).to(torch.float32)).to(torch.bool))
        # target_tensor = F.interpolate(target_tensor, scale_factor=0.2, mode="bicubic")
    else:
        raise NotImplementedError

    assert torch.equal(target_tensor_mask[target_tensor == 0], ~target_tensor.to(torch.bool)[target_tensor == 0])

    # Get rid of small vegetation if ever it was missed during data preparation
    target_tensor[target_tensor < 15] = 0
    target_tensor_mask[target_tensor < 15] = 1

    target_tensor_mask = remove_isolated(target_tensor_mask)

    return target_tensor, target_tensor_mask
