from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_ellipse_outside_fraction_torch(
    centroids: torch.Tensor,
    bbox_sizes: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    num_samples_x: int = 64,
    num_samples_y: int = 64,
) -> torch.Tensor:
    """Differentiable approximate outside-channel fraction for axis-aligned ellipses.

    Centroids are in image pixel coordinates as ``(x, y)``. Bbox sizes are pixel
    ``(width, height)``. The channel mask uses 1/True for admissible channel
    pixels. Samples outside the image are counted as outside the channel.
    """
    if centroids.shape[-1] != 2:
        raise ValueError(f"centroids must have shape (..., 2), got {tuple(centroids.shape)}")
    if bbox_sizes.shape[-1] != 2 or bbox_sizes.shape[:-1] != centroids.shape[:-1]:
        raise ValueError("bbox_sizes must have shape matching centroids, (..., 2)")
    if channel_mask.ndim != 2:
        raise ValueError(f"channel_mask must be 2D, got {tuple(channel_mask.shape)}")
    if num_samples_x <= 0 or num_samples_y <= 0:
        raise ValueError("num_samples_x and num_samples_y must be positive")

    device = centroids.device
    dtype = centroids.dtype
    mask = channel_mask.to(device=device, dtype=dtype)
    height, width = mask.shape
    bbox_sizes = bbox_sizes.to(device=device, dtype=dtype)

    if torch.any(~torch.isfinite(centroids)) or torch.any(~torch.isfinite(bbox_sizes)):
        raise ValueError("centroids and bbox_sizes must be finite")
    if torch.any(bbox_sizes <= 0):
        raise ValueError("bbox_sizes must be positive")

    sample_x = torch.linspace(-1.0, 1.0, num_samples_x, device=device, dtype=dtype)
    sample_y = torch.linspace(-1.0, 1.0, num_samples_y, device=device, dtype=dtype)
    unit_y, unit_x = torch.meshgrid(sample_y, sample_x, indexing="ij")
    unit_disk = unit_x.square() + unit_y.square() <= 1.0
    offsets = torch.stack([unit_x[unit_disk], unit_y[unit_disk]], dim=-1)

    flat_centroids = centroids.reshape(-1, 2)
    flat_sizes = bbox_sizes.reshape(-1, 2)
    sample_pixels = flat_centroids[:, None, :] + offsets[None, :, :] * (flat_sizes[:, None, :] / 2.0)

    normalized_x = 2.0 * sample_pixels[..., 0] / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * sample_pixels[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([normalized_x, normalized_y], dim=-1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        mask.reshape(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(flat_centroids.shape[0], -1)
    return (1.0 - sampled.mean(dim=1)).reshape(centroids.shape[:-1])
