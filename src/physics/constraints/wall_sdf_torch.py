from __future__ import annotations

import torch
import torch.nn.functional as F

from src.physics.geometry.wall_sdf import WallSDF


_GRADIENT_EPS = 1.0e-6


def wall_sdf_to_torch(wall_sdf: WallSDF, *, device=None, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sdf = torch.as_tensor(wall_sdf.sdf, device=device, dtype=dtype)
    grad_x = torch.as_tensor(wall_sdf.grad_x, device=device, dtype=dtype)
    grad_y = torch.as_tensor(wall_sdf.grad_y, device=device, dtype=dtype)
    return sdf, grad_x, grad_y


def _sample_bilinear_torch(grid_hw: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a (H, W) grid at pixel-coordinate points.

    Uses the same align_corners=True pixel<->normalized-grid convention as
    compute_ellipse_outside_fraction_torch in geometry_loss.py.
    """
    height, width = grid_hw.shape
    flat_points = points_xy.reshape(-1, 2)
    normalized_x = 2.0 * flat_points[:, 0] / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * flat_points[:, 1] / max(height - 1, 1) - 1.0
    sample_grid = torch.stack([normalized_x, normalized_y], dim=-1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        grid_hw.reshape(1, 1, height, width),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(flat_points.shape[0])
    return sampled.reshape(points_xy.shape[:-1])


def clamp_to_channel_torch(
    candidate_xy: torch.Tensor,
    bbox_wh: torch.Tensor,
    sdf: torch.Tensor,
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
) -> torch.Tensor:
    """Differentiable counterpart of wall_sdf.clamp_to_channel_numpy.

    Pushes (x, y) along the local wall-normal (sdf gradient) direction until the
    predicted axis-aligned ellipse (from bbox_wh) is at least its support-function
    radius away from the nearest wall. A no-op wherever already contained.
    """
    if candidate_xy.shape != bbox_wh.shape:
        raise ValueError(
            f"candidate_xy and bbox_wh must share shape, got {tuple(candidate_xy.shape)} and {tuple(bbox_wh.shape)}"
        )
    height, width = sdf.shape
    in_bounds_x = candidate_xy[..., 0].clamp(0.0, float(width - 1))
    in_bounds_y = candidate_xy[..., 1].clamp(0.0, float(height - 1))
    in_bounds = torch.stack([in_bounds_x, in_bounds_y], dim=-1)

    sdf_value = _sample_bilinear_torch(sdf, in_bounds)
    grad_x_value = _sample_bilinear_torch(grad_x, in_bounds)
    grad_y_value = _sample_bilinear_torch(grad_y, in_bounds)

    # sqrt's own gradient (0.5/sqrt(x)) is +inf at x == 0, and that poisons the result with
    # NaN even through a torch.where/clamp_min that only masks the *forward* value -- the
    # unused branch's backward is still computed, and 0 * inf/nan is nan in IEEE arithmetic.
    # Clamping the squared magnitude *before* the sqrt (not the sqrt's output after) keeps
    # sqrt's argument bounded away from zero, so its gradient stays finite everywhere,
    # including at true medial-axis points where the sdf gradient is genuinely (0, 0).
    grad_norm_sq = (grad_x_value**2 + grad_y_value**2).clamp_min(_GRADIENT_EPS**2)
    safe_norm = torch.sqrt(grad_norm_sq)
    direction_x = grad_x_value / safe_norm
    direction_y = grad_y_value / safe_norm

    half_width = bbox_wh[..., 0] / 2.0
    half_height = bbox_wh[..., 1] / 2.0
    r_eff_sq = ((half_width * direction_x) ** 2 + (half_height * direction_y) ** 2).clamp_min(_GRADIENT_EPS**2)
    r_eff = torch.sqrt(r_eff_sq)
    deficit = (r_eff - sdf_value).clamp_min(0.0)

    new_x = in_bounds_x + deficit * direction_x
    new_y = in_bounds_y + deficit * direction_y
    return torch.stack([new_x, new_y], dim=-1)
