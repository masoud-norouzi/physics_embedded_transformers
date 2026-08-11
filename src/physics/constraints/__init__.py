"""Training-time physics constraints."""

from .geometry_loss import compute_ellipse_outside_fraction_torch
from .wall_sdf_torch import clamp_to_channel_torch, wall_sdf_to_torch

__all__ = [
    "compute_ellipse_outside_fraction_torch",
    "clamp_to_channel_torch",
    "wall_sdf_to_torch",
]
