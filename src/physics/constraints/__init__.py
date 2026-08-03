"""Training-time physics constraints."""

from .geometry_loss import compute_ellipse_outside_fraction_torch

__all__ = ["compute_ellipse_outside_fraction_torch"]
