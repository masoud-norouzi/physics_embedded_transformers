from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


_GRADIENT_EPS = 1.0e-6


@dataclass(frozen=True)
class WallSDF:
    """Signed distance to the nearest wall, positive inside the channel."""

    sdf: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray

    def __post_init__(self) -> None:
        if self.sdf.ndim != 2:
            raise ValueError(f"sdf must be 2D, got shape {self.sdf.shape}")
        if self.grad_x.shape != self.sdf.shape or self.grad_y.shape != self.sdf.shape:
            raise ValueError("sdf, grad_x, and grad_y must share the same shape")


def build_wall_sdf(channel_mask: np.ndarray) -> WallSDF:
    """Build a signed distance field from a boolean channel mask.

    Positive values are inside the channel (distance to the nearest wall),
    negative values are outside, and the zero level set is the wall boundary.
    """
    mask = np.asarray(channel_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"channel_mask must be 2D, got shape {mask.shape}")
    if not mask.any():
        raise ValueError("channel_mask must contain at least one True pixel")
    if mask.all():
        raise ValueError("channel_mask must contain at least one False pixel to define a wall")
    inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    sdf = (inside_distance - outside_distance).astype(np.float32)
    grad_y, grad_x = np.gradient(sdf)
    return WallSDF(sdf=sdf, grad_x=grad_x.astype(np.float32), grad_y=grad_y.astype(np.float32))


def ellipse_support_radius(half_width: np.ndarray, half_height: np.ndarray, direction_x: np.ndarray, direction_y: np.ndarray) -> np.ndarray:
    """Axis-aligned ellipse support function: extent of the ellipse toward (direction_x, direction_y)."""
    return np.sqrt((half_width * direction_x) ** 2 + (half_height * direction_y) ** 2)


def sample_wall_sdf_numpy(wall_sdf: WallSDF, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly sample sdf, grad_x, grad_y at arbitrary (x, y) pixel points."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape[-1] != 2:
        raise ValueError(f"points_xy must have shape (..., 2), got {points.shape}")
    flat = points.reshape(-1, 2)
    coords = np.stack([flat[:, 1], flat[:, 0]], axis=0)
    sdf_value = map_coordinates(wall_sdf.sdf, coords, order=1, mode="nearest")
    grad_x = map_coordinates(wall_sdf.grad_x, coords, order=1, mode="nearest")
    grad_y = map_coordinates(wall_sdf.grad_y, coords, order=1, mode="nearest")
    out_shape = points.shape[:-1]
    return sdf_value.reshape(out_shape), grad_x.reshape(out_shape), grad_y.reshape(out_shape)


def clamp_to_channel_numpy(candidate_xy: np.ndarray, bbox_wh: np.ndarray, wall_sdf: WallSDF) -> np.ndarray:
    """Push (x, y) back toward the channel interior so the predicted ellipse stays contained.

    The push is purely along the local wall-normal direction (the SDF gradient), so the
    along-channel component of the position is left unchanged. A no-op when already contained.

    Sampling coordinates are first clipped to the sdf array bounds. ``sdf`` is an exact
    Euclidean distance field everywhere inside those bounds (|grad(sdf)| == 1 a.e.), so a
    single push of length ``deficit`` along the gradient lands exactly on sdf == r_eff even
    for candidates deep outside the channel -- this clip only matters for candidates predicted
    outside the array entirely, where it snaps to the array edge before pushing inward, rather
    than silently under-correcting on a border-extrapolated sample.
    """
    candidate = np.asarray(candidate_xy, dtype=np.float64)
    wh = np.asarray(bbox_wh, dtype=np.float64)
    if candidate.shape != wh.shape:
        raise ValueError(f"candidate_xy and bbox_wh must share shape, got {candidate.shape} and {wh.shape}")
    height, width = wall_sdf.sdf.shape
    in_bounds_x = np.clip(candidate[..., 0], 0.0, float(width - 1))
    in_bounds_y = np.clip(candidate[..., 1], 0.0, float(height - 1))
    in_bounds = np.stack([in_bounds_x, in_bounds_y], axis=-1)
    sdf_value, grad_x, grad_y = sample_wall_sdf_numpy(wall_sdf, in_bounds)
    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    safe_norm = np.maximum(grad_norm, _GRADIENT_EPS)
    direction_x = np.where(grad_norm > _GRADIENT_EPS, grad_x / safe_norm, 0.0)
    direction_y = np.where(grad_norm > _GRADIENT_EPS, grad_y / safe_norm, 0.0)
    half_width = wh[..., 0] / 2.0
    half_height = wh[..., 1] / 2.0
    r_eff = ellipse_support_radius(half_width, half_height, direction_x, direction_y)
    deficit = np.maximum(r_eff - sdf_value, 0.0)
    new_x = in_bounds_x + deficit * direction_x
    new_y = in_bounds_y + deficit * direction_y
    return np.stack([new_x, new_y], axis=-1)
