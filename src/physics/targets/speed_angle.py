from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch


SPEED_ANGLE_TARGET_FEATURES = ("speed", "angular_correction", "bbox_w", "bbox_h")
_VELOCITY_TARGET_FEATURES = ("vx", "vy", "bbox_w", "bbox_h")


def velocity_target_features_for_parameterization(target_features: tuple[str, ...]) -> tuple[str, ...]:
    if tuple(target_features) == SPEED_ANGLE_TARGET_FEATURES:
        return _VELOCITY_TARGET_FEATURES
    return tuple(target_features)


def wrap_angle_np(angle) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def derive_speed_angle_targets_np(
    future_features: np.ndarray,
    previous_features: np.ndarray,
    feature_index: Mapping[str, int],
    *,
    cfd_flip_y: bool = True,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Derive ``speed, angular_correction, bbox_w, bbox_h`` targets.

    ``previous_features`` supplies the base direction available at prediction
    time. CFD y is flipped by default because canonical model ``vy`` is
    image-down while CFD enrichment stores device-y components.
    """
    vx = future_features[..., feature_index["vx"]]
    vy = future_features[..., feature_index["vy"]]
    speed = np.hypot(vx, vy)
    true_angle = np.arctan2(vy, vx)
    base_angle, _ = base_angle_np(previous_features, feature_index, cfd_flip_y=cfd_flip_y, eps=eps)
    correction = wrap_angle_np(true_angle - base_angle)
    return np.stack(
        [
            speed,
            correction,
            future_features[..., feature_index["bbox_w"]],
            future_features[..., feature_index["bbox_h"]],
        ],
        axis=-1,
    ).astype(np.float32)


def derive_speed_angle_targets_torch(
    future_features: torch.Tensor,
    previous_features: torch.Tensor,
    feature_index: Mapping[str, int],
    *,
    cfd_flip_y: bool = True,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    vx = future_features[..., feature_index["vx"]]
    vy = future_features[..., feature_index["vy"]]
    speed = torch.sqrt(vx.square() + vy.square())
    true_angle = torch.atan2(vy, vx)
    base_angle, fallback = base_angle_torch(previous_features, feature_index, cfd_flip_y=cfd_flip_y, eps=eps)
    correction = wrap_angle_torch(true_angle - base_angle)
    target = torch.stack(
        [
            speed,
            correction,
            future_features[..., feature_index["bbox_w"]],
            future_features[..., feature_index["bbox_h"]],
        ],
        dim=-1,
    )
    return target, fallback


def reconstruct_velocity_from_speed_angle_torch(
    speed_angle_targets: torch.Tensor,
    base_features: torch.Tensor,
    feature_index: Mapping[str, int],
    *,
    cfd_flip_y: bool = True,
    max_angular_correction: float | None = None,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    speed = torch.clamp_min(speed_angle_targets[..., 0], 0.0)
    correction = speed_angle_targets[..., 1]
    if max_angular_correction is not None:
        correction = torch.clamp(
            correction,
            min=-float(max_angular_correction),
            max=float(max_angular_correction),
        )
    base_angle, fallback = base_angle_torch(base_features, feature_index, cfd_flip_y=cfd_flip_y, eps=eps)
    angle = base_angle + correction
    velocity = torch.stack([speed * torch.cos(angle), speed * torch.sin(angle)], dim=-1)
    return velocity, fallback


def base_angle_np(
    features: np.ndarray,
    feature_index: Mapping[str, int],
    *,
    cfd_flip_y: bool = True,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    cfd_x = features[..., feature_index["cfd_u_norm"]]
    cfd_y = features[..., feature_index["cfd_v_norm"]]
    if cfd_flip_y:
        cfd_y = -cfd_y
    cfd_norm = np.hypot(cfd_x, cfd_y)
    cfd_valid = np.isfinite(cfd_x) & np.isfinite(cfd_y) & (cfd_norm > eps)

    vx = features[..., feature_index["vx"]]
    vy = features[..., feature_index["vy"]]
    vel_norm = np.hypot(vx, vy)
    vel_valid = np.isfinite(vx) & np.isfinite(vy) & (vel_norm > eps)

    base_x = np.where(cfd_valid, cfd_x, np.where(vel_valid, vx, 1.0))
    base_y = np.where(cfd_valid, cfd_y, np.where(vel_valid, vy, 0.0))
    return np.arctan2(base_y, base_x), ~cfd_valid


def base_angle_torch(
    features: torch.Tensor,
    feature_index: Mapping[str, int],
    *,
    cfd_flip_y: bool = True,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfd_x = features[..., feature_index["cfd_u_norm"]]
    cfd_y = features[..., feature_index["cfd_v_norm"]]
    if cfd_flip_y:
        cfd_y = -cfd_y
    eps2 = float(eps) ** 2
    cfd_norm_sq = cfd_x.square() + cfd_y.square()
    cfd_valid = torch.isfinite(cfd_x) & torch.isfinite(cfd_y) & (cfd_norm_sq > eps2)

    vx = features[..., feature_index["vx"]]
    vy = features[..., feature_index["vy"]]
    vel_norm_sq = vx.square() + vy.square()
    vel_valid = torch.isfinite(vx) & torch.isfinite(vy) & (vel_norm_sq > eps2)

    ones = torch.ones_like(cfd_x)
    zeros = torch.zeros_like(cfd_x)
    base_x = torch.where(cfd_valid, cfd_x, torch.where(vel_valid, vx, ones))
    base_y = torch.where(cfd_valid, cfd_y, torch.where(vel_valid, vy, zeros))
    return torch.atan2(base_y, base_x), ~cfd_valid
