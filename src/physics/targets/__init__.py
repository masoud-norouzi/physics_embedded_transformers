"""Target parameterizations for learned rollout models."""

from .speed_angle import (
    SPEED_ANGLE_TARGET_FEATURES,
    velocity_target_features_for_parameterization,
    derive_speed_angle_targets_np,
    derive_speed_angle_targets_torch,
    reconstruct_velocity_from_speed_angle_torch,
    wrap_angle_np,
    wrap_angle_torch,
)

__all__ = [
    "SPEED_ANGLE_TARGET_FEATURES",
    "velocity_target_features_for_parameterization",
    "derive_speed_angle_targets_np",
    "derive_speed_angle_targets_torch",
    "reconstruct_velocity_from_speed_angle_torch",
    "wrap_angle_np",
    "wrap_angle_torch",
]
