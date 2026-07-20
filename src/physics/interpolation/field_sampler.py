from __future__ import annotations

import numpy as np
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

from src.physics.cfd.domain import inside_junction_domain
from src.physics.geometry.coordinates import CoordinateConvention
from src.physics.cfd.solver import UM_TO_M

from .types import InterpolatedVelocityField, SampledVelocityField


ZERO_SPEED_DIRECTION_THRESHOLD_M_PER_S = 1.0e-14


def sample_velocity_field_cfd(field: InterpolatedVelocityField, points_cfd_um: np.ndarray) -> SampledVelocityField:
    """Sample an interpolated P2 velocity field in the frozen CFD native frame."""
    points = np.asarray(points_cfd_um, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_um must have shape (N, 2), got {points.shape}")

    inside = inside_junction_domain(points, field.mesh.geometry, tolerance_um=0.0)
    velocity = np.full((len(points), 2), np.nan, dtype=float)
    if np.any(inside):
        basis = velocity_basis(field.nodes_um, field.elements)
        coefficients = paired_velocity_to_basis_coefficients(basis, field.velocity_dof_m_per_s)
        inside_points_m = points[inside].T * UM_TO_M
        velocity[inside] = _evaluate_basis_interpolator(basis, coefficients, inside_points_m)

    speed = np.linalg.norm(velocity, axis=1)
    direction = np.full_like(velocity, np.nan)
    nonzero = np.isfinite(speed) & (speed > ZERO_SPEED_DIRECTION_THRESHOLD_M_PER_S)
    direction[nonzero] = velocity[nonzero] / speed[nonzero, None]
    return SampledVelocityField(
        points_um=points,
        u_x_m_per_s=velocity[:, 0],
        u_y_m_per_s=velocity[:, 1],
        speed_m_per_s=speed,
        direction_x=direction[:, 0],
        direction_y=direction[:, 1],
        inside_domain=inside,
        units={
            "position": "um in frozen CFD native frame",
            "velocity": "m/s in frozen CFD native frame",
            "speed": "m/s",
            "direction": "unit vector; NaN where speed is near zero or point is outside",
        },
        coordinate_frame="cfd_native_y_down",
    )


def sample_velocity_field_device(
    field: InterpolatedVelocityField,
    points_device_um: np.ndarray,
    convention: CoordinateConvention,
) -> SampledVelocityField:
    """Sample at device-Cartesian points and return device-Cartesian vectors."""
    points_device = np.asarray(points_device_um, dtype=float)
    points_cfd = convention.device_points_to_cfd(points_device)
    sampled_cfd = sample_velocity_field_cfd(field, points_cfd)
    velocity_cfd = np.column_stack([sampled_cfd.u_x_m_per_s, sampled_cfd.u_y_m_per_s])
    velocity_device = np.full_like(velocity_cfd, np.nan)
    finite = np.isfinite(velocity_cfd).all(axis=1)
    if np.any(finite):
        velocity_device[finite] = convention.cfd_vectors_to_device(velocity_cfd[finite])
    speed = np.linalg.norm(velocity_device, axis=1)
    direction = np.full_like(velocity_device, np.nan)
    nonzero = np.isfinite(speed) & (speed > ZERO_SPEED_DIRECTION_THRESHOLD_M_PER_S)
    direction[nonzero] = velocity_device[nonzero] / speed[nonzero, None]
    return SampledVelocityField(
        points_um=points_device,
        u_x_m_per_s=velocity_device[:, 0],
        u_y_m_per_s=velocity_device[:, 1],
        speed_m_per_s=speed,
        direction_x=direction[:, 0],
        direction_y=direction[:, 1],
        inside_domain=sampled_cfd.inside_domain,
        units={
            "position": "um in device Cartesian frame",
            "velocity": "m/s in device Cartesian frame",
            "speed": "m/s",
            "direction": "unit vector; NaN where speed is near zero or point is outside",
        },
        coordinate_frame="device_cartesian_y_up",
    )


sample_velocity_field = sample_velocity_field_cfd


def velocity_basis(nodes_um: np.ndarray, elements: np.ndarray) -> Basis:
    skmesh = MeshTri(np.asarray(nodes_um, dtype=float).T * UM_TO_M, np.asarray(elements, dtype=np.int64).T)
    return Basis(skmesh, ElementVector(ElementTriP2()), intorder=4)


def paired_velocity_to_basis_coefficients(basis: Basis, velocity_dof_m_per_s: np.ndarray) -> np.ndarray:
    values = np.asarray(velocity_dof_m_per_s, dtype=float)
    component_x, component_y = basis.split_indices()
    if values.shape != (len(component_x), 2):
        raise ValueError(f"velocity_dof_m_per_s has shape {values.shape}; expected {(len(component_x), 2)}")
    coefficients = np.zeros(basis.N, dtype=float)
    coefficients[component_x] = values[:, 0]
    coefficients[component_y] = values[:, 1]
    return coefficients


def _evaluate_basis_interpolator(basis: Basis, coefficients: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    interpolator = basis.interpolator(coefficients)
    try:
        evaluated = np.asarray(interpolator(points_m), dtype=float)
        return _normalize_interpolator_output(evaluated, points_m.shape[1])
    except ValueError:
        sampled = np.full((points_m.shape[1], 2), np.nan, dtype=float)
        for idx in range(points_m.shape[1]):
            try:
                evaluated = np.asarray(interpolator(points_m[:, idx : idx + 1]), dtype=float)
                sampled[idx] = _normalize_interpolator_output(evaluated, 1)[0]
            except ValueError:
                continue
        return sampled


def _normalize_interpolator_output(evaluated: np.ndarray, count: int) -> np.ndarray:
    if evaluated.shape == (2, count):
        return evaluated.T
    if evaluated.shape == (count, 2):
        return evaluated
    if count == 1 and evaluated.shape == (2,):
        return evaluated.reshape(1, 2)
    raise ValueError(f"Unexpected FEM interpolator output shape {evaluated.shape} for {count} points")
