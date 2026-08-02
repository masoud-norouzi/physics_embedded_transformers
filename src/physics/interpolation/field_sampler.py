from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

from src.physics.cfd.domain import inside_junction_domain
from src.physics.cfd.solver import UM_TO_M
from src.physics.full_device_cfd.domain import inside_full_device_domain
from src.physics.geometry.coordinates import CoordinateConvention

from .types import InterpolatedVelocityField, SampledVelocityField


ZERO_SPEED_DIRECTION_THRESHOLD_M_PER_S = 1.0e-14
_PROJECTION_LOOKUP_CACHE: dict[tuple[int, int, int], "_ValidPointProjectionLookup"] = {}
_VELOCITY_BASIS_CACHE: dict[tuple[int, int, int], tuple[Basis, np.ndarray, np.ndarray]] = {}


def sample_velocity_field_cfd(field: InterpolatedVelocityField, points_cfd_um: np.ndarray) -> SampledVelocityField:
    """Sample an interpolated P2 velocity field in the frozen CFD native frame."""
    points = np.asarray(points_cfd_um, dtype=float)
    if points.ndim == 1 and points.shape == (2,):
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_um must have shape (N, 2), got {points.shape}")

    original_valid = _inside_fluid_domain(points, field.mesh.geometry)
    sample_points = _sampling_points_with_projection(points, original_valid, field)
    inside = _inside_fluid_domain(sample_points, field.mesh.geometry)
    velocity = np.full((len(points), 2), np.nan, dtype=float)
    if np.any(inside):
        basis, component_x, component_y = velocity_basis_components(field)
        coefficients = paired_velocity_to_basis_coefficients(
            basis,
            field.velocity_dof_m_per_s,
            component_x=component_x,
            component_y=component_y,
        )
        inside_points_m = sample_points[inside].T * UM_TO_M
        velocity[inside] = _evaluate_basis_interpolator(basis, coefficients, inside_points_m)

    speed = np.linalg.norm(velocity, axis=1)
    projection_distance = np.linalg.norm(sample_points - points, axis=1)
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
        original_valid=original_valid,
        sample_points_um=sample_points,
        projection_distance_um=projection_distance,
        inlet_reference_velocity_m_per_s=field.inlet_reference_velocity_m_per_s,
        units={
            "position": "um in frozen CFD native frame",
            "sample_position": "um in frozen CFD native frame; outside queries are projected to nearest cached valid CFD point",
            "projection_distance": "um",
            "velocity": "m/s in frozen CFD native frame",
            "speed": "m/s",
            "normalized_velocity": "dimensionless; velocity divided by analytical inlet centerline maximum",
            "inlet_reference_velocity": "m/s",
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
    if points_device.ndim == 1 and points_device.shape == (2,):
        points_device = points_device.reshape(1, 2)
    points_cfd = convention.device_points_to_cfd(points_device)
    sampled_cfd = sample_velocity_field_cfd(field, points_cfd)
    velocity_cfd = np.column_stack([sampled_cfd.u_x_m_per_s, sampled_cfd.u_y_m_per_s])
    sample_points_device = convention.cfd_points_to_device(sampled_cfd.sample_points_um)
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
        original_valid=sampled_cfd.original_valid,
        sample_points_um=sample_points_device,
        projection_distance_um=sampled_cfd.projection_distance_um,
        inlet_reference_velocity_m_per_s=field.inlet_reference_velocity_m_per_s,
        units={
            "position": "um in device Cartesian frame",
            "sample_position": "um in device Cartesian frame; outside queries are projected to nearest cached valid CFD point",
            "projection_distance": "um",
            "velocity": "m/s in device Cartesian frame",
            "speed": "m/s",
            "normalized_velocity": "dimensionless; velocity divided by analytical inlet centerline maximum",
            "inlet_reference_velocity": "m/s",
            "direction": "unit vector; NaN where speed is near zero or point is outside",
        },
        coordinate_frame="device_cartesian_y_up",
    )


sample_velocity_field = sample_velocity_field_cfd


def _inside_fluid_domain(points_um: np.ndarray, geometry) -> np.ndarray:
    if hasattr(geometry, "outer_ring_um") and hasattr(geometry, "inner_ring_um"):
        return inside_full_device_domain(points_um, geometry, tolerance_um=0.0)
    return inside_junction_domain(points_um, geometry, tolerance_um=0.0)


def _sampling_points_with_projection(points: np.ndarray, original_valid: np.ndarray, field: InterpolatedVelocityField) -> np.ndarray:
    if np.all(original_valid):
        return points.copy()
    geometry = field.mesh.geometry
    if not (hasattr(geometry, "outer_ring_um") and hasattr(geometry, "inner_ring_um")):
        return points.copy()
    sample_points = points.copy()
    outside = ~original_valid
    lookup = _projection_lookup(field)
    sample_points[outside] = lookup.nearest_valid_points(points[outside])
    return sample_points


def _projection_lookup(field: InterpolatedVelocityField) -> "_ValidPointProjectionLookup":
    key = (id(field.mesh), len(field.nodes_um), len(field.elements))
    cached = _PROJECTION_LOOKUP_CACHE.get(key)
    if cached is None:
        cached = _ValidPointProjectionLookup.from_field(field)
        _PROJECTION_LOOKUP_CACHE[key] = cached
    return cached


class _ValidPointProjectionLookup:
    def __init__(self, valid_points_um: np.ndarray) -> None:
        if len(valid_points_um) == 0:
            raise ValueError("Cannot build CFD projection lookup without valid sample points")
        self.valid_points_um = np.asarray(valid_points_um, dtype=float)
        self.tree = cKDTree(self.valid_points_um)

    @classmethod
    def from_field(cls, field: InterpolatedVelocityField) -> "_ValidPointProjectionLookup":
        nodes = np.asarray(field.nodes_um, dtype=float)
        elements = np.asarray(field.elements, dtype=np.int64)
        tri = nodes[elements]
        edge_midpoints = np.vstack(
            [
                0.5 * (tri[:, 0, :] + tri[:, 1, :]),
                0.5 * (tri[:, 1, :] + tri[:, 2, :]),
                0.5 * (tri[:, 2, :] + tri[:, 0, :]),
            ]
        )
        centroids = tri.mean(axis=1)
        barycentric_samples = np.vstack(
            [
                0.6 * tri[:, 0, :] + 0.2 * tri[:, 1, :] + 0.2 * tri[:, 2, :],
                0.2 * tri[:, 0, :] + 0.6 * tri[:, 1, :] + 0.2 * tri[:, 2, :],
                0.2 * tri[:, 0, :] + 0.2 * tri[:, 1, :] + 0.6 * tri[:, 2, :],
            ]
        )
        candidates = np.vstack([nodes, edge_midpoints, centroids, barycentric_samples])
        candidates = np.unique(np.round(candidates, decimals=9), axis=0)
        inside = _inside_fluid_domain(candidates, field.mesh.geometry)
        interpolation_valid = np.zeros(len(candidates), dtype=bool)
        if np.any(inside):
            interpolation_valid[np.flatnonzero(inside)] = _finite_interpolation_support(field, candidates[inside])
        return cls(candidates[inside & interpolation_valid])

    def nearest_valid_points(self, points_um: np.ndarray) -> np.ndarray:
        _, indices = self.tree.query(np.asarray(points_um, dtype=float), k=1)
        return self.valid_points_um[np.asarray(indices, dtype=np.int64)]


def _finite_interpolation_support(
    field: InterpolatedVelocityField,
    candidates_um: np.ndarray,
    chunk_size: int = 5000,
) -> np.ndarray:
    basis, _, _ = velocity_basis_components(field)
    coefficients = np.zeros(basis.N, dtype=float)
    valid = np.zeros(len(candidates_um), dtype=bool)
    for start in range(0, len(candidates_um), int(chunk_size)):
        stop = min(start + int(chunk_size), len(candidates_um))
        evaluated = _evaluate_basis_interpolator(basis, coefficients, candidates_um[start:stop].T * UM_TO_M)
        valid[start:stop] = np.isfinite(evaluated).all(axis=1)
    return valid


def velocity_basis(nodes_um: np.ndarray, elements: np.ndarray) -> Basis:
    nodes_m = np.ascontiguousarray(np.asarray(nodes_um, dtype=float).T * UM_TO_M)
    element_indices = np.ascontiguousarray(np.asarray(elements, dtype=np.int64).T)
    skmesh = MeshTri(nodes_m, element_indices)
    return Basis(skmesh, ElementVector(ElementTriP2()), intorder=4)


def velocity_basis_components(field: InterpolatedVelocityField) -> tuple[Basis, np.ndarray, np.ndarray]:
    key = (id(field.mesh), len(field.nodes_um), len(field.elements))
    cached = _VELOCITY_BASIS_CACHE.get(key)
    if cached is None:
        basis = velocity_basis(field.nodes_um, field.elements)
        component_x, component_y = basis.split_indices()
        cached = (basis, component_x, component_y)
        _VELOCITY_BASIS_CACHE[key] = cached
    return cached


def paired_velocity_to_basis_coefficients(
    basis: Basis,
    velocity_dof_m_per_s: np.ndarray,
    *,
    component_x: np.ndarray | None = None,
    component_y: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(velocity_dof_m_per_s, dtype=float)
    if component_x is None or component_y is None:
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
