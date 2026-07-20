from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.physics.cfd.mesh import TriangularMesh


@dataclass(frozen=True)
class VelocityFieldCase:
    """One frozen CFD Version 1 split case loaded from disk."""

    case_id: str
    path: Path
    left_fraction: float
    right_fraction: float
    velocity_dof_m_per_s: np.ndarray
    velocity_dof_coordinates_um: np.ndarray
    velocity_node_m_per_s: np.ndarray
    nodes_um: np.ndarray
    elements: np.ndarray
    mesh: TriangularMesh
    metadata: dict[str, Any]
    flux_report: dict[str, Any]
    cfd_version: str
    mesh_version: str
    units: dict[str, str]


@dataclass(frozen=True)
class SampledVelocityField:
    """Vectorized samples of an interpolated velocity field."""

    points_um: np.ndarray
    u_x_m_per_s: np.ndarray
    u_y_m_per_s: np.ndarray
    speed_m_per_s: np.ndarray
    direction_x: np.ndarray
    direction_y: np.ndarray
    inside_domain: np.ndarray
    units: dict[str, str]
    coordinate_frame: str


@dataclass(frozen=True)
class InterpolatedVelocityField:
    """Continuous-in-split velocity field on the frozen CFD production mesh."""

    requested_left_fraction: float
    requested_right_fraction: float
    lower_library_fraction: float
    upper_library_fraction: float
    interpolation_weight: float
    velocity_dof_m_per_s: np.ndarray
    velocity_dof_coordinates_um: np.ndarray
    velocity_node_m_per_s: np.ndarray
    nodes_um: np.ndarray
    elements: np.ndarray
    mesh: TriangularMesh
    velocity_basis_metadata: dict[str, Any]
    units: dict[str, str]
    cfd_version: str
    mesh_version: str
    exact_match: bool
    lower_case_id: str
    upper_case_id: str

    def sample_cfd(self, points_cfd_um: np.ndarray) -> SampledVelocityField:
        """Sample at points in the frozen CFD native frame; returns CFD-frame vectors."""
        from .field_sampler import sample_velocity_field_cfd

        return sample_velocity_field_cfd(self, points_cfd_um)

    def sample_device(self, points_device_um: np.ndarray, convention) -> SampledVelocityField:
        """Sample at device-Cartesian points; returns device-Cartesian vectors."""
        from .field_sampler import sample_velocity_field_device

        return sample_velocity_field_device(self, points_device_um, convention)

    def sample(self, points_um: np.ndarray) -> SampledVelocityField:
        """Backward-compatible alias for sample_cfd(points_cfd_um)."""
        return self.sample_cfd(points_um)
