from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.csgraph import structural_rank
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, MeshTri, asm, condense
from skfem.helpers import ddot, div, grad

from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry
from src.physics.full_device_cfd.mesh import generate_full_device_mesh
from src.physics.full_device_cfd.solver import (
    UL_PER_HR_TO_M3_PER_S,
    UM_TO_M,
    _boundary_scalar_dof_mask,
    _pressure_gauge_dof,
    _velocity_dirichlet_values,
)


def _assembled_problem():
    geometry = build_full_device_cfd_geometry()
    mesh = generate_full_device_mesh(geometry, target_size_um=24.0, boundary_size_um=12.0)
    skmesh = MeshTri(mesh.nodes_um.T * UM_TO_M, mesh.elements.T)
    basis_u = Basis(skmesh, ElementVector(ElementTriP2()), intorder=4)
    basis_p = Basis(skmesh, ElementTriP1(), intorder=4)

    @BilinearForm
    def viscous(u, v, w):
        return 0.001 * ddot(grad(u), grad(v))

    @BilinearForm
    def pressure_divergence(u, q, w):
        return -q * div(u)

    stiffness = asm(viscous, basis_u)
    divergence = asm(pressure_divergence, basis_u, basis_p)
    system = bmat([[stiffness, divergence.T], [divergence, None]], format="csr")
    inlet_flux = 1960.0 * UL_PER_HR_TO_M3_PER_S / (geometry.channel_height_um * UM_TO_M)
    mean_velocity = inlet_flux / (geometry.channel_width_um * UM_TO_M)
    velocity_dofs, velocity_values = _velocity_dirichlet_values(basis_u, mesh, mean_velocity)
    return geometry, mesh, basis_u, basis_p, divergence, system, velocity_dofs, velocity_values


def test_full_device_velocity_dirichlet_dofs_are_only_labeled_boundary_dofs() -> None:
    _geometry, mesh, basis_u, _basis_p, _divergence, _system, velocity_dofs, _velocity_values = _assembled_problem()
    xidx, yidx = basis_u.split_indices()
    constrained_scalar = _vector_to_scalar_indices(velocity_dofs, xidx, yidx)
    boundary = (
        _boundary_scalar_dof_mask(basis_u, mesh, "inlet")
        | _boundary_scalar_dof_mask(basis_u, mesh, "outlet")
        | _boundary_scalar_dof_mask(basis_u, mesh, "wall")
    )

    assert np.all(boundary[constrained_scalar])


def test_full_device_no_free_pressure_row_loses_divergence_coupling() -> None:
    _geometry, _mesh, basis_u, _basis_p, divergence, _system, velocity_dofs, _velocity_values = _assembled_problem()
    constrained = np.zeros(basis_u.N, dtype=bool)
    constrained[velocity_dofs] = True
    b_free = divergence[:, np.flatnonzero(~constrained)].tocsr()

    assert np.count_nonzero(np.diff(b_free.indptr) == 0) == 0
    assert structural_rank(b_free) == b_free.shape[0]


def test_full_device_unstabilized_condensed_system_is_structurally_full_rank_after_one_gauge() -> None:
    geometry, _mesh, basis_u, basis_p, _divergence, system, velocity_dofs, velocity_values = _assembled_problem()
    values = np.zeros(system.shape[0])
    values[velocity_dofs] = velocity_values
    pressure_dof = np.asarray([_pressure_gauge_dof(basis_p, geometry) + basis_u.N], dtype=np.int64)
    matrix, _vector, _expanded, _kept = condense(system, np.zeros(system.shape[0]), x=values, D=np.unique(np.concatenate([velocity_dofs, pressure_dof])))

    assert structural_rank(matrix) == matrix.shape[0]


def test_full_device_solver_has_no_pressure_mass_stabilization() -> None:
    source = Path("src/physics/full_device_cfd/solver.py").read_text(encoding="utf-8")

    assert "pressure-mass" not in source.lower()
    assert "pmass" not in source
    assert "-eps" not in source
    assert "stabilization" not in source.lower()


def _vector_to_scalar_indices(vector_dofs: np.ndarray, xidx: np.ndarray, yidx: np.ndarray) -> np.ndarray:
    x_lookup = {int(dof): i for i, dof in enumerate(xidx)}
    y_lookup = {int(dof): i for i, dof in enumerate(yidx)}
    scalar = set()
    for dof in vector_dofs:
        if int(dof) in x_lookup:
            scalar.add(x_lookup[int(dof)])
        if int(dof) in y_lookup:
            scalar.add(y_lookup[int(dof)])
    return np.asarray(sorted(scalar), dtype=np.int64)
