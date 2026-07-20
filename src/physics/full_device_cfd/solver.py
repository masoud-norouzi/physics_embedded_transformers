from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.csgraph import structural_rank
from scipy.sparse.linalg import MatrixRankWarning
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, MeshTri, asm, condense, solve
from skfem.helpers import ddot, div, grad

from .domain import (
    FullDeviceCFDGeometry,
    cross_section_points,
    inside_full_device_domain,
    project_to_centerline,
    resistance_weights,
)
from .mesh import FullDeviceMesh, evaluate_full_device_mesh

UM_TO_M = 1.0e-6
UL_PER_HR_TO_M3_PER_S = 1.0e-9 / 3600.0


@dataclass(frozen=True)
class FullDeviceCFDSolution:
    case_id: str
    target_left_fraction: float
    alpha_left_pa_s_per_m2: float
    alpha_right_pa_s_per_m2: float
    mesh: FullDeviceMesh
    viscosity_pa_s: float
    total_flow_rate_ul_per_hr: float
    inlet_flux_m2_per_s: float
    velocity_node_m_per_s: np.ndarray
    pressure_node_pa: np.ndarray
    velocity_dof_coordinates_um: np.ndarray
    velocity_dof_m_per_s: np.ndarray
    pressure_dof_coordinates_um: np.ndarray
    pressure_dof_pa: np.ndarray
    fluxes_m2_per_s: dict[str, float]
    actual_left_fraction: float
    solver_backend: str
    solve_runtime_s: float
    linear_system_diagnostics: dict[str, Any]


def solve_full_device_stokes(
    mesh: FullDeviceMesh,
    *,
    target_left_fraction: float,
    alpha_left_pa_s_per_m2: float,
    alpha_right_pa_s_per_m2: float,
    total_flow_rate_ul_per_hr: float = 1960.0,
    viscosity_pa_s: float = 0.001,
    case_id: str | None = None,
) -> FullDeviceCFDSolution:
    """Solve steady incompressible Stokes-Brinkman flow on the full device."""
    start = time.perf_counter()
    if alpha_left_pa_s_per_m2 < 0 or alpha_right_pa_s_per_m2 < 0:
        raise ValueError("Resistance alpha values must be nonnegative")
    skmesh = MeshTri(mesh.nodes_um.T * UM_TO_M, mesh.elements.T)
    basis_u = Basis(skmesh, ElementVector(ElementTriP2()), intorder=4)
    basis_p = Basis(skmesh, ElementTriP1(), intorder=4)
    geometry = mesh.geometry
    inlet_flux = total_flow_rate_ul_per_hr * UL_PER_HR_TO_M3_PER_S / (geometry.channel_height_um * UM_TO_M)
    mean_velocity = inlet_flux / (geometry.channel_width_um * UM_TO_M)

    @BilinearForm
    def viscous(u, v, w):
        return viscosity_pa_s * ddot(grad(u), grad(v))

    @BilinearForm
    def brinkman(u, v, w):
        pts_um = np.column_stack([w.x[0].ravel(), w.x[1].ravel()]) / UM_TO_M
        weights = resistance_weights(pts_um, geometry)
        alpha = alpha_left_pa_s_per_m2 * weights["left"] + alpha_right_pa_s_per_m2 * weights["right"]
        return alpha.reshape(w.x[0].shape) * (u[0] * v[0] + u[1] * v[1])

    @BilinearForm
    def pressure_divergence(u, q, w):
        return -q * div(u)

    stiffness = asm(viscous, basis_u) + asm(brinkman, basis_u)
    divergence = asm(pressure_divergence, basis_u, basis_p)
    system = bmat([[stiffness, divergence.T], [divergence, None]], format="csr")
    rhs = np.zeros(system.shape[0])
    values = np.zeros(system.shape[0])
    velocity_dofs, velocity_values = _velocity_dirichlet_values(basis_u, mesh, mean_velocity)
    values[velocity_dofs] = velocity_values
    pressure_dofs_local = np.asarray([_pressure_gauge_dof(basis_p, geometry)], dtype=np.int64)
    pressure_dofs_local = np.unique(
        np.concatenate([pressure_dofs_local, _uncoupled_pressure_gauge_dofs(divergence, velocity_dofs, basis_u.N)])
    )
    pressure_dofs = pressure_dofs_local + basis_u.N
    constrained = np.unique(np.concatenate([velocity_dofs, pressure_dofs]))
    condensed = condense(system, rhs, x=values, D=constrained)
    matrix, vector, _expanded, kept = condensed
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        solution_vector = solve(*condensed)
    if not np.isfinite(solution_vector).all():
        raise RuntimeError("Full-device direct solve returned non-finite coefficients")
    velocity_coeff = solution_vector[: basis_u.N]
    pressure_coeff = solution_vector[basis_u.N :]
    velocity_nodes = _velocity_at_points(basis_u, velocity_coeff, mesh.nodes_um)
    pressure_nodes = _scalar_at_points(basis_p, pressure_coeff, mesh.nodes_um)
    fluxes = full_device_fluxes(mesh, basis_u, velocity_coeff)
    branch_total = fluxes["left_branch"] + fluxes["right_branch"]
    actual = float(fluxes["left_branch"] / branch_total) if abs(branch_total) > 0 else float("nan")
    return FullDeviceCFDSolution(
        case_id=case_id or f"left_fraction_{_token(target_left_fraction)}",
        target_left_fraction=float(target_left_fraction),
        alpha_left_pa_s_per_m2=float(alpha_left_pa_s_per_m2),
        alpha_right_pa_s_per_m2=float(alpha_right_pa_s_per_m2),
        mesh=mesh,
        viscosity_pa_s=float(viscosity_pa_s),
        total_flow_rate_ul_per_hr=float(total_flow_rate_ul_per_hr),
        inlet_flux_m2_per_s=float(inlet_flux),
        velocity_node_m_per_s=velocity_nodes,
        pressure_node_pa=pressure_nodes,
        velocity_dof_coordinates_um=_paired_vector_dof_coordinates(basis_u),
        velocity_dof_m_per_s=_paired_vector_dof_values(basis_u, velocity_coeff),
        pressure_dof_coordinates_um=basis_p.doflocs.T / UM_TO_M,
        pressure_dof_pa=pressure_coeff,
        fluxes_m2_per_s=fluxes,
        actual_left_fraction=actual,
        solver_backend="scikit-fem/direct",
        solve_runtime_s=float(time.perf_counter() - start),
        linear_system_diagnostics={
            "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "structural_rank": int(structural_rank(matrix)),
            "free_velocity_dofs": int(np.count_nonzero(np.asarray(kept) < basis_u.N)),
            "free_pressure_dofs": int(np.count_nonzero(np.asarray(kept) >= basis_u.N)),
            "velocity_dofs": int(basis_u.N),
            "pressure_dofs": int(basis_p.N),
            "constrained_pressure_dofs": int(len(pressure_dofs)),
            "uncoupled_pressure_gauge_dofs": int(len(_uncoupled_pressure_gauge_dofs(divergence, velocity_dofs, basis_u.N))),
            "alpha_units": "Pa s m^-2; effective distributed momentum-resistance coefficient in alpha*u",
            "pressure_boundary_condition": "single pressure gauge DOF; no pressure Dirichlet condition on the prescribed-velocity inlet; outlet uses the natural traction condition",
        },
    )


def full_device_fluxes(mesh: FullDeviceMesh, basis_u: Basis, velocity_coeff: np.ndarray) -> dict[str, float]:
    g = mesh.geometry
    return {
        "inlet": _boundary_facet_flux(mesh, basis_u, velocity_coeff, "inlet"),
        "outlet": _boundary_facet_flux(mesh, basis_u, velocity_coeff, "outlet"),
        "left_branch": _section_flux_at_s(mesh, basis_u, velocity_coeff, "left", g.centerlines["left"].length_um / 2.0),
        "right_branch": _section_flux_at_s(mesh, basis_u, velocity_coeff, "right", g.centerlines["right"].length_um / 2.0),
    }


def save_full_device_solution(solution: FullDeviceCFDSolution, output_dir: str | Path, calibration_history: list[dict[str, Any]], overwrite: bool = False) -> None:
    out = Path(output_dir)
    diag = out / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    diag.mkdir(parents=True, exist_ok=True)
    files = [out / "stokes_solution.npz", out / "metadata.json", out / "calibration_history.csv"]
    if not overwrite and any(p.exists() for p in files):
        raise FileExistsError(f"Pilot output exists: {out}")
    np.savez_compressed(
        out / "stokes_solution.npz",
        nodes_um=solution.mesh.nodes_um,
        elements=solution.mesh.elements,
        velocity_nodes_m_per_s=solution.velocity_node_m_per_s,
        pressure_nodes_pa=solution.pressure_node_pa,
        velocity_dof_coordinates_um=solution.velocity_dof_coordinates_um,
        velocity_dofs_m_per_s=solution.velocity_dof_m_per_s,
        pressure_dof_coordinates_um=solution.pressure_dof_coordinates_um,
        pressure_dofs_pa=solution.pressure_dof_pa,
        alpha_left_pa_s_per_m2=np.array([solution.alpha_left_pa_s_per_m2]),
        alpha_right_pa_s_per_m2=np.array([solution.alpha_right_pa_s_per_m2]),
    )
    metadata = {
        "case_id": solution.case_id,
        "split_convention": "f_left = Q_left / (Q_left + Q_right)",
        "target_left_fraction": solution.target_left_fraction,
        "achieved_left_fraction": solution.actual_left_fraction,
        "alpha_left_pa_s_per_m2": solution.alpha_left_pa_s_per_m2,
        "alpha_right_pa_s_per_m2": solution.alpha_right_pa_s_per_m2,
        "alpha_dimensional_convention": "Pa s m^-2 in -alpha(x)u; effective distributed momentum resistance, not a literal porous-medium property.",
        "fluxes_m2_per_s": solution.fluxes_m2_per_s,
        "mass_error_inlet_outlet_m2_per_s": solution.fluxes_m2_per_s["inlet"] + solution.fluxes_m2_per_s["outlet"],
        "mass_error_junction_m2_per_s": solution.fluxes_m2_per_s["left_branch"] + solution.fluxes_m2_per_s["right_branch"] - solution.fluxes_m2_per_s["outlet"],
        "solver_backend": solution.solver_backend,
        "solve_runtime_s": solution.solve_runtime_s,
        "linear_system_diagnostics": solution.linear_system_diagnostics,
        "mesh_quality": asdict(evaluate_full_device_mesh(solution.mesh)),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_history(out / "calibration_history.csv", calibration_history)
    _save_figures(solution, diag)


def _velocity_dirichlet_values(basis_u: Basis, mesh: FullDeviceMesh, mean_velocity: float) -> tuple[np.ndarray, np.ndarray]:
    xidx, yidx = basis_u.split_indices()
    coords = basis_u.doflocs[:, xidx].T / UM_TO_M
    g = mesh.geometry
    inlet = _boundary_scalar_dof_mask(basis_u, mesh, "inlet")
    outlet = _boundary_scalar_dof_mask(basis_u, mesh, "outlet")
    wall = _boundary_scalar_dof_mask(basis_u, mesh, "wall") & ~inlet & ~outlet
    constrained_scalar = inlet | wall
    boundary = inlet | outlet | wall
    off_boundary = constrained_scalar & ~boundary
    if np.any(off_boundary):
        bad = np.flatnonzero(off_boundary)[:10].tolist()
        raise RuntimeError(f"Velocity Dirichlet selection constrained non-boundary P2 DOFs: {bad}")
    values = np.zeros((len(coords), 2))
    tangent = g.centerlines["inlet"].tangents[0]
    proj = project_to_centerline(coords, g.centerlines["inlet"], g.channel_width_um)
    speed = 1.5 * mean_velocity * np.maximum(0.0, 1.0 - proj.eta**2)
    values[inlet] = speed[inlet, None] * tangent
    dofs = np.concatenate([xidx[constrained_scalar], yidx[constrained_scalar]])
    vals = np.concatenate([values[constrained_scalar, 0], values[constrained_scalar, 1]])
    return dofs.astype(np.int64), vals


def _near_wall(points: np.ndarray, geometry: FullDeviceCFDGeometry, tol_um: float) -> np.ndarray:
    wall = np.zeros(len(points), dtype=bool)
    for line in geometry.centerlines.values():
        proj = project_to_centerline(points, line, geometry.channel_width_um)
        wall |= proj.inside & (np.abs(np.abs(proj.eta) - 1.0) <= tol_um / geometry.half_width_um)
    return wall


def _boundary_scalar_dof_mask(basis_u: Basis, mesh: FullDeviceMesh, label: str) -> np.ndarray:
    xidx, _ = basis_u.split_indices()
    scalar_basis = Basis(basis_u.mesh, ElementTriP2(), intorder=4)
    mask = np.zeros(len(xidx), dtype=bool)
    facets = _skfem_facet_indices(basis_u.mesh, mesh.boundary_facets.get(label, np.empty((0, 2), dtype=np.int64)))
    if len(facets) == 0:
        return mask
    dofs = scalar_basis.get_dofs(facets=facets).all()
    mask[np.asarray(dofs, dtype=np.int64)] = True
    return mask


def _skfem_facet_indices(skmesh: MeshTri, edges: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return np.array([], dtype=np.int64)
    lookup = {tuple(sorted(map(int, skmesh.facets[:, i]))): i for i in range(skmesh.facets.shape[1])}
    ids = []
    missing = []
    for edge in np.asarray(edges, dtype=np.int64):
        key = tuple(sorted(map(int, edge)))
        if key in lookup:
            ids.append(lookup[key])
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(f"Boundary facets are missing from scikit-fem mesh: {missing[:5]}")
    return np.asarray(ids, dtype=np.int64)


def _near_wall_facets(points: np.ndarray, mesh: FullDeviceMesh, tol_um: float) -> np.ndarray:
    wall_edges = mesh.boundary_facets.get("wall", np.empty((0, 2), dtype=np.int64))
    if len(wall_edges) == 0:
        return np.zeros(len(points), dtype=bool)
    near = np.zeros(len(points), dtype=bool)
    nodes = mesh.nodes_um
    for edge in wall_edges:
        start, end = nodes[edge[0]], nodes[edge[1]]
        vec = end - start
        length2 = float(np.dot(vec, vec))
        if length2 <= 0:
            continue
        rel = points - start
        a = np.clip((rel @ vec) / length2, 0.0, 1.0)
        closest = start + a[:, None] * vec
        near |= np.linalg.norm(points - closest, axis=1) <= tol_um
    return near


def _near_cut(points: np.ndarray, center: np.ndarray, tangent: np.ndarray, width: float, tol_um: float) -> np.ndarray:
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    rel = points - center
    return (np.abs(rel @ tangent) <= tol_um) & (np.abs(rel @ normal) <= width / 2.0 + tol_um)


def _pressure_gauge_dof(basis_p: Basis, geometry: FullDeviceCFDGeometry) -> int:
    coords = basis_p.doflocs.T / UM_TO_M
    return int(np.argmin(np.linalg.norm(coords - geometry.outlet_cut_center_um, axis=1)))


def _uncoupled_pressure_gauge_dofs(divergence, velocity_dofs: np.ndarray, velocity_dof_count: int) -> np.ndarray:
    constrained_u = np.zeros(velocity_dof_count, dtype=bool)
    constrained_u[np.asarray(velocity_dofs, dtype=np.int64)] = True
    free_u = np.flatnonzero(~constrained_u)
    coupling = divergence[:, free_u].tocsr()
    row_nnz = np.diff(coupling.indptr)
    return np.flatnonzero(row_nnz == 0).astype(np.int64)


def _section_flux_at_s(mesh: FullDeviceMesh, basis_u: Basis, coeff: np.ndarray, branch: str, s_um: float) -> float:
    line = mesh.geometry.centerlines[branch]
    center = np.array([np.interp(s_um, line.s_um, line.points_um[:, 0]), np.interp(s_um, line.s_um, line.points_um[:, 1])])
    tangent = np.array([np.interp(s_um, line.s_um, line.tangents[:, 0]), np.interp(s_um, line.s_um, line.tangents[:, 1])])
    tangent /= np.linalg.norm(tangent)
    return _section_flux(mesh, basis_u, coeff, center, tangent, outward=False)


def _boundary_facet_flux(mesh: FullDeviceMesh, basis_u: Basis, coeff: np.ndarray, label: str) -> float:
    flux = 0.0
    for edge in mesh.boundary_facets[label]:
        start, end = mesh.nodes_um[edge[0]], mesh.nodes_um[edge[1]]
        midpoint = 0.5 * (start + end)
        tangent = end - start
        length_um = float(np.linalg.norm(tangent))
        if length_um <= 0.0:
            continue
        tangent /= length_um
        normal = np.array([-tangent[1], tangent[0]])
        if inside_full_device_domain((midpoint + normal * 0.5)[None, :], mesh.geometry)[0]:
            normal = -normal
        points = np.vstack([start, midpoint, end])
        velocity = _velocity_at_points(basis_u, coeff, points)
        normal_velocity = velocity @ normal
        if not np.isfinite(normal_velocity).all():
            continue
        flux += float((normal_velocity[0] + 4.0 * normal_velocity[1] + normal_velocity[2]) / 6.0 * length_um * UM_TO_M)
    return flux


def _section_flux(mesh: FullDeviceMesh, basis_u: Basis, coeff: np.ndarray, center: np.ndarray, tangent: np.ndarray, outward: bool) -> float:
    points = cross_section_points(center, tangent, mesh.geometry.channel_width_um, n=101)
    velocity = _velocity_at_points(basis_u, coeff, points)
    sign = -1.0 if outward else 1.0
    u_t = velocity @ (sign * tangent / np.linalg.norm(tangent))
    finite = np.isfinite(u_t)
    eta = np.linspace(-1.0, 1.0, len(points))
    return float(np.trapezoid(u_t[finite], eta[finite]) * mesh.geometry.channel_width_um / 2.0 * UM_TO_M)


def _velocity_at_points(basis_u: Basis, coeff: np.ndarray, points_um: np.ndarray) -> np.ndarray:
    interp = basis_u.interpolator(coeff)
    points_m = np.asarray(points_um).T * UM_TO_M
    out = np.full((points_m.shape[1], 2), np.nan)
    for i in range(points_m.shape[1]):
        try:
            val = np.asarray(interp(points_m[:, i : i + 1]), dtype=float)
            out[i] = val.reshape(2, -1)[:, 0]
        except ValueError:
            continue
    return out


def _scalar_at_points(basis: Basis, coeff: np.ndarray, points_um: np.ndarray) -> np.ndarray:
    interp = basis.interpolator(coeff)
    points_m = np.asarray(points_um).T * UM_TO_M
    out = np.full(points_m.shape[1], np.nan)
    for i in range(points_m.shape[1]):
        try:
            out[i] = float(np.asarray(interp(points_m[:, i : i + 1])).ravel()[0])
        except ValueError:
            continue
    return out


def _paired_vector_dof_coordinates(basis_u: Basis) -> np.ndarray:
    xidx, _ = basis_u.split_indices()
    return basis_u.doflocs[:, xidx].T / UM_TO_M


def _paired_vector_dof_values(basis_u: Basis, coeff: np.ndarray) -> np.ndarray:
    xidx, yidx = basis_u.split_indices()
    return np.column_stack([coeff[xidx], coeff[yidx]])


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    import csv

    if not history:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _save_figures(solution: FullDeviceCFDSolution, out: Path) -> None:
    tri = mtri.Triangulation(solution.mesh.nodes_um[:, 0], solution.mesh.nodes_um[:, 1], solution.mesh.elements)
    speed = np.linalg.norm(solution.velocity_node_m_per_s, axis=1)
    _tripcolor(tri, speed, out / "speed_magnitude.png", "speed (m/s)", f"Speed, {solution.case_id}", cmap="magma")
    _tripcolor(tri, solution.pressure_node_pa, out / "pressure.png", "pressure (Pa)", f"Pressure, {solution.case_id}", cmap="coolwarm")
    weights = resistance_weights(solution.mesh.nodes_um, solution.mesh.geometry)
    alpha = solution.alpha_left_pa_s_per_m2 * weights["left"] + solution.alpha_right_pa_s_per_m2 * weights["right"]
    _tripcolor(tri, alpha, out / "alpha_field.png", "alpha (Pa s m^-2)", f"Effective resistance alpha, {solution.case_id}", cmap="viridis")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.triplot(tri, linewidth=0.12, color="#d1d5db")
    idx = np.arange(len(solution.mesh.nodes_um))[:: max(1, len(solution.mesh.nodes_um) // 220)]
    ax.quiver(solution.mesh.nodes_um[idx, 0], solution.mesh.nodes_um[idx, 1], solution.velocity_node_m_per_s[idx, 0], solution.velocity_node_m_per_s[idx, 1], scale_units="xy", angles="xy", scale=0.002)
    _format(ax, f"Velocity quiver, {solution.case_id}")
    fig.tight_layout()
    fig.savefig(out / "velocity_quiver.png", dpi=180)
    plt.close(fig)


def _tripcolor(tri, values, path: Path, label: str, title: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap)
    fig.colorbar(im, ax=ax, label=label)
    ax.triplot(tri, linewidth=0.08, color="white", alpha=0.2)
    _format(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format(ax, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title(title)


def _token(value: float) -> str:
    return f"0p{int(round(value * 100)):02d}"
