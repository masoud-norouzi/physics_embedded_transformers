from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import bmat
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, MeshTri, asm
from skfem.helpers import ddot, div, grad

from .domain import JunctionGeometry, build_junction_geometry, load_junction_cfd_config
from .mesh import TriangularMesh, generate_mesh
from .solver import (
    UM_TO_M,
    _load_flow_inputs,
    _paired_vector_dof_coordinates,
    _paired_vector_dof_values,
    _pressure_reference_dof,
    _scale_open_profiles_to_flux,
    _split_config,
    _solution_config,
    _solve_condensed_system,
    _unit,
    _velocity_at_mesh_nodes,
    _velocity_dirichlet_values,
    _volume_flow_to_2d_flux,
)


@dataclass(frozen=True)
class CrossSectionDiagnostic:
    name: str
    arc_length_um: float
    center_um: list[float]
    tangent: list[float]
    normal: list[float]
    mean_axial_velocity_m_per_s: float
    maximum_axial_velocity_m_per_s: float
    max_to_mean_ratio: float
    rmse_m_per_s: float
    max_abs_cross_velocity_m_per_s: float
    cross_velocity_relative_to_mean: float
    left_wall_velocity_m_per_s: float
    right_wall_velocity_m_per_s: float
    integrated_flux_m2_per_s: float
    expected_flux_m2_per_s: float
    downstream_sign_ok: bool
    normal_velocity_ok: bool
    symmetry_error_m_per_s: float
    symmetry_ok: bool
    flux_relative_error: float
    flux_ok: bool


def run_inlet_profile_diagnostic(
    config: str | Path | dict[str, Any],
    output_root: str | Path | None = None,
    sample_count: int = 61,
) -> list[CrossSectionDiagnostic]:
    cfg = load_junction_cfd_config(config)
    solution_cfg = _solution_config(cfg)
    geometry = build_junction_geometry(cfg)
    mesh = generate_mesh(geometry)
    basis_u, velocity_solution, inlet_flux, mean_velocity = _solve_raw_velocity_coefficients(cfg, mesh)
    if output_root is None:
        output_root = Path(solution_cfg["output_root"]) / "inlet_profile_diagnostics"
    output_root = Path(output_root)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    (output_root / "samples").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    sections = _inlet_cross_sections(geometry)
    diagnostics = []
    for name, arc_length_um in sections:
        sample = _sample_cross_section(
            basis_u,
            velocity_solution,
            geometry,
            name,
            arc_length_um,
            sample_count,
            mean_velocity,
            inlet_flux,
        )
        diagnostics.append(sample["diagnostic"])
        _write_sample_csv(output_root / "samples" / f"{name}.csv", sample["rows"])
        _save_profile_plot(output_root / "figures" / f"{name}.png", sample["rows"], sample["diagnostic"])

    report = [asdict(item) for item in diagnostics]
    (output_root / "reports" / "inlet_profile_diagnostics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_root / "reports" / "inlet_profile_diagnostics.md").write_text(
        _markdown_report(diagnostics),
        encoding="utf-8",
    )
    return diagnostics


def _solve_raw_velocity_coefficients(
    cfg: dict[str, Any],
    mesh: TriangularMesh,
) -> tuple[Basis, np.ndarray, float, float]:
    solution_cfg = _solution_config(cfg)
    split_cfg = _split_config(solution_cfg)
    viscosity = float(solution_cfg["viscosity_pa_s"])
    flow = _load_flow_inputs(cfg)
    inlet_flux = _volume_flow_to_2d_flux(flow["total_flow_rate_ul_per_hr"], flow["channel_height_um"])
    mean_velocity = inlet_flux / (mesh.geometry.channel_width_um * UM_TO_M)

    skmesh = MeshTri(mesh.nodes_um.T * UM_TO_M, mesh.elements.T)
    basis_u = Basis(skmesh, ElementVector(ElementTriP2()), intorder=4)
    basis_p = Basis(skmesh, ElementTriP1(), intorder=4)

    @BilinearForm
    def viscous(u, v, w):
        return viscosity * ddot(grad(u), grad(v))

    @BilinearForm
    def pressure_divergence(u, q, w):
        return -q * div(u)

    stiffness = asm(viscous, basis_u)
    divergence = asm(pressure_divergence, basis_u, basis_p)
    system = bmat([[stiffness, divergence.T], [divergence, None]], format="csr")
    rhs = np.zeros(system.shape[0], dtype=float)
    values = np.zeros(system.shape[0], dtype=float)
    velocity_dofs, velocity_values = _velocity_dirichlet_values(
        basis_u,
        mesh,
        mean_velocity,
        inlet_flux,
        split_cfg["left_fraction"],
        inlet_profile=str(solution_cfg["inlet_profile"]),
    )
    velocity_values = _scale_open_profiles_to_flux(
        basis_u,
        mesh,
        velocity_dofs,
        velocity_values,
        inlet_flux,
        split_cfg["left_fraction"],
    )
    values[velocity_dofs] = velocity_values
    pressure_dofs = np.asarray([_pressure_reference_dof(basis_p, mesh.geometry) + basis_u.N], dtype=np.int64)
    values[pressure_dofs] = float(solution_cfg["outlet_pressure_pa"])
    constrained = np.unique(np.concatenate([velocity_dofs, pressure_dofs]))
    solution_vector, _ = _solve_condensed_system(system, rhs, values, constrained)
    return basis_u, solution_vector[: basis_u.N], inlet_flux, mean_velocity


def _inlet_cross_sections(geometry: JunctionGeometry) -> list[tuple[str, float]]:
    length = float(geometry.branch_arc_lengths_um["inlet"][-1])
    return [
        ("near_inlet", min(geometry.channel_width_um * 0.25, length * 0.1)),
        ("mid_inlet", length * 0.5),
        ("one_width_upstream_junction", max(0.0, length - geometry.channel_width_um)),
    ]


def _sample_cross_section(
    basis_u: Basis,
    velocity_solution: np.ndarray,
    geometry: JunctionGeometry,
    name: str,
    arc_length_um: float,
    sample_count: int,
    mean_velocity: float,
    inlet_flux: float,
) -> dict[str, Any]:
    center = _interpolate_polyline(geometry.branch_centerlines_um["inlet"], geometry.branch_arc_lengths_um["inlet"], arc_length_um)
    tangent = _interpolate_vectors(geometry.branch_tangents["inlet"], geometry.branch_arc_lengths_um["inlet"], arc_length_um)
    tangent = _unit(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    width = geometry.channel_width_um
    s_um = np.linspace(-width / 2.0, width / 2.0, sample_count)
    points_um = center + s_um[:, None] * normal
    probes = basis_u.probes(points_um.T * UM_TO_M)
    raw = probes @ velocity_solution
    ux = raw[:sample_count]
    uy = raw[sample_count:]
    velocity = np.column_stack([ux, uy])
    u_parallel = velocity @ tangent
    u_normal = velocity @ normal
    analytical = 1.5 * mean_velocity * (1.0 - (2.0 * s_um / width) ** 2)
    flux = float(np.trapezoid(u_parallel, s_um * UM_TO_M))
    mean_axial = flux / (width * UM_TO_M)
    max_axial = float(np.max(u_parallel))
    rmse = float(np.sqrt(np.mean((u_parallel - analytical) ** 2)))
    max_cross = float(np.max(np.abs(u_normal)))
    symmetry = float(np.max(np.abs(u_parallel - u_parallel[::-1])))
    wall_left = float(u_parallel[0])
    wall_right = float(u_parallel[-1])
    rows = [
        {
            "cross_section": name,
            "x_um": float(point[0]),
            "y_um": float(point[1]),
            "s_um": float(s),
            "u_x_m_per_s": float(vec[0]),
            "u_y_m_per_s": float(vec[1]),
            "u_parallel_m_per_s": float(up),
            "u_normal_m_per_s": float(un),
            "u_analytical_m_per_s": float(ua),
        }
        for point, s, vec, up, un, ua in zip(points_um, s_um, velocity, u_parallel, u_normal, analytical)
    ]
    diagnostic = CrossSectionDiagnostic(
        name=name,
        arc_length_um=float(arc_length_um),
        center_um=[float(center[0]), float(center[1])],
        tangent=[float(tangent[0]), float(tangent[1])],
        normal=[float(normal[0]), float(normal[1])],
        mean_axial_velocity_m_per_s=float(mean_axial),
        maximum_axial_velocity_m_per_s=max_axial,
        max_to_mean_ratio=float(max_axial / mean_axial) if abs(mean_axial) > 0 else float("nan"),
        rmse_m_per_s=rmse,
        max_abs_cross_velocity_m_per_s=max_cross,
        cross_velocity_relative_to_mean=float(max_cross / abs(mean_axial)) if abs(mean_axial) > 0 else float("nan"),
        left_wall_velocity_m_per_s=wall_left,
        right_wall_velocity_m_per_s=wall_right,
        integrated_flux_m2_per_s=flux,
        expected_flux_m2_per_s=inlet_flux,
        downstream_sign_ok=bool(mean_axial > 0),
        normal_velocity_ok=bool(max_cross / abs(mean_axial) < 0.05) if abs(mean_axial) > 0 else False,
        symmetry_error_m_per_s=symmetry,
        symmetry_ok=bool(symmetry / abs(max_axial) < 0.05) if abs(max_axial) > 0 else False,
        flux_relative_error=float((flux - inlet_flux) / inlet_flux),
        flux_ok=bool(abs((flux - inlet_flux) / inlet_flux) < 0.05),
    )
    return {"rows": rows, "diagnostic": diagnostic}


def _interpolate_polyline(points: np.ndarray, arc_lengths: np.ndarray, arc_length_um: float) -> np.ndarray:
    x = np.interp(arc_length_um, arc_lengths, points[:, 0])
    y = np.interp(arc_length_um, arc_lengths, points[:, 1])
    return np.array([x, y], dtype=float)


def _interpolate_vectors(vectors: np.ndarray, arc_lengths: np.ndarray, arc_length_um: float) -> np.ndarray:
    x = np.interp(arc_length_um, arc_lengths, vectors[:, 0])
    y = np.interp(arc_length_um, arc_lengths, vectors[:, 1])
    return np.array([x, y], dtype=float)


def _write_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_profile_plot(path: Path, rows: list[dict[str, Any]], diagnostic: CrossSectionDiagnostic) -> None:
    s = np.array([row["s_um"] for row in rows])
    u_parallel = np.array([row["u_parallel_m_per_s"] for row in rows]) / UM_TO_M
    u_normal = np.array([row["u_normal_m_per_s"] for row in rows]) / UM_TO_M
    analytical = np.array([row["u_analytical_m_per_s"] for row in rows]) / UM_TO_M
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s, u_parallel, label="raw FEM u_parallel", linewidth=2)
    ax.plot(s, analytical, "--", label="analytical planar Poiseuille", linewidth=2)
    ax.plot(s, u_normal, label="raw FEM u_normal", linewidth=1.5)
    ax.axvline(-max(abs(s)), color="black", linestyle=":", label="walls")
    ax.axvline(max(abs(s)), color="black", linestyle=":")
    ax.set_xlabel("transverse coordinate s (um)")
    ax.set_ylabel("velocity (um/s)")
    ax.set_title(f"Inlet profile diagnostic: {diagnostic.name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_report(diagnostics: list[CrossSectionDiagnostic]) -> str:
    lines = [
        "# Raw FEM Inlet Poiseuille Diagnostic",
        "",
        "Velocity is evaluated directly from the finite-element basis on inlet cross-sections.",
        "The regular-grid streamline interpolation is not used for this diagnostic.",
        "FEM coordinates, velocity components, and plots all use the mesh/image convention with y increasing downward.",
        "",
    ]
    for item in diagnostics:
        lines.extend(
            [
                f"## {item.name}",
                "",
                f"- Arc length from inlet: {item.arc_length_um:.3f} um",
                f"- Mean axial velocity: {item.mean_axial_velocity_m_per_s:.6e} m/s",
                f"- Maximum axial velocity: {item.maximum_axial_velocity_m_per_s:.6e} m/s",
                f"- Max/mean ratio: {item.max_to_mean_ratio:.6f}",
                f"- RMSE vs analytical: {item.rmse_m_per_s:.6e} m/s",
                f"- Max absolute cross-channel velocity: {item.max_abs_cross_velocity_m_per_s:.6e} m/s",
                f"- Cross-channel velocity / mean axial velocity: {item.cross_velocity_relative_to_mean:.6e}",
                f"- Left/right wall velocity: {item.left_wall_velocity_m_per_s:.6e}, {item.right_wall_velocity_m_per_s:.6e} m/s",
                f"- Integrated flux: {item.integrated_flux_m2_per_s:.6e} m^2/s",
                f"- Expected inlet flux: {item.expected_flux_m2_per_s:.6e} m^2/s",
                f"- Flux relative error: {item.flux_relative_error:.6e}",
                f"- Downstream sign ok: {item.downstream_sign_ok}",
                f"- Normal velocity approximately zero: {item.normal_velocity_ok}",
                f"- Symmetric about centerline: {item.symmetry_ok}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw FEM inlet velocity profiles against planar Poiseuille flow.")
    parser.add_argument("--config", type=Path, default=Path("configs/physics/junction_cfd.yml"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=61)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = run_inlet_profile_diagnostic(args.config, args.output_root, sample_count=args.sample_count)
    print("Raw FEM inlet profile diagnostic completed")
    for item in diagnostics:
        print(
            f"  {item.name}: mean={item.mean_axial_velocity_m_per_s:.6e} m/s, "
            f"max/mean={item.max_to_mean_ratio:.4f}, "
            f"rmse={item.rmse_m_per_s:.6e} m/s, "
            f"cross/mean={item.cross_velocity_relative_to_mean:.3e}, "
            f"flux_error={item.flux_relative_error:.3e}"
        )


if __name__ == "__main__":
    main()
