from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.csgraph import structural_rank
from scipy.sparse import bmat
from scipy.sparse.linalg import MatrixRankWarning, svds
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, MeshTri, asm, condense, solve
from skfem.helpers import ddot, div, grad

from src.config import load_experiment_config

from .domain import JunctionGeometry, classify_boundary_points, inside_junction_domain, load_junction_cfd_config
from .mesh import TriangularMesh, evaluate_mesh, generate_mesh


UM_TO_M = 1e-6
UL_PER_HR_TO_M3_PER_S = 1e-9 / 3600.0


@dataclass(frozen=True)
class StokesSolution:
    """Single-phase steady Stokes solution for one junction-flow case."""

    case_id: str
    mesh: TriangularMesh
    viscosity_pa_s: float
    total_flow_rate_ul_per_hr: float
    inlet_flux_m2_per_s: float
    requested_left_fraction: float
    requested_right_fraction: float
    channel_height_um: float
    inlet_mean_velocity_m_per_s: float
    velocity_node_m_per_s: np.ndarray
    pressure_node_pa: np.ndarray
    velocity_dof_coordinates_um: np.ndarray
    velocity_dof_m_per_s: np.ndarray
    pressure_dof_coordinates_um: np.ndarray
    pressure_dof_pa: np.ndarray
    fluxes_m2_per_s: dict[str, float]
    split_fraction: dict[str, float]
    outlet_pressure_pa: float
    solver_backend: str
    boundary_condition_summary: dict[str, Any]
    linear_system_diagnostics: dict[str, Any]
    boundary_dof_diagnostics: dict[str, Any]
    divergence_diagnostics: dict[str, float]
    assumptions: list[str]
    postprocessing_config: dict[str, Any]


@dataclass(frozen=True)
class StokesReport:
    case_id: str
    solver_backend: str
    element_pair: str
    coordinate_units_geometry: str
    coordinate_units_solve: str
    viscosity_pa_s: float
    total_flow_rate_ul_per_hr: float
    channel_height_um: float
    inlet_flux_m2_per_s: float
    requested_left_fraction: float
    requested_right_fraction: float
    inlet_mean_velocity_m_per_s: float
    maximum_velocity_m_per_s: float
    mean_velocity_m_per_s: float
    minimum_pressure_pa: float
    maximum_pressure_pa: float
    inlet_flux_signed_m2_per_s: float
    left_outlet_flux_m2_per_s: float
    right_outlet_flux_m2_per_s: float
    net_flux_residual_m2_per_s: float
    left_split_fraction: float
    right_split_fraction: float
    mesh_nodes: int
    mesh_elements: int
    outlet_pressure_pa: float


@dataclass(frozen=True)
class StreamlineDiagnostics:
    """Post-processing summary for streamlines sampled from the solved velocity field."""

    interpolation_grid_resolution_um: float
    dense_streamline_density: float
    inlet_seed_count: int
    seed_offset_um: float
    trace_step_um: float
    max_trace_steps: int
    terminated_left: int
    terminated_right: int
    terminated_other: int
    termination_reason_counts: dict[str, int]
    note: str


def solve_junction_stokes(
    config: str | Path | dict[str, Any],
    mesh: TriangularMesh | None = None,
) -> StokesSolution:
    """Solve the first single-phase Stokes case on the existing junction mesh."""
    cfg = load_junction_cfd_config(config)
    solution_cfg = _solution_config(cfg)
    case_id = str(solution_cfg["case_id"])
    split_cfg = _split_config(solution_cfg)
    viscosity = float(solution_cfg["viscosity_pa_s"])
    outlet_pressure = float(solution_cfg["outlet_pressure_pa"])
    if viscosity <= 0:
        raise ValueError("solution.viscosity_pa_s must be positive")

    if mesh is None:
        mesh = generate_mesh(_geometry_from_config(cfg))

    flow = _load_flow_inputs(cfg)
    total_flow_ul_hr = flow["total_flow_rate_ul_per_hr"]
    channel_height_um = flow["channel_height_um"]
    inlet_flux = _volume_flow_to_2d_flux(total_flow_ul_hr, channel_height_um)
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
    values[pressure_dofs] = outlet_pressure
    constrained = np.unique(np.concatenate([velocity_dofs, pressure_dofs]))
    linear_system_diagnostics = _condensed_system_diagnostics(system, rhs, values, constrained, basis_u.N)
    boundary_dof_diagnostics = _boundary_dof_diagnostics(basis_u, mesh.geometry, velocity_dofs, velocity_values, inlet_flux)

    solution_vector, linear_solver = _solve_condensed_system(system, rhs, values, constrained)
    velocity_solution = solution_vector[: basis_u.N]
    pressure_solution = solution_vector[basis_u.N :]
    divergence_diagnostics = _divergence_diagnostics(basis_u, velocity_solution)

    velocity_nodes = _velocity_at_mesh_nodes(mesh, basis_u, velocity_solution)
    pressure_nodes = _pressure_at_mesh_nodes(mesh, basis_p, pressure_solution)
    fluxes = _boundary_fluxes_fem_quadrature(mesh, basis_u, velocity_solution)
    outlet_total = fluxes["left_outlet"] + fluxes["right_outlet"]
    split = {
        "left": float(fluxes["left_outlet"] / outlet_total) if abs(outlet_total) > 0 else float("nan"),
        "right": float(fluxes["right_outlet"] / outlet_total) if abs(outlet_total) > 0 else float("nan"),
    }

    return StokesSolution(
        case_id=case_id,
        mesh=mesh,
        viscosity_pa_s=viscosity,
        total_flow_rate_ul_per_hr=total_flow_ul_hr,
        inlet_flux_m2_per_s=inlet_flux,
        requested_left_fraction=split_cfg["left_fraction"],
        requested_right_fraction=split_cfg["right_fraction"],
        channel_height_um=channel_height_um,
        inlet_mean_velocity_m_per_s=mean_velocity,
        velocity_node_m_per_s=velocity_nodes,
        pressure_node_pa=pressure_nodes,
        velocity_dof_coordinates_um=_paired_vector_dof_coordinates(basis_u),
        velocity_dof_m_per_s=_paired_vector_dof_values(basis_u, velocity_solution),
        pressure_dof_coordinates_um=basis_p.doflocs.T / UM_TO_M,
        pressure_dof_pa=pressure_solution,
        fluxes_m2_per_s=fluxes,
        split_fraction=split,
        outlet_pressure_pa=outlet_pressure,
        solver_backend=f"scikit-fem/{linear_solver}",
        boundary_condition_summary={
            "walls": "no-slip velocity Dirichlet condition",
            "inlet": f"{solution_cfg['inlet_profile']} velocity Dirichlet profile matching total experiment flow",
            "outlets": (
                f"prescribed parabolic outlet velocity profiles with requested left/right split "
                f"{split_cfg['left_fraction']:.4f}/{split_cfg['right_fraction']:.4f} and a single pressure gauge of {outlet_pressure} Pa"
            ),
            "pressure": "no physical pressure Dirichlet condition is applied at inlet or outlets; one pressure DOF is fixed only as a gauge.",
            "flow_conversion": "3D volumetric flow rate Q is converted to 2D flux per unit depth as q_2D = Q / channel_height.",
        },
        linear_system_diagnostics=linear_system_diagnostics,
        boundary_dof_diagnostics=boundary_dof_diagnostics,
        divergence_diagnostics=divergence_diagnostics,
        assumptions=[
            "The solve is steady, incompressible, Newtonian, and single phase.",
            "The two-dimensional domain represents a depth-averaged cross-section with finite channel height used only to convert Q to area flux.",
            "The dispersed flow contribution is included in the prescribed total inlet flow for this first 50/50 single-phase case.",
            "No-slip is applied on reconstructed channel walls; this first case prescribes a 50/50 outlet flux split.",
            "A sparse direct solve is required; singular systems fail loudly rather than being accepted as least-squares Stokes fields.",
        ],
        postprocessing_config=_postprocessing_config(mesh.geometry, cfg),
    )


def evaluate_solution(solution: StokesSolution) -> StokesReport:
    speed = np.linalg.norm(solution.velocity_node_m_per_s, axis=1)
    flux = solution.fluxes_m2_per_s
    mesh_report = evaluate_mesh(solution.mesh)
    return StokesReport(
        case_id=solution.case_id,
        solver_backend=solution.solver_backend,
        element_pair="P2 velocity / P1 pressure",
        coordinate_units_geometry="um",
        coordinate_units_solve="m",
        viscosity_pa_s=solution.viscosity_pa_s,
        total_flow_rate_ul_per_hr=solution.total_flow_rate_ul_per_hr,
        channel_height_um=solution.channel_height_um,
        inlet_flux_m2_per_s=solution.inlet_flux_m2_per_s,
        requested_left_fraction=solution.requested_left_fraction,
        requested_right_fraction=solution.requested_right_fraction,
        inlet_mean_velocity_m_per_s=solution.inlet_mean_velocity_m_per_s,
        maximum_velocity_m_per_s=float(np.max(speed)),
        mean_velocity_m_per_s=float(np.mean(speed)),
        minimum_pressure_pa=float(np.min(solution.pressure_node_pa)),
        maximum_pressure_pa=float(np.max(solution.pressure_node_pa)),
        inlet_flux_signed_m2_per_s=float(flux["inlet"]),
        left_outlet_flux_m2_per_s=float(flux["left_outlet"]),
        right_outlet_flux_m2_per_s=float(flux["right_outlet"]),
        net_flux_residual_m2_per_s=float(flux["inlet"] + flux["left_outlet"] + flux["right_outlet"]),
        left_split_fraction=float(solution.split_fraction["left"]),
        right_split_fraction=float(solution.split_fraction["right"]),
        mesh_nodes=mesh_report.number_of_nodes,
        mesh_elements=mesh_report.number_of_elements,
        outlet_pressure_pa=solution.outlet_pressure_pa,
    )


def save_solution_outputs(solution: StokesSolution, output_root: str | Path, overwrite: bool = False) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    reports = root / "reports"
    fields = root / "fields"
    for path in (figures, reports, fields):
        path.mkdir(parents=True, exist_ok=True)

    outputs = [
        fields / "stokes_solution.npz",
        reports / "solution_metadata.json",
        reports / "solution_report.md",
        figures / "velocity_magnitude.png",
        figures / "velocity_vectors.png",
        figures / "pressure_field.png",
        figures / "velocity_streamlines.png",
        figures / "velocity_streamlines_dense.png",
        figures / "velocity_streamlines_inlet_seeded.png",
        reports / "streamline_seed_diagnostics.csv",
    ]
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Solution outputs already exist. Use --overwrite: {existing}")

    report = evaluate_solution(solution)
    streamline_cfg = solution.postprocessing_config
    streamline_traces = _trace_inlet_seeded_streamlines(solution, streamline_cfg)
    streamline_diagnostics = evaluate_streamline_diagnostics(solution, streamline_cfg, streamline_traces)
    np.savez(
        fields / "stokes_solution.npz",
        nodes_um=solution.mesh.nodes_um,
        elements=solution.mesh.elements,
        velocity_nodes_m_per_s=solution.velocity_node_m_per_s,
        pressure_nodes_pa=solution.pressure_node_pa,
        velocity_dof_coordinates_um=solution.velocity_dof_coordinates_um,
        velocity_dofs_m_per_s=solution.velocity_dof_m_per_s,
        pressure_dof_coordinates_um=solution.pressure_dof_coordinates_um,
        pressure_dofs_pa=solution.pressure_dof_pa,
        inlet_streamline_seeds_um=_inlet_seed_points(solution.mesh.geometry, streamline_cfg),
    )
    metadata = {
        "report": asdict(report),
        "streamline_diagnostics": asdict(streamline_diagnostics),
        "boundary_conditions": solution.boundary_condition_summary,
        "boundary_dof_diagnostics": solution.boundary_dof_diagnostics,
        "linear_system_diagnostics": solution.linear_system_diagnostics,
        "divergence_diagnostics": solution.divergence_diagnostics,
        "cfd_version": "1.0",
        "mesh_version": "production_v1",
        "requested_left_fraction": solution.requested_left_fraction,
        "requested_right_fraction": solution.requested_right_fraction,
        "total_inlet_flow_ul_per_hr": solution.total_flow_rate_ul_per_hr,
        "assumptions": solution.assumptions,
        "units": {
            "geometry_coordinates": "um",
            "solve_coordinates": "m",
            "velocity": "m/s",
            "pressure": "Pa",
            "flux_2d": "m^2/s",
            "viscosity": "Pa s",
        },
    }
    (reports / "solution_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (reports / "solution_report.md").write_text(_markdown_report(report, solution), encoding="utf-8")
    _save_velocity_magnitude_figure(solution, figures / "velocity_magnitude.png")
    _save_velocity_vector_figure(solution, figures / "velocity_vectors.png")
    _save_pressure_figure(solution, figures / "pressure_field.png")
    _save_dense_streamline_figure(solution, figures / "velocity_streamlines_dense.png", streamline_cfg)
    _save_seed_diagnostics_csv(reports / "streamline_seed_diagnostics.csv", streamline_traces)
    _save_inlet_seeded_streamline_figure(
        solution,
        figures / "velocity_streamlines_inlet_seeded.png",
        streamline_cfg,
        streamline_diagnostics,
        streamline_traces,
    )
    _save_dense_streamline_figure(solution, figures / "velocity_streamlines.png", streamline_cfg)


def _geometry_from_config(cfg: dict[str, Any]) -> JunctionGeometry:
    from .domain import build_junction_geometry

    return build_junction_geometry(cfg)


def _solve_condensed_system(
    system,
    rhs: np.ndarray,
    values: np.ndarray,
    constrained: np.ndarray,
) -> tuple[np.ndarray, str]:
    condensed = condense(system, rhs, x=values, D=constrained)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        direct = solve(*condensed)
    if not np.isfinite(direct).all():
        raise RuntimeError("Direct Stokes solve returned non-finite coefficients")
    return direct, "direct"


def _condensed_system_diagnostics(
    system,
    rhs: np.ndarray,
    values: np.ndarray,
    constrained: np.ndarray,
    velocity_dof_count: int,
) -> dict[str, Any]:
    matrix, vector, _expanded, kept = condense(system, rhs, x=values, D=constrained)
    kept = np.asarray(kept, dtype=np.int64)
    singular_values: list[float] = []
    singular_error = None
    if matrix.shape[0] <= 1800:
        try:
            k = max(1, min(6, min(matrix.shape) - 2))
            singular_values = sorted(float(item) for item in svds(matrix, k=k, which="SM", return_singular_vectors=False))
        except Exception as exc:  # pragma: no cover - diagnostic only
            singular_error = str(exc)
    else:
        singular_error = "Skipped because the condensed matrix is larger than the practical diagnostic threshold."
    tolerance = 1.0e-10
    pressure_constant_mode = np.zeros(system.shape[0], dtype=float)
    pressure_constant_mode[velocity_dof_count:] = 1.0
    residual = system @ pressure_constant_mode
    return {
        "condensed_matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "free_velocity_dofs": int(np.count_nonzero(kept < velocity_dof_count)),
        "free_pressure_dofs": int(np.count_nonzero(kept >= velocity_dof_count)),
        "constrained_velocity_dofs": int(np.count_nonzero(constrained < velocity_dof_count)),
        "constrained_pressure_dofs": int(np.count_nonzero(constrained >= velocity_dof_count)),
        "structural_rank": int(structural_rank(matrix)),
        "smallest_singular_value_estimates": singular_values,
        "singular_value_estimation_error": singular_error,
        "detected_nullity_from_estimates": int(sum(value < tolerance for value in singular_values)),
        "pressure_constant_mode_residual_norm_unconstrained": float(np.linalg.norm(residual)),
        "rhs_norm_after_condensation": float(np.linalg.norm(vector)),
        "note": "The only pressure constraint is one gauge DOF; direct solve is required and no least-squares fallback is accepted.",
    }


def _divergence_diagnostics(basis_u: Basis, velocity_solution: np.ndarray) -> dict[str, float]:
    field = basis_u.interpolate(velocity_solution)
    divergence = np.asarray(field.grad[0, 0] + field.grad[1, 1])
    weights = np.asarray(basis_u.dx)
    element_integrals = np.sum(divergence * weights, axis=1)
    element_areas = np.sum(weights, axis=1)
    element_means = element_integrals / element_areas
    return {
        "l2_divergence_s_inv": float(np.sqrt(np.sum((divergence**2) * weights))),
        "max_abs_divergence_s_inv": float(np.max(np.abs(divergence))),
        "integrated_divergence_m2_per_s": float(np.sum(element_integrals)),
        "elementwise_mean_divergence_min_s_inv": float(np.min(element_means)),
        "elementwise_mean_divergence_max_s_inv": float(np.max(element_means)),
        "elementwise_mean_divergence_mean_s_inv": float(np.mean(element_means)),
    }


def _solution_config(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("solution")
    if not isinstance(raw, dict):
        raise ValueError("Junction CFD config is missing solution mapping")
    required = ["case_id", "viscosity_pa_s", "inlet_profile", "outlet_pressure_pa", "output_root"]
    for key in required:
        if key not in raw:
            raise ValueError(f"Junction CFD solution config is missing required key: solution.{key}")
    if raw["inlet_profile"] != "parabolic":
        raise ValueError("Only solution.inlet_profile: parabolic is supported in this first solve")
    return dict(raw)


def _split_config(solution_cfg: dict[str, Any]) -> dict[str, float]:
    left = float(solution_cfg.get("left_fraction", 0.5))
    if not 0.0 < left < 1.0:
        raise ValueError("solution.left_fraction must satisfy 0 < left_fraction < 1")
    return {"left_fraction": left, "right_fraction": 1.0 - left}


def _case_id_from_left_fraction(left_fraction: float) -> str:
    if not 0.0 < left_fraction < 1.0:
        raise ValueError("left_fraction must satisfy 0 < left_fraction < 1")
    return f"split_0p{int(round(left_fraction * 100)):02d}"


def configure_solution_split(cfg: dict[str, Any], left_fraction: float) -> dict[str, Any]:
    """Return a shallow config copy with an explicit prescribed left outlet fraction."""
    case_id = _case_id_from_left_fraction(left_fraction)
    updated = dict(cfg)
    solution = dict(_solution_config(updated))
    solution["left_fraction"] = float(left_fraction)
    solution["right_fraction"] = 1.0 - float(left_fraction)
    solution["case_id"] = case_id
    solution["output_root"] = str(Path("outputs/physics/junction_cfd/solutions") / case_id)
    updated["solution"] = solution
    return updated


def _postprocessing_config(geometry: JunctionGeometry, cfg: dict[str, Any] | None) -> dict[str, Any]:
    raw = cfg.get("postprocessing", {}) if isinstance(cfg, dict) else {}
    config = {
        "interpolation_grid_resolution_um": float(raw.get("interpolation_grid_resolution_um", 4.0)),
        "dense_streamline_density": float(raw.get("dense_streamline_density", 2.4)),
        "inlet_seed_count": int(raw.get("inlet_seed_count", 31)),
        "streamline_seed_offset_um": float(raw.get("streamline_seed_offset_um", 8.0)),
        "streamline_trace_step_um": float(raw.get("streamline_trace_step_um", 4.0)),
        "streamline_max_steps": int(raw.get("streamline_max_steps", 1200)),
        "streamline_low_speed_um_per_s": float(raw.get("streamline_low_speed_um_per_s", 1.0e-6)),
        "outlet_termination_band_um": float(raw.get("outlet_termination_band_um", geometry.target_element_size_um)),
    }
    if config["interpolation_grid_resolution_um"] <= 0:
        raise ValueError("postprocessing.interpolation_grid_resolution_um must be positive")
    if config["dense_streamline_density"] <= 0:
        raise ValueError("postprocessing.dense_streamline_density must be positive")
    if not 2 <= config["inlet_seed_count"] <= 200:
        raise ValueError("postprocessing.inlet_seed_count must be between 2 and 200")
    if config["streamline_seed_offset_um"] <= 0:
        raise ValueError("postprocessing.streamline_seed_offset_um must be positive")
    if config["streamline_trace_step_um"] <= 0:
        raise ValueError("postprocessing.streamline_trace_step_um must be positive")
    if config["streamline_max_steps"] <= 0:
        raise ValueError("postprocessing.streamline_max_steps must be positive")
    if config["streamline_low_speed_um_per_s"] < 0:
        raise ValueError("postprocessing.streamline_low_speed_um_per_s must be non-negative")
    if config["outlet_termination_band_um"] <= 0:
        raise ValueError("postprocessing.outlet_termination_band_um must be positive")
    if config["streamline_seed_offset_um"] >= geometry.junction_padding_um:
        raise ValueError("postprocessing.streamline_seed_offset_um must be smaller than junction_padding_um")
    return config


def _load_flow_inputs(cfg: dict[str, Any]) -> dict[str, float]:
    experiment_path = cfg.get("experiment_config")
    if not experiment_path:
        raise ValueError("Junction CFD config is missing experiment_config")
    loaded = load_experiment_config(experiment_path)
    experiment = loaded["experiment"]["experiment"]
    device = loaded["device"]["device"]
    total_flow = float(experiment.get("derived", {}).get("total_flow_rate_ul_per_hr"))
    channel_height = float(device.get("channel", {}).get("height_um"))
    if total_flow <= 0 or channel_height <= 0:
        raise ValueError("Experiment total flow and device channel height must be positive")
    return {"total_flow_rate_ul_per_hr": total_flow, "channel_height_um": channel_height}


def _volume_flow_to_2d_flux(flow_ul_per_hr: float, channel_height_um: float) -> float:
    return flow_ul_per_hr * UL_PER_HR_TO_M3_PER_S / (channel_height_um * UM_TO_M)


def _velocity_dirichlet_values(
    basis_u: Basis,
    mesh: TriangularMesh,
    inlet_mean_velocity_m_per_s: float,
    inlet_flux_m2_per_s: float,
    left_fraction: float,
    inlet_profile: str,
) -> tuple[np.ndarray, np.ndarray]:
    if inlet_profile != "parabolic":
        raise ValueError("Only parabolic inlet profile is implemented")
    geometry = mesh.geometry
    coords_um = basis_u.doflocs.T / UM_TO_M
    component_x, component_y = basis_u.split_indices()
    site_to_dofs = {
        _coord_key(coords_um[x_dof]): (int(x_dof), int(y_dof))
        for x_dof, y_dof in zip(component_x, component_y)
    }
    half_width = geometry.half_width_um
    profiles = {
        "inlet": (
            geometry.boundary_endpoints_um["inlet"],
            _unit(geometry.branch_tangents["inlet"][0]),
            _unit(geometry.branch_normals["inlet"][0]),
            inlet_mean_velocity_m_per_s,
        ),
        "left_outlet": (
            geometry.boundary_endpoints_um["left_outlet"],
            _unit(geometry.branch_tangents["left"][-1]),
            _unit(geometry.branch_normals["left"][-1]),
            left_fraction * inlet_flux_m2_per_s / (geometry.channel_width_um * UM_TO_M),
        ),
        "right_outlet": (
            geometry.boundary_endpoints_um["right_outlet"],
            _unit(geometry.branch_tangents["right"][-1]),
            _unit(geometry.branch_normals["right"][-1]),
            (1.0 - left_fraction) * inlet_flux_m2_per_s / (geometry.channel_width_um * UM_TO_M),
        ),
    }
    prescribed: dict[int, list[float]] = {}
    for label, facets in mesh.boundary_facets.items():
        for edge in facets:
            start = mesh.nodes_um[int(edge[0])]
            end = mesh.nodes_um[int(edge[1])]
            for point in (start, 0.5 * (start + end), end):
                key = _coord_key(point)
                if key not in site_to_dofs:
                    raise ValueError(f"Could not find P2 velocity DOFs on boundary trace at {point}")
                x_dof, y_dof = site_to_dofs[key]
                if label == "wall":
                    vector = np.array([0.0, 0.0], dtype=float)
                else:
                    endpoint, tangent, normal, mean_speed = profiles[label]
                    offset = float(np.clip((point - endpoint) @ normal, -half_width, half_width))
                    scalar_speed = 1.5 * mean_speed * (1.0 - (offset / half_width) ** 2)
                    vector = scalar_speed * tangent
                prescribed.setdefault(x_dof, []).append(float(vector[0]))
                prescribed.setdefault(y_dof, []).append(float(vector[1]))
    dofs = np.asarray(sorted(prescribed), dtype=np.int64)
    values = np.asarray([_single_prescribed_value(dof, prescribed[dof]) for dof in dofs], dtype=float)
    return dofs.astype(np.int64), values


def _single_prescribed_value(dof: int, values: list[float]) -> float:
    if max(values) - min(values) > 1.0e-10:
        raise ValueError(f"Conflicting velocity Dirichlet values prescribed for DOF {dof}: {values}")
    return float(np.mean(values))


def _scale_open_profiles_to_flux(
    basis_u: Basis,
    mesh: TriangularMesh,
    velocity_dofs: np.ndarray,
    velocity_values: np.ndarray,
    inlet_flux_m2_per_s: float,
    left_fraction: float,
) -> np.ndarray:
    coords_um = basis_u.doflocs.T / UM_TO_M
    labels = classify_boundary_points(coords_um[velocity_dofs], mesh.geometry)
    scaled = velocity_values.copy()
    targets = {
        "inlet": -inlet_flux_m2_per_s,
        "left_outlet": left_fraction * inlet_flux_m2_per_s,
        "right_outlet": (1.0 - left_fraction) * inlet_flux_m2_per_s,
    }
    for name, target in targets.items():
        trial = np.zeros(basis_u.N, dtype=float)
        trial[velocity_dofs] = scaled
        current_flux = _boundary_fluxes_fem_quadrature(mesh, basis_u, trial)[name]
        if abs(current_flux) < 1e-15:
            raise ValueError(f"{name} boundary profile has near-zero flux and cannot be scaled")
        scaled[labels == name] *= target / current_flux
    return scaled


def _boundary_fluxes_fem_quadrature(
    mesh: TriangularMesh,
    basis_u: Basis,
    velocity_solution: np.ndarray,
    quadrature_order: int = 8,
) -> dict[str, float]:
    points, weights = np.polynomial.legendre.leggauss(quadrature_order)
    interpolator = basis_u.interpolator(velocity_solution)
    adjacent_elements = _boundary_edge_to_element(mesh.elements)
    fluxes = {"inlet": 0.0, "left_outlet": 0.0, "right_outlet": 0.0}
    directions = {
        "inlet": -_unit(mesh.geometry.branch_tangents["inlet"][0]),
        "left_outlet": _unit(mesh.geometry.branch_tangents["left"][-1]),
        "right_outlet": _unit(mesh.geometry.branch_tangents["right"][-1]),
    }
    for name, direction in directions.items():
        for edge in mesh.boundary_facets[name]:
            start_um = mesh.nodes_um[edge[0]]
            end_um = mesh.nodes_um[edge[1]]
            length_m = float(np.linalg.norm(end_um - start_um) * UM_TO_M)
            sample_um = 0.5 * (start_um + end_um)[:, None] + 0.5 * np.outer(end_um - start_um, points)
            element_id = adjacent_elements[tuple(sorted((int(edge[0]), int(edge[1]))))]
            centroid_um = mesh.nodes_um[mesh.elements[element_id]].mean(axis=0)
            midpoint_um = 0.5 * (start_um + end_um)
            inward_um = centroid_um - midpoint_um
            sample_um = sample_um + inward_um[:, None] * 1.0e-9
            velocity = interpolator(sample_um * UM_TO_M)
            fluxes[name] += float(np.sum((velocity.T @ direction) * weights) * 0.5 * length_m)
    return fluxes


def _boundary_edge_to_element(elements: np.ndarray) -> dict[tuple[int, int], int]:
    owners: dict[tuple[int, int], list[int]] = {}
    for element_id, element in enumerate(elements):
        for edge in (element[[0, 1]], element[[1, 2]], element[[2, 0]]):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            owners.setdefault(key, []).append(element_id)
    return {edge: ids[0] for edge, ids in owners.items() if len(ids) == 1}


def _boundary_dof_diagnostics(
    basis_u: Basis,
    geometry: JunctionGeometry,
    velocity_dofs: np.ndarray,
    velocity_values: np.ndarray,
    inlet_flux_m2_per_s: float,
) -> dict[str, Any]:
    coords_um = basis_u.doflocs.T / UM_TO_M
    label_sets = _boundary_label_sets(coords_um, geometry)
    names = ["inlet", "left_outlet", "right_outlet", "wall"]
    intersections: dict[str, int] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            intersections[f"{left}__{right}"] = int(len(label_sets[left] & label_sets[right]))
    assigned_labels = classify_boundary_points(coords_um[velocity_dofs], geometry)
    assigned_counts = {str(label): int(count) for label, count in zip(*np.unique(assigned_labels, return_counts=True))}
    target_fluxes = {
        "inlet": -float(inlet_flux_m2_per_s),
        "left_outlet": 0.5 * float(inlet_flux_m2_per_s),
        "right_outlet": 0.5 * float(inlet_flux_m2_per_s),
    }
    return {
        "formulation": {
            "inlet_velocity": "prescribed parabolic Dirichlet velocity",
            "left_outlet_velocity": "prescribed parabolic Dirichlet velocity carrying 50% of inlet flux",
            "right_outlet_velocity": "prescribed parabolic Dirichlet velocity carrying 50% of inlet flux",
            "wall_velocity": "no-slip Dirichlet velocity",
            "pressure_boundaries": "none",
            "pressure_gauge": "exactly one P1 pressure DOF is fixed to remove the constant pressure null mode",
        },
        "velocity_dirichlet_dof_count": int(len(velocity_dofs)),
        "independent_boundary_dof_intersections": intersections,
        "assigned_velocity_dof_label_counts": assigned_counts,
        "corner_policy": "Open-boundary labels intentionally override wall labels at cut-wall corners; parabolic speed is zero at these endpoints.",
        "target_boundary_fluxes_m2_per_s": target_fluxes,
        "discrete_flux_residual_target_m2_per_s": float(sum(target_fluxes.values())),
    }


def _boundary_label_sets(coords_um: np.ndarray, geometry: JunctionGeometry) -> dict[str, set[int]]:
    from .domain import _distance_to_segment, _near_strip_wall

    tol = max(geometry.target_element_size_um * 0.25, 1e-9)
    sets = {"wall": set(np.flatnonzero(_near_strip_wall(coords_um, geometry, tol)).tolist())}
    for name, section in geometry.boundary_sections_um.items():
        sets[name] = set(np.flatnonzero(_distance_to_segment(coords_um, section[0], section[1]) <= tol).tolist())
    return sets


def _pressure_outlet_dofs(basis_p: Basis, geometry: JunctionGeometry) -> np.ndarray:
    labels = classify_boundary_points(basis_p.doflocs.T / UM_TO_M, geometry)
    return np.flatnonzero((labels == "left_outlet") | (labels == "right_outlet")).astype(np.int64)


def _pressure_reference_dof(basis_p: Basis, geometry: JunctionGeometry) -> int:
    outlet_dofs = _pressure_outlet_dofs(basis_p, geometry)
    if len(outlet_dofs):
        target = geometry.boundary_endpoints_um["left_outlet"]
        coords = basis_p.doflocs.T[outlet_dofs] / UM_TO_M
        return int(outlet_dofs[np.argmin(np.linalg.norm(coords - target, axis=1))])
    target = geometry.boundary_endpoints_um["left_outlet"]
    coords = basis_p.doflocs.T / UM_TO_M
    return int(np.argmin(np.linalg.norm(coords - target, axis=1)))


def _pressure_open_boundary_dofs(basis_p: Basis, geometry: JunctionGeometry) -> np.ndarray:
    dofs = _pressure_outlet_dofs(basis_p, geometry)
    if len(dofs) == 0:
        raise ValueError("No pressure degrees of freedom were found on outlet boundaries")
    return dofs


def _velocity_at_mesh_nodes(mesh: TriangularMesh, basis_u: Basis, velocity_solution: np.ndarray) -> np.ndarray:
    coords = _paired_vector_dof_coordinates(basis_u)
    values = _paired_vector_dof_values(basis_u, velocity_solution)
    return _values_at_nodes(mesh.nodes_um, coords, values)


def _pressure_at_mesh_nodes(mesh: TriangularMesh, basis_p: Basis, pressure_solution: np.ndarray) -> np.ndarray:
    return _values_at_nodes(mesh.nodes_um, basis_p.doflocs.T / UM_TO_M, pressure_solution[:, None]).ravel()


def _paired_vector_dof_coordinates(basis_u: Basis) -> np.ndarray:
    component_x, _ = basis_u.split_indices()
    return basis_u.doflocs[:, component_x].T / UM_TO_M


def _paired_vector_dof_values(basis_u: Basis, velocity_solution: np.ndarray) -> np.ndarray:
    component_x, component_y = basis_u.split_indices()
    return np.column_stack([velocity_solution[component_x], velocity_solution[component_y]])


def _values_at_nodes(nodes_um: np.ndarray, coords_um: np.ndarray, values: np.ndarray) -> np.ndarray:
    lookup = {_coord_key(coord): value for coord, value in zip(coords_um, values)}
    found = []
    for node in nodes_um:
        key = _coord_key(node)
        if key not in lookup:
            raise ValueError("Could not map finite-element field values back to mesh nodes")
        found.append(lookup[key])
    return np.asarray(found, dtype=float)


def _coord_key(coord: np.ndarray) -> tuple[float, float]:
    return (round(float(coord[0]), 9), round(float(coord[1]), 9))


def _boundary_fluxes(mesh: TriangularMesh, velocity_nodes_m_per_s: np.ndarray) -> dict[str, float]:
    edges = _boundary_edges(mesh.elements)
    labels = mesh.boundary_labels
    fluxes = {"inlet": 0.0, "left_outlet": 0.0, "right_outlet": 0.0}
    directions = {
        "inlet": -_unit(mesh.geometry.branch_tangents["inlet"][0]),
        "left_outlet": _unit(mesh.geometry.branch_tangents["left"][-1]),
        "right_outlet": _unit(mesh.geometry.branch_tangents["right"][-1]),
    }
    for name in fluxes:
        for edge in mesh.boundary_facets[name]:
            length_m = float(np.linalg.norm(mesh.nodes_um[edge[1]] - mesh.nodes_um[edge[0]]) * UM_TO_M)
            avg_velocity = velocity_nodes_m_per_s[edge].mean(axis=0)
            fluxes[name] += float(avg_velocity @ directions[name]) * length_m
    return fluxes


def _boundary_edges(elements: np.ndarray) -> np.ndarray:
    edges = np.vstack([elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return np.asarray(vector, dtype=float) / norm


def _markdown_report(report: StokesReport, solution: StokesSolution) -> str:
    matrix = solution.linear_system_diagnostics
    boundary = solution.boundary_dof_diagnostics
    return "\n".join(
        [
            "# Junction Stokes Solution Report",
            "",
            f"- Case ID: `{report.case_id}`",
            f"- Solver backend: `{report.solver_backend}`",
            f"- Element pair: {report.element_pair}",
            f"- Geometry coordinates: `{report.coordinate_units_geometry}`",
            f"- Solver coordinates: `{report.coordinate_units_solve}`",
            f"- Viscosity: {report.viscosity_pa_s:.6g} Pa s",
            f"- Total configured flow: {report.total_flow_rate_ul_per_hr:.3f} uL/hr",
            f"- Channel height used for 3D-to-2D conversion: {report.channel_height_um:.3f} um",
            f"- Inlet 2D flux: {report.inlet_flux_m2_per_s:.6e} m^2/s",
            f"- Requested left/right split: {report.requested_left_fraction:.4f} / {report.requested_right_fraction:.4f}",
            f"- Inlet mean velocity: {report.inlet_mean_velocity_m_per_s:.6e} m/s",
            f"- Maximum nodal velocity magnitude: {report.maximum_velocity_m_per_s:.6e} m/s",
            f"- Pressure range: {report.minimum_pressure_pa:.6e} to {report.maximum_pressure_pa:.6e} Pa",
            f"- Inlet signed flux: {report.inlet_flux_signed_m2_per_s:.6e} m^2/s",
            f"- Left outlet flux: {report.left_outlet_flux_m2_per_s:.6e} m^2/s",
            f"- Right outlet flux: {report.right_outlet_flux_m2_per_s:.6e} m^2/s",
            f"- Net flux residual: {report.net_flux_residual_m2_per_s:.6e} m^2/s",
            f"- Left/right split: {report.left_split_fraction:.4f} / {report.right_split_fraction:.4f}",
            "",
            "## Streamline Diagnostics",
            "",
            "- Streamlines are post-processing samples of the solved continuous velocity field.",
            "- The displayed streamline count is controlled by visualization density and inlet seed settings, not by the finite-element mesh resolution.",
            "- Dense streamlines are interpolated onto a masked regular grid so plotted streamlines remain inside the CFD domain.",
            "- Inlet-seeded streamlines use deterministic seed locations across the inlet cut for repeatable comparison across future split cases.",
            "",
            "## Boundary Conditions",
            "",
            *[f"- {key}: {value}" for key, value in solution.boundary_condition_summary.items()],
            "",
            "## Boundary DOF Diagnostics",
            "",
            f"- Velocity Dirichlet DOFs: {boundary['velocity_dirichlet_dof_count']}",
            f"- Boundary DOF intersections: {boundary['independent_boundary_dof_intersections']}",
            f"- Assigned velocity DOF label counts: {boundary['assigned_velocity_dof_label_counts']}",
            f"- Target boundary flux residual: {boundary['discrete_flux_residual_target_m2_per_s']:.6e} m^2/s",
            f"- Corner policy: {boundary['corner_policy']}",
            "",
            "## Linear System Diagnostics",
            "",
            f"- Condensed matrix shape: {matrix['condensed_matrix_shape']}",
            f"- Free velocity DOFs: {matrix['free_velocity_dofs']}",
            f"- Free pressure DOFs: {matrix['free_pressure_dofs']}",
            f"- Constrained velocity DOFs: {matrix['constrained_velocity_dofs']}",
            f"- Constrained pressure DOFs: {matrix['constrained_pressure_dofs']}",
            f"- Structural rank: {matrix['structural_rank']}",
            f"- Detected nullity from singular-value estimates: {matrix['detected_nullity_from_estimates']}",
            f"- Pressure constant-mode residual before gauge: {matrix['pressure_constant_mode_residual_norm_unconstrained']:.6e}",
            f"- Singular-value estimate note: {matrix['singular_value_estimation_error']}",
            "",
            "## Divergence Diagnostics",
            "",
            f"- L2 divergence: {solution.divergence_diagnostics['l2_divergence_s_inv']:.6e} s^-1",
            f"- Max absolute divergence: {solution.divergence_diagnostics['max_abs_divergence_s_inv']:.6e} s^-1",
            f"- Integrated divergence: {solution.divergence_diagnostics['integrated_divergence_m2_per_s']:.6e} m^2/s",
            "",
            "## Assumptions",
            "",
            *[f"- {item}" for item in solution.assumptions],
            "",
        ]
    )


def _triangulation(solution: StokesSolution) -> mtri.Triangulation:
    mesh = solution.mesh
    return mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)


def _format_axes(ax: plt.Axes, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title)


def _save_velocity_magnitude_figure(solution: StokesSolution, path: Path) -> None:
    tri = _triangulation(solution)
    speed_um_s = np.linalg.norm(solution.velocity_node_m_per_s, axis=1) / UM_TO_M
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.tripcolor(tri, speed_um_s, shading="gouraud", cmap="magma")
    fig.colorbar(image, ax=ax, label="velocity magnitude (um/s)")
    ax.triplot(tri, linewidth=0.15, color="white", alpha=0.25)
    _format_axes(ax, "Stokes velocity magnitude")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_velocity_vector_figure(solution: StokesSolution, path: Path) -> None:
    nodes = solution.mesh.nodes_um
    velocity_um_s = solution.velocity_node_m_per_s / UM_TO_M
    speed = np.linalg.norm(velocity_um_s, axis=1)
    keep = speed > np.nanpercentile(speed, 10)
    sample = np.flatnonzero(keep)[:: max(1, keep.sum() // 120)]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.triplot(_triangulation(solution), linewidth=0.2, color="#d1d5db")
    ax.quiver(
        nodes[sample, 0],
        nodes[sample, 1],
        velocity_um_s[sample, 0],
        velocity_um_s[sample, 1],
        speed[sample],
        cmap="viridis",
        scale_units="xy",
        angles="xy",
        scale=12000,
        width=0.003,
    )
    _format_axes(ax, "Stokes velocity vectors")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_pressure_figure(solution: StokesSolution, path: Path) -> None:
    tri = _triangulation(solution)
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.tripcolor(tri, solution.pressure_node_pa, shading="gouraud", cmap="coolwarm")
    fig.colorbar(image, ax=ax, label="pressure (Pa)")
    ax.triplot(tri, linewidth=0.15, color="white", alpha=0.25)
    _format_axes(ax, "Stokes pressure field")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_streamline_diagnostics(
    solution: StokesSolution,
    config: dict[str, Any],
    traces: list[dict[str, Any]] | None = None,
) -> StreamlineDiagnostics:
    if traces is None:
        traces = _trace_inlet_seeded_streamlines(solution, config)
    counts = {"left": 0, "right": 0, "other": 0}
    reason_counts: dict[str, int] = {}
    for trace in traces:
        counts[trace["termination"]] += 1
        reason = str(trace["termination_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return StreamlineDiagnostics(
        interpolation_grid_resolution_um=float(config["interpolation_grid_resolution_um"]),
        dense_streamline_density=float(config["dense_streamline_density"]),
        inlet_seed_count=int(config["inlet_seed_count"]),
        seed_offset_um=float(config["streamline_seed_offset_um"]),
        trace_step_um=float(config["streamline_trace_step_um"]),
        max_trace_steps=int(config["streamline_max_steps"]),
        terminated_left=counts["left"],
        terminated_right=counts["right"],
        terminated_other=counts["other"],
        termination_reason_counts=reason_counts,
        note=(
            "Streamlines are post-processing samples of the solved continuous velocity field. "
            "The number displayed is controlled by visualization settings and is not a CFD mesh-resolution limit."
        ),
    )


def _save_dense_streamline_figure(solution: StokesSolution, path: Path, config: dict[str, Any]) -> None:
    grid = _interpolated_velocity_grid(solution, config)
    fig, ax = plt.subplots(figsize=(7, 7))
    speed = np.ma.sqrt(grid["u"] ** 2 + grid["v"] ** 2)
    tri = _triangulation(solution)
    ax.tricontourf(tri, np.linalg.norm(solution.velocity_node_m_per_s / UM_TO_M, axis=1), levels=32, cmap="magma")
    ax.streamplot(
        grid["x"],
        grid["y"],
        grid["u"],
        grid["v"],
        color=speed,
        density=float(config["dense_streamline_density"]),
        linewidth=0.75,
        cmap="viridis",
        arrowsize=0.75,
        minlength=0.04,
    )
    ax.triplot(tri, linewidth=0.12, color="white", alpha=0.2)
    _format_axes(ax, "Dense post-processed streamlines")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_inlet_seeded_streamline_figure(
    solution: StokesSolution,
    path: Path,
    config: dict[str, Any],
    diagnostics: StreamlineDiagnostics,
    traces: list[dict[str, Any]] | None = None,
) -> None:
    if traces is None:
        traces = _trace_inlet_seeded_streamlines(solution, config)
    tri = _triangulation(solution)
    speed_um_s = np.linalg.norm(solution.velocity_node_m_per_s / UM_TO_M, axis=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.tripcolor(tri, speed_um_s, shading="gouraud", cmap="magma", alpha=0.85)
    colors = {
        "reached_left_outlet": "#38bdf8",
        "reached_right_outlet": "#a7f3d0",
        "hit_wall": "#f97316",
        "left_domain": "#ef4444",
        "interpolation_failure": "#eab308",
        "low_speed": "#c084fc",
        "max_steps": "#fbbf24",
        "invalid_element": "#ffffff",
        "other": "#9ca3af",
    }
    for trace in traces:
        pts = trace["points_um"]
        if len(pts) < 2:
            continue
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=colors.get(trace["termination_reason"], colors["other"]),
            linewidth=0.8,
            alpha=0.95,
        )
    seeds = _inlet_seed_points(solution.mesh.geometry, config)
    ax.scatter(seeds[:, 0], seeds[:, 1], s=12, c="white", edgecolors="black", linewidths=0.3, zorder=4)
    ax.text(
        0.02,
        0.02,
        f"left={diagnostics.terminated_left}, right={diagnostics.terminated_right}, other={diagnostics.terminated_other}",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3},
    )
    ax.triplot(tri, linewidth=0.12, color="white", alpha=0.2)
    _format_axes(ax, "Inlet-seeded post-processed streamlines")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _interpolated_velocity_grid(solution: StokesSolution, config: dict[str, Any]) -> dict[str, Any]:
    nodes = solution.velocity_dof_coordinates_um
    velocity = solution.velocity_dof_m_per_s / UM_TO_M
    spacing = float(config["interpolation_grid_resolution_um"])
    x = np.arange(nodes[:, 0].min(), nodes[:, 0].max() + spacing, spacing)
    y = np.arange(nodes[:, 1].min(), nodes[:, 1].max() + spacing, spacing)
    xx, yy = np.meshgrid(x, y)
    interp_u = LinearNDInterpolator(nodes, velocity[:, 0])
    interp_v = LinearNDInterpolator(nodes, velocity[:, 1])
    uu = interp_u(xx, yy)
    vv = interp_v(xx, yy)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    outside = ~inside_junction_domain(points, solution.mesh.geometry, tolerance_um=0.0)
    mask = np.isnan(uu).ravel() | np.isnan(vv).ravel() | outside
    return {
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "u": np.ma.array(uu, mask=mask.reshape(xx.shape)),
        "v": np.ma.array(vv, mask=mask.reshape(xx.shape)),
    }


def _inlet_seed_points(geometry: JunctionGeometry, config: dict[str, Any]) -> np.ndarray:
    section = geometry.boundary_sections_um["inlet"]
    count = int(config["inlet_seed_count"])
    margin = max(geometry.channel_width_um * 0.28, geometry.target_element_size_um * 0.75)
    t = np.linspace(margin / geometry.channel_width_um, 1.0 - margin / geometry.channel_width_um, count)
    seeds = section[0] + t[:, None] * (section[1] - section[0])
    tangent = _unit(geometry.branch_tangents["inlet"][0])
    return seeds + tangent * float(config["streamline_seed_offset_um"])


def _save_seed_diagnostics_csv(path: Path, traces: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_id",
                "seed_x",
                "seed_y",
                "termination_reason",
                "final_x",
                "final_y",
                "steps",
                "arc_length",
            ],
        )
        writer.writeheader()
        for trace in traces:
            final = trace["points_um"][-1]
            seed = trace["seed_um"]
            writer.writerow(
                {
                    "seed_id": trace["seed_id"],
                    "seed_x": f"{seed[0]:.9g}",
                    "seed_y": f"{seed[1]:.9g}",
                    "termination_reason": trace["termination_reason"],
                    "final_x": f"{final[0]:.9g}",
                    "final_y": f"{final[1]:.9g}",
                    "steps": trace["steps"],
                    "arc_length": f"{trace['arc_length_um']:.9g}",
                }
            )


class _LocalTriangleVelocityField:
    """Piecewise-linear velocity lookup inside the containing mesh triangle."""

    def __init__(self, nodes_um: np.ndarray, elements: np.ndarray, velocity_um_per_s: np.ndarray) -> None:
        self.nodes_um = np.asarray(nodes_um, dtype=float)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.velocity_um_per_s = np.asarray(velocity_um_per_s, dtype=float)
        self.triangulation = mtri.Triangulation(self.nodes_um[:, 0], self.nodes_um[:, 1], self.elements)
        self.finder = self.triangulation.get_trifinder()

    def evaluate(self, point_um: np.ndarray) -> tuple[np.ndarray, str]:
        point = np.asarray(point_um, dtype=float)
        elem_id = int(self.finder(float(point[0]), float(point[1])))
        if elem_id < 0:
            return np.zeros(2), "invalid_element"
        tri = self.elements[elem_id]
        vertices = self.nodes_um[tri]
        weights = _barycentric_weights(point, vertices)
        if weights is None or not np.isfinite(weights).all():
            return np.zeros(2), "interpolation_failure"
        velocity = weights @ self.velocity_um_per_s[tri]
        if not np.isfinite(velocity).all():
            return np.zeros(2), "interpolation_failure"
        return velocity, "ok"


def _barycentric_weights(point: np.ndarray, vertices: np.ndarray) -> np.ndarray | None:
    a, b, c = vertices
    matrix = np.column_stack([a - c, b - c])
    rhs = point - c
    det = float(np.linalg.det(matrix))
    if abs(det) <= 1e-12:
        return None
    l1, l2 = np.linalg.solve(matrix, rhs)
    l3 = 1.0 - l1 - l2
    return np.array([l1, l2, l3], dtype=float)


def _trace_inlet_seeded_streamlines(solution: StokesSolution, config: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = _inlet_seed_points(solution.mesh.geometry, config)
    field = _LocalTriangleVelocityField(solution.mesh.nodes_um, solution.mesh.elements, solution.velocity_node_m_per_s / UM_TO_M)
    traces = []
    for seed_id, seed in enumerate(seeds):
        trace = _trace_streamline(seed_id, seed, solution.mesh.geometry, field, config)
        traces.append(trace)
    return traces


def _trace_streamline(
    seed_id: int,
    seed_um: np.ndarray,
    geometry: JunctionGeometry,
    field: "_LocalTriangleVelocityField",
    config: dict[str, Any],
) -> dict[str, Any]:
    step = float(config["streamline_trace_step_um"])
    max_steps = int(config["streamline_max_steps"])
    points = [np.asarray(seed_um, dtype=float)]
    reason = "max_steps"
    for _ in range(max_steps):
        current = points[-1]
        termination = _classify_trace_termination(current, geometry, config)
        if termination in {"reached_left_outlet", "reached_right_outlet"}:
            reason = termination
            break
        delta, step_reason = _streamline_rk4_delta(current, step, field, config)
        if delta is None:
            reason = step_reason
            break
        candidate = current + delta
        if not inside_junction_domain(candidate[None, :], geometry, tolerance_um=step * 0.05)[0]:
            candidate = _adaptive_inside_step(current, delta, geometry)
            if candidate is None:
                reason = "hit_wall" if _near_wall(current, geometry) else "left_domain"
                break
        points.append(candidate)
    points_array = np.asarray(points)
    termination_group = "other"
    if reason == "reached_left_outlet":
        termination_group = "left"
    elif reason == "reached_right_outlet":
        termination_group = "right"
    return {
        "seed_id": seed_id,
        "seed_um": np.asarray(seed_um, dtype=float),
        "points_um": points_array,
        "termination_reason": reason,
        "termination": termination_group,
        "steps": int(max(0, len(points_array) - 1)),
        "arc_length_um": float(np.sum(np.linalg.norm(np.diff(points_array, axis=0), axis=1))) if len(points_array) > 1 else 0.0,
    }


def _streamline_direction(point_um: np.ndarray, field: "_LocalTriangleVelocityField", config: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    velocity, status = field.evaluate(point_um)
    if status != "ok":
        return None, status
    speed = float(np.linalg.norm(velocity))
    if speed <= float(config["streamline_low_speed_um_per_s"]):
        return None, "low_speed"
    return velocity / speed, "ok"


def _streamline_rk4_delta(
    point_um: np.ndarray,
    step_um: float,
    field: "_LocalTriangleVelocityField",
    config: dict[str, Any],
) -> tuple[np.ndarray | None, str]:
    k1, reason = _streamline_direction(point_um, field, config)
    if k1 is None:
        return None, reason
    k2, reason = _streamline_direction(point_um + 0.5 * step_um * k1, field, config)
    if k2 is None:
        return step_um * k1, "ok"
    k3, reason = _streamline_direction(point_um + 0.5 * step_um * k2, field, config)
    if k3 is None:
        return step_um * k2, "ok"
    k4, reason = _streamline_direction(point_um + step_um * k3, field, config)
    if k4 is None:
        return step_um * k3, "ok"
    return step_um * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, "ok"


def _classify_trace_termination(point_um: np.ndarray, geometry: JunctionGeometry, config: dict[str, Any]) -> str:
    band_um = float(config["outlet_termination_band_um"])
    if _inside_outlet_band(point_um, geometry, "left_outlet", "left", band_um):
        return "reached_left_outlet"
    if _inside_outlet_band(point_um, geometry, "right_outlet", "right", band_um):
        return "reached_right_outlet"
    return "other"


def _inside_outlet_band(
    point_um: np.ndarray,
    geometry: JunctionGeometry,
    section_name: str,
    branch_name: str,
    band_um: float,
) -> bool:
    endpoint = geometry.boundary_endpoints_um[section_name]
    tangent = _unit(geometry.branch_tangents[branch_name][-1])
    normal = _unit(geometry.branch_normals[branch_name][-1])
    rel = point_um - endpoint
    upstream_distance = -float(rel @ tangent)
    transverse = abs(float(rel @ normal))
    return 0.0 <= upstream_distance <= band_um and transverse <= geometry.half_width_um * 0.98


def _near_wall(point_um: np.ndarray, geometry: JunctionGeometry) -> bool:
    label = classify_boundary_points(point_um[None, :], geometry)[0]
    return label == "wall"


def _adaptive_inside_step(current: np.ndarray, delta: np.ndarray, geometry: JunctionGeometry) -> np.ndarray | None:
    scale = 0.5
    for _ in range(10):
        candidate = current + delta * scale
        if inside_junction_domain(candidate[None, :], geometry, tolerance_um=0.0)[0]:
            return candidate
        scale *= 0.5
    return None


def _distance_to_segment(point_um: np.ndarray, segment_um: np.ndarray) -> float:
    start, end = segment_um
    vector = end - start
    length2 = float(np.dot(vector, vector))
    if length2 <= 0:
        return float(np.linalg.norm(point_um - start))
    t = np.clip(float(np.dot(point_um - start, vector) / length2), 0.0, 1.0)
    projected = start + t * vector
    return float(np.linalg.norm(point_um - projected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the first single-phase junction Stokes case.")
    parser.add_argument("--config", type=Path, default=Path("configs/physics/junction_cfd.yml"))
    parser.add_argument("--left-fraction", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_junction_cfd_config(args.config)
    if args.left_fraction is not None:
        cfg = configure_solution_split(cfg, args.left_fraction)
    solution = solve_junction_stokes(cfg)
    output_root = _solution_config(cfg)["output_root"]
    save_solution_outputs(solution, output_root, overwrite=args.overwrite)
    report = evaluate_solution(solution)
    print("Junction Stokes solve completed")
    print(f"  case: {report.case_id}")
    print(f"  backend: {report.solver_backend}")
    print(f"  elements: {report.mesh_elements}")
    print(f"  nodes: {report.mesh_nodes}")
    print(f"  inlet mean velocity: {report.inlet_mean_velocity_m_per_s:.6e} m/s")
    print(f"  max velocity: {report.maximum_velocity_m_per_s:.6e} m/s")
    print(f"  pressure range: {report.minimum_pressure_pa:.6e} to {report.maximum_pressure_pa:.6e} Pa")
    print(f"  split left/right: {report.left_split_fraction:.4f} / {report.right_split_fraction:.4f}")
    print(f"  net flux residual: {report.net_flux_residual_m2_per_s:.6e} m^2/s")


if __name__ == "__main__":
    main()
