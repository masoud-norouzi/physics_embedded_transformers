from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

from .domain import build_junction_geometry, inside_junction_domain, load_junction_cfd_config
from .inlet_profile_diagnostic import run_inlet_profile_diagnostic
from .mesh import TriangularMesh, evaluate_mesh, evaluate_mesh_topology, generate_mesh, save_mesh_outputs
from .solver import UM_TO_M, StokesSolution, save_solution_outputs, solve_junction_stokes


FINE_MESH_OVERRIDES = {
    "target_element_size_um": 12.0,
    "mesh_point_min_distance_um": 4.0,
}


def run_default_vs_fine_mesh_check(
    config: str | Path | dict[str, Any],
    output_root: str | Path = "outputs/physics/junction_cfd/mesh_convergence/fine",
) -> dict[str, Any]:
    """Run a focused default-vs-fine mesh check for the validated 50/50 case."""
    default_cfg = load_junction_cfd_config(config)
    fine_cfg = {**default_cfg, **FINE_MESH_OVERRIDES}
    output_root = Path(output_root)
    if _is_relative_to(output_root.resolve(), Path("outputs/physics/junction_cfd/version_1").resolve()):
        raise ValueError("Fine mesh convergence output must not be written inside the frozen Version 1 baseline.")
    output_root.mkdir(parents=True, exist_ok=True)

    default_mesh = generate_mesh(build_junction_geometry(default_cfg))
    fine_mesh = generate_mesh(build_junction_geometry(fine_cfg))

    save_mesh_outputs(fine_mesh, evaluate_mesh(fine_mesh), output_root, overwrite=True)
    default_solution = solve_junction_stokes(default_cfg, mesh=default_mesh)
    fine_solution = solve_junction_stokes(fine_cfg, mesh=fine_mesh)
    fine_solution_root = output_root / "solution_split_0p50"
    save_solution_outputs(fine_solution, fine_solution_root, overwrite=True)
    fine_inlet = run_inlet_profile_diagnostic(fine_cfg, output_root=fine_solution_root / "inlet_profile_diagnostics")

    comparison = {
        "default_mesh_is_production_default": True,
        "fine_mesh_is_comparison_only": True,
        "fine_config_overrides": FINE_MESH_OVERRIDES,
        "default_mesh": _mesh_summary(default_mesh),
        "fine_mesh": _mesh_summary(fine_mesh),
        "default_solution": _solution_summary(default_solution),
        "fine_solution": _solution_summary(fine_solution),
        "fine_inlet_poiseuille_diagnostics": [asdict(item) for item in fine_inlet],
        "differences": _difference_summary(default_solution, fine_solution),
        "velocity_field_comparison": _velocity_field_comparison(default_solution, fine_solution),
    }
    comparison["acceptance"] = _acceptance_decision(comparison)
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "default_vs_fine_mesh_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (reports / "default_vs_fine_mesh_comparison.md").write_text(_markdown_report(comparison), encoding="utf-8")
    return comparison


def fine_mesh_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    cfg = load_junction_cfd_config(config)
    return {**cfg, **FINE_MESH_OVERRIDES}


def _mesh_summary(mesh: TriangularMesh) -> dict[str, Any]:
    quality = evaluate_mesh(mesh)
    topology = evaluate_mesh_topology(mesh)
    return {
        "quality": asdict(quality),
        "topology": asdict(topology),
    }


def _solution_summary(solution: StokesSolution) -> dict[str, Any]:
    fluxes = solution.fluxes_m2_per_s
    return {
        "solver_backend": solution.solver_backend,
        "maximum_velocity_m_per_s": float(np.max(np.linalg.norm(solution.velocity_node_m_per_s, axis=1))),
        "minimum_pressure_pa": float(np.min(solution.pressure_node_pa)),
        "maximum_pressure_pa": float(np.max(solution.pressure_node_pa)),
        "pressure_range_pa": float(np.max(solution.pressure_node_pa) - np.min(solution.pressure_node_pa)),
        "pressure_drop_inlet_to_left_pa": _pressure_drop(solution, "inlet", "left_outlet"),
        "pressure_drop_inlet_to_right_pa": _pressure_drop(solution, "inlet", "right_outlet"),
        "inlet_flux_m2_per_s": float(fluxes["inlet"]),
        "left_outlet_flux_m2_per_s": float(fluxes["left_outlet"]),
        "right_outlet_flux_m2_per_s": float(fluxes["right_outlet"]),
        "net_flux_residual_m2_per_s": float(sum(fluxes.values())),
        "streamline_termination_counts": _streamline_counts(solution),
    }


def _pressure_drop(solution: StokesSolution, inlet: str, outlet: str) -> float:
    labels = solution.mesh.boundary_labels
    inlet_pressure = np.mean(solution.pressure_node_pa[labels[inlet]])
    outlet_pressure = np.mean(solution.pressure_node_pa[labels[outlet]])
    return float(inlet_pressure - outlet_pressure)


def _streamline_counts(solution: StokesSolution) -> dict[str, int]:
    from .solver import _trace_inlet_seeded_streamlines

    traces = _trace_inlet_seeded_streamlines(solution, solution.postprocessing_config)
    counts = {"reached_left_outlet": 0, "reached_right_outlet": 0, "other": 0}
    for trace in traces:
        reason = trace["termination_reason"]
        if reason in counts:
            counts[reason] += 1
        else:
            counts["other"] += 1
    return counts


def _difference_summary(default: StokesSolution, fine: StokesSolution) -> dict[str, float]:
    d = _solution_summary(default)
    f = _solution_summary(fine)
    keys = [
        "inlet_flux_m2_per_s",
        "left_outlet_flux_m2_per_s",
        "right_outlet_flux_m2_per_s",
        "maximum_velocity_m_per_s",
        "pressure_range_pa",
        "pressure_drop_inlet_to_left_pa",
        "pressure_drop_inlet_to_right_pa",
    ]
    return {f"{key}_relative_difference": _relative_difference(d[key], f[key]) for key in keys}


def _velocity_field_comparison(default: StokesSolution, fine: StokesSolution) -> dict[str, float]:
    points_um = _junction_probe_points(default.mesh.geometry)
    default_velocity = _velocity_at_points(default, points_um)
    fine_velocity = _velocity_at_points(fine, points_um)
    valid = np.isfinite(default_velocity).all(axis=1) & np.isfinite(fine_velocity).all(axis=1)
    default_velocity = default_velocity[valid]
    fine_velocity = fine_velocity[valid]
    diff = np.linalg.norm(fine_velocity - default_velocity, axis=1)
    default_speed = np.linalg.norm(default_velocity, axis=1)
    fine_speed = np.linalg.norm(fine_velocity, axis=1)
    speed_scale = max(float(np.mean(fine_speed)), 1.0e-15)
    direction_mask = (default_speed > speed_scale * 1.0e-3) & (fine_speed > speed_scale * 1.0e-3)
    angles = _direction_angles_deg(default_velocity[direction_mask], fine_velocity[direction_mask])
    return {
        "sample_point_count": int(len(points_um)),
        "valid_shared_point_count": int(np.count_nonzero(valid)),
        "mean_velocity_magnitude_difference_m_per_s": float(np.mean(diff)),
        "maximum_velocity_magnitude_difference_m_per_s": float(np.max(diff)),
        "mean_velocity_magnitude_relative_difference": float(np.mean(diff) / speed_scale),
        "maximum_velocity_magnitude_relative_difference": float(np.max(diff) / max(float(np.max(fine_speed)), 1.0e-15)),
        "mean_flow_direction_difference_deg": float(np.mean(angles)) if len(angles) else float("nan"),
        "maximum_flow_direction_difference_deg": float(np.max(angles)) if len(angles) else float("nan"),
    }


def _junction_probe_points(geometry, spacing_um: float = 8.0) -> np.ndarray:
    center = geometry.junction_center_um
    radius = geometry.channel_width_um * 1.6
    x = np.arange(center[0] - radius, center[0] + radius + spacing_um, spacing_um)
    y = np.arange(center[1] - radius, center[1] + radius + spacing_um, spacing_um)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points[inside_junction_domain(points, geometry)]


def _velocity_at_points(solution: StokesSolution, points_um: np.ndarray) -> np.ndarray:
    skmesh = MeshTri(solution.mesh.nodes_um.T * UM_TO_M, solution.mesh.elements.T)
    basis = Basis(skmesh, ElementVector(ElementTriP2()), intorder=4)
    coeffs = np.zeros(basis.N, dtype=float)
    component_x, component_y = basis.split_indices()
    coeffs[component_x] = solution.velocity_dof_m_per_s[:, 0]
    coeffs[component_y] = solution.velocity_dof_m_per_s[:, 1]
    try:
        return basis.interpolator(coeffs)(points_um.T * UM_TO_M).T
    except ValueError:
        values = np.full((len(points_um), 2), np.nan, dtype=float)
        for index, point in enumerate(points_um):
            try:
                values[index] = basis.interpolator(coeffs)(point[:, None] * UM_TO_M).ravel()
            except ValueError:
                continue
        return values


def _direction_angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.clip(dot / np.maximum(norms, 1.0e-15), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _relative_difference(default: float, fine: float) -> float:
    return float(abs(fine - default) / max(abs(fine), 1.0e-15))


def _acceptance_decision(comparison: dict[str, Any]) -> dict[str, Any]:
    diffs = comparison["differences"]
    field = comparison["velocity_field_comparison"]
    integral_ok = all(abs(value) < 0.01 for value in diffs.values())
    stream_default = comparison["default_solution"]["streamline_termination_counts"]
    stream_fine = comparison["fine_solution"]["streamline_termination_counts"]
    stream_ok = stream_default == stream_fine == {"reached_left_outlet": 15, "reached_right_outlet": 15, "other": 1}
    field_ok = (
        field["mean_velocity_magnitude_relative_difference"] < 0.05
        and field["mean_flow_direction_difference_deg"] < 5.0
    )
    return {
        "default_mesh_accepted_as_production": bool(integral_ok and stream_ok and field_ok),
        "criteria": {
            "integral_quantities_relative_difference_lt_1_percent": bool(integral_ok),
            "seeded_streamline_topology_unchanged": bool(stream_ok),
            "mean_velocity_difference_lt_5_percent_and_mean_direction_lt_5_deg": bool(field_ok),
        },
        "note": "The fine mesh is retained only as convergence evidence; it is not promoted automatically.",
    }


def _markdown_report(comparison: dict[str, Any]) -> str:
    default_q = comparison["default_mesh"]["quality"]
    fine_q = comparison["fine_mesh"]["quality"]
    default_s = comparison["default_solution"]
    fine_s = comparison["fine_solution"]
    diffs = comparison["differences"]
    field = comparison["velocity_field_comparison"]
    acceptance = comparison["acceptance"]
    lines = [
        "# Default-vs-Fine Mesh Check",
        "",
        "The improved mesh remains the production default. The fine mesh is used only as convergence evidence.",
        "",
        "## Mesh Quality",
        "",
        "| quantity | default | fine |",
        "|---|---:|---:|",
        f"| nodes | {default_q['number_of_nodes']} | {fine_q['number_of_nodes']} |",
        f"| elements | {default_q['number_of_elements']} | {fine_q['number_of_elements']} |",
        f"| minimum angle (deg) | {default_q['minimum_angle_deg']:.6f} | {fine_q['minimum_angle_deg']:.6f} |",
        f"| mean aspect ratio | {default_q['mean_aspect_ratio']:.6f} | {fine_q['mean_aspect_ratio']:.6f} |",
        f"| maximum aspect ratio | {default_q['maximum_aspect_ratio']:.6f} | {fine_q['maximum_aspect_ratio']:.6f} |",
        "",
        "## CFD Quantities",
        "",
        "| quantity | default | fine | relative difference |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "inlet_flux_m2_per_s",
        "left_outlet_flux_m2_per_s",
        "right_outlet_flux_m2_per_s",
        "maximum_velocity_m_per_s",
        "pressure_range_pa",
        "pressure_drop_inlet_to_left_pa",
        "pressure_drop_inlet_to_right_pa",
    ]:
        lines.append(f"| {key} | {default_s[key]:.6e} | {fine_s[key]:.6e} | {diffs[key + '_relative_difference']:.6e} |")
    lines.extend(
        [
            "",
            f"- Default net mass residual: {default_s['net_flux_residual_m2_per_s']:.6e} m^2/s",
            f"- Fine net mass residual: {fine_s['net_flux_residual_m2_per_s']:.6e} m^2/s",
            f"- Default streamline split: {default_s['streamline_termination_counts']}",
            f"- Fine streamline split: {fine_s['streamline_termination_counts']}",
            "",
            "## Junction Velocity Field",
            "",
            f"- Shared sample points: {field['valid_shared_point_count']} / {field['sample_point_count']}",
            f"- Mean velocity-magnitude difference: {field['mean_velocity_magnitude_difference_m_per_s']:.6e} m/s",
            f"- Maximum velocity-magnitude difference: {field['maximum_velocity_magnitude_difference_m_per_s']:.6e} m/s",
            f"- Mean relative velocity-magnitude difference: {field['mean_velocity_magnitude_relative_difference']:.6e}",
            f"- Maximum relative velocity-magnitude difference: {field['maximum_velocity_magnitude_relative_difference']:.6e}",
            f"- Mean flow-direction difference: {field['mean_flow_direction_difference_deg']:.6f} deg",
            f"- Maximum flow-direction difference: {field['maximum_flow_direction_difference_deg']:.6f} deg",
            "",
            "## Acceptance",
            "",
            f"- Default mesh accepted as production: {acceptance['default_mesh_accepted_as_production']}",
            f"- Criteria: {acceptance['criteria']}",
            f"- Note: {acceptance['note']}",
            "",
        ]
    )
    return "\n".join(lines)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the focused default-vs-fine mesh comparison.")
    parser.add_argument("--config", type=Path, default=Path("configs/physics/junction_cfd.yml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/physics/junction_cfd/mesh_convergence/fine"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run_default_vs_fine_mesh_check(args.config, args.output_root)
    fine = comparison["fine_mesh"]["quality"]
    acceptance = comparison["acceptance"]
    print("Default-vs-fine mesh check completed")
    print(f"  fine nodes: {fine['number_of_nodes']}")
    print(f"  fine elements: {fine['number_of_elements']}")
    print(f"  fine minimum angle: {fine['minimum_angle_deg']:.3f} deg")
    print(f"  fine maximum aspect ratio: {fine['maximum_aspect_ratio']:.3f}")
    print(f"  default accepted as production: {acceptance['default_mesh_accepted_as_production']}")


if __name__ == "__main__":
    main()
