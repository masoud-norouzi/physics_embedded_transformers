from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.physics.cfd.solver import UM_TO_M, _boundary_fluxes_fem_quadrature, _divergence_diagnostics

from .field_sampler import paired_velocity_to_basis_coefficients, sample_velocity_field, velocity_basis
from .library import DEFAULT_CONFIG_PATH, DEFAULT_LIBRARY_PATH, VelocityFieldLibrary
from .split_interpolator import interpolate_split
from .types import InterpolatedVelocityField, VelocityFieldCase


DEFAULT_WITHHELD_FRACTIONS = (0.20, 0.40, 0.60, 0.80)


@dataclass(frozen=True)
class WithheldValidationMetrics:
    left_fraction: float
    lower_fraction: float
    upper_fraction: float
    interpolation_weight: float
    dof_component_rmse_m_per_s: float
    dof_speed_rmse_m_per_s: float
    sample_component_rmse_m_per_s: float
    sample_speed_rmse_m_per_s: float
    mean_relative_velocity_error: float
    maximum_relative_velocity_error: float
    mean_angular_error_deg: float
    maximum_angular_error_deg: float
    inlet_flux_error_m2_per_s: float
    left_outlet_flux_error_m2_per_s: float
    right_outlet_flux_error_m2_per_s: float
    mass_balance_residual_m2_per_s: float
    divergence_l2_norm: float
    divergence_max_abs: float
    inside_sample_count: int


def validate_velocity_interpolation(
    library_path: str | Path = DEFAULT_LIBRARY_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_root: str | Path = "outputs/physics/interpolation/validation",
    withheld_fractions: tuple[float, ...] = DEFAULT_WITHHELD_FRACTIONS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate split interpolation by withholding stored CFD cases."""
    library = VelocityFieldLibrary.from_directory(library_path, config_path=config_path)
    out = Path(output_root)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Interpolation validation output already exists. Use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    sample_points = _interior_sample_points(library.cases[0])
    metrics = []
    for alpha in withheld_fractions:
        direct = library.case_for_fraction(alpha)
        training_cases = [case for case in library.cases if not math.isclose(case.left_fraction, alpha, abs_tol=1.0e-12)]
        interpolated = interpolate_split(training_cases, alpha)
        case_dir = out / f"withheld_{_fraction_token(alpha)}"
        case_dir.mkdir(exist_ok=True)
        item = _compare_withheld_case(direct, interpolated, sample_points, case_dir, figures)
        metrics.append(item)

    summary = {
        "library_path": str(library.root),
        "cfd_version": library.cases[0].cfd_version,
        "mesh_version": library.cases[0].mesh_version,
        "interpolation_method": "linear interpolation of paired P2 FEM velocity coefficients",
        "withheld_fractions": list(withheld_fractions),
        "sample_point_count": int(len(sample_points)),
        "metrics": [asdict(item) for item in metrics],
        "scientific_consistency": _scientific_consistency_checks(library, sample_points),
    }
    (out / "interpolation_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "interpolation_validation_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    _save_summary_figure(metrics, figures / "withheld_error_summary.png")
    return summary


def _compare_withheld_case(
    direct: VelocityFieldCase,
    interpolated: InterpolatedVelocityField,
    sample_points_um: np.ndarray,
    case_dir: Path,
    figures_dir: Path,
) -> WithheldValidationMetrics:
    diff_dof = interpolated.velocity_dof_m_per_s - direct.velocity_dof_m_per_s
    direct_speed_dof = np.linalg.norm(direct.velocity_dof_m_per_s, axis=1)
    interp_speed_dof = np.linalg.norm(interpolated.velocity_dof_m_per_s, axis=1)

    direct_samples = sample_velocity_field(_field_from_case(direct), sample_points_um)
    interp_samples = interpolated.sample(sample_points_um)
    valid = direct_samples.inside_domain & interp_samples.inside_domain & np.isfinite(interp_samples.speed_m_per_s)
    direct_uv = np.column_stack([direct_samples.u_x_m_per_s, direct_samples.u_y_m_per_s])
    interp_uv = np.column_stack([interp_samples.u_x_m_per_s, interp_samples.u_y_m_per_s])
    diff_sample = interp_uv[valid] - direct_uv[valid]
    direct_speed = direct_samples.speed_m_per_s[valid]
    interp_speed = interp_samples.speed_m_per_s[valid]
    speed_threshold = max(1.0e-8, 0.01 * float(np.nanmax(direct_speed)))
    energetic = valid.copy()
    energetic[valid] = direct_speed > speed_threshold
    energetic_direct = direct_uv[energetic]
    energetic_interp = interp_uv[energetic]
    energetic_speed = direct_samples.speed_m_per_s[energetic]
    rel = np.linalg.norm(energetic_interp - energetic_direct, axis=1) / energetic_speed
    angular = _angular_errors_deg(energetic_direct, energetic_interp)

    basis = velocity_basis(interpolated.nodes_um, interpolated.elements)
    coeff = paired_velocity_to_basis_coefficients(basis, interpolated.velocity_dof_m_per_s)
    flux = _boundary_fluxes_fem_quadrature(interpolated.mesh, basis, coeff)
    direct_flux = direct.flux_report
    div = _divergence_diagnostics(basis, coeff)

    samples_table = np.column_stack(
        [
            sample_points_um[valid],
            direct_uv[valid],
            interp_uv[valid],
            direct_speed,
            interp_speed,
        ]
    )
    np.savetxt(
        case_dir / "sample_comparison.csv",
        samples_table,
        delimiter=",",
        header="x_um,y_um,direct_u_x_m_per_s,direct_u_y_m_per_s,interp_u_x_m_per_s,interp_u_y_m_per_s,direct_speed_m_per_s,interp_speed_m_per_s",
        comments="",
    )
    _save_case_figures(direct, interpolated, figures_dir)

    return WithheldValidationMetrics(
        left_fraction=float(direct.left_fraction),
        lower_fraction=float(interpolated.lower_library_fraction),
        upper_fraction=float(interpolated.upper_library_fraction),
        interpolation_weight=float(interpolated.interpolation_weight),
        dof_component_rmse_m_per_s=_rmse(diff_dof),
        dof_speed_rmse_m_per_s=_rmse(interp_speed_dof - direct_speed_dof),
        sample_component_rmse_m_per_s=_rmse(diff_sample),
        sample_speed_rmse_m_per_s=_rmse(interp_speed - direct_speed),
        mean_relative_velocity_error=float(np.mean(rel)),
        maximum_relative_velocity_error=float(np.max(rel)),
        mean_angular_error_deg=float(np.mean(angular)),
        maximum_angular_error_deg=float(np.max(angular)),
        inlet_flux_error_m2_per_s=float(flux["inlet"] - direct_flux["inlet_flux"]),
        left_outlet_flux_error_m2_per_s=float(flux["left_outlet"] - direct_flux["left_outlet_flux"]),
        right_outlet_flux_error_m2_per_s=float(flux["right_outlet"] - direct_flux["right_outlet_flux"]),
        mass_balance_residual_m2_per_s=float(flux["inlet"] + flux["left_outlet"] + flux["right_outlet"]),
        divergence_l2_norm=float(div["l2_divergence_s_inv"]),
        divergence_max_abs=float(div["max_abs_divergence_s_inv"]),
        inside_sample_count=int(np.count_nonzero(valid)),
    )


def _field_from_case(case: VelocityFieldCase) -> InterpolatedVelocityField:
    return InterpolatedVelocityField(
        requested_left_fraction=case.left_fraction,
        requested_right_fraction=case.right_fraction,
        lower_library_fraction=case.left_fraction,
        upper_library_fraction=case.left_fraction,
        interpolation_weight=0.0,
        velocity_dof_m_per_s=case.velocity_dof_m_per_s,
        velocity_dof_coordinates_um=case.velocity_dof_coordinates_um,
        velocity_node_m_per_s=case.velocity_node_m_per_s,
        nodes_um=case.nodes_um,
        elements=case.elements,
        mesh=case.mesh,
        velocity_basis_metadata={"element_pair": "P2 velocity / P1 pressure"},
        units=case.units,
        cfd_version=case.cfd_version,
        mesh_version=case.mesh_version,
        exact_match=True,
        lower_case_id=case.case_id,
        upper_case_id=case.case_id,
    )


def _scientific_consistency_checks(library: VelocityFieldLibrary, sample_points: np.ndarray) -> dict[str, Any]:
    fields = [library.interpolate(alpha) for alpha in (0.10, 0.33, 0.50, 0.67, 0.90)]
    finite_inside = []
    mass = []
    divergence = []
    inlet_reference = None
    inlet_unchanged = []
    wall_speeds = []
    for field in fields:
        samples = field.sample(sample_points)
        finite_inside.append(bool(np.isfinite(samples.speed_m_per_s[samples.inside_domain]).all()))
        basis = velocity_basis(field.nodes_um, field.elements)
        coeff = paired_velocity_to_basis_coefficients(basis, field.velocity_dof_m_per_s)
        flux = _boundary_fluxes_fem_quadrature(field.mesh, basis, coeff)
        mass.append(float(flux["inlet"] + flux["left_outlet"] + flux["right_outlet"]))
        divergence.append(_divergence_diagnostics(basis, coeff))
        inlet_dofs = _dofs_near_boundary(field, "inlet")
        inlet_values = field.velocity_dof_m_per_s[inlet_dofs]
        if inlet_reference is None:
            inlet_reference = inlet_values
        inlet_unchanged.append(bool(np.allclose(inlet_values, inlet_reference, rtol=0.0, atol=1.0e-10)))
        wall_dofs = _dofs_near_boundary(field, "wall")
        wall_speeds.append(float(np.max(np.linalg.norm(field.velocity_dof_m_per_s[wall_dofs], axis=1))) if len(wall_dofs) else float("nan"))
    return {
        "checked_fractions": [field.requested_left_fraction for field in fields],
        "no_nan_inside_domain": all(finite_inside),
        "maximum_mass_balance_residual_m2_per_s": float(np.max(np.abs(mass))),
        "maximum_divergence_l2_norm": float(max(item["l2_divergence_s_inv"] for item in divergence)),
        "maximum_divergence_abs": float(max(item["max_abs_divergence_s_inv"] for item in divergence)),
        "inlet_profile_unchanged_across_alpha": all(inlet_unchanged),
        "maximum_wall_speed_m_per_s": float(np.nanmax(wall_speeds)),
    }


def _dofs_near_boundary(field: InterpolatedVelocityField, boundary_name: str) -> np.ndarray:
    facets = field.mesh.boundary_facets[boundary_name]
    if len(facets) == 0:
        return np.asarray([], dtype=np.int64)
    coords = field.velocity_dof_coordinates_um
    tolerance_um = max(field.mesh.geometry.target_element_size_um * 1.0e-3, 1.0e-9)
    keep = np.zeros(len(coords), dtype=bool)
    for edge in facets:
        start = field.nodes_um[int(edge[0])]
        end = field.nodes_um[int(edge[1])]
        keep |= _distance_to_segment(coords, start, end) <= tolerance_um
    return np.flatnonzero(keep)


def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    length2 = float(vector @ vector)
    if length2 <= 0.0:
        return np.linalg.norm(points - start, axis=1)
    rel = points - start
    t = np.clip((rel @ vector) / length2, 0.0, 1.0)
    projected = start + t[:, None] * vector
    return np.linalg.norm(points - projected, axis=1)


def _interior_sample_points(case: VelocityFieldCase, spacing_um: float = 18.0) -> np.ndarray:
    nodes = case.nodes_um
    x = np.arange(nodes[:, 0].min(), nodes[:, 0].max() + spacing_um, spacing_um)
    y = np.arange(nodes[:, 1].min(), nodes[:, 1].max() + spacing_um, spacing_um)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    from src.physics.cfd.domain import inside_junction_domain

    inside = inside_junction_domain(points, case.mesh.geometry, tolerance_um=-case.mesh.geometry.target_element_size_um * 0.1)
    return points[inside]


def _save_case_figures(direct: VelocityFieldCase, interpolated: InterpolatedVelocityField, figures_dir: Path) -> None:
    tri = mtri.Triangulation(direct.nodes_um[:, 0], direct.nodes_um[:, 1], direct.elements)
    direct_speed = np.linalg.norm(direct.velocity_node_m_per_s, axis=1)
    interp_speed = np.linalg.norm(interpolated.velocity_node_m_per_s, axis=1)
    speed_error = np.abs(interp_speed - direct_speed)
    direct_uv = direct.velocity_node_m_per_s
    interp_uv = interpolated.velocity_node_m_per_s
    angle_error = np.zeros(len(direct_uv), dtype=float)
    speed = np.linalg.norm(direct_uv, axis=1)
    valid = speed > max(1.0e-8, 0.01 * float(np.nanmax(speed)))
    angle_error[valid] = _angular_errors_deg(direct_uv[valid], interp_uv[valid])
    token = _fraction_token(direct.left_fraction)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    images = [
        axes[0].tricontourf(tri, interp_speed, levels=32, cmap="magma"),
        axes[1].tricontourf(tri, speed_error, levels=32, cmap="viridis"),
        axes[2].tricontourf(tri, angle_error, levels=32, cmap="cividis"),
    ]
    titles = [
        f"Interpolated speed, alpha={direct.left_fraction:.2f}",
        "|speed interpolation error|",
        "Angular error away from low speed",
    ]
    for ax, image, title in zip(axes, images, titles):
        ax.triplot(tri, color="white", linewidth=0.15, alpha=0.25)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(figures_dir / f"withheld_{token}_velocity_errors.png", dpi=180)
    plt.close(fig)


def _save_summary_figure(metrics: list[WithheldValidationMetrics], path: Path) -> None:
    x = np.array([item.left_fraction for item in metrics])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(x, [item.dof_component_rmse_m_per_s for item in metrics], "o-")
    axes[0].set_title("DOF component RMSE")
    axes[1].plot(x, [item.sample_speed_rmse_m_per_s for item in metrics], "o-")
    axes[1].set_title("Sample speed RMSE")
    axes[2].plot(x, [item.mean_relative_velocity_error for item in metrics], "o-")
    axes[2].set_title("Mean relative velocity error")
    axes[3].plot(x, [abs(item.mass_balance_residual_m2_per_s) for item in metrics], "o-")
    axes[3].set_title("Mass residual")
    for ax in axes:
        ax.set_xlabel("withheld left fraction")
        ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Velocity Interpolation Validation",
        "",
        f"- Library: `{summary['library_path']}`",
        f"- CFD version: `{summary['cfd_version']}`",
        f"- Mesh version: `{summary['mesh_version']}`",
        f"- Method: {summary['interpolation_method']}",
        "",
        "| withheld alpha | neighbors | weight | DOF RMSE | sample speed RMSE | mean rel. error | max angle error | mass residual |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["metrics"]:
        lines.append(
            f"| {item['left_fraction']:.2f} | {item['lower_fraction']:.2f}/{item['upper_fraction']:.2f} | "
            f"{item['interpolation_weight']:.3f} | {item['dof_component_rmse_m_per_s']:.3e} | "
            f"{item['sample_speed_rmse_m_per_s']:.3e} | {item['mean_relative_velocity_error']:.3e} | "
            f"{item['maximum_angular_error_deg']:.3e} | {item['mass_balance_residual_m2_per_s']:.3e} |"
        )
    lines.extend(["", "## Scientific Consistency Checks", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["scientific_consistency"].items())
    return "\n".join(lines)


def _rmse(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    return float(np.sqrt(np.nanmean(data * data)))


def _angular_errors_deg(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref_norm = np.linalg.norm(reference, axis=1)
    cand_norm = np.linalg.norm(candidate, axis=1)
    dot = np.sum(reference * candidate, axis=1) / (ref_norm * cand_norm)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def _fraction_token(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")
