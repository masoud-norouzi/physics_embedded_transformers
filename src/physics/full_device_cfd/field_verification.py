from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

from .domain import FullDeviceCFDGeometry, inside_full_device_domain, project_to_centerline
from .mesh import FullDeviceMesh, evaluate_full_device_mesh, generate_full_device_mesh, label_boundary_facets
from .solver import UM_TO_M, full_device_fluxes, solve_full_device_stokes


@dataclass(frozen=True)
class CommonGrid:
    x_um: np.ndarray
    y_um: np.ndarray
    xx_um: np.ndarray
    yy_um: np.ndarray
    points_um: np.ndarray
    inside: np.ndarray
    spacing_um: float


@dataclass(frozen=True)
class GridField:
    ux_m_per_s: np.ndarray
    uy_m_per_s: np.ndarray
    speed_m_per_s: np.ndarray


def build_common_grid(geometry: FullDeviceCFDGeometry, spacing_um: float = 6.0) -> CommonGrid:
    rings = np.vstack([geometry.outer_ring_um, geometry.inner_ring_um])
    x = np.arange(np.floor(rings[:, 0].min()), np.ceil(rings[:, 0].max()) + spacing_um, spacing_um)
    y = np.arange(np.floor(rings[:, 1].min()), np.ceil(rings[:, 1].max()) + spacing_um, spacing_um)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = inside_full_device_domain(points, geometry)
    return CommonGrid(x, y, xx, yy, points, inside, float(spacing_um))


def region_masks(grid: CommonGrid, geometry: FullDeviceCFDGeometry, half_widths_um: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
    hx, hy = half_widths_um or (2.6 * geometry.channel_width_um, 2.2 * geometry.channel_width_um)
    upper = _rect_mask(grid.points_um, geometry.upper_junction_um, hx, hy)
    lower = _rect_mask(grid.points_um, geometry.lower_junction_um, hx, hy)
    return {
        "full_domain": grid.inside.copy(),
        "inlet_junction": grid.inside & upper,
        "outlet_junction": grid.inside & lower,
    }


def evaluate_solution_on_grid(
    nodes_um: np.ndarray,
    elements: np.ndarray,
    velocity_dof_m_per_s: np.ndarray,
    grid: CommonGrid,
) -> GridField:
    velocity = np.full((len(grid.points_um), 2), np.nan, dtype=float)
    if np.any(grid.inside):
        basis = velocity_basis(nodes_um, elements)
        coeff = paired_velocity_to_basis_coefficients(basis, velocity_dof_m_per_s)
        velocity[grid.inside] = evaluate_velocity_basis(basis, coeff, grid.points_um[grid.inside])
    ux = velocity[:, 0].reshape(grid.xx_um.shape)
    uy = velocity[:, 1].reshape(grid.xx_um.shape)
    speed = np.linalg.norm(np.dstack([ux, uy]), axis=2)
    return GridField(ux, uy, speed)


def vector_error_metrics(
    candidate: GridField,
    reference: GridField,
    masks: dict[str, np.ndarray],
    low_speed_fraction: float = 0.01,
    speed_floor_fraction: float = 0.01,
) -> list[dict[str, float | str | int]]:
    rows = []
    cand = _flat_velocity(candidate)
    ref = _flat_velocity(reference)
    ref_speed = np.linalg.norm(ref, axis=1)
    cand_speed = np.linalg.norm(cand, axis=1)
    domain_max_speed = float(np.nanmax(ref_speed))
    angular_threshold = low_speed_fraction * domain_max_speed
    speed_floor = speed_floor_fraction * domain_max_speed
    for name, mask in masks.items():
        valid = mask & np.isfinite(cand).all(axis=1) & np.isfinite(ref).all(axis=1)
        diff = cand[valid] - ref[valid]
        diff_norm = np.linalg.norm(diff, axis=1)
        ref_norm = np.linalg.norm(ref[valid], axis=1)
        candidate_norm = np.linalg.norm(cand[valid], axis=1)
        denom = max(float(np.sqrt(np.sum(ref[valid] ** 2))), 1.0e-30)
        angular_valid = valid & (ref_speed > angular_threshold) & (cand_speed > angular_threshold)
        angle = angular_error_degrees(cand[angular_valid], ref[angular_valid])
        speed_valid = valid & (ref_speed > speed_floor)
        rel_speed = np.abs(cand_speed[speed_valid] - ref_speed[speed_valid]) / np.maximum(ref_speed[speed_valid], speed_floor)
        rows.append(
            {
                "region": name,
                "valid_points": int(np.count_nonzero(valid)),
                "vector_l2_relative_error": float(np.sqrt(np.sum(diff**2)) / denom),
                "mean_absolute_vector_error_m_per_s": _safe_stat(diff_norm, np.mean),
                "median_absolute_vector_error_m_per_s": _safe_stat(diff_norm, np.median),
                "p95_absolute_vector_error_m_per_s": _safe_percentile(diff_norm, 95),
                "max_absolute_vector_error_m_per_s": _safe_stat(diff_norm, np.max),
                "mean_absolute_vector_error_um_per_s": _safe_stat(diff_norm * 1.0e6, np.mean),
                "median_absolute_vector_error_um_per_s": _safe_stat(diff_norm * 1.0e6, np.median),
                "p95_absolute_vector_error_um_per_s": _safe_percentile(diff_norm * 1.0e6, 95),
                "max_absolute_vector_error_um_per_s": _safe_stat(diff_norm * 1.0e6, np.max),
                "angular_low_speed_threshold_m_per_s": angular_threshold,
                "angular_excluded_fraction": float(1.0 - np.count_nonzero(angular_valid) / max(np.count_nonzero(valid), 1)),
                "mean_angular_error_deg": _safe_stat(angle, np.mean),
                "median_angular_error_deg": _safe_stat(angle, np.median),
                "p95_angular_error_deg": _safe_percentile(angle, 95),
                "max_angular_error_deg": _safe_stat(angle, np.max),
                "speed_floor_m_per_s": speed_floor,
                "speed_l2_relative_error": float(np.linalg.norm(candidate_norm - ref_norm) / max(np.linalg.norm(ref_norm), 1.0e-30)),
                "mean_relative_speed_error": _safe_stat(rel_speed, np.mean),
                "median_relative_speed_error": _safe_stat(rel_speed, np.median),
                "p95_relative_speed_error": _safe_percentile(rel_speed, 95),
            }
        )
    return rows


def angular_error_degrees(candidate_velocity: np.ndarray, reference_velocity: np.ndarray) -> np.ndarray:
    cand_norm = np.linalg.norm(candidate_velocity, axis=1)
    ref_norm = np.linalg.norm(reference_velocity, axis=1)
    denom = cand_norm * ref_norm
    valid = denom > 0
    out = np.full(len(candidate_velocity), np.nan, dtype=float)
    cos_theta = np.sum(candidate_velocity[valid] * reference_velocity[valid], axis=1) / denom[valid]
    out[valid] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return out


def classify_separatrix(
    field: GridField,
    grid: CommonGrid,
    geometry: FullDeviceCFDGeometry,
    seed_count: int = 31,
    wall_margin_fraction: float = 0.08,
    step_um: float = 4.0,
    max_steps: int = 2500,
) -> dict[str, Any]:
    seeds, signed_offsets = inlet_seed_points(geometry, seed_count, wall_margin_fraction)
    classifications = []
    traces = []
    for seed_id, seed in enumerate(seeds):
        result = trace_branch(field, grid, geometry, seed, step_um=step_um, max_steps=max_steps)
        classifications.append(
            {
                "seed_id": seed_id,
                "seed_x_um": float(seed[0]),
                "seed_y_um": float(seed[1]),
                "signed_offset_um": float(signed_offsets[seed_id]),
                "classification": result["classification"],
                "final_x_um": float(result["final"][0]),
                "final_y_um": float(result["final"][1]),
                "steps": int(result["steps"]),
            }
        )
        traces.append(result["trace"])
    transition = separatrix_seed_location(classifications)
    return {
        "seed_count": int(seed_count),
        "wall_margin_fraction": float(wall_margin_fraction),
        "step_um": float(step_um),
        "transition_signed_offset_um": transition,
        "classifications": classifications,
        "traces": traces,
    }


def inlet_seed_points(
    geometry: FullDeviceCFDGeometry,
    seed_count: int = 31,
    wall_margin_fraction: float = 0.08,
    inward_offset_um: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    inlet = geometry.centerlines["inlet"]
    center = geometry.inlet_cut_center_um + inward_offset_um * inlet.tangents[0]
    normal = inlet.normals[0]
    half = geometry.half_width_um * (1.0 - wall_margin_fraction)
    offsets = np.linspace(-half, half, seed_count)
    return center + offsets[:, None] * normal, offsets


def trace_branch(
    field: GridField,
    grid: CommonGrid,
    geometry: FullDeviceCFDGeometry,
    seed_um: np.ndarray,
    step_um: float = 4.0,
    max_steps: int = 2500,
) -> dict[str, Any]:
    point = np.asarray(seed_um, dtype=float)
    trace = [point.copy()]
    for step in range(max_steps):
        velocity = interpolate_grid_velocity(field, grid, point)
        if not np.isfinite(velocity).all():
            break
        speed = float(np.linalg.norm(velocity))
        if speed <= 1.0e-12:
            break
        direction = velocity / speed
        point = point + step_um * direction
        trace.append(point.copy())
        if not inside_full_device_domain(point[None, :], geometry)[0]:
            break
        label = _branch_reached(point, geometry)
        if label in {"left", "right"}:
            return {"classification": label, "final": point, "steps": step + 1, "trace": np.asarray(trace)}
    return {"classification": "unclassified", "final": point, "steps": len(trace) - 1, "trace": np.asarray(trace)}


def separatrix_seed_location(classifications: list[dict[str, Any]]) -> float | None:
    ordered = sorted(classifications, key=lambda row: row["signed_offset_um"])
    for left_row, right_row in zip(ordered[:-1], ordered[1:]):
        pair = {left_row["classification"], right_row["classification"]}
        if pair == {"left", "right"}:
            return float(0.5 * (left_row["signed_offset_um"] + right_row["signed_offset_um"]))
    left_offsets = [float(row["signed_offset_um"]) for row in ordered if row["classification"] == "left"]
    right_offsets = [float(row["signed_offset_um"]) for row in ordered if row["classification"] == "right"]
    if left_offsets and right_offsets and max(left_offsets) < min(right_offsets):
        return float(0.5 * (max(left_offsets) + min(right_offsets)))
    return None


def interpolate_grid_velocity(field: GridField, grid: CommonGrid, point_um: np.ndarray) -> np.ndarray:
    x, y = float(point_um[0]), float(point_um[1])
    if x < grid.x_um[0] or x > grid.x_um[-1] or y < grid.y_um[0] or y > grid.y_um[-1]:
        return np.array([np.nan, np.nan])
    ix = int(np.searchsorted(grid.x_um, x) - 1)
    iy = int(np.searchsorted(grid.y_um, y) - 1)
    ix = int(np.clip(ix, 0, len(grid.x_um) - 2))
    iy = int(np.clip(iy, 0, len(grid.y_um) - 2))
    x0, x1 = grid.x_um[ix], grid.x_um[ix + 1]
    y0, y1 = grid.y_um[iy], grid.y_um[iy + 1]
    tx = (x - x0) / max(x1 - x0, 1.0e-30)
    ty = (y - y0) / max(y1 - y0, 1.0e-30)
    values = np.array(
        [
            [field.ux_m_per_s[iy, ix], field.uy_m_per_s[iy, ix]],
            [field.ux_m_per_s[iy, ix + 1], field.uy_m_per_s[iy, ix + 1]],
            [field.ux_m_per_s[iy + 1, ix], field.uy_m_per_s[iy + 1, ix]],
            [field.ux_m_per_s[iy + 1, ix + 1], field.uy_m_per_s[iy + 1, ix + 1]],
        ]
    )
    if not np.isfinite(values).all():
        return np.array([np.nan, np.nan])
    return (1 - tx) * (1 - ty) * values[0] + tx * (1 - ty) * values[1] + (1 - tx) * ty * values[2] + tx * ty * values[3]


def run_field_verification(
    output_dir: str | Path = "outputs/physics/full_device_cfd/convergence/field_comparison_24_vs_12",
    grid_spacing_um: float = 6.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geometry = _build_geometry()
    mesh24, data24, source24 = _load_or_solve_case(24.0, geometry, Path("outputs/physics/full_device_cfd/alpha0_equal_split_smoke/stokes_solution.npz"))
    mesh12, data12, source12 = _load_or_solve_case(12.0, geometry, output / "native_12um_solution.npz")
    grid = build_common_grid(geometry, grid_spacing_um)
    masks = region_masks(grid, geometry)
    field24 = evaluate_solution_on_grid(mesh24.nodes_um, mesh24.elements, data24["velocity_dofs_m_per_s"], grid)
    field12 = evaluate_solution_on_grid(mesh12.nodes_um, mesh12.elements, data12["velocity_dofs_m_per_s"], grid)
    metrics = vector_error_metrics(field24, field12, masks)
    sep24 = classify_separatrix(field24, grid, geometry)
    sep12 = classify_separatrix(field12, grid, geometry)
    sep_diff = _separatrix_difference(sep24, sep12, geometry.channel_width_um)
    flux_audit = {
        "method": "boundary facets integrated from the FEM P2 velocity solution using Simpson quadrature at facet endpoints and midpoint",
        "units": "m^2/s depth-integrated 2D flux per channel height",
        "case_24um": _flux_summary(mesh24, data24),
        "case_12um": _flux_summary(mesh12, data12),
    }
    summary = _summary(metrics, sep_diff, data24, data12, mesh24, mesh12, source24, source12, grid, geometry, flux_audit, time.perf_counter() - start)
    _save_common_outputs(output, grid, field24, field12, masks, metrics, summary, sep24, sep12, sep_diff, flux_audit)
    return summary


def velocity_basis(nodes_um: np.ndarray, elements: np.ndarray) -> Basis:
    return Basis(MeshTri(np.asarray(nodes_um).T * UM_TO_M, np.asarray(elements, dtype=np.int64).T), ElementVector(ElementTriP2()), intorder=4)


def paired_velocity_to_basis_coefficients(basis: Basis, velocity_dof_m_per_s: np.ndarray) -> np.ndarray:
    xidx, yidx = basis.split_indices()
    values = np.asarray(velocity_dof_m_per_s, dtype=float)
    if values.shape != (len(xidx), 2):
        raise ValueError(f"Expected paired velocity shape {(len(xidx), 2)}, got {values.shape}")
    coeff = np.zeros(basis.N)
    coeff[xidx] = values[:, 0]
    coeff[yidx] = values[:, 1]
    return coeff


def evaluate_velocity_basis(basis: Basis, coeff: np.ndarray, points_um: np.ndarray) -> np.ndarray:
    interp = basis.interpolator(coeff)
    pts_m = np.asarray(points_um).T * UM_TO_M
    out = np.full((pts_m.shape[1], 2), np.nan)
    for idx in range(pts_m.shape[1]):
        try:
            val = np.asarray(interp(pts_m[:, idx : idx + 1]), dtype=float)
            out[idx] = val.reshape(2, -1)[:, 0]
        except ValueError:
            continue
    return out


def _build_geometry() -> FullDeviceCFDGeometry:
    from .domain import build_full_device_cfd_geometry

    return build_full_device_cfd_geometry()


def _load_or_solve_case(target_size_um: float, geometry: FullDeviceCFDGeometry, path: Path) -> tuple[FullDeviceMesh, dict[str, np.ndarray], str]:
    if path.exists():
        data = dict(np.load(path))
        mesh = _mesh_from_arrays(data["nodes_um"], data["elements"], geometry)
        return mesh, data, str(path)
    mesh = generate_full_device_mesh(geometry, target_size_um=target_size_um, boundary_size_um=0.5 * target_size_um)
    solution = solve_full_device_stokes(
        mesh,
        target_left_fraction=0.5,
        alpha_left_pa_s_per_m2=0.0,
        alpha_right_pa_s_per_m2=0.0,
        case_id=f"alpha0_mesh_{target_size_um:g}um_field_reference",
    )
    np.savez_compressed(
        path,
        nodes_um=mesh.nodes_um,
        elements=mesh.elements,
        velocity_nodes_m_per_s=solution.velocity_node_m_per_s,
        pressure_nodes_pa=solution.pressure_node_pa,
        velocity_dof_coordinates_um=solution.velocity_dof_coordinates_um,
        velocity_dofs_m_per_s=solution.velocity_dof_m_per_s,
        pressure_dof_coordinates_um=solution.pressure_dof_coordinates_um,
        pressure_dofs_pa=solution.pressure_dof_pa,
        alpha_left_pa_s_per_m2=np.array([0.0]),
        alpha_right_pa_s_per_m2=np.array([0.0]),
    )
    return mesh, dict(np.load(path)), f"{path} (regenerated because saved convergence fields were incomplete)"


def _mesh_from_arrays(nodes_um: np.ndarray, elements: np.ndarray, geometry: FullDeviceCFDGeometry) -> FullDeviceMesh:
    facets = label_boundary_facets(nodes_um, elements, geometry)
    nodes = np.unique(np.concatenate([edges.ravel() for edges in facets.values() if len(edges)])).astype(np.int64)
    labels = {name: np.unique(edges.ravel()).astype(np.int64) if len(edges) else np.array([], dtype=np.int64) for name, edges in facets.items()}
    return FullDeviceMesh(nodes_um, elements, geometry, nodes, labels, facets, 0.0)


def _flux_summary(mesh: FullDeviceMesh, data: dict[str, np.ndarray]) -> dict[str, float]:
    basis = velocity_basis(mesh.nodes_um, mesh.elements)
    coeff = paired_velocity_to_basis_coefficients(basis, data["velocity_dofs_m_per_s"])
    flux = full_device_fluxes(mesh, basis, coeff)
    return {
        **{key: float(value) for key, value in flux.items()},
        "mass_mismatch_inlet_outlet_m2_per_s": float(flux["inlet"] + flux["outlet"]),
        "relative_mass_mismatch": float(abs((flux["inlet"] + flux["outlet"]) / flux["inlet"])),
    }


def _save_common_outputs(
    output: Path,
    grid: CommonGrid,
    field24: GridField,
    field12: GridField,
    masks: dict[str, np.ndarray],
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    sep24: dict[str, Any],
    sep12: dict[str, Any],
    sep_diff: dict[str, Any],
    flux_audit: dict[str, Any],
) -> None:
    np.savez_compressed(output / "common_grid_coordinates.npz", x_um=grid.x_um, y_um=grid.y_um, xx_um=grid.xx_um, yy_um=grid.yy_um, inside=grid.inside.reshape(grid.xx_um.shape))
    np.savez_compressed(output / "velocity_24um_common_grid.npz", ux_m_per_s=field24.ux_m_per_s, uy_m_per_s=field24.uy_m_per_s, speed_m_per_s=field24.speed_m_per_s)
    np.savez_compressed(output / "velocity_12um_common_grid.npz", ux_m_per_s=field12.ux_m_per_s, uy_m_per_s=field12.uy_m_per_s, speed_m_per_s=field12.speed_m_per_s)
    _write_metrics_csv(output / "field_comparison_metrics.csv", metrics)
    (output / "field_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "separatrix_comparison.json").write_text(json.dumps({"case_24um": _json_sep(sep24), "case_12um": _json_sep(sep12), "difference": sep_diff}, indent=2), encoding="utf-8")
    (output / "flux_integration_audit.json").write_text(json.dumps(flux_audit, indent=2), encoding="utf-8")
    _save_figures(output, grid, field24, field12, masks)


def _save_figures(output: Path, grid: CommonGrid, field24: GridField, field12: GridField, masks: dict[str, np.ndarray]) -> None:
    vmax = float(np.nanpercentile(np.r_[field24.speed_m_per_s.ravel(), field12.speed_m_per_s.ravel()], 99.5))
    diff = np.linalg.norm(np.dstack([field24.ux_m_per_s - field12.ux_m_per_s, field24.uy_m_per_s - field12.uy_m_per_s]), axis=2)
    angle = angular_error_degrees(_flat_velocity(field24), _flat_velocity(field12)).reshape(grid.xx_um.shape)
    _plot_scalar(grid, field24.speed_m_per_s, output / "velocity_magnitude_24um.png", "24 um velocity magnitude (m/s)", vmin=0.0, vmax=vmax, cmap="magma")
    _plot_scalar(grid, field12.speed_m_per_s, output / "velocity_magnitude_12um.png", "12 um velocity magnitude (m/s)", vmin=0.0, vmax=vmax, cmap="magma")
    _plot_scalar(grid, diff, output / "absolute_vector_error_24_vs_12.png", "absolute vector error (m/s)", vmin=0.0, vmax=float(np.nanpercentile(diff, 99.0)), cmap="viridis")
    _plot_scalar(grid, angle, output / "angular_error_24_vs_12.png", "angular error (deg)", vmin=0.0, vmax=float(np.nanpercentile(angle, 99.0)), cmap="plasma")
    _plot_zoom(grid, field24, field12, diff, masks["inlet_junction"].reshape(grid.xx_um.shape), output / "inlet_junction_error_zoom.png", "Inlet split junction error")
    _plot_zoom(grid, field24, field12, diff, masks["outlet_junction"].reshape(grid.xx_um.shape), output / "outlet_junction_error_zoom.png", "Outlet merge junction error")


def _plot_scalar(grid: CommonGrid, values: np.ndarray, path: Path, title: str, *, vmin: float, vmax: float, cmap: str) -> None:
    masked = np.where(grid.inside.reshape(grid.xx_um.shape), values, np.nan)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.pcolormesh(grid.xx_um, grid.yy_um, masked, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_zoom(grid: CommonGrid, field24: GridField, field12: GridField, diff: np.ndarray, mask2d: np.ndarray, path: Path, title: str) -> None:
    ys, xs = np.where(mask2d)
    pad = 4
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, grid.yy_um.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, grid.xx_um.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, label in zip(axes, [field12.speed_m_per_s, field24.speed_m_per_s, diff], ["12 um speed", "24 um speed", "error"]):
        local = np.where(mask2d[y0:y1, x0:x1], values[y0:y1, x0:x1], np.nan)
        im = ax.pcolormesh(grid.xx_um[y0:y1, x0:x1], grid.yy_um[y0:y1, x0:x1], local, shading="auto", cmap="magma" if label != "error" else "viridis")
        fig.colorbar(im, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary(
    metrics: list[dict[str, Any]],
    sep_diff: dict[str, Any],
    data24: dict[str, np.ndarray],
    data12: dict[str, np.ndarray],
    mesh24: FullDeviceMesh,
    mesh12: FullDeviceMesh,
    source24: str,
    source12: str,
    grid: CommonGrid,
    geometry: FullDeviceCFDGeometry,
    flux_audit: dict[str, Any],
    runtime_s: float,
) -> dict[str, Any]:
    by_region = {row["region"]: row for row in metrics}
    scalar = {
        "max_velocity_difference_m_per_s": float(abs(np.nanmax(np.linalg.norm(data24["velocity_nodes_m_per_s"], axis=1)) - np.nanmax(np.linalg.norm(data12["velocity_nodes_m_per_s"], axis=1)))),
        "pressure_drop_difference_pa": float(abs(np.nanmax(data24["pressure_nodes_pa"]) - np.nanmin(data24["pressure_nodes_pa"]) - (np.nanmax(data12["pressure_nodes_pa"]) - np.nanmin(data12["pressure_nodes_pa"])))),
    }
    accepted = (
        by_region["full_domain"]["vector_l2_relative_error"] <= 0.02
        and by_region["inlet_junction"]["vector_l2_relative_error"] <= 0.02
        and by_region["outlet_junction"]["vector_l2_relative_error"] <= 0.02
        and by_region["full_domain"]["median_angular_error_deg"] <= 1.0
        and by_region["full_domain"]["p95_angular_error_deg"] <= 3.0
        and sep_diff["normalized_by_channel_width"] is not None
        and sep_diff["normalized_by_channel_width"] <= 0.02
    )
    return {
        "candidate_mesh_target_um": 24.0,
        "reference_mesh_target_um": 12.0,
        "grid_spacing_um": grid.spacing_um,
        "grid_valid_points": int(np.count_nonzero(grid.inside)),
        "region_bounds_um": _region_bounds(geometry),
        "solution_sources": {"case_24um": source24, "case_12um": source12},
        "mesh_quality": {"case_24um": asdict(evaluate_full_device_mesh(mesh24)), "case_12um": asdict(evaluate_full_device_mesh(mesh12))},
        "metrics_by_region": by_region,
        "separatrix_difference": sep_diff,
        "scalar_differences": scalar,
        "flux_integration_method": flux_audit["method"],
        "facet_quadrature_mass_mismatch": {
            "case_24um": flux_audit["case_24um"]["relative_mass_mismatch"],
            "case_12um": flux_audit["case_12um"]["relative_mass_mismatch"],
        },
        "acceptance_criteria_passed": bool(accepted),
        "recommendation": "freeze_24um" if accepted else "use_24um_globally_with_local_junction_refinement",
        "runtime_s": float(runtime_s),
    }


def _separatrix_difference(sep24: dict[str, Any], sep12: dict[str, Any], channel_width_um: float) -> dict[str, Any]:
    a = sep24["transition_signed_offset_um"]
    b = sep12["transition_signed_offset_um"]
    if a is None or b is None:
        return {"absolute_difference_um": None, "normalized_by_channel_width": None}
    diff = abs(float(a) - float(b))
    return {"absolute_difference_um": diff, "normalized_by_channel_width": diff / channel_width_um}


def _branch_reached(point: np.ndarray, geometry: FullDeviceCFDGeometry) -> str:
    for branch in ("left", "right"):
        proj = project_to_centerline(point[None, :], geometry.centerlines[branch], geometry.channel_width_um)
        if bool(proj.inside[0]) and proj.s_um[0] > 1.5 * geometry.channel_width_um:
            return branch
    return "none"


def _rect_mask(points: np.ndarray, center: np.ndarray, hx: float, hy: float) -> np.ndarray:
    return (np.abs(points[:, 0] - center[0]) <= hx) & (np.abs(points[:, 1] - center[1]) <= hy)


def _flat_velocity(field: GridField) -> np.ndarray:
    return np.column_stack([field.ux_m_per_s.ravel(), field.uy_m_per_s.ravel()])


def _safe_stat(values: np.ndarray, fn) -> float:
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if len(finite) else float("nan")


def _safe_percentile(values: np.ndarray, pct: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, pct)) if len(finite) else float("nan")


def _write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def _region_bounds(geometry: FullDeviceCFDGeometry) -> dict[str, dict[str, float]]:
    hx, hy = 2.6 * geometry.channel_width_um, 2.2 * geometry.channel_width_um
    return {
        "inlet_junction": _bounds(geometry.upper_junction_um, hx, hy),
        "outlet_junction": _bounds(geometry.lower_junction_um, hx, hy),
    }


def _bounds(center: np.ndarray, hx: float, hy: float) -> dict[str, float]:
    return {"x_min_um": float(center[0] - hx), "x_max_um": float(center[0] + hx), "y_min_um": float(center[1] - hy), "y_max_um": float(center[1] + hy)}


def _json_sep(sep: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sep.items() if key != "traces"}
