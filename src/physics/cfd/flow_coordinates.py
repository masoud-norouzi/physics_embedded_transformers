from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, LinearForm, MeshTri, asm, condense, solve
from skfem.helpers import dot, grad

from src.physics.full_device_cfd.domain import FullDeviceCFDGeometry, inside_full_device_domain
from src.physics.full_device_cfd.mesh import FullDeviceMesh
from src.physics.interpolation.types import VelocityFieldCase

from .solver import UM_TO_M
from .streamlines import StreamlineTrace, TriangleVelocityField, trace_streamline_time


FLOW_COORDINATE_VERSION = "flow_coordinates_v1"


@dataclass(frozen=True)
class FlowCoordinateBuildConfig:
    seed_count: int = 121
    wall_margin_fraction: float = 0.08
    inward_offset_um: float = 8.0
    max_step_um: float = 4.0
    max_time_s: float = 10.0
    max_steps: int = 5000
    low_speed_um_per_s: float = 1.0
    min_trace_points: int = 4
    interpolant_fill: str = "nearest"


@dataclass(frozen=True)
class SampledFlowCoordinates:
    points_um: np.ndarray
    psi: np.ndarray
    n: np.ndarray
    time_s: np.ndarray
    arc_length_um: np.ndarray
    valid: np.ndarray


class FlowCoordinateMap:
    """Lookup map from device coordinates to flow-aligned psi/time coordinates."""

    def __init__(
        self,
        *,
        nodes_um: np.ndarray,
        elements: np.ndarray,
        psi_nodes: np.ndarray,
        psi_center: float,
        sample_points_um: np.ndarray,
        sample_time_s: np.ndarray,
        sample_arc_length_um: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        self.nodes_um = np.asarray(nodes_um, dtype=float)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.psi_nodes = np.asarray(psi_nodes, dtype=float)
        self.psi_center = float(psi_center)
        self.sample_points_um = np.asarray(sample_points_um, dtype=float)
        self.sample_time_s = np.asarray(sample_time_s, dtype=float)
        self.sample_arc_length_um = np.asarray(sample_arc_length_um, dtype=float)
        self.metadata = dict(metadata)
        self._psi_linear = LinearNDInterpolator(self.nodes_um, self.psi_nodes, fill_value=np.nan)
        self._psi_nearest = NearestNDInterpolator(self.nodes_um, self.psi_nodes)
        self._time_linear = LinearNDInterpolator(self.sample_points_um, self.sample_time_s, fill_value=np.nan)
        self._time_nearest = NearestNDInterpolator(self.sample_points_um, self.sample_time_s)
        self._arc_linear = LinearNDInterpolator(self.sample_points_um, self.sample_arc_length_um, fill_value=np.nan)
        self._arc_nearest = NearestNDInterpolator(self.sample_points_um, self.sample_arc_length_um)

    def sample(self, points_um: np.ndarray, geometry: FullDeviceCFDGeometry | None = None) -> SampledFlowCoordinates:
        points = np.asarray(points_um, dtype=float)
        if points.ndim == 1 and points.shape == (2,):
            points = points.reshape(1, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points_um must have shape (N, 2), got {points.shape}")
        psi = _linear_with_nearest_fill(self._psi_linear, self._psi_nearest, points)
        time = _linear_with_nearest_fill(self._time_linear, self._time_nearest, points)
        arc = _linear_with_nearest_fill(self._arc_linear, self._arc_nearest, points)
        finite = np.isfinite(psi) & np.isfinite(time) & np.isfinite(arc)
        if geometry is not None:
            finite &= inside_full_device_domain(points, geometry)
        return SampledFlowCoordinates(
            points_um=points,
            psi=psi,
            n=psi - self.psi_center,
            time_s=time,
            arc_length_um=arc,
            valid=finite,
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            nodes_um=self.nodes_um,
            elements=self.elements,
            psi_nodes=self.psi_nodes,
            psi_center=np.asarray([self.psi_center]),
            sample_points_um=self.sample_points_um,
            sample_time_s=self.sample_time_s,
            sample_arc_length_um=self.sample_arc_length_um,
            metadata_json=np.asarray(json.dumps(self.metadata, indent=2)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlowCoordinateMap":
        with np.load(path, allow_pickle=False) as loaded:
            metadata = json.loads(str(loaded["metadata_json"]))
            return cls(
                nodes_um=loaded["nodes_um"],
                elements=loaded["elements"],
                psi_nodes=loaded["psi_nodes"],
                psi_center=float(loaded["psi_center"][0]),
                sample_points_um=loaded["sample_points_um"],
                sample_time_s=loaded["sample_time_s"],
                sample_arc_length_um=loaded["sample_arc_length_um"],
                metadata=metadata,
            )


def build_flow_coordinate_map(
    case: VelocityFieldCase,
    *,
    config: FlowCoordinateBuildConfig | None = None,
) -> tuple[FlowCoordinateMap, list[StreamlineTrace]]:
    cfg = config or FlowCoordinateBuildConfig()
    mesh = case.mesh
    geometry = mesh.geometry
    if not isinstance(mesh, FullDeviceMesh) or not isinstance(geometry, FullDeviceCFDGeometry):
        raise TypeError("Flow-coordinate maps currently require a full-device CFD case")
    psi_nodes = solve_stream_function_poisson(
        case.nodes_um,
        case.elements,
        case.velocity_dof_m_per_s,
        case.velocity_node_m_per_s,
        boundary_facets=mesh.boundary_facets,
        geometry=geometry,
        fluxes_m2_per_s=_case_fluxes(case),
        left_fraction=case.left_fraction,
    )
    seeds = inlet_seed_points(geometry, cfg.seed_count, cfg.wall_margin_fraction, cfg.inward_offset_um)
    field = TriangleVelocityField(case.nodes_um, case.elements, case.velocity_node_m_per_s / UM_TO_M)
    traces = [
        trace_streamline_time(
            seed_id,
            seed,
            field,
            inside=lambda pts: inside_full_device_domain(pts, geometry),
            max_step_um=cfg.max_step_um,
            max_time_s=cfg.max_time_s,
            max_steps=cfg.max_steps,
            low_speed_um_per_s=cfg.low_speed_um_per_s,
        )
        for seed_id, seed in enumerate(seeds)
    ]
    sample_points, sample_time, sample_arc = streamline_samples(traces, min_trace_points=cfg.min_trace_points)
    psi_center = float(np.interp(0.0, inlet_seed_offsets(geometry, cfg.seed_count, cfg.wall_margin_fraction), psi_at_points(case.nodes_um, psi_nodes, seeds)))
    metadata = {
        "version": FLOW_COORDINATE_VERSION,
        "case_id": case.case_id,
        "left_fraction": float(case.left_fraction),
        "right_fraction": float(case.right_fraction),
        "psi_method": "FEM Poisson stream-function solve: laplacian(psi) = -vorticity, with one constant Dirichlet value per solid wall component; offsets set by CFD branch fluxes",
        "time_method": "RK4 integration of dx/dt = v(x), seeded across the inlet",
        "units": {
            "position": "um in device Cartesian frame",
            "velocity": "m/s",
            "psi": "m^2/s up to an arbitrary additive gauge",
            "time": "s",
            "arc_length": "um",
        },
        "build_config": cfg.__dict__,
        "trace_count": int(len(traces)),
        "trace_termination_counts": _termination_counts(traces),
        "trace_point_count": int(len(sample_points)),
        "time_stats_s": _stats(sample_time),
        "arc_length_stats_um": _stats(sample_arc),
    }
    return (
        FlowCoordinateMap(
            nodes_um=case.nodes_um,
            elements=case.elements,
            psi_nodes=psi_nodes,
            psi_center=psi_center,
            sample_points_um=sample_points,
            sample_time_s=sample_time,
            sample_arc_length_um=sample_arc,
            metadata=metadata,
        ),
        traces,
    )


def solve_stream_function_poisson(
    nodes_um: np.ndarray,
    elements: np.ndarray,
    velocity_dof_m_per_s: np.ndarray,
    velocity_node_m_per_s: np.ndarray,
    *,
    boundary_facets: dict[str, np.ndarray],
    geometry: FullDeviceCFDGeometry,
    fluxes_m2_per_s: dict[str, float],
    left_fraction: float,
) -> np.ndarray:
    """Solve laplacian(psi) = -omega on a P1 scalar space."""
    nodes = np.asarray(nodes_um, dtype=float)
    elements = np.asarray(elements, dtype=np.int64)
    velocity_nodes = np.asarray(velocity_node_m_per_s, dtype=float)
    velocity_dofs = np.asarray(velocity_dof_m_per_s, dtype=float)
    if velocity_nodes.shape != nodes.shape:
        raise ValueError(f"velocity_node_m_per_s must have shape {nodes.shape}, got {velocity_nodes.shape}")
    mesh = MeshTri(nodes.T * UM_TO_M, elements.T)
    basis = Basis(mesh, ElementTriP1(), intorder=4)
    vector_basis = Basis(mesh, ElementVector(ElementTriP2()), intorder=4)
    vector_coeff = _p2_vector_coefficients(vector_basis, velocity_dofs)

    @BilinearForm
    def laplace(u, v, _w):
        return dot(grad(u), grad(v))

    @LinearForm
    def rhs(v, w):
        vel = w["velocity"]
        omega = vel.grad[1, 0] - vel.grad[0, 1]
        return omega * v

    matrix = asm(laplace, basis)
    vector = asm(rhs, basis, velocity=vector_basis.interpolate(vector_coeff))
    boundary_values = stream_function_boundary_values(nodes, boundary_facets, geometry, fluxes_m2_per_s, left_fraction)
    boundary_nodes = np.flatnonzero(np.isfinite(boundary_values)).astype(np.int64)
    if len(boundary_nodes) == 0:
        raise ValueError("Cannot solve stream-function Poisson problem without boundary Dirichlet nodes")
    values = np.zeros(basis.N, dtype=float)
    values[boundary_nodes] = boundary_values[boundary_nodes]
    solution = solve(*condense(matrix, vector, x=values, D=boundary_nodes))
    solution = np.asarray(solution, dtype=float)
    solution -= float(np.nanmin(solution[boundary_nodes]))
    return solution


def stream_function_boundary_values(
    nodes_um: np.ndarray,
    boundary_facets: dict[str, np.ndarray],
    geometry: FullDeviceCFDGeometry,
    fluxes_m2_per_s: dict[str, float],
    left_fraction: float,
) -> np.ndarray:
    """Return exact constant Dirichlet psi values for each solid wall component."""
    nodes = np.asarray(nodes_um, dtype=float)
    wall_edges = np.asarray(boundary_facets.get("wall", np.empty((0, 2), dtype=np.int64)), dtype=np.int64).reshape(-1, 2)
    components = _wall_components(wall_edges)
    values = np.full(len(nodes), np.nan, dtype=float)
    if len(components) != 3:
        raise ValueError(f"Expected exactly three full-device wall components, got {len(components)}")
    constants = _full_device_wall_constants(nodes, components, geometry, fluxes_m2_per_s, left_fraction)
    for component_index, constant in constants.items():
        values[components[component_index]] = float(constant)
    if not np.isfinite(values).any():
        raise ValueError("No stream-function wall boundary values could be assigned")
    return values


def inlet_seed_points(
    geometry: FullDeviceCFDGeometry,
    seed_count: int,
    wall_margin_fraction: float,
    inward_offset_um: float,
) -> np.ndarray:
    offsets = inlet_seed_offsets(geometry, seed_count, wall_margin_fraction)
    inlet = geometry.centerlines["inlet"]
    center = geometry.inlet_cut_center_um + float(inward_offset_um) * inlet.tangents[0]
    return center + offsets[:, None] * inlet.normals[0]


def inlet_seed_offsets(geometry: FullDeviceCFDGeometry, seed_count: int, wall_margin_fraction: float) -> np.ndarray:
    if seed_count < 2:
        raise ValueError("seed_count must be at least 2")
    if not 0.0 <= wall_margin_fraction < 0.5:
        raise ValueError("wall_margin_fraction must satisfy 0 <= margin < 0.5")
    half = geometry.half_width_um * (1.0 - float(wall_margin_fraction))
    return np.linspace(-half, half, int(seed_count))


def streamline_samples(
    traces: Iterable[StreamlineTrace],
    *,
    min_trace_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_parts = []
    time_parts = []
    arc_parts = []
    for trace in traces:
        if len(trace.points_um) < min_trace_points:
            continue
        point_parts.append(trace.points_um)
        time_parts.append(trace.elapsed_s)
        arc_parts.append(trace.arc_length_um)
    if not point_parts:
        raise ValueError("No streamline traces had enough points to build a coordinate map")
    points = np.vstack(point_parts)
    time = np.concatenate(time_parts)
    arc = np.concatenate(arc_parts)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(time) & np.isfinite(arc)
    if not np.any(finite):
        raise ValueError("Streamline samples are all non-finite")
    rounded, keep = np.unique(np.round(points[finite], decimals=9), axis=0, return_index=True)
    return rounded, time[finite][keep], arc[finite][keep]


def psi_at_points(nodes_um: np.ndarray, psi_nodes: np.ndarray, points_um: np.ndarray) -> np.ndarray:
    linear = LinearNDInterpolator(np.asarray(nodes_um, dtype=float), np.asarray(psi_nodes, dtype=float), fill_value=np.nan)
    nearest = NearestNDInterpolator(np.asarray(nodes_um, dtype=float), np.asarray(psi_nodes, dtype=float))
    return _linear_with_nearest_fill(linear, nearest, np.asarray(points_um, dtype=float))


def save_flow_coordinate_diagnostics(
    coord_map: FlowCoordinateMap,
    traces: list[StreamlineTrace],
    geometry: FullDeviceCFDGeometry,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tri = mtri.Triangulation(coord_map.nodes_um[:, 0], coord_map.nodes_um[:, 1], coord_map.elements)
    _save_tripcolor(tri, coord_map.psi_nodes, out / "psi_nodes.png", "stream function psi")
    _save_scatter_field(
        coord_map.sample_points_um,
        coord_map.sample_time_s,
        out / "advective_time_samples.png",
        "advective time T (s)",
        robust=True,
    )
    _save_trace_overlay(coord_map, traces, geometry, out / "streamline_coordinate_traces.png")
    (out / "flow_coordinate_metadata.json").write_text(json.dumps(coord_map.metadata, indent=2), encoding="utf-8")


def _wall_components(edges: np.ndarray) -> list[np.ndarray]:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return []
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    seen: set[int] = set()
    components = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def _full_device_wall_constants(
    nodes_um: np.ndarray,
    components: list[np.ndarray],
    geometry: FullDeviceCFDGeometry,
    fluxes_m2_per_s: dict[str, float],
    left_fraction: float,
) -> dict[int, float]:
    inner_index = _inner_wall_component_index(nodes_um, components, geometry)
    outer = [idx for idx in range(len(components)) if idx != inner_index]
    if len(outer) != 2:
        raise ValueError("Full-device wall constants require two outer wall components and one inner island component")
    outer_sorted = sorted(outer, key=lambda idx: float(np.mean(nodes_um[components[idx], 0])))
    left_outer, right_outer = outer_sorted
    left_flux, right_flux = _branch_fluxes(fluxes_m2_per_s, left_fraction)
    return {
        left_outer: 0.0,
        inner_index: float(left_flux),
        right_outer: float(left_flux + right_flux),
    }


def _inner_wall_component_index(
    nodes_um: np.ndarray,
    components: list[np.ndarray],
    geometry: FullDeviceCFDGeometry,
) -> int:
    inner_centroid = np.mean(geometry.inner_ring_um, axis=0)
    distances = [
        float(np.linalg.norm(np.mean(nodes_um[component], axis=0) - inner_centroid))
        for component in components
    ]
    return int(np.argmin(distances))


def _branch_fluxes(fluxes_m2_per_s: dict[str, float], left_fraction: float) -> tuple[float, float]:
    left = _first_finite_abs(fluxes_m2_per_s, ("left_branch", "left_outlet"))
    right = _first_finite_abs(fluxes_m2_per_s, ("right_branch", "right_outlet"))
    if left is None or right is None:
        inlet = _first_finite_abs(fluxes_m2_per_s, ("inlet",))
        if inlet is None:
            raise ValueError("Cannot determine stream-function wall offsets without branch or inlet fluxes")
        left = float(inlet) * float(left_fraction)
        right = float(inlet) * (1.0 - float(left_fraction))
    if left <= 0.0 or right <= 0.0:
        raise ValueError(f"Branch fluxes must be positive after abs(), got left={left}, right={right}")
    return float(left), float(right)


def _first_finite_abs(values: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in values:
            continue
        value = abs(float(values[key]))
        if np.isfinite(value) and value > 0.0:
            return value
    return None


def _case_fluxes(case: VelocityFieldCase) -> dict[str, float]:
    fluxes = case.metadata.get("fluxes_m2_per_s")
    if isinstance(fluxes, dict):
        return {str(key): float(value) for key, value in fluxes.items()}
    fluxes = case.flux_report.get("fluxes_m2_per_s")
    if isinstance(fluxes, dict):
        return {str(key): float(value) for key, value in fluxes.items()}
    report = case.metadata.get("report", {})
    if isinstance(report, dict):
        result = {}
        for source, target in (
            ("inlet_flux_signed_m2_per_s", "inlet"),
            ("left_outlet_flux_m2_per_s", "left_outlet"),
            ("right_outlet_flux_m2_per_s", "right_outlet"),
        ):
            if source in report:
                result[target] = float(report[source])
        if result:
            return result
    return {}


def _p2_vector_coefficients(vector_basis: Basis, velocity_dof_m_per_s: np.ndarray) -> np.ndarray:
    xidx, yidx = vector_basis.split_indices()
    coeff = np.zeros(vector_basis.N, dtype=float)
    values = np.asarray(velocity_dof_m_per_s, dtype=float)
    if values.shape != (len(xidx), 2):
        raise ValueError(f"velocity_dof_m_per_s must have shape {(len(xidx), 2)}, got {values.shape}")
    coeff[xidx] = values[:, 0]
    coeff[yidx] = values[:, 1]
    return coeff


def _linear_with_nearest_fill(linear, nearest, points: np.ndarray) -> np.ndarray:
    values = np.asarray(linear(points), dtype=float)
    missing = ~np.isfinite(values)
    if np.any(missing):
        values[missing] = np.asarray(nearest(points[missing]), dtype=float)
    return values


def _termination_counts(traces: Iterable[StreamlineTrace]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trace in traces:
        counts[trace.termination_reason] = counts.get(trace.termination_reason, 0) + 1
    return counts


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _save_tripcolor(tri: mtri.Triangulation, values: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.tripcolor(tri, values, shading="gouraud", cmap="viridis")
    fig.colorbar(image, ax=ax)
    ax.triplot(tri, linewidth=0.08, color="white", alpha=0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_scatter_field(points_um: np.ndarray, values: np.ndarray, path: Path, title: str, *, robust: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    kwargs = {}
    if robust:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            kwargs["vmin"] = float(np.min(finite))
            kwargs["vmax"] = float(np.percentile(finite, 99))
    image = ax.scatter(points_um[:, 0], points_um[:, 1], c=values, s=2, cmap="magma", **kwargs)
    fig.colorbar(image, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_trace_overlay(
    coord_map: FlowCoordinateMap,
    traces: list[StreamlineTrace],
    geometry: FullDeviceCFDGeometry,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(geometry.outer_ring_um[:, 0], geometry.outer_ring_um[:, 1], color="#111827", linewidth=0.8)
    ax.plot(geometry.inner_ring_um[:, 0], geometry.inner_ring_um[:, 1], color="#111827", linewidth=0.8)
    for trace in traces:
        if len(trace.points_um) < 2:
            continue
        ax.plot(trace.points_um[:, 0], trace.points_um[:, 1], linewidth=0.5, alpha=0.65)
    ax.scatter(coord_map.sample_points_um[:, 0], coord_map.sample_points_um[:, 1], s=0.5, color="#111827", alpha=0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title("inlet-seeded advective coordinate traces")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
