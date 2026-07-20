from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import triangle as tr

from .domain import FullDeviceCFDGeometry, build_full_device_cfd_geometry, inside_full_device_domain, resistance_weights


@dataclass(frozen=True)
class FullDeviceMesh:
    nodes_um: np.ndarray
    elements: np.ndarray
    geometry: FullDeviceCFDGeometry
    boundary_node_indices: np.ndarray
    boundary_labels: dict[str, np.ndarray]
    boundary_facets: dict[str, np.ndarray]
    generation_runtime_s: float


@dataclass(frozen=True)
class FullDeviceMeshQuality:
    nodes: int
    elements: int
    minimum_angle_deg: float
    maximum_angle_deg: float
    median_aspect_ratio: float
    maximum_aspect_ratio: float
    element_area_um2: dict[str, float]
    boundary_length_um: float
    representative_edge_lengths_um: dict[str, float]
    generation_runtime_s: float


def generate_full_device_mesh(
    geometry: FullDeviceCFDGeometry | None = None,
    *,
    target_size_um: float = 24.0,
    boundary_size_um: float = 12.0,
) -> FullDeviceMesh:
    start = time.perf_counter()
    geometry = geometry or build_full_device_cfd_geometry()
    pslg = _build_constrained_pslg(geometry, boundary_size_um)
    max_area = np.sqrt(3.0) / 4.0 * target_size_um**2
    result = tr.triangulate(pslg, f"pq25a{max_area:.8f}")
    if "triangles" not in result:
        raise RuntimeError("Constrained full-device triangulation did not return elements")
    nodes = np.asarray(result["vertices"], dtype=float)
    elements = np.asarray(result["triangles"], dtype=np.int64)
    centroids = nodes[elements].mean(axis=1)
    keep = inside_full_device_domain(centroids, geometry, tolerance_um=1.0e-9)
    elements = elements[keep]
    areas = _signed_triangle_areas(nodes[elements])
    elements = elements[np.abs(areas) > 1.0e-8]
    inverted = _signed_triangle_areas(nodes[elements]) < 0.0
    elements[inverted] = elements[inverted][:, [0, 2, 1]]
    boundary_facets = label_boundary_facets(nodes, elements, geometry)
    boundary_nodes = np.unique(np.concatenate([edges.ravel() for edges in boundary_facets.values() if len(edges)]))
    labels = {name: np.unique(edges.ravel()).astype(np.int64) if len(edges) else np.array([], dtype=np.int64) for name, edges in boundary_facets.items()}
    return FullDeviceMesh(nodes, elements, geometry, boundary_nodes.astype(np.int64), labels, boundary_facets, time.perf_counter() - start)


def label_boundary_facets(nodes_um: np.ndarray, elements: np.ndarray, geometry: FullDeviceCFDGeometry) -> dict[str, np.ndarray]:
    facets = {"inlet": [], "outlet": [], "wall": []}
    for edge in _boundary_edges(elements):
        midpoint = nodes_um[edge].mean(axis=0)
        label = classify_boundary_midpoint(midpoint, geometry)
        facets[label].append([int(edge[0]), int(edge[1])])
    return {k: np.asarray(v, dtype=np.int64).reshape((-1, 2)) for k, v in facets.items()}


def classify_boundary_midpoint(point_um: np.ndarray, geometry: FullDeviceCFDGeometry) -> str:
    inlet = geometry.centerlines["inlet"]
    outlet = geometry.centerlines["outlet"]
    inlet_dist = _cut_distance(point_um, geometry.inlet_cut_center_um, inlet.tangents[0], geometry.channel_width_um)
    outlet_dist = _cut_distance(point_um, geometry.outlet_cut_center_um, outlet.tangents[-1], geometry.channel_width_um)
    tol = max(geometry.channel_width_um * 0.15, 10.0)
    if inlet_dist < tol:
        return "inlet"
    if outlet_dist < tol:
        return "outlet"
    return "wall"


def evaluate_full_device_mesh(mesh: FullDeviceMesh) -> FullDeviceMeshQuality:
    vertices = mesh.nodes_um[mesh.elements]
    edges = _edge_lengths(vertices)
    angles = _triangle_angles(edges)
    aspect = np.max(edges, axis=1) / np.maximum(np.min(edges, axis=1), 1.0e-12)
    areas = np.abs(_signed_triangle_areas(vertices))
    return FullDeviceMeshQuality(
        nodes=int(len(mesh.nodes_um)),
        elements=int(len(mesh.elements)),
        minimum_angle_deg=float(np.min(angles)),
        maximum_angle_deg=float(np.max(angles)),
        median_aspect_ratio=float(np.median(aspect)),
        maximum_aspect_ratio=float(np.max(aspect)),
        element_area_um2={
            "min": float(np.min(areas)),
            "median": float(np.median(areas)),
            "mean": float(np.mean(areas)),
            "max": float(np.max(areas)),
            "total": float(np.sum(areas)),
        },
        boundary_length_um=float(sum(np.sum(np.linalg.norm(mesh.nodes_um[e[:, 1]] - mesh.nodes_um[e[:, 0]], axis=1)) for e in mesh.boundary_facets.values())),
        representative_edge_lengths_um=_representative_edges(mesh, edges),
        generation_runtime_s=float(mesh.generation_runtime_s),
    )


def save_full_device_mesh(mesh: FullDeviceMesh, output_dir: str | Path = "outputs/physics/full_device_cfd/mesh", overwrite: bool = False) -> FullDeviceMeshQuality:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    quality = evaluate_full_device_mesh(mesh)
    files = [out / "full_device_mesh.npz", out / "full_device_mesh_metadata.json", out / "mesh_quality_summary.json", out / "full_device_mesh.png"]
    if not overwrite and any(p.exists() for p in files):
        raise FileExistsError(f"Full-device mesh outputs already exist: {out}")
    np.savez(
        out / "full_device_mesh.npz",
        nodes_um=mesh.nodes_um,
        elements=mesh.elements,
        boundary_node_indices=mesh.boundary_node_indices,
        inlet_facets=mesh.boundary_facets["inlet"],
        outlet_facets=mesh.boundary_facets["outlet"],
        wall_facets=mesh.boundary_facets["wall"],
    )
    (out / "full_device_mesh_metadata.json").write_text(json.dumps(_mesh_metadata(mesh), indent=2), encoding="utf-8")
    (out / "mesh_quality_summary.json").write_text(json.dumps(asdict(quality), indent=2), encoding="utf-8")
    _save_mesh_plot(mesh, out / "full_device_mesh.png")
    return quality


def _sample_boundaries(geometry: FullDeviceCFDGeometry, spacing: float) -> np.ndarray:
    pts = [
        _resample(geometry.outer_ring_um, spacing),
        _resample(geometry.inner_ring_um, spacing),
        _resample(geometry.inlet_cut_um, spacing),
        _resample(geometry.outlet_cut_um, spacing),
    ]
    return np.vstack(pts)


def _build_constrained_pslg(geometry: FullDeviceCFDGeometry, spacing: float) -> dict[str, np.ndarray]:
    outer = _open_ring(_resample(geometry.outer_ring_um, spacing))
    inner = _open_ring(_resample(geometry.inner_ring_um, spacing))
    vertices = np.vstack([outer, inner])
    outer_segments = _ring_segments(0, len(outer))
    inner_segments = _ring_segments(len(outer), len(inner))
    return {
        "vertices": vertices,
        "segments": np.vstack([outer_segments, inner_segments]).astype(np.int64),
        "holes": np.asarray([np.mean(inner, axis=0)], dtype=float),
    }


def _open_ring(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1.0e-8:
        pts = pts[:-1]
    return _unique_rows_preserving_order(pts)


def _ring_segments(start: int, count: int) -> np.ndarray:
    indices = np.arange(start, start + count, dtype=np.int64)
    return np.column_stack([indices, np.roll(indices, -1)])


def _unique_rows_preserving_order(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values, decimals=8)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return values[np.sort(idx)]


def _sample_centerlines(geometry: FullDeviceCFDGeometry, spacing: float) -> np.ndarray:
    return np.vstack([_resample(line.points_um, spacing) for line in geometry.centerlines.values()])


def _sample_grid(geometry: FullDeviceCFDGeometry, spacing: float) -> np.ndarray:
    pts = geometry.all_centerline_points_um
    pad = geometry.channel_width_um
    x = np.arange(pts[:, 0].min() - pad, pts[:, 0].max() + pad + spacing, spacing)
    y = np.arange(pts[:, 1].min() - pad, pts[:, 1].max() + pad + spacing, spacing)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    near_junction = []
    for center in (geometry.upper_junction_um, geometry.lower_junction_um):
        local_spacing = spacing * 0.5
        lx = np.arange(center[0] - 1.6 * geometry.channel_width_um, center[0] + 1.6 * geometry.channel_width_um + local_spacing, local_spacing)
        ly = np.arange(center[1] - 1.6 * geometry.channel_width_um, center[1] + 1.6 * geometry.channel_width_um + local_spacing, local_spacing)
        lxx, lyy = np.meshgrid(lx, ly)
        near_junction.append(np.column_stack([lxx.ravel(), lyy.ravel()]))
    all_grid = np.vstack([grid, *near_junction])
    return all_grid[inside_full_device_domain(all_grid, geometry, tolerance_um=spacing * 0.2)]


def _cut_points(center: np.ndarray, tangent: np.ndarray, width: float, spacing: float) -> np.ndarray:
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    n = max(3, int(np.ceil(width / spacing)) + 1)
    d = np.linspace(-width / 2.0, width / 2.0, n)
    return center + d[:, None] * normal


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    samples = np.arange(0.0, s[-1] + spacing * 0.5, spacing)
    if samples[-1] < s[-1]:
        samples = np.append(samples, s[-1])
    return np.column_stack([np.interp(samples, s, points[:, 0]), np.interp(samples, s, points[:, 1])])


def _tangent(points: np.ndarray) -> np.ndarray:
    delta = np.empty_like(points)
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    delta[1:-1] = points[2:] - points[:-2]
    return delta / np.linalg.norm(delta, axis=1)[:, None]


def _cut_distance(point: np.ndarray, center: np.ndarray, tangent: np.ndarray, width: float) -> float:
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    rel = point - center
    along = abs(float(rel @ tangent))
    transverse = max(0.0, abs(float(rel @ normal)) - width / 2.0)
    return float(np.hypot(along, transverse))


def _unique_rows(values: np.ndarray) -> np.ndarray:
    rounded = np.round(values, decimals=8)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return values[np.sort(idx)]


def _boundary_edges(elements: np.ndarray) -> np.ndarray:
    edges = np.vstack([elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _signed_triangle_areas(vertices: np.ndarray) -> np.ndarray:
    ab = vertices[:, 1] - vertices[:, 0]
    ac = vertices[:, 2] - vertices[:, 0]
    return (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) / 2.0


def _edge_lengths(vertices: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
        np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
        np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
    ])


def _triangle_angles(edges: np.ndarray) -> np.ndarray:
    a, b, c = edges[:, 0], edges[:, 1], edges[:, 2]
    return np.degrees(np.column_stack([_angle(c, a, b), _angle(a, b, c), _angle(b, c, a)]))


def _angle(adj_a: np.ndarray, adj_b: np.ndarray, opp: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip((adj_a**2 + adj_b**2 - opp**2) / (2.0 * adj_a * adj_b), -1.0, 1.0))


def _representative_edges(mesh: FullDeviceMesh, edges: np.ndarray) -> dict[str, float]:
    centroids = mesh.nodes_um[mesh.elements].mean(axis=1)
    weights = resistance_weights(centroids, mesh.geometry)
    near_upper = np.linalg.norm(centroids - mesh.geometry.upper_junction_um, axis=1) < mesh.geometry.channel_width_um
    near_lower = np.linalg.norm(centroids - mesh.geometry.lower_junction_um, axis=1) < mesh.geometry.channel_width_um
    curved = (weights["left"] > 0.5) | (weights["right"] > 0.5)
    straight = ~(near_upper | near_lower | curved)
    median_edge = np.median(edges, axis=1)
    return {
        "inlet_junction": float(np.median(median_edge[near_upper])) if np.any(near_upper) else float("nan"),
        "outlet_junction": float(np.median(median_edge[near_lower])) if np.any(near_lower) else float("nan"),
        "curved_branches": float(np.median(median_edge[curved])) if np.any(curved) else float("nan"),
        "straight_channels": float(np.median(median_edge[straight])) if np.any(straight) else float("nan"),
    }


def _mesh_metadata(mesh: FullDeviceMesh) -> dict:
    return {
        "device_id": mesh.geometry.device_id,
        "coordinate_frame": mesh.geometry.coordinate_frame,
        "position_units": "um",
        "channel_width_um": mesh.geometry.channel_width_um,
        "channel_height_um": mesh.geometry.channel_height_um,
        "resistance_margin_um": mesh.geometry.resistance_margin_um,
        "resistance_ramp_um": mesh.geometry.resistance_ramp_um,
        "boundary_counts": {name: int(len(edges)) for name, edges in mesh.boundary_facets.items()},
        "source_paths": mesh.geometry.source_paths,
    }


def _save_mesh_plot(mesh: FullDeviceMesh, path: Path) -> None:
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.triplot(tri, linewidth=0.18, color="#1f2937")
    for line in mesh.geometry.centerlines.values():
        ax.plot(line.points_um[:, 0], line.points_um[:, 1], linewidth=1.0, label=line.branch)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title("Full-device CFD mesh")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
