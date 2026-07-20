from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .domain import (
    JunctionGeometry,
    build_junction_geometry,
    classify_boundary_points,
    distance_to_centerline_segments,
    inside_junction_domain,
)


@dataclass(frozen=True)
class TriangularMesh:
    """Triangular finite-element mesh in micrometer coordinates."""

    nodes_um: np.ndarray
    elements: np.ndarray
    geometry: JunctionGeometry
    boundary_node_indices: np.ndarray
    boundary_labels: dict[str, np.ndarray]
    boundary_facets: dict[str, np.ndarray]


@dataclass(frozen=True)
class MeshQualityReport:
    number_of_elements: int
    number_of_nodes: int
    minimum_angle_deg: float
    maximum_angle_deg: float
    mean_aspect_ratio: float
    maximum_aspect_ratio: float
    element_area_um2: dict[str, float]
    boundary_length_um: float
    estimated_hydraulic_domain_area_um2: float
    mesh_resolution_near_junction_um: dict[str, float]
    coordinate_units: str


@dataclass(frozen=True)
class MeshTopologyDiagnostics:
    connected_fluid_components: int
    domain_holes: int
    invalid_or_inverted_elements: int
    zero_area_elements: int
    exterior_boundary_facets: int
    interior_boundary_facets: int
    boundary_loop_count: int
    boundary_facet_counts: dict[str, int]
    defect_region_element_ids: list[int]
    white_region_classification: str


def generate_mesh(geometry: JunctionGeometry) -> TriangularMesh:
    """Generate a robust triangular mesh for the idealized junction patch."""
    target = geometry.target_element_size_um
    near = target * geometry.boundary_refinement_factor
    if near <= 0:
        raise ValueError("Refined element size must be positive")

    points = [_sample_grid_points(geometry, spacing_um=target)]
    points.append(_sample_offset_boundary_points(geometry, spacing_um=near))
    points.append(geometry.junction_center_um[None, :])
    nodes = _unique_rows(np.vstack(points), decimals=8)
    inside = inside_junction_domain(nodes, geometry, tolerance_um=near * 0.25)
    nodes = nodes[inside]

    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    triangles = triangulation.triangles
    centroids = nodes[triangles].mean(axis=1)
    keep = inside_junction_domain(centroids, geometry, tolerance_um=0.0)
    triangles = triangles[keep]
    positive_area = _triangle_areas(nodes[triangles]) > 1e-9
    triangles = triangles[positive_area]
    used = np.unique(triangles.ravel())
    remap = np.full(nodes.shape[0], -1, dtype=int)
    remap[used] = np.arange(len(used))
    compact_nodes = nodes[used]
    compact_triangles = remap[triangles]
    boundary_nodes = _estimate_boundary_nodes(compact_nodes, geometry, tolerance_um=max(near, 1e-9) * 0.8)
    boundary_labels = label_mesh_boundaries(compact_nodes, boundary_nodes, geometry)
    boundary_facets = label_mesh_boundary_facets(compact_nodes, compact_triangles, geometry)
    return TriangularMesh(
        nodes_um=compact_nodes,
        elements=compact_triangles.astype(np.int64),
        geometry=geometry,
        boundary_node_indices=boundary_nodes,
        boundary_labels=boundary_labels,
        boundary_facets=boundary_facets,
    )


def label_mesh_boundaries(
    nodes_um: np.ndarray,
    boundary_node_indices: np.ndarray,
    geometry: JunctionGeometry,
) -> dict[str, np.ndarray]:
    """Attach named boundary labels to mesh nodes."""
    labels: dict[str, list[int]] = {"inlet": [], "left_outlet": [], "right_outlet": [], "wall": []}
    classified_all = classify_boundary_points(nodes_um, geometry)
    for name in ("inlet", "left_outlet", "right_outlet"):
        labels[name] = np.flatnonzero(classified_all == name).astype(np.int64).tolist()
    endpoint_nodes = set(labels["inlet"] + labels["left_outlet"] + labels["right_outlet"])
    classified = classify_boundary_points(nodes_um[boundary_node_indices], geometry)
    for node_idx, label in zip(boundary_node_indices, classified):
        if label == "wall" and int(node_idx) not in endpoint_nodes:
            labels[str(label)].append(int(node_idx))
    return {name: np.asarray(sorted(set(indices)), dtype=np.int64) for name, indices in labels.items()}


def label_mesh_boundary_facets(nodes_um: np.ndarray, elements: np.ndarray, geometry: JunctionGeometry) -> dict[str, np.ndarray]:
    """Attach named labels to exterior mesh edges."""
    facets: dict[str, list[list[int]]] = {"inlet": [], "left_outlet": [], "right_outlet": [], "wall": []}
    for edge in _boundary_edges(elements):
        midpoint = nodes_um[edge].mean(axis=0, keepdims=True)
        label = str(classify_boundary_points(midpoint, geometry)[0])
        if label in facets:
            facets[label].append([int(edge[0]), int(edge[1])])
    return {name: np.asarray(edges, dtype=np.int64).reshape((-1, 2)) for name, edges in facets.items()}


def evaluate_mesh(mesh: TriangularMesh) -> MeshQualityReport:
    """Compute finite-element mesh quality metrics."""
    vertices = mesh.nodes_um[mesh.elements]
    edges = _triangle_edge_lengths(vertices)
    areas = _triangle_areas(vertices)
    angles = _triangle_angles(edges)
    aspect = _triangle_aspect_ratios(edges)
    centroids = vertices.mean(axis=1)
    near = np.linalg.norm(centroids - mesh.geometry.junction_center_um, axis=1) <= mesh.geometry.channel_width_um
    near_edges = edges[near].ravel() if np.any(near) else edges.ravel()
    return MeshQualityReport(
        number_of_elements=int(len(mesh.elements)),
        number_of_nodes=int(len(mesh.nodes_um)),
        minimum_angle_deg=float(np.min(angles)),
        maximum_angle_deg=float(np.max(angles)),
        mean_aspect_ratio=float(np.mean(aspect)),
        maximum_aspect_ratio=float(np.max(aspect)),
        element_area_um2={
            "minimum": float(np.min(areas)),
            "maximum": float(np.max(areas)),
            "mean": float(np.mean(areas)),
            "median": float(np.median(areas)),
            "total": float(np.sum(areas)),
        },
        boundary_length_um=_estimated_smooth_boundary_length(mesh.geometry),
        estimated_hydraulic_domain_area_um2=float(np.sum(areas)),
        mesh_resolution_near_junction_um={
            "minimum_edge_length": float(np.min(near_edges)),
            "mean_edge_length": float(np.mean(near_edges)),
            "median_edge_length": float(np.median(near_edges)),
            "maximum_edge_length": float(np.max(near_edges)),
        },
        coordinate_units="um",
    )


def save_mesh_outputs(mesh: TriangularMesh, report: MeshQualityReport, output_root: Path, overwrite: bool = False) -> None:
    geometry_dir = output_root / "geometry"
    mesh_dir = output_root / "meshes"
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"
    for path in (geometry_dir, mesh_dir, reports_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    outputs = [
        geometry_dir / "junction_geometry.json",
        mesh_dir / "junction_mesh.npz",
        reports_dir / "mesh_quality_report.json",
        reports_dir / "mesh_quality_report.md",
        figures_dir / "idealized_geometry.png",
        figures_dir / "centerline_overlay_geometry.png",
        figures_dir / "junction_mesh.png",
        figures_dir / "mesh_quality_aspect_ratio.png",
    ]
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist. Use --overwrite: {existing}")

    _save_geometry_json(mesh.geometry, geometry_dir / "junction_geometry.json")
    topology = evaluate_mesh_topology(mesh)
    np.savez(
        mesh_dir / "junction_mesh.npz",
        nodes_um=mesh.nodes_um,
        elements=mesh.elements,
        boundary_node_indices=mesh.boundary_node_indices,
        inlet_nodes=mesh.boundary_labels["inlet"],
        left_outlet_nodes=mesh.boundary_labels["left_outlet"],
        right_outlet_nodes=mesh.boundary_labels["right_outlet"],
        wall_nodes=mesh.boundary_labels["wall"],
        inlet_facets=mesh.boundary_facets["inlet"],
        left_outlet_facets=mesh.boundary_facets["left_outlet"],
        right_outlet_facets=mesh.boundary_facets["right_outlet"],
        wall_facets=mesh.boundary_facets["wall"],
    )
    report_dict = asdict(report)
    (reports_dir / "mesh_quality_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    (reports_dir / "mesh_quality_report.md").write_text(_markdown_report(report), encoding="utf-8")
    (reports_dir / "mesh_topology_diagnostics.json").write_text(
        json.dumps(asdict(topology), indent=2),
        encoding="utf-8",
    )
    (reports_dir / "mesh_topology_diagnostics.md").write_text(_topology_markdown_report(topology), encoding="utf-8")
    _save_geometry_figure(mesh.geometry, figures_dir / "idealized_geometry.png", overlay_centerline=False)
    _save_geometry_figure(mesh.geometry, figures_dir / "centerline_overlay_geometry.png", overlay_centerline=True)
    _save_mesh_figure(mesh, figures_dir / "junction_mesh.png")
    _save_quality_figure(mesh, figures_dir / "mesh_quality_aspect_ratio.png")
    _save_boundary_label_figure(mesh, figures_dir / "boundary_labels.png")


def evaluate_mesh_topology(mesh: TriangularMesh) -> MeshTopologyDiagnostics:
    """Diagnose mesh connectivity, holes, invalid elements, and former defect region."""
    signed_areas = _signed_triangle_areas(mesh.nodes_um[mesh.elements])
    invalid = np.flatnonzero(signed_areas <= -1e-9)
    zero = np.flatnonzero(np.abs(signed_areas) <= 1e-9)
    components = _element_components(mesh.elements)
    boundary_edges = _boundary_edges(mesh.elements)
    loops = _edge_graph_components(boundary_edges)
    loop_lengths = [_polyline_edge_length(mesh.nodes_um, loop) for loop in loops]
    exterior_loop_index = int(np.argmax(loop_lengths)) if loop_lengths else -1
    interior_facets = int(sum(len(loop) for i, loop in enumerate(loops) if i != exterior_loop_index))
    holes = max(0, len(loops) - len(components))
    centroids = mesh.nodes_um[mesh.elements].mean(axis=1)
    rel = centroids - mesh.geometry.junction_center_um
    defect_region = (
        (np.abs(rel[:, 0]) <= mesh.geometry.channel_width_um * 0.8)
        & (rel[:, 1] >= -mesh.geometry.channel_width_um * 0.1)
        & (rel[:, 1] <= mesh.geometry.channel_width_um * 0.9)
    )
    classification = (
        "No true hole, unmeshed region, invalid element, inverted element, or interpolation mask remains near "
        "the former lower-center construction seam; remaining white gaps in streamline figures are plotting masks outside the CFD domain."
    )
    if holes:
        classification = "At least one true topological hole remains in the mesh boundary graph."
    elif len(invalid) or len(zero):
        classification = "Invalid, inverted, or zero-area elements remain in the triangulation."
    return MeshTopologyDiagnostics(
        connected_fluid_components=len(components),
        domain_holes=holes,
        invalid_or_inverted_elements=int(len(invalid)),
        zero_area_elements=int(len(zero)),
        exterior_boundary_facets=int(len(boundary_edges) - interior_facets),
        interior_boundary_facets=interior_facets,
        boundary_loop_count=len(loops),
        boundary_facet_counts={name: int(len(edges)) for name, edges in mesh.boundary_facets.items()},
        defect_region_element_ids=np.flatnonzero(defect_region).astype(int).tolist(),
        white_region_classification=classification,
    )


def _sample_offset_boundary_points(geometry: JunctionGeometry, spacing_um: float) -> np.ndarray:
    points = []
    half = geometry.half_width_um
    for name, centerline in geometry.branch_centerlines_um.items():
        sampled = _resample_polyline(centerline, spacing_um)
        tangents, normals = _tangent_normal(sampled)
        points.append(sampled + normals * half)
        points.append(sampled - normals * half)
    for section in geometry.boundary_sections_um.values():
        points.append(_resample_polyline(section, spacing_um))
    return np.vstack(points)


def _sample_grid_points(geometry: JunctionGeometry, spacing_um: float) -> np.ndarray:
    pts = geometry.all_centerline_points_um
    pad = geometry.half_width_um + spacing_um
    grid = _triangular_lattice(
        np.min(pts[:, 0]) - pad,
        np.max(pts[:, 0]) + pad,
        np.min(pts[:, 1]) - pad,
        np.max(pts[:, 1]) + pad,
        spacing_um,
    )
    return grid[inside_junction_domain(grid, geometry, tolerance_um=spacing_um * 0.2)]


def _sample_near_junction_grid_points(geometry: JunctionGeometry, spacing_um: float) -> np.ndarray:
    radius = geometry.channel_width_um * 1.5
    center = geometry.junction_center_um
    grid = _triangular_lattice(
        center[0] - radius,
        center[0] + radius,
        center[1] - radius,
        center[1] + radius,
        spacing_um,
    )
    near = np.linalg.norm(grid - center, axis=1) <= radius
    grid = grid[near]
    return grid[inside_junction_domain(grid, geometry, tolerance_um=spacing_um * 0.2)]


def _triangular_lattice(xmin: float, xmax: float, ymin: float, ymax: float, spacing_um: float) -> np.ndarray:
    dy = spacing_um * np.sqrt(3.0) / 2.0
    rows = []
    y = ymin
    row = 0
    while y <= ymax + dy:
        offset = spacing_um / 2.0 if row % 2 else 0.0
        x = np.arange(xmin + offset, xmax + spacing_um, spacing_um)
        rows.append(np.column_stack([x, np.full_like(x, y)]))
        y += dy
        row += 1
    return np.vstack(rows)


def _resample_polyline(points: np.ndarray, spacing_um: float) -> np.ndarray:
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] <= 0:
        raise ValueError("Cannot resample zero-length polyline")
    n = max(2, int(np.ceil(s[-1] / spacing_um)) + 1)
    sample_s = np.linspace(0.0, s[-1], n)
    x = np.interp(sample_s, s, points[:, 0])
    y = np.interp(sample_s, s, points[:, 1])
    return np.column_stack([x, y])


def _tangent_normal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.empty_like(points)
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    delta[1:-1] = points[2:] - points[:-2]
    tangent = delta / np.linalg.norm(delta, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal


def _cap_points(center: np.ndarray, direction: np.ndarray, radius: float, spacing_um: float) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    n = max(8, int(np.ceil(np.pi * radius / spacing_um)) + 1)
    theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, n)
    return center + radius * (np.cos(theta)[:, None] * normal + np.sin(theta)[:, None] * direction)


def _unique_rows(values: np.ndarray, decimals: int = 8) -> np.ndarray:
    rounded = np.round(values, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return values[np.sort(idx)]


def _estimate_boundary_nodes(nodes: np.ndarray, geometry: JunctionGeometry, tolerance_um: float) -> np.ndarray:
    labels = classify_boundary_points(nodes, geometry)
    return np.flatnonzero(labels != "interior")


def _triangle_edge_lengths(vertices: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
            np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
        ]
    )


def _triangle_areas(vertices: np.ndarray) -> np.ndarray:
    ab = vertices[:, 1] - vertices[:, 0]
    ac = vertices[:, 2] - vertices[:, 0]
    cross_z = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    return np.abs(cross_z) / 2.0


def _signed_triangle_areas(vertices: np.ndarray) -> np.ndarray:
    ab = vertices[:, 1] - vertices[:, 0]
    ac = vertices[:, 2] - vertices[:, 0]
    return (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) / 2.0


def _element_components(elements: np.ndarray) -> list[list[int]]:
    edge_to_elements: dict[tuple[int, int], list[int]] = {}
    for elem_id, tri in enumerate(elements):
        for edge in (tri[[0, 1]], tri[[1, 2]], tri[[2, 0]]):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            edge_to_elements.setdefault(key, []).append(elem_id)
    adjacency = [set() for _ in range(len(elements))]
    for owners in edge_to_elements.values():
        if len(owners) == 2:
            a, b = owners
            adjacency[a].add(b)
            adjacency[b].add(a)
    return _graph_components(adjacency)


def _edge_graph_components(edges: np.ndarray) -> list[np.ndarray]:
    if len(edges) == 0:
        return []
    node_to_edges: dict[int, list[int]] = {}
    for edge_id, edge in enumerate(edges):
        node_to_edges.setdefault(int(edge[0]), []).append(edge_id)
        node_to_edges.setdefault(int(edge[1]), []).append(edge_id)
    adjacency = [set() for _ in range(len(edges))]
    for edge_ids in node_to_edges.values():
        for edge_id in edge_ids:
            adjacency[edge_id].update(other for other in edge_ids if other != edge_id)
    return [edges[component] for component in _graph_components(adjacency)]


def _graph_components(adjacency: list[set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(adjacency)):
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
        components.append(component)
    return components


def _polyline_edge_length(nodes: np.ndarray, edges: np.ndarray) -> float:
    if len(edges) == 0:
        return 0.0
    return float(np.sum(np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)))


def _triangle_angles(edges: np.ndarray) -> np.ndarray:
    a, b, c = edges[:, 0], edges[:, 1], edges[:, 2]
    angles = np.column_stack(
        [
            _law_of_cosines_angle(c, a, b),
            _law_of_cosines_angle(a, b, c),
            _law_of_cosines_angle(b, c, a),
        ]
    )
    return np.degrees(angles)


def _law_of_cosines_angle(adjacent_a: np.ndarray, adjacent_b: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    cos_value = (adjacent_a**2 + adjacent_b**2 - opposite**2) / (2.0 * adjacent_a * adjacent_b)
    return np.arccos(np.clip(cos_value, -1.0, 1.0))


def _triangle_aspect_ratios(edges: np.ndarray) -> np.ndarray:
    return np.max(edges, axis=1) / np.maximum(np.min(edges, axis=1), 1e-12)


def _boundary_edges(elements: np.ndarray) -> np.ndarray:
    edges = np.vstack([elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _estimated_smooth_boundary_length(geometry: JunctionGeometry) -> float:
    centerline_length = sum(float(lengths[-1]) for lengths in geometry.branch_arc_lengths_um.values())
    open_cuts = 3.0 * geometry.channel_width_um
    return 2.0 * centerline_length + open_cuts


def _save_geometry_json(geometry: JunctionGeometry, path: Path) -> None:
    payload: dict[str, Any] = {
        "device_id": geometry.device_id,
        "source_centerline_path": str(geometry.source_centerline_path),
        "coordinate_units": geometry.coordinate_units,
        "channel_width_um": geometry.channel_width_um,
        "channel_width_px": geometry.channel_width_px,
        "um_per_px": geometry.um_per_px,
        "junction_padding_um": geometry.junction_padding_um,
        "target_element_size_um": geometry.target_element_size_um,
        "boundary_refinement_factor": geometry.boundary_refinement_factor,
        "junction_center_um": geometry.junction_center_um.tolist(),
        "boundary_endpoints_um": {name: value.tolist() for name, value in geometry.boundary_endpoints_um.items()},
        "boundary_sections_um": {name: value.tolist() for name, value in geometry.boundary_sections_um.items()},
        "branch_point_counts": {name: int(len(points)) for name, points in geometry.branch_centerlines_um.items()},
        "branch_lengths_um": {
            name: float(lengths[-1]) for name, lengths in geometry.branch_arc_lengths_um.items()
        },
        "assumptions": geometry.assumptions,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown_report(report: MeshQualityReport) -> str:
    area = report.element_area_um2
    near = report.mesh_resolution_near_junction_um
    return "\n".join(
        [
            "# Junction CFD Mesh Quality Report",
            "",
            f"- Coordinate units: `{report.coordinate_units}`",
            f"- Number of nodes: {report.number_of_nodes}",
            f"- Number of elements: {report.number_of_elements}",
            f"- Minimum angle: {report.minimum_angle_deg:.3f} deg",
            f"- Maximum angle: {report.maximum_angle_deg:.3f} deg",
            f"- Mean aspect ratio: {report.mean_aspect_ratio:.3f}",
            f"- Maximum aspect ratio: {report.maximum_aspect_ratio:.3f}",
            f"- Boundary length: {report.boundary_length_um:.3f} um",
            f"- Estimated hydraulic domain area: {report.estimated_hydraulic_domain_area_um2:.3f} um^2",
            "",
            "## Element Area Statistics",
            "",
            f"- Minimum: {area['minimum']:.3f} um^2",
            f"- Median: {area['median']:.3f} um^2",
            f"- Mean: {area['mean']:.3f} um^2",
            f"- Maximum: {area['maximum']:.3f} um^2",
            f"- Total: {area['total']:.3f} um^2",
            "",
            "## Mesh Resolution Near Junction",
            "",
            f"- Minimum edge length: {near['minimum_edge_length']:.3f} um",
            f"- Median edge length: {near['median_edge_length']:.3f} um",
            f"- Mean edge length: {near['mean_edge_length']:.3f} um",
            f"- Maximum edge length: {near['maximum_edge_length']:.3f} um",
            "",
        ]
    )


def _topology_markdown_report(report: MeshTopologyDiagnostics) -> str:
    counts = report.boundary_facet_counts
    return "\n".join(
        [
            "# Junction CFD Mesh Topology Diagnostics",
            "",
            f"- Connected fluid components: {report.connected_fluid_components}",
            f"- Domain holes: {report.domain_holes}",
            f"- Invalid or inverted elements: {report.invalid_or_inverted_elements}",
            f"- Zero-area elements: {report.zero_area_elements}",
            f"- Exterior boundary facets: {report.exterior_boundary_facets}",
            f"- Interior boundary facets: {report.interior_boundary_facets}",
            f"- Boundary loop count: {report.boundary_loop_count}",
            "",
            "## Boundary Facet Counts",
            "",
            f"- Inlet: {counts.get('inlet', 0)}",
            f"- Left outlet: {counts.get('left_outlet', 0)}",
            f"- Right outlet: {counts.get('right_outlet', 0)}",
            f"- Wall: {counts.get('wall', 0)}",
            "",
            "## Former Lower-Center Defect Region",
            "",
            f"- Surrounding element IDs: {report.defect_region_element_ids}",
            f"- Classification: {report.white_region_classification}",
            "",
        ]
    )


def _save_geometry_figure(geometry: JunctionGeometry, path: Path, overlay_centerline: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    nodes = _sample_grid_points(geometry, spacing_um=geometry.target_element_size_um / 2.0)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=2, c="#d8ecff", label="idealized channel")
    if overlay_centerline:
        for name, points in geometry.branch_centerlines_um.items():
            ax.plot(points[:, 0], points[:, 1], linewidth=2, label=f"{name} centerline")
        ax.scatter(*geometry.junction_center_um, c="black", s=25, label="bifurcation")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Idealized junction geometry" + (" with centerline" if overlay_centerline else ""))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_mesh_figure(mesh: TriangularMesh, path: Path) -> None:
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.triplot(tri, linewidth=0.35, color="#1f2937")
    for name, points in mesh.geometry.branch_centerlines_um.items():
        ax.plot(points[:, 0], points[:, 1], linewidth=1.4, label=name)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Generated triangular junction mesh")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_quality_figure(mesh: TriangularMesh, path: Path) -> None:
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    aspect = _triangle_aspect_ratios(_triangle_edge_lengths(mesh.nodes_um[mesh.elements]))
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.tripcolor(tri, aspect, shading="flat", cmap="viridis")
    fig.colorbar(image, ax=ax, label="aspect ratio (longest edge / shortest edge)")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Mesh quality coloring")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_boundary_label_figure(mesh: TriangularMesh, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    ax.triplot(tri, linewidth=0.25, color="#d1d5db")
    colors = {
        "wall": "black",
        "inlet": "tab:blue",
        "left_outlet": "tab:green",
        "right_outlet": "tab:red",
    }
    for label, indices in mesh.boundary_labels.items():
        if len(indices) == 0:
            continue
        pts = mesh.nodes_um[indices]
        ax.scatter(pts[:, 0], pts[:, 1], s=20, label=label, color=colors[label])
        center = pts.mean(axis=0)
        ax.annotate(label, xy=center, xytext=(4, 4), textcoords="offset points", color=colors[label])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("CFD boundary labels")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase-1 junction CFD geometry and mesh.")
    parser.add_argument("--config", type=Path, default=Path("configs/physics/junction_cfd.yml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/physics/junction_cfd"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry = build_junction_geometry(args.config)
    mesh = generate_mesh(geometry)
    report = evaluate_mesh(mesh)
    save_mesh_outputs(mesh, report, args.output_root, overwrite=args.overwrite)
    print("Junction CFD mesh generated")
    print(f"  nodes: {report.number_of_nodes}")
    print(f"  elements: {report.number_of_elements}")
    print(f"  minimum angle: {report.minimum_angle_deg:.3f} deg")
    print(f"  maximum angle: {report.maximum_angle_deg:.3f} deg")
    print(f"  mean aspect ratio: {report.mean_aspect_ratio:.3f}")
    print(f"  maximum aspect ratio: {report.maximum_aspect_ratio:.3f}")
    print(f"  area: {report.estimated_hydraulic_domain_area_um2:.3f} um^2")
    print(f"  boundary length: {report.boundary_length_um:.3f} um")


if __name__ == "__main__":
    main()
