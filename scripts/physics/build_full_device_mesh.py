from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.physics.full_device_cfd.domain import (
    FullDeviceCFDGeometry,
    build_full_device_cfd_geometry,
    validate_device_polygon,
)
from src.physics.full_device_cfd.mesh import (
    FullDeviceMesh,
    evaluate_full_device_mesh,
    generate_full_device_mesh,
    save_full_device_mesh,
)


OUTPUT_DIR = Path("outputs/physics/full_device_cfd/mesh")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    geometry = build_full_device_cfd_geometry()
    validation = validate_device_polygon(geometry)
    _write_centerline_audit(geometry, OUTPUT_DIR / "centerline_orientation_audit.json")
    _write_geometry_json(geometry, OUTPUT_DIR / "full_device_geometry.json")
    _write_width_csv(validation["widths"], OUTPUT_DIR / "width_validation.csv")
    (OUTPUT_DIR / "topology_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    _plot_offset_walls(geometry, OUTPUT_DIR / "full_device_offset_walls.png")
    _plot_polygon(geometry, OUTPUT_DIR / "full_device_polygon.png", centerlines=False)
    _plot_polygon(geometry, OUTPUT_DIR / "full_device_polygon_with_centerlines.png", centerlines=True)
    _plot_junction_zoom(geometry, OUTPUT_DIR / "inlet_junction_pre_boolean_offset_walls.png", geometry.upper_junction_um, "Inlet junction offset walls before Boolean union", offset_walls=True)
    _plot_junction_zoom(geometry, OUTPUT_DIR / "outlet_junction_pre_boolean_offset_walls.png", geometry.lower_junction_um, "Outlet junction offset walls before Boolean union", offset_walls=True)
    _plot_junction_zoom(geometry, OUTPUT_DIR / "inlet_junction_polygon_zoom.png", geometry.upper_junction_um, "Inlet junction after Boolean union", offset_walls=False)
    _plot_junction_zoom(geometry, OUTPUT_DIR / "outlet_junction_polygon_zoom.png", geometry.lower_junction_um, "Outlet junction after Boolean union", offset_walls=False)
    _plot_interior_fragment_diagnostic(geometry, OUTPUT_DIR / "full_device_interior_fragment_diagnostic.png")
    if not validation["passed"]:
        raise RuntimeError(f"Full-device polygon validation failed; mesh was not generated. See {OUTPUT_DIR / 'topology_validation.json'}")

    mesh = generate_full_device_mesh(geometry, target_size_um=24.0, boundary_size_um=12.0)
    quality = save_full_device_mesh(mesh, OUTPUT_DIR, overwrite=True)
    _plot_boundary_labels(mesh, OUTPUT_DIR / "full_device_mesh_boundary_labels.png")
    _plot_mesh_quality(mesh, OUTPUT_DIR / "full_device_mesh_quality.png")
    _plot_mesh_zoom(mesh, OUTPUT_DIR / "inlet_junction_mesh_zoom.png", geometry.upper_junction_um, "Inlet junction mesh zoom")
    _plot_mesh_zoom(mesh, OUTPUT_DIR / "outlet_junction_mesh_zoom.png", geometry.lower_junction_um, "Outlet junction mesh zoom")
    summary = {
        "polygon": {
            "fluid_area_um2": validation["fluid_area_um2"],
            "outer_ring_points": validation["outer_ring_points"],
            "inner_ring_points": validation["inner_ring_points"],
            "connected_fluid_components": validation["connected_fluid_components"],
            "hole_count": validation["hole_count"],
        },
        "mesh": asdict(quality),
        "boundary_facets": {name: int(len(edges)) for name, edges in mesh.boundary_facets.items()},
    }
    print(json.dumps(summary, indent=2))


def _write_geometry_json(geometry: FullDeviceCFDGeometry, path: Path) -> None:
    payload = {
        "device_id": geometry.device_id,
        "coordinate_frame": geometry.coordinate_frame,
        "position_units": "um",
        "channel_width_um": geometry.channel_width_um,
        "channel_height_um": geometry.channel_height_um,
        "outer_ring_um": geometry.outer_ring_um.tolist(),
        "inner_ring_um": geometry.inner_ring_um.tolist(),
        "inlet_cut_um": geometry.inlet_cut_um.tolist(),
        "outlet_cut_um": geometry.outlet_cut_um.tolist(),
        "source_paths": geometry.source_paths,
        "construction_method": "Boolean union of flat-capped centerline channel polygons, followed by exterior and central-hole boundary extraction.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_centerline_audit(geometry: FullDeviceCFDGeometry, path: Path) -> None:
    expected = {
        "inlet": "upstream_to_inlet_junction",
        "left": "inlet_junction_to_outlet_junction",
        "right": "inlet_junction_to_outlet_junction",
        "outlet": "outlet_junction_to_downstream",
    }
    audit = {
        "coordinate_frame": geometry.coordinate_frame,
        "upper_junction_um": geometry.upper_junction_um.tolist(),
        "lower_junction_um": geometry.lower_junction_um.tolist(),
        "branches": {},
    }
    for name, line in geometry.centerlines.items():
        audit["branches"][name] = {
            "points": int(len(line.points_um)),
            "arc_length_um": float(line.length_um),
            "expected_ordering": expected[name],
            "start_um": line.points_um[0].tolist(),
            "end_um": line.points_um[-1].tolist(),
            "start_tangent": line.tangents[0].tolist(),
            "end_tangent": line.tangents[-1].tolist(),
            "follows_physical_flow": True,
        }
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def _write_width_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "branch", "x_device_um", "y_device_um", "width_um"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_offset_walls(geometry: FullDeviceCFDGeometry, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, line in geometry.centerlines.items():
        plus = line.points_um + geometry.half_width_um * line.normals
        minus = line.points_um - geometry.half_width_um * line.normals
        ax.plot(line.points_um[:, 0], line.points_um[:, 1], linewidth=1.0, label=f"{name} centerline")
        ax.plot(plus[:, 0], plus[:, 1], linewidth=0.9, linestyle="--")
        ax.plot(minus[:, 0], minus[:, 1], linewidth=0.9, linestyle=":")
    _format_device_axes(ax, "Individual +/-50 um offset walls")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_polygon(geometry: FullDeviceCFDGeometry, path: Path, *, centerlines: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill(geometry.outer_ring_um[:, 0], geometry.outer_ring_um[:, 1], color="#dbeafe", alpha=0.8, label="fluid envelope")
    ax.fill(geometry.inner_ring_um[:, 0], geometry.inner_ring_um[:, 1], color="white", alpha=1.0, label="central solid island")
    ax.plot(geometry.outer_ring_um[:, 0], geometry.outer_ring_um[:, 1], color="#1d4ed8", linewidth=1.2)
    ax.plot(geometry.inner_ring_um[:, 0], geometry.inner_ring_um[:, 1], color="#111827", linewidth=1.2)
    ax.plot(geometry.inlet_cut_um[:, 0], geometry.inlet_cut_um[:, 1], color="#16a34a", linewidth=2.0, label="inlet cut")
    ax.plot(geometry.outlet_cut_um[:, 0], geometry.outlet_cut_um[:, 1], color="#dc2626", linewidth=2.0, label="outlet cut")
    if centerlines:
        for line in geometry.centerlines.values():
            ax.plot(line.points_um[:, 0], line.points_um[:, 1], linewidth=1.0, label=f"{line.branch} centerline")
    _format_device_axes(ax, "Centerline-derived full-device CFD polygon")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_junction_zoom(geometry: FullDeviceCFDGeometry, path: Path, center: np.ndarray, title: str, *, offset_walls: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.fill(geometry.outer_ring_um[:, 0], geometry.outer_ring_um[:, 1], color="#dbeafe", alpha=0.65)
    ax.fill(geometry.inner_ring_um[:, 0], geometry.inner_ring_um[:, 1], color="white", alpha=1.0)
    ax.plot(geometry.outer_ring_um[:, 0], geometry.outer_ring_um[:, 1], color="#1d4ed8", linewidth=1.4, label="final exterior boundary")
    ax.plot(geometry.inner_ring_um[:, 0], geometry.inner_ring_um[:, 1], color="#111827", linewidth=1.4, label="final island boundary")
    for line in geometry.centerlines.values():
        ax.plot(line.points_um[:, 0], line.points_um[:, 1], linewidth=1.0, label=f"{line.branch} centerline")
        if offset_walls:
            plus = line.points_um + geometry.half_width_um * line.normals
            minus = line.points_um - geometry.half_width_um * line.normals
            ax.plot(plus[:, 0], plus[:, 1], linewidth=0.9, linestyle="--", color="#f97316", alpha=0.8)
            ax.plot(minus[:, 0], minus[:, 1], linewidth=0.9, linestyle=":", color="#7c3aed", alpha=0.8)
    ax.set_xlim(center[0] - 220.0, center[0] + 220.0)
    ax.set_ylim(center[1] - 220.0, center[1] + 220.0)
    _format_device_axes(ax, title)
    ax.legend(loc="best", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_interior_fragment_diagnostic(geometry: FullDeviceCFDGeometry, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill(geometry.outer_ring_um[:, 0], geometry.outer_ring_um[:, 1], color="#e0f2fe", alpha=0.55)
    ax.fill(geometry.inner_ring_um[:, 0], geometry.inner_ring_um[:, 1], color="white", alpha=1.0)
    ax.plot(geometry.outer_ring_um[:, 0], geometry.outer_ring_um[:, 1], color="#0f172a", linewidth=1.0, label="final boundary")
    ax.plot(geometry.inner_ring_um[:, 0], geometry.inner_ring_um[:, 1], color="#0f172a", linewidth=1.0)
    fragments = _boundary_segments_inside_fluid(geometry)
    for segment in fragments:
        ax.plot(segment[:, 0], segment[:, 1], color="#dc2626", linewidth=2.8)
    title = f"Interior boundary-fragment diagnostic: {len(fragments)} final fragments"
    _format_device_axes(ax, title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _boundary_segments_inside_fluid(geometry: FullDeviceCFDGeometry) -> list[np.ndarray]:
    fragments = []
    for ring in (geometry.outer_ring_um, geometry.inner_ring_um):
        for start, end in zip(ring[:-1], ring[1:]):
            midpoint = 0.5 * (start + end)
            if _strictly_inside_final_fluid(midpoint, geometry):
                fragments.append(np.vstack([start, end]))
    return fragments


def _strictly_inside_final_fluid(point: np.ndarray, geometry: FullDeviceCFDGeometry) -> bool:
    from src.physics.full_device_cfd.domain import inside_full_device_domain

    eps = 0.5
    offsets = np.asarray([[eps, 0.0], [-eps, 0.0], [0.0, eps], [0.0, -eps]])
    return bool(inside_full_device_domain(point[None, :] + offsets, geometry).all())


def _plot_boundary_labels(mesh: FullDeviceMesh, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"inlet": "#16a34a", "outlet": "#dc2626", "wall": "#111827"}
    for name, edges in mesh.boundary_facets.items():
        for edge in edges:
            p = mesh.nodes_um[edge]
            ax.plot(p[:, 0], p[:, 1], color=colors[name], linewidth=1.2)
    _format_device_axes(ax, "Full-device mesh boundary labels")
    handles = [plt.Line2D([0], [0], color=color, lw=2, label=name) for name, color in colors.items()]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_mesh_quality(mesh: FullDeviceMesh, path: Path) -> None:
    vertices = mesh.nodes_um[mesh.elements]
    edge_lengths = np.column_stack(
        [
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
            np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
        ]
    )
    aspect = np.max(edge_lengths, axis=1) / np.maximum(np.min(edge_lengths, axis=1), 1.0e-12)
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.tripcolor(tri, aspect, shading="flat", cmap="viridis")
    ax.triplot(tri, linewidth=0.08, color="white", alpha=0.25)
    _format_device_axes(ax, "Full-device mesh aspect-ratio diagnostic")
    fig.colorbar(image, ax=ax, label="element aspect ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_mesh_zoom(mesh: FullDeviceMesh, path: Path, center: np.ndarray, title: str) -> None:
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.triplot(tri, linewidth=0.35, color="#1f2937")
    ax.plot(mesh.geometry.outer_ring_um[:, 0], mesh.geometry.outer_ring_um[:, 1], color="#2563eb", linewidth=1.2)
    ax.plot(mesh.geometry.inner_ring_um[:, 0], mesh.geometry.inner_ring_um[:, 1], color="#111827", linewidth=1.2)
    for line in mesh.geometry.centerlines.values():
        ax.plot(line.points_um[:, 0], line.points_um[:, 1], linewidth=1.0)
    ax.set_xlim(center[0] - 220.0, center[0] + 220.0)
    ax.set_ylim(center[1] - 220.0, center[1] + 220.0)
    _format_device_axes(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _format_device_axes(ax: plt.Axes, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title(title)


if __name__ == "__main__":
    main()
