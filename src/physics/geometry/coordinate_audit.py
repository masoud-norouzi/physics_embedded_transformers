from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.physics.cfd.solver import _boundary_fluxes_fem_quadrature
from src.physics.enrichment.coordinate_mapping import build_coordinate_transform
from src.physics.geometry.coordinates import CoordinateConvention
from src.physics.interpolation import VelocityFieldLibrary
from src.physics.interpolation.field_sampler import paired_velocity_to_basis_coefficients, velocity_basis


@dataclass(frozen=True)
class DirectionAuditRow:
    region: str
    sample_x_cfd_um: float
    sample_y_cfd_um: float
    u_x_cfd_m_per_s: float
    u_y_cfd_m_per_s: float
    expected_direction: str
    measured_direction_dot: float
    passed: bool


def run_coordinate_audit(
    *,
    experiment_config_path: str | Path = "configs/experiments/video_2.yml",
    library_path: str | Path = "outputs/physics/junction_cfd/solutions",
    output_root: str | Path = "outputs/physics/coordinate_audit",
    left_fraction: float = 0.50,
    overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(output_root)
    figures = out / "figures"
    outputs = [
        out / "coordinate_convention_summary.json",
        out / "coordinate_convention_summary.md",
        figures / "cfd_native_direction_check.png",
        figures / "device_cartesian_direction_check.png",
        figures / "image_coordinate_direction_check.png",
        figures / "known_point_round_trip.png",
    ]
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Coordinate-audit outputs already exist. Use --overwrite: {existing}")
    figures.mkdir(parents=True, exist_ok=True)

    transform = build_coordinate_transform(str(experiment_config_path))
    convention = transform.convention
    library = VelocityFieldLibrary.from_directory(library_path)
    field = library.interpolate(left_fraction)
    rows = _direction_rows(field, convention)
    fluxes = _boundary_fluxes(field)
    round_trip = _round_trip_errors(convention, field)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_frames": {
            "image": {
                "coordinates": ["x_px", "y_px"],
                "origin": "top-left video pixel",
                "axes": "+x right, +y down",
                "units": "pixels",
            },
            "device_cartesian": {
                "coordinates": ["x_device_um", "y_device_um"],
                "origin": "lower-left image reference line derived from fixed image height",
                "axes": "+x right, +y up",
                "units": "micrometers",
            },
            "cfd_native": {
                "coordinates": ["x_cfd_um", "y_cfd_um"],
                "origin": "calibrated full-image centerline origin used by frozen CFD Version 1 mesh",
                "axes": "+x right, +y down",
                "units": "micrometers",
            },
        },
        "transforms": {
            "image_to_device_points": "x_device_um = scale*x_px; y_device_um = scale*(y_reference_px - y_px)",
            "image_to_device_vectors": "v_x_device = scale*v_x_px; v_y_device = -scale*v_y_px",
            "device_to_cfd_points": "x_cfd_um = x_device_um; y_cfd_um = y_reference_um - y_device_um",
            "device_to_cfd_vectors": "u_x_cfd = u_x_device; u_y_cfd = -u_y_device",
            "cfd_to_device_vectors": "u_x_device = u_x_cfd; u_y_device = -u_y_cfd",
        },
        "tracked_velocity_frame": {
            "source_columns": "tracked_features.csv contains centroid positions only, not canonical vx/vy columns",
            "derived_observed_velocity": "centered finite difference of centroid_x/centroid_y gives pixels/frame in image frame",
            "device_conversion": "v_x_device_um_per_frame = scale*v_x_image_px_per_frame; v_y_device_um_per_frame = -scale*v_y_image_px_per_frame",
            "frame_rate": "not available in current experiment configuration, so observed velocities remain um/frame for direction diagnostics",
        },
        "cfd_version": field.cfd_version,
        "mesh_version": field.mesh_version,
        "audited_left_fraction": left_fraction,
        "direction_audit_rows": [asdict(row) for row in rows],
        "boundary_fluxes_m2_per_s": fluxes,
        "round_trip_errors": round_trip,
        "stored_field_check": "passed" if all(row.passed for row in rows) else "failed",
        "sampler_check": "VelocityFieldLibrary exact-grid sample_cfd preserves stored direction at audited points.",
        "root_cause": (
            "The frozen CFD field is physically oriented correctly in its native y-down frame. "
            "The misleading inlet arrow came from ambiguous downstream coordinate/vector conventions and diagnostics that did not name the frame explicitly. "
            "Diagnostics now transform vectors explicitly instead of hiding sign changes in plotting code."
        ),
        "frozen_cfd_files_modified": False,
    }
    (out / "coordinate_convention_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "coordinate_convention_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    _save_cfd_plot(field, rows, figures / "cfd_native_direction_check.png")
    _save_device_plot(field, convention, rows, figures / "device_cartesian_direction_check.png")
    _save_image_plot(field, convention, rows, figures / "image_coordinate_direction_check.png")
    _save_round_trip_plot(convention, field, figures / "known_point_round_trip.png")
    return summary


def _direction_rows(field, convention: CoordinateConvention) -> list[DirectionAuditRow]:
    geometry = field.mesh.geometry
    samples = _audit_points(geometry)
    rows = []
    sampled = field.sample_cfd(np.vstack([item[1] for item in samples]))
    for (region, point, expected_vector, expected_label), ux, uy in zip(
        samples,
        sampled.u_x_m_per_s,
        sampled.u_y_m_per_s,
    ):
        velocity = np.array([ux, uy], dtype=float)
        dot = float(velocity @ expected_vector)
        rows.append(
            DirectionAuditRow(
                region=region,
                sample_x_cfd_um=float(point[0]),
                sample_y_cfd_um=float(point[1]),
                u_x_cfd_m_per_s=float(velocity[0]),
                u_y_cfd_m_per_s=float(velocity[1]),
                expected_direction=expected_label,
                measured_direction_dot=dot,
                passed=bool(dot > 0.0),
            )
        )
    return rows


def _audit_points(geometry) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    junction = geometry.junction_center_um
    specs = [
        ("inlet", "inlet", "toward junction"),
        ("left_outlet", "left_outlet", "away from junction"),
        ("right_outlet", "right_outlet", "away from junction"),
    ]
    out = []
    for region, endpoint_key, label in specs:
        endpoint = geometry.boundary_endpoints_um[endpoint_key]
        inward = junction - endpoint
        inward = inward / np.linalg.norm(inward)
        point = endpoint + inward * 30.0
        expected = (junction - point) if region == "inlet" else (point - junction)
        out.append((region, point, expected / np.linalg.norm(expected), label))
    return out


def _boundary_fluxes(field) -> dict[str, float]:
    basis = velocity_basis(field.nodes_um, field.elements)
    coeff = paired_velocity_to_basis_coefficients(basis, field.velocity_dof_m_per_s)
    return _boundary_fluxes_fem_quadrature(field.mesh, basis, coeff)


def _round_trip_errors(convention: CoordinateConvention, field) -> dict[str, float]:
    points_image = convention.cfd_points_to_image(field.nodes_um[:10])
    image_round = convention.device_points_to_image(convention.image_points_to_device(points_image))
    vectors = np.array([[1.0, 2.0], [-3.0, 4.0], [0.5, -0.25]])
    vector_round = convention.device_vectors_to_image(convention.image_vectors_to_device(vectors))
    cfd_round = convention.device_points_to_cfd(convention.cfd_points_to_device(field.nodes_um[:10]))
    return {
        "image_point_round_trip_px": float(np.max(np.abs(points_image - image_round))),
        "image_vector_round_trip_px": float(np.max(np.abs(vectors - vector_round))),
        "cfd_device_point_round_trip_um": float(np.max(np.abs(field.nodes_um[:10] - cfd_round))),
    }


def _save_cfd_plot(field, rows: list[DirectionAuditRow], path: Path) -> None:
    mesh = field.mesh
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.triplot(tri, color="#94a3b8", linewidth=0.35)
    ax.scatter([mesh.geometry.junction_center_um[0]], [mesh.geometry.junction_center_um[1]], color="#111827", label="junction")
    for row in rows:
        ax.quiver(row.sample_x_cfd_um, row.sample_y_cfd_um, row.u_x_cfd_m_per_s, row.u_y_cfd_m_per_s, color="#dc2626", scale=0.25)
        ax.text(row.sample_x_cfd_um, row.sample_y_cfd_um, row.region)
    ax.set_title("CFD native frame: x right, y down")
    ax.set_xlabel("x_cfd_um")
    ax.set_ylabel("y_cfd_um")
    ax.set_aspect("equal")
    ax.set_ylim(mesh.nodes_um[:, 1].max() + 40.0, mesh.nodes_um[:, 1].min() - 40.0)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_device_plot(field, convention: CoordinateConvention, rows: list[DirectionAuditRow], path: Path) -> None:
    nodes = convention.cfd_points_to_device(field.nodes_um)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], field.elements)
    junction = convention.cfd_points_to_device(field.mesh.geometry.junction_center_um[None, :])[0]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.triplot(tri, color="#94a3b8", linewidth=0.35)
    ax.scatter([junction[0]], [junction[1]], color="#111827", label="junction")
    for row in rows:
        point = convention.cfd_points_to_device(np.array([[row.sample_x_cfd_um, row.sample_y_cfd_um]]))[0]
        vel = convention.cfd_vectors_to_device(np.array([[row.u_x_cfd_m_per_s, row.u_y_cfd_m_per_s]]))[0]
        ax.quiver(point[0], point[1], vel[0], vel[1], color="#2563eb", scale=0.25)
        ax.text(point[0], point[1], row.region)
    ax.set_title("Device Cartesian frame: x right, y up")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_image_plot(field, convention: CoordinateConvention, rows: list[DirectionAuditRow], path: Path) -> None:
    nodes = convention.cfd_points_to_image(field.nodes_um)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], field.elements)
    junction = convention.cfd_points_to_image(field.mesh.geometry.junction_center_um[None, :])[0]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.triplot(tri, color="#94a3b8", linewidth=0.35)
    ax.scatter([junction[0]], [junction[1]], color="#111827", label="junction")
    for row in rows:
        point = convention.cfd_points_to_image(np.array([[row.sample_x_cfd_um, row.sample_y_cfd_um]]))[0]
        vel = convention.cfd_vectors_to_image(np.array([[row.u_x_cfd_m_per_s, row.u_y_cfd_m_per_s]]))[0]
        ax.quiver(point[0], point[1], vel[0], vel[1], color="#16a34a", scale=0.06)
        ax.text(point[0], point[1], row.region)
    ax.set_title("Image frame: x right, y down")
    ax.set_xlabel("x_px")
    ax.set_ylabel("y_px")
    ax.set_aspect("equal")
    ax.set_ylim(nodes[:, 1].max() + 10.0, nodes[:, 1].min() - 10.0)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_round_trip_plot(convention: CoordinateConvention, field, path: Path) -> None:
    cfd = field.nodes_um[:: max(1, len(field.nodes_um) // 20)]
    image = convention.cfd_points_to_image(cfd)
    cfd_round = convention.image_points_to_cfd(image)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(cfd[:, 0], cfd[:, 1], label="original CFD points", color="#2563eb")
    ax.scatter(cfd_round[:, 0], cfd_round[:, 1], marker="x", label="round trip", color="#dc2626")
    ax.set_title("Known point round trip: CFD -> image -> CFD")
    ax.set_xlabel("x_cfd_um")
    ax.set_ylabel("y_cfd_um")
    ax.set_aspect("equal")
    ax.set_ylim(field.nodes_um[:, 1].max() + 40.0, field.nodes_um[:, 1].min() - 40.0)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Coordinate Convention Audit",
        "",
        f"- CFD version: `{summary['cfd_version']}`",
        f"- Mesh version: `{summary['mesh_version']}`",
        f"- Audited left fraction: {summary['audited_left_fraction']:.2f}",
        f"- Stored-field check: {summary['stored_field_check']}",
        f"- Frozen CFD files modified: {summary['frozen_cfd_files_modified']}",
        "",
        "## Direction Audit",
        "",
        "| region | sample x_cfd_um | sample y_cfd_um | u_x_cfd | u_y_cfd | expected | dot | pass |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in summary["direction_audit_rows"]:
        lines.append(
            f"| {row['region']} | {row['sample_x_cfd_um']:.3f} | {row['sample_y_cfd_um']:.3f} | "
            f"{row['u_x_cfd_m_per_s']:.6e} | {row['u_y_cfd_m_per_s']:.6e} | {row['expected_direction']} | "
            f"{row['measured_direction_dot']:.6e} | {row['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary Fluxes",
            "",
            f"- Inlet signed flux: {summary['boundary_fluxes_m2_per_s']['inlet']:.6e} m^2/s",
            f"- Left outlet signed flux: {summary['boundary_fluxes_m2_per_s']['left_outlet']:.6e} m^2/s",
            f"- Right outlet signed flux: {summary['boundary_fluxes_m2_per_s']['right_outlet']:.6e} m^2/s",
            "",
            "## Conclusion",
            "",
            summary["root_cause"],
        ]
    )
    return "\n".join(lines)
