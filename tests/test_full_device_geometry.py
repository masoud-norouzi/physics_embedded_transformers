from __future__ import annotations

import numpy as np

from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry, inside_full_device_domain, validate_device_polygon
from src.physics.full_device_cfd.mesh import evaluate_full_device_mesh, generate_full_device_mesh


def test_full_device_centerlines_follow_physical_flow_direction() -> None:
    geometry = build_full_device_cfd_geometry()

    assert geometry.centerlines["inlet"].points_um[0, 1] > geometry.centerlines["inlet"].points_um[-1, 1]
    assert np.linalg.norm(geometry.centerlines["inlet"].points_um[-1] - geometry.upper_junction_um) < 1.0e-6
    assert np.linalg.norm(geometry.centerlines["left"].points_um[0] - geometry.upper_junction_um) < geometry.channel_width_um
    assert np.linalg.norm(geometry.centerlines["right"].points_um[0] - geometry.upper_junction_um) < geometry.channel_width_um
    assert np.linalg.norm(geometry.centerlines["left"].points_um[-1] - geometry.lower_junction_um) < geometry.channel_width_um
    assert np.linalg.norm(geometry.centerlines["right"].points_um[-1] - geometry.lower_junction_um) < geometry.channel_width_um
    assert np.linalg.norm(geometry.centerlines["outlet"].points_um[0] - geometry.lower_junction_um) < 1.0e-6
    assert geometry.centerlines["outlet"].points_um[0, 1] > geometry.centerlines["outlet"].points_um[-1, 1]


def test_full_device_polygon_has_expected_topology_and_width() -> None:
    geometry = build_full_device_cfd_geometry()
    validation = validate_device_polygon(geometry)

    assert validation["passed"]
    assert validation["connected_fluid_components"] == 1
    assert validation["hole_count"] == 1
    assert validation["outer_ring_self_intersections"] == 0
    assert validation["inner_ring_self_intersections"] == 0
    assert validation["inner_ring_inside_outer"]
    assert geometry.channel_width_um == 100.0
    assert all(abs(row["width_um"] - 100.0) <= 2.0 for row in validation["widths"])


def test_full_device_domain_has_no_central_shortcut() -> None:
    geometry = build_full_device_cfd_geometry()
    y = np.linspace(geometry.lower_junction_um[1] + 1.2 * geometry.half_width_um, geometry.upper_junction_um[1] - 1.2 * geometry.half_width_um, 31)
    probe = np.column_stack([np.full_like(y, geometry.upper_junction_um[0]), y])

    assert not inside_full_device_domain(probe, geometry).any()


def test_full_device_mesh_is_constrained_and_boundary_labels_are_disjoint() -> None:
    geometry = build_full_device_cfd_geometry()
    mesh = generate_full_device_mesh(geometry, target_size_um=24.0, boundary_size_um=12.0)
    quality = evaluate_full_device_mesh(mesh)

    assert quality.minimum_angle_deg > 20.0
    assert quality.maximum_aspect_ratio < 3.0
    assert len(mesh.boundary_facets["inlet"]) > 0
    assert len(mesh.boundary_facets["outlet"]) > 0
    assert len(mesh.boundary_facets["wall"]) > 0
    inlet = {tuple(sorted(edge)) for edge in mesh.boundary_facets["inlet"]}
    outlet = {tuple(sorted(edge)) for edge in mesh.boundary_facets["outlet"]}
    wall = {tuple(sorted(edge)) for edge in mesh.boundary_facets["wall"]}
    assert inlet.isdisjoint(outlet)
    assert inlet.isdisjoint(wall)
    assert outlet.isdisjoint(wall)


def test_full_device_mesh_generation_is_deterministic() -> None:
    geometry = build_full_device_cfd_geometry()
    first = generate_full_device_mesh(geometry, target_size_um=24.0, boundary_size_um=12.0)
    second = generate_full_device_mesh(geometry, target_size_um=24.0, boundary_size_um=12.0)

    np.testing.assert_allclose(first.nodes_um, second.nodes_um)
    np.testing.assert_array_equal(first.elements, second.elements)
