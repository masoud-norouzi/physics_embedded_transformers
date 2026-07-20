from pathlib import Path

import numpy as np

from src.physics.cfd.domain import build_junction_geometry
from src.physics.cfd.mesh import evaluate_mesh, evaluate_mesh_topology, generate_mesh


CONFIG = Path("configs/physics/junction_cfd.yml")


def test_junction_geometry_preserves_micrometer_units() -> None:
    geometry = build_junction_geometry(CONFIG)
    assert geometry.coordinate_units == "um"
    assert geometry.channel_width_um == 100.0
    assert geometry.um_per_px == 4.0
    assert np.allclose(geometry.junction_center_um, [1320.0, 860.0])


def test_junction_patch_contains_inlet_and_two_downstream_branches() -> None:
    geometry = build_junction_geometry(CONFIG)
    assert set(geometry.branch_centerlines_um) == {"inlet", "left", "right"}
    for lengths in geometry.branch_arc_lengths_um.values():
        assert lengths[-1] >= geometry.junction_padding_um - 1e-9


def test_generate_mesh_and_quality_report_are_positive() -> None:
    geometry = build_junction_geometry(CONFIG)
    mesh = generate_mesh(geometry)
    report = evaluate_mesh(mesh)
    assert mesh.nodes_um.shape[1] == 2
    assert mesh.elements.shape[1] == 3
    assert report.coordinate_units == "um"
    assert report.number_of_nodes > 0
    assert report.number_of_elements > 0
    assert report.minimum_angle_deg > 0
    assert report.estimated_hydraulic_domain_area_um2 > 0


def test_refined_junction_mesh_quality_and_topology_regression() -> None:
    geometry = build_junction_geometry(CONFIG)
    mesh = generate_mesh(geometry)
    report = evaluate_mesh(mesh)
    topology = evaluate_mesh_topology(mesh)

    assert report.minimum_angle_deg > 15.0
    assert report.maximum_aspect_ratio < 5.0
    assert topology.connected_fluid_components == 1
    assert topology.domain_holes == 0
    assert topology.invalid_or_inverted_elements == 0
    assert topology.zero_area_elements == 0
    assert topology.interior_boundary_facets == 0
    assert topology.boundary_facet_counts["inlet"] > 0
    assert topology.boundary_facet_counts["left_outlet"] > 0
    assert topology.boundary_facet_counts["right_outlet"] > 0
    assert topology.boundary_facet_counts["wall"] > 0
