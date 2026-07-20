from pathlib import Path

import numpy as np

from src.physics.cfd.domain import build_junction_geometry
from src.physics.cfd.mesh import evaluate_mesh, generate_mesh


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
