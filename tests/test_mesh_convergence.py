from pathlib import Path

from src.physics.cfd.domain import build_junction_geometry
from src.physics.cfd.mesh import evaluate_mesh, generate_mesh
from src.physics.cfd.mesh_convergence import fine_mesh_config


CONFIG = Path("configs/physics/junction_cfd.yml")


def test_improved_mesh_remains_default_configuration() -> None:
    geometry = build_junction_geometry(CONFIG)
    mesh = generate_mesh(geometry)
    report = evaluate_mesh(mesh)

    assert geometry.target_element_size_um == 20.0
    assert geometry.mesh_point_min_distance_um == 6.0
    assert report.number_of_nodes == 425
    assert report.number_of_elements == 656
    assert report.minimum_angle_deg > 20.0
    assert report.maximum_aspect_ratio < 3.0


def test_fine_mesh_configuration_is_separate_from_default() -> None:
    default_geometry = build_junction_geometry(CONFIG)
    fine_geometry = build_junction_geometry(fine_mesh_config(CONFIG))
    fine_mesh = generate_mesh(fine_geometry)
    fine_report = evaluate_mesh(fine_mesh)

    assert default_geometry.target_element_size_um == 20.0
    assert fine_geometry.target_element_size_um == 12.0
    assert fine_geometry.mesh_point_min_distance_um == 4.0
    assert fine_report.number_of_nodes > 425
    assert fine_report.number_of_elements > 656
