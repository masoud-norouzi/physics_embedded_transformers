from __future__ import annotations

import numpy as np

from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry
from src.physics.full_device_cfd.field_verification import (
    CommonGrid,
    GridField,
    angular_error_degrees,
    build_common_grid,
    classify_separatrix,
    evaluate_solution_on_grid,
    inlet_seed_points,
    interpolate_grid_velocity,
    paired_velocity_to_basis_coefficients,
    region_masks,
    separatrix_seed_location,
    vector_error_metrics,
    velocity_basis,
)
from src.physics.full_device_cfd.mesh import generate_full_device_mesh


def test_common_grid_interpolation_is_consistent_for_constant_p2_field() -> None:
    geometry = build_full_device_cfd_geometry()
    mesh = generate_full_device_mesh(geometry, target_size_um=80.0, boundary_size_um=40.0)
    basis = velocity_basis(mesh.nodes_um, mesh.elements)
    xidx, _yidx = basis.split_indices()
    paired = np.tile(np.array([[0.012, -0.004]]), (len(xidx), 1))
    grid = build_common_grid(geometry, spacing_um=80.0)

    field = evaluate_solution_on_grid(mesh.nodes_um, mesh.elements, paired, grid)

    assert np.nanmax(np.abs(field.ux_m_per_s.ravel()[grid.inside] - 0.012)) < 1.0e-12
    assert np.nanmax(np.abs(field.uy_m_per_s.ravel()[grid.inside] + 0.004)) < 1.0e-12


def test_paired_velocity_coefficients_preserve_all_dofs() -> None:
    geometry = build_full_device_cfd_geometry()
    mesh = generate_full_device_mesh(geometry, target_size_um=100.0, boundary_size_um=50.0)
    basis = velocity_basis(mesh.nodes_um, mesh.elements)
    xidx, yidx = basis.split_indices()
    values = np.column_stack([np.arange(len(xidx)), -np.arange(len(xidx))])

    coeff = paired_velocity_to_basis_coefficients(basis, values)

    np.testing.assert_allclose(coeff[xidx], values[:, 0])
    np.testing.assert_allclose(coeff[yidx], values[:, 1])


def test_fluid_domain_mask_and_junction_region_masks_are_subset_of_domain() -> None:
    geometry = build_full_device_cfd_geometry()
    grid = build_common_grid(geometry, spacing_um=50.0)
    masks = region_masks(grid, geometry)

    assert np.any(grid.inside)
    assert np.any(masks["inlet_junction"])
    assert np.any(masks["outlet_junction"])
    assert np.all(masks["inlet_junction"] <= grid.inside)
    assert np.all(masks["outlet_junction"] <= grid.inside)


def test_vector_field_error_is_zero_for_identical_grid_fields() -> None:
    grid = _toy_grid()
    field = GridField(np.ones_like(grid.xx_um), np.zeros_like(grid.yy_um), np.ones_like(grid.xx_um))
    metrics = vector_error_metrics(field, field, {"full_domain": grid.inside})

    assert metrics[0]["vector_l2_relative_error"] == 0.0
    assert metrics[0]["median_angular_error_deg"] == 0.0
    assert metrics[0]["speed_l2_relative_error"] == 0.0


def test_angular_error_calculation_reports_right_angle() -> None:
    candidate = np.array([[1.0, 0.0], [0.0, 1.0]])
    reference = np.array([[0.0, 1.0], [0.0, 2.0]])

    angle = angular_error_degrees(candidate, reference)

    np.testing.assert_allclose(angle, [90.0, 0.0], atol=1.0e-12)


def test_low_speed_masking_excludes_stagnation_points_from_angle_statistics() -> None:
    grid = _toy_grid()
    reference = GridField(np.array([[1.0, 0.0], [1.0, 0.0]]), np.zeros((2, 2)), np.array([[1.0, 0.0], [1.0, 0.0]]))
    candidate = GridField(np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [0.0, 1.0]]), np.sqrt(2.0) * np.ones((2, 2)))

    metrics = vector_error_metrics(candidate, reference, {"full_domain": grid.inside}, low_speed_fraction=0.01)

    assert metrics[0]["angular_excluded_fraction"] == 0.5


def test_grid_velocity_interpolation_uses_identical_grid_coordinates() -> None:
    grid = _toy_grid()
    field = GridField(grid.xx_um.copy(), grid.yy_um.copy(), np.hypot(grid.xx_um, grid.yy_um))

    velocity = interpolate_grid_velocity(field, grid, np.array([0.5, 0.5]))

    np.testing.assert_allclose(velocity, [0.5, 0.5])


def test_inlet_seed_points_exclude_walls() -> None:
    geometry = build_full_device_cfd_geometry()

    seeds, offsets = inlet_seed_points(geometry, seed_count=11, wall_margin_fraction=0.1)

    assert len(seeds) == 11
    assert np.max(np.abs(offsets)) < geometry.half_width_um


def test_separatrix_transition_location_is_midpoint_between_changed_labels() -> None:
    classifications = [
        {"signed_offset_um": -10.0, "classification": "left"},
        {"signed_offset_um": 0.0, "classification": "left"},
        {"signed_offset_um": 10.0, "classification": "right"},
    ]

    assert separatrix_seed_location(classifications) == 5.0


def test_separatrix_classification_runs_on_synthetic_downward_field() -> None:
    geometry = build_full_device_cfd_geometry()
    grid = build_common_grid(geometry, spacing_um=30.0)
    ux = np.zeros_like(grid.xx_um)
    uy = -np.ones_like(grid.yy_um)
    field = GridField(ux, uy, np.ones_like(ux))

    result = classify_separatrix(field, grid, geometry, seed_count=5, step_um=20.0, max_steps=20)

    assert result["seed_count"] == 5
    assert len(result["classifications"]) == 5


def _toy_grid() -> CommonGrid:
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return CommonGrid(x, y, xx, yy, points, np.ones(4, dtype=bool), 1.0)
