from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.physics.interpolation import VelocityFieldLibrary
from src.physics.interpolation.validation import validate_velocity_interpolation


LIBRARY_PATH = Path("outputs/physics/junction_cfd/solutions")


@pytest.fixture(scope="module")
def library() -> VelocityFieldLibrary:
    if not (LIBRARY_PATH / "library_index.json").exists():
        pytest.skip("Frozen CFD Version 1 library is not available")
    return VelocityFieldLibrary.from_directory(LIBRARY_PATH)


def test_library_discovery_finds_exactly_19_ordered_cases(library: VelocityFieldLibrary) -> None:
    assert len(library.cases) == 19
    assert library.fractions == tuple(round(0.05 * i, 2) for i in range(1, 20))
    assert all(a < b for a, b in zip(library.fractions, library.fractions[1:]))


def test_all_cases_share_the_same_mesh(library: VelocityFieldLibrary) -> None:
    reference = library.cases[0]
    for case in library.cases[1:]:
        assert np.array_equal(case.elements, reference.elements)
        assert np.allclose(case.nodes_um, reference.nodes_um, rtol=0.0, atol=1.0e-12)


def test_exact_match_returns_stored_field_values_unchanged(library: VelocityFieldLibrary) -> None:
    case = library.case_for_fraction(0.50)
    field = library.interpolate(0.50)

    assert field.exact_match
    assert field.lower_case_id == "split_0p50"
    assert field.upper_case_id == "split_0p50"
    assert np.array_equal(field.velocity_dof_m_per_s, case.velocity_dof_m_per_s)
    assert np.array_equal(field.velocity_node_m_per_s, case.velocity_node_m_per_s)


def test_midpoint_interpolation_uses_correct_neighbors_and_weight(library: VelocityFieldLibrary) -> None:
    field = library.interpolate(0.325)

    assert not field.exact_match
    assert field.lower_library_fraction == 0.30
    assert field.upper_library_fraction == 0.35
    assert math.isclose(field.interpolation_weight, 0.5, abs_tol=1.0e-12)


def test_arbitrary_interpolation_selects_expected_neighbors(library: VelocityFieldLibrary) -> None:
    field = library.interpolate(0.33)

    assert field.lower_library_fraction == 0.30
    assert field.upper_library_fraction == 0.35
    assert math.isclose(field.interpolation_weight, 0.6, rel_tol=0.0, abs_tol=1.0e-12)


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 0.049, 0.951, 1.0, float("nan"), float("inf")])
def test_invalid_fractions_fail_clearly(library: VelocityFieldLibrary, alpha: float) -> None:
    with pytest.raises(ValueError):
        library.interpolate(alpha)


def test_vectorized_point_sampling_shapes_and_outside_flags(library: VelocityFieldLibrary) -> None:
    field = library.interpolate(0.33)
    inside_points = field.velocity_dof_coordinates_um[[10, 100, 500]]
    points = np.vstack([inside_points, np.array([[0.0, 0.0]])])
    samples = field.sample(points)

    assert samples.u_x_m_per_s.shape == (4,)
    assert samples.u_y_m_per_s.shape == (4,)
    assert samples.speed_m_per_s.shape == (4,)
    assert samples.direction_x.shape == (4,)
    assert samples.inside_domain.shape == (4,)
    assert samples.inside_domain[:3].all()
    assert not samples.inside_domain[3]
    assert np.isnan(samples.speed_m_per_s[3])


def test_exact_cfd_fields_are_reproduced_at_stored_fractions(library: VelocityFieldLibrary) -> None:
    for alpha in (0.05, 0.50, 0.95):
        case = library.case_for_fraction(alpha)
        field = library.interpolate(alpha)
        assert np.array_equal(field.velocity_dof_m_per_s, case.velocity_dof_m_per_s)


def test_interpolation_does_not_modify_input_arrays(library: VelocityFieldLibrary) -> None:
    low = library.case_for_fraction(0.30)
    high = library.case_for_fraction(0.35)
    low_before = low.velocity_dof_m_per_s.copy()
    high_before = high.velocity_dof_m_per_s.copy()

    _ = library.interpolate(0.33)

    assert np.array_equal(low.velocity_dof_m_per_s, low_before)
    assert np.array_equal(high.velocity_dof_m_per_s, high_before)


def test_withheld_case_validation_runs_successfully(tmp_path: Path) -> None:
    if not (LIBRARY_PATH / "library_index.json").exists():
        pytest.skip("Frozen CFD Version 1 library is not available")

    summary = validate_velocity_interpolation(
        library_path=LIBRARY_PATH,
        output_root=tmp_path / "validation",
        withheld_fractions=(0.20,),
        overwrite=True,
    )

    assert summary["withheld_fractions"] == [0.20]
    metric = summary["metrics"][0]
    assert metric["lower_fraction"] == 0.15
    assert metric["upper_fraction"] == 0.25
    assert metric["dof_component_rmse_m_per_s"] < 1.0e-10
    assert abs(metric["mass_balance_residual_m2_per_s"]) < 1.0e-12
    assert summary["scientific_consistency"]["no_nan_inside_domain"]


def test_frozen_cfd_outputs_remain_untouched(library: VelocityFieldLibrary, tmp_path: Path) -> None:
    watched = [case.path / "fields" / "stokes_solution.npz" for case in library.cases[:3]]
    before = {path: path.stat().st_mtime_ns for path in watched}

    _ = library.interpolate(0.33)
    _ = validate_velocity_interpolation(
        library_path=LIBRARY_PATH,
        output_root=tmp_path / "validation",
        withheld_fractions=(0.40,),
        overwrite=True,
    )

    after = {path: path.stat().st_mtime_ns for path in watched}
    assert after == before
