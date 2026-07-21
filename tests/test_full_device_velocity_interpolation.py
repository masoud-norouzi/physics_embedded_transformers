from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry
from src.physics.interpolation import VelocityFieldLibrary


LIBRARY_PATH = Path("outputs/physics/full_device_cfd/library")


@pytest.fixture(scope="module")
def full_device_library() -> VelocityFieldLibrary:
    if not (LIBRARY_PATH / "production_split_library.csv").exists():
        pytest.skip("Full-device production CFD library is not available")
    return VelocityFieldLibrary.from_directory(LIBRARY_PATH)


def test_full_device_exact_split_recovers_stored_field(full_device_library: VelocityFieldLibrary) -> None:
    case = full_device_library.case_for_fraction(0.5000000004936045)
    field = full_device_library.interpolate(0.5000000004936045)

    assert field.exact_match
    assert field.lower_case_id == "production_fL_0p5"
    assert field.upper_case_id == "production_fL_0p5"
    assert np.array_equal(field.velocity_dof_m_per_s, case.velocity_dof_m_per_s)
    assert np.array_equal(field.velocity_node_m_per_s, case.velocity_node_m_per_s)


def test_full_device_interpolation_between_adjacent_splits(full_device_library: VelocityFieldLibrary) -> None:
    low = full_device_library.case_for_fraction(0.5000000004936045)
    high = full_device_library.case_for_fraction(0.5200000001654056)
    target = 0.5 * (low.left_fraction + high.left_fraction)
    field = full_device_library.interpolate(target)

    assert not field.exact_match
    assert field.lower_case_id == "production_fL_0p5"
    assert field.upper_case_id == "production_fL_0p52"
    assert math.isclose(field.interpolation_weight, 0.5, abs_tol=1.0e-12)
    assert np.allclose(field.velocity_dof_m_per_s, 0.5 * (low.velocity_dof_m_per_s + high.velocity_dof_m_per_s))


def test_full_device_sampling_succeeds_in_all_major_regions(full_device_library: VelocityFieldLibrary) -> None:
    geometry = build_full_device_cfd_geometry()
    points = np.vstack(
        [
            geometry.centerlines["inlet"].points_um[len(geometry.centerlines["inlet"].points_um) // 2],
            geometry.upper_junction_um,
            geometry.centerlines["left"].points_um[len(geometry.centerlines["left"].points_um) // 2],
            geometry.centerlines["right"].points_um[len(geometry.centerlines["right"].points_um) // 2],
            geometry.lower_junction_um,
            geometry.centerlines["outlet"].points_um[len(geometry.centerlines["outlet"].points_um) // 2],
        ]
    )
    field = full_device_library.interpolate(0.5000000004936045)
    samples = field.sample_cfd(points)

    assert samples.cfd_valid.tolist() == [True] * 6
    assert np.isfinite(samples.cfd_u).all()
    assert np.isfinite(samples.cfd_v).all()
    assert np.isfinite(samples.cfd_speed).all()
    assert np.isfinite(samples.cfd_dir_x).all()
    assert np.isfinite(samples.cfd_dir_y).all()


def test_full_device_outside_domain_returns_nan_without_extrapolation(full_device_library: VelocityFieldLibrary) -> None:
    field = full_device_library.interpolate(0.5000000004936045)
    outside = np.array([[0.0, 0.0]])
    samples = field.sample_cfd(outside)

    assert samples.cfd_valid.tolist() == [False]
    assert np.isnan(samples.cfd_u[0])
    assert np.isnan(samples.cfd_v[0])
    assert np.isnan(samples.cfd_speed[0])
    assert np.isnan(samples.cfd_dir_x[0])
    assert np.isnan(samples.cfd_dir_y[0])


def test_full_device_scalar_and_vectorized_sampling_agree(full_device_library: VelocityFieldLibrary) -> None:
    geometry = build_full_device_cfd_geometry()
    point = geometry.centerlines["right"].points_um[len(geometry.centerlines["right"].points_um) // 2]
    field = full_device_library.interpolate(0.510000000329505)

    scalar = field.sample_cfd(point)
    vectorized = field.sample_cfd(point.reshape(1, 2))

    assert scalar.cfd_valid.shape == (1,)
    assert np.array_equal(scalar.cfd_valid, vectorized.cfd_valid)
    assert np.allclose(scalar.cfd_u, vectorized.cfd_u, equal_nan=True)
    assert np.allclose(scalar.cfd_v, vectorized.cfd_v, equal_nan=True)
    assert np.allclose(scalar.cfd_speed, vectorized.cfd_speed, equal_nan=True)
