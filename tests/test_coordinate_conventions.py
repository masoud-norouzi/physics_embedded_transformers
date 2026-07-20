from __future__ import annotations

import numpy as np
import pytest

from src.physics.geometry.coordinate_audit import _audit_points
from src.physics.geometry.coordinates import CoordinateConvention
from src.physics.interpolation import VelocityFieldLibrary


def test_image_device_point_round_trip() -> None:
    convention = CoordinateConvention(pixel_scale_um_per_px=4.0, y_reference_px=596.0)
    points = np.array([[0.0, 0.0], [330.0, 215.0], [623.0, 596.0]])

    round_trip = convention.device_points_to_image(convention.image_points_to_device(points))

    np.testing.assert_allclose(round_trip, points)


def test_device_cfd_point_round_trip() -> None:
    convention = CoordinateConvention(pixel_scale_um_per_px=4.0, y_reference_px=596.0)
    points = np.array([[1320.0, 1524.0], [1016.0, 1524.0], [1624.0, 1522.0]])

    round_trip = convention.cfd_points_to_device(convention.device_points_to_cfd(points))

    np.testing.assert_allclose(round_trip, points)


def test_image_device_vector_round_trip_and_y_reflection() -> None:
    convention = CoordinateConvention(pixel_scale_um_per_px=4.0, y_reference_px=596.0)
    vectors_px = np.array([[1.0, 2.0], [-3.0, -4.0]])

    device = convention.image_vectors_to_device(vectors_px)
    round_trip = convention.device_vectors_to_image(device)

    np.testing.assert_allclose(device, [[4.0, -8.0], [-12.0, 16.0]])
    np.testing.assert_allclose(round_trip, vectors_px)


def test_translations_do_not_affect_vectors() -> None:
    base = CoordinateConvention(pixel_scale_um_per_px=4.0, y_reference_px=596.0, cfd_origin_device_um=(0.0, 0.0))
    shifted = CoordinateConvention(pixel_scale_um_per_px=4.0, y_reference_px=596.0, cfd_origin_device_um=(100.0, 200.0))
    vector = np.array([[3.0, -5.0]])

    np.testing.assert_allclose(base.device_vectors_to_cfd(vector), shifted.device_vectors_to_cfd(vector))


@pytest.fixture(scope="module")
def exact_field():
    return VelocityFieldLibrary.from_directory("outputs/physics/junction_cfd/solutions").interpolate(0.50)


def test_known_inlet_and_outlet_points_map_inside_cfd_domain(exact_field) -> None:
    points = np.vstack([item[1] for item in _audit_points(exact_field.mesh.geometry)])
    sampled = exact_field.sample_cfd(points)

    assert sampled.inside_domain.all()


def test_stored_inlet_velocity_points_toward_junction_and_outlets_away(exact_field) -> None:
    samples = _audit_points(exact_field.mesh.geometry)
    points = np.vstack([item[1] for item in samples])
    sampled = exact_field.sample_cfd(points)
    velocities = np.column_stack([sampled.u_x_m_per_s, sampled.u_y_m_per_s])

    for (_, _, expected, _), velocity in zip(samples, velocities):
        assert float(velocity @ expected) > 0.0


def test_interpolated_exact_grid_sampling_preserves_stored_velocity_direction(exact_field) -> None:
    point = _audit_points(exact_field.mesh.geometry)[0][1][None, :]
    stored = exact_field.sample_cfd(point)
    sampled = exact_field.sample(point)

    assert stored.coordinate_frame == "cfd_native_y_down"
    np.testing.assert_allclose(sampled.u_x_m_per_s, stored.u_x_m_per_s)
    np.testing.assert_allclose(sampled.u_y_m_per_s, stored.u_y_m_per_s)


def test_device_frame_sampling_returns_reflected_cfd_vector(exact_field) -> None:
    convention = CoordinateConvention(pixel_scale_um_per_px=4.0, y_reference_px=596.0)
    point_cfd = _audit_points(exact_field.mesh.geometry)[0][1][None, :]
    point_device = convention.cfd_points_to_device(point_cfd)

    sampled_cfd = exact_field.sample_cfd(point_cfd)
    sampled_device = exact_field.sample_device(point_device, convention)

    assert sampled_device.coordinate_frame == "device_cartesian_y_up"
    np.testing.assert_allclose(sampled_device.u_x_m_per_s, sampled_cfd.u_x_m_per_s)
    np.testing.assert_allclose(sampled_device.u_y_m_per_s, -sampled_cfd.u_y_m_per_s)
