from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoordinateConvention:
    """Point and vector transforms between image, device, and frozen-CFD frames.

    Image frame:
        x_px right, y_px downward, origin at the upper-left image pixel.

    Device Cartesian frame:
        x_device_um right, y_device_um upward, fixed origin at the lower-left
        image reference line y_reference_px.

    Frozen CFD Version 1 native frame:
        x_cfd_um right, y_cfd_um downward, origin inherited from the calibrated
        full-image centerline. This is a native data frame, not the preferred
        device Cartesian frame for cross-module vectors.
    """

    pixel_scale_um_per_px: float
    y_reference_px: float
    cfd_origin_device_um: tuple[float, float] = (0.0, 0.0)

    @property
    def y_reference_um(self) -> float:
        return float(self.pixel_scale_um_per_px * self.y_reference_px)

    @property
    def point_image_to_device_matrix(self) -> np.ndarray:
        return np.array([[self.pixel_scale_um_per_px, 0.0], [0.0, -self.pixel_scale_um_per_px]])

    @property
    def point_image_to_device_offset(self) -> np.ndarray:
        return np.array([0.0, self.y_reference_um])

    @property
    def point_device_to_cfd_matrix(self) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, -1.0]])

    @property
    def point_device_to_cfd_offset(self) -> np.ndarray:
        origin = np.asarray(self.cfd_origin_device_um, dtype=float)
        return np.array([-origin[0], self.y_reference_um + origin[1]])

    def image_points_to_device(self, points_px: np.ndarray) -> np.ndarray:
        points = _points(points_px)
        return points @ self.point_image_to_device_matrix.T + self.point_image_to_device_offset

    def device_points_to_image(self, points_device_um: np.ndarray) -> np.ndarray:
        points = _points(points_device_um)
        x_px = points[:, 0] / self.pixel_scale_um_per_px
        y_px = self.y_reference_px - points[:, 1] / self.pixel_scale_um_per_px
        return np.column_stack([x_px, y_px])

    def image_vectors_to_device(self, vectors_px: np.ndarray) -> np.ndarray:
        vectors = _points(vectors_px)
        return vectors @ self.point_image_to_device_matrix.T

    def device_vectors_to_image(self, vectors_device_um: np.ndarray) -> np.ndarray:
        vectors = _points(vectors_device_um)
        return np.column_stack(
            [
                vectors[:, 0] / self.pixel_scale_um_per_px,
                -vectors[:, 1] / self.pixel_scale_um_per_px,
            ]
        )

    def device_points_to_cfd(self, points_device_um: np.ndarray) -> np.ndarray:
        points = _points(points_device_um)
        return points @ self.point_device_to_cfd_matrix.T + self.point_device_to_cfd_offset

    def cfd_points_to_device(self, points_cfd_um: np.ndarray) -> np.ndarray:
        points = _points(points_cfd_um)
        origin = np.asarray(self.cfd_origin_device_um, dtype=float)
        return np.column_stack([points[:, 0] + origin[0], self.y_reference_um + origin[1] - points[:, 1]])

    def device_vectors_to_cfd(self, vectors_device: np.ndarray) -> np.ndarray:
        vectors = _points(vectors_device)
        return vectors @ self.point_device_to_cfd_matrix.T

    def cfd_vectors_to_device(self, vectors_cfd: np.ndarray) -> np.ndarray:
        vectors = _points(vectors_cfd)
        return vectors @ self.point_device_to_cfd_matrix.T

    def image_points_to_cfd(self, points_px: np.ndarray) -> np.ndarray:
        return self.device_points_to_cfd(self.image_points_to_device(points_px))

    def cfd_points_to_image(self, points_cfd_um: np.ndarray) -> np.ndarray:
        return self.device_points_to_image(self.cfd_points_to_device(points_cfd_um))

    def image_vectors_to_cfd(self, vectors_px: np.ndarray) -> np.ndarray:
        return self.device_vectors_to_cfd(self.image_vectors_to_device(vectors_px))

    def cfd_vectors_to_image(self, vectors_cfd: np.ndarray) -> np.ndarray:
        return self.device_vectors_to_image(self.cfd_vectors_to_device(vectors_cfd))


def _points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected an array with shape (N, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Coordinate and vector arrays must be finite")
    return points
