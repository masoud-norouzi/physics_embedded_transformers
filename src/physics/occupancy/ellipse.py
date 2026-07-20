from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EllipseRaster:
    """Local rasterized footprint for an axis-aligned bbox ellipse."""

    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray
    image_boundary_clipped: bool

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.mask))


def rasterize_bbox_ellipse(
    center_x: float,
    center_y: float,
    bbox_width: float,
    bbox_height: float,
    image_shape: tuple[int, int],
) -> EllipseRaster:
    """Rasterize an axis-aligned ellipse at pixel centers inside a local window.

    Pixel centers are taken at integer image coordinates: x is the column index
    and y is the row index, matching the tracking and region-label convention.
    """
    values = np.asarray([center_x, center_y, bbox_width, bbox_height], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Ellipse centroid and bounding-box values must be finite")
    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError("Bounding-box width and height must be positive")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")

    rx = bbox_width / 2.0
    ry = bbox_height / 2.0
    raw_x0 = int(np.floor(center_x - rx))
    raw_x1 = int(np.ceil(center_x + rx)) + 1
    raw_y0 = int(np.floor(center_y - ry))
    raw_y1 = int(np.ceil(center_y + ry)) + 1
    x0 = max(0, raw_x0)
    y0 = max(0, raw_y0)
    x1 = min(width, raw_x1)
    y1 = min(height, raw_y1)
    clipped = x0 != raw_x0 or y0 != raw_y0 or x1 != raw_x1 or y1 != raw_y1
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Ellipse bounding window does not intersect the image")

    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
    xx += x0
    yy += y0
    mask = ((xx - center_x) / rx) ** 2 + ((yy - center_y) / ry) ** 2 <= 1.0
    if not np.any(mask):
        raise ValueError("Ellipse contains zero raster pixels")
    return EllipseRaster(y0=y0, y1=y1, x0=x0, x1=x1, mask=mask, image_boundary_clipped=clipped)
