from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_experiment_config
from src.physics.geometry.coordinates import CoordinateConvention

from .types import COORDINATE_TRANSFORM_VERSION


@dataclass(frozen=True)
class CoordinateTransform:
    """Explicit map from tracking pixels to CFD physical coordinates."""

    um_per_px: float
    y_reference_px: float
    tracking_x_column: str
    tracking_y_column: str
    image_origin: str = "upper_left"
    y_axis_orientation: str = "image_y_down"
    cfd_origin_description: str = "Version 1 CFD mesh uses the same full-device physical coordinate frame"
    transform_version: str = COORDINATE_TRANSFORM_VERSION

    @property
    def description(self) -> str:
        return (
            "x_device_um = x_px * um_per_px; "
            "y_device_um = (y_reference_px - y_px) * um_per_px; "
            "x_cfd_um = x_device_um; y_cfd_um = y_reference_um - y_device_um. "
            "Vectors use the linear parts only, so image/device and device/CFD transforms each reflect y exactly once."
        )

    @property
    def convention(self) -> CoordinateConvention:
        return CoordinateConvention(self.um_per_px, self.y_reference_px)


def build_coordinate_transform(experiment_config_path: str) -> CoordinateTransform:
    loaded = load_experiment_config(experiment_config_path)
    device = loaded["device"]["device"]
    um_per_px = float(device["calibration"]["um_per_px"])
    if um_per_px <= 0 or not np.isfinite(um_per_px):
        raise ValueError("Device calibration um_per_px must be positive and finite")
    y_reference_px = _image_y_reference_px(device)
    return CoordinateTransform(
        um_per_px=um_per_px,
        y_reference_px=y_reference_px,
        tracking_x_column="centroid_x",
        tracking_y_column="centroid_y",
    )


def map_tracking_coordinates(tracking: pd.DataFrame, transform: CoordinateTransform) -> pd.DataFrame:
    """Return x/y physical coordinates and local CFD coordinates in micrometers."""
    missing = {transform.tracking_x_column, transform.tracking_y_column}.difference(tracking.columns)
    if missing:
        raise ValueError(f"Tracking table is missing coordinate columns: {sorted(missing)}")
    pixels = tracking[[transform.tracking_x_column, transform.tracking_y_column]].to_numpy(float)
    if not np.isfinite(pixels).all():
        raise ValueError("Tracking coordinate columns must be finite")
    device = transform.convention.image_points_to_device(pixels)
    cfd = transform.convention.device_points_to_cfd(device)
    return pd.DataFrame(
        {
            "x_device_um": device[:, 0],
            "y_device_um": device[:, 1],
            "x_cfd_um": cfd[:, 0],
            "y_cfd_um": cfd[:, 1],
        },
        index=tracking.index,
    )


def transform_metadata(transform: CoordinateTransform) -> dict[str, Any]:
    return {
        "version": transform.transform_version,
        "um_per_px": transform.um_per_px,
        "y_reference_px": transform.y_reference_px,
        "y_reference_um": transform.convention.y_reference_um,
        "tracking_x_column": transform.tracking_x_column,
        "tracking_y_column": transform.tracking_y_column,
        "image_origin": transform.image_origin,
        "y_axis_orientation": transform.y_axis_orientation,
        "cfd_origin_description": transform.cfd_origin_description,
        "equations": transform.description,
    }


def _image_y_reference_px(device: dict[str, Any]) -> float:
    metadata_path = device.get("geometry", {}).get("region_metadata_path")
    if metadata_path and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        shape = metadata.get("image_shape")
        if isinstance(shape, list) and len(shape) >= 1:
            return float(shape[0] - 1)
    raise ValueError("Device geometry metadata must provide a fixed image_shape for y_reference_px")
