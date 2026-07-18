from dataclasses import dataclass

import numpy as np


def _validate_array(name: str, value: np.ndarray, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape_tail and array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have trailing shape {shape_tail}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class BranchCenterline:
    """Ordered centerline geometry for one labeled branch."""

    branch: str
    xy: np.ndarray
    s_px: np.ndarray
    s_um: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    total_length_px: float
    total_length_um: float

    def __post_init__(self) -> None:
        xy = _validate_array("xy", self.xy, (2,))
        s_px = _validate_array("s_px", self.s_px)
        s_um = _validate_array("s_um", self.s_um)
        tangents = _validate_array("tangents", self.tangents, (2,))
        normals = _validate_array("normals", self.normals, (2,))
        n_points = xy.shape[0]
        if n_points < 2:
            raise ValueError(f"Branch {self.branch} must contain at least two points")
        for name, array in [("s_px", s_px), ("s_um", s_um), ("tangents", tangents), ("normals", normals)]:
            if array.shape[0] != n_points:
                raise ValueError(f"{name} length does not match xy for branch {self.branch}")
        if np.any(np.diff(s_px) < -1e-12) or np.any(np.diff(s_um) < -1e-12):
            raise ValueError(f"Arc length must be non-decreasing for branch {self.branch}")
        if not np.isclose(s_px[-1], self.total_length_px):
            raise ValueError(f"s_px endpoint does not match total_length_px for branch {self.branch}")
        if not np.isclose(s_um[-1], self.total_length_um):
            raise ValueError(f"s_um endpoint does not match total_length_um for branch {self.branch}")


@dataclass(frozen=True)
class DeviceGeometry:
    """Geometry representation for one physical device."""

    device_id: str
    calibration: dict[str, float]
    branches: dict[str, BranchCenterline]
    upper_junction_xy: np.ndarray
    lower_junction_xy: np.ndarray
    endpoint_mismatch_distances_px: dict[str, float]

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("device_id must be non-empty")
        if "um_per_px" not in self.calibration:
            raise ValueError("calibration must include um_per_px")
        if not np.isfinite(float(self.calibration["um_per_px"])):
            raise ValueError("um_per_px must be finite")
        expected = {"inlet", "left", "right", "outlet"}
        missing = expected.difference(self.branches)
        if missing:
            raise ValueError(f"Missing required branches: {sorted(missing)}")
        _validate_array("upper_junction_xy", self.upper_junction_xy)
        _validate_array("lower_junction_xy", self.lower_junction_xy)
