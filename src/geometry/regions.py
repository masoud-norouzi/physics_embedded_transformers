from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd


class RegionLabel(IntEnum):
    """Integer IDs stored in device region label maps."""

    UNASSIGNED = 0
    INLET = 1
    OUTLET = 2
    LEFT_BRANCH = 3
    RIGHT_BRANCH = 4
    UPPER_JUNCTION = 5
    LOWER_JUNCTION = 6


LABEL_NAMES = {
    RegionLabel.UNASSIGNED: "unassigned",
    RegionLabel.INLET: "inlet",
    RegionLabel.OUTLET: "outlet",
    RegionLabel.LEFT_BRANCH: "left_branch",
    RegionLabel.RIGHT_BRANCH: "right_branch",
    RegionLabel.UPPER_JUNCTION: "upper_junction",
    RegionLabel.LOWER_JUNCTION: "lower_junction",
}

BRANCH_TO_LABEL = {
    "inlet": RegionLabel.INLET,
    "outlet": RegionLabel.OUTLET,
    "left": RegionLabel.LEFT_BRANCH,
    "right": RegionLabel.RIGHT_BRANCH,
}


@dataclass(frozen=True)
class JunctionDefinition:
    """Pixel-space junction box derived from physical device config."""

    name: str
    center_px: np.ndarray
    size_um: np.ndarray
    size_px: np.ndarray


def junction_mask(shape: tuple[int, int], junction: JunctionDefinition) -> np.ndarray:
    """Rasterize a centered, axis-aligned junction box."""
    if np.any(junction.size_px <= 0) or not np.all(np.isfinite(junction.center_px)):
        raise ValueError(f"Invalid junction definition: {junction}")
    height, width = shape
    yy, xx = np.indices((height, width))
    half = junction.size_px / 2.0
    x0, y0 = junction.center_px - half
    x1, y1 = junction.center_px + half
    return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)


def load_junctions(device_config: dict[str, Any]) -> dict[str, JunctionDefinition]:
    """Read validated junction definitions from the device configuration."""
    device = device_config["device"]
    um_per_px = float(device["calibration"]["um_per_px"])
    if um_per_px <= 0:
        raise ValueError("Pixel scale um_per_px must be positive")
    junction_config = device.get("geometry", {}).get("junctions", {})
    junctions = {}
    for name in ("upper", "lower"):
        raw = junction_config.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"Missing geometry.junctions.{name} definition")
        center_px = np.asarray(raw.get("center_px"), dtype=float)
        size_um = np.asarray(raw.get("size_um"), dtype=float)
        if center_px.shape != (2,) or size_um.shape != (2,):
            raise ValueError(f"Junction {name} must define center_px and size_um as length-2 arrays")
        if np.any(size_um <= 0) or not np.all(np.isfinite(size_um)):
            raise ValueError(f"Junction {name} has invalid physical dimensions")
        junctions[name] = JunctionDefinition(
            name=name,
            center_px=center_px,
            size_um=size_um,
            size_px=size_um / um_per_px,
        )
    return junctions


def build_region_label_map(
    channel_mask: np.ndarray,
    centerlines: pd.DataFrame,
    device_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a mutually exclusive six-region label map from mask and centerlines."""
    mask = np.asarray(channel_mask, dtype=bool)
    required = {"x", "y", "channel"}
    missing = required.difference(centerlines.columns)
    if missing:
        raise ValueError(f"Centerline file is missing columns: {sorted(missing)}")

    junctions = load_junctions(device_config)
    upper = junction_mask(mask.shape, junctions["upper"]) & mask
    lower = junction_mask(mask.shape, junctions["lower"]) & mask
    if np.any(upper & lower):
        raise ValueError("Upper and lower junction masks overlap")

    label_map = np.zeros(mask.shape, dtype=np.uint8)
    label_map[upper] = int(RegionLabel.UPPER_JUNCTION)
    label_map[lower] = int(RegionLabel.LOWER_JUNCTION)

    assignable = mask & (label_map == int(RegionLabel.UNASSIGNED))
    assigned_by_branch = _nearest_centerline_assignment(assignable, centerlines)
    overlap_before_precedence = int(np.count_nonzero(upper & lower))
    for branch, region_label in BRANCH_TO_LABEL.items():
        label_map[assigned_by_branch == branch] = int(region_label)

    diagnostics = validate_region_label_map(label_map, mask)
    diagnostics.update(
        {
            "overlap_pixels_before_precedence_resolution": overlap_before_precedence,
            "junctions": junctions,
            "connectivity": check_region_connectivity(label_map),
        }
    )
    failed = [name for name, ok in diagnostics["connectivity"].items() if not ok]
    if failed:
        raise ValueError(f"Region connectivity checks failed: {failed}")
    return label_map, diagnostics


def _nearest_centerline_assignment(assignable: np.ndarray, centerlines: pd.DataFrame) -> np.ndarray:
    coords_yx = np.argwhere(assignable)
    assignment = np.full(assignable.shape, "", dtype=object)
    if len(coords_yx) == 0:
        return assignment
    coords_xy = coords_yx[:, [1, 0]].astype(float)
    best_dist = np.full(len(coords_xy), np.inf)
    best_branch = np.full(len(coords_xy), "", dtype=object)
    for branch in ("inlet", "outlet", "left", "right"):
        points = centerlines[centerlines["channel"].astype(str) == branch][["x", "y"]].to_numpy(float)
        if len(points) == 0:
            raise ValueError(f"Missing centerline points for branch '{branch}'")
        dist = _min_squared_distances(coords_xy, points)
        update = dist < best_dist
        best_dist[update] = dist[update]
        best_branch[update] = branch
    assignment[coords_yx[:, 0], coords_yx[:, 1]] = best_branch
    return assignment


def _min_squared_distances(coords_xy: np.ndarray, points_xy: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    out = np.empty(len(coords_xy), dtype=float)
    for start in range(0, len(coords_xy), chunk_size):
        chunk = coords_xy[start : start + chunk_size]
        delta = chunk[:, None, :] - points_xy[None, :, :]
        out[start : start + chunk_size] = np.min(np.sum(delta * delta, axis=2), axis=1)
    return out


def validate_region_label_map(label_map: np.ndarray, channel_mask: np.ndarray) -> dict[str, Any]:
    """Validate label-map shape, IDs, exclusivity, and mask containment."""
    labels = np.asarray(label_map)
    mask = np.asarray(channel_mask, dtype=bool)
    if labels.shape != mask.shape:
        raise ValueError(f"Label-map shape {labels.shape} differs from channel-mask shape {mask.shape}")
    valid_ids = {int(label) for label in RegionLabel}
    present_ids = {int(value) for value in np.unique(labels)}
    unknown = present_ids.difference(valid_ids)
    if unknown:
        raise ValueError(f"Unknown region label IDs: {sorted(unknown)}")
    assigned_outside = int(np.count_nonzero((labels != 0) & ~mask))
    if assigned_outside:
        raise ValueError(f"Found {assigned_outside} assigned pixels outside channel mask")
    counts = {LABEL_NAMES[RegionLabel(label)]: int(np.count_nonzero(labels == label)) for label in valid_ids}
    for label in RegionLabel:
        if label == RegionLabel.UNASSIGNED:
            continue
        if counts[LABEL_NAMES[label]] == 0:
            raise ValueError(f"Physical region is empty: {LABEL_NAMES[label]}")
    total_channel = int(np.count_nonzero(mask))
    unassigned_inside = int(np.count_nonzero((labels == 0) & mask))
    background = int(np.count_nonzero(~mask))
    assigned_inside = total_channel - unassigned_inside
    return {
        "label_counts": counts,
        "total_channel_pixels": total_channel,
        "unassigned_within_channel_pixels": unassigned_inside,
        "background_pixel_count": background,
        "assigned_pixels_outside_channel": assigned_outside,
        "assigned_channel_fraction": assigned_inside / total_channel if total_channel else 0.0,
    }


def check_region_connectivity(label_map: np.ndarray) -> dict[str, bool]:
    """Check that branch regions touch their expected junctions."""
    labels = np.asarray(label_map)
    upper = labels == int(RegionLabel.UPPER_JUNCTION)
    lower = labels == int(RegionLabel.LOWER_JUNCTION)
    return {
        "inlet_connects_upper_junction": _touches(labels == int(RegionLabel.INLET), upper),
        "outlet_connects_lower_junction": _touches(labels == int(RegionLabel.OUTLET), lower),
        "left_branch_connects_upper_and_lower_junctions": _touches_both(
            labels == int(RegionLabel.LEFT_BRANCH), upper, lower
        ),
        "right_branch_connects_upper_and_lower_junctions": _touches_both(
            labels == int(RegionLabel.RIGHT_BRANCH), upper, lower
        ),
    }


def _touches(region: np.ndarray, target: np.ndarray) -> bool:
    dilated = _dilate8(region)
    return bool(np.any(dilated & target))


def _touches_both(region: np.ndarray, first: np.ndarray, second: np.ndarray) -> bool:
    component = _component_touching(region, first)
    return component is not None and bool(np.any(_dilate8(component) & second))


def _component_touching(region: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    starts = np.argwhere(region & _dilate8(target))
    if len(starts) == 0:
        return None
    visited = np.zeros(region.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque([tuple(starts[0])])
    visited[tuple(starts[0])] = True
    height, width = region.shape
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width and region[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    queue.append((yy, xx))
    return visited


def _dilate8(region: np.ndarray) -> np.ndarray:
    padded = np.pad(region, 1, mode="constant", constant_values=False)
    out = np.zeros_like(region, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + region.shape[0], dx : dx + region.shape[1]]
    return out
