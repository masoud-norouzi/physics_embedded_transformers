from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import numpy as np
import yaml

from src.physics.geometry.centerlines import build_device_geometry, cumulative_arc_length, tangent_normal_vectors


@dataclass(frozen=True)
class JunctionGeometry:
    """Idealized upper-junction CFD geometry stored in micrometers."""

    device_id: str
    source_centerline_path: Path
    coordinate_units: str
    channel_width_um: float
    channel_width_px: float
    um_per_px: float
    junction_padding_um: float
    target_element_size_um: float
    boundary_refinement_factor: float
    junction_center_um: np.ndarray
    branch_centerlines_um: dict[str, np.ndarray]
    branch_arc_lengths_um: dict[str, np.ndarray]
    branch_tangents: dict[str, np.ndarray]
    branch_normals: dict[str, np.ndarray]
    boundary_endpoints_um: dict[str, np.ndarray]
    boundary_sections_um: dict[str, np.ndarray]
    assumptions: list[str]

    @property
    def half_width_um(self) -> float:
        return self.channel_width_um / 2.0

    @property
    def all_centerline_points_um(self) -> np.ndarray:
        return np.vstack(list(self.branch_centerlines_um.values()))


def load_junction_cfd_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a junction-CFD config from a path or return a shallow dict copy."""
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(f"Junction CFD config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Junction CFD config must be a non-empty mapping: {path}")
    return loaded


def build_junction_geometry(config: str | Path | dict[str, Any]) -> JunctionGeometry:
    """Build an idealized upper-bifurcation channel patch in micrometers."""
    cfg = load_junction_cfd_config(config)
    device_config_path = Path(_required(cfg, "device_config"))
    if not device_config_path.exists():
        raise FileNotFoundError(f"Device config does not exist: {device_config_path}")
    with device_config_path.open("r", encoding="utf-8") as handle:
        device_config = yaml.safe_load(handle)
    if not isinstance(device_config, dict) or "device" not in device_config:
        raise ValueError(f"Device config must contain a top-level device mapping: {device_config_path}")

    device = device_config["device"]
    centerline_path = Path(cfg.get("centerline_path") or device["geometry"]["centerline"])
    geometry, _ = build_device_geometry(centerline_path, device_config)
    um_per_px = float(device["calibration"]["um_per_px"])
    channel_width_um = float(_required(cfg, "channel_width_um"))
    channel_width_px = float(_required(cfg, "channel_width_px"))
    padding_um = float(_required(cfg, "junction_padding_um"))
    target_size_um = float(_required(cfg, "target_element_size_um"))
    refinement = float(_required(cfg, "boundary_refinement_factor"))
    if min(channel_width_um, channel_width_px, padding_um, target_size_um, refinement) <= 0:
        raise ValueError("Geometry and mesh sizing config values must be positive")

    junction_um = geometry.upper_junction_xy * um_per_px
    patch: dict[str, np.ndarray] = {}
    for branch_name in ("inlet", "left", "right"):
        branch = geometry.branches[branch_name]
        xy_um = branch.xy * um_per_px
        if branch_name == "inlet":
            patch[branch_name] = _remove_adjacent_duplicates(_take_upstream_segment(xy_um, padding_um))
        else:
            patch[branch_name] = _remove_adjacent_duplicates(_take_downstream_segment(xy_um, padding_um))

    arc_lengths = {name: cumulative_arc_length(points) for name, points in patch.items()}
    tangents: dict[str, np.ndarray] = {}
    normals: dict[str, np.ndarray] = {}
    for name, points in patch.items():
        tangent, normal = tangent_normal_vectors(points)
        tangents[name] = tangent
        normals[name] = normal
    boundary_endpoints = {
        "inlet": patch["inlet"][0],
        "left_outlet": patch["left"][-1],
        "right_outlet": patch["right"][-1],
    }
    boundary_sections = {
        "inlet": _cross_section(patch["inlet"][0], tangents["inlet"][0], channel_width_um),
        "left_outlet": _cross_section(patch["left"][-1], tangents["left"][-1], channel_width_um),
        "right_outlet": _cross_section(patch["right"][-1], tangents["right"][-1], channel_width_um),
    }

    return JunctionGeometry(
        device_id=str(device["id"]),
        source_centerline_path=centerline_path,
        coordinate_units="um",
        channel_width_um=channel_width_um,
        channel_width_px=channel_width_px,
        um_per_px=um_per_px,
        junction_padding_um=padding_um,
        target_element_size_um=target_size_um,
        boundary_refinement_factor=refinement,
        junction_center_um=junction_um,
        branch_centerlines_um=patch,
        branch_arc_lengths_um=arc_lengths,
        branch_tangents=tangents,
        branch_normals=normals,
        boundary_endpoints_um=boundary_endpoints,
        boundary_sections_um=boundary_sections,
        assumptions=[
            "The computational patch represents the upper inlet bifurcation only.",
            "The channel is idealized as a constant-width finite strip around the extracted centerlines.",
            "The inlet and outlet boundaries are flat cross-sections normal to their local branch centerlines.",
            "Segmentation masks are not used as CFD boundaries.",
            "Coordinates are stored in micrometers throughout the geometry and mesh pipeline.",
        ],
    )


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Junction CFD config is missing required key: {key}")
    return config[key]


def _take_downstream_segment(points_um: np.ndarray, length_um: float) -> np.ndarray:
    s = cumulative_arc_length(points_um)
    keep = s <= length_um
    if keep.sum() < 2:
        raise ValueError("Downstream branch segment is too short for requested patch")
    return _append_interpolated_endpoint(points_um, s, length_um, keep)


def _take_upstream_segment(points_um: np.ndarray, length_um: float) -> np.ndarray:
    reversed_points = points_um[::-1]
    segment = _take_downstream_segment(reversed_points, length_um)
    return segment[::-1]


def _append_interpolated_endpoint(
    points_um: np.ndarray,
    s_um: np.ndarray,
    length_um: float,
    keep: np.ndarray,
) -> np.ndarray:
    selected = points_um[keep]
    if np.isclose(selected[-1], points_um[-1]).all() or length_um >= s_um[-1]:
        return selected
    idx = int(np.searchsorted(s_um, length_um))
    if idx <= 0 or idx >= len(points_um):
        return selected
    t = (length_um - s_um[idx - 1]) / (s_um[idx] - s_um[idx - 1])
    endpoint = points_um[idx - 1] + t * (points_um[idx] - points_um[idx - 1])
    return np.vstack([selected, endpoint])


def _remove_adjacent_duplicates(points_um: np.ndarray) -> np.ndarray:
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points_um, axis=0), axis=1) > 1e-9])
    cleaned = points_um[keep]
    if len(cleaned) < 2:
        raise ValueError("Cleaned junction branch segment has fewer than two points")
    return cleaned


def distance_to_centerline_segments(points: np.ndarray, geometry: JunctionGeometry) -> np.ndarray:
    """Return distance in micrometers from points to the idealized branch centerlines."""
    query = np.asarray(points, dtype=float)
    distances = np.full(query.shape[0], np.inf, dtype=float)
    for centerline in geometry.branch_centerlines_um.values():
        starts = centerline[:-1]
        ends = centerline[1:]
        seg = ends - starts
        seg_len2 = np.sum(seg * seg, axis=1)
        valid = seg_len2 > 0
        starts = starts[valid]
        seg = seg[valid]
        seg_len2 = seg_len2[valid]
        for start, vector, length2 in zip(starts, seg, seg_len2):
            rel = query - start
            t = np.clip(np.sum(rel * vector, axis=1) / length2, 0.0, 1.0)
            projected = start + t[:, None] * vector
            distances = np.minimum(distances, np.linalg.norm(query - projected, axis=1))
    return distances


def inside_junction_domain(points: np.ndarray, geometry: JunctionGeometry, tolerance_um: float = 0.0) -> np.ndarray:
    """Return whether points are inside the idealized finite-width strip domain."""
    query = np.asarray(points, dtype=float)
    inside = np.zeros(query.shape[0], dtype=bool)
    half = geometry.half_width_um + tolerance_um
    for centerline in geometry.branch_centerlines_um.values():
        starts = centerline[:-1]
        ends = centerline[1:]
        seg = ends - starts
        seg_len2 = np.sum(seg * seg, axis=1)
        valid = seg_len2 > 0
        for start, vector, length2 in zip(starts[valid], seg[valid], seg_len2[valid]):
            rel = query - start
            t = np.sum(rel * vector, axis=1) / length2
            projected = start + t[:, None] * vector
            perpendicular = np.linalg.norm(query - projected, axis=1)
            inside |= (t >= -1e-9) & (t <= 1.0 + 1e-9) & (perpendicular <= half)
    inside |= _inside_junction_core(query, geometry, tolerance_um)
    return inside


def _inside_junction_core(points: np.ndarray, geometry: JunctionGeometry, tolerance_um: float) -> np.ndarray:
    """Fill the branch-overlap construction seam at the bifurcation."""
    inlet_axis = geometry.branch_tangents["inlet"][-1]
    outlet_axis = geometry.branch_tangents["right"][0]
    inlet_axis = inlet_axis / np.linalg.norm(inlet_axis)
    outlet_axis = outlet_axis / np.linalg.norm(outlet_axis)
    rel = np.asarray(points, dtype=float) - geometry.junction_center_um
    half = geometry.half_width_um + tolerance_um
    along_inlet = np.abs(rel @ inlet_axis)
    along_outlet = np.abs(rel @ outlet_axis)
    return (along_inlet <= half) & (along_outlet <= half * 1.35)


def classify_boundary_points(points: np.ndarray, geometry: JunctionGeometry) -> np.ndarray:
    """Classify points as inlet, outlets, wall, or interior using stored endpoints."""
    pts = np.asarray(points, dtype=float)
    labels = np.full(pts.shape[0], "interior", dtype=object)
    cut_tol = max(geometry.target_element_size_um * 0.25, 1e-9)
    wall_tol = max(geometry.target_element_size_um * 0.25, 1e-9)
    labels[_near_strip_wall(pts, geometry, wall_tol)] = "wall"
    for name, section in geometry.boundary_sections_um.items():
        near = _distance_to_segment(pts, section[0], section[1]) <= cut_tol
        labels[near] = name
    return labels


def _cross_section(center_um: np.ndarray, tangent: np.ndarray, width_um: float) -> np.ndarray:
    tangent = np.asarray(tangent, dtype=float)
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    return np.vstack([center_um - normal * width_um / 2.0, center_um + normal * width_um / 2.0])


def _near_strip_wall(points: np.ndarray, geometry: JunctionGeometry, tolerance_um: float) -> np.ndarray:
    query = np.asarray(points, dtype=float)
    candidate_probe = np.full_like(query, np.nan, dtype=float)
    candidate = np.zeros(query.shape[0], dtype=bool)
    half = geometry.half_width_um
    for centerline in geometry.branch_centerlines_um.values():
        starts = centerline[:-1]
        ends = centerline[1:]
        seg = ends - starts
        seg_len2 = np.sum(seg * seg, axis=1)
        valid = seg_len2 > 0
        for start, vector, length2 in zip(starts[valid], seg[valid], seg_len2[valid]):
            rel = query - start
            t = np.sum(rel * vector, axis=1) / length2
            projected = start + t[:, None] * vector
            offset = query - projected
            perpendicular = np.linalg.norm(offset, axis=1)
            segment_candidate = (t >= -1e-9) & (t <= 1.0 + 1e-9) & (np.abs(perpendicular - half) <= tolerance_um)
            outward = np.zeros_like(offset)
            nonzero = perpendicular > 1e-12
            outward[nonzero] = offset[nonzero] / perpendicular[nonzero, None]
            update = segment_candidate & ~candidate
            candidate_probe[update] = query[update] + outward[update] * max(tolerance_um, 1.0)
            candidate |= segment_candidate
    near = np.zeros(query.shape[0], dtype=bool)
    if np.any(candidate):
        exterior = ~inside_junction_domain(candidate_probe[candidate], geometry, tolerance_um=0.0)
        near[np.flatnonzero(candidate)[exterior]] = True
    return near


def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    query = np.asarray(points, dtype=float)
    vector = end - start
    length2 = float(np.dot(vector, vector))
    if length2 <= 0:
        return np.linalg.norm(query - start, axis=1)
    t = np.clip(((query - start) @ vector) / length2, 0.0, 1.0)
    projected = start + t[:, None] * vector
    return np.linalg.norm(query - projected, axis=1)
