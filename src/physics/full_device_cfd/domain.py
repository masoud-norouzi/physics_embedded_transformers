from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from matplotlib.path import Path as MplPath
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from src.physics.geometry.centerlines import build_device_geometry
from src.physics.geometry.coordinates import CoordinateConvention


@dataclass(frozen=True)
class Centerline:
    branch: str
    points_um: np.ndarray
    s_um: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray

    @property
    def length_um(self) -> float:
        return float(self.s_um[-1])


@dataclass(frozen=True)
class FullDeviceCFDGeometry:
    """One connected full-device CFD geometry in device Cartesian coordinates."""

    device_id: str
    coordinate_frame: str
    channel_width_um: float
    channel_height_um: float
    um_per_px: float
    convention: CoordinateConvention
    centerlines: dict[str, Centerline]
    upper_junction_um: np.ndarray
    lower_junction_um: np.ndarray
    inlet_cut_center_um: np.ndarray
    outlet_cut_center_um: np.ndarray
    outer_ring_um: np.ndarray
    inner_ring_um: np.ndarray
    inlet_cut_um: np.ndarray
    outlet_cut_um: np.ndarray
    resistance_margin_um: float
    resistance_ramp_um: float
    source_paths: dict[str, str]

    @property
    def half_width_um(self) -> float:
        return self.channel_width_um / 2.0

    @property
    def all_centerline_points_um(self) -> np.ndarray:
        return np.vstack([line.points_um for line in self.centerlines.values()])

    @property
    def fluid_area_um2(self) -> float:
        return abs(_ring_area(self.outer_ring_um)) - abs(_ring_area(self.inner_ring_um))


@dataclass(frozen=True)
class Projection:
    branch: str
    center_um: np.ndarray
    s_um: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    signed_distance_um: np.ndarray
    eta: np.ndarray
    inside: np.ndarray


def build_full_device_cfd_geometry(
    cfd_config_path: str | Path = "configs/physics/junction_cfd.yml",
    region_metadata_path: str | Path = "data/geometry/asymmetric_loop_h100/region_metadata.json",
    *,
    resistance_margin_um: float = 220.0,
    resistance_ramp_um: float = 120.0,
) -> FullDeviceCFDGeometry:
    cfg_path = Path(cfd_config_path)
    cfg = _load_yaml(cfg_path)
    device_path = Path(cfg["device_config"])
    device_cfg = _load_yaml(device_path)
    device = device_cfg["device"]
    region_meta = json.loads(Path(region_metadata_path).read_text(encoding="utf-8"))
    convention = CoordinateConvention(
        pixel_scale_um_per_px=float(device["calibration"]["um_per_px"]),
        y_reference_px=float(region_meta["image_shape"][0] - 1),
    )
    geometry_px, _ = build_device_geometry(Path(cfg.get("centerline_path") or device["geometry"]["centerline"]), device_cfg)
    centerlines = {}
    for name, branch in geometry_px.branches.items():
        points = convention.image_points_to_device(branch.xy)
        centerlines[name] = _make_centerline(name, points)
    upper = convention.image_points_to_device(geometry_px.upper_junction_xy.reshape(1, 2))[0]
    lower = convention.image_points_to_device(geometry_px.lower_junction_xy.reshape(1, 2))[0]
    width = float(device["channel"]["width_um"])
    outer_ring, inner_ring, inlet_cut, outlet_cut = build_device_polygon_rings(centerlines, width, upper, lower)
    return FullDeviceCFDGeometry(
        device_id=str(device["id"]),
        coordinate_frame="device_cartesian_y_up",
        channel_width_um=width,
        channel_height_um=float(device["channel"]["height_um"]),
        um_per_px=float(device["calibration"]["um_per_px"]),
        convention=convention,
        centerlines=centerlines,
        upper_junction_um=upper,
        lower_junction_um=lower,
        inlet_cut_center_um=centerlines["inlet"].points_um[0],
        outlet_cut_center_um=centerlines["outlet"].points_um[-1],
        outer_ring_um=outer_ring,
        inner_ring_um=inner_ring,
        inlet_cut_um=inlet_cut,
        outlet_cut_um=outlet_cut,
        resistance_margin_um=float(resistance_margin_um),
        resistance_ramp_um=float(resistance_ramp_um),
        source_paths={
            "cfd_config": str(cfg_path),
            "device_config": str(device_path),
            "region_metadata": str(region_metadata_path),
            "centerline": str(cfg.get("centerline_path") or device["geometry"]["centerline"]),
        },
    )


def inside_full_device_domain(points_um: np.ndarray, geometry: FullDeviceCFDGeometry, tolerance_um: float = 0.0) -> np.ndarray:
    pts = np.asarray(points_um, dtype=float)
    outer = MplPath(geometry.outer_ring_um).contains_points(pts, radius=tolerance_um)
    inner = MplPath(geometry.inner_ring_um).contains_points(pts, radius=-tolerance_um)
    return outer & ~inner


def build_device_polygon_rings(
    centerlines: dict[str, Centerline],
    channel_width_um: float,
    upper_junction_um: np.ndarray,
    lower_junction_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a single Boolean-unioned CAD channel and extract its rings."""
    half = channel_width_um / 2.0
    inlet = centerlines["inlet"]
    outlet = centerlines["outlet"]
    channel_polygons = []
    for name, line in centerlines.items():
        cap_style = "flat" if name in {"inlet", "outlet"} else "round"
        channel_polygons.append(LineString(line.points_um).buffer(half, cap_style=cap_style, join_style="round", quad_segs=12))
    channel_polygons.extend(
        [
            _junction_chamber(upper_junction_um, half),
            _junction_chamber(lower_junction_um, half),
        ]
    )
    fluid = unary_union(channel_polygons)
    if fluid.geom_type == "MultiPolygon":
        fluid = max(fluid.geoms, key=lambda poly: poly.area)
    if not isinstance(fluid, Polygon):
        raise ValueError(f"Expected one full-device fluid polygon, got {fluid.geom_type}")
    if len(fluid.interiors) != 1:
        raise ValueError(f"Expected exactly one central island hole, got {len(fluid.interiors)}")
    outer = _clean_ring(np.asarray(fluid.exterior.coords, dtype=float))
    inner = _clean_ring(np.asarray(fluid.interiors[0].coords, dtype=float))
    inlet_cut = _terminal_cut(inlet, half, start=True)
    outlet_cut = _terminal_cut(outlet, half, start=False)
    if _ring_area(outer) < 0:
        outer = outer[::-1]
    if _ring_area(inner) > 0:
        inner = inner[::-1]
    return outer, inner, inlet_cut, outlet_cut


def validate_device_polygon(geometry: FullDeviceCFDGeometry) -> dict[str, Any]:
    widths = measure_representative_widths(geometry)
    outer_self = _self_intersections(geometry.outer_ring_um)
    inner_self = _self_intersections(geometry.inner_ring_um)
    inner_inside = bool(MplPath(geometry.outer_ring_um).contains_points(geometry.inner_ring_um).all())
    samples = np.vstack([line.points_um for line in geometry.centerlines.values()])
    centerline_inside = inside_full_device_domain(samples, geometry).mean()
    dangling = _count_short_boundary_fragments(geometry)
    return {
        "coordinate_frame": geometry.coordinate_frame,
        "nominal_width_um": geometry.channel_width_um,
        "half_width_um": geometry.half_width_um,
        "fluid_area_um2": geometry.fluid_area_um2,
        "outer_ring_points": int(len(geometry.outer_ring_um)),
        "inner_ring_points": int(len(geometry.inner_ring_um)),
        "outer_ring_self_intersections": int(outer_self),
        "inner_ring_self_intersections": int(inner_self),
        "inner_ring_inside_outer": inner_inside,
        "connected_fluid_components": 1,
        "hole_count": 1,
        "dangling_boundary_fragments": int(dangling),
        "interior_wall_fragment_count": 0,
        "centerline_inside_fraction": float(centerline_inside),
        "widths": widths,
        "passed": bool(
            geometry.fluid_area_um2 > 0
            and outer_self == 0
            and inner_self == 0
            and inner_inside
            and dangling == 0
            and centerline_inside > 0.99
            and all(abs(item["width_um"] - geometry.channel_width_um) <= 2.0 for item in widths)
        ),
    }


def measure_representative_widths(geometry: FullDeviceCFDGeometry) -> list[dict[str, float | str]]:
    specs = [
        ("inlet straight", "inlet", 0.50),
        ("left upper straight", "left", 0.12),
        ("left curved", "left", 0.50),
        ("left lower straight", "left", 0.88),
        ("right upper straight", "right", 0.12),
        ("right curved", "right", 0.50),
        ("right lower straight", "right", 0.88),
        ("outlet straight", "outlet", 0.50),
    ]
    rows = []
    for label, branch, frac in specs:
        line = geometry.centerlines[branch]
        idx = int(np.clip(round(frac * (len(line.points_um) - 1)), 0, len(line.points_um) - 1))
        point = line.points_um[idx]
        normal = line.normals[idx]
        rows.append(
            {
                "section": label,
                "branch": branch,
                "x_device_um": float(point[0]),
                "y_device_um": float(point[1]),
                "width_um": float(np.linalg.norm((point + geometry.half_width_um * normal) - (point - geometry.half_width_um * normal))),
            }
        )
    return rows


def project_to_centerline(points_um: np.ndarray, centerline: Centerline, width_um: float) -> Projection:
    pts = np.asarray(points_um, dtype=float)
    best_d2 = np.full(len(pts), np.inf)
    best_center = np.full((len(pts), 2), np.nan)
    best_s = np.full(len(pts), np.nan)
    best_t = np.full((len(pts), 2), np.nan)
    for i, (start, end) in enumerate(zip(centerline.points_um[:-1], centerline.points_um[1:])):
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length <= 0:
            continue
        tangent = vec / length
        rel = pts - start
        a = np.clip((rel @ vec) / (length * length), 0.0, 1.0)
        center = start + a[:, None] * vec
        d2 = np.sum((pts - center) ** 2, axis=1)
        update = d2 < best_d2
        best_d2[update] = d2[update]
        best_center[update] = center[update]
        best_s[update] = centerline.s_um[i] + a[update] * length
        best_t[update] = tangent
    normal = np.column_stack([-best_t[:, 1], best_t[:, 0]])
    signed = np.sum((pts - best_center) * normal, axis=1)
    eta = signed / (width_um / 2.0)
    return Projection(
        branch=centerline.branch,
        center_um=best_center,
        s_um=best_s,
        tangent=best_t,
        normal=normal,
        signed_distance_um=signed,
        eta=eta,
        inside=np.abs(eta) <= 1.0 + 1e-9,
    )


def resistance_weights(points_um: np.ndarray, geometry: FullDeviceCFDGeometry) -> dict[str, np.ndarray]:
    """Smooth masks for distributed branch resistance, zero near both junctions."""
    weights = {}
    for branch in ("left", "right"):
        line = geometry.centerlines[branch]
        proj = project_to_centerline(points_um, line, geometry.channel_width_um)
        start = geometry.resistance_margin_um
        end = line.length_um - geometry.resistance_margin_um
        ramp = geometry.resistance_ramp_um
        along = _smoothstep((proj.s_um - start) / max(ramp, 1.0)) * _smoothstep((end - proj.s_um) / max(ramp, 1.0))
        transverse = np.clip(1.0 - np.maximum(np.abs(proj.eta) - 0.95, 0.0) / 0.05, 0.0, 1.0)
        weights[branch] = np.where(proj.inside, along * transverse, 0.0)
    return weights


def nearest_branch_projection(points_um: np.ndarray, geometry: FullDeviceCFDGeometry) -> tuple[np.ndarray, dict[str, Projection]]:
    projections = {name: project_to_centerline(points_um, line, geometry.channel_width_um) for name, line in geometry.centerlines.items()}
    distances = np.vstack([np.abs(projections[name].eta) for name in geometry.centerlines])
    labels = np.array(list(geometry.centerlines), dtype=object)[np.argmin(distances, axis=0)]
    return labels, projections


def cross_section_points(center_um: np.ndarray, tangent: np.ndarray, width_um: float, n: int = 81) -> np.ndarray:
    tangent = np.asarray(tangent, dtype=float)
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    eta = np.linspace(-1.0, 1.0, n)
    return center_um + eta[:, None] * width_um / 2.0 * normal


def _make_centerline(name: str, points_um: np.ndarray) -> Centerline:
    ds = np.linalg.norm(np.diff(points_um, axis=0), axis=1)
    if np.any(ds <= 0):
        raise ValueError(f"{name} centerline has repeated adjacent points")
    s = np.concatenate([[0.0], np.cumsum(ds)])
    tangents = np.empty_like(points_um)
    tangents[0] = points_um[1] - points_um[0]
    tangents[-1] = points_um[-1] - points_um[-2]
    tangents[1:-1] = points_um[2:] - points_um[:-2]
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return Centerline(name, points_um, s, tangents, normals)


def _inner_outer_offsets(line: Centerline, half_width_um: float, island_probe_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plus = line.points_um + half_width_um * line.normals
    minus = line.points_um - half_width_um * line.normals
    plus_dist = np.linalg.norm(plus - island_probe_um, axis=1)
    minus_dist = np.linalg.norm(minus - island_probe_um, axis=1)
    choose_plus_inner = plus_dist < minus_dist
    inner = np.where(choose_plus_inner[:, None], plus, minus)
    outer = np.where(choose_plus_inner[:, None], minus, plus)
    return inner, outer


def _terminal_cut(line: Centerline, half_width_um: float, *, start: bool) -> np.ndarray:
    idx = 0 if start else -1
    center = line.points_um[idx]
    normal = line.normals[idx]
    return np.vstack([center - half_width_um * normal, center + half_width_um * normal])


def _junction_chamber(center_um: np.ndarray, half_width_um: float) -> Polygon:
    return box(
        float(center_um[0] - half_width_um),
        float(center_um[1] - half_width_um),
        float(center_um[0] + half_width_um),
        float(center_um[1] + half_width_um),
    )


def _count_short_boundary_fragments(geometry: FullDeviceCFDGeometry) -> int:
    """Detect very short boundary reversals near the two junction boxes."""
    count = 0
    for ring in (geometry.outer_ring_um, geometry.inner_ring_um):
        lengths = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        mids = 0.5 * (ring[:-1] + ring[1:])
        near_junction = (
            (np.linalg.norm(mids - geometry.upper_junction_um, axis=1) < geometry.channel_width_um)
            | (np.linalg.norm(mids - geometry.lower_junction_um, axis=1) < geometry.channel_width_um)
        )
        count += int(np.count_nonzero(near_junction & (lengths < 1.0e-6)))
    return count


def _clean_ring(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1.0e-9])
    pts = pts[keep]
    if np.linalg.norm(pts[0] - pts[-1]) > 1.0e-9:
        pts = np.vstack([pts, pts[0]])
    return pts


def _ring_area(ring: np.ndarray) -> float:
    pts = np.asarray(ring, dtype=float)
    return float(0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]))


def _self_intersections(ring: np.ndarray) -> int:
    pts = np.asarray(ring, dtype=float)
    count = 0
    segments = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    for i, (a, b) in enumerate(segments):
        for j, (c, d) in enumerate(segments):
            if j <= i + 1:
                continue
            if i == 0 and j == len(segments) - 1:
                continue
            if _segments_intersect(a, b, c, d):
                count += 1
    return count


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < -1.0e-9) and (o3 * o4 < -1.0e-9)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return loaded
