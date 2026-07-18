from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .types import BranchCenterline, DeviceGeometry

BRANCH_ALIASES = ("branch", "branch_name", "branch_id", "region", "segment", "channel")
X_ALIASES = ("x", "x_px", "centerline_x")
Y_ALIASES = ("y", "y_px", "centerline_y")
ORDER_ALIASES = ("order", "point_index", "index", "sequence", "s_px")
EXPECTED_BRANCHES = ("inlet", "left", "right", "outlet")
BRANCH_NAME_MAP = {
    "inlet": "inlet",
    "outlet": "outlet",
    "left": "left",
    "right": "right",
}


def inspect_centerline_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a centerline CSV and report its schema diagnostics."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Centerline CSV does not exist: {path}")
    df = pd.read_csv(path)
    columns = list(df.columns)
    branch_col = _select_column(columns, BRANCH_ALIASES, "branch label", required=False)
    x_col = _select_column(columns, X_ALIASES, "x coordinate")
    y_col = _select_column(columns, Y_ALIASES, "y coordinate")
    order_col = _select_column(columns, ORDER_ALIASES, "point order", required=False)
    duplicate_xy = int(df.duplicated(subset=[x_col, y_col]).sum())
    info = {
        "columns": columns,
        "row_count": int(len(df)),
        "branch_column": branch_col,
        "x_column": x_col,
        "y_column": y_col,
        "order_column": order_col,
        "unique_branch_labels": sorted(df[branch_col].dropna().astype(str).unique().tolist()) if branch_col else [],
        "missing_value_counts": {key: int(value) for key, value in df.isna().sum().to_dict().items()},
        "duplicate_coordinate_count": duplicate_xy,
    }
    print("Centerline CSV schema")
    print(f"  columns: {info['columns']}")
    print(f"  row count: {info['row_count']}")
    print(f"  unique branch labels: {info['unique_branch_labels']}")
    print(f"  missing values: {info['missing_value_counts']}")
    print(f"  duplicate coordinate count: {duplicate_xy}")
    return df, info


def _select_column(columns: list[str], aliases: tuple[str, ...], label: str, required: bool = True) -> str | None:
    lowered = {column.lower(): column for column in columns}
    matches = [lowered[alias] for alias in aliases if alias in lowered]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {label} columns: {matches}")
    if not matches:
        if required:
            raise ValueError(f"Could not identify {label} column. Supported aliases: {aliases}")
        return None
    return matches[0]


def normalize_branch_names(labels: list[str]) -> dict[str, str]:
    """Map discovered labels to required branch names."""
    mapping: dict[str, str] = {}
    for label in labels:
        key = str(label).strip().lower()
        if key not in BRANCH_NAME_MAP:
            raise ValueError(
                f"Unknown branch label '{label}'. Add an explicit mapping in BRANCH_NAME_MAP."
            )
        mapping[str(label)] = BRANCH_NAME_MAP[key]
    normalized = set(mapping.values())
    missing = set(EXPECTED_BRANCHES).difference(normalized)
    if missing:
        raise ValueError(f"Missing expected physical branches after normalization: {sorted(missing)}")
    print(f"Discovered branch labels: {sorted(labels)}")
    print(f"Normalized branch mapping: {mapping}")
    return mapping


def cumulative_arc_length(xy: np.ndarray) -> np.ndarray:
    """Return cumulative Euclidean arc length in pixels."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise ValueError("xy must have shape (N, 2) with at least two points")
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(ds <= 0):
        raise ValueError("Centerline contains repeated adjacent points")
    return np.concatenate([[0.0], np.cumsum(ds)])


def tangent_normal_vectors(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit tangents and normals where normal=(-ty, tx)."""
    xy = np.asarray(xy, dtype=float)
    delta = np.empty_like(xy)
    delta[0] = xy[1] - xy[0]
    delta[-1] = xy[-1] - xy[-2]
    if len(xy) > 2:
        delta[1:-1] = xy[2:] - xy[:-2]
    norms = np.linalg.norm(delta, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-length tangent vector encountered")
    tangents = delta / norms[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return tangents, normals


def order_branch_points(points: np.ndarray, order_values: np.ndarray | None = None) -> np.ndarray:
    """Order one branch polyline from explicit order values or coordinate adjacency."""
    points = np.asarray(points, dtype=float)
    if order_values is not None:
        order = np.argsort(order_values, kind="mergesort")
        ordered = points[order]
        cumulative_arc_length(ordered)
        return ordered
    try:
        return reconstruct_path_order(points)
    except ValueError as exc:
        cumulative_arc_length(points)
        print(
            "Adjacency reconstruction failed; preserving CSV row order as the implicit "
            f"point sequence. Reason: {exc}"
        )
        return points


def reconstruct_path_order(points: np.ndarray) -> np.ndarray:
    """Reconstruct a single non-branching path from unsorted points."""
    points = np.asarray(points, dtype=float)
    if len(points) != len({tuple(row) for row in points}):
        raise ValueError("Duplicate coordinates prevent deterministic adjacency ordering")

    integer_like = np.allclose(points, np.round(points), atol=1e-6)
    if integer_like:
        adjacency = _integer_adjacency(points)
        if _is_single_path(adjacency):
            return _traverse_path(points, adjacency)
        degrees = [len(neighbors) for neighbors in adjacency]
        if any(degree > 2 for degree in degrees) or all(degree == 2 for degree in degrees):
            raise ValueError(
                "Integer centerline is forked or looped; "
                f"degree counts are {dict(sorted((degree, degrees.count(degree)) for degree in set(degrees)))}"
            )

    adjacency = _nearest_neighbor_adjacency(points)
    if not _is_single_path(adjacency):
        degrees = [len(neighbors) for neighbors in adjacency]
        raise ValueError(
            "Branch must be a single non-branching path; "
            f"degree counts are {dict(sorted((degree, degrees.count(degree)) for degree in set(degrees)))}"
        )
    return _traverse_path(points, adjacency)


def _integer_adjacency(points: np.ndarray) -> list[set[int]]:
    rounded = np.round(points).astype(int)
    by_coord = {tuple(coord): idx for idx, coord in enumerate(rounded)}
    adjacency = [set() for _ in range(len(points))]
    for idx, (x, y) in enumerate(rounded):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                other = by_coord.get((x + dx, y + dy))
                if other is not None:
                    adjacency[idx].add(other)
    return adjacency


def _nearest_neighbor_adjacency(points: np.ndarray) -> list[set[int]]:
    n_points = len(points)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    typical_spacing = float(np.median(nearest))
    tolerance = min(float(np.percentile(nearest, 95)) * 1.5, typical_spacing * 20.0)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Could not derive nearest-neighbor tolerance")
    print(f"Nearest-neighbor ordering tolerance: {tolerance:.3f} px (typical spacing {typical_spacing:.3f} px)")

    edges: list[tuple[float, int, int]] = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance = distances[i, j]
            if distance <= tolerance:
                edges.append((float(distance), i, j))
    edges.sort()
    adjacency = [set() for _ in range(n_points)]
    parent = list(range(n_points))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, i, j in edges:
        if len(adjacency[i]) >= 2 or len(adjacency[j]) >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj and sum(len(neighbors) for neighbors in adjacency) // 2 < n_points - 1:
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)
        parent[ri] = rj
        if sum(len(neighbors) for neighbors in adjacency) // 2 == n_points - 1:
            break
    return adjacency


def _is_single_path(adjacency: list[set[int]]) -> bool:
    degrees = [len(neighbors) for neighbors in adjacency]
    if degrees.count(1) != 2 or any(degree not in (1, 2) for degree in degrees):
        return False
    seen = set()
    stack = [degrees.index(1)]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node].difference(seen))
    return len(seen) == len(adjacency)


def _traverse_path(points: np.ndarray, adjacency: list[set[int]]) -> np.ndarray:
    endpoints = [idx for idx, neighbors in enumerate(adjacency) if len(neighbors) == 1]
    current = min(endpoints, key=lambda idx: (points[idx, 1], points[idx, 0]))
    previous = None
    order = []
    while True:
        order.append(current)
        candidates = [idx for idx in adjacency[current] if idx != previous]
        if not candidates:
            break
        previous, current = current, candidates[0]
    if len(order) != len(points):
        raise ValueError("Could not traverse all centerline points")
    return points[order]


def standardize_orientation(branch_points: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    """Orient inlet/loop/outlet branches with physical flow direction."""
    endpoints = {name: (points[0], points[-1]) for name, points in branch_points.items()}
    inlet_end = _closest_endpoint(endpoints["inlet"][1], [endpoints["left"][0], endpoints["left"][1], endpoints["right"][0], endpoints["right"][1]])
    outlet_start = _closest_endpoint(endpoints["outlet"][0], [endpoints["left"][0], endpoints["left"][1], endpoints["right"][0], endpoints["right"][1]])

    oriented = dict(branch_points)
    if np.linalg.norm(endpoints["inlet"][0] - inlet_end) < np.linalg.norm(endpoints["inlet"][1] - inlet_end):
        oriented["inlet"] = oriented["inlet"][::-1]
    upper = oriented["inlet"][-1]

    for name in ("left", "right"):
        points = oriented[name]
        if np.linalg.norm(points[-1] - upper) < np.linalg.norm(points[0] - upper):
            oriented[name] = points[::-1]

    lower_candidates = [oriented["left"][-1], oriented["right"][-1]]
    lower = np.mean(lower_candidates, axis=0)
    outlet = oriented["outlet"]
    if np.linalg.norm(outlet[-1] - lower) < np.linalg.norm(outlet[0] - lower):
        oriented["outlet"] = outlet[::-1]
    lower = oriented["outlet"][0]

    mismatch = {
        "upper_inlet_left_px": float(np.linalg.norm(oriented["inlet"][-1] - oriented["left"][0])),
        "upper_inlet_right_px": float(np.linalg.norm(oriented["inlet"][-1] - oriented["right"][0])),
        "lower_left_outlet_px": float(np.linalg.norm(oriented["left"][-1] - oriented["outlet"][0])),
        "lower_right_outlet_px": float(np.linalg.norm(oriented["right"][-1] - oriented["outlet"][0])),
    }
    spacing = np.median(
        np.concatenate([np.linalg.norm(np.diff(points, axis=0), axis=1) for points in oriented.values()])
    )
    tolerance = max(2.0, float(spacing) * 3.0)
    if any(distance > tolerance for distance in mismatch.values()):
        raise ValueError(f"Junction endpoint mismatch exceeds {tolerance:.3f} px: {mismatch}")
    return oriented, upper, lower, mismatch


def _closest_endpoint(point: np.ndarray, candidates: list[np.ndarray]) -> np.ndarray:
    distances = [np.linalg.norm(point - candidate) for candidate in candidates]
    return candidates[int(np.argmin(distances))]


def build_device_geometry(
    centerline_path: str | Path,
    device_config: dict[str, Any],
    length_tolerance_px: float = 1.0,
) -> tuple[DeviceGeometry, dict[str, Any]]:
    """Build typed device geometry and validation metadata from a centerline CSV."""
    df, schema = inspect_centerline_csv(centerline_path)
    branch_col = schema["branch_column"]
    x_col = schema["x_column"]
    y_col = schema["y_column"]
    order_col = schema["order_column"]
    if branch_col is None:
        raise ValueError("Centerline CSV requires a branch label column")

    mapping = normalize_branch_names(schema["unique_branch_labels"])
    raw_points: dict[str, np.ndarray] = {}
    for raw_name, normalized in mapping.items():
        subset = df[df[branch_col].astype(str) == raw_name]
        order_values = subset[order_col].to_numpy() if order_col else None
        raw_points[normalized] = order_branch_points(subset[[x_col, y_col]].to_numpy(float), order_values)

    oriented, upper, lower, mismatch = standardize_orientation(raw_points)
    device = device_config["device"]
    um_per_px = float(device["calibration"]["um_per_px"])
    branches: dict[str, BranchCenterline] = {}
    rows = []
    for branch_name in EXPECTED_BRANCHES:
        xy = oriented[branch_name]
        s_px = cumulative_arc_length(xy)
        s_um = s_px * um_per_px
        tangents, normals = tangent_normal_vectors(xy)
        branch = BranchCenterline(
            branch=branch_name,
            xy=xy,
            s_px=s_px,
            s_um=s_um,
            tangents=tangents,
            normals=normals,
            total_length_px=float(s_px[-1]),
            total_length_um=float(s_um[-1]),
        )
        branches[branch_name] = branch
        for idx in range(len(xy)):
            rows.append(
                {
                    "branch": branch_name,
                    "point_index": idx,
                    "x_px": xy[idx, 0],
                    "y_px": xy[idx, 1],
                    "s_px": s_px[idx],
                    "s_um": s_um[idx],
                    "tangent_x": tangents[idx, 0],
                    "tangent_y": tangents[idx, 1],
                    "normal_x": normals[idx, 0],
                    "normal_y": normals[idx, 1],
                }
            )

    geometry = DeviceGeometry(
        device_id=device["id"],
        calibration={"um_per_px": um_per_px},
        branches=branches,
        upper_junction_xy=upper,
        lower_junction_xy=lower,
        endpoint_mismatch_distances_px=mismatch,
    )
    validation = validate_loop_lengths(geometry, device_config, length_tolerance_px)
    metadata = {
        "schema": schema,
        "branch_mapping": mapping,
        "point_rows": rows,
        "length_validation": validation,
    }
    return geometry, metadata


def validate_loop_lengths(
    geometry: DeviceGeometry,
    device_config: dict[str, Any],
    tolerance_px: float = 1.0,
) -> dict[str, Any]:
    """Compare loop lengths against configured references."""
    device = device_config["device"]
    refs = device["loop"]["branches"]
    correction_px = float(device.get("channel", {}).get("width_px", 0.0)) * 2.0 - 1.0
    report: dict[str, Any] = {"branches": {}, "junction_correction_px": correction_px}
    for branch_name in ("left", "right"):
        branch = geometry.branches[branch_name]
        configured_px = float(refs[branch_name]["length_px"])
        configured_um = float(refs[branch_name]["length_um"])
        comparable_px = branch.total_length_px - correction_px
        comparable_um = comparable_px * float(geometry.calibration["um_per_px"])
        diff_px = abs(comparable_px - configured_px)
        report["branches"][branch_name] = {
            "raw_centerline_length_px": branch.total_length_px,
            "computed_length_px": comparable_px,
            "configured_length_px": configured_px,
            "absolute_difference_px": diff_px,
            "relative_difference_percent": diff_px / configured_px * 100.0,
            "computed_length_um": comparable_um,
            "configured_length_um": configured_um,
        }
        if diff_px > tolerance_px:
            raise ValueError(
                f"{branch_name} branch length differs by {diff_px:.3f} px, "
                f"exceeding tolerance {tolerance_px:.3f} px"
            )
    left = report["branches"]["left"]["computed_length_px"]
    right = report["branches"]["right"]["computed_length_px"]
    if left <= right:
        raise ValueError("Expected left branch to be longer than right branch")
    report["long_to_short_ratio"] = left / right
    report["asymmetry_alpha"] = (left - right) / right
    return report


def save_geometry_artifacts(
    geometry: DeviceGeometry,
    metadata: dict[str, Any],
    output_dir: str | Path,
    centerline_path: str | Path,
    device_config: dict[str, Any],
    overwrite: bool = False,
) -> None:
    """Save CSV, JSON, NPZ, and PNG geometry artifacts."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists. Use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metadata["point_rows"]).to_csv(output_dir / "centerline_geometry.csv", index=False)

    summary = _geometry_summary(geometry, metadata, centerline_path, device_config)
    with (output_dir / "geometry_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    arrays = {
        "upper_junction_xy": geometry.upper_junction_xy,
        "lower_junction_xy": geometry.lower_junction_xy,
    }
    for name, branch in geometry.branches.items():
        arrays[f"{name}_xy"] = branch.xy
        arrays[f"{name}_s_px"] = branch.s_px
        arrays[f"{name}_s_um"] = branch.s_um
        arrays[f"{name}_tangents"] = branch.tangents
        arrays[f"{name}_normals"] = branch.normals
    np.savez(output_dir / "device_geometry.npz", **arrays)
    _save_validation_plot(geometry, metadata, output_dir / "centerline_geometry.png")


def _geometry_summary(
    geometry: DeviceGeometry,
    metadata: dict[str, Any],
    centerline_path: str | Path,
    device_config: dict[str, Any],
) -> dict[str, Any]:
    validation = metadata["length_validation"]
    return {
        "device_id": geometry.device_id,
        "source_centerline_file": str(centerline_path),
        "calibration": geometry.calibration,
        "branch_point_counts": {name: int(len(branch.xy)) for name, branch in geometry.branches.items()},
        "branch_lengths": {
            name: {"px": branch.total_length_px, "um": branch.total_length_um}
            for name, branch in geometry.branches.items()
        },
        "configured_reference_lengths": device_config["device"]["loop"]["branches"],
        "length_differences": validation,
        "long_to_short_ratio": validation["long_to_short_ratio"],
        "asymmetry_alpha": validation["asymmetry_alpha"],
        "upper_junction_coordinate": geometry.upper_junction_xy.tolist(),
        "lower_junction_coordinate": geometry.lower_junction_xy.tolist(),
        "endpoint_mismatch_distances": geometry.endpoint_mismatch_distances_px,
        "coordinate_convention": "x increases right; y increases downward",
        "tangent_convention": "centered finite difference for interior points; one-sided at endpoints",
        "normal_convention": "normal_x=-tangent_y; normal_y=tangent_x",
        "schema": metadata["schema"],
        "normalized_branch_mapping": metadata["branch_mapping"],
    }


def _save_validation_plot(geometry: DeviceGeometry, metadata: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, branch in geometry.branches.items():
        ax.plot(branch.xy[:, 0], branch.xy[:, 1], label=name)
        mid = len(branch.xy) // 2
        ax.text(branch.xy[mid, 0], branch.xy[mid, 1], name)
        step = max(1, len(branch.xy) // 12)
        ax.quiver(
            branch.xy[::step, 0],
            branch.xy[::step, 1],
            branch.tangents[::step, 0],
            branch.tangents[::step, 1],
            angles="xy",
            scale_units="xy",
            scale=0.08,
            width=0.003,
        )
    ax.scatter(*geometry.upper_junction_xy, marker="o", c="black", label="upper junction")
    ax.scatter(*geometry.lower_junction_xy, marker="s", c="black", label="lower junction")
    lengths = metadata["length_validation"]["branches"]
    ax.set_title(
        "Centerline geometry "
        f"(left {lengths['left']['computed_length_px']:.2f} px, "
        f"right {lengths['right']['computed_length_px']:.2f} px)"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel("x_px")
    ax.set_ylabel("y_px")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
