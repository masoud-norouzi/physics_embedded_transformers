from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from src.physics.interpolation import VelocityFieldLibrary

from .coordinate_mapping import build_coordinate_transform, map_tracking_coordinates, transform_metadata
from .types import COORDINATE_TRANSFORM_VERSION, ENRICHMENT_VERSION, EnrichmentConfig, EnrichmentSummary
from .validation import (
    validate_hydraulic_state,
    validate_row_preservation,
    validate_sampled_fields,
    validate_tracking_hydraulic_join,
    validation_summary,
)


def build_physics_enriched_tracking(config: EnrichmentConfig, overwrite: bool = False) -> tuple[pd.DataFrame, EnrichmentSummary]:
    """Build a downstream physics-enriched tracking table without modifying upstream artifacts."""
    output_root = Path(config.output_root)
    output_csv = output_root / "physics_enriched_tracked_features.csv"
    summary_json = output_root / "physics_enriched_tracking_summary.json"
    summary_md = output_root / "physics_enriched_tracking_summary.md"
    diagnostics_dir = output_root / "diagnostics"
    outputs = [
        output_csv,
        summary_json,
        summary_md,
        diagnostics_dir / "sampled_field_overlay.png",
        diagnostics_dir / "inside_domain_by_frame.png",
        diagnostics_dir / "flow_direction_alignment.png",
    ]
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Enrichment outputs already exist. Use --overwrite: {existing}")
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    tracking = pd.read_csv(config.tracking_path)
    hydraulic = pd.read_csv(config.hydraulic_state_path)
    validate_hydraulic_state(hydraulic)

    transform = build_coordinate_transform(str(config.experiment_config_path))
    mapped = map_tracking_coordinates(tracking, transform)
    hydraulic_features = _prepare_hydraulic_features(hydraulic)
    joined = tracking.merge(hydraulic_features, on="frame", how="left", sort=False, validate="many_to_one")
    validate_tracking_hydraulic_join(tracking, hydraulic_features, joined)

    enriched = pd.concat([joined, mapped], axis=1)
    enriched = _order_columns(tracking.columns.tolist(), enriched)
    library = VelocityFieldLibrary.from_directory(config.cfd_library_path)
    sampled = _sample_cfd_background(enriched, library, transform.convention)
    enriched = pd.concat([enriched, sampled], axis=1)

    occupancy_regions = _load_occupancy_regions(config.occupancy_path)
    if occupancy_regions is not None:
        enriched = enriched.merge(occupancy_regions, on=["frame", "track_id"], how="left", sort=False, validate="one_to_one")

    observed = _derive_observed_velocity_direction(enriched)
    enriched = pd.concat([enriched, observed], axis=1)
    validate_row_preservation(tracking, enriched)
    validate_sampled_fields(enriched)
    direction_alignment = _flow_direction_alignment(enriched)
    inside_by_region = _inside_domain_by_region(enriched)

    missing_counts = {
        column: int(enriched[column].isna().sum())
        for column in [
            "cfd_u",
            "cfd_v",
            "cfd_speed",
            "cfd_dir_x",
            "cfd_dir_y",
            "background_u_x_device_m_per_s",
            "background_u_y_device_m_per_s",
            "background_speed_m_per_s",
            "background_direction_x",
            "background_direction_y",
        ]
    }
    summary = EnrichmentSummary(
        experiment_id=config.experiment_id,
        source_tracking_path=str(config.tracking_path),
        source_tracking_sha256=_sha256(config.tracking_path),
        hydraulic_input_path=str(config.hydraulic_state_path),
        hydraulic_input_sha256=_sha256(config.hydraulic_state_path),
        cfd_library_path=str(config.cfd_library_path),
        cfd_version=library.cases[0].cfd_version,
        mesh_version=library.cases[0].mesh_version,
        interpolation_module_version=_git_commit(),
        coordinate_transform_version=COORDINATE_TRANSFORM_VERSION,
        coordinate_transform_description=transform.description,
        output_path=str(output_csv),
        row_count=int(len(enriched)),
        column_count=int(len(enriched.columns)),
        original_column_count=int(len(tracking.columns)),
        inside_cfd_domain_rows=int(enriched["inside_cfd_domain"].sum()),
        inside_cfd_domain_fraction=float(enriched["inside_cfd_domain"].mean()),
        unique_tracks_inside_cfd_domain=int(enriched.loc[enriched["inside_cfd_domain"], "track_id"].nunique()),
        inside_domain_by_region=inside_by_region,
        flow_direction_alignment=direction_alignment,
        missing_value_counts=missing_counts,
        validation=validation_summary(tracking, enriched, hydraulic_features),
        generation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    enriched.to_csv(output_csv, index=False)
    summary_dict = asdict(summary)
    summary_dict["coordinate_transform"] = transform_metadata(transform)
    summary_json.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    summary_md.write_text(_markdown_summary(summary_dict), encoding="utf-8")
    _save_diagnostics(enriched, library, diagnostics_dir)
    return enriched, summary


def _prepare_hydraulic_features(hydraulic: pd.DataFrame) -> pd.DataFrame:
    total = hydraulic["total_mixture_input_flow_ul_hr"].to_numpy(float)
    if np.any(total <= 0):
        raise ValueError("total_mixture_input_flow_ul_hr must be positive")
    left_fraction = hydraulic["left_flow_ul_hr"].to_numpy(float) / total
    right_fraction = hydraulic["right_flow_ul_hr"].to_numpy(float) / total
    features = pd.DataFrame(
        {
            "frame": hydraulic["frame"].to_numpy(),
            "left_flow_fraction": left_fraction,
            "right_flow_fraction": right_fraction,
            "inlet_superficial_velocity_m_per_s": _flow_ul_hr_to_velocity_m_s(
                hydraulic["total_mixture_input_flow_ul_hr"].to_numpy(float)
            ),
            "left_branch_superficial_velocity_m_per_s": hydraulic["left_velocity_um_s"].to_numpy(float) * 1.0e-6,
            "right_branch_superficial_velocity_m_per_s": hydraulic["right_velocity_um_s"].to_numpy(float) * 1.0e-6,
        }
    )
    if not np.allclose(left_fraction + right_fraction, 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("Hydraulic left/right flow fractions do not sum to one")
    return features


def _flow_ul_hr_to_velocity_m_s(flow_ul_hr: np.ndarray, channel_width_um: float = 100.0, channel_height_um: float = 100.0) -> np.ndarray:
    flow_m3_s = flow_ul_hr * 1.0e-9 / 3600.0
    area_m2 = channel_width_um * 1.0e-6 * channel_height_um * 1.0e-6
    return flow_m3_s / area_m2


def _order_columns(original_columns: list[str], table: pd.DataFrame) -> pd.DataFrame:
    added = [column for column in table.columns if column not in original_columns]
    return table[original_columns + added]


def _sample_cfd_background(enriched: pd.DataFrame, library: VelocityFieldLibrary, convention=None) -> pd.DataFrame:
    points = _sampling_points(enriched, library)
    output = pd.DataFrame(index=enriched.index)
    for column in [
        "cfd_u",
        "cfd_v",
        "cfd_speed",
        "cfd_dir_x",
        "cfd_dir_y",
        "background_u_x_device_m_per_s",
        "background_u_y_device_m_per_s",
        "background_speed_m_per_s",
        "background_direction_x",
        "background_direction_y",
    ]:
        output[column] = np.nan
    output["cfd_valid"] = False
    output["inside_cfd_domain"] = False
    output["cfd_left_fraction"] = enriched["left_flow_fraction"].to_numpy(float)
    output["cfd_right_fraction"] = enriched["right_flow_fraction"].to_numpy(float)
    alpha_low, alpha_high, weight, exact = _neighbor_provenance(library.fractions, output["cfd_left_fraction"].to_numpy(float))
    output["cfd_alpha_low"] = alpha_low
    output["cfd_alpha_high"] = alpha_high
    output["cfd_interpolation_weight"] = weight
    output["cfd_exact_match"] = exact
    output["physics_enrichment_version"] = ENRICHMENT_VERSION
    output["coordinate_transform_version"] = COORDINATE_TRANSFORM_VERSION

    sampled_cases = {}
    valid_cases = {}
    for case in library.cases:
        samples = library.interpolate(case.left_fraction).sample_cfd(points)
        uv = np.column_stack([samples.cfd_u, samples.cfd_v])
        sampled_cases[case.left_fraction] = uv
        valid_cases[case.left_fraction] = samples.cfd_valid & np.isfinite(uv).all(axis=1)

    velocities = np.full((len(enriched), 2), np.nan, dtype=float)
    valid = np.zeros(len(enriched), dtype=bool)
    for low in np.unique(alpha_low):
        low = float(low)
        mask_low = np.isclose(alpha_low, low, atol=1.0e-12, rtol=0.0)
        for high in np.unique(alpha_high[mask_low]):
            high = float(high)
            mask = mask_low & np.isclose(alpha_high, high, atol=1.0e-12, rtol=0.0)
            w = weight[mask]
            velocities[mask] = (1.0 - w[:, None]) * sampled_cases[low][mask] + w[:, None] * sampled_cases[high][mask]
            valid[mask] = valid_cases[low][mask] & valid_cases[high][mask] & np.isfinite(velocities[mask]).all(axis=1)

    output.loc[valid, "cfd_valid"] = True
    output.loc[valid, "inside_cfd_domain"] = True
    output.loc[valid, "cfd_u"] = velocities[valid, 0]
    output.loc[valid, "cfd_v"] = velocities[valid, 1]
    cfd_speed = np.linalg.norm(velocities[valid], axis=1)
    output.loc[valid, "cfd_speed"] = cfd_speed
    cfd_dirs = np.full((len(cfd_speed), 2), np.nan, dtype=float)
    cfd_nonzero = cfd_speed > 1.0e-14
    cfd_dirs[cfd_nonzero] = velocities[valid][cfd_nonzero] / cfd_speed[cfd_nonzero, None]
    output.loc[valid, "cfd_dir_x"] = cfd_dirs[:, 0]
    output.loc[valid, "cfd_dir_y"] = cfd_dirs[:, 1]

    velocities_out = _vectors_to_device_if_needed(velocities[valid], library, convention)
    output.loc[valid, "background_u_x_device_m_per_s"] = velocities_out[:, 0]
    output.loc[valid, "background_u_y_device_m_per_s"] = velocities_out[:, 1]
    speed = np.linalg.norm(velocities_out, axis=1)
    output.loc[valid, "background_speed_m_per_s"] = speed
    nonzero = speed > 1.0e-14
    dirs = np.full((len(speed), 2), np.nan, dtype=float)
    dirs[nonzero] = velocities_out[nonzero] / speed[nonzero, None]
    output.loc[valid, "background_direction_x"] = dirs[:, 0]
    output.loc[valid, "background_direction_y"] = dirs[:, 1]
    return output


def _sampling_points(enriched: pd.DataFrame, library: VelocityFieldLibrary) -> np.ndarray:
    geometry = library.cases[0].mesh.geometry
    if hasattr(geometry, "coordinate_frame") and geometry.coordinate_frame == "device_cartesian_y_up":
        return enriched[["x_device_um", "y_device_um"]].to_numpy(float)
    return enriched[["x_cfd_um", "y_cfd_um"]].to_numpy(float)


def _vectors_to_device_if_needed(velocities: np.ndarray, library: VelocityFieldLibrary, convention) -> np.ndarray:
    geometry = library.cases[0].mesh.geometry
    if hasattr(geometry, "coordinate_frame") and geometry.coordinate_frame == "device_cartesian_y_up":
        return velocities
    return convention.cfd_vectors_to_device(velocities) if convention is not None else velocities


def _neighbor_provenance(fractions: tuple[float, ...], alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(fractions, dtype=float)
    if np.any(~np.isfinite(alpha)):
        raise ValueError("Hydraulic left fractions must be finite")
    if np.any(alpha < grid[0] - 1.0e-12) or np.any(alpha > grid[-1] + 1.0e-12):
        raise ValueError(f"Hydraulic left fractions are outside CFD interpolation range [{grid[0]}, {grid[-1]}]")
    exact = np.isclose(alpha[:, None], grid[None, :], atol=1.0e-12, rtol=0.0).any(axis=1)
    upper_idx = np.searchsorted(grid, alpha, side="right")
    exact_idx = np.argmin(np.abs(alpha[:, None] - grid[None, :]), axis=1)
    upper_idx[exact] = exact_idx[exact]
    upper_idx = np.clip(upper_idx, 1, len(grid) - 1)
    lower_idx = upper_idx - 1
    lower_idx[exact] = exact_idx[exact]
    low = grid[lower_idx]
    high = grid[upper_idx]
    denom = high - low
    weight = np.zeros_like(alpha, dtype=float)
    blended = denom > 0
    weight[blended] = (alpha[blended] - low[blended]) / denom[blended]
    return low, high, weight, exact


def _load_occupancy_regions(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    usecols = ["frame", "track_id", "dominant_region"]
    occupancy = pd.read_csv(path, usecols=usecols)
    if occupancy[["frame", "track_id"]].duplicated().any():
        return None
    return occupancy


def _derive_observed_velocity_direction(enriched: pd.DataFrame) -> pd.DataFrame:
    observed = pd.DataFrame(index=enriched.index)
    observed["observed_v_x_device_um_per_frame"] = np.nan
    observed["observed_v_y_device_um_per_frame"] = np.nan
    for _, group in enriched.groupby("track_id", sort=False):
        order = group.sort_values("frame").index.to_numpy()
        if len(order) < 3:
            continue
        frames = enriched.loc[order, "frame"].to_numpy(int)
        x = enriched.loc[order, "x_device_um"].to_numpy(float)
        y = enriched.loc[order, "y_device_um"].to_numpy(float)
        valid = (frames[1:-1] - frames[:-2] == 1) & (frames[2:] - frames[1:-1] == 1)
        middle = order[1:-1][valid]
        observed.loc[middle, "observed_v_x_device_um_per_frame"] = (x[2:][valid] - x[:-2][valid]) / 2.0
        observed.loc[middle, "observed_v_y_device_um_per_frame"] = (y[2:][valid] - y[:-2][valid]) / 2.0
    speed = np.sqrt(observed["observed_v_x_device_um_per_frame"] ** 2 + observed["observed_v_y_device_um_per_frame"] ** 2)
    observed["observed_speed_um_per_frame"] = speed
    observed["observed_direction_x_device"] = observed["observed_v_x_device_um_per_frame"] / speed
    observed["observed_direction_y_device"] = observed["observed_v_y_device_um_per_frame"] / speed
    observed.loc[speed <= 1.0e-12, ["observed_direction_x_device", "observed_direction_y_device"]] = np.nan
    return observed


def _flow_direction_alignment(enriched: pd.DataFrame) -> dict[str, float]:
    required = [
        "inside_cfd_domain",
        "background_direction_x",
        "background_direction_y",
        "observed_direction_x_device",
        "observed_direction_y_device",
    ]
    valid = enriched[required].notna().all(axis=1) & enriched["inside_cfd_domain"].astype(bool)
    if not valid.any():
        return {
            "valid_comparison_count": 0,
            "mean_cosine_similarity": float("nan"),
            "median_cosine_similarity": float("nan"),
            "mean_angular_difference_deg": float("nan"),
            "median_angular_difference_deg": float("nan"),
        }
    observed = enriched.loc[valid, ["observed_direction_x_device", "observed_direction_y_device"]].to_numpy(float)
    background = enriched.loc[valid, ["background_direction_x", "background_direction_y"]].to_numpy(float)
    cosine = np.clip(np.sum(observed * background, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    return {
        "valid_comparison_count": int(len(cosine)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "median_cosine_similarity": float(np.median(cosine)),
        "mean_angular_difference_deg": float(np.mean(angles)),
        "median_angular_difference_deg": float(np.median(angles)),
    }


def _inside_domain_by_region(enriched: pd.DataFrame) -> dict[str, int]:
    if "dominant_region" not in enriched.columns:
        return {}
    values = enriched.loc[enriched["inside_cfd_domain"], "dominant_region"].fillna("unknown")
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _save_diagnostics(enriched: pd.DataFrame, library: VelocityFieldLibrary, diagnostics_dir: Path) -> None:
    _save_overlay(enriched, library, diagnostics_dir / "sampled_field_overlay.png")
    _save_inside_by_frame(enriched, diagnostics_dir / "inside_domain_by_frame.png")
    _save_alignment(enriched, diagnostics_dir / "flow_direction_alignment.png")


def _save_overlay(enriched: pd.DataFrame, library: VelocityFieldLibrary, path: Path) -> None:
    subset = enriched
    inside = subset["inside_cfd_domain"].to_numpy(bool)
    points = _sampling_points(subset, library)
    mesh = library.cases[0].mesh
    tri = mtri.Triangulation(mesh.nodes_um[:, 0], mesh.nodes_um[:, 1], mesh.elements)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.triplot(tri, color="#94a3b8", linewidth=0.35, alpha=0.7)
    ax.scatter(points[~inside, 0], points[~inside, 1], s=4, color="#9ca3af", alpha=0.35, label="invalid CFD")
    ax.scatter(points[inside, 0], points[inside, 1], s=5, color="#ef4444", alpha=0.45, label="valid CFD")
    if inside.any():
        valid_positions = np.flatnonzero(inside)
        stride = max(1, len(valid_positions) // 1200)
        draw = valid_positions[::stride]
        ax.quiver(
            points[draw, 0],
            points[draw, 1],
            subset.iloc[draw]["cfd_u"],
            subset.iloc[draw]["cfd_v"],
            color="#0f766e",
            scale=0.8,
            width=0.002,
            alpha=0.65,
        )
    ax.set_title("Sampled full-device CFD vectors on tracked droplet centroids")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_inside_by_frame(enriched: pd.DataFrame, path: Path) -> None:
    counts = enriched.groupby("frame")["inside_cfd_domain"].agg(["sum", "count"])
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(counts.index, counts["sum"], linewidth=1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("droplets inside CFD domain")
    ax.set_title("Full-device CFD coverage over time")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_alignment(enriched: pd.DataFrame, path: Path) -> None:
    valid = (
        enriched["inside_cfd_domain"].astype(bool)
        & enriched[["background_direction_x", "background_direction_y", "observed_direction_x_device", "observed_direction_y_device"]].notna().all(axis=1)
    )
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    if valid.any():
        observed = enriched.loc[valid, ["observed_direction_x_device", "observed_direction_y_device"]].to_numpy(float)
        background = enriched.loc[valid, ["background_direction_x", "background_direction_y"]].to_numpy(float)
        cosine = np.clip(np.sum(observed * background, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        ax.hist(angles, bins=40, color="#2563eb", alpha=0.8)
    ax.set_xlabel("angular difference (deg)")
    ax.set_ylabel("row count")
    ax.set_title("Observed droplet direction vs sampled background flow")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _markdown_summary(summary: dict[str, Any]) -> str:
    alignment = summary["flow_direction_alignment"]
    lines = [
        "# Physics-Enriched Tracking Summary",
        "",
        f"- Experiment ID: `{summary['experiment_id']}`",
        f"- Source tracking: `{summary['source_tracking_path']}`",
        f"- Hydraulic state: `{summary['hydraulic_input_path']}`",
        f"- CFD library: `{summary['cfd_library_path']}`",
        f"- Output: `{summary['output_path']}`",
        f"- Rows: {summary['row_count']}",
        f"- Columns: {summary['column_count']}",
        f"- Valid CFD rows: {summary['inside_cfd_domain_rows']} ({summary['inside_cfd_domain_fraction']:.2%})",
        f"- Invalid CFD rows: {summary['row_count'] - summary['inside_cfd_domain_rows']}",
        f"- Unique tracks entering CFD domain: {summary['unique_tracks_inside_cfd_domain']}",
        "",
        "## Coordinate Transform",
        "",
        summary["coordinate_transform_description"],
        "",
        "## Flow-Direction Alignment",
        "",
        f"- Valid comparisons: {alignment['valid_comparison_count']}",
        f"- Mean cosine similarity: {alignment['mean_cosine_similarity']:.6f}",
        f"- Median cosine similarity: {alignment['median_cosine_similarity']:.6f}",
        f"- Mean angular difference: {alignment['mean_angular_difference_deg']:.3f} deg",
        f"- Median angular difference: {alignment['median_angular_difference_deg']:.3f} deg",
        "",
        "## Missing Background-Field Values",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary["missing_value_counts"].items())
    lines.extend(["", "## Validation", ""])
    lines.extend(f"- {key}: {value}" for key, value in summary["validation"].items())
    if summary["inside_domain_by_region"]:
        lines.extend(["", "## Inside-Domain Rows by Region", ""])
        lines.extend(f"- {key}: {value}" for key, value in summary["inside_domain_by_region"].items())
    return "\n".join(lines)
