from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from src.physics.full_device_cfd.domain import FullDeviceCFDGeometry, build_full_device_cfd_geometry


REGION_LABELS = {
    "inlet": "inlet channel",
    "upper_junction": "inlet junction",
    "left_branch": "left branch",
    "right_branch": "right branch",
    "lower_junction": "outlet junction",
    "outlet": "outlet channel",
}


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.enriched_csv)
    geometry = build_full_device_cfd_geometry()
    normalized = _with_regions(table)

    summary = _regional_summary(normalized)
    summary_path = out / "regional_cfd_validity_summary.csv"
    summary.to_csv(summary_path, index=False)
    _print_summary(summary)
    _save_invalid_bar_chart(summary, out / "invalid_percentage_by_region.png")

    _save_invalid_overview(normalized, geometry, out / "invalid_samples_full_device_overlay.png")
    invalid_regions = [region for region in REGION_LABELS.values() if int(summary.loc[summary["region"] == region, "invalid_rows"].iloc[0]) > 0]
    for region in invalid_regions:
        _save_invalid_zoom(normalized, geometry, region, out / f"invalid_samples_zoom_{_token(region)}.png")

    _print_random_invalid_samples(normalized, geometry, seed=args.seed, count=args.random_count)
    print(f"\nArtifacts written to: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose full-device CFD validity in the physics-enriched tracking dataset.")
    parser.add_argument(
        "--enriched-csv",
        type=Path,
        default=Path("outputs/physics/video_2/enrichment/physics_enriched_tracked_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/physics/video_2/enrichment/diagnostics/final_validity"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-count", type=int, default=20)
    return parser.parse_args()


def _with_regions(table: pd.DataFrame) -> pd.DataFrame:
    required = {"frame", "track_id", "x_device_um", "y_device_um", "cfd_valid", "dominant_region"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Enriched dataset is missing required columns: {sorted(missing)}")
    out = table.copy()
    out["region"] = out["dominant_region"].map(REGION_LABELS).fillna("unknown")
    out["cfd_valid"] = out["cfd_valid"].astype(bool)
    return out


def _regional_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGION_LABELS.values():
        subset = table[table["region"] == region]
        total = int(len(subset))
        valid = int(subset["cfd_valid"].sum())
        invalid = total - valid
        rows.append(
            {
                "region": region,
                "total_rows": total,
                "valid_rows": valid,
                "invalid_rows": invalid,
                "valid_fraction": valid / total if total else np.nan,
                "invalid_fraction": invalid / total if total else np.nan,
                "invalid_percentage": 100.0 * invalid / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame) -> None:
    total = int(summary["total_rows"].sum())
    invalid = int(summary["invalid_rows"].sum())
    valid = int(summary["valid_rows"].sum())
    print("Regional CFD validity summary")
    print(f"  total rows in named device regions: {total}")
    print(f"  valid CFD rows: {valid} ({valid / total:.2%})")
    print(f"  invalid CFD rows: {invalid} ({invalid / total:.2%})")
    print("")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.region}: invalid {row.invalid_rows}/{row.total_rows} "
            f"({row.invalid_percentage:.3f}%), valid {row.valid_rows}/{row.total_rows}"
        )


def _save_invalid_bar_chart(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(summary["region"], summary["invalid_percentage"], color="#dc2626", alpha=0.85)
    ax.set_ylabel("invalid CFD samples (%)")
    ax.set_xlabel("device region")
    ax.set_title("Invalid CFD sample percentage by device region")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_invalid_overview(table: pd.DataFrame, geometry: FullDeviceCFDGeometry, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _draw_geometry(ax, geometry)
    valid = table["cfd_valid"].to_numpy(bool)
    ax.scatter(table.loc[valid, "x_device_um"], table.loc[valid, "y_device_um"], s=2, color="#cbd5e1", alpha=0.18, rasterized=True, label="valid")
    ax.scatter(table.loc[~valid, "x_device_um"], table.loc[~valid, "y_device_um"], s=7, color="#dc2626", alpha=0.75, rasterized=True, label="invalid")
    ax.set_title("Invalid CFD samples over full-device geometry")
    _format_device_axes(ax)
    ax.legend(loc="best")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_invalid_zoom(table: pd.DataFrame, geometry: FullDeviceCFDGeometry, region: str, path: Path) -> None:
    subset = table[table["region"] == region]
    invalid = subset[~subset["cfd_valid"]]
    if invalid.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    _draw_geometry(ax, geometry)
    valid = subset["cfd_valid"].to_numpy(bool)
    ax.scatter(subset.loc[valid, "x_device_um"], subset.loc[valid, "y_device_um"], s=4, color="#cbd5e1", alpha=0.25, rasterized=True, label="valid")
    ax.scatter(invalid["x_device_um"], invalid["y_device_um"], s=12, color="#dc2626", alpha=0.8, rasterized=True, label="invalid")
    xmin, xmax = invalid["x_device_um"].min(), invalid["x_device_um"].max()
    ymin, ymax = invalid["y_device_um"].min(), invalid["y_device_um"].max()
    pad = max(80.0, 0.2 * max(xmax - xmin, ymax - ymin, 1.0))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_title(f"Invalid CFD samples: {region}")
    _format_device_axes(ax)
    ax.legend(loc="best")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _print_random_invalid_samples(table: pd.DataFrame, geometry: FullDeviceCFDGeometry, *, seed: int, count: int) -> None:
    invalid = table[~table["cfd_valid"]].copy()
    if invalid.empty:
        print("\nNo invalid CFD samples were found.")
        return
    sample = invalid.sample(n=min(count, len(invalid)), random_state=seed)
    boundary = _boundary_lines(geometry)
    print("\nRandom invalid CFD samples")
    print("  frame, track_id, x_um, y_um, region, distance_to_boundary_um")
    for row in sample.itertuples(index=False):
        point = Point(float(row.x_device_um), float(row.y_device_um))
        distance = boundary.distance(point)
        print(
            f"  {int(row.frame)}, {int(row.track_id)}, "
            f"{float(row.x_device_um):.3f}, {float(row.y_device_um):.3f}, "
            f"{row.region}, {distance:.3f}"
        )


def _boundary_lines(geometry: FullDeviceCFDGeometry) -> MultiLineString:
    return MultiLineString([LineString(geometry.outer_ring_um), LineString(geometry.inner_ring_um)])


def _draw_geometry(ax: plt.Axes, geometry: FullDeviceCFDGeometry) -> None:
    outer = geometry.outer_ring_um
    inner = geometry.inner_ring_um
    ax.plot(outer[:, 0], outer[:, 1], color="#111827", linewidth=1.0, label="outer wall")
    ax.plot(inner[:, 0], inner[:, 1], color="#111827", linewidth=1.0)
    for name, line in geometry.centerlines.items():
        ax.plot(line.points_um[:, 0], line.points_um[:, 1], linewidth=0.8, alpha=0.6, label=f"{name} centerline")


def _format_device_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.grid(alpha=0.2)


def _token(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    main()
