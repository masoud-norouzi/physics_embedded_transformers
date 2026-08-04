from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.velocity import load_velocity_conversion_from_experiment


DEFAULT_TRACKS = Path("outputs/physics/video_2/enrichment/physics_enriched_tracked_features.csv")
DEFAULT_REGION_LABELS = Path("data/geometry/asymmetric_loop_h100/region_labels.npy")
DEFAULT_EXPERIMENT = Path("configs/experiments/video_2.yml")
DEFAULT_OUTPUT = Path("outputs/evaluation/velocity_gap_study")
REGION_NAMES = {
    0: "outside",
    1: "inlet_channel",
    2: "outlet_channel",
    3: "left_branch",
    4: "right_branch",
    5: "inlet_junction",
    6: "outlet_junction",
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gaps = tuple(int(item) for item in str(args.gaps).split(",") if item.strip())
    if not gaps or any(gap < 1 for gap in gaps):
        raise ValueError(f"gaps must be positive integers, got {args.gaps!r}")

    scale = load_velocity_conversion_from_experiment(args.experiment_config)["velocity_mm_s_per_px_frame"]
    tracks = load_tracks(args.tracks_csv)
    region_labels = np.load(args.region_labels)
    samples = build_velocity_samples(tracks, region_labels, gaps, float(scale), int(args.max_rows_per_gap))
    if samples.empty:
        raise RuntimeError("No valid velocity samples were built.")

    summary = summarize(samples, gaps)
    by_region = summarize_by_region(samples, gaps)
    comparison = compare_to_reference(samples, int(args.reference_gap))
    samples.to_csv(output_dir / "velocity_gap_samples.csv", index=False)
    summary.to_csv(output_dir / "velocity_gap_summary.csv", index=False)
    by_region.to_csv(output_dir / "velocity_gap_by_region.csv", index=False)
    comparison.to_csv(output_dir / "velocity_gap_reference_comparison.csv", index=False)
    save_plots(samples, summary, by_region, comparison, output_dir)
    payload = {
        "tracks_csv": str(args.tracks_csv),
        "region_labels": str(args.region_labels),
        "experiment_config": str(args.experiment_config),
        "velocity_mm_s_per_px_frame": float(scale),
        "gaps": list(gaps),
        "reference_gap": int(args.reference_gap),
        "rows": int(len(samples)),
        "outputs": {
            "samples": str(output_dir / "velocity_gap_samples.csv"),
            "summary": str(output_dir / "velocity_gap_summary.csv"),
            "by_region": str(output_dir / "velocity_gap_by_region.csv"),
            "comparison": str(output_dir / "velocity_gap_reference_comparison.csv"),
        },
        "headline": headline_metrics(summary, comparison),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Velocity gap study complete")
    print(f"  output: {output_dir}")
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare observed velocity target quality across temporal gaps.")
    parser.add_argument("--tracks-csv", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--region-labels", type=Path, default=DEFAULT_REGION_LABELS)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gaps", type=str, default="1,2,3,5,10")
    parser.add_argument("--reference-gap", type=int, default=10)
    parser.add_argument("--max-rows-per-gap", type=int, default=200000)
    return parser.parse_args(argv)


def load_tracks(path: Path) -> pd.DataFrame:
    columns = [
        "frame",
        "track_id",
        "centroid_x",
        "centroid_y",
        "background_direction_x",
        "background_direction_y",
        "cfd_u_norm",
        "cfd_v_norm",
        "cfd_valid",
        "dominant_region",
    ]
    tracks = pd.read_csv(path, usecols=lambda name: name in columns)
    tracks = tracks.rename(columns={"centroid_x": "x", "centroid_y": "y"})
    tracks = tracks.dropna(subset=["frame", "track_id", "x", "y"])
    tracks["frame"] = tracks["frame"].astype(int)
    tracks["track_id"] = tracks["track_id"].astype(int)
    return tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)


def build_velocity_samples(
    tracks: pd.DataFrame,
    region_labels: np.ndarray,
    gaps: tuple[int, ...],
    velocity_mm_s_per_px_frame: float,
    max_rows_per_gap: int,
) -> pd.DataFrame:
    pieces = []
    for gap in gaps:
        current = tracks.copy()
        future = tracks[["track_id", "frame", "x", "y"]].copy()
        future["frame"] -= int(gap)
        future = future.rename(columns={"x": "x_future", "y": "y_future"})
        merged = current.merge(future, on=["track_id", "frame"], how="inner", validate="one_to_one")
        merged = merged[np.isfinite(merged[["x", "y", "x_future", "y_future"]]).all(axis=1)].copy()
        if max_rows_per_gap > 0 and len(merged) > max_rows_per_gap:
            merged = merged.sample(n=max_rows_per_gap, random_state=gap).sort_values(["track_id", "frame"])
        dx = merged["x_future"].to_numpy(float) - merged["x"].to_numpy(float)
        dy = merged["y_future"].to_numpy(float) - merged["y"].to_numpy(float)
        merged["gap"] = int(gap)
        merged["dx_px"] = dx
        merged["dy_px"] = dy
        merged["displacement_px"] = np.hypot(dx, dy)
        merged["vx_mm_s"] = dx * float(velocity_mm_s_per_px_frame) / float(gap)
        merged["vy_mm_s"] = dy * float(velocity_mm_s_per_px_frame) / float(gap)
        merged["speed_mm_s"] = np.hypot(merged["vx_mm_s"], merged["vy_mm_s"])
        merged["angle_rad"] = np.arctan2(merged["vy_mm_s"], merged["vx_mm_s"])
        merged["region_id"] = region_ids(merged["x"].to_numpy(float), merged["y"].to_numpy(float), region_labels)
        merged["region"] = [REGION_NAMES.get(int(item), "unknown") for item in merged["region_id"]]
        direction_x, direction_y = reference_direction(merged)
        dot = merged["vx_mm_s"].to_numpy(float) * direction_x + merged["vy_mm_s"].to_numpy(float) * direction_y
        direction_norm = np.hypot(direction_x, direction_y)
        merged["has_reference_direction"] = np.isfinite(direction_x) & np.isfinite(direction_y) & (direction_norm > 1.0e-8)
        merged["wrong_way"] = merged["has_reference_direction"] & (dot < 0.0)
        merged["reference_angle_error_deg"] = angular_error_deg(
            merged["vx_mm_s"].to_numpy(float),
            merged["vy_mm_s"].to_numpy(float),
            direction_x,
            direction_y,
        )
        pieces.append(
            merged[
                [
                    "track_id",
                    "frame",
                    "gap",
                    "x",
                    "y",
                    "x_future",
                    "y_future",
                    "dx_px",
                    "dy_px",
                    "displacement_px",
                    "vx_mm_s",
                    "vy_mm_s",
                    "speed_mm_s",
                    "angle_rad",
                    "region_id",
                    "region",
                    "has_reference_direction",
                    "wrong_way",
                    "reference_angle_error_deg",
                ]
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def region_ids(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cols = np.rint(x).astype(int)
    rows = np.rint(y).astype(int)
    valid = (rows >= 0) & (rows < labels.shape[0]) & (cols >= 0) & (cols < labels.shape[1])
    out = np.zeros(len(x), dtype=int)
    out[valid] = labels[rows[valid], cols[valid]].astype(int)
    return out


def reference_direction(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if {"background_direction_x", "background_direction_y"}.issubset(table.columns):
        dx = table["background_direction_x"].to_numpy(float)
        dy = -table["background_direction_y"].to_numpy(float)
        finite = np.isfinite(dx) & np.isfinite(dy) & (np.hypot(dx, dy) > 1.0e-8)
        if finite.any():
            return dx, dy
    if {"cfd_u_norm", "cfd_v_norm"}.issubset(table.columns):
        return table["cfd_u_norm"].to_numpy(float), -table["cfd_v_norm"].to_numpy(float)
    return np.full(len(table), np.nan), np.full(len(table), np.nan)


def angular_error_deg(vx: np.ndarray, vy: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
    speed = np.hypot(vx, vy)
    ref_norm = np.hypot(ref_x, ref_y)
    valid = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(ref_x) & np.isfinite(ref_y) & (speed > 1.0e-8) & (ref_norm > 1.0e-8)
    out = np.full(len(vx), np.nan, dtype=float)
    dot = vx[valid] * ref_x[valid] + vy[valid] * ref_y[valid]
    cosine = np.clip(dot / (speed[valid] * ref_norm[valid]), -1.0, 1.0)
    out[valid] = np.degrees(np.arccos(cosine))
    return out


def summarize(samples: pd.DataFrame, gaps: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for gap in gaps:
        part = samples[samples["gap"] == int(gap)]
        rows.append(summary_row(part, {"gap": int(gap)}))
    return pd.DataFrame(rows)


def summarize_by_region(samples: pd.DataFrame, gaps: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for (gap, region), part in samples.groupby(["gap", "region"], sort=True):
        row = summary_row(part, {"gap": int(gap), "region": str(region)})
        rows.append(row)
    return pd.DataFrame(rows)


def summary_row(part: pd.DataFrame, prefix: dict[str, Any]) -> dict[str, Any]:
    speed = part["speed_mm_s"].to_numpy(float)
    displacement = part["displacement_px"].to_numpy(float)
    angle = part["reference_angle_error_deg"].to_numpy(float)
    return {
        **prefix,
        "count": int(len(part)),
        "median_displacement_px": finite_percentile(displacement, 50),
        "p10_displacement_px": finite_percentile(displacement, 10),
        "p90_displacement_px": finite_percentile(displacement, 90),
        "mean_speed_mm_s": finite_mean(speed),
        "median_speed_mm_s": finite_percentile(speed, 50),
        "speed_cv": finite_std(speed) / max(finite_mean(speed), 1.0e-12),
        "p95_abs_accel_like_delta_speed_mm_s": acceleration_like_tail(part),
        "wrong_way_fraction": finite_mean(part.loc[part["has_reference_direction"], "wrong_way"].astype(float).to_numpy()),
        "median_reference_angle_error_deg": finite_percentile(angle, 50),
        "p90_reference_angle_error_deg": finite_percentile(angle, 90),
    }


def acceleration_like_tail(part: pd.DataFrame) -> float:
    values = []
    for _, group in part.sort_values(["track_id", "frame"]).groupby("track_id", sort=False):
        speed = group["speed_mm_s"].to_numpy(float)
        if len(speed) > 1:
            values.append(np.abs(np.diff(speed)))
    if not values:
        return float("nan")
    return finite_percentile(np.concatenate(values), 95)


def compare_to_reference(samples: pd.DataFrame, reference_gap: int) -> pd.DataFrame:
    ref = samples[samples["gap"] == int(reference_gap)][["track_id", "frame", "vx_mm_s", "vy_mm_s"]].rename(
        columns={"vx_mm_s": "ref_vx_mm_s", "vy_mm_s": "ref_vy_mm_s"}
    )
    rows = []
    for gap, part in samples.groupby("gap", sort=True):
        if int(gap) == int(reference_gap):
            continue
        merged = part.merge(ref, on=["track_id", "frame"], how="inner")
        error = angular_error_deg(
            merged["vx_mm_s"].to_numpy(float),
            merged["vy_mm_s"].to_numpy(float),
            merged["ref_vx_mm_s"].to_numpy(float),
            merged["ref_vy_mm_s"].to_numpy(float),
        )
        rows.append(
            {
                "gap": int(gap),
                "reference_gap": int(reference_gap),
                "count": int(len(merged)),
                "median_angle_error_to_ref_deg": finite_percentile(error, 50),
                "p90_angle_error_to_ref_deg": finite_percentile(error, 90),
                "wrong_way_vs_ref_fraction": finite_mean((error > 90.0).astype(float)),
            }
        )
    return pd.DataFrame(rows)


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_std(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def finite_percentile(values, percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else float("nan")


def headline_metrics(summary: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    return {
        "summary": summary.to_dict("records"),
        "reference_comparison": comparison.to_dict("records"),
    }


def save_plots(samples: pd.DataFrame, summary: pd.DataFrame, by_region: pd.DataFrame, comparison: pd.DataFrame, output_dir: Path) -> None:
    plot_summary_lines(summary, output_dir / "velocity_gap_summary.png")
    plot_region_wrong_way(by_region, output_dir / "wrong_way_by_region.png")
    plot_reference_comparison(comparison, output_dir / "angle_error_to_gap_reference.png")
    plot_speed_distributions(samples, output_dir / "speed_distribution_by_gap.png")
    plot_displacement_distributions(samples, output_dir / "displacement_distribution_by_gap.png")


def plot_summary_lines(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    x = summary["gap"]
    axes[0].plot(x, summary["median_displacement_px"], marker="o")
    axes[0].set_ylabel("median displacement [px]")
    axes[1].plot(x, summary["speed_cv"], marker="o")
    axes[1].set_ylabel("speed CV")
    axes[2].plot(x, summary["wrong_way_fraction"], marker="o")
    axes[2].set_ylabel("wrong-way fraction")
    axes[3].plot(x, summary["median_reference_angle_error_deg"], marker="o", label="median")
    axes[3].plot(x, summary["p90_reference_angle_error_deg"], marker="o", label="p90")
    axes[3].set_ylabel("angle error to CFD/background [deg]")
    axes[3].legend()
    for ax in axes:
        ax.set_xlabel("velocity gap [frames]")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_region_wrong_way(by_region: pd.DataFrame, path: Path) -> None:
    pivot = by_region.pivot(index="gap", columns="region", values="wrong_way_fraction")
    ax = pivot.plot(marker="o", figsize=(11, 6))
    ax.set_ylabel("wrong-way fraction")
    ax.set_xlabel("velocity gap [frames]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)


def plot_reference_comparison(comparison: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(comparison["gap"], comparison["median_angle_error_to_ref_deg"], marker="o", label="median")
    ax.plot(comparison["gap"], comparison["p90_angle_error_to_ref_deg"], marker="o", label="p90")
    ax.set_xlabel("velocity gap [frames]")
    ax.set_ylabel("angle error to reference gap [deg]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_speed_distributions(samples: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [samples.loc[samples["gap"] == gap, "speed_mm_s"].to_numpy(float) for gap in sorted(samples["gap"].unique())]
    ax.boxplot(data, tick_labels=[str(gap) for gap in sorted(samples["gap"].unique())], showfliers=False)
    ax.set_xlabel("velocity gap [frames]")
    ax.set_ylabel("speed [mm/s]")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_displacement_distributions(samples: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [samples.loc[samples["gap"] == gap, "displacement_px"].to_numpy(float) for gap in sorted(samples["gap"].unique())]
    ax.boxplot(data, tick_labels=[str(gap) for gap in sorted(samples["gap"].unique())], showfliers=False)
    ax.set_xlabel("velocity gap [frames]")
    ax.set_ylabel("displacement [px]")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
