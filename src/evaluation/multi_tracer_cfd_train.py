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

from src.evaluation.single_tracer_cfd_streamline import (
    REGION_NAMES,
    hydraulics_from_one_hot,
    one_hot_occupancy,
    region_id_at,
    save_json,
    write_mp4,
)
from src.physics.runtime import load_physics_runtime_context


DEFAULT_TRACKS = Path("outputs/physics/video_2/enrichment/physics_enriched_tracked_features.csv")
DEFAULT_EXPERIMENT = Path("configs/experiments/video_2.yml")
DEFAULT_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_OUTPUT = Path("outputs/evaluation/multi_tracer_cfd_train")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_physics_runtime_context(
        experiment_config_path=args.experiment_config,
        cfd_library_path=args.cfd_library,
    )
    tracks = load_tracking_table(args.tracks_csv)
    initial = initial_tracers_from_frame(tracks, context.region_labels, int(args.seed_frame))
    arrivals = inlet_arrival_frames(tracks)
    schedule = injection_schedule_from_arrivals(
        arrivals,
        seed_frame=int(args.seed_frame),
        existing_count=len(initial),
        target_count=int(args.tracer_count),
    )
    table = rollout_tracer_train(
        context=context,
        initial=initial,
        schedule=schedule,
        steps=int(args.steps),
        start_x=float(args.start_x),
        start_y=float(args.start_y),
        target_count=int(args.tracer_count),
    )
    table.to_csv(output_dir / "trajectory.csv", index=False)
    summary = summarize(table, schedule, initial)
    save_train_plot(table, context.region_labels, output_dir / "trajectory.png")
    animation_path = save_train_animation(table, context.region_labels, output_dir / "tracer_train_animation.mp4")
    save_json(
        output_dir / "summary.json",
        {
            "tracks_csv": str(args.tracks_csv),
            "experiment": str(args.experiment_config),
            "cfd_library": str(args.cfd_library),
            "seed_frame": int(args.seed_frame),
            "start_x": float(args.start_x),
            "start_y": float(args.start_y),
            "target_tracer_count": int(args.tracer_count),
            "steps": int(args.steps),
            "initial_tracer_count": len(initial),
            "injection_schedule": schedule,
            "animation": str(animation_path),
            "summary": summary,
        },
    )
    print("Multi-tracer CFD train complete")
    print(f"  output: {output_dir}")
    print(f"  animation: {animation_path}")
    print(
        f"  initial={len(initial)} injected={len(schedule)} max_active={summary['max_active_tracers']} "
        f"final_active={summary['final_active_tracers']} first_outside_step={summary['first_outside_step']}"
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone CFD tracer train using observed inlet spacing.")
    parser.add_argument("--tracks-csv", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-frame", type=int, default=27410)
    parser.add_argument("--start-x", type=float, default=331.0)
    parser.add_argument("--start-y", type=float, default=50.0)
    parser.add_argument("--tracer-count", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1200)
    return parser.parse_args(argv)


def load_tracking_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=["frame", "track_id", "centroid_x", "centroid_y"])
    table = table.rename(columns={"centroid_x": "x", "centroid_y": "y"})
    table = table.dropna(subset=["frame", "track_id", "x", "y"])
    table["frame"] = table["frame"].astype(int)
    table["track_id"] = table["track_id"].astype(int)
    return table


def initial_tracers_from_frame(tracks: pd.DataFrame, region_labels: np.ndarray, seed_frame: int) -> list[dict[str, Any]]:
    active = tracks.loc[tracks["frame"] == int(seed_frame)]
    if active.empty:
        raise KeyError(f"Frame {seed_frame} not found in tracking CSV")
    tracers = []
    for row in active.itertuples(index=False):
        x = float(row.x)
        y = float(row.y)
        region_id = region_id_at(x, y, region_labels)
        if region_id <= 0:
            continue
        tracers.append(
            {
                "tracer_id": int(row.track_id),
                "x": x,
                "y": y,
                "active": True,
                "injection_step": 0,
                "source": "observed_seed_frame",
            }
        )
    return tracers


def inlet_arrival_frames(tracks: pd.DataFrame) -> list[int]:
    arrivals: list[int] = []
    inlet = tracks[
        tracks["x"].between(315.0, 345.0, inclusive="neither")
        & tracks["y"].between(0.0, 120.0, inclusive="neither")
    ]
    for _, group in inlet.sort_values("frame").groupby("track_id", sort=False):
        arrivals.append(int(group["frame"].iloc[0]))
    return sorted(arrivals)


def injection_schedule_from_arrivals(
    arrivals: list[int],
    *,
    seed_frame: int,
    existing_count: int,
    target_count: int,
) -> list[dict[str, Any]]:
    future = [int(frame) for frame in arrivals if int(frame) > int(seed_frame)]
    needed = max(int(target_count) - int(existing_count), 0)
    schedule = []
    for new_index, frame in enumerate(future[:needed], start=1):
        schedule.append(
            {
                "tracer_id": int(1_000_000 + new_index),
                "frame": int(frame),
                "injection_step": int(frame - int(seed_frame)),
            }
        )
    if len(schedule) < needed:
        raise RuntimeError(f"Only found {len(schedule)} future arrivals after seed frame; need {needed}")
    return schedule


def rollout_tracer_train(
    *,
    context,
    initial: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
    steps: int,
    start_x: float,
    start_y: float,
    target_count: int,
) -> pd.DataFrame:
    tracers = [dict(item) for item in initial[: int(target_count)]]
    schedule_by_step: dict[int, list[dict[str, Any]]] = {}
    for item in schedule:
        schedule_by_step.setdefault(int(item["injection_step"]), []).append(item)
    rows: list[dict[str, Any]] = []
    for step in range(int(steps) + 1):
        for item in schedule_by_step.get(step, []):
            if len(tracers) >= int(target_count):
                break
            tracers.append(
                {
                    "tracer_id": int(item["tracer_id"]),
                    "x": float(start_x),
                    "y": float(start_y),
                    "active": True,
                    "injection_step": int(step),
                    "source": "observed_inlet_spacing",
                }
            )
        active_indices = [idx for idx, tracer in enumerate(tracers) if tracer["active"]]
        occupancies = [one_hot_occupancy(region_id_at(tracers[idx]["x"], tracers[idx]["y"], context.region_labels)) for idx in active_indices]
        total_occupancy = sum_occupancies(occupancies)
        hydraulics = hydraulics_from_total_one_hot(total_occupancy, context, step, len(active_indices))
        active_points = np.asarray(
            [[float(tracers[idx]["x"]), float(tracers[idx]["y"])] for idx in active_indices],
            dtype=float,
        )
        cfd_samples = sample_cfd_at_image_points(active_points, float(hydraulics["left_flow_fraction"]), context)
        next_updates = []
        for sample_offset, idx in enumerate(active_indices):
            tracer = tracers[idx]
            x = float(tracer["x"])
            y = float(tracer["y"])
            region_id = region_id_at(x, y, context.region_labels)
            occupancy = one_hot_occupancy(region_id)
            vx_mm_s = float(cfd_samples["u_x_m_per_s"][sample_offset] * 1000.0)
            vy_mm_s = float(-cfd_samples["u_y_m_per_s"][sample_offset] * 1000.0)
            rows.append(
                {
                    "step": int(step),
                    "tracer_id": int(tracer["tracer_id"]),
                    "source": str(tracer["source"]),
                    "injection_step": int(tracer["injection_step"]),
                    "x": x,
                    "y": y,
                    "region_id": int(region_id),
                    "region": REGION_NAMES.get(int(region_id), "unknown"),
                    "inside_channel": bool(region_id > 0),
                    "vx_mm_s": vx_mm_s,
                    "vy_mm_s": vy_mm_s,
                    "speed_mm_s": float(cfd_samples["speed_m_per_s"][sample_offset] * 1000.0),
                    "left_flow_fraction": float(hydraulics["left_flow_fraction"]),
                    "n_left_eff": float(hydraulics["n_left_eff"]),
                    "n_right_eff": float(hydraulics["n_right_eff"]),
                    "active_tracer_count": int(len(active_indices)),
                    "cfd_original_valid": bool(cfd_samples["original_valid"][sample_offset]),
                    "cfd_projection_distance_um": float(cfd_samples["projection_distance_um"][sample_offset]),
                    **occupancy,
                }
            )
            next_updates.append((idx, x + vx_mm_s / context.velocity_mm_s_per_px_frame, y + vy_mm_s / context.velocity_mm_s_per_px_frame))
        if step == int(steps):
            break
        for idx, x_next, y_next in next_updates:
            tracers[idx]["x"] = float(x_next)
            tracers[idx]["y"] = float(y_next)
    return pd.DataFrame(rows)


def sample_cfd_at_image_points(points_px: np.ndarray, left_flow_fraction: float, context) -> dict[str, np.ndarray]:
    split_min, split_max = context.cfd_split_bounds
    split = float(np.clip(left_flow_fraction, split_min, split_max))
    field = context.cfd_library.interpolate(split)
    geometry = context.cfd_library.cases[0].mesh.geometry
    if getattr(geometry, "coordinate_frame", "") == "device_cartesian_y_up":
        points_um = context.coordinate_convention.image_points_to_device(points_px)
    else:
        points_um = context.coordinate_convention.image_points_to_cfd(points_px)
    samples = field.sample_cfd(points_um)
    values = {
        "u_x_m_per_s": np.asarray(samples.u_x_m_per_s, dtype=float),
        "u_y_m_per_s": np.asarray(samples.u_y_m_per_s, dtype=float),
        "speed_m_per_s": np.asarray(samples.speed_m_per_s, dtype=float),
        "original_valid": np.asarray(samples.original_valid, dtype=bool),
        "projection_distance_um": np.asarray(samples.projection_distance_um, dtype=float),
    }
    finite = np.isfinite(values["u_x_m_per_s"]) & np.isfinite(values["u_y_m_per_s"]) & np.isfinite(values["speed_m_per_s"])
    if not bool(finite.all()):
        bad = points_px[np.flatnonzero(~finite)[:5]]
        raise ValueError(f"CFD sampling returned non-finite velocities for points {bad.tolist()}")
    return values


def sum_occupancies(occupancies: list[dict[str, float]]) -> dict[str, float]:
    total = {name: 0.0 for name in ("occupancy_inlet_channel", "occupancy_inlet_junction", "occupancy_left_branch", "occupancy_right_branch", "occupancy_outlet_junction", "occupancy_outlet_channel")}
    for occupancy in occupancies:
        for name in total:
            total[name] += float(occupancy.get(name, 0.0))
    return total


def hydraulics_from_total_one_hot(total_occupancy: dict[str, float], context, frame: int, n_active: int) -> dict[str, Any]:
    # Reuse the single-tracer helper with scaled branch occupancy, then patch total count.
    hydraulics = hydraulics_from_one_hot(
        {
            **{name: 0.0 for name in total_occupancy},
            "occupancy_left_branch": float(total_occupancy["occupancy_left_branch"]),
            "occupancy_right_branch": float(total_occupancy["occupancy_right_branch"]),
        },
        context,
        frame,
    )
    hydraulics["n_droplets_total"] = int(n_active)
    return hydraulics


def summarize(table: pd.DataFrame, schedule: list[dict[str, Any]], initial: list[dict[str, Any]]) -> dict[str, Any]:
    outside = ~table["inside_channel"].astype(bool)
    final = table.loc[table["step"] == table["step"].max()]
    return {
        "initial_tracer_count": int(len(initial)),
        "scheduled_injection_count": int(len(schedule)),
        "rows": int(len(table)),
        "max_active_tracers": int(table["active_tracer_count"].max()),
        "final_active_tracers": int(final["tracer_id"].nunique()),
        "first_outside_step": None if not bool(outside.any()) else int(table.loc[outside, "step"].min()),
        "outside_row_fraction": float(outside.mean()),
        "mean_speed_mm_s": float(table["speed_mm_s"].mean()),
        "max_speed_mm_s": float(table["speed_mm_s"].max()),
        "min_left_flow_fraction": float(table["left_flow_fraction"].min()),
        "max_left_flow_fraction": float(table["left_flow_fraction"].max()),
        "regions_visited": sorted(str(region) for region in table["region"].unique()),
        "cfd_projection_row_fraction": float((table["cfd_projection_distance_um"] > 0.0).mean()),
    }


def save_train_plot(table: pd.DataFrame, region_labels: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(region_labels > 0, cmap="gray", alpha=0.24, origin="upper")
    for _, group in table.groupby("tracer_id"):
        ax.plot(group["x"], group["y"], linewidth=0.8, alpha=0.35)
    ax.set_xlim(250, 540)
    ax.set_ylim(region_labels.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title("100-tracer CFD train")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_train_animation(table: pd.DataFrame, region_labels: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    channel_mask = region_labels > 0
    frames = []
    max_step = int(table["step"].max())
    frame_steps = np.unique(np.linspace(0, max_step, min(max_step + 1, 260), dtype=int))
    for step in frame_steps:
        ax.clear()
        current = table.loc[table["step"] == int(step)]
        recent = table[(table["step"] <= int(step)) & (table["step"] >= max(0, int(step) - 80))]
        ax.imshow(channel_mask, cmap="gray", alpha=0.22, origin="upper")
        for _, group in recent.groupby("tracer_id"):
            ax.plot(group["x"], group["y"], color="#2563eb", linewidth=0.7, alpha=0.20)
        ax.scatter(current["x"], current["y"], s=18, color="#dc2626", alpha=0.85)
        ax.text(
            0.02,
            0.98,
            (
                f"step={step}\n"
                f"active={current['tracer_id'].nunique()}\n"
                f"left flow={current['left_flow_fraction'].mean():.3f}\n"
                f"left eff={current['n_left_eff'].mean():.0f} right eff={current['n_right_eff'].mean():.0f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        ax.set_xlim(250, 540)
        ax.set_ylim(region_labels.shape[0], 0)
        ax.set_aspect("equal")
        ax.set_title("Observed-spacing train of volume-free CFD tracers")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(path, frames, fps=16)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
