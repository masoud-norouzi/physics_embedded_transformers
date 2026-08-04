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

from src.physics.hydraulics import compute_frame_baseline_hydraulics_from_occupancies
from src.physics.runtime import load_physics_runtime_context


DEFAULT_EXPERIMENT = Path("configs/experiments/video_2.yml")
DEFAULT_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_OUTPUT = Path("outputs/evaluation/single_tracer_cfd_streamline")
REGION_NAMES = {
    0: "outside",
    1: "inlet_channel",
    2: "outlet_channel",
    3: "left_branch",
    4: "right_branch",
    5: "inlet_junction",
    6: "outlet_junction",
}
OCCUPANCY_NAMES = (
    "occupancy_inlet_channel",
    "occupancy_inlet_junction",
    "occupancy_left_branch",
    "occupancy_right_branch",
    "occupancy_outlet_junction",
    "occupancy_outlet_channel",
)
REGION_TO_OCCUPANCY = {
    1: "occupancy_inlet_channel",
    2: "occupancy_outlet_channel",
    3: "occupancy_left_branch",
    4: "occupancy_right_branch",
    5: "occupancy_inlet_junction",
    6: "occupancy_outlet_junction",
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_physics_runtime_context(
        experiment_config_path=args.experiment_config,
        cfd_library_path=args.cfd_library,
    )
    table = rollout_tracer(
        context=context,
        start_x=float(args.start_x),
        start_y=float(args.start_y),
        steps=int(args.steps),
    )
    table.to_csv(output_dir / "trajectory.csv", index=False)
    summary = summarize(table)
    save_trajectory_plot(table, context.region_labels, output_dir / "trajectory.png")
    animation_path = save_animation(table, context.region_labels, output_dir / "tracer_streamline_animation.mp4")
    save_json(
        output_dir / "summary.json",
        {
            "experiment": str(args.experiment_config),
            "cfd_library": str(args.cfd_library),
            "start_x": float(args.start_x),
            "start_y": float(args.start_y),
            "steps": int(args.steps),
            "animation": str(animation_path),
            "summary": summary,
        },
    )
    print("Single-tracer CFD streamline complete")
    print(f"  output: {output_dir}")
    print(f"  animation: {animation_path}")
    print(
        f"  final=({summary['final_x']:.2f}, {summary['final_y']:.2f}) "
        f"region={summary['final_region']} leaves_channel={summary['trajectory_ever_leaves_channel']} "
        f"distance={summary['total_traveled_distance_px']:.2f}px "
        f"speed={summary['mean_speed_mm_s']:.2f}..{summary['max_speed_mm_s']:.2f} mm/s"
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone one-particle CFD tracer using runtime hydraulics.")
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-x", type=float, default=331.0)
    parser.add_argument("--start-y", type=float, default=50.0)
    parser.add_argument("--steps", type=int, default=200)
    return parser.parse_args(argv)


def rollout_tracer(*, context, start_x: float, start_y: float, steps: int) -> pd.DataFrame:
    x = float(start_x)
    y = float(start_y)
    rows: list[dict[str, Any]] = []
    for step in range(int(steps) + 1):
        region_id = region_id_at(x, y, context.region_labels)
        occupancy = one_hot_occupancy(region_id)
        hydraulics = hydraulics_from_one_hot(occupancy, context, step)
        sample = sample_cfd_at_image_point(x, y, float(hydraulics["left_flow_fraction"]), context)
        vx_mm_s = float(sample["u_x_m_per_s"] * 1000.0)
        vy_mm_s = float(-sample["u_y_m_per_s"] * 1000.0)
        speed_mm_s = float(sample["speed_m_per_s"] * 1000.0)
        rows.append(
            {
                "step": int(step),
                "x": x,
                "y": y,
                "region_id": int(region_id),
                "region": REGION_NAMES.get(int(region_id), "unknown"),
                "inside_channel": bool(region_id > 0),
                "vx_mm_s": vx_mm_s,
                "vy_mm_s": vy_mm_s,
                "speed_mm_s": speed_mm_s,
                "cfd_u_norm": float(sample["cfd_u_norm"]),
                "cfd_v_norm": float(sample["cfd_v_norm"]),
                "cfd_original_valid": bool(sample["original_valid"]),
                "cfd_projection_distance_um": float(sample["projection_distance_um"]),
                "left_flow_fraction": float(hydraulics["left_flow_fraction"]),
                "left_flow_ul_hr": float(hydraulics["left_flow_ul_hr"]),
                "right_flow_ul_hr": float(hydraulics["right_flow_ul_hr"]),
                "n_left_eff": float(hydraulics["n_left_eff"]),
                "n_right_eff": float(hydraulics["n_right_eff"]),
                **occupancy,
            }
        )
        if step == int(steps):
            break
        x += vx_mm_s / float(context.velocity_mm_s_per_px_frame)
        y += vy_mm_s / float(context.velocity_mm_s_per_px_frame)
    return pd.DataFrame(rows)


def hydraulics_from_one_hot(occupancy: dict[str, float], context, frame: int) -> dict[str, Any]:
    constants = context.hydraulic_constants
    result = compute_frame_baseline_hydraulics_from_occupancies(
        float(occupancy["occupancy_left_branch"]),
        float(occupancy["occupancy_right_branch"]),
        frame=int(frame),
        n_droplets_total=1,
        left_length_um=constants["left_length_um"],
        right_length_um=constants["right_length_um"],
        droplet_equivalent_length_um=constants["droplet_equivalent_length_um"],
        total_mixture_flow_ul_hr=constants["total_mixture_flow_ul_hr"],
        channel_width_um=constants["channel_width_um"],
        channel_height_um=constants["channel_height_um"],
        continuous_flow_ul_hr=constants.get("continuous_flow_ul_hr"),
        dispersed_flow_ul_hr=constants.get("dispersed_flow_ul_hr"),
    )
    total_flow = float(result["left_flow_ul_hr"] + result["right_flow_ul_hr"])
    result["left_flow_fraction"] = float(result["left_flow_ul_hr"] / total_flow)
    return result


def sample_cfd_at_image_point(x: float, y: float, left_flow_fraction: float, context) -> dict[str, float | bool]:
    split_min, split_max = context.cfd_split_bounds
    split = float(np.clip(left_flow_fraction, split_min, split_max))
    field = context.cfd_library.interpolate(split)
    points_px = np.asarray([[float(x), float(y)]], dtype=float)
    geometry = context.cfd_library.cases[0].mesh.geometry
    if getattr(geometry, "coordinate_frame", "") == "device_cartesian_y_up":
        points_um = context.coordinate_convention.image_points_to_device(points_px)
    else:
        points_um = context.coordinate_convention.image_points_to_cfd(points_px)
    samples = field.sample_cfd(points_um)
    if not np.isfinite([samples.u_x_m_per_s[0], samples.u_y_m_per_s[0], samples.speed_m_per_s[0]]).all():
        raise ValueError(f"CFD sampling returned non-finite velocity at x={x}, y={y}")
    return {
        "u_x_m_per_s": float(samples.u_x_m_per_s[0]),
        "u_y_m_per_s": float(samples.u_y_m_per_s[0]),
        "speed_m_per_s": float(samples.speed_m_per_s[0]),
        "cfd_u_norm": float(samples.cfd_u_norm[0]),
        "cfd_v_norm": float(samples.cfd_v_norm[0]),
        "original_valid": bool(samples.original_valid[0]),
        "projection_distance_um": float(samples.projection_distance_um[0]),
    }


def one_hot_occupancy(region_id: int) -> dict[str, float]:
    occupancy = {name: 0.0 for name in OCCUPANCY_NAMES}
    name = REGION_TO_OCCUPANCY.get(int(region_id))
    if name is not None:
        occupancy[name] = 1.0
    return occupancy


def region_id_at(x: float, y: float, region_labels: np.ndarray) -> int:
    col = int(round(float(x)))
    row = int(round(float(y)))
    if row < 0 or row >= region_labels.shape[0] or col < 0 or col >= region_labels.shape[1]:
        return 0
    return int(region_labels[row, col])


def summarize(table: pd.DataFrame) -> dict[str, Any]:
    xy = table[["x", "y"]].to_numpy(float)
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return {
        "final_x": float(table["x"].iloc[-1]),
        "final_y": float(table["y"].iloc[-1]),
        "final_region": str(table["region"].iloc[-1]),
        "total_traveled_distance_px": float(distance.sum()),
        "mean_speed_mm_s": float(table["speed_mm_s"].mean()),
        "max_speed_mm_s": float(table["speed_mm_s"].max()),
        "min_speed_mm_s": float(table["speed_mm_s"].min()),
        "min_left_flow_fraction": float(table["left_flow_fraction"].min()),
        "max_left_flow_fraction": float(table["left_flow_fraction"].max()),
        "trajectory_ever_leaves_channel": bool((~table["inside_channel"].astype(bool)).any()),
        "first_outside_step": (
            None
            if bool(table["inside_channel"].all())
            else int(table.index[~table["inside_channel"].astype(bool)][0])
        ),
        "regions_visited": sorted(str(region) for region in table["region"].unique()),
        "cfd_projection_steps": int((table["cfd_projection_distance_um"] > 0.0).sum()),
    }


def save_trajectory_plot(table: pd.DataFrame, region_labels: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(region_labels > 0, cmap="gray", alpha=0.25, origin="upper")
    ax.plot(table["x"], table["y"], color="#2563eb", linewidth=2.0)
    ax.scatter(table["x"].iloc[0], table["y"].iloc[0], color="#16a34a", s=50, label="start")
    ax.scatter(table["x"].iloc[-1], table["y"].iloc[-1], color="#dc2626", s=50, label="final")
    ax.set_xlim(250, 540)
    ax.set_ylim(region_labels.shape[0], 0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("CFD tracer trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_animation(table: pd.DataFrame, region_labels: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    channel_mask = region_labels > 0
    frames: list[np.ndarray] = []

    for frame_idx in range(len(table)):
        ax.clear()
        current = table.iloc[frame_idx]
        trail = table.iloc[: frame_idx + 1]
        ax.imshow(channel_mask, cmap="gray", alpha=0.24, origin="upper")
        ax.plot(trail["x"], trail["y"], color="#2563eb", linewidth=2.0)
        ax.scatter([current["x"]], [current["y"]], s=70, color="#dc2626", zorder=5)
        arrow_scale = 0.10
        ax.arrow(
            current["x"],
            current["y"],
            current["vx_mm_s"] * arrow_scale,
            current["vy_mm_s"] * arrow_scale,
            width=1.1,
            head_width=7,
            head_length=9,
            color="#f97316",
            length_includes_head=True,
            zorder=6,
        )
        ax.text(
            0.02,
            0.98,
            (
                f"step={int(current['step'])}\n"
                f"region={current['region']}\n"
                f"speed={current['speed_mm_s']:.1f} mm/s\n"
                f"left flow={current['left_flow_fraction']:.3f}\n"
                f"proj={current['cfd_projection_distance_um']:.1f} um"
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
        ax.set_title("Volume-free tracer advected by runtime CFD")
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(path, frames, fps=12)
    plt.close(fig)
    return path


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("No frames were rendered for the animation.")
    try:
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    except ModuleNotFoundError:
        pass
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
