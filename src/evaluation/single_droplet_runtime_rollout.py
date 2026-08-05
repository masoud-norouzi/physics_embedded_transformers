from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import yaml

from src.models.canonical_rollout_transformer import CanonicalRolloutTransformer
from src.physics.runtime import load_physics_runtime_context, step as physics_runtime_step
from src.physics.runtime.state_transition import CANONICAL_RUNTIME_FEATURE_NAMES


DEFAULT_CONFIG = Path("configs/experiments/physics_markovian_v1.yml")
DEFAULT_EXPERIMENT = Path("configs/experiments/video_2.yml")
DEFAULT_DATASET = Path("outputs/processed/2/canonical_dataset_v2/canonical_dataset_v2.npz")
DEFAULT_CHECKPOINT = Path("outputs/models/physics_markovian_v1-fused/best_checkpoint.pt")
DEFAULT_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_OUTPUT = Path("outputs/evaluation/single_droplet_runtime")
REGION_NAMES = {
    0: "outside",
    1: "inlet channel",
    2: "outlet channel",
    3: "left branch",
    4: "right branch",
    5: "inlet junction",
    6: "outlet junction",
}
OCCUPANCY_FEATURES = tuple(name for name in CANONICAL_RUNTIME_FEATURE_NAMES if name.startswith("occupancy_"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch = import_torch()
    device = select_device(torch, args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    dataset = load_dataset(args.dataset)
    feature_names = [str(name) for name in dataset["feature_names"]]
    validate_current_feature_contract(feature_names, config)
    feature_index = {name: index for index, name in enumerate(feature_names)}

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = load_model(torch, checkpoint, device)
    normalization = checkpoint["normalization_stats"]
    target_features = tuple(str(name) for name in checkpoint.get("target_features", config["model"]["target_features"]))
    prediction_mode = prediction_mode_from_targets(target_features)
    runtime_context = load_physics_runtime_context(
        experiment_config_path=args.experiment_config,
        cfd_library_path=args.cfd_library,
        feature_names=feature_names,
    )

    scenarios = select_scenarios(
        dataset=dataset,
        feature_index=feature_index,
        region_labels=runtime_context.region_labels,
        stride=int(config["dataset"]["stride"]),
        t_history=int(checkpoint["model_config"]["T_history"]),
        t_future=int(config["model"]["rollout_horizon"]),
        count=int(args.scenario_count),
        preferred_region=str(args.preferred_region),
        track_id=args.track_id,
        frame=args.frame,
    )
    if not scenarios:
        raise RuntimeError("No single-droplet scenarios were found.")

    summaries = []
    for scenario in scenarios:
        scenario_dir = output_dir / scenario["scenario"]
        scenario_dir.mkdir(parents=True, exist_ok=True)
        rows = rollout_single_droplet(
            torch=torch,
            model=model,
            device=device,
            initial_state=scenario["initial_state"],
            feature_index=feature_index,
            normalization=normalization,
            runtime_context=runtime_context,
            rollout_length=int(args.rollout_length),
            scenario=scenario,
            prediction_mode=prediction_mode,
        )
        table = pd.DataFrame(rows)
        table.to_csv(scenario_dir / "trajectory.csv", index=False)
        save_trajectory_plot(table, runtime_context.region_labels > 0, scenario_dir / "trajectory.png")
        save_ellipse_occupancy_animation(table, runtime_context.region_labels, scenario_dir / "ellipse_occupancy_animation.mp4")
        save_occupancy_fraction_plot(table, scenario_dir / "occupancy_fractions.png")
        save_speed_plot(table, scenario_dir / "speed.png")
        save_flow_plot(table, scenario_dir / "left_flow_fraction.png")
        summary = summarize(table)
        summaries.append(summary)
        save_json(
            scenario_dir / "metadata.json",
            {
                "checkpoint": checkpoint_metadata(args.checkpoint, checkpoint),
                "target_features": target_features,
                "prediction_mode": prediction_mode,
                "scenario": scenario_metadata(scenario),
                "rollout_length": int(args.rollout_length),
                "runtime_feature_names": feature_names,
                "summary": summary,
            },
        )

    save_json(
        output_dir / "summary.json",
        {
            "checkpoint": checkpoint_metadata(args.checkpoint, checkpoint),
            "target_features": target_features,
            "prediction_mode": prediction_mode,
            "scenario_count": len(summaries),
            "summaries": summaries,
        },
    )
    print_summary(output_dir, summaries)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-droplet closed-loop runtime rollout experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rollout-length", type=int, default=100)
    parser.add_argument("--scenario-count", type=int, default=5)
    parser.add_argument("--preferred-region", default="inlet channel")
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for single-droplet runtime rollout.") from exc
    return torch


def select_device(torch, mode: str):
    if mode == "cuda" or (mode == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config is empty or malformed: {path}")
    return data


def load_dataset(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def validate_current_feature_contract(feature_names: list[str], config: dict[str, Any]) -> None:
    expected = list(config["model"]["input_feature_names"])
    if feature_names != expected:
        raise ValueError(f"Dataset feature order does not match config.\nExpected: {expected}\nFound: {feature_names}")
    if feature_names != list(CANONICAL_RUNTIME_FEATURE_NAMES):
        raise ValueError(f"Runtime single-droplet rollout requires current 16-feature state, got {feature_names}")
    target_features = tuple(config["model"]["target_features"])
    if target_features not in {("vx", "vy", "bbox_w", "bbox_h"), ("x", "y", "bbox_w", "bbox_h")}:
        raise ValueError(
            "Runtime single-droplet rollout requires target_features to be either "
            "vx, vy, bbox_w, bbox_h or x, y, bbox_w, bbox_h"
        )


def prediction_mode_from_targets(target_features: tuple[str, ...]) -> str:
    if target_features == ("x", "y", "bbox_w", "bbox_h"):
        return "position"
    if target_features == ("vx", "vy", "bbox_w", "bbox_h"):
        return "velocity"
    raise ValueError(f"Unsupported checkpoint target_features for single-droplet rollout: {target_features}")


def load_model(torch, checkpoint: dict[str, Any], device):
    model = CanonicalRolloutTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def checkpoint_metadata(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {path}")
    return {
        "path": str(path),
        "checkpoint_name": Path(path).name,
        "is_latest_checkpoint": Path(path).name == "latest_checkpoint.pt",
        "epoch": int(checkpoint.get("epoch", -1)),
        "val_loss": float(checkpoint.get("val_loss", np.nan)),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def select_scenarios(
    *,
    dataset: dict[str, Any],
    feature_index: dict[str, int],
    region_labels: np.ndarray,
    stride: int,
    t_history: int,
    t_future: int,
    count: int,
    preferred_region: str,
    track_id: int | None,
    frame: int | None,
) -> list[dict[str, Any]]:
    if track_id is not None or frame is not None:
        if track_id is None or frame is None:
            raise ValueError("--track-id and --frame must be provided together.")
        return [scenario_from_track_frame(dataset, feature_index, int(track_id), int(frame), "explicit")]

    frames = np.asarray(dataset["frames"], dtype=np.int64)
    starts = np.arange(0, len(frames) - (int(t_history) + int(t_future)) + 1, int(stride), dtype=np.int64)
    train_end = int(0.70 * len(starts))
    val_end = int(0.85 * len(starts))
    candidates = []
    for start_index in starts[train_end:val_end]:
        frame_value = int(frames[start_index])
        active_tracks = np.flatnonzero(dataset["mask"][:, start_index].astype(bool))
        for track_index in active_tracks:
            state = dataset["Z"][track_index, start_index, :].astype(np.float32)
            if not initial_state_is_valid(state, feature_index):
                continue
            region = region_at_pixel(float(state[feature_index["x"]]), float(state[feature_index["y"]]), region_labels)
            preferred = region == preferred_region
            inside = region != "outside"
            if not inside:
                continue
            candidates.append((not preferred, frame_value, int(track_index), region))
    scenarios = []
    for rank, (_, frame_value, track_index, region) in enumerate(sorted(candidates)[: max(int(count), 0)]):
        track_id_value = int(dataset["track_ids"][track_index])
        scenario = scenario_from_track_frame(dataset, feature_index, track_id_value, frame_value, f"validation_{rank:02d}")
        scenario["initial_region"] = region
        scenarios.append(scenario)
    return scenarios


def scenario_from_track_frame(
    dataset: dict[str, Any],
    feature_index: dict[str, int],
    track_id: int,
    frame: int,
    name: str,
) -> dict[str, Any]:
    track_matches = np.flatnonzero(dataset["track_ids"].astype(int) == int(track_id))
    frame_matches = np.flatnonzero(dataset["frames"].astype(int) == int(frame))
    if track_matches.size == 0 or frame_matches.size == 0:
        raise KeyError(f"Cannot locate track_id={track_id}, frame={frame} in dataset.")
    track_index = int(track_matches[0])
    frame_index = int(frame_matches[0])
    if not bool(dataset["mask"][track_index, frame_index]):
        raise ValueError(f"Track {track_id} is not active at frame {frame}.")
    state = dataset["Z"][track_index, frame_index, :].astype(np.float32).copy()
    if not initial_state_is_valid(state, feature_index):
        raise ValueError(f"Initial state for track_id={track_id}, frame={frame} is invalid.")
    return {
        "scenario": name,
        "track_id": int(track_id),
        "track_index": track_index,
        "frame": int(frame),
        "frame_index": frame_index,
        "initial_state": state,
    }


def initial_state_is_valid(state: np.ndarray, feature_index: dict[str, int]) -> bool:
    required = ["x", "y", "vx", "vy", "bbox_w", "bbox_h", "cfd_u_norm", "cfd_v_norm", "superficial_velocity", "left_flow_fraction"]
    if not np.isfinite([state[feature_index[name]] for name in required]).all():
        return False
    return bool(state[feature_index["bbox_w"]] > 0.0 and state[feature_index["bbox_h"]] > 0.0)


def rollout_single_droplet(
    *,
    torch,
    model,
    device,
    initial_state: np.ndarray,
    feature_index: dict[str, int],
    normalization: dict[str, Any],
    runtime_context,
    rollout_length: int,
    scenario: dict[str, Any],
    prediction_mode: str,
) -> list[dict[str, Any]]:
    max_droplets = int(model.max_droplets)
    feature_dim = len(runtime_context.feature_names)
    state = np.zeros((max_droplets, feature_dim), dtype=np.float32)
    state[0] = initial_state.astype(np.float32)
    active_mask = np.zeros((max_droplets,), dtype=bool)
    active_mask[0] = True
    history = np.repeat(state[None, :, :], int(model.T_history), axis=0)
    rows = [row_from_state(0, state[0], feature_index, scenario, runtime_context, True, "initial_observed_state")]

    for step_index in range(1, int(rollout_length) + 1):
        history_norm = normalize_state(history, normalization, torch, device)
        history_mask = torch.as_tensor(
            np.repeat(active_mask.reshape(1, max_droplets), int(model.T_history), axis=0).reshape(1, int(model.T_history), max_droplets),
            dtype=torch.bool,
            device=device,
        )
        with torch.no_grad():
            pred_norm = model(history_norm.reshape(1, int(model.T_history), max_droplets, feature_dim), history_mask)[0]
            pred_phys = denormalize_target(pred_norm, normalization, torch, device).detach().cpu().numpy().astype(np.float32)
        try:
            next_state = physics_runtime_step(state, pred_phys, runtime_context, active_mask=active_mask, prediction_mode=prediction_mode)
            runtime_success = True
            source = "runtime_closed_loop"
        except Exception as exc:
            next_state = fallback_kinematic_step(state, pred_phys, feature_index, runtime_context, prediction_mode)
            runtime_success = False
            source = f"kinematic_fallback_after_runtime_error:{type(exc).__name__}"
        state = next_state.astype(np.float32)
        history = np.concatenate([history[1:], state[None, :, :]], axis=0)
        rows.append(row_from_state(step_index, state[0], feature_index, scenario, runtime_context, runtime_success, source))
    return rows


def normalize_state(history: np.ndarray, normalization: dict[str, Any], torch, device):
    mean = torch.as_tensor(normalization["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization["input_std"], dtype=torch.float32, device=device)
    value = torch.as_tensor(history, dtype=torch.float32, device=device)
    return (value - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def denormalize_target(target, normalization: dict[str, Any], torch, device):
    mean = torch.as_tensor(normalization["target_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization["target_std"], dtype=torch.float32, device=device)
    return target * std.view(1, -1) + mean.view(1, -1)


def fallback_kinematic_step(state: np.ndarray, prediction: np.ndarray, feature_index: dict[str, int], runtime_context, prediction_mode: str) -> np.ndarray:
    next_state = state.copy()
    idx = feature_index
    scale = float(runtime_context.velocity_mm_s_per_px_frame)
    if prediction_mode == "position":
        previous_x = float(next_state[0, idx["x"]])
        previous_y = float(next_state[0, idx["y"]])
        next_state[0, idx["x"]] = prediction[0, 0]
        next_state[0, idx["y"]] = prediction[0, 1]
        next_state[0, idx["vx"]] = (prediction[0, 0] - previous_x) * scale
        next_state[0, idx["vy"]] = (prediction[0, 1] - previous_y) * scale
    else:
        next_state[0, idx["x"]] += prediction[0, 0] / scale
        next_state[0, idx["y"]] += prediction[0, 1] / scale
        next_state[0, idx["vx"]] = prediction[0, 0]
        next_state[0, idx["vy"]] = prediction[0, 1]
    next_state[0, idx["bbox_w"]] = max(float(prediction[0, 2]), 1.0e-3)
    next_state[0, idx["bbox_h"]] = max(float(prediction[0, 3]), 1.0e-3)
    next_state[1:] = 0.0
    return next_state


def row_from_state(
    rollout_step: int,
    state: np.ndarray,
    feature_index: dict[str, int],
    scenario: dict[str, Any],
    runtime_context,
    runtime_success: bool,
    source: str,
) -> dict[str, Any]:
    x = float(state[feature_index["x"]])
    y = float(state[feature_index["y"]])
    occupancies = {name: float(state[feature_index[name]]) for name in OCCUPANCY_FEATURES}
    row = {
        "scenario": scenario["scenario"],
        "template_track_id": int(scenario["track_id"]),
        "template_frame": int(scenario["frame"]),
        "rollout_step": int(rollout_step),
        "x": x,
        "y": y,
        "vx": float(state[feature_index["vx"]]),
        "vy": float(state[feature_index["vy"]]),
        "speed": float(np.hypot(state[feature_index["vx"]], state[feature_index["vy"]])),
        "bbox_w": float(state[feature_index["bbox_w"]]),
        "bbox_h": float(state[feature_index["bbox_h"]]),
        "cfd_u_norm": float(state[feature_index["cfd_u_norm"]]),
        "cfd_v_norm": float(state[feature_index["cfd_v_norm"]]),
        "superficial_velocity": float(state[feature_index["superficial_velocity"]]),
        "left_flow_fraction": float(state[feature_index["left_flow_fraction"]]),
        "occupancy_sum": float(sum(occupancies.values())),
        "region": region_at_pixel(x, y, runtime_context.region_labels),
        "inside_channel": region_at_pixel(x, y, runtime_context.region_labels) != "outside",
        "runtime_success": bool(runtime_success),
        "state_source": source,
    }
    row.update(occupancies)
    return row


def region_at_pixel(x: float, y: float, region_labels: np.ndarray) -> str:
    col = int(round(x))
    row = int(round(y))
    if row < 0 or row >= region_labels.shape[0] or col < 0 or col >= region_labels.shape[1]:
        return "outside"
    return REGION_NAMES.get(int(region_labels[row, col]), "unknown")


def summarize(table: pd.DataFrame) -> dict[str, Any]:
    xy = table[["x", "y"]].to_numpy(float)
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    runtime_success = table["runtime_success"].astype(bool)
    return {
        "scenario": str(table["scenario"].iloc[0]),
        "template_track_id": int(table["template_track_id"].iloc[0]),
        "template_frame": int(table["template_frame"].iloc[0]),
        "rollout_steps": int(table["rollout_step"].max()),
        "final_x": float(table["x"].iloc[-1]),
        "final_y": float(table["y"].iloc[-1]),
        "total_traveled_distance_px": float(distance.sum()),
        "mean_speed": float(table["speed"].mean()),
        "max_speed": float(table["speed"].max()),
        "min_left_flow_fraction": float(table["left_flow_fraction"].min()),
        "max_left_flow_fraction": float(table["left_flow_fraction"].max()),
        "final_left_flow_fraction": float(table["left_flow_fraction"].iloc[-1]),
        "runtime_success_fraction": float(runtime_success.mean()),
        "runtime_fallback_steps": int((~runtime_success).sum()),
        "trajectory_ever_leaves_channel": bool((~table["inside_channel"].astype(bool)).any()),
        "regions_visited": sorted(table["region"].astype(str).unique().tolist()),
    }


def scenario_metadata(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": scenario["scenario"],
        "track_id": int(scenario["track_id"]),
        "frame": int(scenario["frame"]),
        "track_index": int(scenario["track_index"]),
        "frame_index": int(scenario["frame_index"]),
        "initial_region": scenario.get("initial_region"),
    }


def save_trajectory_plot(table: pd.DataFrame, channel_mask: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.imshow(channel_mask, cmap="gray", alpha=0.25, origin="upper")
    points = ax.scatter(table["x"], table["y"], c=table["rollout_step"], s=12, cmap="viridis")
    ax.scatter([table["x"].iloc[0]], [table["y"].iloc[0]], s=55, color="lime", edgecolor="black", label="start")
    ax.scatter([table["x"].iloc[-1]], [table["y"].iloc[-1]], s=55, color="red", edgecolor="black", label="end")
    ax.set_xlim(80, 560)
    ax.set_ylim(channel_mask.shape[0], 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(fontsize=8)
    fig.colorbar(points, ax=ax, label="rollout step")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_ellipse_occupancy_animation(table: pd.DataFrame, region_labels: np.ndarray, path: Path) -> None:
    frames = []
    channel_mask = region_labels > 0
    for _, current in table.iterrows():
        history = table.loc[table["rollout_step"] <= int(current["rollout_step"])]
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8])
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_occ = fig.add_subplot(gs[0, 1])

        ax_xy.imshow(channel_mask, cmap="gray", alpha=0.22, origin="upper")
        ax_xy.plot(history["x"], history["y"], color="#2563eb", linewidth=1.4, alpha=0.85)
        ax_xy.scatter([table["x"].iloc[0]], [table["y"].iloc[0]], s=38, color="#22c55e", edgecolor="black", linewidth=0.5)
        ax_xy.scatter([current["x"]], [current["y"]], s=42, color="#ef4444", edgecolor="black", linewidth=0.5)
        ellipse = Ellipse(
            xy=(float(current["x"]), float(current["y"])),
            width=float(current["bbox_w"]),
            height=float(current["bbox_h"]),
            angle=0.0,
            facecolor="#f97316",
            edgecolor="#7c2d12",
            alpha=0.42,
            linewidth=1.5,
        )
        ax_xy.add_patch(ellipse)
        cfd_dx, cfd_dy, cfd_available = cfd_arrow_delta(current)
        if cfd_available:
            ax_xy.arrow(
                float(current["x"]),
                float(current["y"]),
                cfd_dx,
                cfd_dy,
                width=1.0,
                head_width=7.0,
                head_length=8.5,
                length_includes_head=True,
                color="#06b6d4",
                alpha=0.9,
                zorder=5,
            )
        ax_xy.set_title(
            f"{current['scenario']} step {int(current['rollout_step'])} | {current['region']}\n"
            f"bbox={current['bbox_w']:.1f}x{current['bbox_h']:.1f} "
            f"runtime={bool(current['runtime_success'])} cfd={'on' if cfd_available else 'off'}",
            fontsize=10,
        )
        ax_xy.set_xlim(80, 560)
        ax_xy.set_ylim(region_labels.shape[0], 0)
        ax_xy.set_xlabel("x (px)")
        ax_xy.set_ylabel("y (px)")

        labels = [name.replace("occupancy_", "") for name in OCCUPANCY_FEATURES]
        values = [float(current[name]) for name in OCCUPANCY_FEATURES]
        colors = ["#64748b"] * len(values)
        if max(values) > 0.0:
            colors[int(np.argmax(values))] = "#f97316"
        y_pos = np.arange(len(labels))
        ax_occ.barh(y_pos, values, color=colors)
        ax_occ.set_yticks(y_pos)
        ax_occ.set_yticklabels(labels, fontsize=8)
        ax_occ.set_xlim(0.0, 1.0)
        ax_occ.set_xlabel("fraction")
        ax_occ.set_title(f"occupancy sum={float(current['occupancy_sum']):.3f}")
        ax_occ.grid(True, axis="x", alpha=0.25)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    write_mp4(path, frames, fps=10)


def cfd_arrow_delta(row: pd.Series, length_px: float = 32.0) -> tuple[float, float, bool]:
    u = float(row["cfd_u_norm"])
    v = float(row["cfd_v_norm"])
    if not np.isfinite(u) or not np.isfinite(v):
        return 0.0, 0.0, False
    norm = float(np.hypot(u, v))
    if norm <= 1.0e-8:
        return 0.0, 0.0, False
    return length_px * u / norm, -length_px * v / norm, True


def dominant_occupancy(row: pd.Series) -> str:
    values = [(name.replace("occupancy_", ""), float(row[name])) for name in OCCUPANCY_FEATURES]
    name, value = max(values, key=lambda item: item[1])
    if value <= 0.0:
        return "none"
    return f"{name} {value:.2f}"


def save_occupancy_fraction_plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for name in OCCUPANCY_FEATURES:
        label = name.replace("occupancy_", "")
        ax.plot(table["rollout_step"], table[name], linewidth=1.4, label=label)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("occupancy fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_speed_plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(table["rollout_step"], table["speed"])
    ax.set_xlabel("rollout step")
    ax.set_ylabel("speed")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_flow_plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(table["rollout_step"], table["left_flow_fraction"])
    ax.set_xlabel("rollout step")
    ax.set_ylabel("left_flow_fraction")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def print_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    print("Single-droplet runtime rollout complete")
    print(f"  output: {output_dir}")
    for summary in summaries:
        print(
            f"  {summary['scenario']}: track={summary['template_track_id']} frame={summary['template_frame']} "
            f"distance={summary['total_traveled_distance_px']:.2f}px "
            f"speed_mean={summary['mean_speed']:.3f} "
            f"left_flow={summary['min_left_flow_fraction']:.4f}..{summary['max_left_flow_fraction']:.4f} "
            f"runtime_success={summary['runtime_success_fraction']:.3f} "
            f"leaves_channel={summary['trajectory_ever_leaves_channel']}"
        )


if __name__ == "__main__":
    main()
