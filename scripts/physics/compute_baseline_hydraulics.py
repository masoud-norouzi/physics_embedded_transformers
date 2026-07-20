from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_experiment_config
from src.physics.hydraulics import (
    CANONICAL_ML_FEATURES,
    compute_baseline_hydraulic_state,
    compute_branch_flow_rates_ul_hr,
    compute_isolated_droplet_equivalent_length_um,
    flow_rate_ul_hr_to_superficial_velocity_um_s,
)
from src.physics.hydraulics.baseline import BASELINE_ASSUMPTIONS, DIAGNOSTIC_OUTPUTS


def _stats(series: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(series.min()),
        "maximum": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
    }


def _load_physical_constants(config: dict[str, dict[str, Any]]) -> dict[str, Any]:
    experiment = config["experiment"]["experiment"]
    device = config["device"]["device"]
    branches = device["loop"]["branches"]
    left_length = float(branches["left"]["length_um"])
    right_length = float(branches["right"]["length_um"])
    if left_length <= 0 or right_length <= 0:
        raise ValueError("Configured branch lengths must be positive")
    short_branch_name = "left" if left_length <= right_length else "right"
    short_length = min(left_length, right_length)
    resistance = device.get("hydraulics", {}).get("isolated_droplet_resistance", {})
    ratio = float(resistance.get("ratio_to_short_branch", 0.15))
    droplet_length = compute_isolated_droplet_equivalent_length_um(short_length, ratio)
    channel_width = float(device["channel"]["width_um"])
    channel_height = float(device["channel"]["height_um"])
    continuous_flow = _phase_flow_ul_hr(experiment, "continuous")
    dispersed_flow = _phase_flow_ul_hr(experiment, "dispersed")
    total_mixture_flow = continuous_flow + dispersed_flow
    if channel_width <= 0 or channel_height <= 0:
        raise ValueError("Configured channel dimensions must be positive")
    if total_mixture_flow <= 0:
        raise ValueError("Total mixture flow must be positive")
    return {
        "left_length_um": left_length,
        "right_length_um": right_length,
        "short_branch_name": short_branch_name,
        "short_branch_length_um": short_length,
        "isolated_droplet_resistance_ratio": ratio,
        "isolated_droplet_equivalent_length_um": droplet_length,
        "channel_width_um": channel_width,
        "channel_height_um": channel_height,
        "continuous_flow_ul_hr": continuous_flow,
        "dispersed_flow_ul_hr": dispersed_flow,
        "total_mixture_flow_ul_hr": total_mixture_flow,
        "dispersed_flow_fraction": dispersed_flow / total_mixture_flow,
    }


def _phase_flow_ul_hr(experiment: dict[str, Any], phase: str) -> float:
    try:
        value = experiment["phases"][phase]["flow_rate_ul_per_hr"]
    except KeyError as exc:
        raise ValueError(f"Experiment config is missing {phase} flow_rate_ul_per_hr") from exc
    flow = float(value)
    if not np.isfinite(flow):
        raise ValueError(f"{phase} flow_rate_ul_per_hr must be finite")
    if flow < 0:
        raise ValueError(f"{phase} flow_rate_ul_per_hr must be nonnegative")
    return flow


def _validate_occupancy(occupancy: pd.DataFrame) -> None:
    required = {"frame", "track_id", "occupancy_computable", "w_left", "w_right"}
    missing = required.difference(occupancy.columns)
    if missing:
        raise ValueError(f"Occupancy input is missing columns: {sorted(missing)}")
    if (~occupancy["occupancy_computable"].astype(bool)).any():
        raise ValueError("Occupancy input contains noncomputable rows")
    branch = occupancy[["w_left", "w_right"]].to_numpy(float)
    if not np.all(np.isfinite(branch)):
        raise ValueError("Normalized branch occupancies must be finite")
    if np.any(branch < -1e-12) or np.any(branch > 1 + 1e-12):
        raise ValueError("Normalized branch occupancies must lie in [0, 1]")


def _build_summary(
    state: pd.DataFrame,
    config: dict[str, dict[str, Any]],
    constants: dict[str, Any],
    occupancy_path: Path,
) -> dict[str, Any]:
    experiment = config["experiment"]["experiment"]
    device = config["device"]["device"]
    empty_left_flow, empty_right_flow = compute_branch_flow_rates_ul_hr(
        constants["total_mixture_flow_ul_hr"],
        constants["left_length_um"],
        constants["right_length_um"],
    )
    empty_left_velocity = flow_rate_ul_hr_to_superficial_velocity_um_s(
        empty_left_flow,
        constants["channel_width_um"],
        constants["channel_height_um"],
    )
    empty_right_velocity = flow_rate_ul_hr_to_superficial_velocity_um_s(
        empty_right_flow,
        constants["channel_width_um"],
        constants["channel_height_um"],
    )
    return {
        "experiment_id": experiment["id"],
        "device_id": device["id"],
        "occupancy_input_path": str(occupancy_path),
        "total_frame_count": int(len(state)),
        "frame_minimum": int(state["frame"].min()),
        "frame_maximum": int(state["frame"].max()),
        "base_left_branch_length_um": constants["left_length_um"],
        "base_right_branch_length_um": constants["right_length_um"],
        "short_branch_name": constants["short_branch_name"],
        "isolated_droplet_resistance_ratio": constants["isolated_droplet_resistance_ratio"],
        "isolated_droplet_equivalent_length_um": constants["isolated_droplet_equivalent_length_um"],
        "channel_width_um": constants["channel_width_um"],
        "channel_height_um": constants["channel_height_um"],
        "continuous_input_flow_ul_hr": constants["continuous_flow_ul_hr"],
        "dispersed_input_flow_ul_hr": constants["dispersed_flow_ul_hr"],
        "total_mixture_input_flow_ul_hr": constants["total_mixture_flow_ul_hr"],
        "dispersed_flow_fraction": constants["dispersed_flow_fraction"],
        "empty_channel_left_mixture_flow_ul_hr": empty_left_flow,
        "empty_channel_right_mixture_flow_ul_hr": empty_right_flow,
        "empty_channel_left_superficial_mixture_velocity_um_s": empty_left_velocity,
        "empty_channel_right_superficial_mixture_velocity_um_s": empty_right_velocity,
        "n_left_eff": _stats(state["n_left_eff"]),
        "n_right_eff": _stats(state["n_right_eff"]),
        "left_effective_length_um": _stats(state["left_effective_length_um"]),
        "right_effective_length_um": _stats(state["right_effective_length_um"]),
        "left_velocity_um_s": _stats(state["left_velocity_um_s"]),
        "right_velocity_um_s": _stats(state["right_velocity_um_s"]),
        "maximum_absolute_flow_conservation_error_ul_hr": float(
            state["flow_conservation_error_ul_hr"].abs().max()
        ),
        "frames_with_no_branch_occupancy": int(((state["n_left_eff"] == 0) & (state["n_right_eff"] == 0)).sum()),
        "baseline_assumptions": BASELINE_ASSUMPTIONS,
        "diagnostic_outputs": DIAGNOSTIC_OUTPUTS,
        "canonical_ml_features": CANONICAL_ML_FEATURES,
        "velocity_semantics": {
            "left_velocity_um_s": "total superficial mixture velocity in the left branch",
            "right_velocity_um_s": "total superficial mixture velocity in the right branch",
        },
        "flow_column_semantics": {
            "left_flow_ul_hr": "total mixture volumetric flow in the left branch",
            "right_flow_ul_hr": "total mixture volumetric flow in the right branch",
        },
    }


def _save_diagnostics(state: pd.DataFrame, summary: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.ravel()
    frame = state["frame"]
    axes[0].plot(frame, state["n_left_eff"], label="left")
    axes[0].plot(frame, state["n_right_eff"], label="right")
    axes[0].set_title("Effective branch occupancy")
    axes[0].legend()
    axes[1].plot(frame, state["left_effective_length_um"], label="left")
    axes[1].plot(frame, state["right_effective_length_um"], label="right")
    axes[1].set_title("Effective branch length")
    axes[1].legend()
    axes[2].plot(frame, state["left_velocity_um_s"], label="left")
    axes[2].plot(frame, state["right_velocity_um_s"], label="right")
    axes[2].set_title("Superficial velocity")
    axes[2].legend()
    velocity_diff = state["left_velocity_um_s"] - state["right_velocity_um_s"]
    axes[3].plot(frame, velocity_diff, color="black")
    axes[3].set_title("Velocity difference U_left - U_right")
    axes[4].scatter(state["n_left_eff"] - state["n_right_eff"], velocity_diff, s=4, alpha=0.35)
    axes[4].set_title("Occupancy difference vs velocity difference")
    axes[4].set_xlabel("N_left_eff - N_right_eff")
    axes[4].set_ylabel("U_left - U_right (um/s)")
    axes[5].axis("off")
    text = "\n".join(
        [
            f"l_d: {summary['isolated_droplet_equivalent_length_um']:.3f} um",
            f"Q_continuous: {summary['continuous_input_flow_ul_hr']:.3f} uL/hr",
            f"Q_dispersed: {summary['dispersed_input_flow_ul_hr']:.3f} uL/hr",
            f"Q_mixture: {summary['total_mixture_input_flow_ul_hr']:.3f} uL/hr",
            f"dispersed fraction: {summary['dispersed_flow_fraction']:.6f}",
            "empty split: "
            f"L={summary['empty_channel_left_mixture_flow_ul_hr']:.3f}, "
            f"R={summary['empty_channel_right_mixture_flow_ul_hr']:.3f} uL/hr",
            f"W x H: {summary['channel_width_um']:.1f} x {summary['channel_height_um']:.1f} um",
            "velocities: total superficial mixture velocities",
            f"max |flow err|: {summary['maximum_absolute_flow_conservation_error_ul_hr']:.3e} uL/hr",
            "",
            "assumptions:",
            *[f"- {item}" for item in summary["baseline_assumptions"]],
        ]
    )
    axes[5].text(0.0, 1.0, text, va="top", fontsize=8)
    for ax in axes[:4]:
        ax.set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute baseline branch hydraulics.")
    parser.add_argument("--experiment", required=True, type=Path, help="Experiment YAML path.")
    parser.add_argument("--occupancy", type=Path, default=Path("outputs/physics/video_2/droplet_occupancy.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics/video_2"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.occupancy.exists():
        raise FileNotFoundError(f"Occupancy input is missing: {args.occupancy}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "baseline_hydraulic_state.csv",
        args.output_dir / "baseline_hydraulic_summary.json",
        args.output_dir / "baseline_hydraulic_diagnostics.png",
    ]
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist. Use --overwrite: {existing}")

    config = load_experiment_config(args.experiment)
    constants = _load_physical_constants(config)
    occupancy = pd.read_csv(args.occupancy)
    _validate_occupancy(occupancy)
    state = compute_baseline_hydraulic_state(
        occupancy,
        left_length_um=constants["left_length_um"],
        right_length_um=constants["right_length_um"],
        droplet_equivalent_length_um=constants["isolated_droplet_equivalent_length_um"],
        total_mixture_flow_ul_hr=constants["total_mixture_flow_ul_hr"],
        channel_width_um=constants["channel_width_um"],
        channel_height_um=constants["channel_height_um"],
        continuous_flow_ul_hr=constants["continuous_flow_ul_hr"],
        dispersed_flow_ul_hr=constants["dispersed_flow_ul_hr"],
    )
    summary = _build_summary(state, config, constants, args.occupancy)
    state.to_csv(outputs[0], index=False)
    outputs[1].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_diagnostics(state, summary, outputs[2])

    print("Baseline hydraulics summary")
    print(f"  frames processed: {summary['total_frame_count']}")
    print(f"  continuous input flow: {summary['continuous_input_flow_ul_hr']:.6f} uL/hr")
    print(f"  dispersed input flow: {summary['dispersed_input_flow_ul_hr']:.6f} uL/hr")
    print(f"  total mixture input flow: {summary['total_mixture_input_flow_ul_hr']:.6f} uL/hr")
    print(f"  dispersed-flow fraction: {summary['dispersed_flow_fraction']:.6f}")
    print(f"  isolated droplet equivalent length: {summary['isolated_droplet_equivalent_length_um']:.3f} um")
    print(
        "  empty-channel flow split: "
        f"left={summary['empty_channel_left_mixture_flow_ul_hr']:.6f} uL/hr, "
        f"right={summary['empty_channel_right_mixture_flow_ul_hr']:.6f} uL/hr"
    )
    print(
        "  empty-channel velocities: "
        f"left={summary['empty_channel_left_superficial_mixture_velocity_um_s']:.6f} um/s, "
        f"right={summary['empty_channel_right_superficial_mixture_velocity_um_s']:.6f} um/s"
    )
    print(f"  n_left_eff range: {summary['n_left_eff']['minimum']:.6f} to {summary['n_left_eff']['maximum']:.6f}")
    print(f"  n_right_eff range: {summary['n_right_eff']['minimum']:.6f} to {summary['n_right_eff']['maximum']:.6f}")
    print(
        "  velocity ranges: "
        f"left={summary['left_velocity_um_s']['minimum']:.6f}..{summary['left_velocity_um_s']['maximum']:.6f}, "
        f"right={summary['right_velocity_um_s']['minimum']:.6f}..{summary['right_velocity_um_s']['maximum']:.6f} um/s"
    )
    print(
        "  max flow conservation error: "
        f"{summary['maximum_absolute_flow_conservation_error_ul_hr']:.3e} uL/hr"
    )
    print(f"  zero-branch-occupancy frames: {summary['frames_with_no_branch_occupancy']}")


if __name__ == "__main__":
    main()
