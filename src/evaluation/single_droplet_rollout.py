from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


FEATURE_NAMES = [
    "x",
    "y",
    "vx",
    "vy",
    "circularity",
    "cfd_u",
    "cfd_v",
    "left_flow_fraction",
    "occupancy_inlet_channel",
    "occupancy_inlet_junction",
    "occupancy_left_branch",
    "occupancy_right_branch",
    "occupancy_outlet_junction",
    "occupancy_outlet_channel",
    "cfd_valid",
]
OCCUPANCY_FEATURES = FEATURE_NAMES[8:14]
REGION_NAMES = {
    0: "outside",
    1: "inlet channel",
    2: "outlet channel",
    3: "left branch",
    4: "right branch",
    5: "inlet junction",
    6: "outlet junction",
}
REGION_TO_OCCUPANCY = {
    "inlet channel": "occupancy_inlet_channel",
    "inlet junction": "occupancy_inlet_junction",
    "left branch": "occupancy_left_branch",
    "right branch": "occupancy_right_branch",
    "outlet junction": "occupancy_outlet_junction",
    "outlet channel": "occupancy_outlet_channel",
}

DEFAULT_CONFIG = Path("configs/experiments/physics_markovian_v1.yml")
DEFAULT_EXPERIMENT = Path("configs/experiments/video_2.yml")
DEFAULT_DATASET = Path("outputs/processed/2/canonical_dataset_v2/canonical_dataset_v2.npz")
DEFAULT_ENRICHED = Path("outputs/physics/video_2/enrichment/physics_enriched_tracked_features.csv")
DEFAULT_CHECKPOINT = Path("outputs/models/physics_markovian_v1/best_checkpoint.pt")
DEFAULT_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_REGION_LABELS = Path("data/geometry/asymmetric_loop_h100/region_labels.npy")
DEFAULT_OUTPUT = Path("outputs/evaluation/single_droplet")
DEFAULT_LEAD_FRAMES = (80, 60, 40, 20, 10)
NEIGHBOR_COUNT = 10
CFD_CONTRACT_SAMPLE_COUNT = 100
CFD_MAX_ABS_TOLERANCE = 5.0e-6
CFD_MEAN_ABS_TOLERANCE = 1.0e-6
CFD_VALID_AGREEMENT_MINIMUM = 0.99

RAW_TO_CANONICAL_REGION = {
    "inlet": "inlet channel",
    "inlet_channel": "inlet channel",
    "upper_junction": "inlet junction",
    "inlet_junction": "inlet junction",
    "left": "left branch",
    "left_branch": "left branch",
    "right": "right branch",
    "right_branch": "right branch",
    "lower_junction": "outlet junction",
    "outlet_junction": "outlet junction",
    "outlet": "outlet channel",
    "outlet_channel": "outlet channel",
    "inlet channel": "inlet channel",
    "inlet junction": "inlet junction",
    "left branch": "left branch",
    "right branch": "right branch",
    "outlet junction": "outlet junction",
    "outlet channel": "outlet channel",
}


@dataclass
class HistoricalLookup:
    positions: np.ndarray
    values: np.ndarray
    feature_index: dict[str, int]


@dataclass
class PhysicsContext:
    geometry: Any
    cfd_library: Any
    region_labels: np.ndarray
    hydraulic_constants: dict[str, float]
    cfd_min_split: float
    cfd_max_split: float
    fallback_events: list[dict[str, Any]] = field(default_factory=list)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch = import_torch()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_config = load_yaml(args.config)
    dataset = load_canonical_dataset(args.dataset)
    dataset_feature_names = [str(name) for name in dataset["feature_names"]]
    feature_index = {name: idx for idx, name in enumerate(dataset_feature_names)}
    validate_feature_contract(dataset_feature_names, train_config)

    device = select_device(torch, args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_info = checkpoint_metadata(args.checkpoint, checkpoint)
    model = load_model(torch, checkpoint, device)
    normalization = checkpoint["normalization_stats"]

    context = build_physics_context(args)
    historical = build_historical_lookup(dataset, feature_index)
    contract = validate_cfd_feature_contract(
        dataset=dataset,
        feature_index=feature_index,
        context=context,
        sample_count=CFD_CONTRACT_SAMPLE_COUNT,
        output_path=output_root / "cfd_feature_contract_validation.json",
    )

    validation_info = validation_start_frame_values(
        frames=dataset["frames"],
        stride=int(train_config["dataset"]["stride"]),
        t_history=int(train_config["model"]["T_history"]),
        t_future=int(train_config["model"]["rollout_horizon"]),
    )
    enriched = pd.read_csv(
        args.enriched,
        usecols=["frame", "track_id", "centroid_x", "centroid_y", "dominant_region", "cfd_valid"],
    )
    region_info = normalize_enriched_regions(enriched)
    scenarios = select_scenarios(
        enriched=enriched,
        region_column=region_info["normalized_column"],
        dataset=dataset,
        feature_index=feature_index,
        requested_leads=parse_leads(args.lead_frames),
        validation_starts=validation_info["validation_start_frame_values"],
        explicit_track_id=args.track_id,
        explicit_frame=args.frame,
    )
    if not scenarios:
        raise RuntimeError("No suitable validation-template droplets were found.")

    summaries = []
    trajectories = []
    velocity_units = state_velocity_units(dataset)
    for scenario in scenarios:
        scenario_dir = output_root / scenario["scenario"]
        scenario_dir.mkdir(parents=True, exist_ok=True)
        rows = rollout_single_scenario(
            torch=torch,
            model=model,
            device=device,
            dataset=dataset,
            feature_index=feature_index,
            normalization=normalization,
            scenario=scenario,
            rollout_length=int(args.rollout_length),
            context=context,
            historical=historical,
            velocity_to_px_frame=velocity_to_px_frame_scale_from_config(dataset, args.experiment_config),
        )
        table = pd.DataFrame(rows)
        table["velocity_units"] = velocity_units
        table.to_csv(scenario_dir / "trajectory.csv", index=False)
        table.to_csv(scenario_dir / "state_reconstruction.csv", index=False)
        save_velocity_profile(table, scenario_dir / "velocity_profile.png")
        save_speed_plot(table, scenario_dir / "speed_vs_time.png")
        save_left_flow_fraction_plot(table, scenario_dir / "left_flow_fraction_vs_time.png")
        save_overlay_video(table, context.region_labels > 0, scenario_dir / "trajectory_overlay.mp4")
        summary = summarize_trajectory(table)
        metadata = build_metadata(args, scenario, summary, context, checkpoint_info, region_info, validation_info, contract)
        save_json(scenario_dir / "metadata.json", metadata)
        summaries.append(summary)
        trajectories.append(table)

    cross = cross_initialization_summary(trajectories, summaries)
    save_json(output_root / "cross_initialization_summary.json", cross)
    save_json(
        output_root / "summary.json",
        {
            "checkpoint": checkpoint_info,
            "cfd_feature_contract": contract,
            "region_normalization": region_info,
            "validation_split": serializable_validation_info(validation_info),
            "scenarios": summaries,
            "cross_initialization": cross,
        },
    )
    print_summary(summaries, cross, output_root)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-droplet physical sanity rollout.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--region-labels", type=Path, default=DEFAULT_REGION_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rollout-length", type=int, default=100)
    parser.add_argument("--lead-frames", default=",".join(str(item) for item in DEFAULT_LEAD_FRAMES))
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for single-droplet rollout.") from exc
    return torch


def load_model(torch, checkpoint: dict[str, Any], device):
    from src.models.canonical_rollout_transformer import CanonicalRolloutTransformer

    model = CanonicalRolloutTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def select_device(torch, mode: str):
    if mode == "cuda" or (mode == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_metadata(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {path}")
    if Path(path).name == "latest_checkpoint.pt":
        raise ValueError("Default single-droplet evaluation must use a frozen best checkpoint, not latest_checkpoint.pt.")
    return {
        "path": str(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "validation_metric": "val_loss",
        "validation_metric_value": float(checkpoint.get("val_loss", np.nan)),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "selection_reason": "lowest validation loss checkpoint from physics_markovian_v1 output directory",
    }


def build_physics_context(args: argparse.Namespace) -> PhysicsContext:
    from src.config import load_experiment_config
    from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry
    from src.physics.hydraulics import compute_isolated_droplet_equivalent_length_um
    from src.physics.interpolation import VelocityFieldLibrary

    cfg = load_experiment_config(args.experiment_config)
    experiment = cfg["experiment"]["experiment"]
    device = cfg["device"]["device"]
    branches = device["loop"]["branches"]
    left_length = float(branches["left"]["length_um"])
    right_length = float(branches["right"]["length_um"])
    short_length = min(left_length, right_length)
    resistance_ratio = float(device.get("hydraulics", {}).get("isolated_droplet_resistance", {}).get("ratio_to_short_branch", 0.15))
    hydraulic_constants = {
        "left_length_um": left_length,
        "right_length_um": right_length,
        "droplet_equivalent_length_um": compute_isolated_droplet_equivalent_length_um(short_length, resistance_ratio),
        "total_mixture_flow_ul_hr": phase_flow(experiment, "continuous") + phase_flow(experiment, "dispersed"),
        "channel_width_um": float(device["channel"]["width_um"]),
        "channel_height_um": float(device["channel"]["height_um"]),
        "continuous_flow_ul_hr": phase_flow(experiment, "continuous"),
        "dispersed_flow_ul_hr": phase_flow(experiment, "dispersed"),
    }
    geometry = build_full_device_cfd_geometry()
    library = VelocityFieldLibrary.from_directory(args.cfd_library)
    return PhysicsContext(
        geometry=geometry,
        cfd_library=library,
        region_labels=np.load(args.region_labels),
        hydraulic_constants=hydraulic_constants,
        cfd_min_split=float(min(library.fractions)),
        cfd_max_split=float(max(library.fractions)),
    )


def phase_flow(experiment: dict[str, Any], phase: str) -> float:
    return float(experiment["phases"][phase]["flow_rate_ul_per_hr"])


def build_historical_lookup(dataset: dict[str, Any], feature_index: dict[str, int]) -> HistoricalLookup:
    mask = dataset["mask"].astype(bool)
    tracks, frames = np.nonzero(mask)
    values = dataset["Z"][tracks, frames, :].astype(np.float32)
    valid = (
        np.isfinite(values[:, feature_index["x"]])
        & np.isfinite(values[:, feature_index["y"]])
        & (values[:, feature_index["cfd_valid"]] >= 0.5)
    )
    values = values[valid]
    positions = values[:, [feature_index["x"], feature_index["y"]]].astype(np.float32)
    if len(positions) < NEIGHBOR_COUNT:
        raise ValueError("Not enough historical valid states for nearest-neighbor reconstruction.")
    return HistoricalLookup(positions=positions, values=values, feature_index=feature_index)


def validate_cfd_feature_contract(
    dataset: dict[str, Any],
    feature_index: dict[str, int],
    context: PhysicsContext,
    sample_count: int,
    output_path: Path,
) -> dict[str, Any]:
    mask = dataset["mask"].astype(bool)
    tracks, frames = np.nonzero(mask)
    values = dataset["Z"][tracks, frames, :].astype(np.float64)
    valid = (
        np.isfinite(values[:, feature_index["x"]])
        & np.isfinite(values[:, feature_index["y"]])
        & np.isfinite(values[:, feature_index["left_flow_fraction"]])
        & np.isfinite(values[:, feature_index["cfd_u"]])
        & np.isfinite(values[:, feature_index["cfd_v"]])
        & (values[:, feature_index["cfd_valid"]] >= 0.5)
    )
    candidates = values[valid]
    if len(candidates) < sample_count:
        raise ValueError(f"Need at least {sample_count} valid historical CFD rows; found {len(candidates)}")
    rng = np.random.default_rng(20260722)
    selected = candidates[rng.choice(len(candidates), size=sample_count, replace=False)]
    sampled_u = np.empty(sample_count, dtype=float)
    sampled_v = np.empty(sample_count, dtype=float)
    sampled_valid = np.empty(sample_count, dtype=bool)
    for idx, row in enumerate(selected):
        split = float(row[feature_index["left_flow_fraction"]])
        field = context.cfd_library.interpolate(split)
        point_px = row[[feature_index["x"], feature_index["y"]]].reshape(1, 2)
        point_device = context.geometry.convention.image_points_to_device(point_px)
        sample = field.sample_cfd(point_device)
        sampled_u[idx] = float(sample.cfd_u[0]) if bool(sample.cfd_valid[0]) else np.nan
        sampled_v[idx] = float(sample.cfd_v[0]) if bool(sample.cfd_valid[0]) else np.nan
        sampled_valid[idx] = bool(sample.cfd_valid[0])

    stored_u = selected[:, feature_index["cfd_u"]]
    stored_v = selected[:, feature_index["cfd_v"]]
    valid_agreement = sampled_valid == (selected[:, feature_index["cfd_valid"]] >= 0.5)
    finite = sampled_valid & np.isfinite(sampled_u) & np.isfinite(sampled_v)
    if not np.any(finite):
        raise ValueError("CFD feature-contract validation produced no finite sampled rows.")
    err_u = np.abs(sampled_u[finite] - stored_u[finite])
    err_v = np.abs(sampled_v[finite] - stored_v[finite])
    sign_correlation_u = safe_corr(sampled_u[finite], stored_u[finite])
    sign_correlation_v = safe_corr(sampled_v[finite], stored_v[finite])
    report = {
        "rows_tested": int(sample_count),
        "finite_rows_compared": int(np.count_nonzero(finite)),
        "max_abs_error_cfd_u": float(np.max(err_u)),
        "mean_abs_error_cfd_u": float(np.mean(err_u)),
        "max_abs_error_cfd_v": float(np.max(err_v)),
        "mean_abs_error_cfd_v": float(np.mean(err_v)),
        "cfd_valid_agreement_rate": float(np.mean(valid_agreement)),
        "sign_correlation_cfd_u": sign_correlation_u,
        "sign_correlation_cfd_v": sign_correlation_v,
        "coordinate_convention": "image px -> device um via production full-device geometry convention; sample_device returns device-frame m/s",
        "tolerances": {
            "max_abs_error": CFD_MAX_ABS_TOLERANCE,
            "mean_abs_error": CFD_MEAN_ABS_TOLERANCE,
            "minimum_valid_agreement": CFD_VALID_AGREEMENT_MINIMUM,
        },
        "passed": False,
    }
    report["passed"] = bool(
        report["max_abs_error_cfd_u"] <= CFD_MAX_ABS_TOLERANCE
        and report["max_abs_error_cfd_v"] <= CFD_MAX_ABS_TOLERANCE
        and report["mean_abs_error_cfd_u"] <= CFD_MEAN_ABS_TOLERANCE
        and report["mean_abs_error_cfd_v"] <= CFD_MEAN_ABS_TOLERANCE
        and report["cfd_valid_agreement_rate"] >= CFD_VALID_AGREEMENT_MINIMUM
    )
    save_json(output_path, report)
    if not report["passed"]:
        raise ValueError(f"CFD feature-contract validation failed; see {output_path}")
    return report


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.nanstd(a) == 0.0 or np.nanstd(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config is empty or malformed: {path}")
    return data


def load_canonical_dataset(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def validate_feature_contract(dataset_feature_names: list[str], config: dict[str, Any]) -> None:
    expected_feature_names = list(config["model"]["input_feature_names"])
    if expected_feature_names != FEATURE_NAMES:
        raise ValueError("Training config feature order does not match the expected 15-feature state.")
    if dataset_feature_names != expected_feature_names:
        raise ValueError(
            "Canonical dataset feature order does not match the trained model.\n"
            f"Expected: {expected_feature_names}\n"
            f"Found:    {dataset_feature_names}"
        )


def parse_leads(text: str) -> list[int]:
    leads = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not leads:
        raise ValueError("At least one lead frame must be requested.")
    return leads


def validation_start_frame_values(frames: np.ndarray, stride: int, t_history: int, t_future: int) -> dict[str, Any]:
    frame_values = np.asarray(frames, dtype=np.int64)
    starts = np.arange(0, len(frame_values) - (t_history + t_future) + 1, stride, dtype=np.int64)
    train_end = int(0.70 * len(starts))
    val_end = int(0.85 * len(starts))
    val_start_indices = starts[train_end:val_end]
    val_start_frames = {int(frame_values[idx]) for idx in val_start_indices}
    diffs = np.diff(frame_values)
    return {
        "split_function_used": "canonical_window_dataset convention: np.arange start indices, 70/15/15 split, then map indices through dataset['frames']",
        "stride": int(stride),
        "t_history": int(t_history),
        "t_future": int(t_future),
        "train_window_count": int(train_end),
        "validation_window_count": int(len(val_start_indices)),
        "test_window_count": int(len(starts) - val_end),
        "validation_start_indices": val_start_indices,
        "validation_start_frame_values": val_start_frames,
        "minimum_validation_start_frame": int(min(val_start_frames)) if val_start_frames else None,
        "maximum_validation_start_frame": int(max(val_start_frames)) if val_start_frames else None,
        "frame_values_contiguous": bool(np.all(diffs == 1)) if len(diffs) else True,
    }


def serializable_validation_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (sorted(value) if isinstance(value, set) else value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in info.items()
    }


def normalize_enriched_regions(enriched: pd.DataFrame) -> dict[str, Any]:
    if "dominant_region" not in enriched.columns:
        raise KeyError("Enriched dataset is missing dominant_region.")
    raw_labels = sorted(enriched["dominant_region"].dropna().astype(str).unique().tolist())
    mapping = {}
    for label in raw_labels:
        if label not in RAW_TO_CANONICAL_REGION:
            raise ValueError(f"Unsupported enriched dominant_region label: {label!r}")
        mapping[label] = RAW_TO_CANONICAL_REGION[label]
    required = {"inlet channel", "inlet junction"}
    normalized_values = {mapping[label] for label in raw_labels}
    missing = sorted(required.difference(normalized_values))
    if missing:
        raise ValueError(f"Cannot identify required canonical scenario-selection regions: {missing}")
    column = "dominant_region_canonical"
    enriched[column] = enriched["dominant_region"].astype(str).map(mapping)
    return {
        "raw_labels": raw_labels,
        "normalization_map": mapping,
        "normalized_column": column,
        "canonical_labels": sorted(normalized_values),
    }


def select_scenarios(
    enriched: pd.DataFrame,
    region_column: str,
    dataset: dict[str, Any],
    feature_index: dict[str, int],
    requested_leads: list[int],
    validation_starts: set[int],
    explicit_track_id: int | None,
    explicit_frame: int | None,
) -> list[dict[str, Any]]:
    if explicit_track_id is not None or explicit_frame is not None:
        if explicit_track_id is None or explicit_frame is None:
            raise ValueError("--track-id and --frame must be provided together.")
        return [build_scenario(dataset, feature_index, explicit_track_id, explicit_frame, None, None)]

    candidates = []
    valid = enriched[enriched["cfd_valid"].astype(bool)].copy()
    for track_id, group in valid.sort_values("frame").groupby("track_id"):
        frames = group["frame"].astype(int).to_numpy()
        regions = group[region_column].astype(str).to_numpy()
        junction_indices = np.flatnonzero(regions == "inlet junction")
        if junction_indices.size == 0:
            continue
        junction_frame = int(frames[junction_indices[0]])
        before = group[(group["frame"] < junction_frame) & (group[region_column].astype(str) == "inlet channel")]
        for _, row in before.iterrows():
            source_frame = int(row["frame"])
            if source_frame in validation_starts:
                lead = junction_frame - source_frame
                if lead > 0:
                    candidates.append((int(track_id), source_frame, junction_frame, lead))

    scenarios = []
    used = set()
    for requested in requested_leads:
        available = [item for item in candidates if (item[0], item[1]) not in used]
        if not available:
            continue
        selected = min(available, key=lambda item: (abs(item[3] - requested), item[1], item[0]))
        used.add((selected[0], selected[1]))
        scenarios.append(build_scenario(dataset, feature_index, selected[0], selected[1], requested, selected[2]))
    return scenarios


def build_scenario(
    dataset: dict[str, Any],
    feature_index: dict[str, int],
    track_id: int,
    source_frame: int,
    requested_lead: int | None,
    junction_entry_frame: int | None,
) -> dict[str, Any]:
    track_matches = np.flatnonzero(dataset["track_ids"].astype(int) == int(track_id))
    frame_matches = np.flatnonzero(dataset["frames"].astype(int) == int(source_frame))
    if track_matches.size == 0 or frame_matches.size == 0:
        raise KeyError(f"Cannot locate track={track_id}, frame={source_frame} in canonical dataset.")
    track_index = int(track_matches[0])
    frame_index = int(frame_matches[0])
    if not bool(dataset["mask"][track_index, frame_index]):
        raise ValueError(f"Track {track_id} is not present at frame {source_frame}.")
    initial = dataset["Z"][track_index, frame_index, :].astype(np.float32).copy()
    if not np.isfinite(initial).all():
        raise ValueError("Initial observed 15-dimensional state must be finite.")
    return {
        "scenario": f"lead_{requested_lead if requested_lead is not None else 'explicit'}",
        "requested_lead_frames": requested_lead,
        "actual_lead_frames": None if junction_entry_frame is None else int(junction_entry_frame - source_frame),
        "junction_entry_frame": junction_entry_frame,
        "track_id": int(track_id),
        "source_frame": int(source_frame),
        "track_index": track_index,
        "frame_index": frame_index,
        "initial_state": initial,
        "initial_x": float(initial[feature_index["x"]]),
        "initial_y": float(initial[feature_index["y"]]),
    }


def rollout_single_scenario(
    *,
    torch,
    model,
    device,
    dataset: dict[str, Any],
    feature_index: dict[str, int],
    normalization: dict[str, Any],
    scenario: dict[str, Any],
    rollout_length: int,
    context: PhysicsContext,
    historical: HistoricalLookup,
    velocity_to_px_frame: float,
) -> list[dict[str, Any]]:
    max_droplets = int(model.max_droplets)
    feature_dim = len(dataset["feature_names"])
    state = np.zeros((max_droplets, feature_dim), dtype=np.float32)
    state[0] = scenario["initial_state"]
    mask = np.zeros((max_droplets,), dtype=bool)
    mask[0] = True
    assert_exactly_one_active_slot(state, mask)
    initial_historical = state[0].copy()
    initial_reconstruction = reconstruct_nonpredicted_state(
        state[0],
        feature_index,
        context,
        historical,
        step=0,
        scenario_name=scenario["scenario"],
    )
    assert_initial_physics_reconstructed(initial_historical, state[0], feature_index)

    rows = [
        row_from_state(
            0,
            state[0],
            feature_index,
            scenario,
            context,
            initial_reconstruction["nearest_neighbor_distance_mean"],
            initial_reconstruction["nearest_neighbor_distance_max"],
            "observed_kinematics_reconstructed_single_droplet_physics",
        )
    ]
    for step in range(1, rollout_length + 1):
        history_state = normalize_state(state, normalization, device, torch)
        history_state[~torch.as_tensor(mask, dtype=torch.bool, device=device)] = 0.0
        history = history_state.view(1, 1, max_droplets, feature_dim)
        history_mask = torch.as_tensor(mask.reshape(1, 1, max_droplets), dtype=torch.bool, device=device)
        with torch.no_grad():
            pred_norm = model(history, history_mask)[0, 0, :]
        pred_velocity = denormalize_target(pred_norm, normalization, device, torch).detach().cpu().numpy()
        next_state = state[0].copy()
        next_state[feature_index["x"]] = np.float32(next_state[feature_index["x"]] + pred_velocity[0] * velocity_to_px_frame)
        next_state[feature_index["y"]] = np.float32(next_state[feature_index["y"]] + pred_velocity[1] * velocity_to_px_frame)
        next_state[feature_index["vx"]] = np.float32(pred_velocity[0])
        next_state[feature_index["vy"]] = np.float32(pred_velocity[1])
        reconstruction = reconstruct_nonpredicted_state(next_state, feature_index, context, historical, step, scenario["scenario"])
        state[:] = 0.0
        state[0] = next_state
        assert_exactly_one_active_slot(state, mask)
        rows.append(
            row_from_state(
                step,
                state[0],
                feature_index,
                scenario,
                context,
                reconstruction["nearest_neighbor_distance_mean"],
                reconstruction["nearest_neighbor_distance_max"],
                reconstruction["reconstruction_source"],
            )
        )
    return rows


def velocity_to_px_frame_scale_from_config(dataset: dict[str, Any], experiment_config: Path) -> float:
    velocity_units = state_velocity_units(dataset)
    if velocity_units == "mm/s":
        from src.config.velocity import load_velocity_conversion_from_experiment

        conversion = float(load_velocity_conversion_from_experiment(experiment_config)["velocity_mm_s_per_px_frame"])
        if conversion <= 0 or not np.isfinite(conversion):
            raise ValueError(f"Invalid velocity conversion factor: {conversion}")
        return 1.0 / conversion
    return 1.0


def state_velocity_units(dataset: dict[str, Any]) -> str:
    return str(dataset["velocity_units"]) if "velocity_units" in dataset else "px/frame"


def reconstruct_nonpredicted_state(
    state: np.ndarray,
    feature_index: dict[str, int],
    context: PhysicsContext,
    historical: HistoricalLookup,
    step: int,
    scenario_name: str,
) -> dict[str, Any]:
    xy = state[[feature_index["x"], feature_index["y"]]].astype(float)
    estimate = estimate_shape_and_occupancy(xy, historical)
    state[feature_index["circularity"]] = np.float32(estimate["circularity"])
    if region_at_pixel(float(xy[0]), float(xy[1]), context.region_labels) == "outside":
        for name in OCCUPANCY_FEATURES:
            state[feature_index[name]] = 0.0
        estimate["occupancy"] = {name: 0.0 for name in OCCUPANCY_FEATURES}
        estimate["reconstruction_source"] = "outside_defined_regions_all_zero_occupancy"
    else:
        for name, value in estimate["occupancy"].items():
            state[feature_index[name]] = np.float32(value)
    hydraulic = compute_single_droplet_hydraulics(state, feature_index, context)
    left_fraction = float(hydraulic["left_flow_ul_hr"] / (hydraulic["left_flow_ul_hr"] + hydraulic["right_flow_ul_hr"]))
    if not np.isfinite(left_fraction) or not (0.0 <= left_fraction <= 1.0):
        raise ValueError(f"Invalid left_flow_fraction at step {step}: {left_fraction}")
    state[feature_index["left_flow_fraction"]] = np.float32(left_fraction)
    sample_cfd_at_current_split(state, feature_index, context, step, scenario_name)
    validate_reconstructed_state(state, feature_index)
    return estimate


def estimate_shape_and_occupancy(xy: np.ndarray, historical: HistoricalLookup) -> dict[str, Any]:
    diff = historical.positions - xy.reshape(1, 2)
    dist2 = np.einsum("ij,ij->i", diff, diff)
    k = min(NEIGHBOR_COUNT, len(dist2))
    nn = np.argpartition(dist2, k - 1)[:k]
    distances = np.sqrt(dist2[nn])
    weights = 1.0 / np.maximum(distances, 1.0e-6)
    weights = weights / weights.sum()
    values = historical.values[nn]
    idx = historical.feature_index
    circularity = float(np.sum(weights * values[:, idx["circularity"]]))
    occ_values = np.array([np.sum(weights * values[:, idx[name]]) for name in OCCUPANCY_FEATURES], dtype=float)
    occ_values = np.where(np.abs(occ_values) < 1.0e-12, 0.0, occ_values)
    occ_values = np.clip(occ_values, 0.0, None)
    total = float(occ_values.sum())
    if total > 0.0:
        occ_values /= total
    return {
        "circularity": circularity,
        "occupancy": dict(zip(OCCUPANCY_FEATURES, occ_values)),
        "nearest_neighbor_distance_mean": float(np.mean(distances)),
        "nearest_neighbor_distance_max": float(np.max(distances)),
        "reconstruction_source": "inverse_distance_10_nearest_historical_positions",
    }


def compute_single_droplet_hydraulics(state: np.ndarray, feature_index: dict[str, int], context: PhysicsContext) -> dict[str, Any]:
    from src.physics.hydraulics import compute_frame_baseline_hydraulics

    occupancy = pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 1,
                "occupancy_computable": True,
                "w_left": float(state[feature_index["occupancy_left_branch"]]),
                "w_right": float(state[feature_index["occupancy_right_branch"]]),
            }
        ]
    )
    constants = context.hydraulic_constants
    return compute_frame_baseline_hydraulics(
        occupancy,
        frame=0,
        left_length_um=constants["left_length_um"],
        right_length_um=constants["right_length_um"],
        droplet_equivalent_length_um=constants["droplet_equivalent_length_um"],
        total_mixture_flow_ul_hr=constants["total_mixture_flow_ul_hr"],
        channel_width_um=constants["channel_width_um"],
        channel_height_um=constants["channel_height_um"],
        continuous_flow_ul_hr=constants["continuous_flow_ul_hr"],
        dispersed_flow_ul_hr=constants["dispersed_flow_ul_hr"],
    )


def sample_cfd_at_current_split(
    state: np.ndarray,
    feature_index: dict[str, int],
    context: PhysicsContext,
    step: int,
    scenario_name: str,
) -> None:
    split = float(state[feature_index["left_flow_fraction"]])
    sample_split = float(np.clip(split, context.cfd_min_split, context.cfd_max_split))
    if sample_split != split:
        context.fallback_events.append(
            {
                "scenario": scenario_name,
                "step": step,
                "left_flow_fraction": split,
                "sampled_cfd_split": sample_split,
            }
        )
    field = context.cfd_library.interpolate(sample_split)
    point_px = np.array([[state[feature_index["x"]], state[feature_index["y"]]]], dtype=float)
    point_device = context.geometry.convention.image_points_to_device(point_px)
    sample = field.sample_cfd(point_device)
    if bool(sample.cfd_valid[0]) and np.isfinite(sample.cfd_u[0]) and np.isfinite(sample.cfd_v[0]):
        state[feature_index["cfd_u"]] = np.float32(sample.cfd_u[0])
        state[feature_index["cfd_v"]] = np.float32(sample.cfd_v[0])
        state[feature_index["cfd_valid"]] = 1.0
    else:
        state[feature_index["cfd_u"]] = 0.0
        state[feature_index["cfd_v"]] = 0.0
        state[feature_index["cfd_valid"]] = 0.0


def validate_reconstructed_state(state: np.ndarray, feature_index: dict[str, int]) -> None:
    occ = np.array([state[feature_index[name]] for name in OCCUPANCY_FEATURES], dtype=float)
    if not np.isfinite(occ).all():
        raise ValueError("Occupancy fractions must be finite.")
    total = float(occ.sum())
    if total > 0.0 and not np.isclose(total, 1.0, atol=1.0e-5):
        raise ValueError(f"Occupancy fractions do not sum to one: {total}")
    split = float(state[feature_index["left_flow_fraction"]])
    if not np.isfinite(split) or not (0.0 <= split <= 1.0):
        raise ValueError(f"left_flow_fraction is invalid: {split}")
    if not np.isfinite(state[feature_index["cfd_u"]]) or not np.isfinite(state[feature_index["cfd_v"]]):
        raise ValueError("cfd_u and cfd_v must be finite numeric values after invalid handling.")


def assert_initial_physics_reconstructed(before: np.ndarray, after: np.ndarray, feature_index: dict[str, int]) -> None:
    for name in ("x", "y", "vx", "vy"):
        if not np.isclose(before[feature_index[name]], after[feature_index[name]], atol=1.0e-7):
            raise AssertionError(f"Initial {name} must preserve observed kinematics.")
    checked = ["left_flow_fraction", "cfd_u", "cfd_v", "cfd_valid", "circularity", *OCCUPANCY_FEATURES]
    if not all(np.isfinite(after[feature_index[name]]) for name in checked):
        raise AssertionError("Initial reconstructed physics state contains non-finite values.")


def assert_exactly_one_active_slot(state: np.ndarray, mask: np.ndarray) -> None:
    if int(mask.sum()) != 1 or not bool(mask[0]):
        raise AssertionError("Synthetic rollout must have exactly one active droplet slot.")
    if not np.allclose(state[1:], 0.0):
        raise AssertionError("Inactive droplet slots must remain zero-padded.")


def row_from_state(
    rollout_step: int,
    state: np.ndarray,
    feature_index: dict[str, int],
    scenario: dict[str, Any],
    context: PhysicsContext,
    nn_mean: float,
    nn_max: float,
    source: str,
) -> dict[str, Any]:
    from src.physics.full_device_cfd.domain import inside_full_device_domain

    x = float(state[feature_index["x"]])
    y = float(state[feature_index["y"]])
    point_device = context.geometry.convention.image_points_to_device(np.array([[x, y]], dtype=float))
    inside = bool(inside_full_device_domain(point_device, context.geometry)[0])
    occ_sum = float(sum(float(state[feature_index[name]]) for name in OCCUPANCY_FEATURES))
    row = {
        "scenario": scenario["scenario"],
        "requested_lead_frames": scenario["requested_lead_frames"],
        "actual_lead_frames": scenario["actual_lead_frames"],
        "template_track_id": scenario["track_id"],
        "template_source_frame": scenario["source_frame"],
        "rollout_step": int(rollout_step),
        "frame": int(rollout_step),
        "speed": float(np.hypot(state[feature_index["vx"]], state[feature_index["vy"]])),
        "streamline_velocity": float(np.hypot(state[feature_index["cfd_u"]], state[feature_index["cfd_v"]])),
        "region": region_at_pixel(x, y, context.region_labels),
        "inside_channel": inside,
        "nearest_neighbor_distance_mean": float(nn_mean),
        "nearest_neighbor_distance_max": float(nn_max),
        "occupancy_sum": occ_sum,
        "reconstruction_source": source,
    }
    for name in FEATURE_NAMES:
        value = state[feature_index[name]]
        row[name] = bool(value >= 0.5) if name == "cfd_valid" else float(value)
    return row


def normalize_state(state: np.ndarray, normalization: dict[str, Any], device, torch):
    mean = torch.as_tensor(normalization["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization["input_std"], dtype=torch.float32, device=device)
    return (torch.as_tensor(state, dtype=torch.float32, device=device) - mean.view(1, -1)) / std.view(1, -1)


def denormalize_target(target, normalization: dict[str, Any], device, torch):
    mean = torch.as_tensor(normalization["target_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization["target_std"], dtype=torch.float32, device=device)
    return target * std + mean


def region_at_pixel(x: float, y: float, region_labels: np.ndarray) -> str:
    col = int(round(x))
    row = int(round(y))
    if row < 0 or row >= region_labels.shape[0] or col < 0 or col >= region_labels.shape[1]:
        return "outside"
    return REGION_NAMES.get(int(region_labels[row, col]), "unknown")


def summarize_trajectory(table: pd.DataFrame) -> dict[str, Any]:
    xy = table[["x", "y"]].to_numpy(float)
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    regions = table["region"].astype(str).tolist()
    branch = "none"
    for region in regions:
        if region in {"left branch", "right branch"}:
            branch = region
            break
    inlet_entry = first_index(regions, {"inlet junction"})
    branch_entry = first_index(regions, {"left branch", "right branch"})
    outlet_junction_entry = first_index(regions, {"outlet junction"})
    outlet_channel_entry = first_index(regions, {"outlet channel"})
    baseline_slice = table if branch_entry is None else table.loc[table["rollout_step"] < branch_entry]
    prebranch_baseline = float(baseline_slice["left_flow_fraction"].median()) if len(baseline_slice) else float(table["left_flow_fraction"].iloc[0])
    left_branch = table.loc[table["region"] == "left branch", "left_flow_fraction"]
    right_branch = table.loc[table["region"] == "right branch", "left_flow_fraction"]
    outlet_junction_split = None if outlet_junction_entry is None else float(table.loc[table["rollout_step"] == outlet_junction_entry, "left_flow_fraction"].iloc[0])
    after_branch = table.loc[table["rollout_step"] > branch_entry] if branch_entry is not None else table.iloc[0:0]
    returns_toward = None
    if len(after_branch) and ((not left_branch.empty) or (not right_branch.empty)):
        final_departure = abs(float(table["left_flow_fraction"].iloc[-1]) - prebranch_baseline)
        max_departure = float(np.max(np.abs(table["left_flow_fraction"].to_numpy(float) - prebranch_baseline)))
        returns_toward = bool(final_departure < max_departure)
    defined_region = table["region"].isin(REGION_TO_OCCUPANCY)
    inside_normalized = (
        bool(np.allclose(table.loc[defined_region, "occupancy_sum"], 1.0, atol=1.0e-5))
        if bool(defined_region.any())
        else True
    )
    outside_zero = bool(np.allclose(table.loc[~defined_region, "occupancy_sum"], 0.0, atol=1.0e-8))
    return {
        "scenario": str(table["scenario"].iloc[0]),
        "template_track_id": int(table["template_track_id"].iloc[0]),
        "template_source_frame": int(table["template_source_frame"].iloc[0]),
        "requested_lead_frames": none_or_int(table["requested_lead_frames"].iloc[0]),
        "actual_lead_frames": none_or_int(table["actual_lead_frames"].iloc[0]),
        "branch_selected": branch,
        "first_frame_entering_inlet_junction": inlet_entry,
        "first_frame_entering_loop_branch": branch_entry,
        "first_frame_entering_outlet_junction": outlet_junction_entry,
        "first_frame_entering_outlet_channel": outlet_channel_entry,
        "total_traveled_distance_px": float(distances.sum()),
        "velocity_units": str(table["velocity_units"].iloc[0]) if "velocity_units" in table.columns else "px/frame",
        "minimum_speed": float(table["speed"].min()),
        "maximum_speed": float(table["speed"].max()),
        "mean_speed": float(table["speed"].mean()),
        "baseline_left_flow_fraction_before_branch_entry": prebranch_baseline,
        "minimum_left_flow_fraction": float(table["left_flow_fraction"].min()),
        "maximum_left_flow_fraction": float(table["left_flow_fraction"].max()),
        "maximum_absolute_departure_from_prebranch_baseline": float(
            np.max(np.abs(table["left_flow_fraction"].to_numpy(float) - prebranch_baseline))
        ),
        "left_flow_fraction_at_first_branch_entry": None
        if branch_entry is None
        else float(table.loc[table["rollout_step"] == branch_entry, "left_flow_fraction"].iloc[0]),
        "mean_left_flow_fraction_while_in_left_branch": None if left_branch.empty else float(left_branch.mean()),
        "mean_left_flow_fraction_while_in_right_branch": None if right_branch.empty else float(right_branch.mean()),
        "left_flow_fraction_at_outlet_junction_entry": outlet_junction_split,
        "final_left_flow_fraction": float(table["left_flow_fraction"].iloc[-1]),
        "split_returns_toward_empty_device_baseline_after_branch_exit": returns_toward,
        "cfd_valid_ever_false": bool((~table["cfd_valid"].astype(bool)).any()),
        "trajectory_ever_leaves_channel": bool((~table["inside_channel"].astype(bool)).any()),
        "occupancy_fractions_remain_normalized": bool(inside_normalized and outside_zero),
    }


def first_index(values: list[str], targets: set[str]) -> int | None:
    for idx, value in enumerate(values):
        if value in targets:
            return idx
    return None


def none_or_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def cross_initialization_summary(trajectories: list[pd.DataFrame], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    branches = [summary["branch_selected"] for summary in summaries]
    same_branch = len(set(branches)) <= 1
    aligned = []
    for table, summary in zip(trajectories, summaries):
        anchor = summary["first_frame_entering_inlet_junction"]
        if anchor is None:
            anchor = 0
        coords = table.loc[table["rollout_step"] >= anchor, ["x", "y"]].to_numpy(float)
        aligned.append(coords)
    min_len = min((len(item) for item in aligned), default=0)
    max_dev = np.nan
    if min_len > 0 and len(aligned) > 1:
        stack = np.stack([item[:min_len] for item in aligned], axis=0)
        center = np.mean(stack, axis=0, keepdims=True)
        max_dev = float(np.max(np.linalg.norm(stack - center, axis=2)))
    finals = [table[["x", "y"]].iloc[-1].to_numpy(float) for table in trajectories]
    final_diffs = []
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            final_diffs.append(float(np.linalg.norm(finals[i] - finals[j])))
    return {
        "all_runs_select_same_branch": bool(same_branch),
        "branches": branches,
        "trajectory_alignment_anchor": "first inlet-junction entry, falling back to rollout step 0",
        "maximum_spatial_deviation_px": max_dev,
        "maximum_final_position_difference_px": float(max(final_diffs)) if final_diffs else 0.0,
        "velocity_units": str(trajectories[0]["velocity_units"].iloc[0]) if trajectories and "velocity_units" in trajectories[0].columns else "px/frame",
        "speed_profile_final_values": [float(table["speed"].iloc[-1]) for table in trajectories],
        "left_flow_fraction_ranges": [
            [float(table["left_flow_fraction"].min()), float(table["left_flow_fraction"].max())]
            for table in trajectories
        ],
    }


def build_metadata(
    args: argparse.Namespace,
    scenario: dict[str, Any],
    summary: dict[str, Any],
    context: PhysicsContext,
    checkpoint_info: dict[str, Any],
    region_info: dict[str, Any],
    validation_info: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    scenario_fallbacks = [
        event for event in context.fallback_events if event.get("scenario") == scenario["scenario"]
    ]
    return {
        "checkpoint": checkpoint_info,
        "source_validation_sample": scenario["scenario"],
        "target_track_id": int(scenario["track_id"]),
        "initial_frame": int(scenario["source_frame"]),
        "nearest_neighbor_count": NEIGHBOR_COUNT,
        "weighting_method": "inverse Euclidean distance in x-y position",
        "hydraulic_model_config_used": {
            "model": "src.physics.hydraulics.compute_frame_baseline_hydraulics",
            **context.hydraulic_constants,
        },
        "cfd_library_config_used": {
            "library_path": str(args.cfd_library),
            "coordinate": "achieved left-flow fraction",
            "available_split_range": [context.cfd_min_split, context.cfd_max_split],
        },
        "rollout_length": int(args.rollout_length),
        "fallback_behavior": {
            "cfd_split_clamping_events": scenario_fallbacks,
            "invalid_cfd_samples_are_encoded_as_zero_velocity_with_cfd_valid_false": True,
        },
        "region_normalization": region_info,
        "validation_split": serializable_validation_info(validation_info),
        "cfd_feature_contract_validation": contract,
        "diagnostics": summary,
    }


def save_velocity_profile(table: pd.DataFrame, path: Path) -> None:
    velocity_units = str(table["velocity_units"].iloc[0]) if "velocity_units" in table.columns else "px/frame"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].plot(table["rollout_step"], table["vx"])
    axes[1].plot(table["rollout_step"], table["vy"])
    axes[0].set_ylabel(f"vx ({velocity_units})")
    axes[1].set_ylabel(f"vy ({velocity_units})")
    axes[1].set_xlabel("rollout step")
    axes[0].set_title("Predicted velocity components")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_speed_plot(table: pd.DataFrame, path: Path) -> None:
    velocity_units = str(table["velocity_units"].iloc[0]) if "velocity_units" in table.columns else "px/frame"
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(table["rollout_step"], table["speed"], label="predicted droplet")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(f"speed ({velocity_units})")
    ax.set_title("Single-droplet speed over rollout")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_left_flow_fraction_plot(table: pd.DataFrame, path: Path) -> None:
    summary = summarize_trajectory(table)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(table["rollout_step"], table["left_flow_fraction"], color="#1d4ed8")
    markers = [
        ("inlet junction", summary["first_frame_entering_inlet_junction"], "#f97316"),
        ("branch", summary["first_frame_entering_loop_branch"], "#16a34a"),
        ("outlet junction", summary["first_frame_entering_outlet_junction"], "#9333ea"),
        ("outlet channel", summary["first_frame_entering_outlet_channel"], "#dc2626"),
    ]
    for label, step, color in markers:
        if step is not None:
            ax.axvline(step, color=color, linestyle="--", linewidth=1.2, label=label)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("left_flow_fraction")
    ax.set_title("One-droplet hydraulic response")
    ax.grid(True, alpha=0.3)
    if any(step is not None for _, step, _ in markers):
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overlay_video(table: pd.DataFrame, channel_mask: np.ndarray, path: Path) -> None:
    max_step = int(table["rollout_step"].max())
    cmap = plt.get_cmap("viridis")
    rendered = []
    for step in range(max_step + 1):
        group = table[table["rollout_step"] <= step]
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.imshow(channel_mask, cmap="gray", alpha=0.28, origin="upper")
        colors = cmap(group["rollout_step"].to_numpy(float) / max(max_step, 1))
        ax.scatter(group["x"], group["y"], c=colors, s=14, linewidths=0)
        latest = group.iloc[-1]
        ax.scatter([latest["x"]], [latest["y"]], s=55, edgecolor="black", facecolor=colors[-1])
        ax.text(float(latest["x"]) + 5.0, float(latest["y"]) - 5.0, str(step), fontsize=8)
        ax.set_title(f"{table['scenario'].iloc[0]} single-droplet rollout, step {step}")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_xlim(80, 560)
        ax.set_ylim(channel_mask.shape[0], 0)
        fig.canvas.draw()
        rendered.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    write_mp4(path, rendered, fps=12)


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("No frames were rendered for the overlay video.")
    try:
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter could not open the output file")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    except ModuleNotFoundError:
        pass
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps)


def print_summary(summaries: list[dict[str, Any]], cross: dict[str, Any], output_dir: Path) -> None:
    print("Single-droplet physical sanity test complete")
    print(f"  output: {output_dir}")
    for item in summaries:
        print(
            f"  {item['scenario']}: branch={item['branch_selected']} "
            f"distance={item['total_traveled_distance_px']:.2f}px "
            f"speed={item['minimum_speed']:.3f}..{item['maximum_speed']:.3f} {item['velocity_units']} "
            f"fL={item['minimum_left_flow_fraction']:.6f}..{item['maximum_left_flow_fraction']:.6f} "
            f"cfd_invalid={item['cfd_valid_ever_false']} leaves_channel={item['trajectory_ever_leaves_channel']}"
        )
    print(f"  all runs same branch: {cross['all_runs_select_same_branch']}")
    print(f"  max aligned deviation: {cross['maximum_spatial_deviation_px']}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
