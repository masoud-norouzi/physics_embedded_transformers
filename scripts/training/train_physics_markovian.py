from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by environments without torch
    raise ModuleNotFoundError(
        "PyTorch is required to train the physics Markovian model. Install a CPU or CUDA PyTorch "
        "build appropriate for this machine, then rerun this script."
    ) from exc

from src.datasets.canonical_window_dataset import create_train_val_test_datasets
from src.models.canonical_rollout_transformer import CanonicalRolloutTransformer


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

DIAGNOSTIC_STEPS = (1, 5, 10, 20, 30, 40, 50)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.smoke_test:
        apply_smoke_test_overrides(config)

    set_random_seed(int(config["training"]["random_seed"]))
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device_info = select_device(config.get("device", {}).get("mode", "auto"))
    device = device_info["device"]
    print(f"Selected device: {device}")
    if device_info.get("gpu_name"):
        print(f"GPU: {device_info['gpu_name']}")

    save_json(output_dir / "resolved_config.json", config)
    save_json(output_dir / "device_info.json", {k: str(v) for k, v in device_info.items() if k != "device"})

    train_ds, val_ds, test_ds, normalization_stats = create_train_val_test_datasets(
        npz_path=config["dataset"]["path"],
        stride=int(config["dataset"]["stride"]),
        T_history=int(config["model"]["T_history"]),
        T_future=int(config["model"]["rollout_horizon"]),
        max_droplets=int(config["model"]["max_droplets"]),
        target_features=tuple(config["model"]["target_features"]),
    )
    validate_feature_contract(train_ds, config)
    if args.smoke_test:
        train_ds = SubsetByIndex(train_ds, int(config["smoke_test"]["train_windows"]))
        val_ds = SubsetByIndex(val_ds, int(config["smoke_test"]["val_windows"]))

    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")
    print(f"Test windows: {len(test_ds)}")
    print(f"Input dimension: {config['model']['input_dim']}")
    print(f"Prediction targets: {tuple(config['model']['target_features'])}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    model_config = dict(config["model"]["architecture"])
    model_config.update(
        input_dim=int(config["model"]["input_dim"]),
        target_dim=len(config["model"]["target_features"]),
        T_history=int(config["model"]["T_history"]),
        max_droplets=int(config["model"]["max_droplets"]),
    )
    model = CanonicalRolloutTransformer(**model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = rollout_weights(
        int(config["model"]["rollout_horizon"]),
        float(config["training"]["loss_alpha"]),
        device,
    )

    run_shape_test(model, train_loader, train_ds, normalization_stats, weights, device)

    start_time = time.perf_counter()
    if args.smoke_test:
        smoke_summary = run_smoke_test(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            dataset=train_ds,
            normalization_stats=normalization_stats,
            weights=weights,
            device=device,
            config=config,
            model_config=model_config,
            output_dir=output_dir,
        )
        smoke_summary["runtime_seconds"] = time.perf_counter() - start_time
        save_json(output_dir / "smoke_test_summary.json", smoke_summary)
        print(f"Smoke test runtime seconds: {smoke_summary['runtime_seconds']:.2f}")
        return

    train_full(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=train_ds,
        normalization_stats=normalization_stats,
        weights=weights,
        device=device,
        config=config,
        model_config=model_config,
        output_dir=output_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the physics-enabled Markovian rollout Transformer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Training config is empty or malformed: {config_path}")
    config["config_path"] = str(config_path)
    return config


def apply_smoke_test_overrides(config: dict[str, Any]) -> None:
    smoke = config["smoke_test"]
    config["training"]["epochs"] = int(smoke["epochs"])
    config["training"]["batch_size"] = int(smoke["batch_size"])
    config["model"]["rollout_horizon"] = int(smoke["rollout_horizon"])
    config["training"]["log_every_n_batches"] = 1
    config["training"]["output_dir"] = str(Path(config["training"]["output_dir"]) / "smoke_test")


def select_device(mode: str = "auto") -> dict[str, Any]:
    mode = str(mode).lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device mode: {mode}")
    if mode == "cpu":
        device = torch.device("cpu")
    elif mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "device": device,
        "mode": mode,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_feature_contract(dataset, config: dict[str, Any]) -> None:
    if dataset.feature_names != list(config["model"]["input_feature_names"]):
        raise ValueError("Dataset feature order does not match the physics Markovian config.")
    if len(dataset.feature_names) != int(config["model"]["input_dim"]):
        raise ValueError("Dataset feature count does not match configured input_dim.")


class SubsetByIndex:
    def __init__(self, base_dataset, count: int):
        self.base_dataset = base_dataset
        self.count = min(max(int(count), 0), len(base_dataset))
        self.start_frames = base_dataset.start_frames[: self.count]

    def __len__(self):
        return self.count

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    def __getitem__(self, index):
        if index >= self.count:
            raise IndexError(index)
        return self.base_dataset[index]


def rollout_weights(horizon: int, alpha: float, device) -> torch.Tensor:
    if horizon == 1:
        return torch.ones(1, dtype=torch.float32, device=device)
    step_ids = torch.arange(horizon, dtype=torch.float32, device=device)
    return 1.0 + float(alpha) * step_ids / float(horizon - 1)


def run_shape_test(model, train_loader, dataset, normalization_stats, weights, device) -> None:
    model.eval()
    batch = move_batch_to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        rollout = boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
        )
    print(f"history_x:       {tuple(batch['history_x'].shape)}")
    print(f"history_mask:    {tuple(batch['history_mask'].shape)}")
    print(f"future_y:        {tuple(batch['future_y'].shape)}")
    print(f"future_mask:     {tuple(batch['future_mask'].shape)}")
    print(f"cfd_loss_mask:   {tuple(batch['cfd_loss_mask'].shape)}")
    print(f"pred_velocity:   {tuple(rollout['pred_velocity'].shape)}")
    print(f"weighted_loss_internal_only: {float(rollout['weighted_loss_internal_only']):.6f}")
    assert rollout["pred_velocity"].shape == batch["future_y"].shape
    assert rollout["mask"].shape == batch["future_mask"].shape
    assert rollout["supervision_mask"].shape == batch["cfd_loss_mask"].shape


def run_smoke_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    dataset,
    normalization_stats,
    weights,
    device,
    config,
    model_config,
    output_dir: Path,
) -> dict[str, Any]:
    train_summary = train_one_epoch(
        model=model,
        loader=train_loader,
        dataset=dataset,
        optimizer=optimizer,
        normalization_stats=normalization_stats,
        weights=weights,
        device=device,
        grad_clip=float(config["training"]["grad_clip"]),
        log_every=int(config["training"]["log_every_n_batches"]),
        max_batches=int(config["smoke_test"]["optimization_steps"]),
    )
    val_summary = evaluate(
        model=model,
        loader=val_loader,
        dataset=dataset,
        normalization_stats=normalization_stats,
        weights=weights,
        device=device,
        log_every=0,
        max_batches=1,
    )
    checkpoint_path = output_dir / "latest_checkpoint.pt"
    checkpoint = build_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        val_summary=val_summary,
        normalization_stats=normalization_stats,
        config=config,
        model_config=model_config,
    )
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded_model = CanonicalRolloutTransformer(**loaded["model_config"]).to(device)
    reloaded_model.load_state_dict(loaded["model_state_dict"])
    reloaded_model.eval()
    batch = move_batch_to_device(next(iter(val_loader)), device)
    with torch.no_grad():
        rollout = boundary_conditioned_rollout(reloaded_model, batch, dataset, normalization_stats, weights)
    assert torch.isfinite(rollout["weighted_loss_internal_only"])
    return {
        "train_loss": train_summary["weighted_loss_internal_only"],
        "val_loss": val_summary["weighted_loss_internal_only"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload": "ok",
        "finite_losses": True,
        "rollout_shape": list(rollout["pred_velocity"].shape),
    }


def train_full(
    model,
    optimizer,
    train_loader,
    val_loader,
    dataset,
    normalization_stats,
    weights,
    device,
    config,
    model_config,
    output_dir: Path,
) -> None:
    curves_csv_path = output_dir / "training_curves.csv"
    initialize_curves_csv(curves_csv_path)
    best_val_loss = float("inf")

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            dataset=dataset,
            optimizer=optimizer,
            normalization_stats=normalization_stats,
            weights=weights,
            device=device,
            grad_clip=float(config["training"]["grad_clip"]),
            log_every=int(config["training"]["log_every_n_batches"]),
        )
        val_summary = evaluate(
            model=model,
            loader=val_loader,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            device=device,
            log_every=int(config["training"]["log_every_n_batches"]),
        )
        print_epoch_summary(epoch, train_summary, val_summary)
        append_curves_csv(curves_csv_path, epoch, train_summary, val_summary)

        checkpoint = build_checkpoint(model, optimizer, epoch, val_summary, normalization_stats, config, model_config)
        latest_path = output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        if val_summary["weighted_loss_internal_only"] < best_val_loss:
            best_val_loss = val_summary["weighted_loss_internal_only"]
            torch.save(checkpoint, output_dir / "best_checkpoint.pt")
            print(f"Saved best checkpoint: {output_dir / 'best_checkpoint.pt'}")


def train_one_epoch(
    model,
    loader,
    dataset,
    optimizer,
    normalization_stats,
    weights,
    device,
    grad_clip: float,
    log_every: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_supervised = 0
    total_present = 0
    num_batches = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        rollout = boundary_conditioned_rollout(model, batch, dataset, normalization_stats, weights)
        loss = rollout["weighted_loss_internal_only"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_supervised += int(rollout["supervision_mask"].sum().detach().cpu())
        total_present += int(rollout["mask"].sum().detach().cpu())
        num_batches += 1
        if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
            print_progress("train", num_batches, total_batches, total_loss / max(num_batches, 1))
        if max_batches is not None and num_batches >= max_batches:
            break
    return {
        "weighted_loss_internal_only": total_loss / max(num_batches, 1),
        "supervised_samples": float(total_supervised),
        "present_samples": float(total_present),
        "cfd_valid_target_fraction": total_supervised / max(total_present, 1),
    }


def evaluate(
    model,
    loader,
    dataset,
    normalization_stats,
    weights,
    device,
    log_every: int = 0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_supervised = 0
    total_present = 0
    num_batches = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    accumulators = create_accumulators(int(weights.numel()))

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            rollout = boundary_conditioned_rollout(model, batch, dataset, normalization_stats, weights)
            total_loss += float(rollout["weighted_loss_internal_only"].detach().cpu())
            total_supervised += int(rollout["supervision_mask"].sum().detach().cpu())
            total_present += int(rollout["mask"].sum().detach().cpu())
            update_metric_accumulators(accumulators, rollout)
            num_batches += 1
            if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
                print_progress("val", num_batches, total_batches, total_loss / max(num_batches, 1))
            if max_batches is not None and num_batches >= max_batches:
                break

    summary = metrics_from_accumulator(accumulators["overall"])
    summary["weighted_loss_internal_only"] = total_loss / max(num_batches, 1)
    summary["supervised_samples"] = float(total_supervised)
    summary["present_samples"] = float(total_present)
    summary["cfd_valid_target_fraction"] = total_supervised / max(total_present, 1)
    summary["step_rmse_position"] = [
        metrics_from_accumulator(accumulator)["rmse_position"]
        for accumulator in accumulators["steps"]
    ]
    return summary


def boundary_conditioned_rollout(model, batch, dataset, normalization_stats, weights):
    device = batch["history_x"].device
    rollout_history = batch["history_x"].clone()
    history_mask = batch["history_mask"].clone()

    pred_velocities_norm = []
    true_velocities_norm = []
    pred_velocities_phys = []
    true_velocities_phys = []
    pred_positions = []
    true_positions = []
    step_masks = []
    supervision_masks = []
    boundary_masks = []
    step_losses = []

    feature_index = dataset.feature_indices
    true_future_features = get_true_future_features(batch, dataset, device, weights.numel())
    true_future_xy = true_future_features[:, :, :, [feature_index["x"], feature_index["y"]]]

    for step_index in range(weights.numel()):
        previous_last_mask = history_mask[:, -1, :]
        pred_step_norm_raw = model(rollout_history, history_mask)
        pred_step_phys_raw = denormalize_targets(
            pred_step_norm_raw[:, None, :, :],
            normalization_stats,
            device,
        )[:, 0, :, :]

        true_step_norm = batch["future_y"][:, step_index, :, :]
        true_step_phys = denormalize_targets(
            true_step_norm[:, None, :, :],
            normalization_stats,
            device,
        )[:, 0, :, :]

        history_phys = denormalize_features(rollout_history, normalization_stats, device)
        last_frame = history_phys[:, -1, :, :]
        x_next = last_frame[:, :, feature_index["x"]] + pred_step_phys_raw[:, :, 0]
        y_next = last_frame[:, :, feature_index["y"]] + pred_step_phys_raw[:, :, 1]

        new_frame_phys = last_frame.clone()
        new_frame_phys[:, :, feature_index["x"]] = x_next
        new_frame_phys[:, :, feature_index["y"]] = y_next
        new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys_raw[:, :, 0]
        new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys_raw[:, :, 1]

        new_mask = batch["future_mask"][:, step_index, :]
        entering_mask = new_mask & ~previous_last_mask
        true_step_features = true_future_features[:, step_index, :, :]
        true_step_features_finite = torch.isfinite(true_step_features).all(dim=-1)
        boundary_mask = entering_mask & true_step_features_finite
        new_frame_phys[boundary_mask] = true_step_features[boundary_mask]

        refresh_observed_non_target_features(new_frame_phys, true_step_features, new_mask, feature_index)

        pred_step_norm = pred_step_norm_raw.clone()
        pred_step_phys = pred_step_phys_raw.clone()
        pred_step_norm[boundary_mask] = true_step_norm[boundary_mask]
        pred_step_phys[boundary_mask] = true_step_phys[boundary_mask]

        target_cfd_mask = batch.get("cfd_loss_mask", batch["future_mask"])[:, step_index, :]
        supervision_mask = target_cfd_mask & ~boundary_mask
        step_loss = masked_velocity_mse(pred_step_norm, true_step_norm, supervision_mask)
        step_losses.append(step_loss)

        new_frame_norm = normalize_features(new_frame_phys, normalization_stats, device)
        new_frame_norm = torch.where(new_mask[:, :, None], new_frame_norm, torch.zeros_like(new_frame_norm))
        rollout_history = torch.cat([rollout_history[:, 1:, :, :], new_frame_norm[:, None, :, :]], dim=1)
        history_mask = torch.cat([history_mask[:, 1:, :], new_mask[:, None, :]], dim=1)

        pred_velocities_norm.append(pred_step_norm)
        true_velocities_norm.append(true_step_norm)
        pred_velocities_phys.append(pred_step_phys)
        true_velocities_phys.append(true_step_phys)
        pred_positions.append(new_frame_phys[:, :, [feature_index["x"], feature_index["y"]]])
        true_positions.append(true_future_xy[:, step_index, :, :])
        step_masks.append(new_mask)
        supervision_masks.append(supervision_mask)
        boundary_masks.append(boundary_mask)

    step_loss_tensor = torch.stack(step_losses)
    weighted_loss_internal_only = (step_loss_tensor * weights).sum() / weights.sum()
    mask_tensor = torch.stack(step_masks, dim=1)
    supervision_mask_tensor = torch.stack(supervision_masks, dim=1)
    return {
        "weighted_loss": weighted_loss_internal_only,
        "weighted_loss_internal_only": weighted_loss_internal_only,
        "step_losses": step_loss_tensor,
        "pred_velocity_norm": torch.stack(pred_velocities_norm, dim=1),
        "true_velocity_norm": torch.stack(true_velocities_norm, dim=1),
        "pred_velocity": torch.stack(pred_velocities_phys, dim=1),
        "true_velocity": torch.stack(true_velocities_phys, dim=1),
        "pred_position": torch.stack(pred_positions, dim=1),
        "true_position": torch.stack(true_positions, dim=1),
        "mask": mask_tensor,
        "supervision_mask": supervision_mask_tensor,
        "boundary_mask": torch.stack(boundary_masks, dim=1),
        "internal_loss_mask": supervision_mask_tensor,
    }


def refresh_observed_non_target_features(new_frame_phys, true_step_features, new_mask, feature_index) -> None:
    predicted_names = {"x", "y", "vx", "vy"}
    for name, index in feature_index.items():
        if name in predicted_names:
            continue
        values = true_step_features[:, :, index]
        valid = new_mask & torch.isfinite(values)
        new_frame_phys[:, :, index] = torch.where(valid, values, new_frame_phys[:, :, index])


def masked_velocity_mse(prediction, target, mask):
    expanded_mask = mask.unsqueeze(-1).expand_as(target)
    squared_error = (prediction - target) ** 2
    valid_error = squared_error[expanded_mask]
    if valid_error.numel() == 0:
        return squared_error.sum() * 0.0
    return valid_error.mean()


def get_true_future_features(batch, dataset, device, horizon):
    droplet_ids = batch["droplet_ids"].detach().cpu().numpy()
    frame_starts = batch["frame_start"].detach().cpu().numpy()
    track_id_to_index = {int(track_id): index for index, track_id in enumerate(dataset.track_ids)}

    B, M = droplet_ids.shape
    true_features = np.full((B, horizon, M, len(dataset.feature_names)), np.nan, dtype=np.float32)
    for batch_index in range(B):
        start = int(frame_starts[batch_index]) + dataset.T_history
        end = start + horizon
        for slot_index in range(M):
            track_id = int(droplet_ids[batch_index, slot_index])
            if track_id < 0:
                continue
            droplet_index = track_id_to_index.get(track_id)
            if droplet_index is None:
                continue
            true_features[batch_index, :, slot_index, :] = dataset.Z[droplet_index, start:end, :]
    return torch.as_tensor(true_features, dtype=torch.float32, device=device)


def move_batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def denormalize_features(features, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["input_std"], dtype=torch.float32, device=device)
    return features * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def normalize_features(features, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["input_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["input_std"], dtype=torch.float32, device=device)
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def denormalize_targets(targets, normalization_stats, device):
    mean = torch.as_tensor(normalization_stats["target_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization_stats["target_std"], dtype=torch.float32, device=device)
    return targets * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)


def create_accumulators(num_steps):
    return {"overall": new_accumulator(), "steps": [new_accumulator() for _ in range(num_steps)]}


def new_accumulator():
    return {
        "count": 0,
        "sum_sq_vx": 0.0,
        "sum_sq_vy": 0.0,
        "sum_sq_speed": 0.0,
        "position_count": 0,
        "sum_sq_position": 0.0,
    }


def update_metric_accumulators(accumulators, rollout):
    velocity_error = rollout["pred_velocity"] - rollout["true_velocity"]
    speed_error = torch.sqrt(velocity_error[..., 0] ** 2 + velocity_error[..., 1] ** 2)
    position_error = rollout["pred_position"] - rollout["true_position"]
    position_error_norm = torch.sqrt(position_error[..., 0] ** 2 + position_error[..., 1] ** 2)
    position_finite = torch.isfinite(position_error).all(dim=-1)
    update_one_accumulator(
        accumulators["overall"],
        velocity_error,
        speed_error,
        rollout["supervision_mask"],
        position_error_norm,
        rollout["supervision_mask"] & position_finite,
    )
    for step_index in range(rollout["supervision_mask"].shape[1]):
        update_one_accumulator(
            accumulators["steps"][step_index],
            velocity_error[:, step_index, :, :],
            speed_error[:, step_index, :],
            rollout["supervision_mask"][:, step_index, :],
            position_error_norm[:, step_index, :],
            rollout["supervision_mask"][:, step_index, :] & position_finite[:, step_index, :],
        )


def update_one_accumulator(accumulator, velocity_error, speed_error, velocity_mask, position_error_norm, position_mask):
    valid = velocity_mask.bool()
    if valid.sum().item() > 0:
        vx_error = velocity_error[..., 0][valid]
        vy_error = velocity_error[..., 1][valid]
        speed = speed_error[valid]
        accumulator["count"] += int(valid.sum().item())
        accumulator["sum_sq_vx"] += float((vx_error**2).sum().detach().cpu())
        accumulator["sum_sq_vy"] += float((vy_error**2).sum().detach().cpu())
        accumulator["sum_sq_speed"] += float((speed**2).sum().detach().cpu())
    valid_position = position_mask.bool()
    if valid_position.sum().item() > 0:
        position = position_error_norm[valid_position]
        accumulator["position_count"] += int(valid_position.sum().item())
        accumulator["sum_sq_position"] += float((position**2).sum().detach().cpu())


def metrics_from_accumulator(accumulator):
    count = accumulator["count"]
    position_count = accumulator["position_count"]
    return {
        "valid_samples": count,
        "valid_position_samples": position_count,
        "rmse_vx": safe_rmse(accumulator["sum_sq_vx"], count),
        "rmse_vy": safe_rmse(accumulator["sum_sq_vy"], count),
        "rmse_speed": safe_rmse(accumulator["sum_sq_speed"], count),
        "rmse_position": safe_rmse(accumulator["sum_sq_position"], position_count),
    }


def safe_rmse(sum_sq, count):
    return np.sqrt(sum_sq / count) if count else np.nan


def print_progress(label, num_batches, total_batches, running_loss):
    percent = 100.0 * num_batches / max(total_batches, 1)
    print(f"  {label:<5} batch {num_batches:04d}/{total_batches:04d} ({percent:5.1f}%) weighted_loss={running_loss:.6f}")


def print_epoch_summary(epoch, train_summary, val_summary):
    available_steps = min(len(val_summary["step_rmse_position"]), max(DIAGNOSTIC_STEPS))
    step_text = " ".join(
        f"s{step}={val_summary['step_rmse_position'][step - 1]:.6f}"
        for step in DIAGNOSTIC_STEPS
        if step <= available_steps
    )
    print(
        f"epoch {epoch:03d} "
        f"train_weighted_loss_internal_only={train_summary['weighted_loss_internal_only']:.6f} "
        f"val_weighted_loss_internal_only={val_summary['weighted_loss_internal_only']:.6f} "
        f"val_rmse_vx={val_summary['rmse_vx']:.6f} "
        f"val_rmse_vy={val_summary['rmse_vy']:.6f} "
        f"val_rmse_speed={val_summary['rmse_speed']:.6f} "
        f"val_rmse_position={val_summary['rmse_position']:.6f}"
    )
    print(f"  stepwise_val_rmse_position {step_text}")


def initialize_curves_csv(path):
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_weighted_loss_internal_only",
                "val_weighted_loss_internal_only",
                "train_cfd_valid_target_fraction",
                "val_cfd_valid_target_fraction",
                "val_rmse_vx",
                "val_rmse_vy",
                "val_rmse_speed",
                "val_rmse_position",
            ]
        )


def append_curves_csv(path, epoch, train_summary, val_summary):
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                train_summary["weighted_loss_internal_only"],
                val_summary["weighted_loss_internal_only"],
                train_summary["cfd_valid_target_fraction"],
                val_summary["cfd_valid_target_fraction"],
                val_summary["rmse_vx"],
                val_summary["rmse_vy"],
                val_summary["rmse_speed"],
                val_summary["rmse_position"],
            ]
        )


def build_checkpoint(model, optimizer, epoch, val_summary, normalization_stats, config, model_config):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_summary["weighted_loss_internal_only"],
        "normalization_stats": normalization_stats,
        "model_config": model_config,
        "input_feature_names": config["model"]["input_feature_names"],
        "target_features": config["model"]["target_features"],
        "rollout_horizon": config["model"]["rollout_horizon"],
        "loss_alpha": config["training"]["loss_alpha"],
        "stride": config["dataset"]["stride"],
        "random_seed": config["training"]["random_seed"],
        "git_commit": git_commit_hash(),
    }


def git_commit_hash() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
