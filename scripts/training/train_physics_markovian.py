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
from src.physics.runtime import load_physics_runtime_context, step as physics_runtime_step


FEATURE_NAMES = [
    "x",
    "y",
    "vx",
    "vy",
    "bbox_w",
    "bbox_h",
    "cfd_u_norm",
    "cfd_v_norm",
    "superficial_velocity",
    "left_flow_fraction",
    "occupancy_inlet_channel",
    "occupancy_inlet_junction",
    "occupancy_left_branch",
    "occupancy_right_branch",
    "occupancy_outlet_junction",
    "occupancy_outlet_channel",
]

DIAGNOSTIC_STEPS = (1, 5, 10, 20, 30, 40, 50)
RUNTIME_TARGET_FEATURES = ("vx", "vy", "bbox_w", "bbox_h")
CURVES_COLUMNS = [
    "epoch",
    "active_rollout_horizon",
    "train_weighted_loss_internal_only",
    "val_weighted_loss_internal_only",
    "train_cfd_valid_target_fraction",
    "val_cfd_valid_target_fraction",
    "val_rmse_vx",
    "val_rmse_vy",
    "val_rmse_speed",
    "val_rmse_bbox_w",
    "val_rmse_bbox_h",
    "val_rmse_position",
    "train_runtime_step_attempts",
    "train_runtime_step_fallbacks",
    "train_runtime_step_fallback_fraction",
    "val_runtime_step_attempts",
    "val_runtime_step_fallbacks",
    "val_runtime_step_fallback_fraction",
    *[f"val_rmse_position_s{step}" for step in DIAGNOSTIC_STEPS],
    "val_pure_weighted_loss_internal_only",
    "val_pure_rmse_vx",
    "val_pure_rmse_vy",
    "val_pure_rmse_speed",
    "val_pure_rmse_bbox_w",
    "val_pure_rmse_bbox_h",
    "val_pure_rmse_position",
    "val_pure_runtime_step_attempts",
    "val_pure_runtime_step_fallbacks",
    "val_pure_runtime_step_fallback_fraction",
    *[f"val_pure_rmse_position_s{step}" for step in DIAGNOSTIC_STEPS],
    *[f"adaptive_fusion_alpha_s{step}" for step in DIAGNOSTIC_STEPS],
    *[
        f"adaptive_fusion_alpha_{feature}_s{step}"
        for feature in RUNTIME_TARGET_FEATURES
        for step in DIAGNOSTIC_STEPS
    ],
    "adaptive_fusion_alpha_mean",
]


class AdaptiveTargetFusion:
    """EMA-driven measurement-weighted target fusion for recurrent rollout inputs."""

    def __init__(
        self,
        *,
        horizon: int,
        target_dim: int,
        enabled: bool,
        ema_beta: float,
        initial_prediction_variance: float,
        measurement_variance,
        min_alpha: float,
        max_alpha: float,
        mode: str,
        device,
    ) -> None:
        self.enabled = bool(enabled)
        self.horizon = int(horizon)
        self.target_dim = int(target_dim)
        self.ema_beta = float(ema_beta)
        self.measurement_variance = torch.as_tensor(measurement_variance, dtype=torch.float32, device=device)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.mode = str(mode)
        if self.mode not in {"global_ema", "causal_rollout"}:
            raise ValueError(f"Unsupported adaptive_target_fusion mode: {self.mode!r}")
        self.prediction_variance = torch.full(
            (int(horizon), int(target_dim)),
            float(initial_prediction_variance),
            dtype=torch.float32,
            device=device,
        )
        self.last_alpha = self.alpha_tensor().detach().cpu().numpy()

    def alpha_tensor(self) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(self.prediction_variance)
        denominator = self.prediction_variance + self.measurement_variance
        alpha = self.prediction_variance / torch.clamp_min(denominator, 1.0e-12)
        return torch.clamp(alpha, self.min_alpha, self.max_alpha)

    def alpha_from_variance(self, variance: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(variance)
        denominator = variance + self.measurement_variance
        alpha = variance / torch.clamp_min(denominator, 1.0e-12)
        return torch.clamp(alpha, self.min_alpha, self.max_alpha)

    def initial_rollout_variance(self, device) -> torch.Tensor:
        return self.prediction_variance[0].detach().clone().to(device)

    def update_rollout_variance(
        self,
        current_variance: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return current_variance
        mse, count = target_error_mse_for_step(prediction, target, mask)
        valid = count > 0
        updated = self.ema_beta * current_variance + (1.0 - self.ema_beta) * mse
        return torch.where(valid, updated, current_variance).detach()

    def update(self, mse_by_step_feature: torch.Tensor, count_by_step_feature: torch.Tensor) -> None:
        if not self.enabled:
            return
        valid = count_by_step_feature > 0
        if not bool(valid.any().item()):
            return
        mse = torch.where(valid, mse_by_step_feature, self.prediction_variance)
        self.prediction_variance = torch.where(
            valid,
            self.ema_beta * self.prediction_variance + (1.0 - self.ema_beta) * mse,
            self.prediction_variance,
        )
        self.last_alpha = self.alpha_tensor().detach().cpu().numpy()

    def summary(self) -> dict[str, Any]:
        alpha = self.last_alpha if self.enabled else np.zeros_like(self.last_alpha)
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "alpha_by_step_feature": alpha.tolist(),
            "alpha_by_step_mean": alpha.mean(axis=1).tolist(),
            "alpha_mean": float(alpha.mean()),
        }


class ZeroAdaptiveTargetFusion:
    enabled = True
    mode = "zero"

    def __init__(self, horizon: int, target_dim: int, device) -> None:
        self._alpha = torch.zeros((int(horizon), int(target_dim)), dtype=torch.float32, device=device)

    def alpha_tensor(self) -> torch.Tensor:
        return self._alpha


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
        experiment_config=config["dataset"].get("experiment_config", "configs/experiments/video_2.yml"),
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
    runtime_context = load_physics_runtime_context(
        experiment_config_path=config["dataset"].get("experiment_config", "configs/experiments/video_2.yml"),
        cfd_library_path=config["dataset"].get("cfd_library_path", "outputs/physics/full_device_cfd/library"),
        feature_names=tuple(config["model"]["input_feature_names"]),
    )

    initial_runtime_context = runtime_context_for_epoch(config, 1, runtime_context)
    print(f"shape_test physics_refresh={physics_refresh_mode(initial_runtime_context)}")
    run_shape_test(model, train_loader, train_ds, normalization_stats, weights, device, initial_runtime_context)

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
            runtime_context=runtime_context,
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
        runtime_context=runtime_context,
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
    config["training"]["rollout_horizon_schedule"] = [
        {"start_epoch": 1, "horizon": int(smoke["rollout_horizon"])}
    ]
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
    target_features = tuple(config["model"]["target_features"])
    if dataset.feature_names == FEATURE_NAMES and target_features != RUNTIME_TARGET_FEATURES:
        raise ValueError(
            "Closed-loop physics rollout requires target_features to be exactly "
            f"{RUNTIME_TARGET_FEATURES}, got {target_features}."
        )


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


def run_shape_test(model, train_loader, dataset, normalization_stats, weights, device, runtime_context=None) -> None:
    model.eval()
    batch = move_batch_to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        rollout = boundary_conditioned_rollout(
            model=model,
            batch=batch,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=weights,
            runtime_context=runtime_context,
        )
    print(f"history_x:       {tuple(batch['history_x'].shape)}")
    print(f"history_mask:    {tuple(batch['history_mask'].shape)}")
    print(f"future_y:        {tuple(batch['future_y'].shape)}")
    print(f"future_mask:     {tuple(batch['future_mask'].shape)}")
    print(f"cfd_loss_mask:   {tuple(batch['cfd_loss_mask'].shape)}")
    print(f"pred_target:     {tuple(rollout['pred_target'].shape)}")
    print(f"weighted_loss_internal_only: {float(rollout['weighted_loss_internal_only']):.6f}")
    assert rollout["pred_target"].shape == batch["future_y"].shape
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
    runtime_context,
    model_config,
    output_dir: Path,
) -> dict[str, Any]:
    adaptive_fusion = create_adaptive_target_fusion(config, weights, model_config, device, dataset, normalization_stats)
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
        runtime_context=runtime_context,
        adaptive_fusion=adaptive_fusion,
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
        runtime_context=runtime_context,
        adaptive_fusion=adaptive_fusion,
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
        rollout = boundary_conditioned_rollout(
            reloaded_model,
            batch,
            dataset,
            normalization_stats,
            weights,
            runtime_context=runtime_context,
            adaptive_fusion=adaptive_fusion,
        )
    assert torch.isfinite(rollout["weighted_loss_internal_only"])
    return {
        "train_loss": train_summary["weighted_loss_internal_only"],
        "val_loss": val_summary["weighted_loss_internal_only"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload": "ok",
        "finite_losses": True,
        "rollout_shape": list(rollout["pred_target"].shape),
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
    runtime_context,
    model_config,
    output_dir: Path,
) -> None:
    curves_csv_path = output_dir / "training_curves.csv"
    initialize_curves_csv(curves_csv_path)
    best_val_loss = float("inf")
    full_rollout_horizon = int(weights.numel())
    if rollout_horizon_schedule_enabled(config) and adaptive_target_fusion_enabled(config):
        raise ValueError("rollout_horizon_schedule is a no-fusion ablation; set adaptive_target_fusion.enabled=false")
    adaptive_fusion = create_adaptive_target_fusion(
        config,
        weights,
        model_config,
        device,
        dataset,
        normalization_stats,
    )
    alpha_history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        active_rollout_horizon = rollout_horizon_for_epoch(config, epoch, full_rollout_horizon)
        active_weights = rollout_weights(
            active_rollout_horizon,
            float(config["training"]["loss_alpha"]),
            device,
        )
        active_runtime_context = runtime_context_for_epoch(config, epoch, runtime_context)
        active_fusion = adaptive_fusion if active_runtime_context is not None and adaptive_fusion.enabled else None
        if adaptive_fusion is not None and adaptive_fusion.horizon != active_rollout_horizon:
            adaptive_fusion = create_adaptive_target_fusion(
                config,
                active_weights,
                model_config,
                device,
                dataset,
                normalization_stats,
            )
            active_fusion = adaptive_fusion if active_runtime_context is not None and adaptive_fusion.enabled else None
        print(
            f"epoch {epoch:03d} "
            f"physics_refresh={physics_refresh_mode(active_runtime_context)} "
            f"rollout_horizon={active_rollout_horizon}"
        )
        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            dataset=dataset,
            optimizer=optimizer,
            normalization_stats=normalization_stats,
            weights=active_weights,
            device=device,
            grad_clip=float(config["training"]["grad_clip"]),
            log_every=int(config["training"]["log_every_n_batches"]),
            runtime_context=active_runtime_context,
            adaptive_fusion=active_fusion,
        )
        train_summary["active_rollout_horizon"] = float(active_rollout_horizon)
        val_summary = evaluate(
            model=model,
            loader=val_loader,
            dataset=dataset,
            normalization_stats=normalization_stats,
            weights=active_weights,
            device=device,
            log_every=int(config["training"]["log_every_n_batches"]),
            runtime_context=active_runtime_context,
            adaptive_fusion=active_fusion,
        )
        val_summary["active_rollout_horizon"] = float(active_rollout_horizon)
        if active_fusion is None:
            val_pure_summary = dict(val_summary)
        else:
            pure_validation_fusion = ZeroAdaptiveTargetFusion(
                active_rollout_horizon,
                model_config["target_dim"],
                device,
            )
            val_pure_summary = evaluate(
                model=model,
                loader=val_loader,
                dataset=dataset,
                normalization_stats=normalization_stats,
                weights=active_weights,
                device=device,
                log_every=0,
                runtime_context=active_runtime_context,
                adaptive_fusion=pure_validation_fusion,
            )
            val_pure_summary["active_rollout_horizon"] = float(active_rollout_horizon)
        fusion_summary = train_summary.get("adaptive_fusion")
        if fusion_summary is None:
            fusion_summary = adaptive_fusion.summary() if active_fusion is not None else inactive_adaptive_fusion_summary(adaptive_fusion)
            train_summary["adaptive_fusion"] = fusion_summary
        val_summary["pure"] = val_pure_summary
        alpha_history.append({"epoch": epoch, **fusion_summary})
        print_epoch_summary(epoch, train_summary, val_summary)
        append_curves_csv(curves_csv_path, epoch, train_summary, val_summary)
        save_adaptive_fusion_alpha_plot(alpha_history, output_dir / "adaptive_fusion_alpha_by_epoch.png")

        checkpoint = build_checkpoint(model, optimizer, epoch, val_pure_summary, normalization_stats, config, model_config)
        latest_path = output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        if should_update_best_checkpoint(
            active_runtime_context,
            val_pure_summary,
            best_val_loss,
            active_rollout_horizon=active_rollout_horizon,
            full_rollout_horizon=full_rollout_horizon,
        ):
            best_val_loss = val_pure_summary["weighted_loss_internal_only"]
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
    runtime_context=None,
    adaptive_fusion: AdaptiveTargetFusion | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_supervised = 0
    total_present = 0
    total_runtime_attempts = 0
    total_runtime_fallbacks = 0
    alpha_sum = None
    alpha_count = 0
    num_batches = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        rollout = boundary_conditioned_rollout(
            model,
            batch,
            dataset,
            normalization_stats,
            weights,
            runtime_context=runtime_context,
            adaptive_fusion=adaptive_fusion,
        )
        loss = rollout["weighted_loss_internal_only"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_supervised += int(rollout["supervision_mask"].sum().detach().cpu())
        total_present += int(rollout["mask"].sum().detach().cpu())
        total_runtime_attempts += int(rollout["runtime_step_attempts"])
        total_runtime_fallbacks += int(rollout["runtime_step_fallbacks"])
        if adaptive_fusion is not None:
            adaptive_fusion.update(
                rollout["target_error_mse_by_step_feature"],
                rollout["target_error_count_by_step_feature"],
            )
        if "adaptive_alpha_used" in rollout:
            alpha_sum = accumulate_alpha_used(alpha_sum, rollout["adaptive_alpha_used"])
            alpha_count += 1
        num_batches += 1
        if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
            print_progress("train", num_batches, total_batches, total_loss / max(num_batches, 1))
        if max_batches is not None and num_batches >= max_batches:
            break
    summary = {
        "weighted_loss_internal_only": total_loss / max(num_batches, 1),
        "supervised_samples": float(total_supervised),
        "present_samples": float(total_present),
        "cfd_valid_target_fraction": total_supervised / max(total_present, 1),
        "runtime_step_attempts": float(total_runtime_attempts),
        "runtime_step_fallbacks": float(total_runtime_fallbacks),
        "runtime_step_fallback_fraction": total_runtime_fallbacks / max(total_runtime_attempts, 1),
    }
    if adaptive_fusion is not None and alpha_sum is not None:
        summary["adaptive_fusion"] = alpha_used_summary(alpha_sum, alpha_count, adaptive_fusion)
    return summary


def evaluate(
    model,
    loader,
    dataset,
    normalization_stats,
    weights,
    device,
    log_every: int = 0,
    max_batches: int | None = None,
    runtime_context=None,
    adaptive_fusion: AdaptiveTargetFusion | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_supervised = 0
    total_present = 0
    total_runtime_attempts = 0
    total_runtime_fallbacks = 0
    alpha_sum = None
    alpha_count = 0
    num_batches = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    accumulators = create_accumulators(int(weights.numel()))

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            rollout = boundary_conditioned_rollout(
                model,
                batch,
                dataset,
                normalization_stats,
                weights,
                runtime_context=runtime_context,
                adaptive_fusion=adaptive_fusion,
            )
            total_loss += float(rollout["weighted_loss_internal_only"].detach().cpu())
            total_supervised += int(rollout["supervision_mask"].sum().detach().cpu())
            total_present += int(rollout["mask"].sum().detach().cpu())
            total_runtime_attempts += int(rollout["runtime_step_attempts"])
            total_runtime_fallbacks += int(rollout["runtime_step_fallbacks"])
            if "adaptive_alpha_used" in rollout:
                alpha_sum = accumulate_alpha_used(alpha_sum, rollout["adaptive_alpha_used"])
                alpha_count += 1
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
    summary["runtime_step_attempts"] = float(total_runtime_attempts)
    summary["runtime_step_fallbacks"] = float(total_runtime_fallbacks)
    summary["runtime_step_fallback_fraction"] = total_runtime_fallbacks / max(total_runtime_attempts, 1)
    summary["step_rmse_position"] = [
        metrics_from_accumulator(accumulator)["rmse_position"]
        for accumulator in accumulators["steps"]
    ]
    if adaptive_fusion is not None and alpha_sum is not None:
        summary["adaptive_fusion"] = alpha_used_summary(alpha_sum, alpha_count, adaptive_fusion)
    return summary


def create_adaptive_target_fusion(config: dict[str, Any], weights, model_config: dict[str, Any], device, dataset=None, normalization_stats=None):
    fusion = config.get("training", {}).get("adaptive_target_fusion", {})
    measurement_variance = adaptive_measurement_variance(fusion, dataset, normalization_stats, device)
    return AdaptiveTargetFusion(
        horizon=int(weights.numel()),
        target_dim=int(model_config["target_dim"]),
        enabled=bool(fusion.get("enabled", False)),
        ema_beta=float(fusion.get("ema_beta", 0.95)),
        initial_prediction_variance=float(fusion.get("initial_prediction_variance", 1.0)),
        measurement_variance=measurement_variance,
        min_alpha=float(fusion.get("min_alpha", 0.0)),
        max_alpha=float(fusion.get("max_alpha", 0.8)),
        mode=str(fusion.get("mode", "causal_rollout")),
        device=device,
    )


def adaptive_measurement_variance(fusion_config: dict[str, Any], dataset, normalization_stats, device):
    if "detection_position_variance_px2" not in fusion_config:
        return float(fusion_config.get("measurement_variance", 4.0))
    if dataset is None or normalization_stats is None:
        return float(fusion_config.get("measurement_variance", 4.0))

    detection_variance_px2 = float(fusion_config["detection_position_variance_px2"])
    if detection_variance_px2 < 0.0 or not np.isfinite(detection_variance_px2):
        raise ValueError(f"detection_position_variance_px2 must be finite and non-negative, got {detection_variance_px2}")
    target_std = torch.as_tensor(normalization_stats["target_std"], dtype=torch.float32, device=device)
    if target_std.numel() != len(RUNTIME_TARGET_FEATURES):
        raise ValueError(f"Expected {len(RUNTIME_TARGET_FEATURES)} target std values, got {target_std.numel()}")
    velocity_std_px_frame = float(np.sqrt(2.0 * detection_variance_px2))
    velocity_std = velocity_std_px_frame * float(getattr(dataset, "velocity_mm_s_per_px_frame", 1.0))
    bbox_std = float(np.sqrt(2.0 * detection_variance_px2))
    measurement_std = torch.as_tensor(
        [velocity_std, velocity_std, bbox_std, bbox_std],
        dtype=torch.float32,
        device=device,
    )
    return (measurement_std / torch.clamp_min(target_std, 1.0e-12)) ** 2


def inactive_adaptive_fusion_summary(adaptive_fusion: AdaptiveTargetFusion) -> dict[str, Any]:
    alpha = np.zeros_like(adaptive_fusion.last_alpha)
    return {
        "enabled": False,
        "mode": "inactive",
        "alpha_by_step_feature": alpha.tolist(),
        "alpha_by_step_mean": alpha.mean(axis=1).tolist(),
        "alpha_mean": 0.0,
    }


def accumulate_alpha_used(alpha_sum, alpha_used: torch.Tensor):
    value = alpha_used.detach()
    return value.clone() if alpha_sum is None else alpha_sum + value


def alpha_used_summary(alpha_sum: torch.Tensor, alpha_count: int, adaptive_fusion) -> dict[str, Any]:
    alpha = (alpha_sum / max(int(alpha_count), 1)).detach().cpu().numpy()
    return {
        "enabled": bool(getattr(adaptive_fusion, "enabled", False)),
        "mode": str(getattr(adaptive_fusion, "mode", "unknown")),
        "alpha_by_step_feature": alpha.tolist(),
        "alpha_by_step_mean": alpha.mean(axis=1).tolist(),
        "alpha_mean": float(alpha.mean()),
    }


def boundary_conditioned_rollout(
    model,
    batch,
    dataset,
    normalization_stats,
    weights,
    runtime_context=None,
    adaptive_fusion: AdaptiveTargetFusion | None = None,
):
    device = batch["history_x"].device
    rollout_history = batch["history_x"].clone()
    history_mask = batch["history_mask"].clone()

    pred_targets_norm = []
    true_targets_norm = []
    pred_targets_phys = []
    true_targets_phys = []
    pred_positions = []
    true_positions = []
    pred_states = []
    step_masks = []
    supervision_masks = []
    boundary_masks = []
    step_losses = []
    runtime_step_attempts = 0
    runtime_step_fallbacks = 0
    target_dim = int(len(normalization_stats["target_mean"]))
    target_error_sse_by_step_feature = torch.zeros((int(weights.numel()), target_dim), device=device)
    target_error_count_by_step_feature = torch.zeros_like(target_error_sse_by_step_feature)
    adaptive_alpha_used = torch.zeros_like(target_error_sse_by_step_feature)
    rollout_variance = (
        adaptive_fusion.initial_rollout_variance(device)
        if adaptive_fusion is not None and adaptive_fusion.enabled and adaptive_fusion.mode == "causal_rollout"
        else None
    )

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
        new_mask = batch["future_mask"][:, step_index, :]
        continuing_mask = new_mask & previous_last_mask
        entering_mask = new_mask & ~previous_last_mask
        true_step_features = true_future_features[:, step_index, :, :]
        true_step_features_finite = torch.isfinite(true_step_features).all(dim=-1)
        boundary_mask = entering_mask & true_step_features_finite
        alpha_for_step = alpha_for_rollout_step(adaptive_fusion, step_index, pred_step_phys_raw.device, rollout_variance)
        if alpha_for_step is not None:
            adaptive_alpha_used[step_index] = alpha_for_step
        target_for_rollout_phys = fuse_rollout_targets(
            pred_step_phys_raw,
            true_step_phys,
            continuing_mask,
            alpha_for_step,
        )

        if runtime_context is None:
            new_frame_phys = build_stale_refresh_frame(
                last_frame,
                target_for_rollout_phys,
                true_step_features,
                new_mask,
                boundary_mask,
                feature_index,
                dataset,
                device,
            )
        else:
            new_frame_phys = torch.zeros_like(last_frame)
            refreshed_phys, runtime_success = runtime_step_batch(
                last_frame,
                target_for_rollout_phys,
                continuing_mask,
                runtime_context,
            )
            runtime_attempt_rows = continuing_mask.any(dim=1)
            runtime_fallback_rows = runtime_attempt_rows & ~runtime_success
            runtime_step_attempts += int(runtime_attempt_rows.sum().detach().cpu())
            runtime_step_fallbacks += int(runtime_fallback_rows.sum().detach().cpu())

            runtime_mask = continuing_mask & runtime_success[:, None]
            new_frame_phys = torch.where(runtime_mask[:, :, None], refreshed_phys, new_frame_phys)
            if runtime_fallback_rows.any():
                stale_frame_phys = build_stale_refresh_frame(
                    last_frame,
                    target_for_rollout_phys,
                    true_step_features,
                    new_mask,
                    boundary_mask,
                    feature_index,
                    dataset,
                    device,
                    refresh_observed_non_target=False,
                )
                fallback_mask = new_mask & runtime_fallback_rows[:, None]
                new_frame_phys = torch.where(fallback_mask[:, :, None], stale_frame_phys, new_frame_phys)
            new_frame_phys[boundary_mask] = true_step_features[boundary_mask]

        pred_step_norm = pred_step_norm_raw.clone()
        pred_step_phys = pred_step_phys_raw.clone()
        pred_step_norm[boundary_mask] = true_step_norm[boundary_mask]
        pred_step_phys[boundary_mask] = true_step_phys[boundary_mask]

        target_cfd_mask = batch.get("cfd_loss_mask", batch["future_mask"])[:, step_index, :]
        supervision_mask = target_cfd_mask & ~boundary_mask
        step_loss = masked_velocity_mse(pred_step_norm, true_step_norm, supervision_mask)
        step_losses.append(step_loss)
        update_target_error_stats(
            target_error_sse_by_step_feature,
            target_error_count_by_step_feature,
            step_index,
            pred_step_norm_raw,
            true_step_norm,
            supervision_mask,
        )
        if rollout_variance is not None:
            rollout_variance = adaptive_fusion.update_rollout_variance(
                rollout_variance,
                pred_step_norm_raw,
                true_step_norm,
                supervision_mask,
            )

        new_frame_norm = normalize_features(new_frame_phys, normalization_stats, device)
        new_frame_norm = torch.where(new_mask[:, :, None], new_frame_norm, torch.zeros_like(new_frame_norm))
        rollout_history = torch.cat([rollout_history[:, 1:, :, :], new_frame_norm[:, None, :, :]], dim=1)
        history_mask = torch.cat([history_mask[:, 1:, :], new_mask[:, None, :]], dim=1)

        pred_targets_norm.append(pred_step_norm)
        true_targets_norm.append(true_step_norm)
        pred_targets_phys.append(pred_step_phys)
        true_targets_phys.append(true_step_phys)
        pred_states.append(new_frame_phys)
        pred_positions.append(new_frame_phys[:, :, [feature_index["x"], feature_index["y"]]])
        true_positions.append(true_future_xy[:, step_index, :, :])
        step_masks.append(new_mask)
        supervision_masks.append(supervision_mask)
        boundary_masks.append(boundary_mask)

    step_loss_tensor = torch.stack(step_losses)
    weighted_loss_internal_only = (step_loss_tensor * weights).sum() / weights.sum()
    mask_tensor = torch.stack(step_masks, dim=1)
    supervision_mask_tensor = torch.stack(supervision_masks, dim=1)
    pred_target_norm = torch.stack(pred_targets_norm, dim=1)
    true_target_norm = torch.stack(true_targets_norm, dim=1)
    pred_target = torch.stack(pred_targets_phys, dim=1)
    true_target = torch.stack(true_targets_phys, dim=1)
    return {
        "weighted_loss": weighted_loss_internal_only,
        "weighted_loss_internal_only": weighted_loss_internal_only,
        "step_losses": step_loss_tensor,
        "pred_target_norm": pred_target_norm,
        "true_target_norm": true_target_norm,
        "pred_target": pred_target,
        "true_target": true_target,
        "pred_velocity_norm": pred_target_norm[..., :2],
        "true_velocity_norm": true_target_norm[..., :2],
        "pred_velocity": pred_target[..., :2],
        "true_velocity": true_target[..., :2],
        "pred_state": torch.stack(pred_states, dim=1),
        "pred_position": torch.stack(pred_positions, dim=1),
        "true_position": torch.stack(true_positions, dim=1),
        "mask": mask_tensor,
        "supervision_mask": supervision_mask_tensor,
        "boundary_mask": torch.stack(boundary_masks, dim=1),
        "internal_loss_mask": supervision_mask_tensor,
        "runtime_step_attempts": runtime_step_attempts,
        "runtime_step_fallbacks": runtime_step_fallbacks,
        "target_error_mse_by_step_feature": safe_divide_tensor(
            target_error_sse_by_step_feature,
            target_error_count_by_step_feature,
        ).detach(),
        "target_error_count_by_step_feature": target_error_count_by_step_feature.detach(),
        "adaptive_alpha_used": adaptive_alpha_used.detach(),
    }


def alpha_for_rollout_step(adaptive_fusion, step_index: int, device, rollout_variance=None):
    if adaptive_fusion is None or not adaptive_fusion.enabled:
        return None
    if getattr(adaptive_fusion, "mode", "global_ema") == "causal_rollout" and rollout_variance is not None:
        return adaptive_fusion.alpha_from_variance(rollout_variance).to(device)
    return adaptive_fusion.alpha_tensor()[int(step_index)].to(device)


def fuse_rollout_targets(pred_step_phys_raw, true_step_phys, continuing_mask, alpha):
    if alpha is None:
        return pred_step_phys_raw
    true_finite = torch.isfinite(true_step_phys).all(dim=-1)
    fusion_mask = continuing_mask & true_finite
    fused = (1.0 - alpha.view(1, 1, -1)) * pred_step_phys_raw + alpha.view(1, 1, -1) * true_step_phys
    return torch.where(fusion_mask[:, :, None], fused, pred_step_phys_raw)


def update_target_error_stats(
    sse_by_step_feature,
    count_by_step_feature,
    step_index: int,
    prediction,
    target,
    mask,
) -> None:
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    expanded_mask = mask[:, :, None] & finite
    squared = (prediction - target) ** 2
    sse_by_step_feature[step_index] += torch.where(expanded_mask, squared, torch.zeros_like(squared)).sum(dim=(0, 1))
    count_by_step_feature[step_index] += expanded_mask.sum(dim=(0, 1)).to(sse_by_step_feature.dtype)


def target_error_mse_for_step(prediction, target, mask):
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    expanded_mask = mask[:, :, None] & finite
    squared = (prediction - target) ** 2
    sse = torch.where(expanded_mask, squared, torch.zeros_like(squared)).sum(dim=(0, 1))
    count = expanded_mask.sum(dim=(0, 1)).to(sse.dtype)
    mse = torch.where(count > 0, sse / torch.clamp_min(count, 1.0), torch.zeros_like(sse))
    return mse, count


def safe_divide_tensor(numerator, denominator):
    return torch.where(denominator > 0, numerator / torch.clamp_min(denominator, 1.0), torch.zeros_like(numerator))


def build_stale_refresh_frame(
    last_frame,
    pred_step_phys_raw,
    true_step_features,
    new_mask,
    boundary_mask,
    feature_index,
    dataset,
    device,
    refresh_observed_non_target: bool = True,
):
    velocity_to_px_frame = velocity_mm_s_to_px_frame_scale(dataset, device)
    x_next = last_frame[:, :, feature_index["x"]] + pred_step_phys_raw[:, :, 0] * velocity_to_px_frame
    y_next = last_frame[:, :, feature_index["y"]] + pred_step_phys_raw[:, :, 1] * velocity_to_px_frame

    new_frame_phys = last_frame.clone()
    new_frame_phys[:, :, feature_index["x"]] = x_next
    new_frame_phys[:, :, feature_index["y"]] = y_next
    new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys_raw[:, :, 0]
    new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys_raw[:, :, 1]
    if pred_step_phys_raw.shape[-1] >= 4 and "bbox_w" in feature_index and "bbox_h" in feature_index:
        new_frame_phys[:, :, feature_index["bbox_w"]] = pred_step_phys_raw[:, :, 2]
        new_frame_phys[:, :, feature_index["bbox_h"]] = pred_step_phys_raw[:, :, 3]
    new_frame_phys[boundary_mask] = true_step_features[boundary_mask]
    if refresh_observed_non_target:
        refresh_observed_non_target_features(new_frame_phys, true_step_features, new_mask, feature_index)
    return new_frame_phys


def refresh_observed_non_target_features(new_frame_phys, true_step_features, new_mask, feature_index) -> None:
    predicted_names = {"x", "y", "vx", "vy", "bbox_w", "bbox_h"}
    for name, index in feature_index.items():
        if name in predicted_names:
            continue
        values = true_step_features[:, :, index]
        valid = new_mask & torch.isfinite(values)
        new_frame_phys[:, :, index] = torch.where(valid, values, new_frame_phys[:, :, index])


def runtime_step_batch(current_state_phys, model_prediction_phys, active_mask, runtime_context, profile=None):
    device = current_state_phys.device
    dtype = current_state_phys.dtype
    total_start = time.perf_counter() if profile is not None else None

    section_start = time.perf_counter() if profile is not None else None
    current_np = current_state_phys.detach().cpu().numpy().astype(np.float32, copy=True)
    if profile is not None:
        add_profile_time(profile, "batch_current_to_cpu_numpy_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    prediction_np = model_prediction_phys.detach().cpu().numpy().astype(np.float32, copy=True)
    if profile is not None:
        add_profile_time(profile, "batch_prediction_to_cpu_numpy_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    active_np = active_mask.detach().cpu().numpy().astype(bool, copy=True)
    if profile is not None:
        add_profile_time(profile, "batch_active_mask_to_cpu_numpy_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    next_np = np.zeros_like(current_np, dtype=np.float32)
    success_np = np.ones(current_np.shape[0], dtype=bool)
    active_mask_cache: dict[int, np.ndarray] = {}
    if profile is not None:
        add_profile_time(profile, "batch_numpy_allocation_seconds", time.perf_counter() - section_start)

    loop_start = time.perf_counter() if profile is not None else None
    for batch_index in np.flatnonzero(active_np.any(axis=1)):
        active_slots = active_np[batch_index]
        active_count = int(np.count_nonzero(active_slots))
        runtime_active_mask = active_mask_cache.setdefault(active_count, np.ones(active_count, dtype=bool))
        try:
            step_kwargs = {"active_mask": runtime_active_mask}
            if profile is not None:
                step_kwargs["profile"] = profile
            next_active = physics_runtime_step(
                current_np[batch_index, active_slots],
                prediction_np[batch_index, active_slots],
                runtime_context,
                **step_kwargs,
            )
        except Exception:
            success_np[batch_index] = False
            continue
        next_np[batch_index, active_slots] = next_active
    if profile is not None:
        add_profile_time(profile, "batch_runtime_loop_seconds", time.perf_counter() - loop_start)

    section_start = time.perf_counter() if profile is not None else None
    success = torch.as_tensor(success_np, dtype=torch.bool, device=device)
    if profile is not None:
        add_profile_time(profile, "batch_success_to_device_seconds", time.perf_counter() - section_start)

    section_start = time.perf_counter() if profile is not None else None
    next_tensor = torch.as_tensor(next_np, dtype=dtype, device=device)
    if profile is not None:
        add_profile_time(profile, "batch_next_to_device_seconds", time.perf_counter() - section_start)
        add_profile_time(profile, "batch_runtime_step_total_seconds", time.perf_counter() - total_start)
    return next_tensor, success


def add_profile_time(profile: dict[str, float], key: str, elapsed: float) -> None:
    profile[key] = float(profile.get(key, 0.0) + elapsed)


def masked_velocity_mse(prediction, target, mask):
    expanded_mask = mask.unsqueeze(-1).expand_as(target)
    squared_error = (prediction - target) ** 2
    valid_error = squared_error[expanded_mask]
    if valid_error.numel() == 0:
        return squared_error.sum() * 0.0
    return valid_error.mean()


def velocity_mm_s_to_px_frame_scale(dataset, device):
    velocity_units = getattr(dataset, "velocity_units", "px/frame")
    if velocity_units == "mm/s":
        conversion = float(getattr(dataset, "velocity_mm_s_per_px_frame", 1.0))
        if conversion <= 0 or not np.isfinite(conversion):
            raise ValueError(f"Invalid velocity conversion factor: {conversion}")
        return torch.as_tensor(1.0 / conversion, dtype=torch.float32, device=device)
    return torch.as_tensor(1.0, dtype=torch.float32, device=device)


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
        "sum_sq_bbox_w": 0.0,
        "sum_sq_bbox_h": 0.0,
        "position_count": 0,
        "sum_sq_position": 0.0,
    }


def update_metric_accumulators(accumulators, rollout):
    velocity_error = rollout["pred_velocity"] - rollout["true_velocity"]
    bbox_error = rollout["pred_target"][..., 2:4] - rollout["true_target"][..., 2:4]
    speed_error = torch.sqrt(velocity_error[..., 0] ** 2 + velocity_error[..., 1] ** 2)
    position_error = rollout["pred_position"] - rollout["true_position"]
    position_error_norm = torch.sqrt(position_error[..., 0] ** 2 + position_error[..., 1] ** 2)
    position_finite = torch.isfinite(position_error).all(dim=-1)
    update_one_accumulator(
        accumulators["overall"],
        velocity_error,
        bbox_error,
        speed_error,
        rollout["supervision_mask"],
        position_error_norm,
        rollout["supervision_mask"] & position_finite,
    )
    for step_index in range(rollout["supervision_mask"].shape[1]):
        update_one_accumulator(
            accumulators["steps"][step_index],
            velocity_error[:, step_index, :, :],
            bbox_error[:, step_index, :, :],
            speed_error[:, step_index, :],
            rollout["supervision_mask"][:, step_index, :],
            position_error_norm[:, step_index, :],
            rollout["supervision_mask"][:, step_index, :] & position_finite[:, step_index, :],
        )


def update_one_accumulator(accumulator, velocity_error, bbox_error, speed_error, velocity_mask, position_error_norm, position_mask):
    valid = velocity_mask.bool()
    if valid.sum().item() > 0:
        vx_error = velocity_error[..., 0][valid]
        vy_error = velocity_error[..., 1][valid]
        bbox_w_error = bbox_error[..., 0][valid]
        bbox_h_error = bbox_error[..., 1][valid]
        speed = speed_error[valid]
        accumulator["count"] += int(valid.sum().item())
        accumulator["sum_sq_vx"] += float((vx_error**2).sum().detach().cpu())
        accumulator["sum_sq_vy"] += float((vy_error**2).sum().detach().cpu())
        accumulator["sum_sq_speed"] += float((speed**2).sum().detach().cpu())
        accumulator["sum_sq_bbox_w"] += float((bbox_w_error**2).sum().detach().cpu())
        accumulator["sum_sq_bbox_h"] += float((bbox_h_error**2).sum().detach().cpu())
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
        "rmse_bbox_w": safe_rmse(accumulator["sum_sq_bbox_w"], count),
        "rmse_bbox_h": safe_rmse(accumulator["sum_sq_bbox_h"], count),
        "rmse_position": safe_rmse(accumulator["sum_sq_position"], position_count),
    }


def safe_rmse(sum_sq, count):
    return np.sqrt(sum_sq / count) if count else np.nan


def print_progress(label, num_batches, total_batches, running_loss):
    percent = 100.0 * num_batches / max(total_batches, 1)
    print(f"  {label:<5} batch {num_batches:04d}/{total_batches:04d} ({percent:5.1f}%) weighted_loss={running_loss:.6f}")


def print_epoch_summary(epoch, train_summary, val_summary):
    active_rollout_horizon = int(train_summary.get("active_rollout_horizon", len(val_summary["step_rmse_position"])))
    available_steps = min(len(val_summary["step_rmse_position"]), max(DIAGNOSTIC_STEPS))
    step_text = " ".join(
        f"s{step}={val_summary['step_rmse_position'][step - 1]:.6f}"
        for step in DIAGNOSTIC_STEPS
        if step <= available_steps
    )
    print(
        f"epoch {epoch:03d} "
        f"rollout_horizon={active_rollout_horizon} "
        f"train_weighted_loss_internal_only={train_summary['weighted_loss_internal_only']:.6f} "
        f"val_weighted_loss_internal_only={val_summary['weighted_loss_internal_only']:.6f} "
        f"val_rmse_vx={val_summary['rmse_vx']:.6f} "
        f"val_rmse_vy={val_summary['rmse_vy']:.6f} "
        f"val_rmse_speed={val_summary['rmse_speed']:.6f} "
        f"val_rmse_bbox_w={val_summary['rmse_bbox_w']:.6f} "
        f"val_rmse_bbox_h={val_summary['rmse_bbox_h']:.6f} "
        f"val_rmse_position={val_summary['rmse_position']:.6f} "
        f"train_runtime_fallback={train_summary.get('runtime_step_fallback_fraction', 0.0):.6f} "
        f"val_runtime_fallback={val_summary.get('runtime_step_fallback_fraction', 0.0):.6f}"
    )
    print(f"  stepwise_val_rmse_position {step_text}")
    pure = val_summary.get("pure")
    if pure is not None:
        pure_available_steps = min(len(pure["step_rmse_position"]), max(DIAGNOSTIC_STEPS))
        pure_step_text = " ".join(
            f"s{step}={pure['step_rmse_position'][step - 1]:.6f}"
            for step in DIAGNOSTIC_STEPS
            if step <= pure_available_steps
        )
        print(
            f"  pure_alpha0_val "
            f"weighted_loss_internal_only={pure['weighted_loss_internal_only']:.6f} "
            f"rmse_position={pure['rmse_position']:.6f} "
            f"runtime_fallback={pure.get('runtime_step_fallback_fraction', 0.0):.6f}"
        )
        print(f"  pure_alpha0_stepwise_val_rmse_position {pure_step_text}")
    fusion = train_summary.get("adaptive_fusion", {})
    if fusion.get("enabled"):
        alpha_values = fusion.get("alpha_by_step_mean", [])
        alpha_text = " ".join(
            f"a{step}={alpha_values[step - 1]:.6f}"
            for step in DIAGNOSTIC_STEPS
            if step <= len(alpha_values)
        )
        print(f"  adaptive_fusion_alpha_measurement_weight {alpha_text}")


def runtime_context_for_epoch(config: dict[str, Any], epoch: int, runtime_context):
    refresh = config.get("training", {}).get("physics_refresh", {})
    runtime_start = int(refresh.get("runtime_start_epoch", 1))
    return runtime_context if int(epoch) >= runtime_start else None


def rollout_horizon_for_epoch(config: dict[str, Any], epoch: int, full_rollout_horizon: int) -> int:
    schedule = config.get("training", {}).get("rollout_horizon_schedule", [])
    if not schedule:
        return int(full_rollout_horizon)
    active_horizon = int(full_rollout_horizon)
    active_start = -1
    for item in schedule:
        start_epoch = int(item["start_epoch"])
        horizon = int(item["horizon"])
        if horizon < 1:
            raise ValueError(f"rollout_horizon_schedule horizon must be >= 1, got {horizon}")
        if horizon > int(full_rollout_horizon):
            raise ValueError(
                f"rollout_horizon_schedule horizon {horizon} exceeds model rollout_horizon {full_rollout_horizon}"
            )
        if start_epoch <= int(epoch) and start_epoch >= active_start:
            active_start = start_epoch
            active_horizon = horizon
    return active_horizon


def rollout_horizon_schedule_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("rollout_horizon_schedule", []))


def adaptive_target_fusion_enabled(config: dict[str, Any]) -> bool:
    fusion = config.get("training", {}).get("adaptive_target_fusion", {})
    return bool(fusion.get("enabled", False))


def physics_refresh_mode(runtime_context) -> str:
    return "runtime" if runtime_context is not None else "stale"


def should_update_best_checkpoint(
    runtime_context,
    val_summary: dict[str, Any],
    best_val_loss: float,
    *,
    active_rollout_horizon: int | None = None,
    full_rollout_horizon: int | None = None,
) -> bool:
    if runtime_context is None:
        return False
    if (
        active_rollout_horizon is not None
        and full_rollout_horizon is not None
        and int(active_rollout_horizon) != int(full_rollout_horizon)
    ):
        return False
    return float(val_summary["weighted_loss_internal_only"]) < float(best_val_loss)


def initialize_curves_csv(path):
    if path.exists():
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, None)
        if existing_header == CURVES_COLUMNS:
            return
        archive = path.with_name(f"{path.stem}_legacy_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        path.replace(archive)
        print(f"Archived incompatible training curves CSV: {archive}")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CURVES_COLUMNS)


def diagnostic_step_value(summary: dict[str, Any], step: int) -> float:
    values = summary.get("step_rmse_position", [])
    return float(values[step - 1]) if step <= len(values) else np.nan


def diagnostic_alpha_value(summary: dict[str, Any], step: int) -> float:
    fusion = summary.get("adaptive_fusion", {})
    values = fusion.get("alpha_by_step_mean", [])
    return float(values[step - 1]) if step <= len(values) else 0.0


def diagnostic_alpha_feature_value(summary: dict[str, Any], feature_index: int, step: int) -> float:
    fusion = summary.get("adaptive_fusion", {})
    values = fusion.get("alpha_by_step_feature", [])
    if step > len(values):
        return 0.0
    step_values = values[step - 1]
    return float(step_values[feature_index]) if feature_index < len(step_values) else 0.0


def pure_summary_value(summary: dict[str, Any], key: str) -> float:
    return float(summary.get("pure", {}).get(key, summary.get(key, np.nan)))


def pure_diagnostic_step_value(summary: dict[str, Any], step: int) -> float:
    pure = summary.get("pure", summary)
    return diagnostic_step_value(pure, step)


def save_adaptive_fusion_alpha_plot(alpha_history: list[dict[str, Any]], path: Path) -> None:
    if not alpha_history:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    plt.figure(figsize=(8, 5))
    for item in alpha_history:
        values = item.get("alpha_by_step_mean", [])
        if not values:
            continue
        steps = np.arange(1, len(values) + 1)
        plt.plot(steps, values, alpha=0.35, linewidth=1.0, label=f"epoch {item['epoch']}")
    plt.xlabel("rollout step")
    plt.ylabel("alpha (measurement weight)")
    plt.ylim(0.0, 1.0)
    if len(alpha_history) <= 8:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def append_curves_csv(path, epoch, train_summary, val_summary):
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                train_summary.get("active_rollout_horizon", np.nan),
                train_summary["weighted_loss_internal_only"],
                val_summary["weighted_loss_internal_only"],
                train_summary["cfd_valid_target_fraction"],
                val_summary["cfd_valid_target_fraction"],
                val_summary["rmse_vx"],
                val_summary["rmse_vy"],
                val_summary["rmse_speed"],
                val_summary["rmse_bbox_w"],
                val_summary["rmse_bbox_h"],
                val_summary["rmse_position"],
                train_summary.get("runtime_step_attempts", 0.0),
                train_summary.get("runtime_step_fallbacks", 0.0),
                train_summary.get("runtime_step_fallback_fraction", 0.0),
                val_summary.get("runtime_step_attempts", 0.0),
                val_summary.get("runtime_step_fallbacks", 0.0),
                val_summary.get("runtime_step_fallback_fraction", 0.0),
                *[diagnostic_step_value(val_summary, step) for step in DIAGNOSTIC_STEPS],
                pure_summary_value(val_summary, "weighted_loss_internal_only"),
                pure_summary_value(val_summary, "rmse_vx"),
                pure_summary_value(val_summary, "rmse_vy"),
                pure_summary_value(val_summary, "rmse_speed"),
                pure_summary_value(val_summary, "rmse_bbox_w"),
                pure_summary_value(val_summary, "rmse_bbox_h"),
                pure_summary_value(val_summary, "rmse_position"),
                pure_summary_value(val_summary, "runtime_step_attempts"),
                pure_summary_value(val_summary, "runtime_step_fallbacks"),
                pure_summary_value(val_summary, "runtime_step_fallback_fraction"),
                *[pure_diagnostic_step_value(val_summary, step) for step in DIAGNOSTIC_STEPS],
                *[diagnostic_alpha_value(train_summary, step) for step in DIAGNOSTIC_STEPS],
                *[
                    diagnostic_alpha_feature_value(train_summary, feature_index, step)
                    for feature_index, _feature in enumerate(RUNTIME_TARGET_FEATURES)
                    for step in DIAGNOSTIC_STEPS
                ],
                train_summary.get("adaptive_fusion", {}).get("alpha_mean", 0.0),
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
