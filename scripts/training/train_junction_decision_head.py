from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.training import train_physics_markovian as base


DEFAULT_CONFIG = Path("configs/experiments/physics_markovian_v1_decision.yml")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = base.load_config(args.config)
    if args.smoke_test:
        base.apply_smoke_test_overrides(config)

    base.set_random_seed(int(config["training"]["random_seed"]))
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device_info = base.select_device(config.get("device", {}).get("mode", "auto"))
    device = device_info["device"]
    print(f"Selected device: {device}")

    base.save_json(output_dir / "resolved_config.json", config)

    train_ds, val_ds, test_ds, normalization_stats = base.create_train_val_test_datasets(
        npz_path=config["dataset"]["path"],
        stride=int(config["dataset"]["stride"]),
        T_history=int(config["model"]["T_history"]),
        T_future=int(config["model"]["rollout_horizon"]),
        max_droplets=int(config["model"]["max_droplets"]),
        target_features=tuple(config["model"]["target_features"]),
        experiment_config=config["dataset"].get("experiment_config", "configs/experiments/video_2.yml"),
    )
    base.validate_feature_contract(train_ds, config)
    target_parameterization = base.target_parameterization_from_config(config)
    target_deadband = base.target_deadband_from_config(config)
    event_exclusion = base.event_exclusion_from_config(config)
    if args.smoke_test:
        train_ds = base.SubsetByIndex(train_ds, int(config["smoke_test"]["train_windows"]))
        val_ds = base.SubsetByIndex(val_ds, int(config["smoke_test"]["val_windows"]))

    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")
    print(f"Test windows: {len(test_ds)}")

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

    decision_section = config.get("training", {}).get("decision_head", {})
    if not bool(decision_section.get("enabled", False)):
        raise ValueError("This script requires training.decision_head.enabled=true in the config.")
    if not bool(config["model"]["architecture"].get("predict_branch_decision", False)):
        raise ValueError("This script requires model.architecture.predict_branch_decision=true in the config.")

    model_config = dict(config["model"]["architecture"])
    model_config.update(
        input_dim=int(config["model"]["input_dim"]),
        target_dim=len(config["model"]["target_features"]),
        T_history=int(config["model"]["T_history"]),
        max_droplets=int(config["model"]["max_droplets"]),
    )
    model = base.CanonicalRolloutTransformer(**model_config).to(device)

    warm_start_path = decision_section.get("warm_start_checkpoint")
    if warm_start_path is not None:
        checkpoint = torch.load(warm_start_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        missing_non_decision = [name for name in missing if not name.startswith("decision_head")]
        if missing_non_decision:
            raise ValueError(
                f"Warm-start checkpoint is missing trunk parameters (not just decision_head): {missing_non_decision}"
            )
        if unexpected:
            raise ValueError(f"Warm-start checkpoint has unexpected parameters for this architecture: {unexpected}")
        print(
            f"Warm-started trunk from {warm_start_path} "
            f"(epoch {checkpoint.get('epoch', '?')}); decision_head initialized fresh"
        )

    freeze_trunk = bool(decision_section.get("freeze_trunk", True))
    if freeze_trunk:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("decision_head")
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"Trunk frozen; trainable parameters: {trainable}")
        loss_key = "weighted_decision_loss"
    else:
        loss_key = "total_loss"

    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    weights = base.rollout_weights(
        int(config["model"]["rollout_horizon"]),
        float(config["training"]["loss_alpha"]),
        device,
    )
    runtime_context = base.load_physics_runtime_context(
        experiment_config_path=config["dataset"].get("experiment_config", "configs/experiments/video_2.yml"),
        cfd_library_path=config["dataset"].get("cfd_library_path", "outputs/physics/full_device_cfd/library"),
        feature_names=tuple(config["model"]["input_feature_names"]),
    )
    geometry_constraint = base.create_geometry_constraint(config, runtime_context, device)
    hard_wall_containment = base.create_hard_wall_containment(config, runtime_context, device)
    branch_decision = base.create_branch_decision_training(config)
    print(
        f"Branch decision labels: {branch_decision.branch_label.shape} "
        f"loss_weight={branch_decision.loss_weight}"
    )

    base.train_full(
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
        geometry_constraint=geometry_constraint,
        hard_wall_containment=hard_wall_containment,
        target_parameterization=target_parameterization,
        target_deadband=target_deadband,
        event_exclusion=event_exclusion,
        model_config=model_config,
        output_dir=output_dir,
        branch_decision=branch_decision,
        loss_key=loss_key,
        decision_calibration_every_n_epochs=int(decision_section.get("calibration_every_n_epochs", 5)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: train the junction branch-decision auxiliary head.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
