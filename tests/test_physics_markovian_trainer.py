from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.training import train_physics_markovian as trainer
from src.datasets.canonical_window_dataset import CanonicalWindowDataset
from src.models.canonical_rollout_transformer import CanonicalRolloutTransformer


V1_FEATURES = ["x", "y", "vx", "vy", "circularity"]
V2_FEATURES = trainer.FEATURE_NAMES


def test_cpu_device_selection_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    info = trainer.select_device("auto")
    assert str(info["device"]) == "cpu"
    assert info["cuda_available"] is False


def test_cuda_auto_selection_without_requiring_cuda(monkeypatch) -> None:
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "get_device_name", lambda index: "Mock CUDA GPU")
    info = trainer.select_device("auto")
    assert str(info["device"]) == "cuda"
    assert info["gpu_name"] == "Mock CUDA GPU"


def test_model_accepts_16_dimensional_markovian_state() -> None:
    model = _small_model(input_dim=16, horizon=1, max_droplets=4, target_dim=4)
    history_x = torch.randn(2, 1, 4, 16)
    history_mask = torch.ones(2, 1, 4, dtype=torch.bool)
    output = model(history_x, history_mask)
    assert output.shape == (2, 4, 4)


def test_batch_unpacking_and_cfd_loss_mask_from_loader(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "v2.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(npz, start_frames=[0], T_history=1, T_future=3, max_droplets=4)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch["history_x"].shape == (1, 1, 4, 16)
    assert batch["cfd_loss_mask"].shape == (1, 3, 4)
    assert batch["future_mask"][0, 1, 0].item() is True
    assert batch["cfd_loss_mask"][0, 1, 0].item() is True


def test_cfd_loss_mask_controls_supervised_loss(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "v2.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(npz, start_frames=[0], T_history=1, T_future=3, max_droplets=4)
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    model = _small_model(input_dim=16, horizon=1, max_droplets=4)
    stats = _identity_stats(16)
    weights = trainer.rollout_weights(3, 2.0, torch.device("cpu"))
    rollout = trainer.boundary_conditioned_rollout(model, batch, dataset, stats, weights)
    assert rollout["mask"][0, 1, 0].item() is True
    assert rollout["supervision_mask"][0, 1, 0].item() is True


def test_invalid_targets_make_zero_loss_contribution() -> None:
    prediction = torch.tensor([[[1.0, 2.0], [10.0, 10.0]]])
    target = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    loss = trainer.masked_velocity_mse(prediction, target, mask)
    assert loss.item() == pytest.approx(0.0)


def test_no_valid_cfd_targets_do_not_produce_nan() -> None:
    prediction = torch.ones(1, 2, 2)
    target = torch.zeros(1, 2, 2)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    loss = trainer.masked_velocity_mse(prediction, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_checkpoint_save_load_with_map_location(tmp_path: Path) -> None:
    model = _small_model(input_dim=16, horizon=1, max_droplets=4, target_dim=4)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "input_dim": 16,
            "target_dim": 4,
            "T_history": 1,
            "max_droplets": 4,
            "d_model": 16,
            "n_heads": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
        },
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    reloaded = CanonicalRolloutTransformer(**loaded["model_config"])
    reloaded.load_state_dict(loaded["model_state_dict"])
    assert isinstance(reloaded, CanonicalRolloutTransformer)


def test_one_optimization_step_and_short_rollout(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "v2.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(npz, start_frames=[0], T_history=1, T_future=3, max_droplets=4)
    loader = DataLoader(dataset, batch_size=1)
    model = _small_model(input_dim=16, horizon=1, max_droplets=4)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    stats = _identity_stats(16)
    weights = trainer.rollout_weights(3, 2.0, torch.device("cpu"))
    summary = trainer.train_one_epoch(
        model,
        loader,
        dataset,
        optimizer,
        stats,
        weights,
        torch.device("cpu"),
        grad_clip=1.0,
        log_every=0,
        max_batches=1,
    )
    assert np.isfinite(summary["weighted_loss_internal_only"])


def test_original_5_feature_dataset_remains_practical(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "v1.npz", V1_FEATURES)
    dataset = CanonicalWindowDataset(npz, start_frames=[0], T_history=1, T_future=2, max_droplets=4)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch["history_x"].shape == (1, 1, 4, 5)
    assert torch.equal(batch["cfd_loss_mask"], batch["future_mask"])


def test_closed_loop_runtime_is_called_for_every_predicted_step(tmp_path: Path, monkeypatch) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    model = _ConstantPredictionModel([1.0, 2.0, 21.0, 13.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 2.0, torch.device("cpu"))
    recorder = _RuntimeRecorder(dataset.feature_indices)
    monkeypatch.setattr(trainer, "physics_runtime_step", recorder)

    trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=_runtime_context(dataset),
    )

    assert len(recorder.calls) == 3
    assert [len(call["active_mask"]) for call in recorder.calls] == [2, 2, 2]
    assert [int(call["active_mask"].sum()) for call in recorder.calls] == [2, 2, 2]


def test_runtime_batch_preserves_independent_batch_rows(tmp_path: Path, monkeypatch) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    batch = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in batch.items()}
    batch["history_x"][1, :, :, dataset.feature_indices["x"]] += 10.0
    model = _ConstantPredictionModel([1.0, 2.0, 21.0, 13.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))
    recorder = _RuntimeRecorder(dataset.feature_indices)
    monkeypatch.setattr(trainer, "physics_runtime_step", recorder)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=_runtime_context(dataset),
    )

    x_index = dataset.feature_indices["x"]
    assert len(recorder.calls) == 2
    assert [call["current_state"].shape[0] for call in recorder.calls] == [2, 2]
    assert rollout["pred_state"][0, 0, 0, x_index].item() == pytest.approx(100.0)
    assert rollout["pred_state"][1, 0, 0, x_index].item() == pytest.approx(101.0)


def test_closed_loop_runtime_state_becomes_next_model_input(tmp_path: Path, monkeypatch) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    model = _ConstantPredictionModel([1.0, 2.0, 21.0, 13.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(2, 2.0, torch.device("cpu"))
    recorder = _RuntimeRecorder(dataset.feature_indices)
    monkeypatch.setattr(trainer, "physics_runtime_step", recorder)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=_runtime_context(dataset),
    )

    cfd_index = dataset.feature_indices["cfd_u_norm"]
    x_index = dataset.feature_indices["x"]
    assert rollout["pred_state"][0, 0, 0, cfd_index].item() == pytest.approx(7.0)
    assert model.seen_history[1][0, -1, 0, cfd_index].item() == pytest.approx(7.0)
    assert model.seen_history[1][0, -1, 0, x_index].item() == pytest.approx(100.0)


def test_closed_loop_does_not_refresh_stale_physics_from_future_truth(tmp_path: Path, monkeypatch) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    model = _ConstantPredictionModel([1.0, 2.0, 21.0, 13.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))
    recorder = _RuntimeRecorder(dataset.feature_indices)
    monkeypatch.setattr(trainer, "physics_runtime_step", recorder)

    def fail_refresh(*args, **kwargs):
        raise AssertionError("refresh_observed_non_target_features should not run for closed-loop continuing slots")

    monkeypatch.setattr(trainer, "refresh_observed_non_target_features", fail_refresh)
    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=_runtime_context(dataset),
    )

    cfd_index = dataset.feature_indices["cfd_u_norm"]
    assert rollout["pred_state"][0, 0, 0, cfd_index].item() == pytest.approx(7.0)
    assert rollout["pred_state"][0, 0, 0, cfd_index].item() != pytest.approx(0.1)


def test_closed_loop_preserves_entering_truth_and_inactive_zero_padding(tmp_path: Path, monkeypatch) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    batch["history_mask"][0, :, 1] = False
    model = _ConstantPredictionModel([1.0, 2.0, 21.0, 13.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))
    recorder = _RuntimeRecorder(dataset.feature_indices)
    monkeypatch.setattr(trainer, "physics_runtime_step", recorder)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=_runtime_context(dataset),
    )

    expected_entering = torch.as_tensor(dataset.Z[1, 1, :], dtype=torch.float32)
    assert recorder.calls[0]["active_mask"].tolist() == [True]
    assert torch.allclose(rollout["pred_state"][0, 0, 1], expected_entering)
    assert torch.allclose(rollout["pred_state"][0, 0, 3], torch.zeros(16))


def test_runtime_failure_falls_back_to_stale_physics_for_that_step(tmp_path: Path, monkeypatch) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    model = _ConstantPredictionModel([1.0, 2.0, 31.0, 17.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))

    def fail_runtime(*args, **kwargs):
        raise ValueError("synthetic runtime failure")

    monkeypatch.setattr(trainer, "physics_runtime_step", fail_runtime)
    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=_runtime_context(dataset),
    )

    idx = dataset.feature_indices
    assert rollout["runtime_step_attempts"] == 1
    assert rollout["runtime_step_fallbacks"] == 1
    assert rollout["pred_state"][0, 0, 0, idx["bbox_w"]].item() == pytest.approx(31.0)
    assert rollout["pred_state"][0, 0, 0, idx["bbox_h"]].item() == pytest.approx(17.0)
    assert rollout["pred_state"][0, 0, 0, idx["cfd_u_norm"]].item() == pytest.approx(
        dataset.Z[0, 0, idx["cfd_u_norm"]]
    )


def test_scheduled_sampling_p_truth_1_uses_truth_for_rollout_but_loss_uses_raw_prediction(tmp_path: Path) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    model = _ConstantPredictionModel([10.0, 20.0, 31.0, 17.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        p_truth=1.0,
    )

    idx = dataset.feature_indices
    # The primary loss still compares the model's raw prediction to ground truth...
    assert rollout["pred_target"][0, 0, 0, 0].item() == pytest.approx(10.0)
    assert rollout["true_target"][0, 0, 0, 0].item() == pytest.approx(1.0)
    assert rollout["weighted_loss_internal_only"].item() > 0.0
    # ...but with p_truth=1.0, every continuing droplet's propagated state is ground truth, not the raw prediction.
    assert rollout["pred_state"][0, 0, 0, idx["vx"]].item() == pytest.approx(1.0)
    assert rollout["pred_state"][0, 0, 0, idx["bbox_w"]].item() == pytest.approx(20.0)


def test_geometry_constraint_uses_predicted_bbox_not_true_future_bbox(tmp_path: Path) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    model = _ConstantPredictionModel([0.0, 0.0, 80.0, 80.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))
    channel_mask = torch.zeros((64, 64), dtype=torch.float32)
    channel_mask[0:20, 0:20] = 1.0
    geometry = trainer.GeometryConstraint(
        enabled=True,
        channel_mask=channel_mask,
        weight=2.0,
        tolerance=0.02,
        num_samples_x=16,
        num_samples_y=16,
    )

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        geometry_constraint=geometry,
    )

    assert rollout["geometry_count"] == 2
    assert rollout["geometry_loss"].item() > 0.0
    assert rollout["weighted_geometry_loss"].item() == pytest.approx(2.0 * rollout["geometry_loss"].item())
    assert rollout["total_loss"].item() == pytest.approx(
        rollout["weighted_loss_internal_only"].item() + rollout["weighted_geometry_loss"].item()
    )


def test_geometry_constraint_excludes_boundary_injected_droplets(tmp_path: Path) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    batch["history_mask"][0, :, 1] = False
    model = _ConstantPredictionModel([0.0, 0.0, 80.0, 80.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))
    geometry = trainer.GeometryConstraint(
        enabled=True,
        channel_mask=torch.ones((64, 64), dtype=torch.float32),
        weight=1.0,
        tolerance=0.02,
        num_samples_x=8,
        num_samples_y=8,
    )

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        geometry_constraint=geometry,
    )

    assert rollout["boundary_mask"][0, 0, 1].item() is True
    assert rollout["geometry_mask"][0, 0, 1].item() is False


def test_hard_wall_containment_backward_survives_boundary_injected_droplets(tmp_path: Path) -> None:
    # Regression test: raw_position was previously mutated in place (via boundary_mask
    # indexing) after being passed into clamp_to_channel_torch, which only trips PyTorch's
    # autograd version check when boundary_mask actually has a True entry and a real
    # parameterized model is used -- a tiny constant-prediction / no-boundary-injection
    # smoke test does not exercise this path, which is how it slipped through originally.
    dataset, batch = _v2_four_target_batch(tmp_path)
    batch["history_mask"][0, :, 1] = False
    model = _small_model(input_dim=16, horizon=1, max_droplets=4, target_dim=4)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 2.0, torch.device("cpu"))
    mask = np.ones((64, 64), dtype=bool)
    mask[63, :] = False
    mask[:, 63] = False
    from src.physics.constraints import wall_sdf_to_torch
    from src.physics.geometry.wall_sdf import build_wall_sdf

    sdf, grad_x, grad_y = wall_sdf_to_torch(build_wall_sdf(mask))
    hard_wall_containment = trainer.HardWallContainment(enabled=True, sdf=sdf, grad_x=grad_x, grad_y=grad_y)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        hard_wall_containment=hard_wall_containment,
    )
    rollout["total_loss"].backward()

    assert all(
        param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters()
    )


def test_p_truth_for_epoch_follows_piecewise_schedule() -> None:
    sampling = trainer.ScheduledSampling(
        enabled=True,
        schedule=((1, 1.0), (20, 0.5), (60, 0.2), (120, 0.05)),
    )
    assert trainer.p_truth_for_epoch(sampling, 1) == pytest.approx(1.0)
    assert trainer.p_truth_for_epoch(sampling, 19) == pytest.approx(1.0)
    assert trainer.p_truth_for_epoch(sampling, 20) == pytest.approx(0.5)
    assert trainer.p_truth_for_epoch(sampling, 59) == pytest.approx(0.5)
    assert trainer.p_truth_for_epoch(sampling, 120) == pytest.approx(0.05)
    assert trainer.p_truth_for_epoch(sampling, 10_000) == pytest.approx(0.05)


def test_p_truth_for_epoch_returns_none_when_disabled_or_absent() -> None:
    disabled = trainer.ScheduledSampling(enabled=False, schedule=((1, 1.0),))
    assert trainer.p_truth_for_epoch(disabled, 50) is None
    assert trainer.p_truth_for_epoch(None, 50) is None


def test_create_scheduled_sampling_validates_p_truth_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        trainer.create_scheduled_sampling(
            {"training": {"scheduled_sampling": {"enabled": True, "schedule": [{"start_epoch": 1, "p_truth": 1.5}]}}}
        )


def test_sample_rollout_targets_is_hard_select_not_a_blend() -> None:
    pred = torch.zeros(4, 5, 4)
    true = torch.ones(4, 5, 4)
    continuing_mask = torch.ones(4, 5, dtype=torch.bool)

    torch.manual_seed(0)
    result = trainer.sample_rollout_targets(pred, true, continuing_mask, 0.5)

    # Every selected entry is exactly one tensor or the other -- never an in-between blend.
    is_pred = torch.isclose(result, pred)
    is_true = torch.isclose(result, true)
    assert torch.all(is_pred | is_true)
    # With p_truth=0.5 over 20 independent draws, expect a genuine mix, not all-one-value.
    assert bool(is_true.any()) and bool(is_pred.any())


def test_sample_rollout_targets_p_truth_none_is_pure_self_conditioning() -> None:
    pred = torch.zeros(2, 3, 4)
    true = torch.ones(2, 3, 4)
    continuing_mask = torch.ones(2, 3, dtype=torch.bool)
    result = trainer.sample_rollout_targets(pred, true, continuing_mask, None)
    assert torch.equal(result, pred)


def test_sample_rollout_targets_respects_continuing_mask_and_nan_truth() -> None:
    pred = torch.full((1, 3, 4), 2.0)
    true = torch.ones(1, 3, 4)
    true[0, 1, 0] = float("nan")  # not finite -> must not be selected even at p_truth=1.0
    continuing_mask = torch.tensor([[True, True, False]])  # slot 2 is not continuing

    result = trainer.sample_rollout_targets(pred, true, continuing_mask, 1.0)

    assert torch.equal(result[0, 0], true[0, 0])  # continuing + finite -> truth
    assert torch.equal(result[0, 1], pred[0, 1])  # continuing but non-finite truth -> prediction
    assert torch.equal(result[0, 2], pred[0, 2])  # not continuing -> prediction


def test_training_curves_include_step_rmse_and_scheduled_sampling_p_truth(tmp_path: Path) -> None:
    path = tmp_path / "training_curves.csv"
    train_summary = {
        "active_rollout_horizon": 50.0,
        "weighted_loss_internal_only": 1.0,
        "cfd_valid_target_fraction": 0.5,
        "runtime_step_attempts": 2.0,
        "runtime_step_fallbacks": 0.0,
        "runtime_step_fallback_fraction": 0.0,
        "scheduled_sampling_p_truth": 0.35,
    }
    val_summary = {
        "weighted_loss_internal_only": 2.0,
        "cfd_valid_target_fraction": 0.6,
        "rmse_vx": 1.0,
        "rmse_vy": 2.0,
        "rmse_speed": 3.0,
        "rmse_bbox_w": 3.5,
        "rmse_bbox_h": 3.6,
        "rmse_position": 4.0,
        "runtime_step_attempts": 3.0,
        "runtime_step_fallbacks": 0.0,
        "runtime_step_fallback_fraction": 0.0,
        "step_rmse_position": [float(step) for step in range(1, 51)],
        "pure": {
            "weighted_loss_internal_only": 3.0,
            "rmse_vx": 4.0,
            "rmse_vy": 5.0,
            "rmse_speed": 6.0,
            "rmse_bbox_w": 6.5,
            "rmse_bbox_h": 6.6,
            "rmse_position": 7.0,
            "runtime_step_attempts": 8.0,
            "runtime_step_fallbacks": 1.0,
            "runtime_step_fallback_fraction": 0.125,
            "step_rmse_position": [float(step * 10) for step in range(1, 51)],
        },
    }

    trainer.initialize_curves_csv(path)
    trainer.append_curves_csv(path, 1, train_summary, val_summary)

    lines = path.read_text().splitlines()
    assert "active_rollout_horizon" in lines[0]
    assert "train_total_loss" in lines[0]
    assert "val_geometry_loss" in lines[0]
    assert "val_geometry_violation_fraction" in lines[0]
    assert "val_rmse_bbox_w" in lines[0]
    assert "val_pure_rmse_bbox_h" in lines[0]
    assert "val_rmse_position_s50" in lines[0]
    assert "val_pure_rmse_position_s50" in lines[0]
    assert "scheduled_sampling_p_truth" in lines[0]
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("active_rollout_horizon")] == "50.0"
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("val_rmse_bbox_w")] == "3.5"
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("val_pure_rmse_bbox_h")] == "6.6"
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("val_rmse_position_s50")] == "50.0"
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("val_pure_rmse_position_s50")] == "500.0"
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("scheduled_sampling_p_truth")] == "0.35"


def test_rollout_horizon_schedule_uses_latest_started_entry() -> None:
    config = {
        "training": {
            "rollout_horizon_schedule": [
                {"start_epoch": 1, "horizon": 10},
                {"start_epoch": 4, "horizon": 20},
                {"start_epoch": 9, "horizon": 50},
            ]
        }
    }

    assert trainer.rollout_horizon_for_epoch(config, 1, 50) == 10
    assert trainer.rollout_horizon_for_epoch(config, 3, 50) == 10
    assert trainer.rollout_horizon_for_epoch(config, 4, 50) == 20
    assert trainer.rollout_horizon_for_epoch(config, 9, 50) == 50


def test_physics_refresh_epoch_schedule_switches_from_stale_to_runtime() -> None:
    config = {"training": {"physics_refresh": {"runtime_start_epoch": 2}}}
    runtime_context = object()
    assert trainer.runtime_context_for_epoch(config, 1, runtime_context) is None
    assert trainer.runtime_context_for_epoch(config, 2, runtime_context) is runtime_context
    assert trainer.physics_refresh_mode(None) == "stale"
    assert trainer.physics_refresh_mode(runtime_context) == "runtime"


def test_best_checkpoint_selection_ignores_stale_physics_epochs() -> None:
    runtime_context = object()
    stale_summary = {"weighted_loss_internal_only": 0.1}
    runtime_better_summary = {"weighted_loss_internal_only": 0.2}
    runtime_worse_summary = {"weighted_loss_internal_only": 0.3}

    assert not trainer.should_update_best_checkpoint(None, stale_summary, float("inf"))
    assert trainer.should_update_best_checkpoint(runtime_context, runtime_better_summary, float("inf"))
    assert not trainer.should_update_best_checkpoint(runtime_context, runtime_worse_summary, 0.2)
    assert not trainer.should_update_best_checkpoint(
        runtime_context,
        runtime_better_summary,
        float("inf"),
        active_rollout_horizon=10,
        full_rollout_horizon=50,
    )
    assert trainer.should_update_best_checkpoint(
        runtime_context,
        runtime_better_summary,
        float("inf"),
        active_rollout_horizon=50,
        full_rollout_horizon=50,
    )


def test_stale_refresh_uses_predicted_bbox_and_observed_non_target_physics(tmp_path: Path) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    model = _ConstantPredictionModel([1.0, 2.0, 31.0, 17.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
    )

    idx = dataset.feature_indices
    assert rollout["pred_state"][0, 0, 0, idx["bbox_w"]].item() == pytest.approx(31.0)
    assert rollout["pred_state"][0, 0, 0, idx["bbox_h"]].item() == pytest.approx(17.0)
    assert rollout["pred_state"][0, 0, 0, idx["cfd_u_norm"]].item() == pytest.approx(
        dataset.Z[0, 1, idx["cfd_u_norm"]]
    )
    assert rollout["pred_state"][0, 0, 0, idx["left_flow_fraction"]].item() == pytest.approx(
        dataset.Z[0, 1, idx["left_flow_fraction"]]
    )


def test_position_targets_directly_update_rollout_state(tmp_path: Path) -> None:
    dataset, batch = _v2_position_target_batch(tmp_path)
    model = _ConstantPredictionModel([12.5, 34.5, 31.0, 17.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        target_parameterization={"mode": "position"},
    )

    idx = dataset.feature_indices
    scale = dataset.velocity_mm_s_per_px_frame
    previous_x = float(dataset.Z[0, 0, idx["x"]])
    previous_y = float(dataset.Z[0, 0, idx["y"]])
    assert rollout["pred_state"][0, 0, 0, idx["x"]].item() == pytest.approx(12.5)
    assert rollout["pred_state"][0, 0, 0, idx["y"]].item() == pytest.approx(34.5)
    assert rollout["pred_state"][0, 0, 0, idx["vx"]].item() == pytest.approx((12.5 - previous_x) * scale)
    assert rollout["pred_state"][0, 0, 0, idx["vy"]].item() == pytest.approx((34.5 - previous_y) * scale)
    assert rollout["pred_target"][0, 0, 0, 0].item() == pytest.approx(12.5)
    assert rollout["true_target"][0, 0, 0, 0].item() == pytest.approx(dataset.Z[0, 1, idx["x"]])


def test_position_targets_runtime_uses_position_prediction_mode(tmp_path: Path, monkeypatch) -> None:
    dataset, batch = _v2_position_target_batch(tmp_path)
    model = _ConstantPredictionModel([12.5, 34.5, 31.0, 17.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))
    recorder = _RuntimeRecorder(dataset.feature_indices)
    monkeypatch.setattr(trainer, "physics_runtime_step", recorder)

    trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=_runtime_context(dataset),
        target_parameterization={"mode": "position"},
    )

    assert recorder.calls[0]["prediction_mode"] == "position"
    assert recorder.calls[0]["model_prediction"][0, 0] == pytest.approx(12.5)
    assert recorder.calls[0]["model_prediction"][0, 1] == pytest.approx(34.5)


def _small_model(
    input_dim: int,
    horizon: int,
    max_droplets: int,
    target_dim: int = 2,
) -> CanonicalRolloutTransformer:
    return CanonicalRolloutTransformer(
        input_dim=input_dim,
        target_dim=target_dim,
        T_history=horizon,
        max_droplets=max_droplets,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )


def _identity_stats(feature_dim: int, target_dim: int = 2) -> dict[str, np.ndarray]:
    return {
        "input_mean": np.zeros(feature_dim, dtype=np.float32),
        "input_std": np.ones(feature_dim, dtype=np.float32),
        "target_mean": np.zeros(target_dim, dtype=np.float32),
        "target_std": np.ones(target_dim, dtype=np.float32),
    }


def _v2_four_target_batch(tmp_path: Path):
    npz = _write_npz(tmp_path / "v2_four_target.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=1,
        T_future=3,
        max_droplets=4,
        target_features=trainer.RUNTIME_TARGET_FEATURES,
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    return dataset, batch


def _v2_position_target_batch(tmp_path: Path):
    npz = _write_npz(tmp_path / "v2_position_target.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=1,
        T_future=3,
        max_droplets=4,
        target_features=trainer.POSITION_TARGET_FEATURES,
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    return dataset, batch


def _runtime_context(dataset: CanonicalWindowDataset):
    return SimpleNamespace(feature_index=dataset.feature_indices)


class _ConstantPredictionModel(torch.nn.Module):
    def __init__(self, prediction) -> None:
        super().__init__()
        self.register_buffer("prediction", torch.as_tensor(prediction, dtype=torch.float32))
        self.seen_history = []

    def forward(self, history_x, history_mask):
        self.seen_history.append(history_x.detach().clone())
        B, _, M, _ = history_x.shape
        return self.prediction.view(1, 1, -1).expand(B, M, -1).clone()


class _RuntimeRecorder:
    def __init__(self, feature_index: dict[str, int]) -> None:
        self.feature_index = feature_index
        self.calls = []

    def __call__(self, current_state, model_prediction, context, active_mask=None, prediction_mode="velocity", **_kwargs):
        active = np.asarray(active_mask, dtype=bool)
        self.calls.append(
            {
                "current_state": np.asarray(current_state).copy(),
                "model_prediction": np.asarray(model_prediction).copy(),
                "active_mask": active.copy(),
                "prediction_mode": prediction_mode,
            }
        )
        out = np.zeros_like(current_state, dtype=np.float32)
        idx = self.feature_index
        if np.any(active):
            out[active] = current_state[active]
            out[active, idx["x"]] = 100.0 + len(self.calls) - 1
            out[active, idx["y"]] = 200.0 + len(self.calls) - 1
            out[active, idx["vx"]] = model_prediction[active, 0]
            out[active, idx["vy"]] = model_prediction[active, 1]
            out[active, idx["bbox_w"]] = model_prediction[active, 2]
            out[active, idx["bbox_h"]] = model_prediction[active, 3]
            out[active, idx["cfd_u_norm"]] = 7.0
            out[active, idx["cfd_v_norm"]] = -7.0
            out[active, idx["superficial_velocity"]] = 56.0
            out[active, idx["left_flow_fraction"]] = 0.42
            for name in V2_FEATURES:
                if name.startswith("occupancy_"):
                    out[active, idx[name]] = 1.0 / 6.0
        return out


def _write_npz(path: Path, feature_names: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = 2
    frames = 6
    feature_dim = len(feature_names)
    idx = {name: i for i, name in enumerate(feature_names)}
    Z = np.full((tracks, frames, feature_dim), np.nan, dtype=np.float32)
    mask = np.ones((tracks, frames), dtype=bool)
    for track in range(tracks):
        for frame in range(frames):
            Z[track, frame, idx["x"]] = frame + track
            Z[track, frame, idx["y"]] = 2 * frame + track
            Z[track, frame, idx["vx"]] = 1.0
            Z[track, frame, idx["vy"]] = 2.0
            if "circularity" in idx:
                Z[track, frame, idx["circularity"]] = 0.9
            if "bbox_w" in idx:
                Z[track, frame, idx["bbox_w"]] = 20.0
                Z[track, frame, idx["bbox_h"]] = 12.0
            if "cfd_u_norm" in idx:
                Z[track, frame, idx["cfd_u_norm"]] = 0.1
                Z[track, frame, idx["cfd_v_norm"]] = 0.2
                Z[track, frame, idx["superficial_velocity"]] = 56.944444
                Z[track, frame, idx["left_flow_fraction"]] = 0.5
                for name in feature_names:
                    if name.startswith("occupancy_"):
                        Z[track, frame, idx[name]] = 1.0 / 6.0
    arrays = {
        "Z": Z,
        "mask": mask,
        "track_ids": np.asarray([10, 20], dtype=np.int64),
        "frames": np.arange(frames, dtype=np.int64),
        "feature_names": np.asarray(feature_names),
    }
    np.savez(path, **arrays)
    return path
