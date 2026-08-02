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


def test_adaptive_fusion_uses_truth_for_rollout_but_loss_uses_raw_prediction(tmp_path: Path) -> None:
    dataset, batch = _v2_four_target_batch(tmp_path)
    model = _ConstantPredictionModel([10.0, 20.0, 31.0, 17.0])
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(1, 2.0, torch.device("cpu"))
    fusion = trainer.AdaptiveTargetFusion(
        horizon=1,
        target_dim=4,
        enabled=True,
        ema_beta=0.0,
        initial_prediction_variance=100.0,
        measurement_variance=1.0e-9,
        min_alpha=0.0,
        max_alpha=1.0,
        device=torch.device("cpu"),
    )

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        adaptive_fusion=fusion,
    )

    idx = dataset.feature_indices
    assert rollout["pred_target"][0, 0, 0, 0].item() == pytest.approx(10.0)
    assert rollout["true_target"][0, 0, 0, 0].item() == pytest.approx(1.0)
    assert rollout["weighted_loss_internal_only"].item() > 0.0
    assert rollout["pred_state"][0, 0, 0, idx["vx"]].item() == pytest.approx(1.0)
    assert rollout["pred_state"][0, 0, 0, idx["bbox_w"]].item() == pytest.approx(20.0)


def test_adaptive_fusion_alpha_decreases_with_lower_prediction_variance() -> None:
    fusion = trainer.AdaptiveTargetFusion(
        horizon=2,
        target_dim=4,
        enabled=True,
        ema_beta=0.0,
        initial_prediction_variance=4.0,
        measurement_variance=4.0,
        min_alpha=0.0,
        max_alpha=0.8,
        device=torch.device("cpu"),
    )
    assert fusion.alpha_tensor()[0, 0].item() == pytest.approx(0.5)

    mse = torch.zeros(2, 4)
    counts = torch.ones(2, 4)
    fusion.update(mse, counts)

    assert fusion.alpha_tensor()[0, 0].item() == pytest.approx(0.0)


def test_zero_adaptive_fusion_forces_pure_prediction_rollout() -> None:
    fusion = trainer.ZeroAdaptiveTargetFusion(horizon=3, target_dim=4, device=torch.device("cpu"))
    assert fusion.enabled is True
    assert torch.equal(fusion.alpha_tensor(), torch.zeros(3, 4))


def test_training_curves_include_step_rmse_and_adaptive_alpha(tmp_path: Path) -> None:
    path = tmp_path / "training_curves.csv"
    train_summary = {
        "weighted_loss_internal_only": 1.0,
        "cfd_valid_target_fraction": 0.5,
        "runtime_step_attempts": 2.0,
        "runtime_step_fallbacks": 0.0,
        "runtime_step_fallback_fraction": 0.0,
        "adaptive_fusion": {
            "alpha_by_step_feature": [[0.1, 0.2, 0.3, 0.4] for _ in range(50)],
            "alpha_by_step_mean": [0.1] * 50,
            "alpha_mean": 0.1,
        },
    }
    val_summary = {
        "weighted_loss_internal_only": 2.0,
        "cfd_valid_target_fraction": 0.6,
        "rmse_vx": 1.0,
        "rmse_vy": 2.0,
        "rmse_speed": 3.0,
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
    assert "val_rmse_position_s50" in lines[0]
    assert "val_pure_rmse_position_s50" in lines[0]
    assert "adaptive_fusion_alpha_s50" in lines[0]
    assert "adaptive_fusion_alpha_bbox_h_s50" in lines[0]
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("val_rmse_position_s50")] == "50.0"
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("val_pure_rmse_position_s50")] == "500.0"
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("adaptive_fusion_alpha_s50")] == "0.1"
    assert lines[1].split(",")[trainer.CURVES_COLUMNS.index("adaptive_fusion_alpha_bbox_h_s50")] == "0.4"


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

    def __call__(self, current_state, model_prediction, context, active_mask=None):
        active = np.asarray(active_mask, dtype=bool)
        self.calls.append(
            {
                "current_state": np.asarray(current_state).copy(),
                "model_prediction": np.asarray(model_prediction).copy(),
                "active_mask": active.copy(),
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
