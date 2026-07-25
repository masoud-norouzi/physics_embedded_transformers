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
