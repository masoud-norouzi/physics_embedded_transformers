from __future__ import annotations

from pathlib import Path

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


def test_model_accepts_15_dimensional_markovian_state() -> None:
    model = _small_model(input_dim=15, horizon=1, max_droplets=4)
    history_x = torch.randn(2, 1, 4, 15)
    history_mask = torch.ones(2, 1, 4, dtype=torch.bool)
    output = model(history_x, history_mask)
    assert output.shape == (2, 4, 2)


def test_batch_unpacking_and_cfd_loss_mask_from_loader(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "v2.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(npz, start_frames=[0], T_history=1, T_future=3, max_droplets=4)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch["history_x"].shape == (1, 1, 4, 15)
    assert batch["cfd_loss_mask"].shape == (1, 3, 4)
    assert batch["future_mask"][0, 1, 0].item() is True
    assert batch["cfd_loss_mask"][0, 1, 0].item() is False


def test_cfd_loss_mask_controls_supervised_loss(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "v2.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(npz, start_frames=[0], T_history=1, T_future=3, max_droplets=4)
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    model = _small_model(input_dim=15, horizon=1, max_droplets=4)
    stats = _identity_stats(15)
    weights = trainer.rollout_weights(3, 2.0, torch.device("cpu"))
    rollout = trainer.boundary_conditioned_rollout(model, batch, dataset, stats, weights)
    assert rollout["mask"][0, 1, 0].item() is True
    assert rollout["supervision_mask"][0, 1, 0].item() is False


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
    model = _small_model(input_dim=15, horizon=1, max_droplets=4)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "input_dim": 15,
            "target_dim": 2,
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
    model = _small_model(input_dim=15, horizon=1, max_droplets=4)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    stats = _identity_stats(15)
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


def _small_model(input_dim: int, horizon: int, max_droplets: int) -> CanonicalRolloutTransformer:
    return CanonicalRolloutTransformer(
        input_dim=input_dim,
        target_dim=2,
        T_history=horizon,
        max_droplets=max_droplets,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )


def _identity_stats(feature_dim: int) -> dict[str, np.ndarray]:
    return {
        "input_mean": np.zeros(feature_dim, dtype=np.float32),
        "input_std": np.ones(feature_dim, dtype=np.float32),
        "target_mean": np.zeros(2, dtype=np.float32),
        "target_std": np.ones(2, dtype=np.float32),
    }


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
            Z[track, frame, idx["circularity"]] = 0.9
            if "cfd_valid" in idx:
                Z[track, frame, idx["cfd_u"]] = 0.1
                Z[track, frame, idx["cfd_v"]] = 0.2
                Z[track, frame, idx["left_flow_fraction"]] = 0.5
                for name in feature_names:
                    if name.startswith("occupancy_"):
                        Z[track, frame, idx[name]] = 1.0 / 6.0
                Z[track, frame, idx["cfd_valid"]] = 1.0
    if "cfd_valid" in idx:
        Z[0, 2, idx["cfd_valid"]] = 0.0
    np.savez(
        path,
        Z=Z,
        mask=mask,
        track_ids=np.asarray([10, 20], dtype=np.int64),
        frames=np.arange(frames, dtype=np.int64),
        feature_names=np.asarray(feature_names),
    )
    return path
