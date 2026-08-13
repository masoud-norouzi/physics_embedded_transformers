from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from scripts.training import train_physics_markovian as trainer
from src.datasets.canonical_window_dataset import CanonicalWindowDataset
from src.models.canonical_rollout_transformer import CanonicalRolloutTransformer


V2_FEATURES = trainer.FEATURE_NAMES


def _model(bbox_stop_gradient: bool, target_dim: int = 4, seed: int = 0) -> CanonicalRolloutTransformer:
    torch.manual_seed(seed)
    return CanonicalRolloutTransformer(
        input_dim=16,
        target_dim=target_dim,
        T_history=1,
        max_droplets=4,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        bbox_stop_gradient=bbox_stop_gradient,
    )


def _history(batch=2, T=1, M=4, F=16):
    history_x = torch.randn(batch, T, M, F)
    history_mask = torch.ones(batch, T, M, dtype=torch.bool)
    return history_x, history_mask


# ---------------------------------------------------------------------------
# Construction / architecture
# ---------------------------------------------------------------------------


def test_bbox_stop_gradient_requires_target_dim_greater_than_2() -> None:
    with pytest.raises(ValueError, match="target_dim"):
        _model(bbox_stop_gradient=True, target_dim=2)


def test_bbox_stop_gradient_false_preserves_single_velocity_head() -> None:
    model = _model(bbox_stop_gradient=False)
    assert model.velocity_head is not None
    assert model.motion_head is None
    assert model.bbox_head is None
    assert model.velocity_head[-1].out_features == 4


def test_bbox_stop_gradient_true_splits_into_motion_and_bbox_heads() -> None:
    model = _model(bbox_stop_gradient=True, target_dim=4)
    assert model.velocity_head is None
    assert model.motion_head is not None
    assert model.bbox_head is not None
    assert model.motion_head[-1].out_features == 2  # target_dim - 2
    assert model.bbox_head[-1].out_features == 2


# ---------------------------------------------------------------------------
# Forward shape / backward compatibility
# ---------------------------------------------------------------------------


def test_bbox_stop_gradient_forward_shape_matches_plain_head() -> None:
    split_model = _model(bbox_stop_gradient=True, seed=3)
    plain_model = _model(bbox_stop_gradient=False, seed=3)
    history_x, history_mask = _history()

    split_prediction = split_model(history_x, history_mask)
    plain_prediction = plain_model(history_x, history_mask)

    assert torch.is_tensor(split_prediction)
    assert split_prediction.shape == plain_prediction.shape == (2, 4, 4)


# ---------------------------------------------------------------------------
# Gradient isolation -- the actual point of bbox_stop_gradient
# ---------------------------------------------------------------------------


def test_bbox_head_gradient_never_reaches_trunk() -> None:
    model = _model(bbox_stop_gradient=True)
    history_x, history_mask = _history()
    prediction = model(history_x, history_mask)
    bbox_prediction = prediction[..., 2:]  # bbox_w, bbox_h

    bbox_prediction.sum().backward()

    # NOTE: prediction = cat([motion_prediction, bbox_prediction]), and cat's backward always
    # propagates a (here all-zero, since we sliced) gradient into the OTHER branch too -- so
    # non-bbox params end up with an explicit zero tensor, not None. "None or exactly zero" is
    # the correct check for "zero contribution," not strict identity-is-None.
    for name, param in model.named_parameters():
        if name.startswith("bbox_head"):
            assert param.grad is not None and float(param.grad.abs().sum()) > 0.0, (
                f"{name} should receive gradient from the bbox loss"
            )
        else:
            zero_contribution = param.grad is None or torch.allclose(
                param.grad, torch.zeros_like(param.grad), atol=1e-8
            )
            assert zero_contribution, f"{name} should receive ZERO gradient -- bbox_head reads h_last.detach()"


def test_motion_head_gradient_still_reaches_trunk() -> None:
    model = _model(bbox_stop_gradient=True)
    history_x, history_mask = _history()
    prediction = model(history_x, history_mask)
    motion_prediction = prediction[..., :2]  # vx, vy

    motion_prediction.sum().backward()

    trunk_grad_present = any(
        param.grad is not None
        for name, param in model.named_parameters()
        if not name.startswith("motion_head") and not name.startswith("bbox_head")
    )
    assert trunk_grad_present, "motion_head must still train the trunk normally"
    for name, param in model.named_parameters():
        if name.startswith("bbox_head"):
            zero_contribution = param.grad is None or torch.allclose(
                param.grad, torch.zeros_like(param.grad), atol=1e-8
            )
            assert zero_contribution, "bbox_head must not receive gradient from the motion loss"


# ---------------------------------------------------------------------------
# End-to-end: garbage bbox labels must not change trunk gradients
# ---------------------------------------------------------------------------


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


def _identity_stats(feature_dim: int, target_dim: int) -> dict[str, np.ndarray]:
    return {
        "input_mean": np.zeros(feature_dim, dtype=np.float32),
        "input_std": np.ones(feature_dim, dtype=np.float32),
        "target_mean": np.zeros(target_dim, dtype=np.float32),
        "target_std": np.ones(target_dim, dtype=np.float32),
    }


def _rollout_batch(tmp_path: Path):
    npz = _write_npz(tmp_path / "bbox_isolation.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=1,
        T_future=2,
        max_droplets=4,
        target_features=trainer.RUNTIME_TARGET_FEATURES,  # (vx, vy, bbox_w, bbox_h)
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    return dataset, batch


def test_garbage_bbox_labels_do_not_change_trunk_gradient(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(2, 2.0, torch.device("cpu"))

    model_clean = _model(bbox_stop_gradient=True, seed=9)
    rollout_clean = trainer.boundary_conditioned_rollout(
        model_clean, batch, dataset, stats, weights, runtime_context=None
    )
    rollout_clean["weighted_loss_internal_only"].backward()
    trunk_grads_clean = {
        name: param.grad.clone()
        for name, param in model_clean.named_parameters()
        if not name.startswith("motion_head") and not name.startswith("bbox_head")
    }

    corrupted_batch = dict(batch)
    corrupted_batch["future_y"] = batch["future_y"].clone()
    corrupted_batch["future_y"][..., 2:] = 1.0e6  # garbage bbox_w/bbox_h target -- vx,vy untouched

    model_corrupted = _model(bbox_stop_gradient=True, seed=9)  # same seed -> bit-identical init
    rollout_corrupted = trainer.boundary_conditioned_rollout(
        model_corrupted, corrupted_batch, dataset, stats, weights, runtime_context=None
    )
    rollout_corrupted["weighted_loss_internal_only"].backward()
    trunk_grads_corrupted = {
        name: param.grad
        for name, param in model_corrupted.named_parameters()
        if not name.startswith("motion_head") and not name.startswith("bbox_head")
    }

    assert set(trunk_grads_clean.keys()) == set(trunk_grads_corrupted.keys())
    for name in trunk_grads_clean:
        assert torch.allclose(trunk_grads_clean[name], trunk_grads_corrupted[name], atol=1e-6), (
            f"{name} trunk gradient changed when bbox labels were corrupted to garbage -- "
            "bbox_stop_gradient is not actually isolating the trunk"
        )
