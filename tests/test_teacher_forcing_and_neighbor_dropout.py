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


def _model(seed: int = 0, **overrides) -> CanonicalRolloutTransformer:
    torch.manual_seed(seed)
    kwargs = dict(
        input_dim=16,
        target_dim=4,
        T_history=1,
        max_droplets=4,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return CanonicalRolloutTransformer(**kwargs)


def _history(batch=1, T=1, M=3, F=16, seed: int = 0):
    torch.manual_seed(seed)
    history_x = torch.randn(batch, T, M, F)
    history_mask = torch.ones(batch, T, M, dtype=torch.bool)
    return history_x, history_mask


# ---------------------------------------------------------------------------
# Model: key_visibility_mask
# ---------------------------------------------------------------------------


def test_key_visibility_mask_none_matches_omitted_kwarg() -> None:
    model = _model(seed=1, max_droplets=3)
    history_x, history_mask = _history()
    with_none = model(history_x, history_mask, key_visibility_mask=None)
    omitted = model(history_x, history_mask)
    assert torch.equal(with_none, omitted)


def test_key_visibility_mask_hidden_droplet_is_invisible_but_still_a_query() -> None:
    model = _model(seed=2, max_droplets=3)
    history_x, history_mask = _history(M=3)

    full_visibility = model(history_x, history_mask)

    key_mask = torch.tensor([[[True, True, False]]])  # slot 2 hidden as a key
    key_masked = model(history_x, history_mask, key_visibility_mask=key_mask)

    absent_mask = torch.tensor([[[True, True, False]]])
    absent_history_x = history_x.clone()
    absent_history_x[:, :, 2, :] = 0.0
    absent = model(absent_history_x, absent_mask)

    # Slots 0/1 only ever see {0,1} as valid keys in both key_masked and absent -- their query
    # embeddings are unchanged (their own history_mask/features are identical), so their output
    # must match exactly regardless of whether slot 2 is "key-hidden" or genuinely absent.
    assert torch.allclose(key_masked[:, :2, :], absent[:, :2, :], atol=1e-6)

    # Slot 2 itself is a REAL query in key_masked (real features, present per history_mask) but
    # not in absent (zeroed features, absent per history_mask) -- its own output must differ.
    assert not torch.allclose(key_masked[:, 2, :], absent[:, 2, :], atol=1e-6)

    # And key_masked's slot 2 must also differ from full_visibility's slot 2: in full_visibility
    # slot 2 can attend to itself as a key; in key_masked it cannot.
    assert not torch.allclose(key_masked[:, 2, :], full_visibility[:, 2, :], atol=1e-6)


def test_key_visibility_mask_degenerate_single_droplet_falls_back_to_full_visibility() -> None:
    model = _model(seed=3, max_droplets=1, predict_branch_decision=True)
    history_x, history_mask = _history(M=1)  # the only droplet in the window

    key_mask = torch.zeros(1, 1, 1, dtype=torch.bool)  # would hide the only droplet as its own key
    masked_output = model(history_x, history_mask, key_visibility_mask=key_mask, return_decision=True)
    unmasked_output = model(history_x, history_mask, return_decision=True)

    assert torch.isfinite(masked_output["prediction"]).all()
    assert torch.isfinite(masked_output["decision_logit"]).all()
    assert torch.allclose(masked_output["prediction"], unmasked_output["prediction"], atol=1e-6)
    assert torch.allclose(masked_output["decision_logit"], unmasked_output["decision_logit"], atol=1e-6)


def test_key_visibility_mask_degenerate_case_is_per_batch_row_not_global() -> None:
    # Row 0: two droplets present, drop droplet 1 as key -- should NOT hit the fallback.
    # Row 1: one droplet present, drop it as key -- SHOULD hit the fallback for that row only.
    model = _model(seed=4, max_droplets=3)
    history_x = torch.randn(2, 1, 3, 16)
    history_mask = torch.tensor([[[True, True, False]], [[True, False, False]]])
    key_mask = torch.tensor([[[True, False, False]], [[False, False, False]]])

    output = model(history_x, history_mask, key_visibility_mask=key_mask)
    assert torch.isfinite(output).all()

    unmasked_row1 = model(history_x[1:2], history_mask[1:2])
    masked_row1 = model(history_x[1:2], history_mask[1:2], key_visibility_mask=key_mask[1:2])
    assert torch.allclose(masked_row1, unmasked_row1, atol=1e-6)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_literal_teacher_forcing_enabled_reads_config() -> None:
    assert trainer.literal_teacher_forcing_enabled({}) is False
    assert trainer.literal_teacher_forcing_enabled({"training": {"literal_teacher_forcing": {"enabled": False}}}) is False
    assert trainer.literal_teacher_forcing_enabled({"training": {"literal_teacher_forcing": {"enabled": True}}}) is True


def test_neighbor_key_dropout_probability_from_config() -> None:
    assert trainer.neighbor_key_dropout_probability_from_config({}) == 0.0
    assert (
        trainer.neighbor_key_dropout_probability_from_config(
            {"training": {"neighbor_key_dropout": {"enabled": False, "probability": 0.9}}}
        )
        == 0.0
    )
    assert (
        trainer.neighbor_key_dropout_probability_from_config(
            {"training": {"neighbor_key_dropout": {"enabled": True, "probability": 0.3}}}
        )
        == 0.3
    )
    with pytest.raises(ValueError, match="probability"):
        trainer.neighbor_key_dropout_probability_from_config(
            {"training": {"neighbor_key_dropout": {"enabled": True, "probability": 1.5}}}
        )


def test_sample_neighbor_key_dropout_mask_edge_cases() -> None:
    device = torch.device("cpu")
    assert trainer.sample_neighbor_key_dropout_mask(4, 8, 0.0, device) is None

    all_hidden = trainer.sample_neighbor_key_dropout_mask(4, 8, 1.0, device)
    assert all_hidden is not None
    assert all_hidden.shape == (4, 8)
    assert not bool(all_hidden.any())

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    mid = trainer.sample_neighbor_key_dropout_mask(64, 64, 0.5, device, generator=generator)
    visible_fraction = float(mid.float().mean())
    assert 0.35 < visible_fraction < 0.65  # ~50% visible, loose statistical tolerance


# ---------------------------------------------------------------------------
# boundary_conditioned_rollout: literal_teacher_forcing + key_visibility_mask
# ---------------------------------------------------------------------------


def _write_npz(path: Path, feature_names: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = 3
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
        "track_ids": np.asarray([10, 20, 30], dtype=np.int64),
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


def _rollout_batch(tmp_path: Path, max_droplets: int = 4):
    npz = _write_npz(tmp_path / "teacher_forcing.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=1,
        T_future=3,
        max_droplets=max_droplets,
        target_features=trainer.RUNTIME_TARGET_FEATURES,
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    return dataset, batch


def test_literal_teacher_forcing_pred_position_matches_true_position_exactly(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=5)  # untrained/random -- predictions are garbage on purpose

    rollout = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, runtime_context=None, literal_teacher_forcing=True
    )

    mask = rollout["mask"]
    assert torch.allclose(rollout["pred_position"][mask], rollout["true_position"][mask], atol=1e-6), (
        "literal_teacher_forcing must drive the next-step history from ground truth regardless "
        "of what the (garbage, untrained) model predicted"
    )


def test_non_teacher_forced_rollout_pred_position_diverges_from_truth(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=5)

    rollout = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, runtime_context=None, p_truth=0.0, literal_teacher_forcing=False
    )

    mask = rollout["mask"]
    assert not torch.allclose(rollout["pred_position"][mask], rollout["true_position"][mask], atol=1e-6), (
        "sanity check that the two rollout modes actually differ -- self-conditioned rollout with "
        "an untrained model should not land on the exact true position"
    )


def test_key_visibility_mask_changes_rollout_output(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path, max_droplets=4)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=6)

    baseline = trainer.boundary_conditioned_rollout(model, batch, dataset, stats, weights, runtime_context=None)

    hide_all_but_first = torch.zeros(batch["history_mask"].shape[0], batch["history_mask"].shape[2], dtype=torch.bool)
    hide_all_but_first[:, 0] = True
    masked = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, runtime_context=None, key_visibility_mask=hide_all_but_first
    )

    assert not torch.allclose(baseline["pred_target"], masked["pred_target"], atol=1e-6), (
        "key_visibility_mask should actually reach the model and change its predictions"
    )


def test_literal_teacher_forcing_falls_back_to_last_frame_for_nan_non_target_feature(tmp_path: Path) -> None:
    # future_mask only guarantees the TARGET dims (vx, vy, bbox_w, bbox_h) are finite -- a
    # present droplet's OTHER columns (e.g. cfd_u_norm) can still be NaN in the raw data. This
    # regression-tests the fix: literal_teacher_forcing must not blindly copy a NaN non-target
    # value into the next-step history (which poisons every subsequent step via the transformer).
    npz = _write_npz(tmp_path / "nan_non_target.npz", V2_FEATURES)
    idx = {name: i for i, name in enumerate(V2_FEATURES)}
    with np.load(npz) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    arrays["Z"][0, 1, idx["cfd_u_norm"]] = np.nan  # track 0, frame 1 (the first predicted step)
    np.savez(npz, **arrays)

    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=1,
        T_future=3,
        max_droplets=4,
        target_features=trainer.RUNTIME_TARGET_FEATURES,
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=8)

    rollout = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, runtime_context=None, literal_teacher_forcing=True
    )

    assert torch.isfinite(rollout["weighted_loss_internal_only"])
    assert torch.isfinite(rollout["total_loss"])
    assert torch.isfinite(rollout["pred_target_norm"]).all()
    assert torch.isfinite(rollout["pred_position"]).all()


def test_target_parameterization_from_config_accepts_velocity_only_targets() -> None:
    config = {"model": {"target_features": ["vx", "vy"]}}
    assert trainer.target_parameterization_from_config(config) == {"mode": "velocity"}


def test_validate_feature_contract_velocity_only_requires_literal_teacher_forcing(tmp_path: Path) -> None:
    dataset, _ = _rollout_batch(tmp_path)  # dataset.feature_names == V2_FEATURES == FEATURE_NAMES
    config = {
        "model": {
            "input_feature_names": V2_FEATURES,
            "input_dim": len(V2_FEATURES),
            "target_features": ["vx", "vy"],
        }
    }
    # Without literal_teacher_forcing, bbox-less targets hit the "closed-loop physics rollout"
    # contract check (that path needs a predicted bbox to compute occupancy) and must be rejected.
    with pytest.raises(ValueError, match="Closed-loop physics rollout requires"):
        trainer.validate_feature_contract(dataset, config, literal_teacher_forcing=False)
    # Under literal_teacher_forcing, that contract doesn't apply -- bbox-less targets are fine.
    trainer.validate_feature_contract(dataset, config, literal_teacher_forcing=True)

    with pytest.raises(ValueError, match="Unsupported target_features under literal_teacher_forcing"):
        trainer.validate_feature_contract(
            dataset,
            {"model": {**config["model"], "target_features": ["vx"]}},
            literal_teacher_forcing=True,
        )


def test_boundary_conditioned_rollout_velocity_only_targets_no_bbox_crash(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "no_bbox_target.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=1,
        T_future=3,
        max_droplets=4,
        target_features=("vx", "vy"),
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    stats = _identity_stats(16, target_dim=2)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=9, target_dim=2, bbox_stop_gradient=False)

    rollout = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, runtime_context=None, literal_teacher_forcing=True
    )
    assert rollout["pred_target"].shape[-1] == 2
    assert torch.isfinite(rollout["weighted_loss_internal_only"])

    # update_metric_accumulators/update_one_accumulator must not crash on a bbox-less target, and
    # must report bbox RMSE as NaN (not 0.0, which would misleadingly imply perfect prediction)
    # rather than silently reusing the vx/vy sample count as the bbox denominator.
    accumulators = trainer.create_accumulators(int(weights.numel()))
    trainer.update_metric_accumulators(accumulators, rollout)
    metrics = trainer.metrics_from_accumulator(accumulators["overall"])
    assert np.isnan(metrics["rmse_bbox_w"])
    assert np.isnan(metrics["rmse_bbox_h"])
    assert np.isfinite(metrics["rmse_vx"])
    assert np.isfinite(metrics["rmse_vy"])


def test_literal_teacher_forcing_skips_runtime_step_even_if_runtime_context_given(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=7)

    class ExplodingRuntimeContext:
        """Any attribute access means the (unused, physics-bypassed) runtime path was reached."""

        def __getattr__(self, name):
            raise AssertionError(f"runtime_context.{name} was accessed -- literal_teacher_forcing should bypass it")

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=ExplodingRuntimeContext(),
        literal_teacher_forcing=True,
    )
    assert torch.isfinite(rollout["weighted_loss_internal_only"])
    assert rollout["runtime_step_attempts"] == 0
    assert rollout["runtime_step_fallbacks"] == 0
