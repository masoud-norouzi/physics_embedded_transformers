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
# Model: key_visibility_mask (kept from the removed literal-teacher-forcing track, unchanged)
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


def test_state_injected_rollout_enabled_reads_config() -> None:
    assert trainer.state_injected_rollout_enabled({}) is False
    assert trainer.state_injected_rollout_enabled({"training": {"state_injected_rollout": {"enabled": False}}}) is False
    assert trainer.state_injected_rollout_enabled({"training": {"state_injected_rollout": {"enabled": True}}}) is True


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


def test_is_ordered_subsequence() -> None:
    full = ["x", "y", "vx", "vy", "bbox_w", "bbox_h", "cfd_u_norm"]
    assert trainer.is_ordered_subsequence(full, full)
    assert trainer.is_ordered_subsequence(["x", "y", "vx", "vy", "cfd_u_norm"], full)  # bbox dropped
    assert not trainer.is_ordered_subsequence(["vx", "x"], full)  # wrong order
    assert not trainer.is_ordered_subsequence(["x", "y", "made_up_feature"], full)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _write_npz(path: Path, feature_names: list[str], vx_value: float = 1.0, vy_value: float = 2.0) -> Path:
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
            Z[track, frame, idx["vx"]] = vx_value
            Z[track, frame, idx["vy"]] = vy_value
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


def _rollout_batch(
    tmp_path: Path,
    max_droplets: int = 4,
    name: str = "state_injected.npz",
    target_features=trainer.RUNTIME_TARGET_FEATURES,
    **npz_kwargs,
):
    npz = _write_npz(tmp_path / name, V2_FEATURES, **npz_kwargs)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=1,
        T_future=3,
        max_droplets=max_droplets,
        target_features=target_features,
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    return dataset, batch


# ---------------------------------------------------------------------------
# boundary_conditioned_rollout: state_injected_rollout
# ---------------------------------------------------------------------------


def test_state_injected_rollout_requires_typical_inlet_velocity(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=5)

    with pytest.raises(ValueError, match="typical_inlet_velocity"):
        trainer.boundary_conditioned_rollout(
            model, batch, dataset, stats, weights, runtime_context=None, state_injected_rollout=True
        )


def test_state_injected_rollout_position_diverges_from_truth_with_garbage_model(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=5)  # untrained/random -- predictions are garbage on purpose
    typical_inlet_velocity = torch.tensor([0.0, 0.0])

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )

    mask = rollout["mask"]
    assert not torch.allclose(rollout["pred_position"][mask], rollout["true_position"][mask], atol=1e-6), (
        "position is now integrated from the model's own (garbage, untrained) velocity prediction, "
        "not injected from truth -- it should NOT land on the exact true position"
    )


def test_state_injected_rollout_position_is_a_real_euler_integration(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path, target_features=("vx", "vy"))
    stats = _identity_stats(16, target_dim=2)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    feature_index = dataset.feature_indices
    typical_inlet_velocity = torch.tensor([0.0, 0.0])

    # A model that always predicts zero velocity: Euler integration of zero is a no-op, so
    # position must stay exactly at its initial (true) value for every step.
    zero_model = _EchoVelocityModel(feature_index["vx"], feature_index["vy"], scale=0.0)
    still_rollout = trainer.boundary_conditioned_rollout(
        zero_model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )
    mask = still_rollout["mask"]
    still_position = still_rollout["pred_position"]
    for step in range(1, 3):
        assert torch.allclose(
            still_position[:, step, :, :][mask[:, step, :]], still_position[:, 0, :, :][mask[:, 0, :]], atol=1e-4
        ), "zero predicted velocity every step must leave position unchanged (Euler integration of zero is a no-op)"

    # A model that always echoes back the (nonzero) input velocity unchanged: position must move
    # by a real, nonzero amount each step -- proof this is genuine integration, not another
    # disguised form of injection.
    moving_model = _EchoVelocityModel(feature_index["vx"], feature_index["vy"], scale=1.0)
    moving_rollout = trainer.boundary_conditioned_rollout(
        moving_model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )
    mask = moving_rollout["mask"]
    moving_position = moving_rollout["pred_position"]
    assert not torch.allclose(
        moving_position[:, 1, :, :][mask[:, 1, :]], moving_position[:, 0, :, :][mask[:, 0, :]], atol=1e-4
    ), "nonzero predicted velocity must actually displace the integrated position step to step"


class _EchoVelocityModel(torch.nn.Module):
    """Deterministic stand-in: predicts scale * (this step's vx, vy input), ignoring everything
    else. Lets a test hand-compute the exact expected prediction at every rollout step and verify
    state_injected_rollout is really feeding the model's OWN prior prediction back in as the next
    step's vx,vy input (not the true value)."""

    def __init__(self, vx_index: int, vy_index: int, scale: float = 2.0) -> None:
        super().__init__()
        self.vx_index = vx_index
        self.vy_index = vy_index
        self.scale = scale
        self.seen_history = []

    def forward(self, history_x, history_mask, key_visibility_mask=None):
        self.seen_history.append(history_x.detach().clone())
        vx = history_x[:, -1, :, self.vx_index]
        vy = history_x[:, -1, :, self.vy_index]
        return torch.stack([vx, vy], dim=-1) * self.scale


def test_state_injected_rollout_continuing_droplet_uses_own_prediction_for_velocity(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path, vx_value=1.0, vy_value=2.0, target_features=("vx", "vy"))
    stats = _identity_stats(16, target_dim=2)  # identity stats -> normalized == physical, echo model reads it directly
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    feature_index = dataset.feature_indices
    model = _EchoVelocityModel(feature_index["vx"], feature_index["vy"], scale=2.0)
    typical_inlet_velocity = torch.tensor([0.0, 0.0])

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )

    # All 3 tracks are present in every frame of this synthetic window (continuing throughout).
    # Step 1 input vx,vy is the true initial value (1.0, 2.0) -> prediction = (2.0, 4.0).
    # Step 2 input vx,vy is step 1's OWN prediction (2.0, 4.0), NOT the true (1.0, 2.0) again
    #   -> prediction = (4.0, 8.0).
    # Step 3 input vx,vy is step 2's OWN prediction (4.0, 8.0) -> prediction = (8.0, 16.0).
    pred = rollout["pred_target"]  # (B, horizon, M, 2), physical units (identity stats)
    mask = rollout["mask"]
    for step_index, expected in enumerate([(2.0, 4.0), (4.0, 8.0), (8.0, 16.0)]):
        step_pred = pred[:, step_index, :, :][mask[:, step_index, :]]
        expected_tensor = torch.tensor(expected).expand_as(step_pred)
        assert torch.allclose(step_pred, expected_tensor, atol=1e-4), (
            f"step {step_index}: expected {expected} (own prediction fed forward), got {step_pred}"
        )


def test_state_injected_rollout_new_entry_seeded_from_typical_inlet_velocity(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "entry.npz", V2_FEATURES, vx_value=99.0, vy_value=99.0)
    idx = {name: i for i, name in enumerate(V2_FEATURES)}
    with np.load(npz) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    # Track 0 is absent from the initial history frame (0) but present from frame 1 onward --
    # it "enters" the window at the first rollout step. Its true vx,vy (99, 99) is deliberately
    # distinguishable from typical_inlet_velocity so the test can tell which one was actually used.
    arrays["mask"][0, 0] = False
    arrays["Z"][0, 0, :] = np.nan
    np.savez(npz, **arrays)

    dataset = CanonicalWindowDataset(
        npz, start_frames=[0], T_history=1, T_future=3, max_droplets=4, target_features=("vx", "vy")
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    stats = _identity_stats(16, target_dim=2)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    feature_index = dataset.feature_indices
    model = _EchoVelocityModel(feature_index["vx"], feature_index["vy"], scale=2.0)
    typical_inlet_velocity = torch.tensor([5.0, -5.0])

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )

    droplet_ids = batch["droplet_ids"][0]
    track0_slot = int((droplet_ids == 10).nonzero(as_tuple=True)[0].item())  # track_ids=[10,20,30]

    # Step 0: track 0 is entering (boundary_mask) -- its OWN "prediction" at step 0 (row_from_state
    # for the entry point) isn't from the echo model at all (it's seeded directly), so step 1's
    # prediction is the first one driven by the echo model reading the seeded velocity.
    # Step 1 input vx,vy for track 0 = typical_inlet_velocity (5, -5), NOT its true (99, 99)
    #   -> step 1 prediction = echo(5, -5) = (10, -10).
    step1_pred = rollout["pred_target"][0, 1, track0_slot, :]
    assert torch.allclose(step1_pred, torch.tensor([10.0, -10.0]), atol=1e-4), (
        f"expected step 1 prediction to reflect typical_inlet_velocity=(5,-5) seeded at entry, got {step1_pred} "
        "-- if this is (198, -198) instead, the true (leaked) vx,vy was used at entry"
    )


class _ConstantVelocityModel(torch.nn.Module):
    def __init__(self, vx: float, vy: float) -> None:
        super().__init__()
        self.vx = vx
        self.vy = vy

    def forward(self, history_x, history_mask, key_visibility_mask=None):
        B, T, M, _ = history_x.shape
        return torch.tensor([self.vx, self.vy]).view(1, 1, 2).expand(B, M, 2).clone()


def test_state_injected_rollout_hard_wall_containment_uses_true_bbox_not_predicted(tmp_path: Path) -> None:
    # The model's predicted target is (vx, vy) only -- 2 dims, no bbox -- so
    # build_stale_refresh_frame's ORIGINAL clamp path (which needs bbox in dims 2:4 of the
    # prediction) could never engage. This proves the NEW clamp_bbox_phys path lets
    # hard_wall_containment still constrain the integrated position using the droplet's real
    # (ground-truth) bbox, without the model ever seeing or predicting it.
    from src.physics.constraints import wall_sdf_to_torch
    from src.physics.geometry.wall_sdf import build_wall_sdf

    idx = {name: i for i, name in enumerate(V2_FEATURES)}
    Z = np.full((1, 2, len(V2_FEATURES)), np.nan, dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    for frame in range(2):
        Z[0, frame, idx["x"]] = 10.0
        Z[0, frame, idx["y"]] = 10.0
        Z[0, frame, idx["vx"]] = 1.0
        Z[0, frame, idx["vy"]] = 0.0
        Z[0, frame, idx["bbox_w"]] = 4.0
        Z[0, frame, idx["bbox_h"]] = 4.0
        Z[0, frame, idx["cfd_u_norm"]] = 0.1
        Z[0, frame, idx["cfd_v_norm"]] = 0.2
        Z[0, frame, idx["superficial_velocity"]] = 56.9
        Z[0, frame, idx["left_flow_fraction"]] = 0.5
        for name in V2_FEATURES:
            if name.startswith("occupancy_"):
                Z[0, frame, idx[name]] = 1.0 / 6.0
    npz_path = tmp_path / "wall_clamp.npz"
    np.savez(
        npz_path,
        Z=Z,
        mask=mask,
        track_ids=np.asarray([1], dtype=np.int64),
        frames=np.arange(2, dtype=np.int64),
        feature_names=np.asarray(V2_FEATURES),
    )
    dataset = CanonicalWindowDataset(
        npz_path, start_frames=[0], T_history=1, T_future=1, max_droplets=1, target_features=("vx", "vy")
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    stats = _identity_stats(16, target_dim=2)
    weights = trainer.rollout_weights(1, 0.0, torch.device("cpu"))
    typical_inlet_velocity = torch.tensor([0.0, 0.0])

    channel_mask = np.ones((20, 30), dtype=bool)
    channel_mask[:, 20:] = False  # wall starts at column (x) = 20
    sdf, grad_x, grad_y = wall_sdf_to_torch(build_wall_sdf(channel_mask))
    hard_wall_containment = trainer.HardWallContainment(enabled=True, sdf=sdf, grad_x=grad_x, grad_y=grad_y)

    model = _ConstantVelocityModel(vx=15.0, vy=0.0)  # would push x from 10 -> 25, past the wall at 20

    common = dict(
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )
    clamped = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, hard_wall_containment=hard_wall_containment, **common
    )
    unclamped = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, hard_wall_containment=None, **common
    )

    clamped_x = clamped["pred_position"][0, 0, 0, 0].item()
    unclamped_x = unclamped["pred_position"][0, 0, 0, 0].item()
    assert unclamped_x > 20.0, f"sanity check: without containment the droplet should cross the wall, got x={unclamped_x}"
    assert clamped_x < unclamped_x - 1.0, (
        "hard_wall_containment should pull the integrated position back using the TRUE bbox (even "
        f"though the model never predicts bbox) -- clamped x={clamped_x}, unclamped x={unclamped_x}"
    )


def test_state_injected_rollout_key_visibility_mask_still_changes_output(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path, max_droplets=4)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=6)
    typical_inlet_velocity = torch.tensor([0.0, 0.0])

    common = dict(
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )
    baseline = trainer.boundary_conditioned_rollout(model, batch, dataset, stats, weights, **common)

    hide_all_but_first = torch.zeros(batch["history_mask"].shape[0], batch["history_mask"].shape[2], dtype=torch.bool)
    hide_all_but_first[:, 0] = True
    masked = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, key_visibility_mask=hide_all_but_first, **common
    )

    assert not torch.allclose(baseline["pred_target"], masked["pred_target"], atol=1e-6), (
        "key_visibility_mask should still reach the model and change its predictions under state_injected_rollout"
    )


def test_state_injected_rollout_falls_back_to_last_frame_for_nan_non_target_feature(tmp_path: Path) -> None:
    # future_mask only guarantees vx,vy (the targets) are finite -- a present droplet's other
    # columns (e.g. cfd_u_norm) can still be NaN in the raw data. Regression test for the fix:
    # state_injected_rollout must not blindly copy a NaN non-target value into next-step history.
    npz = _write_npz(tmp_path / "nan_non_target.npz", V2_FEATURES)
    idx = {name: i for i, name in enumerate(V2_FEATURES)}
    with np.load(npz) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    arrays["Z"][0, 1, idx["cfd_u_norm"]] = np.nan  # track 0, frame 1 (the first predicted step)
    np.savez(npz, **arrays)

    dataset = CanonicalWindowDataset(
        npz, start_frames=[0], T_history=1, T_future=3, max_droplets=4, target_features=trainer.RUNTIME_TARGET_FEATURES
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=8)
    typical_inlet_velocity = torch.tensor([0.0, 0.0])

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )

    assert torch.isfinite(rollout["weighted_loss_internal_only"])
    assert torch.isfinite(rollout["total_loss"])
    assert torch.isfinite(rollout["pred_target_norm"]).all()
    assert torch.isfinite(rollout["pred_position"]).all()


def test_state_injected_rollout_skips_runtime_step_even_if_runtime_context_given(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=7)
    typical_inlet_velocity = torch.tensor([0.0, 0.0])

    class ExplodingRuntimeContext:
        """Any attribute access means the (unused, physics-bypassed) runtime path was reached."""

        def __getattr__(self, name):
            raise AssertionError(f"runtime_context.{name} was accessed -- state_injected_rollout should bypass it")

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=ExplodingRuntimeContext(),
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
    )
    assert torch.isfinite(rollout["weighted_loss_internal_only"])
    assert rollout["runtime_step_attempts"] == 0
    assert rollout["runtime_step_fallbacks"] == 0


def test_non_state_injected_rollout_position_diverges_from_truth(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=5)

    rollout = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, runtime_context=None, p_truth=0.0, state_injected_rollout=False
    )

    mask = rollout["mask"]
    assert not torch.allclose(rollout["pred_position"][mask], rollout["true_position"][mask], atol=1e-6), (
        "sanity check that the two rollout modes actually differ -- self-conditioned rollout with "
        "an untrained model should not land on the exact true position"
    )


# ---------------------------------------------------------------------------
# Reduced model input (bbox dropped from input_feature_names)
# ---------------------------------------------------------------------------


class _ShapeAssertingModel(torch.nn.Module):
    """Stand-in that asserts the actual input width it receives, then returns zeros."""

    def __init__(self, expected_width: int, target_dim: int = 2) -> None:
        super().__init__()
        self.expected_width = expected_width
        self.target_dim = target_dim

    def forward(self, history_x, history_mask, key_visibility_mask=None):
        assert history_x.shape[-1] == self.expected_width, (
            f"expected model input width {self.expected_width}, got {history_x.shape[-1]}"
        )
        B, T, M, _ = history_x.shape
        return torch.zeros(B, M, self.target_dim)


def test_model_input_column_indices_reduces_width_at_model_boundary(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path, target_features=("vx", "vy"))
    stats = _identity_stats(16, target_dim=2)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    typical_inlet_velocity = torch.tensor([0.0, 0.0])
    bbox_indices = {dataset.feature_indices["bbox_w"], dataset.feature_indices["bbox_h"]}
    keep_indices = torch.tensor([i for i in range(16) if i not in bbox_indices], dtype=torch.long)
    model = _ShapeAssertingModel(expected_width=14)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
        model_input_column_indices=keep_indices,
    )
    assert torch.isfinite(rollout["weighted_loss_internal_only"])


def test_model_input_column_indices_none_keeps_full_width(tmp_path: Path) -> None:
    dataset, batch = _rollout_batch(tmp_path, target_features=("vx", "vy"))
    stats = _identity_stats(16, target_dim=2)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    typical_inlet_velocity = torch.tensor([0.0, 0.0])
    model = _ShapeAssertingModel(expected_width=16)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
        model_input_column_indices=None,
    )
    assert torch.isfinite(rollout["weighted_loss_internal_only"])


# ---------------------------------------------------------------------------
# validate_feature_contract: ordered-subset input_feature_names under state_injected_rollout
# ---------------------------------------------------------------------------


def test_validate_feature_contract_allows_dropped_bbox_input_under_state_injected_rollout(tmp_path: Path) -> None:
    dataset, _ = _rollout_batch(tmp_path)  # dataset.feature_names == V2_FEATURES == FEATURE_NAMES
    reduced_input_features = [name for name in V2_FEATURES if name not in ("bbox_w", "bbox_h")]
    config = {
        "model": {
            "input_feature_names": reduced_input_features,
            "input_dim": len(reduced_input_features),
            "target_features": ["vx", "vy"],
        }
    }
    with pytest.raises(ValueError, match="Dataset feature order does not match"):
        trainer.validate_feature_contract(dataset, config, state_injected_rollout=False)
    trainer.validate_feature_contract(dataset, config, state_injected_rollout=True)


def test_target_parameterization_from_config_accepts_velocity_only_targets() -> None:
    config = {"model": {"target_features": ["vx", "vy"]}}
    assert trainer.target_parameterization_from_config(config) == {"mode": "velocity"}


def test_validate_feature_contract_velocity_only_requires_state_injected_rollout(tmp_path: Path) -> None:
    dataset, _ = _rollout_batch(tmp_path)
    config = {
        "model": {
            "input_feature_names": V2_FEATURES,
            "input_dim": len(V2_FEATURES),
            "target_features": ["vx", "vy"],
        }
    }
    with pytest.raises(ValueError, match="Closed-loop physics rollout requires"):
        trainer.validate_feature_contract(dataset, config, state_injected_rollout=False)
    trainer.validate_feature_contract(dataset, config, state_injected_rollout=True)

    with pytest.raises(ValueError, match="Unsupported target_features under state_injected_rollout"):
        trainer.validate_feature_contract(
            dataset,
            {"model": {**config["model"], "target_features": ["vx"]}},
            state_injected_rollout=True,
        )


def test_boundary_conditioned_rollout_velocity_only_targets_no_bbox_crash(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "no_bbox_target.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz, start_frames=[0], T_history=1, T_future=3, max_droplets=4, target_features=("vx", "vy")
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    stats = _identity_stats(16, target_dim=2)
    weights = trainer.rollout_weights(3, 0.0, torch.device("cpu"))
    model = _model(seed=9, target_dim=2, bbox_stop_gradient=False)
    typical_inlet_velocity = torch.tensor([0.0, 0.0])

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        state_injected_rollout=True,
        typical_inlet_velocity=typical_inlet_velocity,
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


# ---------------------------------------------------------------------------
# compute_typical_inlet_velocity
# ---------------------------------------------------------------------------


class _FakeInletDataset:
    def __init__(self, Z: np.ndarray, mask: np.ndarray) -> None:
        self.Z = Z
        self.mask = mask


def test_compute_typical_inlet_velocity() -> None:
    from src.physics.targets.junction_decision import INLET_CHANNEL, LEFT_BRANCH

    feature_index = {"x": 0, "y": 1, "vx": 2, "vy": 3}
    region_labels = np.zeros((10, 10), dtype=np.int64)
    region_labels[0:5, :] = INLET_CHANNEL
    region_labels[5:10, :] = LEFT_BRANCH

    # Track 0: 2 observations in the inlet channel (vx,vy = 2,4 and 4,8 -> mean 3,6), 1 in the
    # branch region (vx,vy = 100,100, must NOT be included).
    Z = np.full((1, 3, 4), np.nan, dtype=np.float32)
    Z[0, 0] = [1.0, 1.0, 2.0, 4.0]  # inlet channel (y=1 -> row 1, INLET_CHANNEL)
    Z[0, 1] = [1.0, 2.0, 4.0, 8.0]  # inlet channel (y=2 -> row 2, INLET_CHANNEL)
    Z[0, 2] = [1.0, 7.0, 100.0, 100.0]  # left branch (y=7 -> row 7, LEFT_BRANCH), excluded
    mask = np.ones((1, 3), dtype=bool)

    dataset = _FakeInletDataset(Z, mask)
    vx, vy = trainer.compute_typical_inlet_velocity(dataset, region_labels, feature_index)
    assert vx == pytest.approx(3.0)
    assert vy == pytest.approx(6.0)


def test_compute_typical_inlet_velocity_raises_when_no_inlet_observations() -> None:
    from src.physics.targets.junction_decision import LEFT_BRANCH

    feature_index = {"x": 0, "y": 1, "vx": 2, "vy": 3}
    region_labels = np.full((10, 10), LEFT_BRANCH, dtype=np.int64)  # no inlet region at all
    Z = np.full((1, 2, 4), np.nan, dtype=np.float32)
    Z[0, 0] = [1.0, 1.0, 2.0, 4.0]
    mask = np.ones((1, 2), dtype=bool)
    dataset = _FakeInletDataset(Z, mask)

    with pytest.raises(ValueError, match="No finite vx,vy observations"):
        trainer.compute_typical_inlet_velocity(dataset, region_labels, feature_index)
