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
from src.physics.targets.junction_decision import (
    INLET_CHANNEL,
    INLET_JUNCTION,
    LEFT_BRANCH,
    OUTLET_CHANNEL,
    RIGHT_BRANCH,
    derive_branch_decision_labels,
    region_codes_for_points,
)


V2_FEATURES = trainer.FEATURE_NAMES


# ---------------------------------------------------------------------------
# src/physics/targets/junction_decision.py -- label/window derivation
# ---------------------------------------------------------------------------


def _region_grid() -> np.ndarray:
    # 1 row x 10 col strip: col 0-2 inlet_channel, 3-4 inlet_junction, 5-9 right_branch,
    # except a second copy below (row 1) used for the left-branch track.
    grid = np.zeros((2, 10), dtype=np.int64)
    grid[0, 0:3] = INLET_CHANNEL
    grid[0, 3:5] = INLET_JUNCTION
    grid[0, 5:10] = RIGHT_BRANCH
    grid[1, 0:3] = INLET_CHANNEL
    grid[1, 3:5] = INLET_JUNCTION
    grid[1, 5:10] = LEFT_BRANCH
    return grid


def test_region_codes_for_points_matches_grid_and_handles_out_of_bounds() -> None:
    region_labels = _region_grid()
    x = np.array([0.0, 4.0, 100.0, np.nan])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    codes = region_codes_for_points(x, y, region_labels)
    assert codes.tolist() == [INLET_CHANNEL, INLET_JUNCTION, 0, 0]


def test_derive_branch_decision_labels_right_branch_track() -> None:
    region_labels = _region_grid()
    n_frames = 8
    # Track 0 walks columns 0..7 along row 0: inlet_channel(0-2), inlet_junction(3-4),
    # right_branch from col 5 (frame 5) onward.
    x = np.arange(n_frames, dtype=np.float32)
    y = np.zeros(n_frames, dtype=np.float32)
    Z = np.zeros((1, n_frames, 2), dtype=np.float32)
    Z[0, :, 0] = x
    Z[0, :, 1] = y
    mask = np.ones((1, n_frames), dtype=bool)
    feature_index = {"x": 0, "y": 1}

    branch_label, in_window, frames_until_commit = derive_branch_decision_labels(Z, mask, feature_index, region_labels)

    assert branch_label.shape == (1, n_frames)
    # Commit frame is 5 (first right_branch observation); window is frames [0, 5).
    np.testing.assert_array_equal(in_window[0], [True, True, True, True, True, False, False, False])
    assert np.all(branch_label[0, :5] == 1.0)
    assert np.all(np.isnan(branch_label[0, 5:]))
    np.testing.assert_array_equal(frames_until_commit[0, :5], [5, 4, 3, 2, 1])
    assert np.all(frames_until_commit[0, 5:] == -1)


def test_derive_branch_decision_labels_left_branch_track_gets_label_zero() -> None:
    region_labels = _region_grid()
    n_frames = 8
    x = np.arange(n_frames, dtype=np.float32)
    y = np.ones(n_frames, dtype=np.float32)  # row 1 -> left_branch
    Z = np.zeros((1, n_frames, 2), dtype=np.float32)
    Z[0, :, 0] = x
    Z[0, :, 1] = y
    mask = np.ones((1, n_frames), dtype=bool)
    feature_index = {"x": 0, "y": 1}

    branch_label, in_window, _ = derive_branch_decision_labels(Z, mask, feature_index, region_labels)

    assert np.all(branch_label[0, :5] == 0.0)
    assert in_window[0, :5].all()
    assert not in_window[0, 5:].any()


def test_derive_branch_decision_labels_track_never_reaching_junction_gets_no_label() -> None:
    region_labels = _region_grid()
    n_frames = 4
    # Stays entirely in outlet_channel territory (never inlet/junction/branch).
    Z = np.zeros((1, n_frames, 2), dtype=np.float32)
    Z[0, :, 0] = -5.0  # out of bounds -> region code 0 every frame
    Z[0, :, 1] = 0.0
    mask = np.ones((1, n_frames), dtype=bool)
    feature_index = {"x": 0, "y": 1}

    branch_label, in_window, frames_until_commit = derive_branch_decision_labels(Z, mask, feature_index, region_labels)

    assert not in_window.any()
    assert np.all(np.isnan(branch_label))
    assert np.all(frames_until_commit == -1)


def test_derive_branch_decision_labels_commit_without_prior_pre_junction_observation() -> None:
    region_labels = _region_grid()
    n_frames = 3
    # Droplet appears directly inside the branch region -- never observed upstream, so no window.
    Z = np.zeros((1, n_frames, 2), dtype=np.float32)
    Z[0, :, 0] = 6.0
    Z[0, :, 1] = 0.0
    mask = np.ones((1, n_frames), dtype=bool)
    feature_index = {"x": 0, "y": 1}

    branch_label, in_window, _ = derive_branch_decision_labels(Z, mask, feature_index, region_labels)

    assert not in_window.any()
    assert np.all(np.isnan(branch_label))


# ---------------------------------------------------------------------------
# CanonicalRolloutTransformer -- decision head construction and forward pass
# ---------------------------------------------------------------------------


def _model(
    predict_branch_decision: bool,
    seed: int = 0,
    decision_stop_gradient: bool = False,
    condition_velocity_on_decision: bool = False,
) -> CanonicalRolloutTransformer:
    torch.manual_seed(seed)
    return CanonicalRolloutTransformer(
        input_dim=16,
        target_dim=4,
        T_history=1,
        max_droplets=4,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        predict_branch_decision=predict_branch_decision,
        decision_stop_gradient=decision_stop_gradient,
        condition_velocity_on_decision=condition_velocity_on_decision,
    )


def _history(batch=2, T=1, M=4, F=16):
    history_x = torch.randn(batch, T, M, F)
    history_mask = torch.ones(batch, T, M, dtype=torch.bool)
    return history_x, history_mask


def test_decision_head_is_none_when_not_requested() -> None:
    model = _model(predict_branch_decision=False)
    assert model.decision_head is None


def test_return_decision_without_head_raises() -> None:
    model = _model(predict_branch_decision=False)
    history_x, history_mask = _history()
    with pytest.raises(ValueError, match="predict_branch_decision"):
        model(history_x, history_mask, return_decision=True)


def test_decision_head_forward_shape_and_probability_range() -> None:
    model = _model(predict_branch_decision=True)
    history_x, history_mask = _history(batch=3)
    output = model(history_x, history_mask, return_decision=True)
    assert set(output.keys()) == {"prediction", "decision_logit"}
    assert output["prediction"].shape == (3, 4, 4)
    assert output["decision_logit"].shape == (3, 4)
    probs = torch.sigmoid(output["decision_logit"])
    assert bool(((probs >= 0.0) & (probs <= 1.0)).all())


def test_plain_call_is_backward_compatible_plain_tensor() -> None:
    model = _model(predict_branch_decision=True)
    history_x, history_mask = _history()
    output = model(history_x, history_mask)
    assert torch.is_tensor(output)
    assert output.shape == (2, 4, 4)


def test_decision_head_does_not_change_velocity_prediction_given_identical_seed() -> None:
    history_x, history_mask = _history()
    model_without_head = _model(predict_branch_decision=False, seed=42)
    model_with_head = _model(predict_branch_decision=True, seed=42)
    # Same seed -> trunk weights are identical up through velocity_head construction order;
    # decision_head params are drawn afterward and don't perturb earlier RNG draws for the trunk.
    pred_without = model_without_head(history_x, history_mask)
    pred_with = model_with_head(history_x, history_mask)
    assert torch.allclose(pred_without, pred_with)


# ---------------------------------------------------------------------------
# Training-loop wiring: decision loss inside boundary_conditioned_rollout
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


def _decision_batch(tmp_path: Path):
    npz = _write_npz(tmp_path / "decision.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=1,
        T_future=2,
        max_droplets=4,
        target_features=trainer.RUNTIME_TARGET_FEATURES,
    )
    batch = trainer.move_batch_to_device(next(iter(DataLoader(dataset, batch_size=1))), torch.device("cpu"))
    return dataset, batch


def _branch_decision(*, all_in_window: bool) -> "trainer.BranchDecisionTraining":
    n_tracks, n_frames = 2, 6
    branch_label = np.full((n_tracks, n_frames), np.nan, dtype=np.float32)
    in_window = np.zeros((n_tracks, n_frames), dtype=bool)
    frames_until_commit = np.full((n_tracks, n_frames), -1, dtype=np.int32)
    if all_in_window:
        branch_label[:] = np.array([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
        in_window[:] = True
        frames_until_commit[:] = np.array([[5, 4, 3, 2, 1, 0]])
    return trainer.BranchDecisionTraining(
        enabled=True,
        loss_weight=1.0,
        branch_label=branch_label,
        in_window=in_window,
        frames_until_commit=frames_until_commit,
    )


def test_decision_loss_is_zero_when_window_is_empty(tmp_path: Path) -> None:
    dataset, batch = _decision_batch(tmp_path)
    model = _model(predict_branch_decision=True)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(2, 2.0, torch.device("cpu"))
    branch_decision = _branch_decision(all_in_window=False)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        branch_decision=branch_decision,
    )

    assert rollout["decision_count"] == 0
    assert rollout["decision_loss"].item() == pytest.approx(0.0)
    assert rollout["weighted_decision_loss"].item() == pytest.approx(0.0)
    assert rollout["total_loss"].item() == pytest.approx(rollout["weighted_loss_internal_only"].item())


def test_decision_loss_nonzero_and_masked_when_window_present(tmp_path: Path) -> None:
    dataset, batch = _decision_batch(tmp_path)
    model = _model(predict_branch_decision=True)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(2, 2.0, torch.device("cpu"))
    branch_decision = _branch_decision(all_in_window=True)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        branch_decision=branch_decision,
    )

    assert rollout["decision_count"] > 0
    assert torch.isfinite(rollout["decision_loss"])
    assert rollout["weighted_decision_loss"].item() == pytest.approx(
        branch_decision.loss_weight * rollout["decision_loss"].item()
    )
    assert rollout["total_loss"].item() == pytest.approx(
        rollout["weighted_loss_internal_only"].item() + rollout["weighted_decision_loss"].item()
    )
    assert rollout["decision_probs"].numel() == rollout["decision_count"]
    assert rollout["decision_labels"].numel() == rollout["decision_count"]


def test_decision_head_ignored_when_model_lacks_it(tmp_path: Path) -> None:
    # A model without predict_branch_decision must behave exactly as before, even if a
    # branch_decision config is supplied -- decision_enabled gates on model.decision_head.
    dataset, batch = _decision_batch(tmp_path)
    model = _model(predict_branch_decision=False)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(2, 2.0, torch.device("cpu"))
    branch_decision = _branch_decision(all_in_window=True)

    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        branch_decision=branch_decision,
    )

    assert rollout["decision_count"] == 0
    assert rollout["total_loss"].item() == pytest.approx(rollout["weighted_loss_internal_only"].item())


def test_frozen_trunk_only_decision_head_updates_after_training_step(tmp_path: Path) -> None:
    dataset, batch = _decision_batch(tmp_path)
    model = _model(predict_branch_decision=True)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(2, 2.0, torch.device("cpu"))
    branch_decision = _branch_decision(all_in_window=True)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("decision_head")
    trunk_before = {
        name: param.detach().clone() for name, param in model.named_parameters() if not name.startswith("decision_head")
    }
    decision_before = {
        name: param.detach().clone() for name, param in model.named_parameters() if name.startswith("decision_head")
    }

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=0.1)
    rollout = trainer.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        weights,
        runtime_context=None,
        branch_decision=branch_decision,
    )
    optimizer.zero_grad()
    rollout["weighted_decision_loss"].backward()
    optimizer.step()

    for name, param in model.named_parameters():
        if name.startswith("decision_head"):
            assert not torch.allclose(param.detach(), decision_before[name]), f"{name} should have changed"
        else:
            assert torch.allclose(param.detach(), trunk_before[name]), f"{name} should NOT have changed (trunk frozen)"


# ---------------------------------------------------------------------------
# decision_stop_gradient -- joint from-scratch training with the trunk trained
# only by the motion/geometry loss, decision_head riding along on a detached copy
# ---------------------------------------------------------------------------


def test_decision_stop_gradient_true_isolates_decision_backward_from_trunk() -> None:
    model = _model(predict_branch_decision=True, decision_stop_gradient=True)
    history_x, history_mask = _history()
    output = model(history_x, history_mask, return_decision=True)
    output["decision_logit"].sum().backward()
    for name, param in model.named_parameters():
        if name.startswith("decision_head"):
            assert param.grad is not None, f"{name} should have received gradient from decision_logit"
        else:
            assert param.grad is None, f"{name} should NOT have received gradient (decision_stop_gradient=True)"


def test_decision_stop_gradient_false_lets_gradient_reach_trunk() -> None:
    model = _model(predict_branch_decision=True, decision_stop_gradient=False)
    history_x, history_mask = _history()
    output = model(history_x, history_mask, return_decision=True)
    output["decision_logit"].sum().backward()
    trunk_grad_present = any(
        param.grad is not None for name, param in model.named_parameters() if not name.startswith("decision_head")
    )
    assert trunk_grad_present, "default (decision_stop_gradient=False) must preserve the pre-existing behavior"


def test_decision_stop_gradient_true_trunk_gradient_unaffected_by_decision_loss(tmp_path: Path) -> None:
    dataset, batch = _decision_batch(tmp_path)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(2, 2.0, torch.device("cpu"))
    branch_decision = _branch_decision(all_in_window=True)

    model_with_decision = _model(predict_branch_decision=True, seed=7, decision_stop_gradient=True)
    rollout_with = trainer.boundary_conditioned_rollout(
        model_with_decision, batch, dataset, stats, weights, runtime_context=None, branch_decision=branch_decision
    )
    rollout_with["total_loss"].backward()
    trunk_grads_with = {
        name: param.grad.clone()
        for name, param in model_with_decision.named_parameters()
        if not name.startswith("decision_head")
    }
    decision_head_grad_norm = sum(
        float(param.grad.abs().sum())
        for name, param in model_with_decision.named_parameters()
        if name.startswith("decision_head")
    )
    assert decision_head_grad_norm > 0.0, "decision_head should still receive gradient from decision_loss"

    # Same seed -> bit-identical initial weights (trunk AND decision_head) -- so any trunk gradient
    # difference between the two runs can only come from whether the decision term was included.
    model_without_decision = _model(predict_branch_decision=True, seed=7, decision_stop_gradient=True)
    rollout_without = trainer.boundary_conditioned_rollout(
        model_without_decision, batch, dataset, stats, weights, runtime_context=None, branch_decision=None
    )
    rollout_without["total_loss"].backward()
    trunk_grads_without = {
        name: param.grad
        for name, param in model_without_decision.named_parameters()
        if not name.startswith("decision_head")
    }

    assert set(trunk_grads_with.keys()) == set(trunk_grads_without.keys())
    for name in trunk_grads_with:
        assert torch.allclose(trunk_grads_with[name], trunk_grads_without[name], atol=1e-6), (
            f"{name} trunk gradient changed when the decision loss was included -- isolation is broken"
        )


# ---------------------------------------------------------------------------
# Phase 2/3: condition_velocity_on_decision (Axis 1 concat, Axis 2/4 gate, Axis 3 override)
# ---------------------------------------------------------------------------


def test_condition_velocity_on_decision_requires_predict_branch_decision() -> None:
    with pytest.raises(ValueError, match="predict_branch_decision"):
        _model(predict_branch_decision=False, condition_velocity_on_decision=True)


def test_condition_velocity_on_decision_changes_velocity_head_input_dim() -> None:
    conditioned = _model(predict_branch_decision=True, condition_velocity_on_decision=True)
    plain = _model(predict_branch_decision=True, condition_velocity_on_decision=False)
    assert conditioned.velocity_head[0].in_features == plain.velocity_head[0].in_features + 1


def test_condition_velocity_on_decision_forward_always_returns_dict() -> None:
    model = _model(predict_branch_decision=True, condition_velocity_on_decision=True)
    history_x, history_mask = _history()
    output = model(history_x, history_mask)  # no return_attention/return_decision requested
    assert isinstance(output, dict)
    assert set(output.keys()) == {"prediction", "decision_logit"}


def test_condition_velocity_on_decision_signal_changes_prediction() -> None:
    model = _model(predict_branch_decision=True, condition_velocity_on_decision=True)
    history_x, history_mask = _history()
    gate = torch.ones(2, 4, dtype=torch.bool)

    output_low = model(history_x, history_mask, decision_condition_signal=torch.zeros(2, 4), decision_condition_gate=gate)
    output_high = model(history_x, history_mask, decision_condition_signal=torch.ones(2, 4), decision_condition_gate=gate)

    assert not torch.allclose(output_low["prediction"], output_high["prediction"])


def test_condition_velocity_on_decision_gate_false_forces_neutral_value() -> None:
    model = _model(predict_branch_decision=True, condition_velocity_on_decision=True)
    history_x, history_mask = _history()
    gate_off = torch.zeros(2, 4, dtype=torch.bool)

    output_low = model(history_x, history_mask, decision_condition_signal=torch.zeros(2, 4), decision_condition_gate=gate_off)
    output_high = model(history_x, history_mask, decision_condition_signal=torch.ones(2, 4), decision_condition_gate=gate_off)

    # Gate off -> both calls fall back to the same neutral 0.5 conditioning value, regardless of signal.
    assert torch.allclose(output_low["prediction"], output_high["prediction"])


def test_condition_velocity_on_decision_signal_none_matches_all_nan_signal() -> None:
    model = _model(predict_branch_decision=True, condition_velocity_on_decision=True)
    history_x, history_mask = _history()

    output_none = model(history_x, history_mask)
    output_nan = model(
        history_x, history_mask, decision_condition_signal=torch.full((2, 4), float("nan"))
    )

    assert torch.allclose(output_none["prediction"], output_nan["prediction"])
    assert torch.allclose(output_none["decision_logit"], output_nan["decision_logit"])


def test_condition_velocity_on_decision_stop_gradient_isolates_trunk_from_conditioning_path() -> None:
    # A same-model, same-forward-value comparison: comparing gradients across two *different*
    # model instances would confound the test, since velocity_head's Linear/GELU/LayerNorm stack
    # is nonlinear and gets evaluated at a different point whenever the conditioning value differs
    # (e.g. a differently-shaped velocity_head, or a different signal value) -- that alone can
    # change d(prediction)/d(h_last) for reasons having nothing to do with gradient isolation.
    # So instead: run the model self-conditioned (decision_head(h_last.detach()) -> signal), then
    # re-run it with that exact same numeric signal supplied as an external override (bypassing
    # decision_head for velocity_head's input entirely). Both calls produce bit-identical
    # `prediction` values, so if decision_head's detached contribution truly adds zero gradient,
    # the trunk gradients from the two calls must match exactly.
    model = _model(predict_branch_decision=True, decision_stop_gradient=True, condition_velocity_on_decision=True)
    history_x, history_mask = _history()

    output_self = model(history_x, history_mask)
    signal_value = torch.sigmoid(output_self["decision_logit"]).detach().clone()
    output_self["prediction"].sum().backward()
    trunk_grads_self = {
        name: param.grad.clone() for name, param in model.named_parameters() if not name.startswith("decision_head")
    }
    decision_grad_norm = sum(
        float(param.grad.abs().sum()) for name, param in model.named_parameters() if name.startswith("decision_head")
    )
    assert decision_grad_norm > 0.0, "decision_head should receive gradient via the motion-loss conditioning path"

    model.zero_grad()

    output_override = model(history_x, history_mask, decision_condition_signal=signal_value)
    assert torch.allclose(output_override["prediction"], output_self["prediction"])
    output_override["prediction"].sum().backward()
    trunk_grads_override = {
        name: param.grad for name, param in model.named_parameters() if not name.startswith("decision_head")
    }

    for name in trunk_grads_self:
        assert torch.allclose(trunk_grads_self[name], trunk_grads_override[name], atol=1e-6), (
            f"{name} trunk gradient differs from the external-override baseline -- "
            "decision_head is leaking gradient into the trunk"
        )


def test_sample_decision_conditioning_signal_p_truth_none_is_all_nan() -> None:
    true_branch_label_step = torch.tensor([1.0, 0.0, float("nan")])
    result = trainer.sample_decision_conditioning_signal(true_branch_label_step, None)
    assert torch.isnan(result).all()


def test_sample_decision_conditioning_signal_p_truth_one_uses_truth_where_finite() -> None:
    true_branch_label_step = torch.tensor([1.0, 0.0, float("nan")])
    result = trainer.sample_decision_conditioning_signal(true_branch_label_step, 1.0)
    assert result[0].item() == pytest.approx(1.0)
    assert result[1].item() == pytest.approx(0.0)
    assert torch.isnan(result[2])  # non-finite truth is never selected, even at p_truth=1.0


def test_sample_decision_conditioning_signal_p_truth_zero_never_uses_truth() -> None:
    true_branch_label_step = torch.tensor([1.0, 0.0])
    result = trainer.sample_decision_conditioning_signal(true_branch_label_step, 0.0)
    assert torch.isnan(result).all()


def test_pre_junction_gate_matches_expected_regions() -> None:
    feature_index = {
        "occupancy_inlet_channel": 0,
        "occupancy_inlet_junction": 1,
        "occupancy_left_branch": 2,
        "occupancy_right_branch": 3,
    }
    last_frame = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # inlet_channel -> pre-junction, gate True
            [0.0, 1.0, 0.0, 0.0],  # inlet_junction -> pre-junction, gate True
            [0.0, 0.0, 1.0, 0.0],  # resolved into left_branch -> gate False
            [0.0, 0.0, 0.0, 1.0],  # resolved into right_branch -> gate False
            [0.0, 0.0, 0.0, 0.0],  # neither (open channel / outlet) -> gate False
        ]
    )
    gate = trainer.pre_junction_gate(last_frame, feature_index)
    assert gate.tolist() == [True, True, False, False, False]


def test_decision_confidence_crossing_frame_finds_far_bin() -> None:
    calibration = {
        "5": {"mean_predicted_p_short": 0.5, "observed_frequency_short": 0.5, "count": 10},
        "3": {"mean_predicted_p_short": 0.85, "observed_frequency_short": 0.9, "count": 10},
        "1": {"mean_predicted_p_short": 0.95, "observed_frequency_short": 1.0, "count": 10},
    }
    assert trainer.decision_confidence_crossing_frame(calibration) == pytest.approx(3.0)


def test_decision_confidence_crossing_frame_nan_when_never_confident() -> None:
    calibration = {
        "5": {"mean_predicted_p_short": 0.5, "observed_frequency_short": 0.5, "count": 10},
        "1": {"mean_predicted_p_short": 0.6, "observed_frequency_short": 0.6, "count": 10},
    }
    assert np.isnan(trainer.decision_confidence_crossing_frame(calibration))


def test_decision_confidence_crossing_frame_empty_is_nan() -> None:
    assert np.isnan(trainer.decision_confidence_crossing_frame({}))


def test_boundary_conditioned_rollout_with_condition_velocity_on_decision(tmp_path: Path) -> None:
    dataset, batch = _decision_batch(tmp_path)
    model = _model(predict_branch_decision=True, condition_velocity_on_decision=True)
    stats = _identity_stats(16, target_dim=4)
    weights = trainer.rollout_weights(2, 2.0, torch.device("cpu"))
    branch_decision = _branch_decision(all_in_window=True)

    rollout = trainer.boundary_conditioned_rollout(
        model, batch, dataset, stats, weights, runtime_context=None, p_truth=0.5, branch_decision=branch_decision
    )

    assert torch.isfinite(rollout["total_loss"])
    assert torch.isfinite(rollout["decision_loss"])
    assert rollout["decision_count"] > 0


# ---------------------------------------------------------------------------
# create_branch_decision_training / get_true_branch_labels
# ---------------------------------------------------------------------------


def test_create_branch_decision_training_loads_npz(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.npz"
    np.savez(
        labels_path,
        branch_label=np.zeros((2, 6), dtype=np.float32),
        in_window=np.ones((2, 6), dtype=bool),
        frames_until_commit=np.full((2, 6), 3, dtype=np.int32),
    )
    config = {
        "training": {
            "decision_head": {
                "enabled": True,
                "loss_weight": 2.5,
                "labels_path": str(labels_path),
            }
        }
    }
    branch_decision = trainer.create_branch_decision_training(config)
    assert branch_decision is not None
    assert branch_decision.enabled
    assert branch_decision.loss_weight == pytest.approx(2.5)
    assert branch_decision.branch_label.shape == (2, 6)


def test_create_branch_decision_training_returns_none_when_disabled() -> None:
    assert trainer.create_branch_decision_training({"training": {}}) is None
    assert trainer.create_branch_decision_training({"training": {"decision_head": {"enabled": False}}}) is None


def test_get_true_branch_labels_alignment(tmp_path: Path) -> None:
    dataset, batch = _decision_batch(tmp_path)
    branch_decision = _branch_decision(all_in_window=True)

    branch_label, in_window, frames_until_commit = trainer.get_true_branch_labels(
        batch, dataset, torch.device("cpu"), horizon=2, branch_decision=branch_decision
    )

    assert branch_label.shape == (1, 2, 4)
    assert in_window.dtype == torch.bool
    # Track ids in the tiny dataset are [10, 20] -> row 0/1 of branch_decision arrays.
    assert bool(in_window.any())
    assert bool(torch.isfinite(branch_label[in_window]).all())


# ---------------------------------------------------------------------------
# decision_validation_metrics
# ---------------------------------------------------------------------------


def test_decision_validation_metrics_empty_returns_nan() -> None:
    metrics = trainer.decision_validation_metrics([], [], [])
    assert np.isnan(metrics["decision_accuracy_near_commitment"])
    assert metrics["decision_calibration"] == {}


def test_decision_validation_metrics_accuracy_and_calibration() -> None:
    probs = [torch.tensor([0.9, 0.2, 0.8])]
    labels = [torch.tensor([1.0, 0.0, 1.0])]
    frames_until_commit = [torch.tensor([1, 1, 3], dtype=torch.int32)]

    metrics = trainer.decision_validation_metrics(probs, labels, frames_until_commit)

    assert metrics["decision_accuracy_near_commitment"] == pytest.approx(1.0)
    assert set(metrics["decision_calibration"].keys()) == {"1", "3"}
    bin1 = metrics["decision_calibration"]["1"]
    assert bin1["count"] == 2
    assert bin1["mean_predicted_p_short"] == pytest.approx((0.9 + 0.2) / 2)
    assert bin1["observed_frequency_short"] == pytest.approx(0.5)
