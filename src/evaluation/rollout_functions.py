from __future__ import annotations

import numpy as np
import torch

from src.physics.runtime import step as physics_runtime_step


RUNTIME_TARGET_FEATURES = ("vx", "vy", "bbox_w", "bbox_h")


def boundary_conditioned_rollout(model, batch, dataset, normalization_stats, weights, runtime_context=None):
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
    boundary_masks = []
    internal_loss_masks = []
    step_losses = []

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

        if runtime_context is None:
            velocity_to_px_frame = velocity_mm_s_to_px_frame_scale(dataset, device, torch)
            x_next = last_frame[:, :, feature_index["x"]] + pred_step_phys_raw[:, :, 0] * velocity_to_px_frame
            y_next = last_frame[:, :, feature_index["y"]] + pred_step_phys_raw[:, :, 1] * velocity_to_px_frame

            new_frame_phys = last_frame.clone()
            new_frame_phys[:, :, feature_index["x"]] = x_next
            new_frame_phys[:, :, feature_index["y"]] = y_next
            new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys_raw[:, :, 0]
            new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys_raw[:, :, 1]
            new_frame_phys[boundary_mask] = true_step_features[boundary_mask]
            refresh_observed_non_target_features(new_frame_phys, true_step_features, new_mask, feature_index)
        else:
            new_frame_phys = torch.zeros_like(last_frame)
            refreshed_phys = runtime_step_batch(last_frame, pred_step_phys_raw, continuing_mask, runtime_context)
            new_frame_phys = torch.where(continuing_mask[:, :, None], refreshed_phys, new_frame_phys)
            new_frame_phys[boundary_mask] = true_step_features[boundary_mask]

        pred_step_norm = pred_step_norm_raw.clone()
        pred_step_phys = pred_step_phys_raw.clone()
        pred_step_norm[boundary_mask] = true_step_norm[boundary_mask]
        pred_step_phys[boundary_mask] = true_step_phys[boundary_mask]

        loss_mask = batch.get("cfd_loss_mask", batch["future_mask"])[:, step_index, :] & ~boundary_mask
        step_loss = masked_velocity_mse(pred_step_norm, true_step_norm, loss_mask)
        step_losses.append(step_loss)

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
        boundary_masks.append(boundary_mask)
        internal_loss_masks.append(loss_mask)

    step_loss_tensor = torch.stack(step_losses)
    weighted_loss_internal_only = (step_loss_tensor * weights).sum() / weights.sum()
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
        "mask": torch.stack(step_masks, dim=1),
        "boundary_mask": torch.stack(boundary_masks, dim=1),
        "internal_loss_mask": torch.stack(internal_loss_masks, dim=1),
    }


def refresh_observed_non_target_features(new_frame_phys, true_step_features, new_mask, feature_index) -> None:
    predicted_names = {"x", "y", "vx", "vy"}
    for name, index in feature_index.items():
        if name in predicted_names:
            continue
        values = true_step_features[:, :, index]
        valid = new_mask & torch.isfinite(values)
        new_frame_phys[:, :, index] = torch.where(valid, values, new_frame_phys[:, :, index])


def runtime_step_batch(current_state_phys, model_prediction_phys, active_mask, runtime_context):
    device = current_state_phys.device
    dtype = current_state_phys.dtype
    current_np = current_state_phys.detach().cpu().numpy().astype(np.float32, copy=True)
    prediction_np = model_prediction_phys.detach().cpu().numpy().astype(np.float32, copy=True)
    active_np = active_mask.detach().cpu().numpy().astype(bool, copy=True)
    next_np = np.zeros_like(current_np, dtype=np.float32)
    active_mask_cache: dict[int, np.ndarray] = {}

    for batch_index in np.flatnonzero(active_np.any(axis=1)):
        active_slots = active_np[batch_index]
        active_count = int(np.count_nonzero(active_slots))
        runtime_active_mask = active_mask_cache.setdefault(active_count, np.ones(active_count, dtype=bool))
        next_active = physics_runtime_step(
            current_np[batch_index, active_slots],
            prediction_np[batch_index, active_slots],
            runtime_context,
            active_mask=runtime_active_mask,
        )
        next_np[batch_index, active_slots] = next_active

    return torch.as_tensor(next_np, dtype=dtype, device=device)


def masked_velocity_mse(prediction, target, mask):
    expanded_mask = mask.unsqueeze(-1).expand_as(target)
    squared_error = (prediction - target) ** 2
    valid_error = squared_error[expanded_mask]
    if valid_error.numel() == 0:
        return squared_error.sum() * 0.0
    return valid_error.mean()


def velocity_mm_s_to_px_frame_scale(dataset, device, torch_module=torch):
    velocity_units = getattr(dataset, "velocity_units", "px/frame")
    if velocity_units == "mm/s":
        conversion = float(getattr(dataset, "velocity_mm_s_per_px_frame", 1.0))
        if conversion <= 0 or not np.isfinite(conversion):
            raise ValueError(f"Invalid velocity conversion factor: {conversion}")
        return torch_module.as_tensor(1.0 / conversion, dtype=torch.float32, device=device)
    return torch_module.as_tensor(1.0, dtype=torch.float32, device=device)


def get_true_future_features(batch, dataset, device, horizon):
    droplet_ids = batch["droplet_ids"].detach().cpu().numpy()
    frame_starts = batch["frame_start"].detach().cpu().numpy()
    track_id_to_index = {int(track_id): index for index, track_id in enumerate(dataset.track_ids)}
    source_Z = dataset.source_Z if hasattr(dataset, "source_Z") else dataset.Z
    input_indices = getattr(dataset, "input_indices", slice(None))

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
            true_features[batch_index, :, slot_index, :] = source_Z[droplet_index, start:end, :][:, input_indices]
    return torch.as_tensor(true_features, dtype=torch.float32, device=device)


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
