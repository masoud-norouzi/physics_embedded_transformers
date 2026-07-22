from __future__ import annotations

import numpy as np
import torch


def boundary_conditioned_rollout(model, batch, dataset, normalization_stats, weights):
    device = batch["history_x"].device
    rollout_history = batch["history_x"].clone()
    history_mask = batch["history_mask"].clone()

    pred_velocities_norm = []
    true_velocities_norm = []
    pred_velocities_phys = []
    true_velocities_phys = []
    pred_positions = []
    true_positions = []
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
        x_next = last_frame[:, :, feature_index["x"]] + pred_step_phys_raw[:, :, 0]
        y_next = last_frame[:, :, feature_index["y"]] + pred_step_phys_raw[:, :, 1]

        new_frame_phys = last_frame.clone()
        new_frame_phys[:, :, feature_index["x"]] = x_next
        new_frame_phys[:, :, feature_index["y"]] = y_next
        new_frame_phys[:, :, feature_index["vx"]] = pred_step_phys_raw[:, :, 0]
        new_frame_phys[:, :, feature_index["vy"]] = pred_step_phys_raw[:, :, 1]

        new_mask = batch["future_mask"][:, step_index, :]
        entering_mask = new_mask & ~previous_last_mask
        true_step_features = true_future_features[:, step_index, :, :]
        true_step_features_finite = torch.isfinite(true_step_features).all(dim=-1)
        boundary_mask = entering_mask & true_step_features_finite
        new_frame_phys[boundary_mask] = true_step_features[boundary_mask]

        refresh_observed_non_target_features(new_frame_phys, true_step_features, new_mask, feature_index)

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

        pred_velocities_norm.append(pred_step_norm)
        true_velocities_norm.append(true_step_norm)
        pred_velocities_phys.append(pred_step_phys)
        true_velocities_phys.append(true_step_phys)
        pred_positions.append(new_frame_phys[:, :, [feature_index["x"], feature_index["y"]]])
        true_positions.append(true_future_xy[:, step_index, :, :])
        step_masks.append(new_mask)
        boundary_masks.append(boundary_mask)
        internal_loss_masks.append(loss_mask)

    step_loss_tensor = torch.stack(step_losses)
    weighted_loss_internal_only = (step_loss_tensor * weights).sum() / weights.sum()

    return {
        "weighted_loss": weighted_loss_internal_only,
        "weighted_loss_internal_only": weighted_loss_internal_only,
        "step_losses": step_loss_tensor,
        "pred_velocity_norm": torch.stack(pred_velocities_norm, dim=1),
        "true_velocity_norm": torch.stack(true_velocities_norm, dim=1),
        "pred_velocity": torch.stack(pred_velocities_phys, dim=1),
        "true_velocity": torch.stack(true_velocities_phys, dim=1),
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


def masked_velocity_mse(prediction, target, mask):
    expanded_mask = mask.unsqueeze(-1).expand_as(target)
    squared_error = (prediction - target) ** 2
    valid_error = squared_error[expanded_mask]
    if valid_error.numel() == 0:
        return squared_error.sum() * 0.0
    return valid_error.mean()


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
