from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config.velocity import DEFAULT_EXPERIMENT_CONFIG, load_velocity_conversion_from_experiment
from src.physics.targets import (
    SPEED_ANGLE_TARGET_FEATURES,
    derive_speed_angle_targets_np,
    velocity_target_features_for_parameterization,
)

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:  # type: ignore[no-redef]
        """Minimal fallback so non-training dataset tests can run without PyTorch."""

        pass

    class _TorchFallback:
        float32 = np.float32
        bool = np.bool_
        long = np.int64

        @staticmethod
        def as_tensor(value, dtype=None):
            return np.asarray(value, dtype=dtype)

    torch = _TorchFallback()  # type: ignore[assignment]


def load_canonical_dataset_arrays(npz_path) -> dict:
    """Load the raw arrays backing CanonicalWindowDataset exactly once.

    Z is a (n_tracks, n_frames, n_features) float32 array that can run into multiple GB for the
    full canonical dataset. create_train_val_test_datasets uses this to share a single in-memory
    copy across the train/val/test splits, instead of each independently reloading and
    materializing its own full copy (three splits x full Z was enough to blow through a
    memory-constrained training container's cgroup limit in practice).
    """
    dataset = np.load(Path(npz_path), allow_pickle=False)
    return {
        "Z": dataset["Z"],
        "mask": dataset["mask"],
        "track_ids": dataset["track_ids"],
        "frames": dataset["frames"],
        "feature_names": [str(name) for name in dataset["feature_names"]],
        "velocity_units": str(dataset["velocity_units"]) if "velocity_units" in dataset.files else "px/frame",
    }


class CanonicalWindowDataset(Dataset):
    """Canonical history/future window dataset, compatible with v1 and v2 tensors."""

    def __init__(
        self,
        npz_path,
        start_frames,
        T_history=20,
        T_future=10,
        max_droplets=64,
        target_features=("vx", "vy"),
        normalization_stats=None,
        fit_normalization=False,
        experiment_config=DEFAULT_EXPERIMENT_CONFIG,
        shared_arrays=None,
    ):
        self.npz_path = Path(npz_path)
        self.start_frames = np.asarray(start_frames, dtype=np.int64)
        self.T_history = int(T_history)
        self.T_future = int(T_future)
        self.T_total = self.T_history + self.T_future
        self.max_droplets = int(max_droplets)
        self.target_features = tuple(target_features)

        if shared_arrays is None:
            shared_arrays = load_canonical_dataset_arrays(self.npz_path)
        self.Z = shared_arrays["Z"]
        self.mask = shared_arrays["mask"]
        self.track_ids = shared_arrays["track_ids"]
        self.frames = shared_arrays["frames"]
        self.feature_names = shared_arrays["feature_names"]
        self.velocity_units = shared_arrays["velocity_units"]
        self.experiment_config = Path(experiment_config)
        self.velocity_mm_s_per_px_frame = self._runtime_velocity_conversion()

        self.feature_indices = self._feature_indices(self.feature_names)
        self.target_source_features = velocity_target_features_for_parameterization(self.target_features)
        self.target_indices = [self.feature_indices[name] for name in self.target_source_features]

        if fit_normalization:
            self.normalization_stats = self._fit_normalization()
        else:
            self.normalization_stats = normalization_stats

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def _runtime_velocity_conversion(self) -> float:
        if self.velocity_units != "mm/s":
            return 1.0
        return float(load_velocity_conversion_from_experiment(self.experiment_config)["velocity_mm_s_per_px_frame"])

    def __len__(self):
        return len(self.start_frames)

    def __getitem__(self, index):
        frame_start = int(self.start_frames[index])
        selected = self._select_droplets(frame_start)
        M = len(selected)

        history_x = np.zeros((self.T_history, self.max_droplets, self.feature_dim), dtype=np.float32)
        future_y = np.zeros((self.T_future, self.max_droplets, len(self.target_features)), dtype=np.float32)
        history_mask = np.zeros((self.T_history, self.max_droplets), dtype=bool)
        future_mask = np.zeros((self.T_future, self.max_droplets), dtype=bool)
        cfd_loss_mask = np.zeros((self.T_future, self.max_droplets), dtype=bool)
        droplet_ids = np.full((self.max_droplets,), -1, dtype=np.int64)

        if M > 0:
            history_slice = slice(frame_start, frame_start + self.T_history)
            future_slice = slice(frame_start + self.T_history, frame_start + self.T_total)

            raw_history = self.Z[selected, history_slice, :]
            raw_future_features = self.Z[selected, future_slice, :]
            raw_previous_features = self.Z[
                selected,
                slice(frame_start + self.T_history - 1, frame_start + self.T_total - 1),
                :,
            ]
            raw_future = self._target_values(raw_future_features, raw_previous_features)
            raw_history_mask = self.mask[selected, history_slice]
            raw_future_mask = self.mask[selected, future_slice]

            raw_history = np.transpose(raw_history, (1, 0, 2))
            raw_future = np.transpose(raw_future, (1, 0, 2))
            raw_history_mask = raw_history_mask.T
            raw_future_mask = raw_future_mask.T

            target_finite_mask = np.isfinite(raw_future).all(axis=2)
            raw_future_mask = raw_future_mask & target_finite_mask
            raw_cfd_loss_mask = self._future_cfd_loss_mask(selected, future_slice).T & raw_future_mask

            history_x[:, :M, :] = np.nan_to_num(raw_history, nan=0.0)
            future_y[:, :M, :] = np.nan_to_num(raw_future, nan=0.0)
            history_mask[:, :M] = raw_history_mask
            future_mask[:, :M] = raw_future_mask
            cfd_loss_mask[:, :M] = raw_cfd_loss_mask
            droplet_ids[:M] = self.track_ids[selected]

        self._normalize_in_place(history_x, future_y, history_mask, future_mask)

        assert history_x.shape == (self.T_history, self.max_droplets, self.feature_dim)
        assert future_y.shape == (self.T_future, self.max_droplets, len(self.target_features))
        assert history_mask.shape == (self.T_history, self.max_droplets)
        assert future_mask.shape == (self.T_future, self.max_droplets)
        assert cfd_loss_mask.shape == (self.T_future, self.max_droplets)
        assert droplet_ids.shape == (self.max_droplets,)

        return {
            "history_x": torch.as_tensor(history_x, dtype=torch.float32),
            "future_y": torch.as_tensor(future_y, dtype=torch.float32),
            "history_mask": torch.as_tensor(history_mask, dtype=torch.bool),
            "future_mask": torch.as_tensor(future_mask, dtype=torch.bool),
            "cfd_loss_mask": torch.as_tensor(cfd_loss_mask, dtype=torch.bool),
            "droplet_ids": torch.as_tensor(droplet_ids, dtype=torch.long),
            "frame_start": torch.as_tensor(frame_start, dtype=torch.long),
        }

    def _future_cfd_loss_mask(self, selected: np.ndarray, future_slice: slice) -> np.ndarray:
        return self.mask[selected, future_slice].copy()

    def _feature_indices(self, feature_names):
        feature_indices = {name: index for index, name in enumerate(feature_names)}
        for required_name in ["x", "y", "vx", "vy"]:
            if required_name not in feature_indices:
                raise KeyError(f"Missing required feature: {required_name}")
        for target_name in velocity_target_features_for_parameterization(self.target_features):
            if target_name not in feature_indices:
                raise KeyError(f"Missing target feature: {target_name}")
        if tuple(self.target_features) == SPEED_ANGLE_TARGET_FEATURES:
            for required_name in ["bbox_w", "bbox_h", "cfd_u_norm", "cfd_v_norm"]:
                if required_name not in feature_indices:
                    raise KeyError(f"Missing speed-angle target dependency: {required_name}")
        return feature_indices

    def _target_values(self, future_features: np.ndarray, previous_features: np.ndarray) -> np.ndarray:
        if tuple(self.target_features) == SPEED_ANGLE_TARGET_FEATURES:
            return derive_speed_angle_targets_np(future_features, previous_features, self.feature_indices)
        return future_features[:, :, self.target_indices]

    def _select_droplets(self, frame_start):
        window_mask = self.mask[:, frame_start : frame_start + self.T_total]
        selected = np.flatnonzero(window_mask.any(axis=1))
        if selected.size == 0:
            return selected

        sort_keys = []
        x_index = self.feature_indices["x"]
        for droplet_index in selected:
            valid_offsets = np.flatnonzero(window_mask[droplet_index])
            first_offset = int(valid_offsets[0])
            first_frame = frame_start + first_offset
            first_x = float(self.Z[droplet_index, first_frame, x_index])
            sort_keys.append((first_frame, first_x, int(self.track_ids[droplet_index]), droplet_index))

        sort_keys.sort()
        return np.asarray([item[3] for item in sort_keys], dtype=np.int64)[: self.max_droplets]

    def _fit_normalization(self):
        input_sum = np.zeros(self.feature_dim, dtype=np.float64)
        input_sumsq = np.zeros(self.feature_dim, dtype=np.float64)
        input_count = np.zeros(self.feature_dim, dtype=np.float64)

        target_sum = np.zeros(len(self.target_features), dtype=np.float64)
        target_sumsq = np.zeros(len(self.target_features), dtype=np.float64)
        target_count = np.zeros(len(self.target_features), dtype=np.float64)

        for frame_start in self.start_frames:
            selected = self._select_droplets(int(frame_start))
            if selected.size == 0:
                continue

            history_slice = slice(int(frame_start), int(frame_start) + self.T_history)
            future_slice = slice(int(frame_start) + self.T_history, int(frame_start) + self.T_total)

            raw_history = self.Z[selected, history_slice, :]
            raw_history_mask = self.mask[selected, history_slice]
            input_valid = raw_history_mask[:, :, None] & np.isfinite(raw_history)
            input_sum += np.where(input_valid, raw_history, 0.0).sum(axis=(0, 1))
            input_sumsq += np.where(input_valid, raw_history * raw_history, 0.0).sum(axis=(0, 1))
            input_count += input_valid.sum(axis=(0, 1))

            raw_future_features = self.Z[selected, future_slice, :]
            raw_previous_features = self.Z[
                selected,
                slice(int(frame_start) + self.T_history - 1, int(frame_start) + self.T_total - 1),
                :,
            ]
            raw_future = self._target_values(raw_future_features, raw_previous_features)
            raw_future_mask = self.mask[selected, future_slice]
            target_valid = raw_future_mask[:, :, None] & np.isfinite(raw_future)
            target_sum += np.where(target_valid, raw_future, 0.0).sum(axis=(0, 1))
            target_sumsq += np.where(target_valid, raw_future * raw_future, 0.0).sum(axis=(0, 1))
            target_count += target_valid.sum(axis=(0, 1))

        input_mean, input_std = _mean_std(input_sum, input_sumsq, input_count)
        for index, name in enumerate(self.feature_names):
            if name == "cfd_valid" or name.startswith("occupancy_"):
                input_mean[index] = 0.0
                input_std[index] = 1.0
        target_mean, target_std = _mean_std(target_sum, target_sumsq, target_count)
        return {
            "feature_names": np.asarray(self.feature_names, dtype=str),
            "input_mean": input_mean.astype(np.float32),
            "input_std": input_std.astype(np.float32),
            "target_features": np.asarray(self.target_features, dtype=str),
            "target_mean": target_mean.astype(np.float32),
            "target_std": target_std.astype(np.float32),
        }

    def _normalize_in_place(self, history_x, future_y, history_mask, future_mask):
        if self.normalization_stats is None:
            return

        input_mean = np.asarray(self.normalization_stats["input_mean"], dtype=np.float32)
        input_std = np.asarray(self.normalization_stats["input_std"], dtype=np.float32)
        target_mean = np.asarray(self.normalization_stats["target_mean"], dtype=np.float32)
        target_std = np.asarray(self.normalization_stats["target_std"], dtype=np.float32)

        valid_history = history_mask[:, :, None] & np.isfinite(history_x)
        history_x[valid_history] = ((history_x - input_mean) / input_std)[valid_history]
        history_x[~valid_history] = 0.0

        valid_future = future_mask[:, :, None] & np.isfinite(future_y)
        future_y[valid_future] = ((future_y - target_mean) / target_std)[valid_future]
        future_y[~valid_future] = 0.0


def create_train_val_test_datasets(
    npz_path,
    stride=5,
    T_history=20,
    T_future=10,
    max_droplets=64,
    target_features=("vx", "vy"),
    experiment_config=DEFAULT_EXPERIMENT_CONFIG,
):
    shared_arrays = load_canonical_dataset_arrays(npz_path)
    T = len(shared_arrays["frames"])
    T_total = T_history + T_future
    all_start_frames = np.arange(0, T - T_total + 1, stride, dtype=np.int64)

    train_end = int(0.70 * len(all_start_frames))
    val_end = int(0.85 * len(all_start_frames))

    train_starts = all_start_frames[:train_end]
    val_starts = all_start_frames[train_end:val_end]
    test_starts = all_start_frames[val_end:]

    train_dataset = CanonicalWindowDataset(
        npz_path=npz_path,
        start_frames=train_starts,
        T_history=T_history,
        T_future=T_future,
        max_droplets=max_droplets,
        target_features=target_features,
        fit_normalization=True,
        experiment_config=experiment_config,
        shared_arrays=shared_arrays,
    )
    normalization_stats = train_dataset.normalization_stats

    val_dataset = CanonicalWindowDataset(
        npz_path=npz_path,
        start_frames=val_starts,
        T_history=T_history,
        T_future=T_future,
        max_droplets=max_droplets,
        target_features=target_features,
        normalization_stats=normalization_stats,
        experiment_config=experiment_config,
        shared_arrays=shared_arrays,
    )
    test_dataset = CanonicalWindowDataset(
        npz_path=npz_path,
        start_frames=test_starts,
        T_history=T_history,
        T_future=T_future,
        max_droplets=max_droplets,
        target_features=target_features,
        normalization_stats=normalization_stats,
        experiment_config=experiment_config,
        shared_arrays=shared_arrays,
    )

    return train_dataset, val_dataset, test_dataset, normalization_stats


def masked_future_velocity_mse_loss(prediction, target, mask):
    expanded_mask = mask.unsqueeze(-1).expand_as(target)
    squared_error = (prediction - target) ** 2
    valid_error = squared_error[expanded_mask]
    if valid_error.numel() == 0:
        return squared_error.sum() * 0.0
    return valid_error.mean()


def _mean_std(value_sum, value_sumsq, value_count):
    safe_count = np.maximum(value_count, 1.0)
    mean = value_sum / safe_count
    variance = value_sumsq / safe_count - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-12))
    std[value_count == 0] = 1.0
    return mean, std
