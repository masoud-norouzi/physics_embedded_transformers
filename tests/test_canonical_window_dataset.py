from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.canonical_window_dataset import (
    CanonicalWindowDataset,
    create_train_val_test_datasets,
)


V1_FEATURES = ["x", "y", "vx", "vy", "circularity"]
V2_FEATURES = [
    "x",
    "y",
    "vx",
    "vy",
    "circularity",
    "cfd_u",
    "cfd_v",
    "left_flow_fraction",
    "occupancy_inlet_channel",
    "occupancy_inlet_junction",
    "occupancy_left_branch",
    "occupancy_right_branch",
    "occupancy_outlet_junction",
    "occupancy_outlet_channel",
    "cfd_valid",
]


def test_v2_loader_keeps_previous_window_counts_splits_and_indexing(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "canonical_v2.npz", V2_FEATURES)

    train, val, test, _ = create_train_val_test_datasets(
        npz,
        stride=2,
        T_history=2,
        T_future=2,
        max_droplets=3,
    )

    expected_starts = np.arange(0, 8 - 4 + 1, 2, dtype=np.int64)
    assert np.array_equal(
        np.concatenate([train.start_frames, val.start_frames, test.start_frames]),
        expected_starts,
    )
    assert (len(train), len(val), len(test)) == (2, 0, 1)

    sample = train[0]
    assert int(sample["frame_start"]) == 0
    assert sample["history_x"].shape == (2, 3, 15)
    assert sample["future_y"].shape == (2, 3, 2)
    assert sample["history_mask"].shape == (2, 3)
    assert sample["future_mask"].shape == (2, 3)
    assert sample["cfd_loss_mask"].shape == (2, 3)


def test_v2_cfd_valid_feature_and_target_loss_mask_are_separate(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "canonical_v2.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=2,
        T_future=2,
        max_droplets=3,
    )

    sample = dataset[0]
    cfd_valid_index = V2_FEATURES.index("cfd_valid")

    history_x = np.asarray(sample["history_x"])
    future_mask = np.asarray(sample["future_mask"])
    cfd_loss_mask = np.asarray(sample["cfd_loss_mask"])

    assert history_x[0, 0, cfd_valid_index] == 1.0
    assert history_x[1, 0, cfd_valid_index] == 0.0

    assert future_mask[0, 0].item() is True
    assert cfd_loss_mask[0, 0].item() is True
    assert future_mask[1, 0].item() is True
    assert cfd_loss_mask[1, 0].item() is False


def test_invalid_cfd_values_do_not_remove_windows_or_future_targets(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "canonical_v2.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0, 1, 2],
        T_history=2,
        T_future=2,
        max_droplets=3,
    )

    assert len(dataset) == 3
    sample = dataset[0]
    assert np.asarray(sample["future_mask"]).sum().item() == 5
    assert np.asarray(sample["cfd_loss_mask"]).sum().item() == 4


def test_original_canonical_dataset_remains_backward_compatible(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "canonical_v1.npz", V1_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=2,
        T_future=2,
        max_droplets=3,
    )

    sample = dataset[0]
    assert sample["history_x"].shape == (2, 3, 5)
    assert set(
        [
            "history_x",
            "future_y",
            "history_mask",
            "future_mask",
            "droplet_ids",
            "frame_start",
        ]
    ).issubset(sample.keys())
    assert np.array_equal(np.asarray(sample["cfd_loss_mask"]), np.asarray(sample["future_mask"]))


def test_rollout_compatible_batch_contract_only_adds_cfd_mask(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "canonical_v2.npz", V2_FEATURES)
    dataset = CanonicalWindowDataset(
        npz,
        start_frames=[0],
        T_history=3,
        T_future=1,
        max_droplets=2,
    )
    sample = dataset[0]

    assert sample["history_x"].shape == (3, 2, 15)
    assert sample["future_y"].shape == (1, 2, 2)
    assert np.asarray(sample["droplet_ids"]).tolist() == [101, 202]
    assert sample["cfd_loss_mask"].shape == sample["future_mask"].shape


def _write_npz(path: Path, feature_names: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_tracks = 3
    num_frames = 8
    feature_dim = len(feature_names)
    features = {name: i for i, name in enumerate(feature_names)}

    Z = np.full((num_tracks, num_frames, feature_dim), np.nan, dtype=np.float32)
    mask = np.zeros((num_tracks, num_frames), dtype=bool)
    track_ids = np.asarray([101, 202, 303], dtype=np.int64)
    frames = np.arange(num_frames, dtype=np.int64)

    for track_index in range(num_tracks):
        for frame in range(num_frames):
            if track_index == 2 and frame not in {1, 2}:
                continue
            mask[track_index, frame] = True
            Z[track_index, frame, features["x"]] = 10 * track_index + frame
            Z[track_index, frame, features["y"]] = 20 * track_index + frame
            Z[track_index, frame, features["vx"]] = 1.0 + track_index
            Z[track_index, frame, features["vy"]] = -1.0 - track_index
            Z[track_index, frame, features["circularity"]] = 0.8

            if "cfd_valid" in features:
                Z[track_index, frame, features["cfd_u"]] = 0.1 * frame
                Z[track_index, frame, features["cfd_v"]] = -0.1 * frame
                Z[track_index, frame, features["left_flow_fraction"]] = 0.5
                for name in feature_names:
                    if name.startswith("occupancy_"):
                        Z[track_index, frame, features[name]] = 1.0 / 6.0
                Z[track_index, frame, features["cfd_valid"]] = 1.0

    if "cfd_valid" in features:
        Z[0, 1, features["cfd_valid"]] = 0.0
        Z[0, 3, features["cfd_valid"]] = 0.0
        Z[0, 3, features["cfd_u"]] = 0.0
        Z[0, 3, features["cfd_v"]] = 0.0

    np.savez_compressed(
        path,
        Z=Z,
        mask=mask,
        track_ids=track_ids,
        frames=frames,
        feature_names=np.asarray(feature_names),
    )
    return path
