from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from src.evaluation import rollout_functions
from src.evaluation.rollout_comparison import AlignedRolloutWindowDataset


FEATURES = (
    "x",
    "y",
    "vx",
    "vy",
    "bbox_w",
    "bbox_h",
    "cfd_u_norm",
    "cfd_v_norm",
    "superficial_velocity",
    "left_flow_fraction",
    "occupancy_inlet_channel",
    "occupancy_inlet_junction",
    "occupancy_left_branch",
    "occupancy_right_branch",
    "occupancy_outlet_junction",
    "occupancy_outlet_channel",
)
TARGETS = ("vx", "vy", "bbox_w", "bbox_h")


def test_aligned_evaluation_dataset_accepts_current_16_feature_state(tmp_path: Path) -> None:
    npz = _write_npz(tmp_path / "canonical_v2.npz")
    stats = _identity_stats()
    selected = np.asarray([[10, 20, -1, -1]], dtype=np.int64)
    dataset = AlignedRolloutWindowDataset(
        npz_path=npz,
        rollout_starts=np.asarray([1], dtype=np.int64),
        selected_track_ids=selected,
        T_history=1,
        T_future=2,
        max_droplets=4,
        normalization_stats=stats,
        input_feature_names=FEATURES,
        target_features=TARGETS,
    )
    sample = dataset[0]
    assert sample["history_x"].shape == (1, 4, 16)
    assert sample["future_y"].shape == (2, 4, 4)
    assert "circularity" not in dataset.feature_indices


def test_evaluation_rollout_uses_runtime_state_for_closed_loop_predictions(tmp_path: Path, monkeypatch) -> None:
    npz = _write_npz(tmp_path / "canonical_v2.npz")
    stats = _identity_stats()
    selected = np.asarray([[10, 20, -1, -1]], dtype=np.int64)
    dataset = AlignedRolloutWindowDataset(
        npz_path=npz,
        rollout_starts=np.asarray([1], dtype=np.int64),
        selected_track_ids=selected,
        T_history=1,
        T_future=2,
        max_droplets=4,
        normalization_stats=stats,
        input_feature_names=FEATURES,
        target_features=TARGETS,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = _ConstantPredictionModel([1.0, 2.0, 21.0, 13.0])
    recorder = _RuntimeRecorder(dataset.feature_indices)
    monkeypatch.setattr(rollout_functions, "physics_runtime_step", recorder)

    rollout = rollout_functions.boundary_conditioned_rollout(
        model,
        batch,
        dataset,
        stats,
        torch.ones(2),
        runtime_context=SimpleNamespace(feature_index=dataset.feature_indices),
    )

    cfd_index = dataset.feature_indices["cfd_u_norm"]
    assert len(recorder.calls) == 2
    assert [len(call["active_mask"]) for call in recorder.calls] == [2, 2]
    assert rollout["pred_target"].shape == (1, 2, 4, 4)
    assert rollout["pred_velocity"].shape == (1, 2, 4, 2)
    assert rollout["pred_state"][0, 0, 0, cfd_index].item() == pytest.approx(5.0)
    assert model.seen_history[1][0, -1, 0, cfd_index].item() == pytest.approx(5.0)


class _ConstantPredictionModel(torch.nn.Module):
    def __init__(self, prediction) -> None:
        super().__init__()
        self.register_buffer("prediction", torch.as_tensor(prediction, dtype=torch.float32))
        self.seen_history = []

    def forward(self, history_x, history_mask, key_visibility_mask=None):
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
        out[active] = current_state[active]
        out[active, idx["x"]] = 50.0 + len(self.calls)
        out[active, idx["y"]] = 60.0 + len(self.calls)
        out[active, idx["vx"]] = model_prediction[active, 0]
        out[active, idx["vy"]] = model_prediction[active, 1]
        out[active, idx["bbox_w"]] = model_prediction[active, 2]
        out[active, idx["bbox_h"]] = model_prediction[active, 3]
        out[active, idx["cfd_u_norm"]] = 5.0
        out[active, idx["cfd_v_norm"]] = -5.0
        return out


def _identity_stats() -> dict[str, np.ndarray]:
    return {
        "input_mean": np.zeros(len(FEATURES), dtype=np.float32),
        "input_std": np.ones(len(FEATURES), dtype=np.float32),
        "target_mean": np.zeros(len(TARGETS), dtype=np.float32),
        "target_std": np.ones(len(TARGETS), dtype=np.float32),
    }


def _write_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = 2
    frames = 5
    idx = {name: i for i, name in enumerate(FEATURES)}
    Z = np.zeros((tracks, frames, len(FEATURES)), dtype=np.float32)
    mask = np.ones((tracks, frames), dtype=bool)
    for track in range(tracks):
        for frame in range(frames):
            Z[track, frame, idx["x"]] = frame + track
            Z[track, frame, idx["y"]] = 2 * frame + track
            Z[track, frame, idx["vx"]] = 1.0
            Z[track, frame, idx["vy"]] = 2.0
            Z[track, frame, idx["bbox_w"]] = 20.0
            Z[track, frame, idx["bbox_h"]] = 12.0
            Z[track, frame, idx["cfd_u_norm"]] = 0.1
            Z[track, frame, idx["cfd_v_norm"]] = -0.1
            Z[track, frame, idx["superficial_velocity"]] = 56.0
            Z[track, frame, idx["left_flow_fraction"]] = 0.5
            for name in FEATURES:
                if name.startswith("occupancy_"):
                    Z[track, frame, idx[name]] = 1.0 / 6.0
    np.savez(
        path,
        Z=Z,
        mask=mask,
        track_ids=np.asarray([10, 20], dtype=np.int64),
        frames=np.arange(frames, dtype=np.int64),
        feature_names=np.asarray(FEATURES),
        velocity_units=np.asarray("mm/s"),
    )
    return path
