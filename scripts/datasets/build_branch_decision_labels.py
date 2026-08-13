from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.physics.runtime import load_physics_runtime_context
from src.physics.targets.junction_decision import derive_branch_decision_labels


DEFAULT_DATASET = Path("outputs/processed/2/canonical_dataset_v2/canonical_dataset_v2.npz")
DEFAULT_EXPERIMENT = Path("configs/experiments/video_2.yml")
DEFAULT_CFD_LIBRARY = Path("outputs/physics/full_device_cfd/library")
DEFAULT_OUTPUT = Path("outputs/processed/2/canonical_dataset_v2/branch_decision_labels.npz")


def main() -> None:
    args = parse_args()
    if args.metadata_json is None:
        args.metadata_json = args.output.with_suffix(".metadata.json")

    with np.load(args.dataset) as loaded:
        Z = loaded["Z"]
        mask = loaded["mask"]
        track_ids = loaded["track_ids"]
        frames = loaded["frames"]
        feature_names = [str(name) for name in loaded["feature_names"]]
    feature_index = {name: index for index, name in enumerate(feature_names)}

    runtime_context = load_physics_runtime_context(
        experiment_config_path=args.experiment_config,
        cfd_library_path=args.cfd_library,
        feature_names=feature_names,
    )

    branch_label, in_window, frames_until_commit = derive_branch_decision_labels(
        Z, mask, feature_index, runtime_context.region_labels
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        branch_label=branch_label,
        in_window=in_window,
        frames_until_commit=frames_until_commit,
        track_ids=track_ids,
        frames=frames,
    )

    labeled = in_window & ~np.isnan(branch_label)
    tracks_with_window = int(in_window.any(axis=1).sum())
    window_lengths = in_window.sum(axis=1)
    window_lengths = window_lengths[window_lengths > 0]
    summary = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "num_tracks": int(Z.shape[0]),
        "num_frames": int(Z.shape[1]),
        "tracks_with_window": tracks_with_window,
        "labeled_track_frame_pairs": int(labeled.sum()),
        "fraction_short_right_branch": float(branch_label[labeled].mean()) if labeled.any() else float("nan"),
        "window_length_min": int(window_lengths.min()) if window_lengths.size else 0,
        "window_length_mean": float(window_lengths.mean()) if window_lengths.size else float("nan"),
        "window_length_median": float(np.median(window_lengths)) if window_lengths.size else float("nan"),
        "window_length_max": int(window_lengths.max()) if window_lengths.size else 0,
    }
    args.metadata_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Branch decision labels complete")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute junction branch-decision labels for canonical_dataset_v2.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--cfd-library", type=Path, default=DEFAULT_CFD_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-json", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
