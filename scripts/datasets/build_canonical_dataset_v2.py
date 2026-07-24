from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.velocity import load_velocity_conversion_from_experiment
from src.datasets.canonical_dataset_builder import CanonicalDatasetV2Builder


DEFAULT_INPUT = Path("outputs/physics/video_2/enrichment/physics_enriched_tracked_features.csv")
DEFAULT_OCCUPANCY = Path("outputs/physics/video_2/droplet_occupancy.csv")
DEFAULT_OUTPUT = Path("outputs/processed/2/canonical_dataset_v2/canonical_dataset_v2.npz")
DEFAULT_EXPERIMENT = Path("configs/experiments/video_2.yml")


def main() -> None:
    args = parse_args()
    velocity = load_velocity_conversion_from_experiment(args.experiment_config)
    builder = CanonicalDatasetV2Builder(
        input_csv=args.input_csv,
        output_npz=args.output_npz,
        occupancy_csv=args.occupancy_csv,
        metadata_json=args.metadata_json,
        inlet_y_max_px=args.inlet_y_max_px,
        pixel_scale_um_per_px=velocity["pixel_scale_um_per_px"],
        frame_rate_fps=velocity["frame_rate_fps"],
    )
    summary = builder.run()
    print("canonical_dataset_v2 complete")
    print(f"  tracks: {summary['num_tracks']}")
    print(f"  frames: {summary['num_frames']}")
    print(f"  features: {summary['feature_count']}")
    print(f"  valid CFD fraction: {summary['valid_cfd_fraction']:.2%}")
    print(f"  velocity units: {summary['velocity_units']}")
    print(f"  velocity conversion: {velocity['velocity_mm_s_per_px_frame']:.6f} mm/s per px/frame from {args.experiment_config}")
    print(f"  metadata: {summary['metadata_json']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build physics-enabled canonical_dataset_v2.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--occupancy-csv", type=Path, default=DEFAULT_OCCUPANCY)
    parser.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_OUTPUT.with_suffix(".metadata.json"))
    parser.add_argument("--inlet-y-max-px", type=float, default=100.0)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
