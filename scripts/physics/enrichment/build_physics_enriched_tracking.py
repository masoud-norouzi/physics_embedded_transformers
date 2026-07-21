from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.physics.enrichment import EnrichmentConfig, build_physics_enriched_tracking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build downstream physics-enriched tracked droplet features.")
    parser.add_argument("--experiment", default="video_2", help="Experiment key; currently supports video_2.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing enrichment outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment != "video_2":
        raise ValueError("Only --experiment video_2 is currently configured")
    config = EnrichmentConfig(experiment_id=args.experiment)
    _, summary = build_physics_enriched_tracking(config, overwrite=args.overwrite)
    print("Physics-enriched tracking artifact complete")
    print(f"  output: {summary.output_path}")
    print(f"  rows: {summary.row_count}")
    print(f"  columns: {summary.column_count}")
    print(f"  valid CFD rows: {summary.inside_cfd_domain_rows} ({summary.inside_cfd_domain_fraction:.2%})")
    print(f"  invalid CFD rows: {summary.row_count - summary.inside_cfd_domain_rows}")
    print(f"  direction comparisons: {summary.flow_direction_alignment['valid_comparison_count']}")


if __name__ == "__main__":
    main()
