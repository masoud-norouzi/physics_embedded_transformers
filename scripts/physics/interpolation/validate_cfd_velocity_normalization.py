from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.physics.interpolation.normalization_validation import validate_cfd_velocity_normalization


def main() -> None:
    args = parse_args()
    summary = validate_cfd_velocity_normalization(
        library_path=args.library,
        left_fraction=args.left_fraction,
        output_root=args.output_root,
        sample_count=args.sample_count,
    )
    print("CFD velocity normalization validation complete")
    print(f"  inlet maximum velocity: {summary['inlet_reference_velocity_m_per_s']:.12g} m/s")
    print(f"  maximum normalized inlet velocity: {summary['maximum_normalized_inlet_velocity']}")
    print(f"  minimum normalized inlet velocity: {summary['minimum_normalized_inlet_velocity']}")
    print(f"  directions unchanged: {summary['directions_unchanged']}")
    print(f"  normalized magnitudes verified: {summary['normalized_magnitudes_equal_raw_divided_by_inlet_reference']}")
    print(f"  output: {args.output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model-facing CFD velocity normalization.")
    parser.add_argument("--library", type=Path, default=Path("outputs/physics/full_device_cfd/library"))
    parser.add_argument("--left-fraction", type=float, default=None)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/physics/interpolation/cfd_normalization_validation"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
