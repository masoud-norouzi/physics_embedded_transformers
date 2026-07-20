from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.physics.interpolation.validation import validate_velocity_interpolation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate interpolation of the frozen CFD Version 1 velocity library.")
    parser.add_argument(
        "--library",
        default="outputs/physics/junction_cfd/solutions",
        help="Frozen CFD Version 1 solution-library directory.",
    )
    parser.add_argument(
        "--config",
        default="configs/physics/junction_cfd.yml",
        help="Junction CFD config used only to reconstruct the shared geometry object.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/physics/interpolation/validation",
        help="Interpolation validation output directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite interpolation validation outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_velocity_interpolation(
        library_path=args.library,
        config_path=args.config,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print("Velocity interpolation validation complete")
    print(f"  withheld fractions: {summary['withheld_fractions']}")
    print(f"  output: {args.output_root}")
    consistency = summary["scientific_consistency"]
    print(f"  max mass residual: {consistency['maximum_mass_balance_residual_m2_per_s']:.3e} m^2/s")
    print(f"  max divergence L2: {consistency['maximum_divergence_l2_norm']:.3e} s^-1")


if __name__ == "__main__":
    main()
