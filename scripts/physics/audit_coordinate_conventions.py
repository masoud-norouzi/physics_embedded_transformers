from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.physics.geometry.coordinate_audit import run_coordinate_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit coordinate and vector conventions across physics modules.")
    parser.add_argument("--experiment-config", default="configs/experiments/video_2.yml")
    parser.add_argument("--library", default="outputs/physics/junction_cfd/solutions")
    parser.add_argument("--output-root", default="outputs/physics/coordinate_audit")
    parser.add_argument("--left-fraction", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_coordinate_audit(
        experiment_config_path=args.experiment_config,
        library_path=args.library,
        output_root=args.output_root,
        left_fraction=args.left_fraction,
        overwrite=args.overwrite,
    )
    print("Coordinate convention audit complete")
    print(f"  output: {args.output_root}")
    print(f"  stored-field check: {summary['stored_field_check']}")
    for row in summary["direction_audit_rows"]:
        print(
            f"  {row['region']}: u=({row['u_x_cfd_m_per_s']:.3e}, {row['u_y_cfd_m_per_s']:.3e}) "
            f"dot={row['measured_direction_dot']:.3e} pass={row['passed']}"
        )


if __name__ == "__main__":
    main()
