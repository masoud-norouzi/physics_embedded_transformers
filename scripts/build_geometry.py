from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_experiment_config
from src.geometry import build_device_geometry, save_geometry_artifacts


def _default_centerline_path(config: dict) -> Path:
    experiment = config["experiment"]["experiment"]
    data_root = Path(experiment["data"]["root"])
    candidate = data_root / "centerlines.csv"
    if candidate.exists():
        return candidate
    device_geometry = config["device"]["device"].get("geometry", {})
    configured = device_geometry.get("centerline")
    if configured:
        return Path(configured)
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable centerline geometry artifacts.")
    parser.add_argument("--experiment", required=True, type=Path, help="Experiment YAML path.")
    parser.add_argument("--centerlines", type=Path, default=None, help="Override centerline CSV path.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/geometry"), help="Output root directory.")
    parser.add_argument("--length-tolerance-px", type=float, default=1.0, help="Loop length tolerance in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing device output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment)
    device = config["device"]["device"]
    centerlines = args.centerlines or _default_centerline_path(config)
    output_dir = args.output_root / device["id"]

    geometry, metadata = build_device_geometry(
        centerlines,
        config["device"],
        length_tolerance_px=args.length_tolerance_px,
    )
    save_geometry_artifacts(
        geometry,
        metadata,
        output_dir,
        centerlines,
        config["device"],
        overwrite=args.overwrite,
    )

    lengths = metadata["length_validation"]["branches"]
    print("Geometry build summary")
    print(f"  device ID: {geometry.device_id}")
    print(f"  source centerlines: {centerlines}")
    print(f"  output: {output_dir}")
    print(f"  upper junction: {geometry.upper_junction_xy.tolist()}")
    print(f"  lower junction: {geometry.lower_junction_xy.tolist()}")
    for branch in ("left", "right"):
        item = lengths[branch]
        print(
            f"  {branch}: computed {item['computed_length_px']:.3f} px "
            f"({item['computed_length_um']:.3f} um), configured "
            f"{item['configured_length_px']:.3f} px, "
            f"diff {item['absolute_difference_px']:.3f} px"
        )


if __name__ == "__main__":
    main()
