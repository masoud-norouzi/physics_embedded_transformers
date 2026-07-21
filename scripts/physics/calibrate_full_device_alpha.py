from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

from src.physics.full_device_cfd.alpha_calibration import run_alpha_calibration_workflow


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    _log("Starting full-device alpha-to-flow-split calibration")
    _log(f"Config: {args.config}")
    _log(f"Output directory: {args.output_dir}")
    summary = run_alpha_calibration_workflow(args.config, args.output_dir, run_identifiability=not args.skip_identifiability)
    brief = {
        "natural_alpha0_split": summary["natural_alpha0_split"],
        "selected_calibration_targets": summary["selected_calibration_targets"],
        "reachable_minimum_left_split": summary["monotonicity_sweep"]["reachable_minimum_left_split"],
        "reachable_maximum_left_split": summary["monotonicity_sweep"]["reachable_maximum_left_split"],
        "canonical_library_coordinate": summary["canonical_library_coordinate"],
        "same_split_identifiability": summary.get("same_split_identifiability", {}).get("one_dimensional_split_library_justified"),
    }
    _log(json.dumps(brief, indent=2))
    _log(f"Finished in {time.perf_counter() - started:.1f} s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate full-device CFD branch-resistance alpha against achieved flow split.")
    parser.add_argument("--config", type=Path, default=Path("configs/physics/full_device_cfd.yml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics/full_device_cfd/alpha_calibration"))
    parser.add_argument("--skip-identifiability", action="store_true", help="Skip the same-split two-alpha identifiability comparison.")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    main()
