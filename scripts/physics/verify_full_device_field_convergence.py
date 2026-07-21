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

from src.physics.full_device_cfd.field_verification import run_field_verification


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    _log("Starting 24 um vs 12 um full-device field verification")
    _log(f"Output directory: {args.output_dir}")
    _log(f"Common-grid spacing: {args.grid_spacing_um:g} um")
    summary = run_field_verification(args.output_dir, grid_spacing_um=args.grid_spacing_um)
    _log(json.dumps(_brief(summary), indent=2))
    _log(f"Finished in {time.perf_counter() - started:.1f} s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the 24 um production-candidate velocity field against the 12 um reference field.")
    parser.add_argument("--grid-spacing-um", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics/full_device_cfd/convergence/field_comparison_24_vs_12"))
    return parser.parse_args()


def _brief(summary: dict) -> dict:
    metrics = summary["metrics_by_region"]
    return {
        "full_domain_vector_l2": metrics["full_domain"]["vector_l2_relative_error"],
        "inlet_junction_vector_l2": metrics["inlet_junction"]["vector_l2_relative_error"],
        "outlet_junction_vector_l2": metrics["outlet_junction"]["vector_l2_relative_error"],
        "full_domain_median_angle_deg": metrics["full_domain"]["median_angular_error_deg"],
        "full_domain_p95_angle_deg": metrics["full_domain"]["p95_angular_error_deg"],
        "separatrix_difference": summary["separatrix_difference"],
        "recommendation": summary["recommendation"],
    }


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    main()
