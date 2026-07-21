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

from src.physics.full_device_cfd.alpha_calibration import run_production_split_library


DEFAULT_TARGETS = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    _log("Starting production full-device CFD split library generation")
    _log(f"Targets: {', '.join(f'{target:.2f}' for target in args.targets)}")
    _log(f"Calibration cache/case dir: {args.calibration_cache_dir}")
    _log(f"Manifest output dir: {args.output_dir}")
    summary = run_production_split_library(
        args.targets,
        args.config,
        calibration_cache_dir=args.calibration_cache_dir,
        output_dir=args.output_dir,
    )
    _log(json.dumps(summary["validation"], indent=2))
    _log(f"Finished in {time.perf_counter() - started:.1f} s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the production full-device CFD library indexed by achieved split.")
    parser.add_argument("--config", type=Path, default=Path("configs/physics/full_device_cfd.yml"))
    parser.add_argument("--calibration-cache-dir", type=Path, default=Path("outputs/physics/full_device_cfd/alpha_calibration"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics/full_device_cfd/library"))
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    main()
