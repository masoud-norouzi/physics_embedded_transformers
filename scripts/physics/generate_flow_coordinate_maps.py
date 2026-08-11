from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

from src.physics.cfd.flow_coordinates import (
    FlowCoordinateBuildConfig,
    build_flow_coordinate_map,
    save_flow_coordinate_diagnostics,
)
from src.physics.interpolation import VelocityFieldLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate flow-aligned psi/T coordinate maps from full-device CFD cases.")
    parser.add_argument("--library", type=Path, default=Path("outputs/physics/full_device_cfd/library"))
    parser.add_argument("--config", type=Path, default=Path("configs/physics/full_device_cfd.yml"))
    parser.add_argument("--case", default=None, help="Exact case_id to generate. Omit with --all to generate every case.")
    parser.add_argument("--all", action="store_true", help="Generate coordinate maps for all cases in the library.")
    parser.add_argument("--seed-count", type=int, default=121)
    parser.add_argument("--max-step-um", type=float, default=4.0)
    parser.add_argument("--max-time-s", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--low-speed-um-per-s", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Write diagnostic figures beside each coordinate map.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and args.case is None:
        raise ValueError("Provide --case CASE_ID or --all")
    library = VelocityFieldLibrary.from_directory(args.library, args.config)
    cases = list(library.cases) if args.all else [_case_by_id(library, str(args.case))]
    build_config = FlowCoordinateBuildConfig(
        seed_count=args.seed_count,
        max_step_um=args.max_step_um,
        max_time_s=args.max_time_s,
        max_steps=args.max_steps,
        low_speed_um_per_s=args.low_speed_um_per_s,
    )
    summary = []
    started = time.perf_counter()
    for case in cases:
        out = Path(case.path) / "flow_coordinates.npz"
        if out.exists() and not args.overwrite:
            print(f"Skipping existing coordinate map: {out}", flush=True)
            continue
        print(f"Building flow coordinates for {case.case_id}", flush=True)
        coord_map, traces = build_flow_coordinate_map(case, config=build_config)
        coord_map.save(out)
        if args.diagnostics:
            save_flow_coordinate_diagnostics(coord_map, traces, case.mesh.geometry, Path(case.path) / "flow_coordinate_diagnostics")
        summary.append(
            {
                "case_id": case.case_id,
                "path": str(out),
                "trace_count": len(traces),
                "trace_point_count": int(len(coord_map.sample_points_um)),
                "termination_counts": coord_map.metadata["trace_termination_counts"],
            }
        )
        print(f"  wrote {out} with {len(coord_map.sample_points_um)} streamline samples", flush=True)
    result = {
        "library": str(args.library),
        "case_count": len(summary),
        "runtime_s": float(time.perf_counter() - started),
        "cases": summary,
    }
    if summary:
        summary_path = args.library / "flow_coordinate_generation_summary.json"
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote summary: {summary_path}", flush=True)


def _case_by_id(library: VelocityFieldLibrary, case_id: str):
    for case in library.cases:
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in library.cases)
    raise KeyError(f"Unknown case_id {case_id!r}. Known cases: {known}")


if __name__ == "__main__":
    main()
