from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from src.physics.full_device_cfd.domain import build_full_device_cfd_geometry
from src.physics.full_device_cfd.mesh import evaluate_full_device_mesh, generate_full_device_mesh
from src.physics.full_device_cfd.solver import solve_full_device_stokes


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()
    _log(f"Starting convergence study")
    _log(f"Output directory: {output_dir}")
    _log(f"Target element sizes: {', '.join(f'{size:g} um' for size in args.target_sizes_um)}")
    _log(f"Boundary factor: {args.boundary_factor:g}")
    _log("Building full-device CAD geometry")
    geometry = build_full_device_cfd_geometry()
    _log("Geometry ready")
    rows = _load_existing(output_dir / "full_device_alpha0_convergence.csv")
    if rows:
        _log(f"Loaded {len(rows)} existing convergence point(s)")
    completed = {float(row["target_element_size_um"]) for row in rows}
    for index, target_size in enumerate(args.target_sizes_um, start=1):
        if float(target_size) in completed and not args.overwrite:
            _log(f"[{index}/{len(args.target_sizes_um)}] Skipping existing convergence point: {target_size:g} um")
            continue
        point_start = time.perf_counter()
        boundary_size = target_size * args.boundary_factor
        _log(
            f"[{index}/{len(args.target_sizes_um)}] Starting mesh size {target_size:g} um "
            f"(boundary {boundary_size:g} um)"
        )
        step_start = time.perf_counter()
        _log("Generating mesh")
        mesh = generate_full_device_mesh(geometry, target_size_um=target_size, boundary_size_um=boundary_size)
        _log(f"Mesh generated in {_elapsed(step_start)}")
        step_start = time.perf_counter()
        _log("Evaluating mesh quality")
        quality = evaluate_full_device_mesh(mesh)
        _log(
            "Mesh quality: "
            f"{quality.nodes} nodes, {quality.elements} elements, "
            f"min angle {quality.minimum_angle_deg:.3g} deg, "
            f"max aspect {quality.maximum_aspect_ratio:.3g}"
        )
        _log(f"Mesh quality evaluated in {_elapsed(step_start)}")
        step_start = time.perf_counter()
        _log("Solving alpha=0 Stokes system")
        solution = solve_full_device_stokes(
            mesh,
            target_left_fraction=0.5,
            alpha_left_pa_s_per_m2=0.0,
            alpha_right_pa_s_per_m2=0.0,
            case_id=f"alpha0_mesh_{target_size:g}um",
        )
        _log(
            "Solve complete: "
            f"backend={solution.solver_backend}, "
            f"solve_runtime={solution.solve_runtime_s:.1f} s, "
            f"wall_time={_elapsed(step_start)}"
        )
        step_start = time.perf_counter()
        _log("Computing summary metrics")
        speed = np.linalg.norm(solution.velocity_node_m_per_s, axis=1)
        flux = solution.fluxes_m2_per_s
        rows.append(
            {
                "target_element_size_um": float(target_size),
                "boundary_element_size_um": float(boundary_size),
                "nodes": quality.nodes,
                "elements": quality.elements,
                "minimum_angle_deg": quality.minimum_angle_deg,
                "maximum_aspect_ratio": quality.maximum_aspect_ratio,
                "solve_runtime_s": solution.solve_runtime_s,
                "solver_backend": solution.solver_backend,
                "inlet_flux_m2_per_s": flux["inlet"],
                "outlet_flux_m2_per_s": flux["outlet"],
                "left_branch_flux_m2_per_s": flux["left_branch"],
                "right_branch_flux_m2_per_s": flux["right_branch"],
                "mass_error_inlet_outlet_m2_per_s": flux["inlet"] + flux["outlet"],
                "mass_error_relative_to_inlet": abs((flux["inlet"] + flux["outlet"]) / flux["inlet"]),
                "left_split_fraction": solution.actual_left_fraction,
                "right_split_fraction": 1.0 - solution.actual_left_fraction,
                "max_velocity_m_per_s": float(np.nanmax(speed)),
                "mean_velocity_m_per_s": float(np.nanmean(speed)),
                "pressure_min_pa": float(np.nanmin(solution.pressure_node_pa)),
                "pressure_max_pa": float(np.nanmax(solution.pressure_node_pa)),
                "pressure_range_pa": float(np.nanmax(solution.pressure_node_pa) - np.nanmin(solution.pressure_node_pa)),
            }
        )
        print(json.dumps(rows[-1], indent=2), flush=True)
        _log(f"Summary metrics computed in {_elapsed(step_start)}")
        step_start = time.perf_counter()
        _log("Writing checkpoint CSV/JSON/plot")
        _save_outputs(output_dir, rows)
        _log(f"Checkpoint written in {_elapsed(step_start)}")
        _log(f"Finished mesh size {target_size:g} um in {_elapsed(point_start)}")
    if rows:
        _log("Writing final convergence outputs")
        _save_outputs(output_dir, rows)
        _log(f"Final outputs written: {output_dir / 'full_device_alpha0_convergence.csv'}")
        _log(f"Final plot written: {output_dir / 'full_device_alpha0_convergence.png'}")
    _log(f"Convergence study complete in {_elapsed(run_start)}")


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _elapsed(start: float) -> str:
    seconds = time.perf_counter() - start
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {remainder:.1f} s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)} h {int(minutes)} min {remainder:.1f} s"


def _with_finest_errors(rows: list[dict]) -> list[dict]:
    finest = min(rows, key=lambda row: row["target_element_size_um"])
    for row in rows:
        for key in ("left_split_fraction", "max_velocity_m_per_s", "pressure_range_pa", "outlet_flux_m2_per_s"):
            denom = max(abs(finest[key]), 1.0e-30)
            row[f"relative_error_vs_finest_{key}"] = abs((row[key] - finest[key]) / denom)
    return rows


def _save_outputs(output_dir: Path, rows: list[dict]) -> None:
    rows = _with_finest_errors(rows)
    _write_csv(output_dir / "full_device_alpha0_convergence.csv", rows)
    (output_dir / "full_device_alpha0_convergence.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _plot(rows, output_dir / "full_device_alpha0_convergence.png")


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            converted = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = value
                    continue
                try:
                    converted[key] = int(value)
                except ValueError:
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        converted[key] = value
            rows.append(converted)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["target_element_size_um"], reverse=True)
    h = np.asarray([row["target_element_size_um"] for row in ordered], dtype=float)
    split = np.asarray([row["left_split_fraction"] for row in ordered], dtype=float)
    vmax = np.asarray([row["max_velocity_m_per_s"] for row in ordered], dtype=float)
    mass = np.asarray([row["mass_error_relative_to_inlet"] for row in ordered], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(h, split, marker="o")
    axes[0].set_ylabel("left split")
    axes[1].plot(h, vmax, marker="o")
    axes[1].set_ylabel("max velocity (m/s)")
    axes[2].semilogy(h, mass, marker="o")
    axes[2].set_ylabel("|Qin + Qout| / |Qin|")
    for ax in axes:
        ax.invert_xaxis()
        ax.set_xlabel("target element size (um)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple alpha=0 full-device CFD mesh convergence study.")
    parser.add_argument(
        "--target-sizes-um",
        type=float,
        nargs="+",
        default=[48.0, 36.0, 24.0, 18.0],
        help="Target element sizes to test. The smallest size is used as the reference.",
    )
    parser.add_argument("--boundary-factor", type=float, default=0.5, help="Boundary element size as a fraction of target size.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics/full_device_cfd/convergence_alpha0"))
    parser.add_argument("--overwrite", action="store_true", help="Recompute sizes even if they already exist in the CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
