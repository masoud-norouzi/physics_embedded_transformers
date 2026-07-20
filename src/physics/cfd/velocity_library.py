from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .domain import load_junction_cfd_config
from .inlet_profile_diagnostic import run_inlet_profile_diagnostic
from .mesh import evaluate_mesh_topology, generate_mesh
from .pilot_splits import _separatrix_seed_index, _seed_counts
from .solver import configure_solution_split, evaluate_solution, save_solution_outputs, solve_junction_stokes


CFD_VERSION = "1.0"
MESH_VERSION = "production_v1"


def split_grid(start: float = 0.05, stop: float = 0.95, step: float = 0.05) -> list[float]:
    values = []
    current = start
    while current <= stop + step * 0.5:
        rounded = round(current, 10)
        if not 0.0 < rounded < 1.0:
            raise ValueError("Version 1 split fractions must satisfy 0 < left_fraction < 1")
        values.append(round(rounded, 2))
        current += step
    return values


def split_case_id(left_fraction: float) -> str:
    return f"split_0p{int(round(left_fraction * 100)):02d}"


def generate_velocity_library(
    config: str | Path | dict[str, Any],
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.05,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = load_junction_cfd_config(config)
    fractions = split_grid(start, stop, step)
    records = []
    skipped = []
    rerun = []

    for left_fraction in fractions:
        case_cfg = configure_solution_split(cfg, left_fraction)
        case_id = case_cfg["solution"]["case_id"]
        output_root = Path(case_cfg["solution"]["output_root"])
        if not overwrite and _existing_case_is_valid(output_root, left_fraction):
            record = _record_from_existing(output_root)
            record["was_skipped"] = True
            skipped.append(case_id)
        else:
            solution = solve_junction_stokes(case_cfg)
            save_solution_outputs(solution, output_root, overwrite=True)
            inlet = run_inlet_profile_diagnostic(case_cfg, output_root=output_root / "inlet_profile_diagnostics")
            record = _record_from_solution(left_fraction, solution, inlet, output_root)
            _write_flux_report(output_root / "reports" / "flux_report.json", record)
            rerun.append(case_id)
        records.append(record)

    monotonicity = _monotonicity_checks(records)
    if not all(record["validation_status"] == "passed" for record in records):
        raise RuntimeError("At least one split case failed validation")
    if not all(monotonicity.values()):
        raise RuntimeError(f"Cross-case monotonicity checks failed: {monotonicity}")

    root = Path("outputs/physics/junction_cfd/solutions")
    index = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "cfd_version": CFD_VERSION,
        "mesh_version": MESH_VERSION,
        "config_path": str(config),
        "experiment_path": str(cfg.get("experiment_config")),
        "split_fractions": fractions,
        "total_runtime_s": float(time.perf_counter() - started),
        "skipped_cases": skipped,
        "rerun_cases": rerun,
        "monotonicity_checks": monotonicity,
        "records": records,
    }
    (root / "library_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (root / "library_summary.md").write_text(_markdown_summary(index), encoding="utf-8")
    _save_summary_figures(index, root)
    return index


def _existing_case_is_valid(output_root: Path, left_fraction: float) -> bool:
    metadata_path = output_root / "reports" / "solution_metadata.json"
    flux_path = output_root / "reports" / "flux_report.json"
    if not metadata_path.exists() or not flux_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        flux = json.loads(flux_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        metadata.get("cfd_version") == CFD_VERSION
        and metadata.get("mesh_version") == MESH_VERSION
        and abs(float(metadata.get("requested_left_fraction", -1.0)) - left_fraction) < 1.0e-12
        and flux.get("validation_status") == "passed"
    )


def _record_from_existing(output_root: Path) -> dict[str, Any]:
    record = json.loads((output_root / "reports" / "flux_report.json").read_text(encoding="utf-8"))
    return record


def _record_from_solution(left_fraction: float, solution, inlet_diagnostics, output_root: Path) -> dict[str, Any]:
    report = evaluate_solution(solution)
    topology = evaluate_mesh_topology(solution.mesh)
    seed_counts = _seed_counts(output_root / "reports" / "streamline_seed_diagnostics.csv")
    record = {
        "split_name": solution.case_id,
        "left_fraction": float(left_fraction),
        "right_fraction": float(1.0 - left_fraction),
        "output_path": str(output_root),
        "cfd_version": CFD_VERSION,
        "mesh_version": MESH_VERSION,
        "solver_backend": report.solver_backend,
        "structural_rank": int(solution.linear_system_diagnostics["structural_rank"]),
        "condensed_matrix_size": int(solution.linear_system_diagnostics["condensed_matrix_shape"][0]),
        "inlet_flux": float(report.inlet_flux_signed_m2_per_s),
        "left_outlet_flux": float(report.left_outlet_flux_m2_per_s),
        "right_outlet_flux": float(report.right_outlet_flux_m2_per_s),
        "measured_left_fraction": float(report.left_split_fraction),
        "measured_right_fraction": float(report.right_split_fraction),
        "mass_balance_residual": float(report.net_flux_residual_m2_per_s),
        "maximum_velocity": float(report.maximum_velocity_m_per_s),
        "pressure_minimum": float(report.minimum_pressure_pa),
        "pressure_maximum": float(report.maximum_pressure_pa),
        "pressure_range": float(report.maximum_pressure_pa - report.minimum_pressure_pa),
        "streamline_counts": seed_counts,
        "separatrix_seed_index": _separatrix_seed_index(output_root / "reports" / "streamline_seed_diagnostics.csv"),
        "inlet_poiseuille_max_flux_error": float(max(abs(item.flux_relative_error) for item in inlet_diagnostics)),
        "inlet_poiseuille_max_transverse_to_mean": float(max(item.cross_velocity_relative_to_mean for item in inlet_diagnostics)),
        "mesh_invalid_or_inverted_elements": int(topology.invalid_or_inverted_elements),
        "mesh_domain_holes": int(topology.domain_holes),
        "mesh_connected_components": int(topology.connected_fluid_components),
    }
    record["validation_status"] = "passed" if _record_is_valid(record) else "failed"
    return record


def _record_is_valid(record: dict[str, Any]) -> bool:
    expected_left_flux = abs(record["inlet_flux"]) * record["left_fraction"]
    expected_right_flux = abs(record["inlet_flux"]) * record["right_fraction"]
    return bool(
        record["solver_backend"] == "scikit-fem/direct"
        and record["structural_rank"] == record["condensed_matrix_size"]
        and record["inlet_flux"] < 0
        and record["left_outlet_flux"] > 0
        and record["right_outlet_flux"] > 0
        and abs(record["left_outlet_flux"] - expected_left_flux) / expected_left_flux < 1.0e-6
        and abs(record["right_outlet_flux"] - expected_right_flux) / expected_right_flux < 1.0e-6
        and abs(record["measured_left_fraction"] - record["left_fraction"]) < 1.0e-6
        and abs(record["measured_right_fraction"] - record["right_fraction"]) < 1.0e-6
        and abs(record["mass_balance_residual"]) < 1.0e-12
        and record["inlet_poiseuille_max_flux_error"] < 1.0e-3
        and record["inlet_poiseuille_max_transverse_to_mean"] < 1.0e-2
        and record["mesh_invalid_or_inverted_elements"] == 0
        and record["mesh_domain_holes"] == 0
        and record["mesh_connected_components"] == 1
    )


def _write_flux_report(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _monotonicity_checks(records: list[dict[str, Any]]) -> dict[str, bool]:
    ordered = sorted(records, key=lambda item: item["left_fraction"])
    left_flux = np.array([item["left_outlet_flux"] for item in ordered])
    right_flux = np.array([item["right_outlet_flux"] for item in ordered])
    measured_left = np.array([item["measured_left_fraction"] for item in ordered])
    measured_right = np.array([item["measured_right_fraction"] for item in ordered])
    requested_left = np.array([item["left_fraction"] for item in ordered])
    separatrix = [item["separatrix_seed_index"] for item in ordered if item["separatrix_seed_index"] is not None]
    separatrix_monotone = bool(len(separatrix) < 2 or np.all(np.diff(separatrix) <= 0))
    return {
        "left_outlet_flux_increases": bool(np.all(np.diff(left_flux) > 0)),
        "right_outlet_flux_decreases": bool(np.all(np.diff(right_flux) < 0)),
        "measured_left_tracks_requested": bool(np.max(np.abs(measured_left - requested_left)) < 1.0e-6),
        "measured_right_tracks_requested": bool(np.max(np.abs(measured_right - (1.0 - requested_left))) < 1.0e-6),
        "maximum_velocity_finite": bool(np.isfinite([item["maximum_velocity"] for item in ordered]).all()),
        "pressure_range_finite": bool(np.isfinite([item["pressure_range"] for item in ordered]).all()),
        "separatrix_seed_indices_monotone_when_present": separatrix_monotone,
    }


def _markdown_summary(index: dict[str, Any]) -> str:
    lines = [
        "# CFD Version 1 Prescribed-Split Velocity Library",
        "",
        f"- CFD version: {index['cfd_version']}",
        f"- Mesh version: {index['mesh_version']}",
        f"- Split count: {len(index['records'])}",
        f"- Total runtime: {index['total_runtime_s']:.2f} s",
        f"- Skipped cases: {index['skipped_cases']}",
        f"- Rerun cases: {index['rerun_cases']}",
        "",
        "| split | requested L/R | measured L/R | mass residual | max velocity | pressure range | streamlines L/R/O | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in sorted(index["records"], key=lambda item: item["left_fraction"]):
        counts = record["streamline_counts"]
        lines.append(
            f"| {record['split_name']} | {record['left_fraction']:.2f}/{record['right_fraction']:.2f} | "
            f"{record['measured_left_fraction']:.4f}/{record['measured_right_fraction']:.4f} | "
            f"{record['mass_balance_residual']:.3e} | {record['maximum_velocity']:.6e} | "
            f"{record['pressure_range']:.6e} | {counts['reached_left_outlet']}/{counts['reached_right_outlet']}/{counts['other']} | "
            f"{record['validation_status']} |"
        )
    lines.extend(["", "## Monotonicity Checks", ""])
    lines.extend(f"- {key}: {value}" for key, value in index["monotonicity_checks"].items())
    return "\n".join(lines)


def _save_summary_figures(index: dict[str, Any], root: Path) -> None:
    figures = root / "library_figures"
    figures.mkdir(parents=True, exist_ok=True)
    records = sorted(index["records"], key=lambda item: item["left_fraction"])
    x = np.array([record["left_fraction"] for record in records])
    measured_left = np.array([record["measured_left_fraction"] for record in records])
    measured_right = np.array([record["measured_right_fraction"] for record in records])
    left_flux = np.array([record["left_outlet_flux"] for record in records])
    right_flux = np.array([record["right_outlet_flux"] for record in records])
    residual = np.array([record["mass_balance_residual"] for record in records])
    pressure = np.array([record["pressure_range"] for record in records])
    max_velocity = np.array([record["maximum_velocity"] for record in records])
    left_counts = np.array([record["streamline_counts"]["reached_left_outlet"] for record in records])
    right_counts = np.array([record["streamline_counts"]["reached_right_outlet"] for record in records])

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    axes[0].plot(x, measured_left, "o-", label="measured left")
    axes[0].plot(x, x, "--", label="requested left")
    axes[0].set_title("Requested vs measured left fraction")
    axes[1].plot(x, measured_right, "o-", label="measured right")
    axes[1].plot(x, 1.0 - x, "--", label="requested right")
    axes[1].set_title("Requested vs measured right fraction")
    axes[2].plot(x, left_flux, "o-", label="left")
    axes[2].plot(x, right_flux, "o-", label="right")
    axes[2].set_title("Outlet fluxes")
    axes[3].plot(x, residual, "o-")
    axes[3].set_title("Mass-balance residual")
    axes[4].plot(x, pressure, "o-", label="pressure range")
    axes[4].plot(x, max_velocity, "o-", label="maximum velocity")
    axes[4].set_title("Pressure range and maximum velocity")
    axes[5].plot(x, left_counts, "o-", label="left seeds")
    axes[5].plot(x, right_counts, "o-", label="right seeds")
    axes[5].set_title("Seeded streamline termination counts")
    for ax in axes:
        ax.set_xlabel("requested left fraction")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures / "library_summary_metrics.png", dpi=170)
    plt.close(fig)
    _save_montage(records, figures / "library_representative_montage.png")


def _save_montage(records: list[dict[str, Any]], path: Path) -> None:
    selected = {0.05, 0.25, 0.50, 0.75, 0.95}
    cases = [record for record in records if round(record["left_fraction"], 2) in selected]
    fig, axes = plt.subplots(2, len(cases), figsize=(4.2 * len(cases), 7))
    for column, record in enumerate(cases):
        root = Path(record["output_path"])
        velocity = plt.imread(root / "figures" / "velocity_magnitude.png")
        seeds = plt.imread(root / "figures" / "velocity_streamlines_inlet_seeded.png")
        axes[0, column].imshow(velocity)
        axes[0, column].axis("off")
        axes[0, column].set_title(f"{record['split_name']} velocity")
        axes[1, column].imshow(seeds)
        axes[1, column].axis("off")
        counts = record["streamline_counts"]
        axes[1, column].set_title(f"L/R/O={counts['reached_left_outlet']}/{counts['reached_right_outlet']}/{counts['other']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str | None]:
    names = ["numpy", "scipy", "matplotlib", "scikit-fem", "pandas", "PyYAML"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the CFD Version 1 prescribed-split velocity library.")
    parser.add_argument("--config", type=Path, default=Path("configs/physics/junction_cfd.yml"))
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = generate_velocity_library(args.config, args.start, args.stop, args.step, overwrite=args.overwrite)
    print("CFD velocity library generation completed")
    print(f"  cases: {len(index['records'])}")
    print(f"  skipped: {index['skipped_cases']}")
    print(f"  rerun: {index['rerun_cases']}")
    print(f"  runtime: {index['total_runtime_s']:.2f} s")


if __name__ == "__main__":
    main()
