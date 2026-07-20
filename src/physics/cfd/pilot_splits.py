from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .domain import load_junction_cfd_config
from .inlet_profile_diagnostic import run_inlet_profile_diagnostic
from .solver import configure_solution_split, evaluate_solution, save_solution_outputs, solve_junction_stokes


PILOT_LEFT_FRACTIONS = (0.10, 0.30, 0.90)


def run_pilot_split_cases(
    config: str | Path | dict[str, Any],
    left_fractions: tuple[float, ...] = PILOT_LEFT_FRACTIONS,
    overwrite: bool = True,
) -> dict[str, Any]:
    base_cfg = load_junction_cfd_config(config)
    cases = []
    for left_fraction in left_fractions:
        cfg = configure_solution_split(base_cfg, left_fraction)
        solution_cfg = cfg["solution"]
        output_root = Path(solution_cfg["output_root"])
        solution = solve_junction_stokes(cfg)
        save_solution_outputs(solution, output_root, overwrite=overwrite)
        inlet = run_inlet_profile_diagnostic(cfg, output_root=output_root / "inlet_profile_diagnostics")
        cases.append(_case_summary(left_fraction, solution, inlet, output_root))

    summary = {
        "cfd_version": "1.0",
        "mesh_version": "production_v1",
        "note": "Pilot cases use prescribed outlet branch-flow splits; CFD computes local velocity and pressure fields but does not predict the split.",
        "cases": cases,
        "physical_response": _physical_response_summary(cases),
    }
    report_root = Path("outputs/physics/junction_cfd/solutions/pilot_splits")
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "pilot_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_root / "pilot_split_summary.md").write_text(_markdown_report(summary), encoding="utf-8")
    _save_comparison_figure(summary, report_root / "pilot_split_comparison.png")
    return summary


def _case_summary(left_fraction: float, solution, inlet_diagnostics, output_root: Path) -> dict[str, Any]:
    report = evaluate_solution(solution)
    seed_counts = _seed_counts(output_root / "reports" / "streamline_seed_diagnostics.csv")
    return {
        "case_id": solution.case_id,
        "output_root": str(output_root),
        "requested_left_fraction": float(left_fraction),
        "requested_right_fraction": float(1.0 - left_fraction),
        "measured_left_fraction": float(report.left_split_fraction),
        "measured_right_fraction": float(report.right_split_fraction),
        "inlet_flux_m2_per_s": float(report.inlet_flux_signed_m2_per_s),
        "left_outlet_flux_m2_per_s": float(report.left_outlet_flux_m2_per_s),
        "right_outlet_flux_m2_per_s": float(report.right_outlet_flux_m2_per_s),
        "net_flux_residual_m2_per_s": float(report.net_flux_residual_m2_per_s),
        "solver_backend": report.solver_backend,
        "structural_rank": int(solution.linear_system_diagnostics["structural_rank"]),
        "condensed_matrix_size": int(solution.linear_system_diagnostics["condensed_matrix_shape"][0]),
        "maximum_velocity_m_per_s": float(report.maximum_velocity_m_per_s),
        "minimum_pressure_pa": float(report.minimum_pressure_pa),
        "maximum_pressure_pa": float(report.maximum_pressure_pa),
        "pressure_range_pa": float(report.maximum_pressure_pa - report.minimum_pressure_pa),
        "divergence_diagnostics": solution.divergence_diagnostics,
        "inlet_poiseuille_diagnostics": [asdict(item) for item in inlet_diagnostics],
        "maximum_inlet_transverse_to_mean": float(max(item.cross_velocity_relative_to_mean for item in inlet_diagnostics)),
        "maximum_inlet_flux_error": float(max(abs(item.flux_relative_error) for item in inlet_diagnostics)),
        "streamline_counts": seed_counts,
        "separatrix_seed_index": _separatrix_seed_index(output_root / "reports" / "streamline_seed_diagnostics.csv"),
    }


def _seed_counts(path: Path) -> dict[str, int]:
    counts = {"reached_left_outlet": 0, "reached_right_outlet": 0, "other": 0}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reason = row["termination_reason"]
            if reason in counts:
                counts[reason] += 1
            else:
                counts["other"] += 1
    return counts


def _separatrix_seed_index(path: Path) -> int | None:
    reasons = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reasons.append(row["termination_reason"])
    other = [index for index, reason in enumerate(reasons) if reason not in {"reached_left_outlet", "reached_right_outlet"}]
    if other:
        return int(other[0])
    for index in range(1, len(reasons)):
        if reasons[index] != reasons[index - 1]:
            return int(index)
    return None


def _physical_response_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda item: item["requested_left_fraction"])
    separatrix = [item["separatrix_seed_index"] for item in ordered]
    return {
        "left_fraction_order": [item["requested_left_fraction"] for item in ordered],
        "left_streamline_counts": [item["streamline_counts"]["reached_left_outlet"] for item in ordered],
        "right_streamline_counts": [item["streamline_counts"]["reached_right_outlet"] for item in ordered],
        "separatrix_seed_indices": separatrix,
        "left_branch_velocity_increases_with_left_fraction": bool(
            ordered[0]["left_outlet_flux_m2_per_s"] < ordered[1]["left_outlet_flux_m2_per_s"] < ordered[2]["left_outlet_flux_m2_per_s"]
        ),
        "right_branch_velocity_decreases_with_left_fraction": bool(
            ordered[0]["right_outlet_flux_m2_per_s"] > ordered[1]["right_outlet_flux_m2_per_s"] > ordered[2]["right_outlet_flux_m2_per_s"]
        ),
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Pilot Prescribed Flow-Split Cases",
        "",
        summary["note"],
        "",
        "| case | requested L/R | measured L/R | mass residual (m^2/s) | max velocity (m/s) | pressure range (Pa) | streamlines L/R/other |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        counts = case["streamline_counts"]
        lines.append(
            f"| {case['case_id']} | {case['requested_left_fraction']:.2f}/{case['requested_right_fraction']:.2f} | "
            f"{case['measured_left_fraction']:.4f}/{case['measured_right_fraction']:.4f} | "
            f"{case['net_flux_residual_m2_per_s']:.3e} | {case['maximum_velocity_m_per_s']:.6e} | "
            f"{case['pressure_range_pa']:.6e} | {counts['reached_left_outlet']}/{counts['reached_right_outlet']}/{counts['other']} |"
        )
    lines.extend(
        [
            "",
            "## Physical Response",
            "",
            f"- Left streamline counts: {summary['physical_response']['left_streamline_counts']}",
            f"- Right streamline counts: {summary['physical_response']['right_streamline_counts']}",
            f"- Separatrix seed indices: {summary['physical_response']['separatrix_seed_indices']}",
            f"- Left outlet flux increases with requested left fraction: {summary['physical_response']['left_branch_velocity_increases_with_left_fraction']}",
            f"- Right outlet flux decreases with requested left fraction: {summary['physical_response']['right_branch_velocity_decreases_with_left_fraction']}",
            "",
        ]
    )
    return "\n".join(lines)


def _save_comparison_figure(summary: dict[str, Any], path: Path) -> None:
    cases = sorted(summary["cases"], key=lambda item: item["requested_left_fraction"])
    fig, axes = plt.subplots(2, len(cases), figsize=(4.8 * len(cases), 7.6))
    for column, case in enumerate(cases):
        root = Path(case["output_root"])
        velocity = plt.imread(root / "figures" / "velocity_magnitude.png")
        streamlines = plt.imread(root / "figures" / "velocity_streamlines_inlet_seeded.png")
        axes[0, column].imshow(velocity)
        axes[0, column].axis("off")
        axes[0, column].set_title(f"{case['case_id']} velocity")
        axes[1, column].imshow(streamlines)
        axes[1, column].axis("off")
        counts = case["streamline_counts"]
        axes[1, column].set_title(
            f"req/meas L={case['requested_left_fraction']:.2f}/{case['measured_left_fraction']:.2f}; "
            f"seeds L/R/O={counts['reached_left_outlet']}/{counts['reached_right_outlet']}/{counts['other']}"
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three Version-1 pilot prescribed split cases.")
    parser.add_argument("--config", type=Path, default=Path("configs/physics/junction_cfd.yml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pilot_split_cases(args.config, overwrite=args.overwrite)
    print("Pilot split cases completed")
    for case in summary["cases"]:
        print(
            f"  {case['case_id']}: requested={case['requested_left_fraction']:.2f}/{case['requested_right_fraction']:.2f}, "
            f"measured={case['measured_left_fraction']:.4f}/{case['measured_right_fraction']:.4f}, "
            f"mass_residual={case['net_flux_residual_m2_per_s']:.3e}"
        )


if __name__ == "__main__":
    main()
