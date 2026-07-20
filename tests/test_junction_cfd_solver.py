from pathlib import Path
import json

import numpy as np

from src.physics.cfd.solver import (
    UL_PER_HR_TO_M3_PER_S,
    UM_TO_M,
    evaluate_solution,
    save_solution_outputs,
    solve_junction_stokes,
)


CONFIG = Path("configs/physics/junction_cfd.yml")


def test_stokes_solver_uses_si_units_and_scikit_fem_backend() -> None:
    solution = solve_junction_stokes(CONFIG)
    report = evaluate_solution(solution)

    expected_flux = 1960.0 * UL_PER_HR_TO_M3_PER_S / (100.0 * UM_TO_M)
    assert solution.solver_backend == "scikit-fem/direct"
    assert report.element_pair == "P2 velocity / P1 pressure"
    assert report.coordinate_units_geometry == "um"
    assert report.coordinate_units_solve == "m"
    assert np.isclose(solution.inlet_flux_m2_per_s, expected_flux)
    assert np.isclose(solution.inlet_mean_velocity_m_per_s, expected_flux / (100.0 * UM_TO_M))


def test_stokes_solution_fields_are_finite_and_nontrivial() -> None:
    solution = solve_junction_stokes(CONFIG)
    report = evaluate_solution(solution)

    assert np.isfinite(solution.velocity_node_m_per_s).all()
    assert np.isfinite(solution.pressure_node_pa).all()
    assert report.maximum_velocity_m_per_s > 0
    assert report.maximum_pressure_pa > report.minimum_pressure_pa
    assert abs(report.maximum_pressure_pa) < 1.0e4
    assert abs(report.minimum_pressure_pa) < 1.0e4


def test_stokes_split_is_reported_for_both_outlets() -> None:
    solution = solve_junction_stokes(CONFIG)

    assert set(solution.split_fraction) == {"left", "right"}
    assert 0.0 < solution.split_fraction["left"] < 1.0
    assert 0.0 < solution.split_fraction["right"] < 1.0
    assert np.isclose(solution.split_fraction["left"] + solution.split_fraction["right"], 1.0)


def test_stokes_solution_outputs_are_written(tmp_path: Path) -> None:
    solution = solve_junction_stokes(CONFIG)
    save_solution_outputs(solution, tmp_path, overwrite=True)

    assert (tmp_path / "fields" / "stokes_solution.npz").exists()
    assert (tmp_path / "reports" / "solution_metadata.json").exists()
    assert (tmp_path / "reports" / "solution_report.md").exists()
    assert (tmp_path / "figures" / "velocity_magnitude.png").exists()
    assert (tmp_path / "figures" / "pressure_field.png").exists()
    assert (tmp_path / "figures" / "velocity_streamlines_dense.png").exists()
    assert (tmp_path / "figures" / "velocity_streamlines_inlet_seeded.png").exists()

    metadata = json.loads((tmp_path / "reports" / "solution_metadata.json").read_text(encoding="utf-8"))
    diagnostics = metadata["streamline_diagnostics"]
    assert diagnostics["inlet_seed_count"] == 31
    assert diagnostics["terminated_left"] + diagnostics["terminated_right"] + diagnostics["terminated_other"] == 31
    assert "post-processing samples" in diagnostics["note"]

    fields = np.load(tmp_path / "fields" / "stokes_solution.npz")
    assert fields["inlet_streamline_seeds_um"].shape == (31, 2)
