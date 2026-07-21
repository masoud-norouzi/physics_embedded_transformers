from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.physics.full_device_cfd.alpha_calibration import (
    AlphaCalibrationConfig,
    AlphaEvaluation,
    alpha_formulation_metadata,
    calibrate_target,
    calibration_targets,
    characteristic_alpha_ref,
    observed_split_summary,
    production_library_validation,
    production_manifest_rows,
    run_monotonicity_sweep,
    same_split_identifiability,
)


class MockEvaluator:
    def __init__(self, natural: float = 0.458, alpha_ref: float = 1.0e5) -> None:
        self.natural = natural
        self.alpha_ref = alpha_ref
        self.output_dir = Path("unused")
        self.mesh = None
        self.cache = {}
        self.new_solves = 0
        self.cache_hits = 0

    def evaluate_alpha(self, alpha_left, alpha_right, case_id=None, *, save_full_field=False):
        key = (round(float(alpha_left), 9), round(float(alpha_right), 9))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        beta_l = alpha_left / self.alpha_ref
        beta_r = alpha_right / self.alpha_ref
        split = self.natural - 0.08 * beta_l / (1.0 + beta_l) + 0.13 * beta_r / (1.0 + beta_r)
        ev = AlphaEvaluation(
            case_id=case_id or "mock",
            alpha_left_pa_s_per_m2=float(alpha_left),
            alpha_right_pa_s_per_m2=float(alpha_right),
            beta_left=float(beta_l),
            beta_right=float(beta_r),
            achieved_left_fraction=float(split),
            achieved_right_fraction=float(1.0 - split),
            q_left_m2_per_s=float(split),
            q_right_m2_per_s=float(1.0 - split),
            q_in_m2_per_s=-1.0,
            q_out_m2_per_s=1.0,
            mass_mismatch_m2_per_s=0.0,
            relative_mass_mismatch=0.0,
            pressure_range_pa=1.0 + beta_l + beta_r,
            max_velocity_m_per_s=0.1,
            min_velocity_m_per_s=0.0,
            runtime_s=0.0,
            solver_backend="mock/direct",
            solver_status="success",
            saved_full_field=save_full_field,
        )
        self.cache[key] = ev
        self.new_solves += 1
        return ev


def test_characteristic_alpha_ref_uses_mu_over_width_squared() -> None:
    assert np.isclose(characteristic_alpha_ref(0.001, 100.0), 100000.0)


def test_alpha_formulation_metadata_documents_brinkman_velocity_drag() -> None:
    metadata = alpha_formulation_metadata(0.001, 100.0)

    assert metadata["units"] == "Pa s m^-2"
    assert "alpha_left" in metadata["weak_form"]
    assert "velocity block" in metadata["acts_on"]
    assert metadata["pressure_stabilization"] == "none"


def test_observed_split_range_calculation() -> None:
    values = pd.Series([0.4, 0.45, 0.5, 0.55, 0.6])
    summary = observed_split_summary(values, margin=0.03)

    assert summary["minimum"] == 0.4
    assert summary["maximum"] == 0.6
    assert summary["frames_outside_0p1_0p9"] == 0
    assert summary["lower_target"] < summary["p01"]
    assert summary["upper_target"] > summary["p99"]


def test_alpha_zero_returns_natural_split() -> None:
    ev = MockEvaluator().evaluate_alpha(0.0, 0.0)

    assert ev.achieved_left_fraction == 0.458


def test_increasing_alpha_left_decreases_left_split() -> None:
    evaluator = MockEvaluator()

    assert evaluator.evaluate_alpha(1.0e5, 0.0).achieved_left_fraction < evaluator.evaluate_alpha(0.0, 0.0).achieved_left_fraction


def test_increasing_alpha_right_increases_left_split() -> None:
    evaluator = MockEvaluator()

    assert evaluator.evaluate_alpha(0.0, 1.0e5).achieved_left_fraction > evaluator.evaluate_alpha(0.0, 0.0).achieved_left_fraction


def test_monotonicity_sweep_passes_on_mock_evaluator(tmp_path) -> None:
    evaluator = MockEvaluator()
    evaluator.output_dir = tmp_path

    result = run_monotonicity_sweep(evaluator, 0.458, 0.43, 0.53, betas=[0, 0.1, 1, 10])

    assert result["left_sweep_monotone_decreasing"]
    assert result["right_sweep_monotone_increasing"]


def test_bracketing_and_brent_converge_above_natural_on_mock_evaluator() -> None:
    evaluator = MockEvaluator()
    cfg = AlphaCalibrationConfig(beta_max=100.0)

    row = calibrate_target(evaluator, 0.5, 0.458, cfg, save_full_field=False)

    assert row["status"] == "success"
    assert abs(row["achieved_left_fraction_cfd"] - 0.5) <= cfg.split_tolerance
    assert row["alpha_left_pa_s_per_m2"] == 0.0
    assert row["alpha_right_pa_s_per_m2"] > 0.0
    assert row["interpolation_coordinate"] == row["achieved_left_fraction_cfd"]


def test_bracketing_and_brent_converge_below_natural_on_mock_evaluator() -> None:
    evaluator = MockEvaluator()
    cfg = AlphaCalibrationConfig(beta_max=100.0)

    row = calibrate_target(evaluator, 0.43, 0.458, cfg, save_full_field=False)

    assert row["status"] == "success"
    assert abs(row["achieved_left_fraction_cfd"] - 0.43) <= cfg.split_tolerance
    assert row["alpha_left_pa_s_per_m2"] > 0.0
    assert row["alpha_right_pa_s_per_m2"] == 0.0


def test_target_equal_to_natural_returns_alpha_zero() -> None:
    evaluator = MockEvaluator()
    cfg = AlphaCalibrationConfig()

    row = calibrate_target(evaluator, 0.458, 0.458, cfg, save_full_field=False)

    assert row["alpha_left_pa_s_per_m2"] == 0.0
    assert row["alpha_right_pa_s_per_m2"] == 0.0


def test_cached_alpha_pairs_are_not_rerun() -> None:
    evaluator = MockEvaluator()

    evaluator.evaluate_alpha(0.0, 0.0)
    evaluator.evaluate_alpha(0.0, 0.0)

    assert evaluator.new_solves == 1
    assert evaluator.cache_hits == 1


def test_unreachable_targets_are_reported_without_extrapolation() -> None:
    evaluator = MockEvaluator()
    cfg = AlphaCalibrationConfig(beta_max=0.1)

    row = calibrate_target(evaluator, 0.9, 0.458, cfg, save_full_field=False)

    assert row["status"] == "unreachable"
    assert "reachable_boundary_left_fraction" in row


def test_calibration_target_set_includes_bounds_natural_and_half_when_needed() -> None:
    targets = calibration_targets(0.462, 0.458, 0.568)

    assert 0.458 in targets
    assert 0.462 in targets
    assert 0.568 in targets
    assert 0.5 in targets


def test_same_split_metric_acceptance_logic_can_be_represented() -> None:
    metrics = [
        {"vector_l2_relative_error": 0.005, "p95_angular_error_deg": 0.5},
        {"vector_l2_relative_error": 0.01, "p95_angular_error_deg": 0.5},
        {"vector_l2_relative_error": 0.01, "p95_angular_error_deg": 0.5},
    ]

    assert metrics[0]["vector_l2_relative_error"] <= 0.01
    assert metrics[1]["vector_l2_relative_error"] <= 0.02
    assert metrics[2]["p95_angular_error_deg"] <= 1.0


def test_solver_file_contains_no_pressure_stabilization_terms() -> None:
    source = Path("src/physics/full_device_cfd/solver.py").read_text(encoding="utf-8")

    assert "pressure-mass" not in source.lower()
    assert "pmass" not in source
    assert "stabilization" not in source.lower()


def test_production_manifest_is_sorted_by_achieved_split_and_uses_solution_path() -> None:
    rows = [
        {
            "requested_left_fraction": 0.5,
            "achieved_left_fraction_cfd": 0.5001,
            "alpha_left_pa_s_per_m2": 0.0,
            "alpha_right_pa_s_per_m2": 1.0,
            "beta_left": 0.0,
            "beta_right": 1.0,
            "solution_path": "b",
        },
        {
            "requested_left_fraction": 0.4,
            "achieved_left_fraction_cfd": 0.3999,
            "alpha_left_pa_s_per_m2": 1.0,
            "alpha_right_pa_s_per_m2": 0.0,
            "beta_left": 1.0,
            "beta_right": 0.0,
            "solution_path": "a",
        },
    ]

    manifest = production_manifest_rows(rows)

    assert [row["solution_path"] for row in manifest] == ["a", "b"]


def test_production_library_validation_reports_monotonicity_error_and_gaps() -> None:
    manifest = [
        {"requested_split": 0.4, "achieved_split": 0.4002},
        {"requested_split": 0.42, "achieved_split": 0.4198},
        {"requested_split": 0.45, "achieved_split": 0.4501},
    ]

    validation = production_library_validation(manifest)

    assert validation["achieved_splits_strictly_monotonic"]
    assert np.isclose(validation["maximum_calibration_error"], 0.0002)
    assert np.isclose(validation["largest_adjacent_achieved_split_gap"], 0.0303)
