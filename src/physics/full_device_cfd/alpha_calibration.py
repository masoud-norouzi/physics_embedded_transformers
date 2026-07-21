from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from .domain import build_full_device_cfd_geometry
from .field_verification import build_common_grid, evaluate_solution_on_grid, region_masks, vector_error_metrics
from .mesh import FullDeviceMesh, generate_full_device_mesh, label_boundary_facets
from .solver import save_full_device_solution, solve_full_device_stokes


DEFAULT_OUTPUT = Path("outputs/physics/full_device_cfd/alpha_calibration")
DEFAULT_PRODUCTION_LIBRARY_OUTPUT = Path("outputs/physics/full_device_cfd/library")
DEFAULT_CONFIG = Path("configs/physics/full_device_cfd.yml")
DEFAULT_HYDRAULIC_STATE = Path("outputs/physics/video_2/baseline_hydraulic_state.csv")
DEFAULT_ALPHA0_SOLUTION = Path("outputs/physics/full_device_cfd/alpha0_equal_split_smoke/stokes_solution.npz")


@dataclass(frozen=True)
class AlphaCalibrationConfig:
    split_tolerance: float = 1.0e-3
    preferred_split_tolerance: float = 5.0e-4
    bracket_growth_factor: float = 3.0
    beta_initial: float = 0.1
    beta_max: float = 100.0
    observed_quantile_low: float = 0.01
    observed_quantile_high: float = 0.99
    split_margin: float = 0.03
    natural_split_tolerance: float = 1.0e-3
    monotonicity_tolerance: float = 1.0e-4
    grid_spacing_um: float = 8.0
    resume: bool = True
    output_dir: Path = DEFAULT_OUTPUT


@dataclass(frozen=True)
class AlphaEvaluation:
    case_id: str
    alpha_left_pa_s_per_m2: float
    alpha_right_pa_s_per_m2: float
    beta_left: float
    beta_right: float
    achieved_left_fraction: float
    achieved_right_fraction: float
    q_left_m2_per_s: float
    q_right_m2_per_s: float
    q_in_m2_per_s: float
    q_out_m2_per_s: float
    mass_mismatch_m2_per_s: float
    relative_mass_mismatch: float
    pressure_range_pa: float
    max_velocity_m_per_s: float
    min_velocity_m_per_s: float
    runtime_s: float
    solver_backend: str
    solver_status: str
    saved_full_field: bool


def load_alpha_calibration_config(path: str | Path = DEFAULT_CONFIG) -> AlphaCalibrationConfig:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = cfg.get("alpha_calibration", {})
    values = {k: raw[k] for k in raw if k in AlphaCalibrationConfig.__dataclass_fields__}
    if "output_dir" in values:
        values["output_dir"] = Path(values["output_dir"])
    return AlphaCalibrationConfig(**values)


def alpha_formulation_metadata(viscosity_pa_s: float, channel_width_um: float) -> dict[str, Any]:
    alpha_ref = characteristic_alpha_ref(viscosity_pa_s, channel_width_um)
    return {
        "weak_form": "a_alpha(u,v)=int_Omega (alpha_left*w_left(x)+alpha_right*w_right(x)) u dot v dOmega",
        "units": "Pa s m^-2",
        "alpha_ref_pa_s_per_m2": alpha_ref,
        "alpha_ref_definition": "mu / W^2",
        "viscosity_pa_s": float(viscosity_pa_s),
        "channel_width_um": float(channel_width_um),
        "acts_on": "velocity block only; pressure changes through the coupled incompressible Stokes solve",
        "drag_interpretation": "distributed Brinkman/Darcy-like linear momentum drag alpha*u",
        "spatial_support": "smooth left and right branch resistance weights from full_device_cfd.domain.resistance_weights; zero near junctions and outside branches",
        "sign_convention": "alpha >= 0 increases momentum resistance; alpha_left should decrease f_L, alpha_right should increase f_L",
        "separate_coefficients": True,
        "pressure_stabilization": "none",
    }


def characteristic_alpha_ref(viscosity_pa_s: float, channel_width_um: float) -> float:
    width_m = channel_width_um * 1.0e-6
    return float(viscosity_pa_s / (width_m * width_m))


def analyze_observed_split_distribution(
    hydraulic_state_path: str | Path = DEFAULT_HYDRAULIC_STATE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    split_margin: float = 0.03,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(hydraulic_state_path)
    if "left_flow_ul_hr" not in df or "right_flow_ul_hr" not in df:
        raise ValueError("Hydraulic state must contain left_flow_ul_hr and right_flow_ul_hr")
    split = df["left_flow_ul_hr"] / (df["left_flow_ul_hr"] + df["right_flow_ul_hr"])
    dist = pd.DataFrame({"frame": df["frame"], "left_flow_fraction": split})
    dist.to_csv(out / "observed_split_distribution.csv", index=False)
    summary = observed_split_summary(split, quantile_low, quantile_high, split_margin)
    (out / "observed_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_histogram(split.to_numpy(), out / "observed_split_histogram.png")
    return summary


def observed_split_summary(split: pd.Series | np.ndarray, q_low: float = 0.01, q_high: float = 0.99, margin: float = 0.03) -> dict[str, Any]:
    values = np.asarray(split, dtype=float)
    p1 = float(np.quantile(values, q_low))
    p99 = float(np.quantile(values, q_high))
    return {
        "count": int(len(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "quantile_low": float(q_low),
        "quantile_high": float(q_high),
        "frames_outside_0p1_0p9": int(np.count_nonzero((values < 0.1) | (values > 0.9))),
        "lower_target": float(max(0.05, p1 - margin)),
        "upper_target": float(min(0.95, p99 + margin)),
    }


class AlphaEvaluator:
    def __init__(
        self,
        mesh: FullDeviceMesh,
        alpha_ref: float,
        output_dir: str | Path = DEFAULT_OUTPUT,
        *,
        total_flow_rate_ul_per_hr: float = 1960.0,
        viscosity_pa_s: float = 0.001,
        resume: bool = True,
    ) -> None:
        self.mesh = mesh
        self.alpha_ref = float(alpha_ref)
        self.output_dir = Path(output_dir)
        self.total_flow_rate_ul_per_hr = float(total_flow_rate_ul_per_hr)
        self.viscosity_pa_s = float(viscosity_pa_s)
        self.cache_path = self.output_dir / "alpha_evaluation_cache.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache: dict[tuple[float, float], AlphaEvaluation] = {}
        self.new_solves = 0
        self.cache_hits = 0
        if resume:
            self._load_cache()

    def evaluate_alpha(
        self,
        alpha_left: float,
        alpha_right: float,
        case_id: str | None = None,
        *,
        save_full_field: bool = False,
    ) -> AlphaEvaluation:
        alpha_left = float(alpha_left)
        alpha_right = float(alpha_right)
        if alpha_left < 0.0 or alpha_right < 0.0:
            raise ValueError("alpha_left and alpha_right must be nonnegative")
        key = _alpha_key(alpha_left, alpha_right)
        existing = self.cache.get(key)
        if existing is not None and (existing.saved_full_field or not save_full_field):
            self.cache_hits += 1
            print(
                f"[alpha-cache] hit {existing.case_id}: "
                f"beta_L={existing.beta_left:.6g}, beta_R={existing.beta_right:.6g}, "
                f"f_L={existing.achieved_left_fraction:.6f}",
                flush=True,
            )
            return existing
        case_id = case_id or alpha_case_id(alpha_left, alpha_right)
        print(
            f"[alpha-solve] start {case_id}: "
            f"beta_L={alpha_left / self.alpha_ref:.6g}, beta_R={alpha_right / self.alpha_ref:.6g}, "
            f"save_full_field={save_full_field}",
            flush=True,
        )
        start = time.perf_counter()
        solution = solve_full_device_stokes(
            self.mesh,
            target_left_fraction=0.5,
            alpha_left_pa_s_per_m2=alpha_left,
            alpha_right_pa_s_per_m2=alpha_right,
            total_flow_rate_ul_per_hr=self.total_flow_rate_ul_per_hr,
            viscosity_pa_s=self.viscosity_pa_s,
            case_id=case_id,
        )
        speed = np.linalg.norm(solution.velocity_node_m_per_s, axis=1)
        flux = solution.fluxes_m2_per_s
        if save_full_field:
            case_dir = self.output_dir / "cases" / case_id
            save_full_device_solution(solution, case_dir, calibration_history=[], overwrite=True)
            _copy_case_figures(case_dir)
            _save_streamline_figure(solution, case_dir / "streamlines.png")
        result = AlphaEvaluation(
            case_id=case_id,
            alpha_left_pa_s_per_m2=alpha_left,
            alpha_right_pa_s_per_m2=alpha_right,
            beta_left=alpha_left / self.alpha_ref,
            beta_right=alpha_right / self.alpha_ref,
            achieved_left_fraction=float(solution.actual_left_fraction),
            achieved_right_fraction=float(1.0 - solution.actual_left_fraction),
            q_left_m2_per_s=float(flux["left_branch"]),
            q_right_m2_per_s=float(flux["right_branch"]),
            q_in_m2_per_s=float(flux["inlet"]),
            q_out_m2_per_s=float(flux["outlet"]),
            mass_mismatch_m2_per_s=float(flux["inlet"] + flux["outlet"]),
            relative_mass_mismatch=float(abs((flux["inlet"] + flux["outlet"]) / flux["inlet"])),
            pressure_range_pa=float(np.nanmax(solution.pressure_node_pa) - np.nanmin(solution.pressure_node_pa)),
            max_velocity_m_per_s=float(np.nanmax(speed)),
            min_velocity_m_per_s=float(np.nanmin(speed)),
            runtime_s=float(time.perf_counter() - start),
            solver_backend=solution.solver_backend,
            solver_status="success",
            saved_full_field=bool(save_full_field),
        )
        self.cache[key] = result
        self.new_solves += 1
        self.save_cache()
        print(
            f"[alpha-solve] done {case_id}: "
            f"f_L={result.achieved_left_fraction:.6f}, "
            f"pressure_range={result.pressure_range_pa:.3f} Pa, "
            f"runtime={result.runtime_s:.1f} s",
            flush=True,
        )
        return result

    def save_cache(self) -> None:
        rows = [evaluation_to_dict(v) for v in sorted(self.cache.values(), key=lambda r: (r.alpha_left_pa_s_per_m2, r.alpha_right_pa_s_per_m2))]
        if rows:
            _write_csv(self.cache_path, rows)

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                converted = {k: _convert(v) for k, v in row.items()}
                ev = AlphaEvaluation(**converted)
                self.cache[_alpha_key(ev.alpha_left_pa_s_per_m2, ev.alpha_right_pa_s_per_m2)] = ev


def run_monotonicity_sweep(
    evaluator: AlphaEvaluator,
    natural_split: float,
    lower_target: float,
    upper_target: float,
    betas: list[float] | None = None,
    tolerance: float = 1.0e-4,
) -> dict[str, Any]:
    betas = betas or [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    rows = []
    for side in ("left", "right"):
        for beta in betas:
            alpha = beta * evaluator.alpha_ref
            left = alpha if side == "left" else 0.0
            right = alpha if side == "right" else 0.0
            ev = evaluator.evaluate_alpha(left, right, f"sweep_{side}_beta_{_token(beta)}")
            rows.append({**evaluation_to_dict(ev), "sweep_side": side, "beta": beta})
            if beta > 0.0 and side == "left" and ev.achieved_left_fraction <= lower_target:
                break
            if beta > 0.0 and side == "right" and ev.achieved_left_fraction >= upper_target:
                break
    _write_csv(evaluator.output_dir / "alpha_monotonicity_sweep.csv", rows)
    _plot_alpha_curve(rows, evaluator.output_dir / "alpha_to_split_curve.png", logx=False)
    _plot_alpha_curve(rows, evaluator.output_dir / "alpha_to_split_curve_logscale.png", logx=True)
    result = {
        "natural_split": float(natural_split),
        "lower_target": float(lower_target),
        "upper_target": float(upper_target),
        "left_sweep_monotone_decreasing": _monotone(rows, "left", decreasing=True, tolerance=tolerance),
        "right_sweep_monotone_increasing": _monotone(rows, "right", decreasing=False, tolerance=tolerance),
        "reachable_minimum_left_split": float(min(row["achieved_left_fraction"] for row in rows)),
        "reachable_maximum_left_split": float(max(row["achieved_left_fraction"] for row in rows)),
    }
    if not result["left_sweep_monotone_decreasing"] or not result["right_sweep_monotone_increasing"]:
        raise RuntimeError(f"Alpha monotonicity check failed: {result}")
    return result


def calibration_targets(lower: float, natural: float, upper: float) -> list[float]:
    targets = [lower, 0.5 * (lower + natural), natural, 0.5 * (natural + upper), upper]
    if all(abs(t - 0.5) > 0.005 for t in targets):
        targets.append(0.5)
    return sorted(set(round(float(t), 6) for t in targets))


def calibrate_target(
    evaluator: AlphaEvaluator,
    target: float,
    natural_split: float,
    cfg: AlphaCalibrationConfig,
    *,
    common_alpha: float = 0.0,
    save_full_field: bool = True,
    case_prefix: str = "calibrated",
) -> dict[str, Any]:
    before_solves, before_hits = evaluator.new_solves, evaluator.cache_hits
    if abs(target - natural_split) <= cfg.natural_split_tolerance and common_alpha == 0.0:
        ev = evaluator.evaluate_alpha(0.0, 0.0, "calibrated_natural_alpha0", save_full_field=save_full_field)
        return _calibration_row(target, ev, 0, before_solves, before_hits, evaluator)
    side = "left" if target < natural_split else "right"

    def pair(alpha_extra: float) -> tuple[float, float]:
        if side == "left":
            return common_alpha + alpha_extra, common_alpha
        return common_alpha, common_alpha + alpha_extra

    def g(alpha_extra: float) -> float:
        left, right = pair(alpha_extra)
        return evaluator.evaluate_alpha(left, right).achieved_left_fraction - target

    lower = 0.0
    lower_value = g(lower)
    upper = max(cfg.beta_initial * evaluator.alpha_ref, 1.0e-12)
    beta_upper = upper / evaluator.alpha_ref
    while beta_upper <= cfg.beta_max:
        value = g(upper)
        if (side == "left" and value <= 0.0) or (side == "right" and value >= 0.0):
            break
        upper *= cfg.bracket_growth_factor
        beta_upper = upper / evaluator.alpha_ref
    else:
        return {
            "requested_left_fraction": float(target),
            "status": "unreachable",
            "side": side,
            "bracket_low_g": float(lower_value),
            "reachable_boundary_left_fraction": float(evaluator.evaluate_alpha(*pair(upper / cfg.bracket_growth_factor)).achieved_left_fraction),
        }

    iterations = {"count": 0}

    def root(alpha_extra: float) -> float:
        iterations["count"] += 1
        return g(alpha_extra)

    alpha_extra = brentq(root, lower, upper, xtol=1.0e-6 * evaluator.alpha_ref, rtol=1.0e-8, maxiter=20)
    left, right = pair(alpha_extra)
    ev = evaluator.evaluate_alpha(left, right, f"{case_prefix}_fL_{_token(target)}", save_full_field=save_full_field)
    return _calibration_row(target, ev, iterations["count"], before_solves, before_hits, evaluator)


def run_alpha_calibration_workflow(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    *,
    run_identifiability: bool = True,
) -> dict[str, Any]:
    cfg = load_alpha_calibration_config(config_path)
    if output_dir is not None:
        cfg = AlphaCalibrationConfig(**{**cfg.__dict__, "output_dir": Path(output_dir)})
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    geometry = build_full_device_cfd_geometry()
    mesh = load_or_generate_production_mesh(geometry)
    alpha_ref = characteristic_alpha_ref(0.001, geometry.channel_width_um)
    formulation = alpha_formulation_metadata(0.001, geometry.channel_width_um)
    (out / "alpha_formulation.json").write_text(json.dumps(formulation, indent=2), encoding="utf-8")
    observed = analyze_observed_split_distribution(DEFAULT_HYDRAULIC_STATE, out, cfg.observed_quantile_low, cfg.observed_quantile_high, cfg.split_margin)
    evaluator = AlphaEvaluator(mesh, alpha_ref, out, resume=cfg.resume)
    natural = evaluator.evaluate_alpha(0.0, 0.0, "natural_alpha0", save_full_field=True)
    sweep = run_monotonicity_sweep(evaluator, natural.achieved_left_fraction, observed["lower_target"], observed["upper_target"], tolerance=cfg.monotonicity_tolerance)
    targets = calibration_targets(observed["lower_target"], natural.achieved_left_fraction, observed["upper_target"])
    target_rows = [{"requested_left_fraction": t} for t in targets]
    _write_csv(out / "calibration_targets.csv", target_rows)
    calibrated = []
    progress = {"targets": [], "started_at_unix": time.time()}
    for target in targets:
        print(f"[alpha-calibration] target f_L={target:.6f}", flush=True)
        row = calibrate_target(evaluator, target, natural.achieved_left_fraction, cfg, save_full_field=True)
        calibrated.append(row)
        progress["targets"].append(row)
        (out / "calibration_progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    summary = {
        "alpha_formulation": formulation,
        "observed_split_summary": observed,
        "natural_alpha0_split": natural.achieved_left_fraction,
        "selected_calibration_targets": targets,
        "monotonicity_sweep": sweep,
        "calibrated_cases": calibrated,
        "canonical_library_coordinate": "achieved_left_fraction",
    }
    if run_identifiability:
        summary["same_split_identifiability"] = same_split_identifiability(evaluator, calibrated, cfg, out / "same_split_identifiability")
    (out / "calibrated_cases_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_production_split_library(
    targets: list[float],
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    calibration_cache_dir: str | Path = DEFAULT_OUTPUT,
    output_dir: str | Path = DEFAULT_PRODUCTION_LIBRARY_OUTPUT,
) -> dict[str, Any]:
    """Generate the production split library using the validated calibration path."""
    cfg = load_alpha_calibration_config(config_path)
    cache_dir = Path(calibration_cache_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geometry = build_full_device_cfd_geometry()
    mesh = load_or_generate_production_mesh(geometry)
    alpha_ref = characteristic_alpha_ref(0.001, geometry.channel_width_um)
    evaluator = AlphaEvaluator(mesh, alpha_ref, cache_dir, resume=True)
    natural = evaluator.evaluate_alpha(0.0, 0.0, "natural_alpha0", save_full_field=True)
    rows = []
    progress = {
        "requested_targets": [float(t) for t in targets],
        "natural_left_fraction": natural.achieved_left_fraction,
        "started_at_unix": time.time(),
        "cases": [],
    }
    for target in targets:
        print(f"[production-library] target requested f_L={target:.6f}", flush=True)
        before_solves = evaluator.new_solves
        before_hits = evaluator.cache_hits
        row = calibrate_target(
            evaluator,
            float(target),
            natural.achieved_left_fraction,
            cfg,
            save_full_field=True,
            case_prefix="production",
        )
        if row.get("status") == "success":
            library_case_id = f"production_fL_{_token(float(target))}"
            row["solution_path"] = str(_stage_production_case(cache_dir / "cases" / row["case_id"], output / "cases" / library_case_id).resolve())
        row["new_cfd_solves_total_for_target"] = evaluator.new_solves - before_solves
        row["cache_hits_total_for_target"] = evaluator.cache_hits - before_hits
        rows.append(row)
        progress["cases"].append(row)
        (output / "production_library_progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    successful = [row for row in rows if row.get("status") == "success"]
    manifest = production_manifest_rows(successful)
    _write_csv(output / "production_split_library.csv", manifest)
    _write_csv(output / "production_split_library_detailed.csv", successful)
    _plot_requested_vs_achieved(rows, output / "requested_vs_achieved_split.png")
    validation = production_library_validation(manifest)
    summary = {
        "requested_targets": [float(t) for t in targets],
        "natural_left_fraction": natural.achieved_left_fraction,
        "alpha_ref_pa_s_per_m2": alpha_ref,
        "calibration_cache_dir": str(cache_dir),
        "successful_case_count": len(successful),
        "unreachable_case_count": len(rows) - len(successful),
        "unreachable_cases": [row for row in rows if row.get("status") != "success"],
        "validation": validation,
        "cases": successful,
        "canonical_library_coordinate": "achieved_split",
        "notes": [
            "Solver, mesh, monotonicity sweep, and same-split identifiability were not modified or rerun.",
            "Manifest is sorted by achieved CFD split.",
        ],
    }
    (output / "production_split_library_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _stage_production_case(source: Path, destination: Path) -> Path:
    if destination.exists() and not source.exists():
        metadata_path = destination / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["case_id"] = destination.name
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return destination
    if not source.exists():
        raise FileNotFoundError(f"Calibrated full-field case does not exist: {source}")
    if source.resolve() == destination.resolve():
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    metadata_path = destination / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["case_id"] = destination.name
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination


def production_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = [
        {
            "requested_split": row["requested_left_fraction"],
            "achieved_split": row["achieved_left_fraction_cfd"],
            "alpha_left": row["alpha_left_pa_s_per_m2"],
            "alpha_right": row["alpha_right_pa_s_per_m2"],
            "beta_left": row["beta_left"],
            "beta_right": row["beta_right"],
            "solution_path": row["solution_path"],
        }
        for row in rows
    ]
    return sorted(manifest, key=lambda row: row["achieved_split"])


def production_library_validation(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    if not manifest:
        return {
            "maximum_calibration_error": None,
            "achieved_splits_strictly_monotonic": False,
            "largest_adjacent_achieved_split_gap": None,
        }
    requested = np.asarray([row["requested_split"] for row in manifest], dtype=float)
    achieved = np.asarray([row["achieved_split"] for row in manifest], dtype=float)
    gaps = np.diff(achieved)
    return {
        "maximum_calibration_error": float(np.max(np.abs(achieved - requested))),
        "achieved_splits_strictly_monotonic": bool(np.all(gaps > 0.0)),
        "largest_adjacent_achieved_split_gap": float(np.max(gaps)) if len(gaps) else 0.0,
        "minimum_achieved_split": float(np.min(achieved)),
        "maximum_achieved_split": float(np.max(achieved)),
        "successful_case_count": int(len(manifest)),
    }


def same_split_identifiability(
    evaluator: AlphaEvaluator,
    calibrated_rows: list[dict[str, Any]],
    cfg: AlphaCalibrationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in calibrated_rows if row.get("status") == "success" and abs(row["requested_left_fraction"] - row["natural_left_fraction"]) > 0.02]
    if not candidates:
        return {"status": "skipped", "reason": "no non-natural calibrated target available"}
    base = min(candidates, key=lambda row: abs(row["requested_left_fraction"] - 0.5))
    target = float(base["achieved_left_fraction_cfd"])
    common_alpha = max(evaluator.alpha_ref, base["alpha_left_pa_s_per_m2"], base["alpha_right_pa_s_per_m2"])
    before_solves, before_hits = evaluator.new_solves, evaluator.cache_hits
    row_b = calibrate_target(
        evaluator,
        target,
        target - 0.05 if base["alpha_right_pa_s_per_m2"] > 0 else target + 0.05,
        cfg,
        common_alpha=common_alpha,
        save_full_field=True,
        case_prefix="same_split_common",
    )
    ev_a = evaluator.evaluate_alpha(base["alpha_left_pa_s_per_m2"], base["alpha_right_pa_s_per_m2"])
    ev_b = evaluator.evaluate_alpha(row_b["alpha_left_pa_s_per_m2"], row_b["alpha_right_pa_s_per_m2"])
    metrics = compare_saved_case_fields(evaluator.output_dir / "cases" / ev_a.case_id / "stokes_solution.npz", evaluator.output_dir / "cases" / ev_b.case_id / "stokes_solution.npz", evaluator.mesh, cfg.grid_spacing_um)
    _write_csv(output_dir / "comparison_metrics.csv", metrics)
    summary = {
        "status": "success",
        "target_left_fraction": target,
        "case_a": evaluation_to_dict(ev_a),
        "case_b": evaluation_to_dict(ev_b),
        "common_alpha_pa_s_per_m2": common_alpha,
        "new_solves": evaluator.new_solves - before_solves,
        "cache_hits": evaluator.cache_hits - before_hits,
        "metrics": metrics,
        "one_dimensional_split_library_justified": bool(
            metrics[0]["vector_l2_relative_error"] <= 0.01
            and metrics[1]["vector_l2_relative_error"] <= 0.02
            and metrics[2]["vector_l2_relative_error"] <= 0.02
            and metrics[0]["p95_angular_error_deg"] <= 1.0
        ),
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def compare_saved_case_fields(path_a: Path, path_b: Path, mesh: FullDeviceMesh, grid_spacing_um: float) -> list[dict[str, Any]]:
    a = dict(np.load(path_a))
    b = dict(np.load(path_b))
    grid = build_common_grid(mesh.geometry, grid_spacing_um)
    masks = region_masks(grid, mesh.geometry)
    field_a = evaluate_solution_on_grid(a["nodes_um"], a["elements"], a["velocity_dofs_m_per_s"], grid)
    field_b = evaluate_solution_on_grid(b["nodes_um"], b["elements"], b["velocity_dofs_m_per_s"], grid)
    return vector_error_metrics(field_b, field_a, masks)


def load_or_generate_production_mesh(geometry=None) -> FullDeviceMesh:
    geometry = geometry or build_full_device_cfd_geometry()
    mesh_path = Path("outputs/physics/full_device_cfd/mesh/full_device_mesh.npz")
    if mesh_path.exists():
        data = np.load(mesh_path)
        nodes = data["nodes_um"]
        elements = data["elements"]
        facets = label_boundary_facets(nodes, elements, geometry)
        boundary_nodes = np.unique(np.concatenate([edges.ravel() for edges in facets.values() if len(edges)])).astype(np.int64)
        labels = {name: np.unique(edges.ravel()).astype(np.int64) if len(edges) else np.array([], dtype=np.int64) for name, edges in facets.items()}
        return FullDeviceMesh(nodes, elements, geometry, boundary_nodes, labels, facets, 0.0)
    return generate_full_device_mesh(geometry, target_size_um=24.0, boundary_size_um=12.0)


def evaluation_to_dict(ev: AlphaEvaluation) -> dict[str, Any]:
    return ev.__dict__.copy()


def alpha_case_id(alpha_left: float, alpha_right: float) -> str:
    return f"alpha_L_{_sci_token(alpha_left)}__alpha_R_{_sci_token(alpha_right)}"


def _calibration_row(target: float, ev: AlphaEvaluation, iterations: int, before_solves: int, before_hits: int, evaluator: AlphaEvaluator) -> dict[str, Any]:
    return {
        "status": "success",
        "requested_left_fraction": float(target),
        "achieved_left_fraction_cfd": ev.achieved_left_fraction,
        "achieved_right_fraction_cfd": ev.achieved_right_fraction,
        "absolute_targeting_error": abs(ev.achieved_left_fraction - target),
        "natural_left_fraction": evaluator.evaluate_alpha(0.0, 0.0).achieved_left_fraction,
        "alpha_left_pa_s_per_m2": ev.alpha_left_pa_s_per_m2,
        "alpha_right_pa_s_per_m2": ev.alpha_right_pa_s_per_m2,
        "beta_left": ev.beta_left,
        "beta_right": ev.beta_right,
        "q_left_m2_per_s": ev.q_left_m2_per_s,
        "q_right_m2_per_s": ev.q_right_m2_per_s,
        "q_in_m2_per_s": ev.q_in_m2_per_s,
        "q_out_m2_per_s": ev.q_out_m2_per_s,
        "pressure_range_pa": ev.pressure_range_pa,
        "max_velocity_m_per_s": ev.max_velocity_m_per_s,
        "min_velocity_m_per_s": ev.min_velocity_m_per_s,
        "mass_mismatch_m2_per_s": ev.mass_mismatch_m2_per_s,
        "relative_mass_mismatch": ev.relative_mass_mismatch,
        "root_finder_iterations": int(iterations),
        "new_cfd_solves": int(evaluator.new_solves - before_solves),
        "cache_hits": int(evaluator.cache_hits - before_hits),
        "runtime_s": ev.runtime_s,
        "case_id": ev.case_id,
        "interpolation_coordinate": ev.achieved_left_fraction,
    }


def _monotone(rows: list[dict[str, Any]], side: str, *, decreasing: bool, tolerance: float) -> bool:
    vals = [row["achieved_left_fraction"] for row in rows if row["sweep_side"] == side]
    diffs = np.diff(vals)
    return bool(np.all(diffs <= tolerance) if decreasing else np.all(diffs >= -tolerance))


def _plot_histogram(values: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=40, color="#2563eb", alpha=0.75)
    ax.set_xlabel("predicted left flow fraction")
    ax.set_ylabel("frames")
    ax.set_title("Observed baseline-hydraulic split distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_alpha_curve(rows: list[dict[str, Any]], path: Path, *, logx: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for side, marker in [("left", "o"), ("right", "s")]:
        side_rows = [row for row in rows if row["sweep_side"] == side]
        beta = np.asarray([row["beta"] for row in side_rows], dtype=float)
        split = np.asarray([row["achieved_left_fraction"] for row in side_rows], dtype=float)
        x = beta if logx else np.asarray([row["alpha_left_pa_s_per_m2"] + row["alpha_right_pa_s_per_m2"] for row in side_rows])
        if logx:
            x = np.maximum(x, 1.0e-6)
        ax.plot(x, split, marker=marker, label=f"{side} resistance")
    if logx:
        ax.set_xscale("log")
        ax.set_xlabel("beta = alpha / alpha_ref")
    else:
        ax.set_xlabel("raw alpha (Pa s m^-2)")
    ax.set_ylabel("achieved left flow fraction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_requested_vs_achieved(rows: list[dict[str, Any]], path: Path) -> None:
    successful = [row for row in rows if row.get("status") == "success"]
    if not successful:
        return

    requested = np.asarray([row["requested_left_fraction"] for row in successful], dtype=float)
    achieved = np.asarray([row["achieved_left_fraction_cfd"] for row in successful], dtype=float)
    lo = float(min(requested.min(), achieved.min()))
    hi = float(max(requested.max(), achieved.max()))
    pad = max(0.005, 0.04 * (hi - lo))

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(requested, achieved, marker="o", color="#2563eb", linewidth=1.5, label="CFD calibrated")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="0.35", linewidth=1.0, label="ideal")
    ax.set_xlabel("requested left-flow fraction")
    ax.set_ylabel("achieved CFD left-flow fraction")
    ax.set_title("Production split-library calibration")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_streamline_figure(solution, path: Path) -> None:
    from matplotlib.tri import Triangulation, LinearTriInterpolator

    nodes = solution.mesh.nodes_um
    tri = Triangulation(nodes[:, 0], nodes[:, 1], solution.mesh.elements)
    ux = LinearTriInterpolator(tri, solution.velocity_node_m_per_s[:, 0])
    uy = LinearTriInterpolator(tri, solution.velocity_node_m_per_s[:, 1])
    x = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 250)
    y = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), 250)
    xx, yy = np.meshgrid(x, y)
    u = np.asarray(ux(xx, yy), dtype=float)
    v = np.asarray(uy(xx, yy), dtype=float)
    speed = np.hypot(u, v)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.streamplot(x, y, u, v, density=2.6, color=speed, cmap="viridis", linewidth=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("x_device_um")
    ax.set_ylabel("y_device_um")
    ax.set_title(f"Streamlines, {solution.case_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _copy_case_figures(case_dir: Path) -> None:
    mapping = {"speed_magnitude.png": "speed.png", "pressure.png": "pressure.png"}
    for src_name, dst_name in mapping.items():
        src = case_dir / "diagnostics" / src_name
        if src.exists():
            shutil.copyfile(src, case_dir / dst_name)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    last_error = None
    for _attempt in range(5):
        try:
            with tmp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5)
    raise PermissionError(f"Could not write CSV after retries: {path}") from last_error


def _convert(value: str) -> Any:
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _alpha_key(alpha_left: float, alpha_right: float) -> tuple[float, float]:
    return (round(float(alpha_left), 9), round(float(alpha_right), 9))


def _token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p").replace("+", "")


def _sci_token(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.6f}e{exponent:+03d}".replace("-", "m").replace("+", "p").replace(".", "p")
