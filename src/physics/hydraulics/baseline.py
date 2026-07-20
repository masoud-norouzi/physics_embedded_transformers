from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CANONICAL_ML_FEATURES = ["left_velocity_um_s", "right_velocity_um_s"]
DIAGNOSTIC_OUTPUTS = [
    "effective occupancies",
    "equivalent lengths",
    "branch flow rates",
    "branch velocities",
]
BASELINE_ASSUMPTIONS = [
    "The total volumetric throughput is the sum of continuous- and dispersed-phase input flow rates.",
    "The total mixture flow is split between the two branches according to their effective hydraulic resistances.",
    "Reported branch velocities are superficial mixture velocities based on the full channel cross-sectional area.",
    "No phase-specific slip or separate phase velocity is modeled in this baseline.",
    "Branch resistance is proportional to equivalent branch length.",
    "Both branches have identical cross-sections.",
    "Fluid properties are identical between branches.",
    "Droplet resistance contributions are additive.",
    "Each isolated droplet contributes 0.15 times the short-branch resistance length.",
    "Fractional branch occupancy scales the droplet contribution linearly.",
    "Inlet, outlet, upper-junction, and lower-junction occupancies do not contribute to baseline resistance.",
    "Droplet-droplet interaction corrections are not modeled.",
    "Junction dynamics are reserved for the future learned model.",
]
REQUIRED_OCCUPANCY_COLUMNS = {"frame", "track_id", "occupancy_computable", "w_left", "w_right"}


def compute_isolated_droplet_equivalent_length_um(
    short_branch_length_um: float,
    resistance_ratio: float,
) -> float:
    """Return isolated-droplet equivalent resistance length in micrometres."""
    if short_branch_length_um <= 0 or not np.isfinite(short_branch_length_um):
        raise ValueError("short_branch_length_um must be positive and finite")
    if resistance_ratio < 0 or not np.isfinite(resistance_ratio):
        raise ValueError("resistance_ratio must be nonnegative and finite")
    return float(short_branch_length_um * resistance_ratio)


def compute_effective_branch_occupancies(frame_occupancy: pd.DataFrame) -> tuple[float, float]:
    """Sum normalized left/right branch occupancies for one frame.

    Junction, inlet, and outlet occupancies are intentionally excluded from this
    baseline resistance model and left for the future learned dynamics model.
    """
    missing = REQUIRED_OCCUPANCY_COLUMNS.difference(frame_occupancy.columns)
    if missing:
        raise ValueError(f"Frame occupancy is missing columns: {sorted(missing)}")
    if (~frame_occupancy["occupancy_computable"].astype(bool)).any():
        raise ValueError("Noncomputable occupancy rows cannot be used by the hydraulics pipeline")
    values = frame_occupancy[["w_left", "w_right"]].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Normalized branch occupancy values must be finite")
    if np.any(values < -1e-12) or np.any(values > 1 + 1e-12):
        raise ValueError("Normalized branch occupancy values must lie in [0, 1]")
    return float(np.sum(values[:, 0])), float(np.sum(values[:, 1]))


def compute_effective_branch_lengths_um(
    left_length_um: float,
    right_length_um: float,
    droplet_equivalent_length_um: float,
    left_effective_occupancy: float,
    right_effective_occupancy: float,
) -> tuple[float, float]:
    """Add linear fractional droplet resistance length to each branch."""
    values = np.asarray(
        [
            left_length_um,
            right_length_um,
            droplet_equivalent_length_um,
            left_effective_occupancy,
            right_effective_occupancy,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Effective branch length inputs must be finite")
    if left_length_um <= 0 or right_length_um <= 0:
        raise ValueError("Branch lengths must be positive")
    if droplet_equivalent_length_um < 0:
        raise ValueError("Droplet equivalent length must be nonnegative")
    if left_effective_occupancy < -1e-12 or right_effective_occupancy < -1e-12:
        raise ValueError("Effective occupancies must be nonnegative")
    left = float(left_length_um + droplet_equivalent_length_um * left_effective_occupancy)
    right = float(right_length_um + droplet_equivalent_length_um * right_effective_occupancy)
    if left <= 0 or right <= 0:
        raise ValueError("Effective branch lengths must be positive")
    return left, right


def compute_branch_flow_rates_ul_hr(
    total_mixture_flow_ul_hr: float,
    left_effective_length_um: float,
    right_effective_length_um: float,
) -> tuple[float, float]:
    """Compute total-mixture branch flow rates under equal pressure drop."""
    values = np.asarray([total_mixture_flow_ul_hr, left_effective_length_um, right_effective_length_um], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Flow split inputs must be finite")
    if total_mixture_flow_ul_hr <= 0:
        raise ValueError("total_mixture_flow_ul_hr must be positive")
    if left_effective_length_um <= 0 or right_effective_length_um <= 0:
        raise ValueError("Effective branch lengths must be positive")
    denom = left_effective_length_um + right_effective_length_um
    left_flow = float(total_mixture_flow_ul_hr * right_effective_length_um / denom)
    right_flow = float(total_mixture_flow_ul_hr * left_effective_length_um / denom)
    if left_flow < -1e-12 or right_flow < -1e-12:
        raise ValueError("Calculated branch flow rates must be nonnegative")
    return left_flow, right_flow


def flow_rate_ul_hr_to_superficial_velocity_um_s(
    flow_rate_ul_hr: float,
    channel_width_um: float,
    channel_height_um: float,
) -> float:
    """Convert mixture flow in uL/hr to average superficial velocity in um/s."""
    values = np.asarray([flow_rate_ul_hr, channel_width_um, channel_height_um], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Velocity conversion inputs must be finite")
    if flow_rate_ul_hr < 0:
        raise ValueError("flow_rate_ul_hr must be nonnegative")
    if channel_width_um <= 0 or channel_height_um <= 0:
        raise ValueError("Channel dimensions must be positive")
    flow_um3_s = flow_rate_ul_hr * 1e9 / 3600.0
    return float(flow_um3_s / (channel_width_um * channel_height_um))


def compute_frame_baseline_hydraulics(
    frame_occupancy: pd.DataFrame,
    *,
    frame: int,
    left_length_um: float,
    right_length_um: float,
    droplet_equivalent_length_um: float,
    total_mixture_flow_ul_hr: float,
    channel_width_um: float,
    channel_height_um: float,
    continuous_flow_ul_hr: float | None = None,
    dispersed_flow_ul_hr: float | None = None,
) -> dict[str, Any]:
    """Compute one frame of baseline branch hydraulics."""
    n_left_eff, n_right_eff = compute_effective_branch_occupancies(frame_occupancy)
    left_eff, right_eff = compute_effective_branch_lengths_um(
        left_length_um,
        right_length_um,
        droplet_equivalent_length_um,
        n_left_eff,
        n_right_eff,
    )
    left_flow, right_flow = compute_branch_flow_rates_ul_hr(total_mixture_flow_ul_hr, left_eff, right_eff)
    left_velocity = flow_rate_ul_hr_to_superficial_velocity_um_s(
        left_flow, channel_width_um, channel_height_um
    )
    right_velocity = flow_rate_ul_hr_to_superficial_velocity_um_s(
        right_flow, channel_width_um, channel_height_um
    )
    total_reconstructed = left_flow + right_flow
    conservation_error = total_reconstructed - total_mixture_flow_ul_hr
    if abs(conservation_error) > 1e-9:
        raise ValueError(f"Flow conservation error exceeds tolerance: {conservation_error}")
    return {
        "frame": int(frame),
        "n_droplets_total": int(len(frame_occupancy)),
        "n_left_eff": n_left_eff,
        "n_right_eff": n_right_eff,
        "left_base_length_um": float(left_length_um),
        "right_base_length_um": float(right_length_um),
        "isolated_droplet_equivalent_length_um": float(droplet_equivalent_length_um),
        "continuous_input_flow_ul_hr": np.nan if continuous_flow_ul_hr is None else float(continuous_flow_ul_hr),
        "dispersed_input_flow_ul_hr": np.nan if dispersed_flow_ul_hr is None else float(dispersed_flow_ul_hr),
        "total_mixture_input_flow_ul_hr": float(total_mixture_flow_ul_hr),
        "left_effective_length_um": left_eff,
        "right_effective_length_um": right_eff,
        "left_flow_ul_hr": left_flow,
        "right_flow_ul_hr": right_flow,
        "left_velocity_um_s": left_velocity,
        "right_velocity_um_s": right_velocity,
        "total_flow_reconstructed_ul_hr": total_reconstructed,
        "flow_conservation_error_ul_hr": conservation_error,
    }


def compute_baseline_hydraulic_state(
    occupancy: pd.DataFrame,
    *,
    left_length_um: float,
    right_length_um: float,
    droplet_equivalent_length_um: float,
    total_mixture_flow_ul_hr: float,
    channel_width_um: float,
    channel_height_um: float,
    continuous_flow_ul_hr: float | None = None,
    dispersed_flow_ul_hr: float | None = None,
) -> pd.DataFrame:
    """Aggregate occupancy by frame and compute baseline hydraulic states."""
    if "frame" not in occupancy.columns:
        raise ValueError("Occupancy input is missing frame column")
    rows = []
    for frame, frame_occupancy in occupancy.groupby("frame", sort=True):
        rows.append(
            compute_frame_baseline_hydraulics(
                frame_occupancy,
                frame=int(frame),
                left_length_um=left_length_um,
                right_length_um=right_length_um,
                droplet_equivalent_length_um=droplet_equivalent_length_um,
                total_mixture_flow_ul_hr=total_mixture_flow_ul_hr,
                channel_width_um=channel_width_um,
                channel_height_um=channel_height_um,
                continuous_flow_ul_hr=continuous_flow_ul_hr,
                dispersed_flow_ul_hr=dispersed_flow_ul_hr,
            )
        )
    state = pd.DataFrame(rows)
    if state["frame"].duplicated().any():
        raise ValueError("Hydraulic state contains duplicate frame rows")
    if len(state) != occupancy["frame"].nunique():
        raise ValueError("Hydraulic state frame count does not match unique occupancy frames")
    return state
